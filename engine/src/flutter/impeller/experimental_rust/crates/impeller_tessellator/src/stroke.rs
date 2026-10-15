use crate::convex::{
    solve_cubic, solve_quad, strip_points_to_mesh, wangs_cubic_subdivisions,
    wangs_quadratic_subdivisions,
};
use crate::trigs::Trigs;
use crate::{StrokeParameters, TessellatedMesh};
use impeller_geometry::{Matrix, Point, Scalar};
use impeller_path::{Cap, CubicPathComponent, Join, Path, PathComponent, QuadPathComponent};

pub const K_MIN_STROKE_SIZE: Scalar = 1.0;
const K_JOIN_PIXEL_THRESHOLD: Scalar = 0.25;

/// Ported from `impeller::SeparatedVector2` in `separated_vector.h`.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct SeparatedVector2 {
    direction: Point,
    magnitude: Scalar,
}

impl SeparatedVector2 {
    #[inline]
    fn new(direction: Point, magnitude: Scalar) -> Self {
        Self {
            direction,
            magnitude,
        }
    }

    #[inline]
    fn get_vector(self) -> Point {
        self.direction * self.magnitude
    }

    #[inline]
    fn get_alignment(self, other: Self) -> Scalar {
        self.direction.dot(other.direction)
    }

    #[inline]
    fn cross(self, other: Self) -> Scalar {
        self.direction.cross(other.direction)
    }
}

/// Ported from `Geometry::ComputeStrokeAlphaCoverage` in `entity/geometry/geometry.cc`.
pub fn compute_stroke_alpha_coverage(transform: &Matrix, stroke_width: Scalar) -> Scalar {
    let scaled_stroke_width = transform.get_max_basis_length_xy() * stroke_width;
    if scaled_stroke_width == 0.0 || scaled_stroke_width >= K_MIN_STROKE_SIZE {
        return 1.0;
    }
    (scaled_stroke_width * 2.0).clamp(0.0, 1.0)
}

fn compute_maximum_join_cosine(scale: Scalar, half_stroke_width: Scalar) -> Scalar {
    let hypotenuse = scale * half_stroke_width;
    if hypotenuse <= K_JOIN_PIXEL_THRESHOLD {
        return -1.1;
    }
    let bisector =
        (hypotenuse * hypotenuse - K_JOIN_PIXEL_THRESHOLD * K_JOIN_PIXEL_THRESHOLD).sqrt();
    let half_cosine = bisector / hypotenuse;
    2.0 * half_cosine * half_cosine - 1.0
}

fn compute_minimum_miter_cosine(miter_limit: Scalar) -> Scalar {
    if miter_limit <= 1.0 {
        return 1.1;
    }
    let half_cosine = 1.0 / miter_limit;
    2.0 * half_cosine * half_cosine - 1.0
}

struct StrokePathSegmentReceiver {
    points: Vec<Point>,
    half_stroke_width: Scalar,
    maximum_join_cosine: Scalar,
    minimum_miter_cosine: Scalar,
    join: Join,
    cap: Cap,
    scale: Scalar,
    trigs: Trigs,
    origin_perpendicular: SeparatedVector2,
    origin_point: Point,
    last_perpendicular: SeparatedVector2,
    last_point: Point,
    has_prior_contour: bool,
    has_prior_segment: bool,
    contour_needs_cap: bool,
}

impl StrokePathSegmentReceiver {
    fn new(stroke: StrokeParameters, scale: Scalar) -> Self {
        let miter_limit = if stroke.miter_limit < 0.0 {
            4.0
        } else {
            stroke.miter_limit
        };
        let min_size = K_MIN_STROKE_SIZE / scale.max(1e-4);
        let half_stroke_width = stroke.width.max(min_size) * 0.5;
        let trigs = Trigs::from_pixel_radius(scale * half_stroke_width);
        Self {
            points: Vec::new(),
            half_stroke_width,
            maximum_join_cosine: compute_maximum_join_cosine(scale, half_stroke_width),
            minimum_miter_cosine: compute_minimum_miter_cosine(miter_limit),
            join: stroke.join,
            cap: stroke.cap,
            scale,
            trigs,
            origin_perpendicular: SeparatedVector2::default(),
            origin_point: Point::ZERO,
            last_perpendicular: SeparatedVector2::default(),
            last_point: Point::ZERO,
            has_prior_contour: false,
            has_prior_segment: false,
            contour_needs_cap: false,
        }
    }

    #[inline]
    fn perpendicular_from_unit_direction(&self, direction: Point) -> SeparatedVector2 {
        SeparatedVector2::new(
            Point::new(-direction.y, direction.x),
            self.half_stroke_width,
        )
    }

    #[inline]
    fn perpendicular_from_points(&self, from: Point, to: Point) -> SeparatedVector2 {
        self.perpendicular_from_unit_direction((to - from).normalize())
    }

    #[inline]
    fn append_vertex(&mut self, p: Point) {
        self.points.push(p);
    }

    #[inline]
    fn append_vertices(&mut self, curve_point: Point, offset: Point) {
        self.points.push(curve_point + offset);
        self.points.push(curve_point - offset);
    }

    fn begin_contour(&mut self, origin: Point, will_be_closed: bool) {
        if self.has_prior_contour {
            self.append_vertex(self.last_point);
            self.append_vertex(self.last_point);
            self.append_vertex(origin);
            self.append_vertex(origin);
        }
        self.has_prior_contour = true;
        self.has_prior_segment = false;
        self.contour_needs_cap = !will_be_closed;
        self.last_point = origin;
        self.origin_point = origin;
    }

    fn record_line(&mut self, p1: Point, p2: Point) {
        if p2 != p1 {
            let dir = (p2 - p1).normalize();
            if dir != Point::ZERO {
                let current_perpendicular = self.perpendicular_from_unit_direction(dir);
                self.handle_previous_join(current_perpendicular);
                self.append_vertices(p2, current_perpendicular.get_vector());
                self.last_perpendicular = current_perpendicular;
                self.last_point = p2;
            }
        }
    }

    fn record_quad(&mut self, q: &QuadPathComponent) {
        let start_dir = if q.p1 != q.cp {
            Some((q.cp - q.p1).normalize())
        } else if q.p1 != q.p2 {
            Some((q.p2 - q.p1).normalize())
        } else {
            None
        };
        let end_dir = if q.p2 != q.cp {
            Some((q.p2 - q.cp).normalize())
        } else if q.p2 != q.p1 {
            Some((q.p2 - q.p1).normalize())
        } else {
            None
        };

        if let (Some(sd), Some(ed)) = (start_dir, end_dir) {
            if sd == Point::ZERO || ed == Point::ZERO {
                return;
            }
            let start_perpendicular = self.perpendicular_from_unit_direction(sd);
            let end_perpendicular = self.perpendicular_from_unit_direction(ed);

            self.handle_previous_join(start_perpendicular);

            let stroke_scale = self.scale * (self.half_stroke_width * 0.25).max(1.0);
            let count = wangs_quadratic_subdivisions(stroke_scale, q.p1, q.cp, q.p2).ceil();
            let count_usize = (count as usize).max(1);

            let mut prev = q.p1;
            let mut prev_perpendicular = start_perpendicular;

            for i in 1..count_usize {
                let cur = solve_quad(q, (i as Scalar) / count);
                if cur != prev {
                    let cur_perpendicular = self.perpendicular_from_points(prev, cur);
                    self.record_curve_segment(prev_perpendicular, cur, cur_perpendicular);
                    prev = cur;
                    prev_perpendicular = cur_perpendicular;
                }
            }

            self.record_curve_segment(prev_perpendicular, q.p2, end_perpendicular);
            self.last_perpendicular = end_perpendicular;
            self.last_point = q.p2;
        }
    }

    fn record_cubic(&mut self, c: &CubicPathComponent) {
        let start_dir = if c.p1 != c.cp1 {
            Some((c.cp1 - c.p1).normalize())
        } else if c.p1 != c.cp2 {
            Some((c.cp2 - c.p1).normalize())
        } else if c.p1 != c.p2 {
            Some((c.p2 - c.p1).normalize())
        } else {
            None
        };
        let end_dir = if c.p2 != c.cp2 {
            Some((c.p2 - c.cp2).normalize())
        } else if c.p2 != c.cp1 {
            Some((c.p2 - c.cp1).normalize())
        } else if c.p2 != c.p1 {
            Some((c.p2 - c.p1).normalize())
        } else {
            None
        };

        if let (Some(sd), Some(ed)) = (start_dir, end_dir) {
            if sd == Point::ZERO || ed == Point::ZERO {
                return;
            }
            let start_perpendicular = self.perpendicular_from_unit_direction(sd);
            let end_perpendicular = self.perpendicular_from_unit_direction(ed);

            self.handle_previous_join(start_perpendicular);

            let stroke_scale = self.scale * (self.half_stroke_width * 0.25).max(1.0);
            let count = wangs_cubic_subdivisions(stroke_scale, c.p1, c.cp1, c.cp2, c.p2).ceil();
            let count_usize = (count as usize).max(1);

            let mut prev = c.p1;
            let mut prev_perpendicular = start_perpendicular;

            for i in 1..count_usize {
                let cur = solve_cubic(c, (i as Scalar) / count);
                if cur != prev {
                    let cur_perpendicular = self.perpendicular_from_points(prev, cur);
                    self.record_curve_segment(prev_perpendicular, cur, cur_perpendicular);
                    prev = cur;
                    prev_perpendicular = cur_perpendicular;
                }
            }

            self.record_curve_segment(prev_perpendicular, c.p2, end_perpendicular);
            self.last_perpendicular = end_perpendicular;
            self.last_point = c.p2;
        }
    }

    fn record_curve_segment(
        &mut self,
        prev_perpendicular: SeparatedVector2,
        cur: Point,
        cur_perpendicular: SeparatedVector2,
    ) {
        if prev_perpendicular.get_alignment(cur_perpendicular) < (self.trigs[1].cos as Scalar) {
            self.append_vertices(cur, prev_perpendicular.get_vector());
            self.add_join(Join::Round, cur, prev_perpendicular, cur_perpendicular);
        }
        self.append_vertices(cur, cur_perpendicular.get_vector());
    }

    fn end_contour(&mut self, origin: Point, with_close: bool) {
        if !self.has_prior_segment {
            // Empty/degenerate contour: fill an axis-aligned cap box/circle at the origin.
            let cap = if self.cap == Cap::Butt {
                Cap::Square
            } else {
                self.cap
            };
            let perpendicular = Point::new(-self.half_stroke_width, 0.0);
            self.add_cap(cap, origin, perpendicular, true);
            if cap == Cap::Round {
                self.append_vertices(origin, perpendicular);
            }
            self.add_cap(cap, origin, perpendicular, false);
        } else if with_close {
            self.add_join(
                self.join,
                origin,
                self.last_perpendicular,
                self.origin_perpendicular,
            );
            self.last_perpendicular = self.origin_perpendicular;
            self.last_point = origin;
        } else {
            self.add_cap(
                self.cap,
                self.last_point,
                self.last_perpendicular.get_vector(),
                false,
            );
        }
        self.has_prior_segment = false;
    }

    fn handle_previous_join(&mut self, new_perpendicular: SeparatedVector2) {
        if self.has_prior_segment {
            self.add_join(
                self.join,
                self.last_point,
                self.last_perpendicular,
                new_perpendicular,
            );
        } else {
            self.has_prior_segment = true;
            let perpendicular_vector = new_perpendicular.get_vector();
            if self.contour_needs_cap {
                self.add_cap(self.cap, self.last_point, perpendicular_vector, true);
            }
            self.append_vertices(self.last_point, perpendicular_vector);
            self.origin_perpendicular = new_perpendicular;
        }
    }

    fn add_cap(&mut self, cap: Cap, path_point: Point, perpendicular: Point, contour_start: bool) {
        match cap {
            Cap::Butt => {}
            Cap::Round => {
                let along = Point::new(perpendicular.y, -perpendicular.x);
                if contour_start {
                    self.append_vertex(path_point - along);
                    let mut i = self.trigs.len() - 2;
                    while i > 0 {
                        let sin_v = self.trigs[i].sin as Scalar;
                        let cos_v = self.trigs[i].cos as Scalar;
                        let center = path_point - along * sin_v;
                        let offset = perpendicular * cos_v;
                        self.append_vertices(center, offset);
                        i -= 1;
                    }
                } else {
                    let end = self.trigs.len() - 1;
                    for i in 1..end {
                        let sin_v = self.trigs[i].sin as Scalar;
                        let cos_v = self.trigs[i].cos as Scalar;
                        let center = path_point + along * sin_v;
                        let offset = perpendicular * cos_v;
                        self.append_vertices(center, offset);
                    }
                    self.append_vertex(path_point + along);
                }
            }
            Cap::Square => {
                let along = Point::new(perpendicular.y, -perpendicular.x);
                let square_center = if contour_start {
                    path_point - along
                } else {
                    path_point + along
                };
                self.append_vertices(square_center, perpendicular);
            }
        }
    }

    fn add_join(
        &mut self,
        join: Join,
        path_point: Point,
        old_perpendicular: SeparatedVector2,
        new_perpendicular: SeparatedVector2,
    ) {
        let cosine = old_perpendicular.get_alignment(new_perpendicular);
        if cosine >= self.maximum_join_cosine {
            return;
        }

        match join {
            Join::Bevel => {}
            Join::Miter => {
                if cosine >= self.minimum_miter_cosine {
                    let miter_vector = (old_perpendicular.get_vector()
                        + new_perpendicular.get_vector())
                        / (cosine + 1.0);
                    if old_perpendicular.cross(new_perpendicular) < 0.0 {
                        self.append_vertex(path_point + miter_vector);
                    } else {
                        self.append_vertex(path_point - miter_vector);
                    }
                }
            }
            Join::Round => {
                let trig1_cos = self.trigs[1].cos as Scalar;
                if cosine >= trig1_cos {
                    // Angle smaller than first trig division: bevel suffices.
                } else if cosine < -trig1_cos {
                    // Nearly 180-degree turn: use a full round cap to round off the bend.
                    self.add_cap(
                        Cap::Round,
                        path_point,
                        old_perpendicular.get_vector(),
                        false,
                    );
                } else {
                    let turning = old_perpendicular.cross(new_perpendicular);
                    let (from_vector, to_vector, begin_end_crossed) = if turning > 0.0 {
                        (
                            -old_perpendicular.get_vector(),
                            -new_perpendicular.get_vector(),
                            false,
                        )
                    } else {
                        (
                            new_perpendicular.get_vector(),
                            old_perpendicular.get_vector(),
                            true,
                        )
                    };

                    if begin_end_crossed {
                        self.append_vertex(path_point + from_vector);
                    }

                    let mut visit_center = false;
                    let middle_vector = from_vector + to_vector;
                    let mut end = self.trigs.len() - 1;

                    for i in 1..end {
                        let p = self.trigs[i].rotate_vector(from_vector);
                        if p.cross(middle_vector) <= 0.0 {
                            end = i;
                            break;
                        }
                        if visit_center {
                            self.append_vertex(path_point);
                            visit_center = false;
                        } else {
                            visit_center = true;
                        }
                        self.append_vertex(path_point + p);
                    }

                    while end > 1 {
                        end -= 1;
                        let p = (-self.trigs[end]).rotate_vector(to_vector);
                        if visit_center {
                            self.append_vertex(path_point);
                            visit_center = false;
                        } else {
                            visit_center = true;
                        }
                        self.append_vertex(path_point + p);
                    }

                    if begin_end_crossed {
                        self.append_vertex(path_point + to_vector);
                    }
                }
            }
        }

        self.append_vertices(path_point, new_perpendicular.get_vector());
    }
}

/// Ported from `StrokeSegmentsGeometry::GenerateSolidStrokeVertices` in `stroke_path_geometry.cc`
/// and `PathPruner` (`is_stroking = true`) in `path_tessellator.cc`.
pub fn generate_solid_stroke_vertices(
    path: &Path,
    stroke: StrokeParameters,
    scale: Scalar,
) -> Vec<Point> {
    if stroke.width < 0.0 || scale <= 0.0 {
        return Vec::new();
    }

    let min_size = K_MIN_STROKE_SIZE / scale;
    let adjusted_stroke = StrokeParameters {
        width: stroke.width.max(min_size),
        ..stroke
    };

    let mut receiver = StrokePathSegmentReceiver::new(adjusted_stroke, scale);

    let mut contour_origin = Point::ZERO;
    let mut current_point = Point::ZERO;
    let mut contour_will_be_closed = false;
    let mut contour_has_segments = false;
    let mut contour_has_points = false;

    let finish_contour = |receiver: &mut StrokePathSegmentReceiver,
                          contour_origin: Point,
                          current_point: &mut Point,
                          contour_will_be_closed: bool,
                          contour_has_segments: &mut bool,
                          contour_has_points: &mut bool| {
        if contour_will_be_closed {
            if !*contour_has_segments {
                receiver.begin_contour(contour_origin, true);
                *contour_has_segments = true;
            }
            if !*contour_has_points {
                receiver.record_line(*current_point, contour_origin);
                *contour_has_points = true;
            } else if *current_point != contour_origin {
                receiver.record_line(*current_point, contour_origin);
            }
            receiver.end_contour(contour_origin, true);
        } else if *contour_has_segments {
            if !*contour_has_points {
                receiver.record_line(contour_origin, contour_origin);
            }
            receiver.end_contour(contour_origin, false);
        }
        *contour_has_segments = false;
        *contour_has_points = false;
    };

    let segment_encountered = |receiver: &mut StrokePathSegmentReceiver,
                               contour_origin: Point,
                               contour_will_be_closed: bool,
                               contour_has_segments: &mut bool| {
        if !*contour_has_segments {
            receiver.begin_contour(contour_origin, contour_will_be_closed);
            *contour_has_segments = true;
        }
    };

    for comp in &path.components {
        match *comp {
            PathComponent::Contour(c) => {
                if contour_has_segments || contour_will_be_closed {
                    finish_contour(
                        &mut receiver,
                        contour_origin,
                        &mut current_point,
                        contour_will_be_closed,
                        &mut contour_has_segments,
                        &mut contour_has_points,
                    );
                }
                contour_origin = c.destination;
                current_point = c.destination;
                contour_will_be_closed = c.is_closed;
                contour_has_segments = false;
                contour_has_points = false;
            }
            PathComponent::Linear(l) => {
                segment_encountered(
                    &mut receiver,
                    contour_origin,
                    contour_will_be_closed,
                    &mut contour_has_segments,
                );
                if l.p2 != current_point {
                    receiver.record_line(current_point, l.p2);
                    current_point = l.p2;
                    contour_has_points = true;
                }
            }
            PathComponent::Quad(q) => {
                if q.cp == current_point || q.p2 == q.cp {
                    segment_encountered(
                        &mut receiver,
                        contour_origin,
                        contour_will_be_closed,
                        &mut contour_has_segments,
                    );
                    if q.p2 != current_point {
                        receiver.record_line(current_point, q.p2);
                        current_point = q.p2;
                        contour_has_points = true;
                    }
                } else {
                    segment_encountered(
                        &mut receiver,
                        contour_origin,
                        contour_will_be_closed,
                        &mut contour_has_segments,
                    );
                    let aligned_q = QuadPathComponent {
                        p1: current_point,
                        cp: q.cp,
                        p2: q.p2,
                    };
                    receiver.record_quad(&aligned_q);
                    current_point = q.p2;
                    contour_has_points = true;
                }
            }
            PathComponent::Cubic(c) => {
                segment_encountered(
                    &mut receiver,
                    contour_origin,
                    contour_will_be_closed,
                    &mut contour_has_segments,
                );
                if c.cp1 != current_point || c.cp2 != current_point || c.p2 != current_point {
                    let aligned_c = CubicPathComponent {
                        p1: current_point,
                        cp1: c.cp1,
                        cp2: c.cp2,
                        p2: c.p2,
                    };
                    receiver.record_cubic(&aligned_c);
                    current_point = c.p2;
                    contour_has_points = true;
                }
            }
        }
    }

    if contour_has_segments || contour_will_be_closed {
        finish_contour(
            &mut receiver,
            contour_origin,
            &mut current_point,
            contour_will_be_closed,
            &mut contour_has_segments,
            &mut contour_has_points,
        );
    }

    receiver.points
}

/// Converts a stroked path into an indexed `TessellatedMesh` (triangle list) using Impeller's
/// solid stroke vertex strip generator.
pub fn tessellate_stroke(
    path: &Path,
    stroke: StrokeParameters,
    tolerance_scale: Scalar,
) -> TessellatedMesh {
    let scale = tolerance_scale.max(f32::EPSILON);
    let strip = generate_solid_stroke_vertices(path, stroke, scale);
    strip_points_to_mesh(&strip)
}
