use crate::convex::{indexed_strip_to_mesh, strip_points_to_mesh};
use crate::stroke::{
    compute_stroke_alpha_coverage, generate_solid_stroke_vertices, K_MIN_STROKE_SIZE,
};
use crate::trigs::{compute_quadrant_divisions, Trigs};
use crate::{StrokeParameters, TessellatedMesh, Tessellator, Vertex2D};
use impeller_geometry::{Matrix, Point, Radians, Rect, Scalar, Size, K_EH_CLOSE_ENOUGH};
use impeller_path::{Cap, FillType, Join, Path, PathBuilder, RoundingRadii};

#[inline]
pub(crate) fn is_translation_scale_only(m: &Matrix) -> bool {
    m.m[1] == 0.0
        && m.m[2] == 0.0
        && m.m[3] == 0.0
        && m.m[4] == 0.0
        && m.m[6] == 0.0
        && m.m[7] == 0.0
        && m.m[8] == 0.0
        && m.m[9] == 0.0
        && m.m[11] == 0.0
        && (m.m[15] - 1.0).abs() <= Scalar::EPSILON
        && m.m[0] != 0.0
        && m.m[5] != 0.0
}

#[inline]
pub(crate) fn transform_rect_bounds(m: &Matrix, rect: Rect) -> Rect {
    let corners = [
        m.transform_point(Point::new(rect.get_left(), rect.get_top())),
        m.transform_point(Point::new(rect.get_right(), rect.get_top())),
        m.transform_point(Point::new(rect.get_left(), rect.get_bottom())),
        m.transform_point(Point::new(rect.get_right(), rect.get_bottom())),
    ];
    let min_x = corners
        .iter()
        .map(|p| p.x)
        .fold(Scalar::INFINITY, Scalar::min);
    let min_y = corners
        .iter()
        .map(|p| p.y)
        .fold(Scalar::INFINITY, Scalar::min);
    let max_x = corners
        .iter()
        .map(|p| p.x)
        .fold(Scalar::NEG_INFINITY, Scalar::max);
    let max_y = corners
        .iter()
        .map(|p| p.y)
        .fold(Scalar::NEG_INFINITY, Scalar::max);
    Rect::from_ltrb(min_x, min_y, max_x, max_y)
}

#[inline]
pub(crate) fn rect_contains_rect(outer: Rect, inner: Rect) -> bool {
    if inner.is_empty() {
        return true;
    }
    if outer.is_empty() {
        return false;
    }
    outer.get_left() <= inner.get_left()
        && outer.get_top() <= inner.get_top()
        && outer.get_right() >= inner.get_right()
        && outer.get_bottom() >= inner.get_bottom()
}

/// Ported from `Tessellator::GenerateStartRoundCap` in `tessellator.cc`.
pub fn generate_start_round_cap(
    p: Point,
    perpendicular: Point,
    trigs: &Trigs,
    out: &mut Vec<Point>,
) {
    let along = Point::new(perpendicular.y, -perpendicular.x);
    for &trig in trigs.as_slice() {
        let relative_along = along * (trig.cos as Scalar);
        let relative_across = perpendicular * (trig.sin as Scalar);
        out.push(p - relative_along + relative_across);
        out.push(p - relative_along - relative_across);
    }
}

/// Ported from `Tessellator::GenerateEndRoundCap` in `tessellator.cc`.
pub fn generate_end_round_cap(p: Point, perpendicular: Point, trigs: &Trigs, out: &mut Vec<Point>) {
    let along = Point::new(perpendicular.y, -perpendicular.x);
    for &trig in trigs.as_slice() {
        let relative_along = along * (trig.sin as Scalar);
        let relative_across = perpendicular * (trig.cos as Scalar);
        out.push(p + relative_along + relative_across);
        out.push(p + relative_along - relative_across);
    }
}

/// Ported from `Tessellator::FilledCircle` / `GenerateFilledCircle` in `tessellator.cc`.
pub fn generate_filled_circle_strip(scale: Scalar, center: Point, radius: Scalar) -> Vec<Point> {
    let radius = radius.max(0.0);
    let divisions = compute_quadrant_divisions(scale * radius);
    let trigs = Trigs::from_divisions(divisions);
    let mut pts = Vec::with_capacity(trigs.len() * 4);

    // Quadrant 1 connecting with Quadrant 4
    for &trig in trigs.as_slice() {
        let offset = trig.scale_radius(radius);
        pts.push(Point::new(center.x - offset.x, center.y + offset.y));
        pts.push(Point::new(center.x - offset.x, center.y - offset.y));
    }

    // Quadrant 2 connecting with Quadrant 3
    for &trig in trigs.as_slice() {
        let offset = trig.scale_radius(radius);
        pts.push(Point::new(center.x + offset.y, center.y + offset.x));
        pts.push(Point::new(center.x + offset.y, center.y - offset.x));
    }

    pts
}

/// Ported from `Tessellator::StrokedCircle` / `GenerateStrokedCircle` in `tessellator.cc`.
pub fn generate_stroked_circle_strip(
    scale: Scalar,
    center: Point,
    radius: Scalar,
    half_width: Scalar,
) -> Vec<Point> {
    if half_width <= 0.0 {
        return generate_filled_circle_strip(scale, center, radius);
    }
    if half_width >= radius {
        return generate_filled_circle_strip(scale, center, radius + half_width);
    }

    let divisions = compute_quadrant_divisions(scale * (radius + half_width));
    let trigs = Trigs::from_divisions(divisions);
    let outer_radius = radius + half_width;
    let inner_radius = radius - half_width;
    let mut pts = Vec::with_capacity(trigs.len() * 8);

    // Quadrant 1
    for &trig in trigs.as_slice() {
        let outer = trig.scale_radius(outer_radius);
        let inner = trig.scale_radius(inner_radius);
        pts.push(Point::new(center.x - outer.x, center.y - outer.y));
        pts.push(Point::new(center.x - inner.x, center.y - inner.y));
    }
    // Quadrant 2
    for &trig in trigs.as_slice() {
        let outer = trig.scale_radius(outer_radius);
        let inner = trig.scale_radius(inner_radius);
        pts.push(Point::new(center.x + outer.y, center.y - outer.x));
        pts.push(Point::new(center.x + inner.y, center.y - inner.x));
    }
    // Quadrant 3
    for &trig in trigs.as_slice() {
        let outer = trig.scale_radius(outer_radius);
        let inner = trig.scale_radius(inner_radius);
        pts.push(Point::new(center.x + outer.x, center.y + outer.y));
        pts.push(Point::new(center.x + inner.x, center.y + inner.y));
    }
    // Quadrant 4
    for &trig in trigs.as_slice() {
        let outer = trig.scale_radius(outer_radius);
        let inner = trig.scale_radius(inner_radius);
        pts.push(Point::new(center.x - outer.y, center.y + outer.x));
        pts.push(Point::new(center.x - inner.y, center.y + inner.x));
    }

    pts
}

/// Ported from `Tessellator::RoundCapLine` / `GenerateRoundCapLine` in `tessellator.cc`.
pub fn generate_round_cap_line_strip(
    scale: Scalar,
    p0: Point,
    p1: Point,
    radius: Scalar,
) -> Vec<Point> {
    let along = p1 - p0;
    let length = along.get_length();
    if length <= K_EH_CLOSE_ENOUGH {
        return generate_filled_circle_strip(scale, p0, radius);
    }

    let divisions = compute_quadrant_divisions(scale * radius);
    let trigs = Trigs::from_divisions(divisions);
    let along_scaled = along * (radius / length);
    let across = Point::new(-along_scaled.y, along_scaled.x);

    let mut pts = Vec::with_capacity(trigs.len() * 4);
    generate_start_round_cap(p0, across, &trigs, &mut pts);
    generate_end_round_cap(p1, across, &trigs, &mut pts);
    pts
}

/// Ported from `Tessellator::FilledEllipse` / `GenerateFilledEllipse` in `tessellator.cc`.
pub fn generate_filled_ellipse_strip(scale: Scalar, bounds: Rect) -> Vec<Point> {
    let w = (bounds.get_right() - bounds.get_left()).abs();
    let h = (bounds.get_bottom() - bounds.get_top()).abs();
    let center = Point::new(bounds.get_left() + w * 0.5, bounds.get_top() + h * 0.5);

    if (w - h).abs() <= Scalar::EPSILON {
        return generate_filled_circle_strip(scale, center, w * 0.5);
    }

    let max_radius = w.max(h);
    let divisions = compute_quadrant_divisions(scale * max_radius);
    let trigs = Trigs::from_divisions(divisions);
    let radii = Size::new(w * 0.5, h * 0.5);
    let mut pts = Vec::with_capacity(trigs.len() * 4);

    // Quadrant 1 connecting with Quadrant 4
    for &trig in trigs.as_slice() {
        let offset = trig.scale_radii(radii);
        pts.push(Point::new(center.x - offset.x, center.y + offset.y));
        pts.push(Point::new(center.x - offset.x, center.y - offset.y));
    }

    // Quadrant 2 connecting with Quadrant 3
    for &trig in trigs.as_slice() {
        let offset = Point::new(
            (trig.sin * radii.width as f64) as Scalar,
            (trig.cos * radii.height as f64) as Scalar,
        );
        pts.push(Point::new(center.x + offset.x, center.y + offset.y));
        pts.push(Point::new(center.x + offset.x, center.y - offset.y));
    }

    pts
}

/// Ported from `Tessellator::FilledRoundRect` / `GenerateFilledRoundRect` in `tessellator.cc`.
pub fn generate_filled_round_rect_strip(scale: Scalar, bounds: Rect, radii: Size) -> Vec<Point> {
    let w = (bounds.get_right() - bounds.get_left()).abs();
    let h = (bounds.get_bottom() - bounds.get_top()).abs();
    let radii = Size::new(
        radii.width.clamp(0.0, w * 0.5),
        radii.height.clamp(0.0, h * 0.5),
    );

    if radii.width * 2.0 < w || radii.height * 2.0 < h {
        let max_radius = radii.width.max(radii.height);
        let divisions = compute_quadrant_divisions(scale * max_radius);
        let trigs = Trigs::from_divisions(divisions);
        let left = bounds.get_left() + radii.width;
        let top = bounds.get_top() + radii.height;
        let right = bounds.get_right() - radii.width;
        let bottom = bounds.get_bottom() - radii.height;

        let mut pts = Vec::with_capacity(trigs.len() * 4);

        // Quadrant 1 connecting with Quadrant 4
        for &trig in trigs.as_slice() {
            let offset = trig.scale_radii(radii);
            pts.push(Point::new(left - offset.x, bottom + offset.y));
            pts.push(Point::new(left - offset.x, top - offset.y));
        }

        // Quadrant 2 connecting with Quadrant 3
        for &trig in trigs.as_slice() {
            let offset = Point::new(
                (trig.sin * radii.width as f64) as Scalar,
                (trig.cos * radii.height as f64) as Scalar,
            );
            pts.push(Point::new(right + offset.x, bottom + offset.y));
            pts.push(Point::new(right + offset.x, top - offset.y));
        }

        pts
    } else {
        generate_filled_ellipse_strip(scale, bounds)
    }
}

/// Ported from `FillPathGeometry` in `fill_path_geometry.cc`.
pub struct FillPathGeometry {
    pub path: Path,
}

impl FillPathGeometry {
    pub fn new(path: Path) -> Self {
        Self { path }
    }

    pub fn get_mesh(&self, scale: Scalar) -> TessellatedMesh {
        Tessellator::tessellate_fill(&self.path, scale)
    }

    pub fn get_coverage(&self, transform: &Matrix) -> Option<Rect> {
        let bbox = self.path.get_bounding_box()?;
        Some(transform_rect_bounds(transform, bbox))
    }

    pub fn covers_area(&self, transform: &Matrix, inner_rect: Option<Rect>, area: Rect) -> bool {
        let Some(inner) = inner_rect else {
            return false;
        };
        if !is_translation_scale_only(transform) {
            return false;
        }
        let coverage = transform_rect_bounds(transform, inner);
        rect_contains_rect(coverage, area)
    }
}

/// Ported from `StrokePathGeometry` in `stroke_path_geometry.cc`.
pub struct StrokePathGeometry {
    pub path: Path,
    pub stroke: StrokeParameters,
}

impl StrokePathGeometry {
    pub fn new(path: Path, stroke: StrokeParameters) -> Self {
        Self { path, stroke }
    }

    pub fn get_mesh(&self, scale: Scalar) -> TessellatedMesh {
        Tessellator::tessellate_stroke(&self.path, self.stroke, scale)
    }

    pub fn generate_strip_vertices(&self, scale: Scalar) -> Vec<Point> {
        generate_solid_stroke_vertices(&self.path, self.stroke, scale)
    }

    pub fn compute_alpha_coverage(&self, transform: &Matrix) -> Scalar {
        compute_stroke_alpha_coverage(transform, self.stroke.width)
    }
}

/// Ported from `FillRectGeometry` and `StrokeRectGeometry` in `rect_geometry.cc`.
pub struct RectGeometry {
    pub rect: Rect,
}

impl RectGeometry {
    pub fn new(rect: Rect) -> Self {
        Self { rect }
    }

    pub fn get_mesh(&self, _scale: Scalar) -> TessellatedMesh {
        if self.rect.is_empty() {
            return TessellatedMesh::default();
        }
        let l = self.rect.get_left();
        let t = self.rect.get_top();
        let r = self.rect.get_right();
        let b = self.rect.get_bottom();
        let vertices = vec![
            Vertex2D::new(l, t),
            Vertex2D::new(r, t),
            Vertex2D::new(l, b),
            Vertex2D::new(r, b),
        ];
        let indices = vec![0, 1, 2, 2, 1, 3];
        TessellatedMesh { vertices, indices }
    }

    pub fn get_stroked_mesh(&self, stroke: StrokeParameters, scale: Scalar) -> TessellatedMesh {
        let strip = self.generate_stroked_strip(stroke, scale);
        strip_points_to_mesh(&strip)
    }

    pub fn generate_stroked_strip(&self, stroke: StrokeParameters, scale: Scalar) -> Vec<Point> {
        if stroke.width < 0.0 || scale <= 0.0 {
            return Vec::new();
        }
        let min_size = K_MIN_STROKE_SIZE / scale;
        let stroke_width = stroke.width.max(min_size);
        let hsw = stroke_width * 0.5;
        let left = self.rect.get_left();
        let top = self.rect.get_top();
        let right = self.rect.get_right();
        let bottom = self.rect.get_bottom();
        let min_dim = (right - left).min(bottom - top);
        let interior_filled = stroke_width >= min_dim;
        let join = if stroke.join == Join::Miter && stroke.miter_limit < std::f32::consts::SQRT_2 {
            Join::Bevel
        } else {
            stroke.join
        };

        match join {
            Join::Round => {
                let trigs = Trigs::from_pixel_radius(hsw * scale);
                let mut pts =
                    Vec::with_capacity(trigs.len() * 4 + if interior_filled { 0 } else { 12 });
                for &trig in trigs.as_slice() {
                    let s = trig.sin as Scalar;
                    let c = trig.cos as Scalar;
                    pts.push(Point::new(left - s * hsw, top - c * hsw));
                    pts.push(Point::new(right + s * hsw, top - c * hsw));
                }
                if !interior_filled {
                    pts.push(Point::new(left - hsw, top + hsw));
                    pts.push(Point::new(right + hsw, top + hsw));
                    pts.push(Point::new(right - hsw, top + hsw));
                    pts.push(Point::new(right + hsw, bottom - hsw));
                    pts.push(Point::new(right - hsw, bottom - hsw));
                    pts.push(Point::new(right - hsw, bottom - hsw));
                    pts.push(Point::new(left + hsw, top + hsw));
                    pts.push(Point::new(left + hsw, top + hsw));
                    pts.push(Point::new(left - hsw, top + hsw));
                    pts.push(Point::new(left + hsw, bottom - hsw));
                    pts.push(Point::new(left - hsw, bottom - hsw));
                    pts.push(Point::new(right + hsw, bottom - hsw));
                }
                for &trig in trigs.as_slice() {
                    let s = trig.sin as Scalar;
                    let c = trig.cos as Scalar;
                    pts.push(Point::new(left - c * hsw, bottom + s * hsw));
                    pts.push(Point::new(right + c * hsw, bottom + s * hsw));
                }
                pts
            }
            Join::Bevel => {
                if interior_filled {
                    vec![
                        Point::new(left, top - hsw),
                        Point::new(right, top - hsw),
                        Point::new(left - hsw, top),
                        Point::new(right + hsw, top),
                        Point::new(left - hsw, bottom),
                        Point::new(right + hsw, bottom),
                        Point::new(left, bottom + hsw),
                        Point::new(right, bottom + hsw),
                    ]
                } else {
                    vec![
                        Point::new(left - hsw, top),
                        Point::new(left + hsw, top + hsw),
                        Point::new(left, top - hsw),
                        Point::new(right - hsw, top + hsw),
                        Point::new(right, top - hsw),
                        Point::new(right - hsw, top + hsw),
                        Point::new(right + hsw, top),
                        Point::new(right - hsw, bottom - hsw),
                        Point::new(right + hsw, bottom),
                        Point::new(right - hsw, bottom - hsw),
                        Point::new(right, bottom + hsw),
                        Point::new(left + hsw, bottom - hsw),
                        Point::new(left, bottom + hsw),
                        Point::new(left + hsw, bottom - hsw),
                        Point::new(left - hsw, bottom),
                        Point::new(left + hsw, top + hsw),
                        Point::new(left - hsw, top),
                    ]
                }
            }
            Join::Miter => {
                if interior_filled {
                    vec![
                        Point::new(left - hsw, top - hsw),
                        Point::new(right + hsw, top - hsw),
                        Point::new(left - hsw, bottom + hsw),
                        Point::new(right + hsw, bottom + hsw),
                    ]
                } else {
                    vec![
                        Point::new(left - hsw, top - hsw),
                        Point::new(left + hsw, top + hsw),
                        Point::new(right + hsw, top - hsw),
                        Point::new(right - hsw, top + hsw),
                        Point::new(right + hsw, bottom + hsw),
                        Point::new(right - hsw, bottom - hsw),
                        Point::new(left - hsw, bottom + hsw),
                        Point::new(left + hsw, bottom - hsw),
                        Point::new(left - hsw, top - hsw),
                        Point::new(left + hsw, top + hsw),
                    ]
                }
            }
        }
    }

    pub fn get_coverage(&self, transform: &Matrix) -> Option<Rect> {
        Some(transform_rect_bounds(transform, self.rect))
    }

    pub fn covers_area(&self, transform: &Matrix, area: Rect) -> bool {
        if !is_translation_scale_only(transform) {
            return false;
        }
        let coverage = transform_rect_bounds(transform, self.rect);
        rect_contains_rect(coverage, area)
    }
}

/// Ported from `CircleGeometry` in `circle_geometry.cc`.
pub struct CircleGeometry {
    pub center: Point,
    pub radius: Scalar,
}

impl CircleGeometry {
    pub fn new(center: Point, radius: Scalar) -> Self {
        Self { center, radius }
    }

    pub fn get_mesh(&self, scale: Scalar) -> TessellatedMesh {
        let strip = generate_filled_circle_strip(scale.max(1.0), self.center, self.radius);
        strip_points_to_mesh(&strip)
    }

    pub fn get_stroked_mesh(&self, stroke_width: Scalar, scale: Scalar) -> TessellatedMesh {
        let max_basis = scale.max(1.0);
        let half_width = LineGeometry::compute_pixel_half_width(max_basis, stroke_width);
        let strip = generate_stroked_circle_strip(max_basis, self.center, self.radius, half_width);
        strip_points_to_mesh(&strip)
    }

    pub fn get_coverage(&self, transform: &Matrix) -> Option<Rect> {
        let r = self.radius.max(0.0);
        let bounds = Rect::from_ltrb(
            self.center.x - r,
            self.center.y - r,
            self.center.x + r,
            self.center.y + r,
        );
        Some(transform_rect_bounds(transform, bounds))
    }

    pub fn get_stroked_coverage(&self, stroke_width: Scalar, transform: &Matrix) -> Option<Rect> {
        let outer = self.radius.max(0.0) + stroke_width.max(0.0) * 0.5;
        let bounds = Rect::from_ltrb(
            self.center.x - outer,
            self.center.y - outer,
            self.center.x + outer,
            self.center.y + outer,
        );
        Some(transform_rect_bounds(transform, bounds))
    }
}

fn normalize_round_rect_radii(rect: Rect, radii: RoundingRadii) -> RoundingRadii {
    let w = (rect.get_right() - rect.get_left()).max(0.0);
    let h = (rect.get_bottom() - rect.get_top()).max(0.0);
    let tl = Size::new(
        radii.top_left.width.max(0.0),
        radii.top_left.height.max(0.0),
    );
    let tr = Size::new(
        radii.top_right.width.max(0.0),
        radii.top_right.height.max(0.0),
    );
    let bl = Size::new(
        radii.bottom_left.width.max(0.0),
        radii.bottom_left.height.max(0.0),
    );
    let br = Size::new(
        radii.bottom_right.width.max(0.0),
        radii.bottom_right.height.max(0.0),
    );

    let mut scale_factor = 1.0f32;
    let top_sum = tl.width + tr.width;
    if top_sum > w && top_sum > 0.0 {
        scale_factor = scale_factor.min(w / top_sum);
    }
    let bottom_sum = bl.width + br.width;
    if bottom_sum > w && bottom_sum > 0.0 {
        scale_factor = scale_factor.min(w / bottom_sum);
    }
    let left_sum = tl.height + bl.height;
    if left_sum > h && left_sum > 0.0 {
        scale_factor = scale_factor.min(h / left_sum);
    }
    let right_sum = tr.height + br.height;
    if right_sum > h && right_sum > 0.0 {
        scale_factor = scale_factor.min(h / right_sum);
    }

    if scale_factor < 1.0 {
        RoundingRadii {
            top_left: Size::new(tl.width * scale_factor, tl.height * scale_factor),
            top_right: Size::new(tr.width * scale_factor, tr.height * scale_factor),
            bottom_left: Size::new(bl.width * scale_factor, bl.height * scale_factor),
            bottom_right: Size::new(br.width * scale_factor, br.height * scale_factor),
        }
    } else {
        RoundingRadii {
            top_left: tl,
            top_right: tr,
            bottom_left: bl,
            bottom_right: br,
        }
    }
}

fn max_corner_radius(radii: RoundingRadii) -> Scalar {
    radii
        .top_left
        .width
        .max(radii.top_left.height)
        .max(radii.top_right.width)
        .max(radii.top_right.height)
        .max(radii.bottom_left.width)
        .max(radii.bottom_left.height)
        .max(radii.bottom_right.width)
        .max(radii.bottom_right.height)
}

fn compute_round_rect_divisions(scale: Scalar, max_r: Scalar) -> usize {
    let pixel_r = scale * max_r;
    let mut divisions = compute_quadrant_divisions(pixel_r * 2.0);
    if pixel_r >= 4.0 {
        divisions = divisions.max(16);
    }
    divisions
}

/// Ported from `RoundRectGeometry` and `FillRoundRectGeometry` in `round_rect_geometry.cc`.
pub struct RoundRectGeometry {
    pub rect: Rect,
    pub radii: RoundingRadii,
}

impl RoundRectGeometry {
    pub fn new(rect: Rect, radii: RoundingRadii) -> Self {
        Self { rect, radii }
    }

    pub fn get_mesh(&self, scale: Scalar) -> TessellatedMesh {
        if self.rect.is_empty() {
            return TessellatedMesh::default();
        }
        let scale = scale.max(1.0);
        let norm = normalize_round_rect_radii(self.rect, self.radii);
        let max_r = max_corner_radius(norm);
        if max_r <= 0.0 {
            return RectGeometry::new(self.rect).get_mesh(scale);
        }

        let divisions = compute_round_rect_divisions(scale, max_r);
        let trigs = Trigs::from_divisions(divisions);
        let left = self.rect.get_left();
        let top = self.rect.get_top();
        let right = self.rect.get_right();
        let bottom = self.rect.get_bottom();

        if norm.top_left == norm.top_right
            && norm.top_left == norm.bottom_left
            && norm.top_left == norm.bottom_right
        {
            let radii = norm.top_left;
            let inner_left = left + radii.width;
            let inner_top = top + radii.height;
            let inner_right = right - radii.width;
            let inner_bottom = bottom - radii.height;
            let mut strip = Vec::with_capacity(trigs.len() * 4);

            for &trig in trigs.as_slice() {
                let offset = trig.scale_radii(radii);
                strip.push(Point::new(inner_left - offset.x, inner_bottom + offset.y));
                strip.push(Point::new(inner_left - offset.x, inner_top - offset.y));
            }

            for &trig in trigs.as_slice() {
                let offset = Point::new(
                    (trig.sin * radii.width as f64) as Scalar,
                    (trig.cos * radii.height as f64) as Scalar,
                );
                strip.push(Point::new(inner_right + offset.x, inner_bottom + offset.y));
                strip.push(Point::new(inner_right + offset.x, inner_top - offset.y));
            }

            return strip_points_to_mesh(&strip);
        }

        let mut contour = Vec::with_capacity(trigs.len() * 4);
        let push_unique = |pts: &mut Vec<Point>, p: Point| {
            if pts
                .last()
                .is_none_or(|last| (*last - p).get_length_squared() > 1e-10)
            {
                pts.push(p);
            }
        };

        // 1. Top-Left: (left, top + tl.h) -> (left + tl.w, top)
        let tl_c = Point::new(left + norm.top_left.width, top + norm.top_left.height);
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            push_unique(
                &mut contour,
                Point::new(
                    tl_c.x - c * norm.top_left.width,
                    tl_c.y - s * norm.top_left.height,
                ),
            );
        }

        // 2. Top-Right: (right - tr.w, top) -> (right, top + tr.h)
        let tr_c = Point::new(right - norm.top_right.width, top + norm.top_right.height);
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            push_unique(
                &mut contour,
                Point::new(
                    tr_c.x + s * norm.top_right.width,
                    tr_c.y - c * norm.top_right.height,
                ),
            );
        }

        // 3. Bottom-Right: (right, bottom - br.h) -> (right - br.w, bottom)
        let br_c = Point::new(
            right - norm.bottom_right.width,
            bottom - norm.bottom_right.height,
        );
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            push_unique(
                &mut contour,
                Point::new(
                    br_c.x + c * norm.bottom_right.width,
                    br_c.y + s * norm.bottom_right.height,
                ),
            );
        }

        // 4. Bottom-Left: (left + bl.w, bottom) -> (left, bottom - bl.h)
        let bl_c = Point::new(
            left + norm.bottom_left.width,
            bottom - norm.bottom_left.height,
        );
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            push_unique(
                &mut contour,
                Point::new(
                    bl_c.x - s * norm.bottom_left.width,
                    bl_c.y + c * norm.bottom_left.height,
                ),
            );
        }

        if contour.len() >= 2
            && (contour[0] - *contour.last().unwrap()).get_length_squared() <= 1e-10
        {
            contour.pop();
        }
        if contour.len() < 3 {
            return TessellatedMesh::default();
        }

        let mut strip_indices = Vec::with_capacity(contour.len());
        strip_indices.push(0);
        let mut a = 1usize;
        let mut b = contour.len() - 1;
        while a < b {
            strip_indices.push(a as u32);
            strip_indices.push(b as u32);
            a += 1;
            b -= 1;
        }
        if a == b {
            strip_indices.push(a as u32);
        }
        indexed_strip_to_mesh(&contour, &strip_indices)
    }

    pub fn get_stroked_mesh(&self, stroke: StrokeParameters, scale: Scalar) -> TessellatedMesh {
        if self.rect.is_empty() || stroke.width < 0.0 || scale <= 0.0 {
            return TessellatedMesh::default();
        }
        let scale = scale.max(1.0);
        let min_size = K_MIN_STROKE_SIZE / scale.max(1e-4);
        let half_width = stroke.width.max(min_size) * 0.5;

        let norm = normalize_round_rect_radii(self.rect, self.radii);
        let max_base_r = max_corner_radius(norm);
        if max_base_r <= 0.0 {
            return RectGeometry::new(self.rect).get_stroked_mesh(stroke, scale);
        }

        let left = self.rect.get_left();
        let top = self.rect.get_top();
        let right = self.rect.get_right();
        let bottom = self.rect.get_bottom();

        let expand_corner = |s: Size| -> Size {
            if s.width <= 0.0 && s.height <= 0.0 && stroke.join != Join::Round {
                Size::ZERO
            } else {
                Size::new(s.width + half_width, s.height + half_width)
            }
        };

        let outer_rect = Rect::from_ltrb(
            left - half_width,
            top - half_width,
            right + half_width,
            bottom + half_width,
        );
        let outer_radii = normalize_round_rect_radii(
            outer_rect,
            RoundingRadii {
                top_left: expand_corner(norm.top_left),
                top_right: expand_corner(norm.top_right),
                bottom_left: expand_corner(norm.bottom_left),
                bottom_right: expand_corner(norm.bottom_right),
            },
        );

        let inner_w = (right - left) - half_width * 2.0;
        let inner_h = (bottom - top) - half_width * 2.0;
        if inner_w <= 0.0 || inner_h <= 0.0 {
            return Self::new(outer_rect, outer_radii).get_mesh(scale);
        }

        let inner_rect = Rect::from_ltrb(
            left + half_width,
            top + half_width,
            right - half_width,
            bottom - half_width,
        );
        let inner_radii = normalize_round_rect_radii(
            inner_rect,
            RoundingRadii {
                top_left: Size::new(
                    (norm.top_left.width - half_width).max(0.0),
                    (norm.top_left.height - half_width).max(0.0),
                ),
                top_right: Size::new(
                    (norm.top_right.width - half_width).max(0.0),
                    (norm.top_right.height - half_width).max(0.0),
                ),
                bottom_left: Size::new(
                    (norm.bottom_left.width - half_width).max(0.0),
                    (norm.bottom_left.height - half_width).max(0.0),
                ),
                bottom_right: Size::new(
                    (norm.bottom_right.width - half_width).max(0.0),
                    (norm.bottom_right.height - half_width).max(0.0),
                ),
            },
        );

        let max_r = max_corner_radius(outer_radii);
        let divisions = compute_round_rect_divisions(scale, max_r);
        let trigs = Trigs::from_divisions(divisions);

        let mut strip = Vec::with_capacity(trigs.len() * 8 + 2);

        // 1. Top-Left
        let out_tl_c = Point::new(
            outer_rect.get_left() + outer_radii.top_left.width,
            outer_rect.get_top() + outer_radii.top_left.height,
        );
        let in_tl_c = Point::new(
            inner_rect.get_left() + inner_radii.top_left.width,
            inner_rect.get_top() + inner_radii.top_left.height,
        );
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            strip.push(Point::new(
                out_tl_c.x - c * outer_radii.top_left.width,
                out_tl_c.y - s * outer_radii.top_left.height,
            ));
            strip.push(Point::new(
                in_tl_c.x - c * inner_radii.top_left.width,
                in_tl_c.y - s * inner_radii.top_left.height,
            ));
        }

        // 2. Top-Right
        let out_tr_c = Point::new(
            outer_rect.get_right() - outer_radii.top_right.width,
            outer_rect.get_top() + outer_radii.top_right.height,
        );
        let in_tr_c = Point::new(
            inner_rect.get_right() - inner_radii.top_right.width,
            inner_rect.get_top() + inner_radii.top_right.height,
        );
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            strip.push(Point::new(
                out_tr_c.x + s * outer_radii.top_right.width,
                out_tr_c.y - c * outer_radii.top_right.height,
            ));
            strip.push(Point::new(
                in_tr_c.x + s * inner_radii.top_right.width,
                in_tr_c.y - c * inner_radii.top_right.height,
            ));
        }

        // 3. Bottom-Right
        let out_br_c = Point::new(
            outer_rect.get_right() - outer_radii.bottom_right.width,
            outer_rect.get_bottom() - outer_radii.bottom_right.height,
        );
        let in_br_c = Point::new(
            inner_rect.get_right() - inner_radii.bottom_right.width,
            inner_rect.get_bottom() - inner_radii.bottom_right.height,
        );
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            strip.push(Point::new(
                out_br_c.x + c * outer_radii.bottom_right.width,
                out_br_c.y + s * outer_radii.bottom_right.height,
            ));
            strip.push(Point::new(
                in_br_c.x + c * inner_radii.bottom_right.width,
                in_br_c.y + s * inner_radii.bottom_right.height,
            ));
        }

        // 4. Bottom-Left
        let out_bl_c = Point::new(
            outer_rect.get_left() + outer_radii.bottom_left.width,
            outer_rect.get_bottom() - outer_radii.bottom_left.height,
        );
        let in_bl_c = Point::new(
            inner_rect.get_left() + inner_radii.bottom_left.width,
            inner_rect.get_bottom() - inner_radii.bottom_left.height,
        );
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            strip.push(Point::new(
                out_bl_c.x - s * outer_radii.bottom_left.width,
                out_bl_c.y + c * outer_radii.bottom_left.height,
            ));
            strip.push(Point::new(
                in_bl_c.x - s * inner_radii.bottom_left.width,
                in_bl_c.y + c * inner_radii.bottom_left.height,
            ));
        }

        if strip.len() >= 2 {
            let first_outer = strip[0];
            let first_inner = strip[1];
            strip.push(first_outer);
            strip.push(first_inner);
        }

        strip_points_to_mesh(&strip)
    }

    pub fn get_stroked_coverage(&self, stroke_width: Scalar, transform: &Matrix) -> Option<Rect> {
        let half_width = stroke_width.max(0.0) * 0.5;
        let outer = Rect::from_ltrb(
            self.rect.get_left() - half_width,
            self.rect.get_top() - half_width,
            self.rect.get_right() + half_width,
            self.rect.get_bottom() + half_width,
        );
        Some(transform_rect_bounds(transform, outer))
    }

    pub fn get_coverage(&self, transform: &Matrix) -> Option<Rect> {
        Some(transform_rect_bounds(transform, self.rect))
    }

    pub fn covers_area(&self, transform: &Matrix, area: Rect) -> bool {
        if !is_translation_scale_only(transform) {
            return false;
        }
        let left_margin = self.radii.top_left.width.max(self.radii.bottom_left.width);
        let right_margin = self
            .radii
            .top_right
            .width
            .max(self.radii.bottom_right.width);
        let top_margin = self.radii.top_left.height.max(self.radii.top_right.height);
        let bottom_margin = self
            .radii
            .bottom_left
            .height
            .max(self.radii.bottom_right.height);

        let w = self.rect.get_right() - self.rect.get_left();
        let h = self.rect.get_bottom() - self.rect.get_top();
        let flat_on_tb = w > left_margin + right_margin;
        let flat_on_lr = h > top_margin + bottom_margin;
        if !flat_on_tb && !flat_on_lr {
            return false;
        }

        if flat_on_tb {
            let vertical_bounds = Rect::from_ltrb(
                self.rect.get_left() + left_margin,
                self.rect.get_top(),
                self.rect.get_right() - right_margin,
                self.rect.get_bottom(),
            );
            let coverage = transform_rect_bounds(transform, vertical_bounds);
            if rect_contains_rect(coverage, area) {
                return true;
            }
        }
        if flat_on_lr {
            let horizontal_bounds = Rect::from_ltrb(
                self.rect.get_left(),
                self.rect.get_top() + top_margin,
                self.rect.get_right(),
                self.rect.get_bottom() - bottom_margin,
            );
            let coverage = transform_rect_bounds(transform, horizontal_bounds);
            if rect_contains_rect(coverage, area) {
                return true;
            }
        }
        false
    }
}

/// Ported from `LineGeometry` in `line_geometry.cc`.
pub struct LineGeometry {
    pub p0: Point,
    pub p1: Point,
    pub width: Scalar,
    pub cap: Cap,
}

impl LineGeometry {
    pub fn new(p0: Point, p1: Point, width: Scalar, cap: Cap) -> Self {
        Self { p0, p1, width, cap }
    }

    pub fn compute_pixel_half_width(scale: Scalar, width: Scalar) -> Scalar {
        if scale <= 0.0 {
            return 0.0;
        }
        let min_size = K_MIN_STROKE_SIZE / scale;
        width.max(min_size) * 0.5
    }

    pub fn compute_along_vector(
        scale: Scalar,
        allow_zero_length: bool,
        p0: Point,
        p1: Point,
        width: Scalar,
    ) -> Point {
        let stroke_half_width = Self::compute_pixel_half_width(scale, width);
        if stroke_half_width < K_EH_CLOSE_ENOUGH {
            return Point::ZERO;
        }
        let along = p1 - p0;
        let length = along.get_length();
        if length < K_EH_CLOSE_ENOUGH {
            if !allow_zero_length {
                return Point::ZERO;
            }
            Point::new(stroke_half_width, 0.0)
        } else {
            along * (stroke_half_width / length)
        }
    }

    pub fn compute_corners(
        scale: Scalar,
        extend_endpoints: bool,
        p0: Point,
        p1: Point,
        width: Scalar,
    ) -> Option<[Point; 4]> {
        let along = Self::compute_along_vector(scale, extend_endpoints, p0, p1, width);
        if along == Point::ZERO {
            return None;
        }
        let across = Point::new(along.y, -along.x);
        let mut corners = [p0 - across, p1 - across, p0 + across, p1 + across];
        if extend_endpoints {
            corners[0] -= along;
            corners[1] += along;
            corners[2] -= along;
            corners[3] += along;
        }
        Some(corners)
    }

    pub fn get_mesh(&self, scale: Scalar) -> TessellatedMesh {
        let scale = scale.max(1.0);
        let radius = Self::compute_pixel_half_width(scale, self.width);
        if self.cap == Cap::Round {
            let strip = generate_round_cap_line_strip(scale, self.p0, self.p1, radius);
            return strip_points_to_mesh(&strip);
        }

        let Some(corners) =
            Self::compute_corners(scale, self.cap == Cap::Square, self.p0, self.p1, self.width)
        else {
            return TessellatedMesh::default();
        };

        let strip = [corners[0], corners[1], corners[2], corners[3]];
        strip_points_to_mesh(&strip)
    }

    pub fn get_coverage(&self, transform: &Matrix) -> Option<Rect> {
        let scale = transform.get_max_basis_length_xy();
        let corners =
            Self::compute_corners(scale, self.cap != Cap::Butt, self.p0, self.p1, self.width)?;
        let transformed = [
            transform.transform_point(corners[0]),
            transform.transform_point(corners[1]),
            transform.transform_point(corners[2]),
            transform.transform_point(corners[3]),
        ];
        let min_x = transformed
            .iter()
            .map(|p| p.x)
            .fold(Scalar::INFINITY, Scalar::min);
        let min_y = transformed
            .iter()
            .map(|p| p.y)
            .fold(Scalar::INFINITY, Scalar::min);
        let max_x = transformed
            .iter()
            .map(|p| p.x)
            .fold(Scalar::NEG_INFINITY, Scalar::max);
        let max_y = transformed
            .iter()
            .map(|p| p.y)
            .fold(Scalar::NEG_INFINITY, Scalar::max);
        Some(Rect::from_ltrb(min_x, min_y, max_x, max_y))
    }

    pub fn is_axis_aligned_rect(&self) -> bool {
        self.cap != Cap::Round && (self.p0.x == self.p1.x || self.p0.y == self.p1.y)
    }

    pub fn covers_area(&self, transform: &Matrix, area: Rect) -> bool {
        if !is_translation_scale_only(transform) || !self.is_axis_aligned_rect() {
            return false;
        }
        let Some(coverage) = self.get_coverage(transform) else {
            return false;
        };
        rect_contains_rect(coverage, area)
    }
}

/// Ported from `EllipseGeometry` in `ellipse_geometry.cc`.
pub struct EllipseGeometry {
    pub bounds: Rect,
}

impl EllipseGeometry {
    pub fn new(bounds: Rect) -> Self {
        Self { bounds }
    }

    pub fn get_mesh(&self, scale: Scalar) -> TessellatedMesh {
        let strip = generate_filled_ellipse_strip(scale.max(1.0), self.bounds);
        strip_points_to_mesh(&strip)
    }

    pub fn get_coverage(&self, transform: &Matrix) -> Option<Rect> {
        Some(transform_rect_bounds(transform, self.bounds))
    }
}

/// Ported from `ArcGeometry` in `arc_geometry.cc` (extended for elliptical `oval_bounds`).
pub struct ArcGeometry {
    pub oval_bounds: Rect,
    pub start: Radians,
    pub sweep: Radians,
    pub use_center: bool,
}

impl ArcGeometry {
    pub fn new(oval_bounds: Rect, start: Radians, sweep: Radians, use_center: bool) -> Self {
        Self {
            oval_bounds,
            start,
            sweep,
            use_center,
        }
    }

    pub fn get_mesh(&self, scale: Scalar) -> TessellatedMesh {
        let bounds = self.oval_bounds.get_positive();
        if bounds.is_empty()
            || !self.start.0.is_finite()
            || !self.sweep.0.is_finite()
            || self.sweep.0.abs() <= 1e-6
        {
            return TessellatedMesh::default();
        }

        let scale = scale.max(1.0);
        let rx = bounds.get_width() * 0.5;
        let ry = bounds.get_height() * 0.5;
        let center = Point::new(bounds.get_left() + rx, bounds.get_top() + ry);

        let sweep_rad = self
            .sweep
            .0
            .clamp(-std::f32::consts::TAU, std::f32::consts::TAU);
        let sweep_abs = sweep_rad.abs();
        let full_circle = sweep_abs >= std::f32::consts::TAU - 1e-5;
        if full_circle {
            return EllipseGeometry::new(bounds).get_mesh(scale);
        }

        let max_radius = rx.max(ry);
        let quad_divs = compute_quadrant_divisions(scale * max_radius);
        let quadrants = (sweep_abs / std::f32::consts::FRAC_PI_2).ceil().max(1.0) as usize;
        let steps = (quad_divs * quadrants).max(2);

        let start_f64 = self.start.0 as f64;
        let step_f64 = (sweep_rad as f64) / (steps as f64);
        let (mut sin_a, mut cos_a) = start_f64.sin_cos();
        let (sin_step, cos_step) = step_f64.sin_cos();

        if self.use_center {
            let mut vertices = Vec::with_capacity(steps + 2);
            let mut indices = Vec::with_capacity(steps * 3);
            vertices.push(Vertex2D::from_point(center));
            for i in 0..=steps {
                let p = Point::new(
                    center.x + (cos_a as Scalar) * rx,
                    center.y + (sin_a as Scalar) * ry,
                );
                vertices.push(Vertex2D::from_point(p));
                if i > 0 {
                    indices.push(0);
                    indices.push(i as u32);
                    indices.push((i + 1) as u32);
                }
                let next_cos = cos_a * cos_step - sin_a * sin_step;
                let next_sin = sin_a * cos_step + cos_a * sin_step;
                cos_a = next_cos;
                sin_a = next_sin;
            }
            TessellatedMesh { vertices, indices }
        } else {
            let mut vertices = Vec::with_capacity(steps + 1);
            let mut indices = Vec::with_capacity((steps - 1) * 3);
            for i in 0..=steps {
                let p = Point::new(
                    center.x + (cos_a as Scalar) * rx,
                    center.y + (sin_a as Scalar) * ry,
                );
                vertices.push(Vertex2D::from_point(p));
                if i >= 2 {
                    indices.push(0);
                    indices.push((i - 1) as u32);
                    indices.push(i as u32);
                }
                let next_cos = cos_a * cos_step - sin_a * sin_step;
                let next_sin = sin_a * cos_step + cos_a * sin_step;
                cos_a = next_cos;
                sin_a = next_sin;
            }
            TessellatedMesh { vertices, indices }
        }
    }

    pub fn get_stroked_mesh(&self, stroke: StrokeParameters, scale: Scalar) -> TessellatedMesh {
        let bounds = self.oval_bounds.get_positive();
        if bounds.is_empty()
            || stroke.width < 0.0
            || scale <= 0.0
            || !self.start.0.is_finite()
            || !self.sweep.0.is_finite()
            || self.sweep.0.abs() <= 1e-6
        {
            return TessellatedMesh::default();
        }

        let scale = scale.max(1.0);
        let min_size = K_MIN_STROKE_SIZE / scale;
        let half_width = stroke.width.max(min_size) * 0.5;
        let rx = bounds.get_width() * 0.5;
        let ry = bounds.get_height() * 0.5;
        let center = Point::new(bounds.get_left() + rx, bounds.get_top() + ry);
        let is_circle = (rx - ry).abs() <= 1e-4;

        let sweep_rad = self
            .sweep
            .0
            .clamp(-std::f32::consts::TAU, std::f32::consts::TAU);
        let sweep_abs = sweep_rad.abs();
        let full_circle = sweep_abs >= std::f32::consts::TAU - 1e-5;

        if full_circle && is_circle {
            let strip = generate_stroked_circle_strip(scale, center, rx, half_width);
            return strip_points_to_mesh(&strip);
        }

        let max_radius = rx.max(ry) + half_width;
        let quad_divs = compute_quadrant_divisions(scale * max_radius);
        let quadrants = (sweep_abs / std::f32::consts::FRAC_PI_2).ceil().max(1.0) as usize;
        let steps = (quad_divs * quadrants).max(2);

        let start_f64 = self.start.0 as f64;
        let step_f64 = (sweep_rad as f64) / (steps as f64);
        let (mut sin_a, mut cos_a) = start_f64.sin_cos();
        let (sin_step, cos_step) = step_f64.sin_cos();

        if self.use_center && !full_circle {
            let mut pb = PathBuilder::new();
            pb.move_to(center);
            for _ in 0..=steps {
                let p = Point::new(
                    center.x + (cos_a as Scalar) * rx,
                    center.y + (sin_a as Scalar) * ry,
                );
                pb.line_to(p);
                let next_cos = cos_a * cos_step - sin_a * sin_step;
                let next_sin = sin_a * cos_step + cos_a * sin_step;
                cos_a = next_cos;
                sin_a = next_sin;
            }
            pb.close();
            let path = pb.build(FillType::NonZero);
            return Tessellator::tessellate_stroke(&path, stroke, scale);
        }

        let unit_normal = |c: Scalar, s: Scalar| -> Point {
            if is_circle {
                Point::new(c, s)
            } else {
                let nx = ry * c;
                let ny = rx * s;
                let len = (nx * nx + ny * ny).sqrt();
                if len > 1e-8 {
                    Point::new(nx / len, ny / len)
                } else {
                    Point::new(c, s)
                }
            }
        };

        let sign: Scalar = if sweep_rad >= 0.0 { 1.0 } else { -1.0 };
        let cap_trigs = if !full_circle && stroke.cap == Cap::Round {
            Some(Trigs::from_divisions(compute_quadrant_divisions(
                scale * half_width,
            )))
        } else {
            None
        };
        let cap_extra = match (&cap_trigs, !full_circle && stroke.cap == Cap::Square) {
            (Some(t), _) => t.len() * 4,
            (None, true) => 4,
            (None, false) => 0,
        };

        let mut strip = Vec::with_capacity((steps + 1) * 2 + cap_extra);

        let c0 = cos_a as Scalar;
        let s0 = sin_a as Scalar;
        let p0 = Point::new(center.x + c0 * rx, center.y + s0 * ry);
        let n0 = unit_normal(c0, s0);
        let t0 = Point::new(-n0.y * sign, n0.x * sign);

        if !full_circle {
            match stroke.cap {
                Cap::Butt => {}
                Cap::Square => {
                    let ext = t0 * half_width;
                    let across = n0 * half_width;
                    strip.push(p0 + across - ext);
                    strip.push(p0 - across - ext);
                }
                Cap::Round => {
                    if let Some(ref trigs) = cap_trigs {
                        let along = t0 * half_width;
                        let across = n0 * half_width;
                        for &trig in trigs.as_slice() {
                            let tc = trig.cos as Scalar;
                            let ts = trig.sin as Scalar;
                            strip.push(p0 - along * tc + across * ts);
                            strip.push(p0 - along * tc - across * ts);
                        }
                    }
                }
            }
        }

        let mut last_p = p0;
        let mut last_n = n0;
        for i in 0..=steps {
            let c = cos_a as Scalar;
            let s = sin_a as Scalar;
            let p = Point::new(center.x + c * rx, center.y + s * ry);
            let n = unit_normal(c, s);
            let outer_p = p + n * half_width;
            let inner_p = if is_circle && half_width >= rx {
                center
            } else {
                p - n * half_width
            };
            strip.push(outer_p);
            strip.push(inner_p);
            if i == steps {
                last_p = p;
                last_n = n;
            } else {
                let next_cos = cos_a * cos_step - sin_a * sin_step;
                let next_sin = sin_a * cos_step + cos_a * sin_step;
                cos_a = next_cos;
                sin_a = next_sin;
            }
        }

        if full_circle {
            if strip.len() >= 2 {
                let first_outer = strip[0];
                let first_inner = strip[1];
                strip.push(first_outer);
                strip.push(first_inner);
            }
        } else {
            let tn = Point::new(-last_n.y * sign, last_n.x * sign);
            match stroke.cap {
                Cap::Butt => {}
                Cap::Square => {
                    let ext = tn * half_width;
                    let across = last_n * half_width;
                    strip.push(last_p + across + ext);
                    strip.push(last_p - across + ext);
                }
                Cap::Round => {
                    if let Some(ref trigs) = cap_trigs {
                        let along = tn * half_width;
                        let across = last_n * half_width;
                        for &trig in trigs.as_slice() {
                            let tc = trig.cos as Scalar;
                            let ts = trig.sin as Scalar;
                            strip.push(last_p + along * ts + across * tc);
                            strip.push(last_p + along * ts - across * tc);
                        }
                    }
                }
            }
        }

        strip_points_to_mesh(&strip)
    }

    pub fn get_coverage(&self, transform: &Matrix) -> Option<Rect> {
        Some(transform_rect_bounds(
            transform,
            self.oval_bounds.get_positive(),
        ))
    }

    pub fn get_stroked_coverage(&self, stroke_width: Scalar, transform: &Matrix) -> Option<Rect> {
        let half_width = stroke_width.max(0.0) * 0.5;
        let b = self.oval_bounds.get_positive();
        let outer = Rect::from_ltrb(
            b.get_left() - half_width,
            b.get_top() - half_width,
            b.get_right() + half_width,
            b.get_bottom() + half_width,
        );
        Some(transform_rect_bounds(transform, outer))
    }
}

/// Analytic nested rounded-rectangle ring geometry (`outer` minus `inner`) matching
/// Flutter `Canvas::drawDRRect` / `BoxDecoration` borders.
pub struct DiffRoundRectGeometry {
    pub outer_rect: Rect,
    pub outer_radii: RoundingRadii,
    pub inner_rect: Rect,
    pub inner_radii: RoundingRadii,
}

impl DiffRoundRectGeometry {
    pub const fn new(
        outer_rect: Rect,
        outer_radii: RoundingRadii,
        inner_rect: Rect,
        inner_radii: RoundingRadii,
    ) -> Self {
        Self {
            outer_rect,
            outer_radii,
            inner_rect,
            inner_radii,
        }
    }

    pub fn get_mesh(&self, scale: Scalar) -> Option<TessellatedMesh> {
        if self.outer_rect.is_empty() {
            return Some(TessellatedMesh::default());
        }
        if self.inner_rect.is_empty() {
            return Some(RoundRectGeometry::new(self.outer_rect, self.outer_radii).get_mesh(scale));
        }

        let outer_rect = self.outer_rect.get_positive();
        let inner_rect = self.inner_rect.get_positive();
        if !rect_contains_rect(outer_rect, inner_rect) {
            return None;
        }

        let scale = scale.max(1.0);
        let outer_radii = normalize_round_rect_radii(outer_rect, self.outer_radii);
        let inner_radii = normalize_round_rect_radii(inner_rect, self.inner_radii);

        let out_tl_c = Point::new(
            outer_rect.get_left() + outer_radii.top_left.width,
            outer_rect.get_top() + outer_radii.top_left.height,
        );
        let in_tl_c = Point::new(
            inner_rect.get_left() + inner_radii.top_left.width,
            inner_rect.get_top() + inner_radii.top_left.height,
        );
        let out_tr_c = Point::new(
            outer_rect.get_right() - outer_radii.top_right.width,
            outer_rect.get_top() + outer_radii.top_right.height,
        );
        let in_tr_c = Point::new(
            inner_rect.get_right() - inner_radii.top_right.width,
            inner_rect.get_top() + inner_radii.top_right.height,
        );
        let out_br_c = Point::new(
            outer_rect.get_right() - outer_radii.bottom_right.width,
            outer_rect.get_bottom() - outer_radii.bottom_right.height,
        );
        let in_br_c = Point::new(
            inner_rect.get_right() - inner_radii.bottom_right.width,
            inner_rect.get_bottom() - inner_radii.bottom_right.height,
        );
        let out_bl_c = Point::new(
            outer_rect.get_left() + outer_radii.bottom_left.width,
            outer_rect.get_bottom() - outer_radii.bottom_left.height,
        );
        let in_bl_c = Point::new(
            inner_rect.get_left() + inner_radii.bottom_left.width,
            inner_rect.get_bottom() - inner_radii.bottom_left.height,
        );

        let k = std::f32::consts::FRAC_1_SQRT_2;
        let out_tl_mid = Point::new(
            out_tl_c.x - k * outer_radii.top_left.width,
            out_tl_c.y - k * outer_radii.top_left.height,
        );
        let in_tl_mid = Point::new(
            in_tl_c.x - k * inner_radii.top_left.width,
            in_tl_c.y - k * inner_radii.top_left.height,
        );
        let out_tr_mid = Point::new(
            out_tr_c.x + k * outer_radii.top_right.width,
            out_tr_c.y - k * outer_radii.top_right.height,
        );
        let in_tr_mid = Point::new(
            in_tr_c.x + k * inner_radii.top_right.width,
            in_tr_c.y - k * inner_radii.top_right.height,
        );
        let out_br_mid = Point::new(
            out_br_c.x + k * outer_radii.bottom_right.width,
            out_br_c.y + k * outer_radii.bottom_right.height,
        );
        let in_br_mid = Point::new(
            in_br_c.x + k * inner_radii.bottom_right.width,
            in_br_c.y + k * inner_radii.bottom_right.height,
        );
        let out_bl_mid = Point::new(
            out_bl_c.x - k * outer_radii.bottom_left.width,
            out_bl_c.y + k * outer_radii.bottom_left.height,
        );
        let in_bl_mid = Point::new(
            in_bl_c.x - k * inner_radii.bottom_left.width,
            in_bl_c.y + k * inner_radii.bottom_left.height,
        );

        let tl_ok = in_tl_mid.x >= out_tl_mid.x - 1e-4 && in_tl_mid.y >= out_tl_mid.y - 1e-4;
        let tr_ok = in_tr_mid.x <= out_tr_mid.x + 1e-4 && in_tr_mid.y >= out_tr_mid.y - 1e-4;
        let br_ok = in_br_mid.x <= out_br_mid.x + 1e-4 && in_br_mid.y <= out_br_mid.y + 1e-4;
        let bl_ok = in_bl_mid.x >= out_bl_mid.x - 1e-4 && in_bl_mid.y <= out_bl_mid.y + 1e-4;
        if !(tl_ok && tr_ok && br_ok && bl_ok) {
            return None;
        }

        let max_r = max_corner_radius(outer_radii).max(max_corner_radius(inner_radii));
        let divisions = compute_round_rect_divisions(scale, max_r);
        let trigs = Trigs::from_divisions(divisions);
        let mut strip = Vec::with_capacity(trigs.len() * 8 + 2);

        // 1. Top-Left
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            strip.push(Point::new(
                out_tl_c.x - c * outer_radii.top_left.width,
                out_tl_c.y - s * outer_radii.top_left.height,
            ));
            strip.push(Point::new(
                in_tl_c.x - c * inner_radii.top_left.width,
                in_tl_c.y - s * inner_radii.top_left.height,
            ));
        }

        // 2. Top-Right
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            strip.push(Point::new(
                out_tr_c.x + s * outer_radii.top_right.width,
                out_tr_c.y - c * outer_radii.top_right.height,
            ));
            strip.push(Point::new(
                in_tr_c.x + s * inner_radii.top_right.width,
                in_tr_c.y - c * inner_radii.top_right.height,
            ));
        }

        // 3. Bottom-Right
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            strip.push(Point::new(
                out_br_c.x + c * outer_radii.bottom_right.width,
                out_br_c.y + s * outer_radii.bottom_right.height,
            ));
            strip.push(Point::new(
                in_br_c.x + c * inner_radii.bottom_right.width,
                in_br_c.y + s * inner_radii.bottom_right.height,
            ));
        }

        // 4. Bottom-Left
        for &trig in trigs.as_slice() {
            let c = trig.cos as Scalar;
            let s = trig.sin as Scalar;
            strip.push(Point::new(
                out_bl_c.x - s * outer_radii.bottom_left.width,
                out_bl_c.y + c * outer_radii.bottom_left.height,
            ));
            strip.push(Point::new(
                in_bl_c.x - s * inner_radii.bottom_left.width,
                in_bl_c.y + c * inner_radii.bottom_left.height,
            ));
        }

        if strip.len() >= 2 {
            let first_outer = strip[0];
            let first_inner = strip[1];
            strip.push(first_outer);
            strip.push(first_inner);
        }

        Some(strip_points_to_mesh(&strip))
    }
}
