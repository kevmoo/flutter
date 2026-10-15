use crate::{TessellatedMesh, Vertex2D};
use impeller_geometry::{Point, Scalar};
use impeller_path::{CubicPathComponent, Path, PathComponent, QuadPathComponent};

const K_WANG_PRECISION: Scalar = 16.0;

/// Ported from `impeller::ComputeQuadradicSubdivisions` in `wangs_formula.cc`.
#[inline]
pub(crate) fn wangs_quadratic_subdivisions(
    scale_factor: Scalar,
    p0: Point,
    p1: Point,
    p2: Point,
) -> Scalar {
    let k = scale_factor * 0.25 * K_WANG_PRECISION;
    (k * (p0 - p1 * 2.0 + p2).get_length()).sqrt()
}

/// Ported from `impeller::ComputeCubicSubdivisions` in `wangs_formula.cc`.
#[inline]
pub(crate) fn wangs_cubic_subdivisions(
    scale_factor: Scalar,
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
) -> Scalar {
    let k = scale_factor * 0.75 * K_WANG_PRECISION;
    let a = p0 - p1 * 2.0 + p2;
    let b = p1 - p2 * 2.0 + p3;
    let max_len_sq = a.get_length_squared().max(b.get_length_squared());
    (k * max_len_sq.sqrt()).sqrt()
}

#[inline]
pub(crate) fn solve_quad(q: &QuadPathComponent, t: Scalar) -> Point {
    let u = 1.0 - t;
    q.p1 * (u * u) + q.cp * (2.0 * u * t) + q.p2 * (t * t)
}

#[inline]
pub(crate) fn solve_cubic(c: &CubicPathComponent, t: Scalar) -> Point {
    let u = 1.0 - t;
    c.p1 * (u * u * u) + c.cp1 * (3.0 * u * u * t) + c.cp2 * (3.0 * u * t * t) + c.p2 * (t * t * t)
}

struct GlesPathVertexWriter {
    points: Vec<Point>,
    indices: Vec<u32>,
    previous_contour_odd_points: bool,
    contour_start: usize,
}

impl GlesPathVertexWriter {
    fn new() -> Self {
        Self {
            points: Vec::new(),
            indices: Vec::new(),
            previous_contour_odd_points: false,
            contour_start: 0,
        }
    }

    fn write(&mut self, point: Point) {
        self.points.push(point);
    }

    fn end_contour(&mut self) {
        if self.points.is_empty() || self.contour_start == self.points.len() - 1 {
            self.contour_start = self.points.len();
            return;
        }

        let start = self.contour_start;
        let mut end = self.points.len() - 1;
        if self.points[end] == self.points[start] && end > start {
            end -= 1;
        }
        if end <= start {
            self.contour_start = self.points.len();
            return;
        }

        if self.contour_start != 0 && !self.indices.is_empty() {
            let back = *self.indices.last().unwrap();
            self.indices.push(back);
            self.indices.push(start as u32);
            self.indices.push(start as u32);
            if self.previous_contour_odd_points {
                self.indices.push(start as u32);
            }
        } else {
            self.indices.push(start as u32);
        }

        let mut a = start + 1;
        let mut b = end;
        while a < b {
            self.indices.push(a as u32);
            self.indices.push(b as u32);
            a += 1;
            b -= 1;
        }
        if a == b {
            self.indices.push(a as u32);
            self.previous_contour_odd_points = false;
        } else {
            self.previous_contour_odd_points = true;
        }
        self.contour_start = self.points.len();
    }
}

/// Ported from `PathTessellator::PathToFilledVertices` + `GLESPathVertexWriter` in `tessellator.cc`.
pub fn tessellate_convex_internal(path: &Path, tolerance: Scalar) -> (Vec<Point>, Vec<u16>) {
    let (points, indices_u32) = tessellate_convex_strip_u32(path, tolerance);
    let indices_u16 = indices_u32.into_iter().map(|i| i as u16).collect();
    (points, indices_u16)
}

pub fn tessellate_convex_strip_u32(path: &Path, tolerance: Scalar) -> (Vec<Point>, Vec<u32>) {
    let scale = tolerance.max(f32::EPSILON);
    let mut writer = GlesPathVertexWriter::new();

    let mut contour_origin = Point::ZERO;
    let mut current_point = Point::ZERO;
    let mut contour_has_segments = false;
    let mut contour_has_points = false;

    let record_line = |p2: Point,
                       writer: &mut GlesPathVertexWriter,
                       contour_origin: Point,
                       current_point: &mut Point,
                       contour_has_segments: &mut bool,
                       contour_has_points: &mut bool| {
        if !*contour_has_segments {
            writer.write(contour_origin);
            *contour_has_segments = true;
        }
        if p2 != *current_point {
            writer.write(p2);
            *current_point = p2;
            *contour_has_points = true;
        }
    };

    for comp in &path.components {
        match *comp {
            PathComponent::Contour(c) => {
                if contour_has_segments {
                    if current_point != contour_origin && contour_has_points {
                        writer.write(contour_origin);
                    }
                    writer.end_contour();
                }
                contour_origin = c.destination;
                current_point = c.destination;
                contour_has_segments = false;
                contour_has_points = false;
            }
            PathComponent::Linear(l) => {
                record_line(
                    l.p2,
                    &mut writer,
                    contour_origin,
                    &mut current_point,
                    &mut contour_has_segments,
                    &mut contour_has_points,
                );
            }
            PathComponent::Quad(q) => {
                if q.cp == current_point || q.p2 == q.cp {
                    record_line(
                        q.p2,
                        &mut writer,
                        contour_origin,
                        &mut current_point,
                        &mut contour_has_segments,
                        &mut contour_has_points,
                    );
                } else {
                    if !contour_has_segments {
                        writer.write(contour_origin);
                        contour_has_segments = true;
                    }
                    let count =
                        wangs_quadratic_subdivisions(scale, q.p1, q.cp, q.p2).ceil() as usize;
                    let count = count.max(1);
                    for i in 1..count {
                        let t = (i as Scalar) / (count as Scalar);
                        writer.write(solve_quad(&q, t));
                    }
                    writer.write(q.p2);
                    current_point = q.p2;
                    contour_has_points = true;
                }
            }
            PathComponent::Cubic(c) => {
                if !contour_has_segments {
                    writer.write(contour_origin);
                    contour_has_segments = true;
                }
                if c.cp1 != current_point || c.cp2 != current_point || c.p2 != current_point {
                    let count =
                        wangs_cubic_subdivisions(scale, c.p1, c.cp1, c.cp2, c.p2).ceil() as usize;
                    let count = count.max(1);
                    for i in 1..count {
                        let t = (i as Scalar) / (count as Scalar);
                        writer.write(solve_cubic(&c, t));
                    }
                    writer.write(c.p2);
                    current_point = c.p2;
                    contour_has_points = true;
                }
            }
        }
    }

    if contour_has_segments {
        if current_point != contour_origin && contour_has_points {
            writer.write(contour_origin);
        }
        writer.end_contour();
    }

    (writer.points, writer.indices)
}

/// Converts a convex path into a `TessellatedMesh` (triangle list) using Impeller's
/// convex zig-zag strip generator while discarding degenerate contour-stitch triangles.
pub fn tessellate_convex(path: &Path, tolerance_scale: Scalar) -> TessellatedMesh {
    let (points, strip_indices) = tessellate_convex_strip_u32(path, tolerance_scale);
    indexed_strip_to_mesh(&points, &strip_indices)
}

/// Converts an unindexed triangle strip of `Point`s into a `TessellatedMesh` (triangle list),
/// filtering out zero-area degenerate stitch triangles.
pub fn strip_points_to_mesh(strip: &[Point]) -> TessellatedMesh {
    if strip.len() < 3 {
        return TessellatedMesh::default();
    }
    let vertices: Vec<Vertex2D> = strip.iter().copied().map(Vertex2D::from_point).collect();
    let mut indices = Vec::with_capacity((strip.len() - 2) * 3);
    for i in 0..(strip.len() - 2) {
        let (i0, i1, i2) = if i % 2 == 0 {
            (i, i + 1, i + 2)
        } else {
            (i + 1, i, i + 2)
        };
        let p0 = strip[i0];
        let p1 = strip[i1];
        let p2 = strip[i2];
        if p0 == p1 || p1 == p2 || p0 == p2 {
            continue;
        }
        let area2 = (p1 - p0).cross(p2 - p0).abs();
        if area2 <= 1e-7 {
            continue;
        }
        indices.push(i0 as u32);
        indices.push(i1 as u32);
        indices.push(i2 as u32);
    }
    TessellatedMesh { vertices, indices }
}

/// Converts an indexed triangle strip into a `TessellatedMesh` (triangle list),
/// filtering out zero-area degenerate stitch triangles.
pub fn indexed_strip_to_mesh(points: &[Point], strip_indices: &[u32]) -> TessellatedMesh {
    if points.is_empty() || strip_indices.len() < 3 {
        return TessellatedMesh::default();
    }
    let vertices: Vec<Vertex2D> = points.iter().copied().map(Vertex2D::from_point).collect();
    let mut indices = Vec::with_capacity((strip_indices.len() - 2) * 3);
    for i in 0..(strip_indices.len() - 2) {
        let (a, b, c) = if i % 2 == 0 {
            (strip_indices[i], strip_indices[i + 1], strip_indices[i + 2])
        } else {
            (strip_indices[i + 1], strip_indices[i], strip_indices[i + 2])
        };
        if a == b || b == c || a == c {
            continue;
        }
        let p0 = points[a as usize];
        let p1 = points[b as usize];
        let p2 = points[c as usize];
        if (p1 - p0).cross(p2 - p0).abs() <= 1e-7 {
            continue;
        }
        indices.push(a);
        indices.push(b);
        indices.push(c);
    }
    TessellatedMesh { vertices, indices }
}
