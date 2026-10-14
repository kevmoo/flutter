#![forbid(unsafe_code)]

mod convex;
mod primitives;
mod pslg;
mod stroke;
mod trigs;

use bytemuck::{Pod, Zeroable};
use impeller_geometry::{Point, Rect, Scalar, Size};
use impeller_path::{Convexity, Path};

pub use primitives::{
    CircleGeometry, EllipseGeometry, FillPathGeometry, LineGeometry, RectGeometry,
    RoundRectGeometry, StrokePathGeometry,
};
pub use stroke::{compute_stroke_alpha_coverage, K_MIN_STROKE_SIZE};
pub use trigs::{compute_quadrant_divisions, Trig, Trigs, K_CIRCLE_TOLERANCE};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vertex2D {
    pub position: [Scalar; 2],
}

impl Vertex2D {
    pub const fn new(x: Scalar, y: Scalar) -> Self {
        Self { position: [x, y] }
    }

    pub const fn from_point(p: Point) -> Self {
        Self {
            position: [p.x, p.y],
        }
    }

    pub const fn to_point(&self) -> Point {
        Point::new(self.position[0], self.position[1])
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TessellatedMesh {
    pub vertices: Vec<Vertex2D>,
    pub indices: Vec<u32>,
}

/// Result status matching `impeller::Tessellator::Result` / `TessellatorLibtess::Result`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TessellationResult {
    Success,
    InputError,
    TessellationError,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct StrokeParameters {
    pub width: Scalar,
    pub cap: impeller_path::Cap,
    pub join: impeller_path::Join,
    pub miter_limit: Scalar,
}

impl Default for StrokeParameters {
    fn default() -> Self {
        Self {
            width: 1.0,
            cap: impeller_path::Cap::Butt,
            join: impeller_path::Join::Miter,
            miter_limit: 4.0,
        }
    }
}

pub struct Tessellator;

impl Tessellator {
    pub const K_CIRCLE_TOLERANCE: Scalar = trigs::K_CIRCLE_TOLERANCE;

    /// Tessellates a path using a Planar Straight-Line Graph (PSLG) decomposition supporting
    /// convex, concave, multi-contour holes, and self-intersecting polygons under both
    /// `FillType::NonZero` and `FillType::Odd`.
    pub fn tessellate(
        path: &Path,
        tolerance_scale: Scalar,
    ) -> Result<TessellatedMesh, TessellationResult> {
        if tolerance_scale <= 0.0 || !tolerance_scale.is_finite() {
            return Err(TessellationResult::InputError);
        }
        pslg::tessellate_pslg(path, tolerance_scale)
    }

    /// Tessellates a path and delivers the mesh to `callback`, matching `TessellatorLibtess::Tessellate`.
    pub fn tessellate_with_callback<F>(
        path: &Path,
        tolerance_scale: Scalar,
        mut callback: F,
    ) -> TessellationResult
    where
        F: FnMut(&[Vertex2D], &[u32]) -> bool,
    {
        match Self::tessellate(path, tolerance_scale) {
            Ok(mesh) => {
                if callback(&mesh.vertices, &mesh.indices) {
                    TessellationResult::Success
                } else {
                    TessellationResult::InputError
                }
            }
            Err(err) => err,
        }
    }

    /// Tessellates a filled path into a `TessellatedMesh`.
    /// Uses the convex strip fast-path when `path.convexity == Convexity::Convex`, and the
    /// general PSLG winding-rule tessellator otherwise.
    pub fn tessellate_fill(path: &Path, tolerance_scale: Scalar) -> TessellatedMesh {
        let scale = tolerance_scale.max(1.0);
        if path.convexity == Convexity::Convex {
            return Self::tessellate_convex(path, scale);
        }
        pslg::tessellate_pslg(path, scale).unwrap_or_default()
    }

    /// Ported from `Tessellator::TessellateConvex` in `tessellator.cc`.
    pub fn tessellate_convex(path: &Path, tolerance_scale: Scalar) -> TessellatedMesh {
        convex::tessellate_convex(path, tolerance_scale.max(1.0))
    }

    /// Ported from `Tessellator::TessellateConvexInternal` in `tessellator.cc`.
    pub fn tessellate_convex_internal(path: &Path, tolerance: Scalar) -> (Vec<Point>, Vec<u16>) {
        convex::tessellate_convex_internal(path, tolerance)
    }

    /// Ported from `StrokePathGeometry` / `StrokeSegmentsGeometry` in `stroke_path_geometry.cc`.
    pub fn tessellate_stroke(
        path: &Path,
        stroke: StrokeParameters,
        tolerance_scale: Scalar,
    ) -> TessellatedMesh {
        stroke::tessellate_stroke(path, stroke, tolerance_scale.max(1.0))
    }

    /// Ported from `StrokeSegmentsGeometry::GenerateSolidStrokeVertices` in `stroke_path_geometry.cc`.
    pub fn generate_solid_stroke_vertices(
        path: &Path,
        stroke: StrokeParameters,
        scale: Scalar,
    ) -> Vec<Point> {
        stroke::generate_solid_stroke_vertices(path, stroke, scale)
    }

    /// Ported from `Tessellator::FilledCircle` in `tessellator.cc`.
    pub fn filled_circle_strip(scale: Scalar, center: Point, radius: Scalar) -> Vec<Point> {
        primitives::generate_filled_circle_strip(scale, center, radius)
    }

    /// Ported from `Tessellator::StrokedCircle` in `tessellator.cc`.
    pub fn stroked_circle_strip(
        scale: Scalar,
        center: Point,
        radius: Scalar,
        half_width: Scalar,
    ) -> Vec<Point> {
        primitives::generate_stroked_circle_strip(scale, center, radius, half_width)
    }

    /// Ported from `Tessellator::RoundCapLine` in `tessellator.cc`.
    pub fn round_cap_line_strip(scale: Scalar, p0: Point, p1: Point, radius: Scalar) -> Vec<Point> {
        primitives::generate_round_cap_line_strip(scale, p0, p1, radius)
    }

    /// Ported from `Tessellator::FilledEllipse` in `tessellator.cc`.
    pub fn filled_ellipse_strip(scale: Scalar, bounds: Rect) -> Vec<Point> {
        primitives::generate_filled_ellipse_strip(scale, bounds)
    }

    /// Ported from `Tessellator::FilledRoundRect` in `tessellator.cc`.
    pub fn filled_round_rect_strip(scale: Scalar, bounds: Rect, radii: Size) -> Vec<Point> {
        primitives::generate_filled_round_rect_strip(scale, bounds, radii)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use impeller_geometry::{Degrees, Matrix, Radians};
    use impeller_path::{Cap, FillType, Join, PathBuilder, RoundingRadii};

    fn assert_point_near(a: Point, b: Point, eps: Scalar) {
        assert!(
            (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps,
            "expected {:?} near {:?} (eps = {})",
            a,
            b,
            eps
        );
    }

    fn mesh_total_area(mesh: &TessellatedMesh) -> Scalar {
        let mut area = 0.0;
        for tri in mesh.indices.as_chunks::<3>().0 {
            let a = mesh.vertices[tri[0] as usize].to_point();
            let b = mesh.vertices[tri[1] as usize].to_point();
            let c = mesh.vertices[tri[2] as usize].to_point();
            area += (b - a).cross(c - a).abs() * 0.5;
        }
        area
    }

    fn mesh_containing_triangle_count(mesh: &TessellatedMesh, p: Point) -> usize {
        let mut count = 0;
        for tri in mesh.indices.as_chunks::<3>().0 {
            let a = mesh.vertices[tri[0] as usize].to_point();
            let b = mesh.vertices[tri[1] as usize].to_point();
            let c = mesh.vertices[tri[2] as usize].to_point();
            let area = (b - a).cross(c - a);
            let (a, b, c) = if area < 0.0 { (a, c, b) } else { (a, b, c) };
            let c1 = (b - a).cross(p - a);
            let c2 = (c - b).cross(p - b);
            let c3 = (a - c).cross(p - c);
            if c1 > 1e-4 && c2 > 1e-4 && c3 > 1e-4 {
                count += 1;
            }
        }
        count
    }

    /// C++ source: tessellator_unittests.cc:TessellatorBuilderReturnsCorrectResultStatus
    #[test]
    fn tessellator_builder_returns_correct_result_status() {
        // Zero points -> InputError
        {
            let path = PathBuilder::new().build(FillType::Odd);
            let res = Tessellator::tessellate_with_callback(&path, 1.0, |_, _| true);
            assert_eq!(res, TessellationResult::InputError);
        }

        // One point -> Success
        {
            let mut pb = PathBuilder::new();
            pb.line_to(Point::new(0.0, 0.0));
            let path = pb.build(FillType::Odd);
            let res = Tessellator::tessellate_with_callback(&path, 1.0, |_, _| true);
            assert_eq!(res, TessellationResult::Success);
        }

        // Two points -> Success
        {
            let mut pb = PathBuilder::new();
            pb.move_to(Point::new(0.0, 0.0))
                .line_to(Point::new(0.0, 1.0));
            let path = pb.build(FillType::Odd);
            let res = Tessellator::tessellate_with_callback(&path, 1.0, |_, _| true);
            assert_eq!(res, TessellationResult::Success);
        }

        // Many points -> Success
        {
            let mut pb = PathBuilder::new();
            for i in 0..1000 {
                let coord = i as Scalar;
                pb.move_to(Point::new(coord, coord))
                    .line_to(Point::new(coord + 1.0, coord + 1.0));
            }
            let path = pb.build(FillType::Odd);
            let res = Tessellator::tessellate_with_callback(&path, 1.0, |_, _| true);
            assert_eq!(res, TessellationResult::Success);
        }

        // Closure fails -> InputError
        {
            let mut pb = PathBuilder::new();
            pb.move_to(Point::new(0.0, 0.0))
                .line_to(Point::new(0.0, 1.0));
            let path = pb.build(FillType::Odd);
            let res = Tessellator::tessellate_with_callback(&path, 1.0, |_, _| false);
            assert_eq!(res, TessellationResult::InputError);
        }
    }

    /// C++ source: tessellator_unittests.cc:TessellateConvex
    #[test]
    fn tessellate_convex() {
        {
            let mut pb = PathBuilder::new();
            pb.add_rect(Rect::from_ltrb(0.0, 0.0, 10.0, 10.0));
            let path = pb.build(FillType::NonZero);
            let (points, indices) = Tessellator::tessellate_convex_internal(&path, 1.0);
            let expected_pts = vec![
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(10.0, 10.0),
                Point::new(0.0, 10.0),
                Point::new(0.0, 0.0),
            ];
            let expected_indices: Vec<u16> = vec![0, 1, 3, 2];
            assert_eq!(points, expected_pts);
            assert_eq!(indices, expected_indices);
        }

        {
            let mut pb = PathBuilder::new();
            pb.add_rect(Rect::from_ltrb(0.0, 0.0, 10.0, 10.0))
                .add_rect(Rect::from_ltrb(20.0, 20.0, 30.0, 30.0));
            let path = pb.build(FillType::NonZero);
            let (points, indices) = Tessellator::tessellate_convex_internal(&path, 1.0);
            let expected_pts = vec![
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(10.0, 10.0),
                Point::new(0.0, 10.0),
                Point::new(0.0, 0.0),
                Point::new(20.0, 20.0),
                Point::new(30.0, 20.0),
                Point::new(30.0, 30.0),
                Point::new(20.0, 30.0),
                Point::new(20.0, 20.0),
            ];
            let expected_indices: Vec<u16> = vec![0, 1, 3, 2, 2, 5, 5, 6, 8, 7];
            assert_eq!(points, expected_pts);
            assert_eq!(indices, expected_indices);
        }
    }

    /// C++ source: tessellator_unittests.cc:TessellateConvexUnclosedPath
    #[test]
    fn tessellate_convex_unclosed_path() {
        let mut pb = PathBuilder::new();
        pb.line_to(Point::new(100.0, 0.0))
            .line_to(Point::new(100.0, 100.0))
            .line_to(Point::new(0.0, 100.0));
        let path = pb.build(FillType::NonZero);
        let (points, indices) = Tessellator::tessellate_convex_internal(&path, 1.0);
        let expected_pts = vec![
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 100.0),
            Point::new(0.0, 100.0),
            Point::new(0.0, 0.0),
        ];
        let expected_indices: Vec<u16> = vec![0, 1, 3, 2];
        assert_eq!(points, expected_pts);
        assert_eq!(indices, expected_indices);
    }

    /// C++ source: tessellator_unittests.cc:EarlyReturnEmptyConvexShape
    #[test]
    fn early_return_empty_convex_shape() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(0.0, 0.0))
            .move_to(Point::new(10.0, 10.0));
        let path = pb.build(FillType::NonZero);
        let (points, indices) = Tessellator::tessellate_convex_internal(&path, 3.0);
        assert!(points.is_empty());
        assert!(indices.is_empty());
    }

    /// C++ source: tessellator_unittests.cc:CircleVertexCounts
    #[test]
    fn circle_vertex_counts() {
        let check = |scale: Scalar, radius: Scalar| {
            let strip = Tessellator::filled_circle_strip(scale, Point::ZERO, radius);
            let quadrant_divisions = strip.len() / 4;
            let angle = std::f64::consts::FRAC_PI_2 / (quadrant_divisions as f64);
            let first = Point::new(radius, 0.0);
            let next = Point::new(
                (angle.cos() * radius as f64) as Scalar,
                (angle.sin() * radius as f64) as Scalar,
            );
            let midpoint = (first + next) * 0.5;
            assert!(
                midpoint.get_length() >= radius - Tessellator::K_CIRCLE_TOLERANCE * 1.01,
                "scale={}, radius={}, divisions={}",
                scale,
                radius,
                quadrant_divisions
            );
        };

        check(1.0, 0.0);
        check(1.0, 0.9);
        check(1.0, 1.0);
        check(1.0, 1.9);
        check(2.0, 0.95);
        check(1.0, 2.0);
        check(2.0, 1.0);
        check(1.0, 11.9);
        check(1.0, 12.0);
        check(1.0, 35.9);
        for i in (36..2000).step_by(40) {
            check(1.0, i as Scalar);
        }
    }

    /// C++ source: tessellator_unittests.cc:StrokedCircleVertexCounts
    #[test]
    fn stroked_circle_vertex_counts() {
        let check = |scale: Scalar, radius: Scalar, half_width: Scalar, label: &str| {
            assert!(half_width < radius, "{}", label);
            let stroked = Tessellator::stroked_circle_strip(scale, Point::ZERO, radius, half_width);
            let filled = Tessellator::filled_circle_strip(scale, Point::ZERO, radius + half_width);
            assert!(
                stroked.len() / 8 >= filled.len() / 4,
                "{}: stroked={}, filled={}",
                label,
                stroked.len(),
                filled.len()
            );
        };

        check(1.0, 2.0, 1.0, "small radius");
        check(1.0, 100.0, 10.0, "large radius");
        check(500.0, 2.0, 1.0, "small radius, scaled up");
        check(500.0, 100.0, 10.0, "large radius, scaled up");
        check(0.002, 1000.0, 10.0, "large radius, scaled down");
    }

    /// C++ source: tessellator_unittests.cc:FilledCircleTessellationVertices
    #[test]
    fn filled_circle_tessellation_vertices() {
        let check = |scale: Scalar, center: Point, radius: Scalar| {
            let vertices = Tessellator::filled_circle_strip(scale, center, radius);
            let vertex_count = vertices.len();
            assert_eq!(vertex_count % 4, 0);

            let quadrant_count = vertex_count / 4;
            for i in 0..quadrant_count {
                let angle =
                    std::f64::consts::FRAC_PI_2 * (i as f64) / ((quadrant_count - 1) as f64);
                let rsin = (angle.sin() * radius as f64) as Scalar;
                let rcos = if i == quadrant_count - 1 {
                    0.0
                } else {
                    (angle.cos() * radius as f64) as Scalar
                };
                assert_point_near(
                    vertices[i * 2],
                    Point::new(center.x - rcos, center.y + rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[i * 2 + 1],
                    Point::new(center.x - rcos, center.y - rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[vertex_count - i * 2 - 1],
                    Point::new(center.x + rcos, center.y - rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[vertex_count - i * 2 - 2],
                    Point::new(center.x + rcos, center.y + rsin),
                    1e-3,
                );
            }
        };

        check(1.0, Point::ZERO, 2.0);
        check(1.0, Point::new(10.0, 10.0), 2.0);
        check(500.0, Point::ZERO, 2.0);
        check(0.002, Point::ZERO, 1000.0);
    }

    /// C++ source: tessellator_unittests.cc:StrokedCircleTessellationVertices
    #[test]
    fn stroked_circle_tessellation_vertices() {
        let check = |scale: Scalar, center: Point, radius: Scalar, half_width: Scalar| {
            assert!(radius > half_width);
            let vertices = Tessellator::stroked_circle_strip(scale, center, radius, half_width);
            let vertex_count = vertices.len();
            assert_eq!(vertex_count % 8, 0);
            let quadrant_count = vertex_count / 8;

            for i in 0..quadrant_count {
                let angle =
                    std::f64::consts::FRAC_PI_2 * (i as f64) / ((quadrant_count - 1) as f64);
                let r_outer = (radius + half_width) as f64;
                let rsin = (angle.sin() * r_outer) as Scalar;
                let rcos = if i == quadrant_count - 1 {
                    0.0
                } else {
                    (angle.cos() * r_outer) as Scalar
                };
                assert_point_near(
                    vertices[i * 2],
                    Point::new(center.x - rcos, center.y - rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[quadrant_count * 2 + i * 2],
                    Point::new(center.x + rsin, center.y - rcos),
                    1e-3,
                );
                assert_point_near(
                    vertices[quadrant_count * 4 + i * 2],
                    Point::new(center.x + rcos, center.y + rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[quadrant_count * 6 + i * 2],
                    Point::new(center.x - rsin, center.y + rcos),
                    1e-3,
                );
            }

            for i in 0..quadrant_count {
                let angle =
                    std::f64::consts::FRAC_PI_2 * (i as f64) / ((quadrant_count - 1) as f64);
                let r_inner = (radius - half_width) as f64;
                let rsin = (angle.sin() * r_inner) as Scalar;
                let rcos = if i == quadrant_count - 1 {
                    0.0
                } else {
                    (angle.cos() * r_inner) as Scalar
                };
                assert_point_near(
                    vertices[i * 2 + 1],
                    Point::new(center.x - rcos, center.y - rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[quadrant_count * 2 + i * 2 + 1],
                    Point::new(center.x + rsin, center.y - rcos),
                    1e-3,
                );
                assert_point_near(
                    vertices[quadrant_count * 4 + i * 2 + 1],
                    Point::new(center.x + rcos, center.y + rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[quadrant_count * 6 + i * 2 + 1],
                    Point::new(center.x - rsin, center.y + rcos),
                    1e-3,
                );
            }
        };

        check(1.0, Point::ZERO, 2.0, 1.0);
        check(1.0, Point::ZERO, 2.0, 0.5);
        check(1.0, Point::new(10.0, 10.0), 2.0, 1.0);
        check(500.0, Point::ZERO, 2.0, 1.0);
        check(0.002, Point::ZERO, 1000.0, 10.0);
    }

    /// C++ source: tessellator_unittests.cc:StrokedCircleWideStrokeTessellationVertices
    #[test]
    fn stroked_circle_wide_stroke_tessellation_vertices() {
        let check = |scale: Scalar, center: Point, radius: Scalar, half_width: Scalar| {
            assert!(half_width >= radius);
            let stroked = Tessellator::stroked_circle_strip(scale, center, radius, half_width);
            let filled = Tessellator::filled_circle_strip(scale, center, radius + half_width);
            assert_eq!(stroked.len(), filled.len());
            for (a, b) in stroked.iter().zip(filled.iter()) {
                assert_point_near(*a, *b, 1e-4);
            }
        };

        check(1.0, Point::ZERO, 2.0, 2.0);
        check(1.0, Point::ZERO, 2.0, 4.0);
        check(1.0, Point::new(10.0, 10.0), 2.0, 2.0);
        check(500.0, Point::ZERO, 2.0, 2.0);
        check(0.002, Point::ZERO, 1000.0, 1000.0);
    }

    /// C++ source: tessellator_unittests.cc:RoundCapLineTessellationVertices
    #[test]
    fn round_cap_line_tessellation_vertices() {
        let check = |scale: Scalar, p0: Point, p1: Point, radius: Scalar| {
            let vertices = Tessellator::round_cap_line_strip(scale, p0, p1, radius);
            let vertex_count = vertices.len();
            assert_eq!(vertex_count % 4, 0);

            let mut along = p1 - p0;
            let length = along.get_length();
            if length > 0.0 {
                along *= radius / length;
            } else {
                along = Point::new(radius, 0.0);
            }
            let across = Point::new(-along.y, along.x);

            let quadrant_count = vertex_count / 4;
            for i in 0..quadrant_count {
                let angle =
                    std::f64::consts::FRAC_PI_2 * (i as f64) / ((quadrant_count - 1) as f64);
                let cos_a = if i == quadrant_count - 1 {
                    0.0
                } else {
                    angle.cos() as Scalar
                };
                let sin_a = angle.sin() as Scalar;
                let rel_along = along * cos_a;
                let rel_across = across * sin_a;

                assert_point_near(vertices[i * 2], p0 - rel_along + rel_across, 1e-3);
                assert_point_near(vertices[i * 2 + 1], p0 - rel_along - rel_across, 1e-3);
                assert_point_near(
                    vertices[vertex_count - i * 2 - 1],
                    p1 + rel_along - rel_across,
                    1e-3,
                );
                assert_point_near(
                    vertices[vertex_count - i * 2 - 2],
                    p1 + rel_along + rel_across,
                    1e-3,
                );
            }
        };

        check(1.0, Point::ZERO, Point::ZERO, 10.0);
        check(1.0, Point::ZERO, Point::new(10.0, 0.0), 2.0);
        check(1.0, Point::new(10.0, 0.0), Point::ZERO, 2.0);
        check(1.0, Point::ZERO, Point::new(10.0, 10.0), 2.0);
        check(500.0, Point::ZERO, Point::new(10.0, 0.0), 2.0);
        check(0.002, Point::ZERO, Point::new(10.0, 10.0), 2.0);
    }

    /// C++ source: tessellator_unittests.cc:FilledEllipseTessellationVertices
    #[test]
    fn filled_ellipse_tessellation_vertices() {
        let check = |scale: Scalar, bounds: Rect| {
            let center = Point::new(
                bounds.get_left() + bounds.get_width() * 0.5,
                bounds.get_top() + bounds.get_height() * 0.5,
            );
            let half_w = bounds.get_width() * 0.5;
            let half_h = bounds.get_height() * 0.5;

            let vertices = Tessellator::filled_ellipse_strip(scale, bounds);
            let vertex_count = vertices.len();
            assert_eq!(vertex_count % 4, 0);

            let quadrant_count = vertex_count / 4;
            for i in 0..quadrant_count {
                let angle =
                    std::f64::consts::FRAC_PI_2 * (i as f64) / ((quadrant_count - 1) as f64);
                let rcos = if i == quadrant_count - 1 {
                    0.0
                } else {
                    (angle.cos() * half_w as f64) as Scalar
                };
                let rsin = (angle.sin() * half_h as f64) as Scalar;

                assert_point_near(
                    vertices[i * 2],
                    Point::new(center.x - rcos, center.y + rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[i * 2 + 1],
                    Point::new(center.x - rcos, center.y - rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[vertex_count - i * 2 - 1],
                    Point::new(center.x + rcos, center.y - rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[vertex_count - i * 2 - 2],
                    Point::new(center.x + rcos, center.y + rsin),
                    1e-3,
                );
            }
        };

        check(1.0, Rect::from_xywh(0.0, 0.0, 2.0, 2.0));
        check(1.0, Rect::from_xywh(0.0, 0.0, 2.0, 3.0));
        check(1.0, Rect::from_xywh(0.0, 0.0, 3.0, 2.0));
        check(1.0, Rect::from_xywh(5.0, 10.0, 2.0, 3.0));
        check(500.0, Rect::from_xywh(5.0, 10.0, 3.0, 2.0));
        check(0.002, Rect::from_xywh(5000.0, 10000.0, 3000.0, 2000.0));
    }

    /// C++ source: tessellator_unittests.cc:FilledRoundRectTessellationVertices
    #[test]
    fn filled_round_rect_tessellation_vertices() {
        let check = |scale: Scalar, bounds: Rect, radii: Size| {
            let middle_left = bounds.get_left() + radii.width;
            let middle_top = bounds.get_top() + radii.height;
            let middle_right = bounds.get_right() - radii.width;
            let middle_bottom = bounds.get_bottom() - radii.height;

            let vertices = Tessellator::filled_round_rect_strip(scale, bounds, radii);
            let vertex_count = vertices.len();
            assert_eq!(vertex_count % 4, 0);

            let quadrant_count = vertex_count / 4;
            for i in 0..quadrant_count {
                let angle =
                    std::f64::consts::FRAC_PI_2 * (i as f64) / ((quadrant_count - 1) as f64);
                let rcos = if i == quadrant_count - 1 {
                    0.0
                } else {
                    (angle.cos() * radii.width as f64) as Scalar
                };
                let rsin = (angle.sin() * radii.height as f64) as Scalar;

                assert_point_near(
                    vertices[i * 2],
                    Point::new(middle_left - rcos, middle_bottom + rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[i * 2 + 1],
                    Point::new(middle_left - rcos, middle_top - rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[vertex_count - i * 2 - 1],
                    Point::new(middle_right + rcos, middle_top - rsin),
                    1e-3,
                );
                assert_point_near(
                    vertices[vertex_count - i * 2 - 2],
                    Point::new(middle_right + rcos, middle_bottom + rsin),
                    1e-3,
                );
            }
        };

        check(
            1.0,
            Rect::from_xywh(0.0, 0.0, 20.0, 20.0),
            Size::new(10.0, 10.0),
        );
        check(
            1.0,
            Rect::from_xywh(0.0, 0.0, 20.0, 20.0),
            Size::new(10.0, 5.0),
        );
        check(
            1.0,
            Rect::from_xywh(0.0, 0.0, 20.0, 20.0),
            Size::new(5.0, 10.0),
        );
        check(
            1.0,
            Rect::from_xywh(0.0, 0.0, 20.0, 30.0),
            Size::new(2.0, 2.0),
        );
        check(
            1.0,
            Rect::from_xywh(5.0, 10.0, 20.0, 30.0),
            Size::new(2.0, 3.0),
        );
        check(
            500.0,
            Rect::from_xywh(5.0, 10.0, 30.0, 20.0),
            Size::new(2.0, 3.0),
        );
        check(
            0.002,
            Rect::from_xywh(5000.0, 10000.0, 3000.0, 2000.0),
            Size::new(50.0, 70.0),
        );
    }

    /// C++ source: tessellator_unittests.cc:TessellateConcaveLShapeAndUShape
    #[test]
    fn tessellate_concave_l_shape_and_u_shape() {
        // Concave L-shape: 100x100 minus top-right 60x60 ([40..100] x [0..60]).
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(40.0, 0.0))
            .line_to(Point::new(40.0, 60.0))
            .line_to(Point::new(100.0, 60.0))
            .line_to(Point::new(100.0, 100.0))
            .line_to(Point::new(0.0, 100.0))
            .close();

        for fill_type in [FillType::NonZero, FillType::Odd] {
            let mesh = Tessellator::tessellate(&pb.build(fill_type), 1.0).unwrap();
            assert_relative_eq!(mesh_total_area(&mesh), 6400.0, epsilon = 0.5);
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(20.0, 20.0)),
                1
            );
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(80.0, 80.0)),
                1
            );
            // Concave notch must NOT be filled!
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(70.0, 30.0)),
                0
            );
        }
    }

    /// C++ source: tessellator_unittests.cc:TessellateMultiContourConcentricRingsNonZeroVsOdd
    #[test]
    fn tessellate_multi_contour_concentric_rings_non_zero_vs_odd() {
        // Outer [0, 100]x[0, 100] CCW + Inner [20, 80]x[20, 80] SAME winding (CCW).
        let mut pb_same = PathBuilder::new();
        pb_same
            .move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(100.0, 0.0))
            .line_to(Point::new(100.0, 100.0))
            .line_to(Point::new(0.0, 100.0))
            .close()
            .move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(80.0, 20.0))
            .line_to(Point::new(80.0, 80.0))
            .line_to(Point::new(20.0, 80.0))
            .close();

        // Under FillType::Odd, winding 2 at center is a hole (area = 10000 - 3600 = 6400).
        let mesh_odd = Tessellator::tessellate(&pb_same.build(FillType::Odd), 1.0).unwrap();
        assert_relative_eq!(mesh_total_area(&mesh_odd), 6400.0, epsilon = 0.5);
        assert_eq!(
            mesh_containing_triangle_count(&mesh_odd, Point::new(10.0, 50.0)),
            1
        );
        assert_eq!(
            mesh_containing_triangle_count(&mesh_odd, Point::new(45.0, 50.0)),
            0
        );

        // Under FillType::NonZero, winding 2 at center is filled with non-overlapping triangles (area = 10000).
        let mesh_nz = Tessellator::tessellate(&pb_same.build(FillType::NonZero), 1.0).unwrap();
        assert_relative_eq!(mesh_total_area(&mesh_nz), 10000.0, epsilon = 0.5);
        assert_eq!(
            mesh_containing_triangle_count(&mesh_nz, Point::new(10.0, 50.0)),
            1
        );
        assert_eq!(
            mesh_containing_triangle_count(&mesh_nz, Point::new(45.0, 50.0)),
            1
        );

        // Opposite winding (Inner CW): hole under BOTH NonZero and Odd!
        let mut pb_opp = PathBuilder::new();
        pb_opp
            .move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(100.0, 0.0))
            .line_to(Point::new(100.0, 100.0))
            .line_to(Point::new(0.0, 100.0))
            .close()
            .move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(20.0, 80.0))
            .line_to(Point::new(80.0, 80.0))
            .line_to(Point::new(80.0, 20.0))
            .close();
        for ft in [FillType::NonZero, FillType::Odd] {
            let mesh = Tessellator::tessellate(&pb_opp.build(ft), 1.0).unwrap();
            assert_relative_eq!(mesh_total_area(&mesh), 6400.0, epsilon = 0.5);
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(50.0, 50.0)),
                0
            );
        }
    }

    /// C++ source: tessellator_unittests.cc:TessellateSelfIntersectingBowtie
    #[test]
    fn tessellate_self_intersecting_bowtie() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(100.0, 100.0))
            .line_to(Point::new(100.0, 0.0))
            .line_to(Point::new(0.0, 100.0))
            .close();

        for ft in [FillType::NonZero, FillType::Odd] {
            let mesh = Tessellator::tessellate(&pb.build(ft), 1.0).unwrap();
            assert_relative_eq!(mesh_total_area(&mesh), 5000.0, epsilon = 0.5);
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(20.0, 50.0)),
                1
            );
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(80.0, 50.0)),
                1
            );
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(50.0, 20.0)),
                0
            );
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(50.0, 80.0)),
                0
            );
        }
    }

    /// C++ source: tessellator_unittests.cc:TessellateSelfIntersectingPentagramNonZeroVsOdd
    #[test]
    fn tessellate_self_intersecting_pentagram_non_zero_vs_odd() {
        let center = Point::new(100.0, 100.0);
        let radius = 80.0;
        let star_order = [0usize, 2, 4, 1, 3];
        let mut pb = PathBuilder::new();
        for (idx, &k) in star_order.iter().enumerate() {
            let angle =
                -std::f32::consts::FRAC_PI_2 + (k as f32) * 2.0 * std::f32::consts::PI / 5.0;
            let pt = Point::new(
                center.x + radius * angle.cos(),
                center.y + radius * angle.sin(),
            );
            if idx == 0 {
                pb.move_to(pt);
            } else {
                pb.line_to(pt);
            }
        }
        pb.close();

        let mesh_nz = Tessellator::tessellate(&pb.build(FillType::NonZero), 1.0).unwrap();
        let mesh_odd = Tessellator::tessellate(&pb.build(FillType::Odd), 1.0).unwrap();

        // Center of pentagram has winding number 2: filled in NonZero, hole in Odd!
        assert_eq!(mesh_containing_triangle_count(&mesh_nz, center), 1);
        assert_eq!(mesh_containing_triangle_count(&mesh_odd, center), 0);
        assert!(mesh_total_area(&mesh_nz) > mesh_total_area(&mesh_odd) + 500.0);
    }

    /// C++ source: tessellator_unittests.cc:TessellateNestedIslandInHole
    #[test]
    fn tessellate_nested_island_in_hole() {
        // Outer [0..100] CCW (10000) + Hole [20..80] CW (-3600) + Island [40..60] CCW (+400) = 6800.
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(100.0, 0.0))
            .line_to(Point::new(100.0, 100.0))
            .line_to(Point::new(0.0, 100.0))
            .close()
            .move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(20.0, 80.0))
            .line_to(Point::new(80.0, 80.0))
            .line_to(Point::new(80.0, 20.0))
            .close()
            .move_to(Point::new(40.0, 40.0))
            .line_to(Point::new(60.0, 40.0))
            .line_to(Point::new(60.0, 60.0))
            .line_to(Point::new(40.0, 60.0))
            .close();

        for ft in [FillType::NonZero, FillType::Odd] {
            let mesh = Tessellator::tessellate(&pb.build(ft), 1.0).unwrap();
            assert_relative_eq!(mesh_total_area(&mesh), 6800.0, epsilon = 0.5);
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(10.0, 50.0)),
                1
            );
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(30.0, 50.0)),
                0
            );
            assert_eq!(
                mesh_containing_triangle_count(&mesh, Point::new(45.0, 50.0)),
                1
            );
        }
    }

    /// C++ source: geometry_unittests.cc:RectGeometryCoversArea
    #[test]
    fn rect_geometry_covers_area() {
        let geom = RectGeometry::new(Rect::from_ltrb(0.0, 0.0, 100.0, 100.0));
        let id = Matrix::IDENTITY;
        assert!(geom.covers_area(&id, Rect::from_ltrb(0.0, 0.0, 100.0, 100.0)));
        assert!(!geom.covers_area(&id, Rect::from_ltrb(-1.0, 0.0, 100.0, 100.0)));
        assert!(geom.covers_area(&id, Rect::from_ltrb(1.0, 1.0, 100.0, 100.0)));
        assert!(geom.covers_area(&id, Rect::ZERO));
    }

    /// C++ source: geometry_unittests.cc:FillPathGeometryCoversArea
    #[test]
    fn fill_path_geometry_covers_area() {
        let mut pb = PathBuilder::new();
        pb.add_rect(Rect::from_ltrb(0.0, 0.0, 100.0, 100.0));
        let geom = FillPathGeometry::new(pb.build(FillType::NonZero));
        let inner = Some(Rect::from_ltrb(0.0, 0.0, 100.0, 100.0));
        let id = Matrix::IDENTITY;
        assert!(geom.covers_area(&id, inner, Rect::from_ltrb(0.0, 0.0, 100.0, 100.0)));
        assert!(!geom.covers_area(&id, inner, Rect::from_ltrb(-1.0, 0.0, 100.0, 100.0)));
        assert!(geom.covers_area(&id, inner, Rect::from_ltrb(1.0, 1.0, 100.0, 100.0)));
        assert!(geom.covers_area(&id, inner, Rect::ZERO));
    }

    /// C++ source: geometry_unittests.cc:FillPathGeometryCoversAreaNoInnerRect
    #[test]
    fn fill_path_geometry_covers_area_no_inner_rect() {
        let mut pb = PathBuilder::new();
        pb.add_rect(Rect::from_ltrb(0.0, 0.0, 100.0, 100.0));
        let geom = FillPathGeometry::new(pb.build(FillType::NonZero));
        let id = Matrix::IDENTITY;
        assert!(!geom.covers_area(&id, None, Rect::from_ltrb(0.0, 0.0, 100.0, 100.0)));
        assert!(!geom.covers_area(&id, None, Rect::from_ltrb(-1.0, 0.0, 100.0, 100.0)));
        assert!(!geom.covers_area(&id, None, Rect::from_ltrb(1.0, 1.0, 100.0, 100.0)));
        assert!(!geom.covers_area(&id, None, Rect::ZERO));
    }

    /// C++ source: geometry_unittests.cc:FillRoundRectGeometryCoversArea
    #[test]
    fn fill_round_rect_geometry_covers_area() {
        let bounds = Rect::from_ltrb(100.0, 100.0, 200.0, 200.0);
        let radii = RoundingRadii::from_radii(
            Size::new(1.0, 11.0),
            Size::new(2.0, 12.0),
            Size::new(3.0, 13.0),
            Size::new(4.0, 14.0),
        );
        let geom = RoundRectGeometry::new(bounds, radii);
        let id = Matrix::IDENTITY;

        assert!(geom.covers_area(&id, Rect::from_ltrb(103.0, 100.0, 196.0, 200.0)));
        assert!(!geom.covers_area(&id, Rect::from_ltrb(102.0, 100.0, 196.0, 200.0)));
        assert!(!geom.covers_area(&id, Rect::from_ltrb(103.0, 99.0, 196.0, 200.0)));
        assert!(!geom.covers_area(&id, Rect::from_ltrb(103.0, 100.0, 197.0, 200.0)));
        assert!(!geom.covers_area(&id, Rect::from_ltrb(103.0, 100.0, 196.0, 201.0)));

        assert!(geom.covers_area(&id, Rect::from_ltrb(100.0, 112.0, 200.0, 186.0)));
        assert!(!geom.covers_area(&id, Rect::from_ltrb(99.0, 112.0, 200.0, 186.0)));
        assert!(!geom.covers_area(&id, Rect::from_ltrb(100.0, 111.0, 200.0, 186.0)));
        assert!(!geom.covers_area(&id, Rect::from_ltrb(100.0, 112.0, 201.0, 186.0)));
        assert!(!geom.covers_area(&id, Rect::from_ltrb(100.0, 112.0, 200.0, 187.0)));
    }

    /// C++ source: geometry_unittests.cc:LineGeometryCoverage
    #[test]
    fn line_geometry_coverage() {
        let id = Matrix::IDENTITY;
        {
            let geom = LineGeometry::new(
                Point::new(10.0, 10.0),
                Point::new(20.0, 10.0),
                2.0,
                Cap::Butt,
            );
            assert_eq!(
                geom.get_coverage(&id),
                Some(Rect::from_ltrb(10.0, 9.0, 20.0, 11.0))
            );
            assert!(geom.covers_area(&id, Rect::from_ltrb(10.0, 9.0, 20.0, 11.0)));
        }
        {
            let geom = LineGeometry::new(
                Point::new(10.0, 10.0),
                Point::new(20.0, 10.0),
                2.0,
                Cap::Square,
            );
            assert_eq!(
                geom.get_coverage(&id),
                Some(Rect::from_ltrb(9.0, 9.0, 21.0, 11.0))
            );
            assert!(geom.covers_area(&id, Rect::from_ltrb(9.0, 9.0, 21.0, 11.0)));
        }
        {
            let geom = LineGeometry::new(
                Point::new(10.0, 10.0),
                Point::new(10.0, 20.0),
                2.0,
                Cap::Butt,
            );
            assert_eq!(
                geom.get_coverage(&id),
                Some(Rect::from_ltrb(9.0, 10.0, 11.0, 20.0))
            );
            assert!(geom.covers_area(&id, Rect::from_ltrb(9.0, 10.0, 11.0, 20.0)));
        }
        {
            let geom = LineGeometry::new(
                Point::new(10.0, 10.0),
                Point::new(10.0, 20.0),
                2.0,
                Cap::Square,
            );
            assert_eq!(
                geom.get_coverage(&id),
                Some(Rect::from_ltrb(9.0, 9.0, 11.0, 21.0))
            );
            assert!(geom.covers_area(&id, Rect::from_ltrb(9.0, 9.0, 11.0, 21.0)));
        }
    }

    /// C++ source: geometry_unittests.cc:RoundRectGeometryCoversArea
    #[test]
    fn round_rect_geometry_covers_area() {
        let geom = RoundRectGeometry::new(
            Rect::from_ltrb(0.0, 0.0, 100.0, 100.0),
            RoundingRadii::from_single_radius(20.0),
        );
        let id = Matrix::IDENTITY;
        assert!(!geom.covers_area(&id, Rect::from_ltrb(15.0, 15.0, 85.0, 85.0)));
        assert!(geom.covers_area(&id, Rect::from_ltrb(20.0, 20.0, 80.0, 80.0)));
        assert!(geom.covers_area(&id, Rect::from_ltrb(30.0, 1.0, 70.0, 99.0)));
        assert!(geom.covers_area(&id, Rect::from_ltrb(1.0, 30.0, 99.0, 70.0)));
    }

    /// C++ source: geometry_unittests.cc:AlphaCoverageStrokePaths
    #[test]
    fn alpha_coverage_stroke_paths() {
        let matrix = Matrix::make_scale(3.0, 3.0, 1.0);
        assert_eq!(compute_stroke_alpha_coverage(&matrix, 0.5), 1.0);
        assert_relative_eq!(
            compute_stroke_alpha_coverage(&matrix, 0.1),
            0.6,
            epsilon = 0.05
        );
        assert_relative_eq!(
            compute_stroke_alpha_coverage(&matrix, 0.05),
            0.3,
            epsilon = 0.05
        );
        assert_relative_eq!(
            compute_stroke_alpha_coverage(&matrix, 0.01),
            0.1,
            epsilon = 0.1
        );
        assert_relative_eq!(
            compute_stroke_alpha_coverage(&matrix, 0.0000005),
            1e-5,
            epsilon = 0.001
        );
        assert_eq!(compute_stroke_alpha_coverage(&matrix, 0.0), 1.0);
        assert_eq!(compute_stroke_alpha_coverage(&matrix, 40.0), 1.0);
    }

    /// C++ source: geometry_unittests.cc:SimpleTwoLineStrokeVerticesButtCap
    #[test]
    fn simple_two_line_stroke_vertices_butt_cap() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(30.0, 20.0))
            .move_to(Point::new(120.0, 20.0))
            .line_to(Point::new(130.0, 20.0));
        let path = pb.build(FillType::NonZero);

        let points = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 10.0,
                cap: Cap::Butt,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );

        let expected = vec![
            Point::new(20.0, 25.0),
            Point::new(20.0, 15.0),
            Point::new(30.0, 25.0),
            Point::new(30.0, 15.0),
            Point::new(30.0, 20.0),
            Point::new(30.0, 20.0),
            Point::new(120.0, 20.0),
            Point::new(120.0, 20.0),
            Point::new(120.0, 25.0),
            Point::new(120.0, 15.0),
            Point::new(130.0, 25.0),
            Point::new(130.0, 15.0),
        ];
        assert_eq!(points, expected);
    }

    /// C++ source: geometry_unittests.cc:SimpleTwoLineStrokeVerticesRoundCap
    #[test]
    fn simple_two_line_stroke_vertices_round_cap() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(30.0, 20.0))
            .move_to(Point::new(120.0, 20.0))
            .line_to(Point::new(130.0, 20.0));
        let path = pb.build(FillType::NonZero);

        let points = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 10.0,
                cap: Cap::Round,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );

        assert_eq!(points.len(), 40);

        let offset = |step: i32, left: bool, backwards: bool| -> Point {
            let angle = std::f32::consts::FRAC_PI_2 * (step as f32 / 4.0);
            let along = Point::new(5.0, 0.0) * angle.cos();
            let across = Point::new(0.0, 5.0) * angle.sin();
            let center = if backwards { -along } else { along };
            if left {
                center + across
            } else {
                center - across
            }
        };

        assert_point_near(points[0], Point::new(15.0, 20.0), 1e-4);
        assert_point_near(
            points[1],
            Point::new(20.0, 20.0) + offset(1, true, true),
            1e-4,
        );
        assert_point_near(
            points[2],
            Point::new(20.0, 20.0) + offset(1, false, true),
            1e-4,
        );
        assert_point_near(
            points[3],
            Point::new(20.0, 20.0) + offset(2, true, true),
            1e-4,
        );
        assert_point_near(
            points[4],
            Point::new(20.0, 20.0) + offset(2, false, true),
            1e-4,
        );
        assert_point_near(
            points[5],
            Point::new(20.0, 20.0) + offset(3, true, true),
            1e-4,
        );
        assert_point_near(
            points[6],
            Point::new(20.0, 20.0) + offset(3, false, true),
            1e-4,
        );
        assert_point_near(points[7], Point::new(20.0, 25.0), 1e-4);
        assert_point_near(points[8], Point::new(20.0, 15.0), 1e-4);
        assert_point_near(points[9], Point::new(30.0, 25.0), 1e-4);
        assert_point_near(points[10], Point::new(30.0, 15.0), 1e-4);
        assert_point_near(
            points[11],
            Point::new(30.0, 20.0) + offset(3, true, false),
            1e-4,
        );
        assert_point_near(
            points[12],
            Point::new(30.0, 20.0) + offset(3, false, false),
            1e-4,
        );
        assert_point_near(points[17], Point::new(35.0, 20.0), 1e-4);
        assert_point_near(points[18], Point::new(30.0, 20.0), 1e-4);
        assert_point_near(points[21], Point::new(120.0, 20.0), 1e-4);
        assert_point_near(points[22], Point::new(115.0, 20.0), 1e-4);
        assert_point_near(points[39], Point::new(135.0, 20.0), 1e-4);
    }

    /// C++ source: geometry_unittests.cc:SimpleTwoLineStrokeVerticesSquareCap
    #[test]
    fn simple_two_line_stroke_vertices_square_cap() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(30.0, 20.0))
            .move_to(Point::new(120.0, 20.0))
            .line_to(Point::new(130.0, 20.0));
        let path = pb.build(FillType::NonZero);

        let points = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 10.0,
                cap: Cap::Square,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );

        let expected = vec![
            Point::new(15.0, 25.0),
            Point::new(15.0, 15.0),
            Point::new(20.0, 25.0),
            Point::new(20.0, 15.0),
            Point::new(30.0, 25.0),
            Point::new(30.0, 15.0),
            Point::new(35.0, 25.0),
            Point::new(35.0, 15.0),
            Point::new(30.0, 20.0),
            Point::new(30.0, 20.0),
            Point::new(120.0, 20.0),
            Point::new(120.0, 20.0),
            Point::new(115.0, 25.0),
            Point::new(115.0, 15.0),
            Point::new(120.0, 25.0),
            Point::new(120.0, 15.0),
            Point::new(130.0, 25.0),
            Point::new(130.0, 15.0),
            Point::new(135.0, 25.0),
            Point::new(135.0, 15.0),
        ];
        assert_eq!(points, expected);
    }

    /// C++ source: geometry_unittests.cc:TwoLineSegmentsRightTurnStrokeVerticesBevelJoin
    #[test]
    fn two_line_segments_right_turn_stroke_vertices_bevel_join() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(30.0, 20.0))
            .line_to(Point::new(30.0, 30.0));
        let path = pb.build(FillType::NonZero);

        let points = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 10.0,
                cap: Cap::Butt,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );

        let expected = vec![
            Point::new(20.0, 25.0),
            Point::new(20.0, 15.0),
            Point::new(30.0, 25.0),
            Point::new(30.0, 15.0),
            Point::new(25.0, 20.0),
            Point::new(35.0, 20.0),
            Point::new(25.0, 30.0),
            Point::new(35.0, 30.0),
        ];
        assert_eq!(points, expected);
    }

    /// C++ source: geometry_unittests.cc:TwoLineSegmentsLeftTurnStrokeVerticesBevelJoin
    #[test]
    fn two_line_segments_left_turn_stroke_vertices_bevel_join() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(30.0, 20.0))
            .line_to(Point::new(30.0, 10.0));
        let path = pb.build(FillType::NonZero);

        let points = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 10.0,
                cap: Cap::Butt,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );

        let expected = vec![
            Point::new(20.0, 25.0),
            Point::new(20.0, 15.0),
            Point::new(30.0, 25.0),
            Point::new(30.0, 15.0),
            Point::new(35.0, 20.0),
            Point::new(25.0, 20.0),
            Point::new(35.0, 10.0),
            Point::new(25.0, 10.0),
        ];
        assert_eq!(points, expected);
    }

    /// C++ source: geometry_unittests.cc:TwoLineSegmentsInDifferentContoursAreNotJoined
    #[test]
    fn two_line_segments_in_different_contours_are_not_joined() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(30.0, 20.0))
            .move_to(Point::new(30.0, 20.0))
            .line_to(Point::new(30.0, 30.0));
        let path = pb.build(FillType::NonZero);

        let points = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 10.0,
                cap: Cap::Butt,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );

        let expected = vec![
            Point::new(20.0, 25.0),
            Point::new(20.0, 15.0),
            Point::new(30.0, 25.0),
            Point::new(30.0, 15.0),
            Point::new(30.0, 20.0),
            Point::new(30.0, 20.0),
            Point::new(30.0, 20.0),
            Point::new(30.0, 20.0),
            Point::new(25.0, 20.0),
            Point::new(35.0, 20.0),
            Point::new(25.0, 30.0),
            Point::new(35.0, 30.0),
        ];
        assert_eq!(points, expected);
    }

    /// C++ source: geometry_unittests.cc:TwoLineSegmentsRightTurnStrokeVerticesMiterJoin
    #[test]
    fn two_line_segments_right_turn_stroke_vertices_miter_join() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(30.0, 20.0))
            .line_to(Point::new(30.0, 30.0));
        let path = pb.build(FillType::NonZero);

        let points = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 10.0,
                cap: Cap::Butt,
                join: Join::Miter,
                miter_limit: 4.0,
            },
            1.0,
        );

        let expected = vec![
            Point::new(20.0, 25.0),
            Point::new(20.0, 15.0),
            Point::new(30.0, 25.0),
            Point::new(30.0, 15.0),
            Point::new(35.0, 15.0),
            Point::new(25.0, 20.0),
            Point::new(35.0, 20.0),
            Point::new(25.0, 30.0),
            Point::new(35.0, 30.0),
        ];
        assert_eq!(points, expected);
    }

    /// C++ source: geometry_unittests.cc:TwoLineSegmentsLeftTurnStrokeVerticesMiterJoin
    #[test]
    fn two_line_segments_left_turn_stroke_vertices_miter_join() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(20.0, 20.0))
            .line_to(Point::new(30.0, 20.0))
            .line_to(Point::new(30.0, 10.0));
        let path = pb.build(FillType::NonZero);

        let points = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 10.0,
                cap: Cap::Butt,
                join: Join::Miter,
                miter_limit: 4.0,
            },
            1.0,
        );

        let expected = vec![
            Point::new(20.0, 25.0),
            Point::new(20.0, 15.0),
            Point::new(30.0, 25.0),
            Point::new(30.0, 15.0),
            Point::new(35.0, 25.0),
            Point::new(35.0, 20.0),
            Point::new(25.0, 20.0),
            Point::new(35.0, 10.0),
            Point::new(25.0, 10.0),
        ];
        assert_eq!(points, expected);
    }

    /// C++ source: geometry_unittests.cc:TinyQuadGeneratesCaps
    #[test]
    fn tiny_quad_generates_caps() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(20.0, 20.0))
            .quad_to(Point::new(20.125, 20.0), Point::new(20.250, 20.0));
        let path = pb.build(FillType::NonZero);

        let points = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 4.0,
                cap: Cap::Square,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );

        let expected = vec![
            Point::new(18.0, 22.0),
            Point::new(18.0, 18.0),
            Point::new(20.0, 22.0),
            Point::new(20.0, 18.0),
            Point::new(20.25, 22.0),
            Point::new(20.25, 18.0),
            Point::new(22.25, 22.0),
            Point::new(22.25, 18.0),
        ];
        assert_eq!(points, expected);
    }

    /// C++ source: geometry_unittests.cc:TinyCubicGeneratesCaps
    #[test]
    fn tiny_cubic_generates_caps() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(20.0, 20.0)).cubic_to(
            Point::new(20.0625, 20.0),
            Point::new(20.125, 20.0),
            Point::new(20.250, 20.0),
        );
        let path = pb.build(FillType::NonZero);

        let points = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 4.0,
                cap: Cap::Square,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );

        let expected = vec![
            Point::new(18.0, 22.0),
            Point::new(18.0, 18.0),
            Point::new(20.0, 22.0),
            Point::new(20.0, 18.0),
            Point::new(20.25, 22.0),
            Point::new(20.25, 18.0),
            Point::new(22.25, 22.0),
            Point::new(22.25, 18.0),
        ];
        assert_eq!(points, expected);
    }

    /// C++ source: geometry_unittests.cc:TwoLineSegmentsMiterLimit
    #[test]
    fn two_line_segments_miter_limit() {
        for degrees in (10..180).step_by(10) {
            for width in 2..=10 {
                let r: Radians = Degrees(degrees as Scalar).into();
                let pixel_delta = Point::new(r.0.cos(), r.0.sin());
                if pixel_delta.get_distance(Point::new(1.0, 0.0)) * (width as Scalar) < 1.0 {
                    continue;
                }
                let r_between: Radians = Degrees((180 - degrees) as Scalar).into();
                let limit = 1.0 / (r_between.0 * 0.5).sin();

                let mut pb = PathBuilder::new();
                pb.move_to(Point::new(20.0, 20.0))
                    .line_to(Point::new(30.0, 20.0))
                    .line_to(Point::new(30.0, 20.0) + pixel_delta * 10.0);
                let path = pb.build(FillType::NonZero);

                // 99% of required limit -> bevel fallback (8 points)
                let pts1 = Tessellator::generate_solid_stroke_vertices(
                    &path,
                    StrokeParameters {
                        width: width as Scalar,
                        cap: Cap::Butt,
                        join: Join::Miter,
                        miter_limit: limit * 0.99,
                    },
                    1.0,
                );
                assert_eq!(pts1.len(), 8, "deg={}, width={}", degrees, width);

                // 101% of required limit -> miter join (9 points)
                let pts2 = Tessellator::generate_solid_stroke_vertices(
                    &path,
                    StrokeParameters {
                        width: width as Scalar,
                        cap: Cap::Butt,
                        join: Join::Miter,
                        miter_limit: limit * 1.01,
                    },
                    1.0,
                );
                assert_eq!(pts2.len(), 9, "deg={}, width={}", degrees, width);
                assert!(
                    pts2[4].get_distance(Point::new(30.0, 20.0))
                        <= (width as Scalar) * limit * 1.05
                );
            }
        }
    }

    /// C++ source: geometry_unittests.cc:TwoLineSegments180DegreeJoins
    #[test]
    fn two_line_segments_180_degree_joins() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(10.0, 10.0))
            .line_to(Point::new(100.0, 10.0))
            .line_to(Point::new(10.0, 10.0));
        let path = pb.build(FillType::NonZero);

        let points_bevel = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 20.0,
                cap: Cap::Butt,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );
        assert_eq!(points_bevel.len(), 8);

        let points_miter = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 20.0,
                cap: Cap::Butt,
                join: Join::Miter,
                miter_limit: 400.0,
            },
            1.0,
        );
        assert_eq!(points_miter.len(), 8);

        let points_round = Tessellator::generate_solid_stroke_vertices(
            &path,
            StrokeParameters {
                width: 20.0,
                cap: Cap::Butt,
                join: Join::Round,
                miter_limit: 4.0,
            },
            1.0,
        );
        assert_eq!(points_round.len(), 19);
    }

    /// C++ source: geometry_unittests.cc:TightQuadratic180DegreeJoins
    #[test]
    fn tight_quadratic_180_degree_joins() {
        let mut pb_ref = PathBuilder::new();
        pb_ref
            .move_to(Point::new(10.0, 10.0))
            .quad_to(Point::new(100.0, 10.0), Point::new(100.0, 100.0));
        let path_ref = pb_ref.build(FillType::NonZero);

        let pts_ref = Tessellator::generate_solid_stroke_vertices(
            &path_ref,
            StrokeParameters {
                width: 20.0,
                cap: Cap::Butt,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );
        assert_eq!(pts_ref.len(), 74);

        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(10.0, 10.0))
            .quad_to(Point::new(100.0, 10.0), Point::new(10.0, 10.0));
        let path = pb.build(FillType::NonZero);

        for join in [Join::Bevel, Join::Miter, Join::Round] {
            let pts = Tessellator::generate_solid_stroke_vertices(
                &path,
                StrokeParameters {
                    width: 20.0,
                    cap: Cap::Butt,
                    join,
                    miter_limit: 400.0,
                },
                1.0,
            );
            assert!(pts.len() > pts_ref.len());
        }
    }

    /// C++ source: geometry_unittests.cc:TightCubic180DegreeJoins
    #[test]
    fn tight_cubic_180_degree_joins() {
        let mut pb_ref = PathBuilder::new();
        pb_ref.move_to(Point::new(10.0, 10.0)).cubic_to(
            Point::new(60.0, 10.0),
            Point::new(100.0, 40.0),
            Point::new(100.0, 90.0),
        );
        let path_ref = pb_ref.build(FillType::NonZero);

        let pts_ref = Tessellator::generate_solid_stroke_vertices(
            &path_ref,
            StrokeParameters {
                width: 20.0,
                cap: Cap::Butt,
                join: Join::Bevel,
                miter_limit: 4.0,
            },
            1.0,
        );
        assert_eq!(pts_ref.len(), 76);

        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(10.0, 10.0)).cubic_to(
            Point::new(60.0, 10.0),
            Point::new(100.0, 40.0),
            Point::new(60.0, 10.0),
        );
        let path = pb.build(FillType::NonZero);

        for join in [Join::Bevel, Join::Miter, Join::Round] {
            let pts = Tessellator::generate_solid_stroke_vertices(
                &path,
                StrokeParameters {
                    width: 20.0,
                    cap: Cap::Butt,
                    join,
                    miter_limit: 400.0,
                },
                1.0,
            );
            assert!(pts.len() > pts_ref.len());
        }
    }

    /// C++ source: geometry_unittests.cc:RotatedFilledCircleGeometryCoverage
    #[test]
    fn rotated_filled_circle_geometry_coverage() {
        let center = Point::new(50.0, 50.0);
        let geom = CircleGeometry::new(center, 50.0);
        let circle_bounds = Rect::from_ltrb(0.0, 0.0, 100.0, 100.0);
        assert_eq!(geom.get_coverage(&Matrix::IDENTITY), Some(circle_bounds));

        let transform45 = Matrix::make_translation(center.x, center.y, 0.0)
            * Matrix::make_rotation_z(Degrees(45.0).into())
            * Matrix::make_translation(-center.x, -center.y, 0.0);
        let bounds = geom.get_coverage(&transform45).unwrap();
        assert!(primitives::rect_contains_rect(bounds, circle_bounds));
    }

    /// C++ source: geometry_unittests.cc:RotatedStrokedCircleGeometryCoverage
    #[test]
    fn rotated_stroked_circle_geometry_coverage() {
        let center = Point::new(50.0, 50.0);
        let geom = CircleGeometry::new(center, 50.0);
        let circle_bounds = Rect::from_ltrb(-5.0, -5.0, 105.0, 105.0);
        assert_eq!(
            geom.get_stroked_coverage(10.0, &Matrix::IDENTITY),
            Some(circle_bounds)
        );

        let transform45 = Matrix::make_translation(center.x, center.y, 0.0)
            * Matrix::make_rotation_z(Degrees(45.0).into())
            * Matrix::make_translation(-center.x, -center.y, 0.0);
        let bounds = geom.get_stroked_coverage(10.0, &transform45).unwrap();
        assert!(primitives::rect_contains_rect(bounds, circle_bounds));
    }

    /// C++ source: geometry_unittests.cc:RoundRectGeometryPerCornerDivisions
    #[test]
    fn test_asymmetric_round_rect_trig_corner_smoothness() {
        let rect = Rect::from_xywh(10.0, 20.0, 160.0, 100.0);
        let radii = RoundingRadii {
            top_left: Size::new(28.0, 28.0),
            top_right: Size::new(6.0, 6.0),
            bottom_left: Size::new(6.0, 6.0),
            bottom_right: Size::new(28.0, 28.0),
        };
        let geom = RoundRectGeometry::new(rect, radii);
        let mesh = geom.get_mesh(1.0);

        // At r_max = 28 >= 4.0, each of the 4 corners has >= 12 divisions (>= 48 perimeter vertices).
        assert!(
            mesh.vertices.len() >= 48,
            "expected >= 48 trig vertices on asymmetric rrect, got {}",
            mesh.vertices.len()
        );

        // Verify vertices on the top-left corner arc (center = (38, 48), r = 28) lie on the true circle.
        let tl_center = Point::new(10.0 + 28.0, 20.0 + 28.0);
        let mut tl_arc_count = 0usize;
        for v in &mesh.vertices {
            let p = v.to_point();
            if p.x <= tl_center.x + 1e-4 && p.y <= tl_center.y + 1e-4 {
                let dist = (p - tl_center).get_length();
                assert_relative_eq!(dist, 28.0, epsilon = 1e-3);
                tl_arc_count += 1;
            }
        }
        assert!(
            tl_arc_count >= 13,
            "expected >= 13 vertices on top-left corner arc, got {}",
            tl_arc_count
        );

        // Exact inscribed polygon area and continuous analytical area.
        let divs = (tl_arc_count - 1) as f32;
        let inscribed_quadrant_factor = 0.5 * divs * (std::f32::consts::FRAC_PI_2 / divs).sin();
        let sum_r2 = 28.0 * 28.0 + 6.0 * 6.0 + 6.0 * 6.0 + 28.0 * 28.0;
        let expected_inscribed = 160.0 * 100.0 - (1.0 - inscribed_quadrant_factor) * sum_r2;
        assert_relative_eq!(mesh_total_area(&mesh), expected_inscribed, epsilon = 0.1);
        let expected_continuous = 160.0 * 100.0 - (1.0 - std::f32::consts::FRAC_PI_4) * sum_r2;
        assert_relative_eq!(mesh_total_area(&mesh), expected_continuous, epsilon = 2.5);
    }

    /// C++ source: geometry_unittests.cc:StrokeRoundRectGeometryCoversRing
    #[test]
    fn test_stroked_round_rect_geometry_ring() {
        let rect = Rect::from_xywh(20.0, 20.0, 120.0, 80.0);
        let radii = RoundingRadii {
            top_left: Size::new(20.0, 20.0),
            top_right: Size::new(8.0, 8.0),
            bottom_left: Size::new(8.0, 8.0),
            bottom_right: Size::new(20.0, 20.0),
        };
        let geom = RoundRectGeometry::new(rect, radii);
        let stroke = StrokeParameters {
            width: 6.0,
            cap: Cap::Butt,
            join: Join::Round,
            miter_limit: 4.0,
        };
        let mesh = geom.get_stroked_mesh(stroke, 1.0);
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());

        // Hollow center is NOT covered.
        assert_eq!(
            mesh_containing_triangle_count(&mesh, Point::new(80.0, 60.0)),
            0
        );
        // Points on the top, bottom, left, right stroke bands ARE covered once.
        assert_eq!(
            mesh_containing_triangle_count(&mesh, Point::new(80.0, 20.5)),
            1
        );
        assert_eq!(
            mesh_containing_triangle_count(&mesh, Point::new(80.0, 99.5)),
            1
        );
        assert_eq!(
            mesh_containing_triangle_count(&mesh, Point::new(20.5, 60.0)),
            1
        );
        assert_eq!(
            mesh_containing_triangle_count(&mesh, Point::new(139.5, 60.0)),
            1
        );

        // Ring area matches analytical outer_rrect_area - inner_rrect_area.
        let rrect_area = |w: f32, h: f32, r_tl: f32, r_tr: f32, r_bl: f32, r_br: f32| -> f32 {
            w * h
                - (1.0 - std::f32::consts::FRAC_PI_4)
                    * (r_tl * r_tl + r_tr * r_tr + r_bl * r_bl + r_br * r_br)
        };
        let outer_area = rrect_area(126.0, 86.0, 23.0, 11.0, 11.0, 23.0);
        let inner_area = rrect_area(114.0, 74.0, 17.0, 5.0, 5.0, 17.0);
        assert_relative_eq!(
            mesh_total_area(&mesh),
            outer_area - inner_area,
            epsilon = 1.5
        );

        // Subpixel stroke width (0.25 at scale 1.0) clamps to 1.0 physical device pixel wide.
        let subpixel_mesh = geom.get_stroked_mesh(
            StrokeParameters {
                width: 0.25,
                ..stroke
            },
            1.0,
        );
        let min_clamped_outer = rrect_area(121.0, 81.0, 20.5, 8.5, 8.5, 20.5);
        let min_clamped_inner = rrect_area(119.0, 79.0, 19.5, 7.5, 7.5, 19.5);
        assert_relative_eq!(
            mesh_total_area(&subpixel_mesh),
            min_clamped_outer - min_clamped_inner,
            epsilon = 1.0
        );
    }
}
