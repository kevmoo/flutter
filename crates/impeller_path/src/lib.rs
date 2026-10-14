#![forbid(unsafe_code)]

use impeller_geometry::{
    compute_cubic_subdivisions, compute_quadratic_subdivisions, Matrix, Point, Radians, Rect,
    Scalar, Size, K_EH_CLOSE_ENOUGH,
};
use std::ops::Mul;

pub const K_ARC_APPROXIMATION_MAGIC: Scalar = 0.551_915_05;

#[inline]
fn scalar_nearly_equal(a: Scalar, b: Scalar) -> bool {
    scalar_nearly_equal_tolerance(a, b, K_EH_CLOSE_ENOUGH)
}

#[inline]
fn scalar_nearly_equal_tolerance(a: Scalar, b: Scalar, tolerance: Scalar) -> bool {
    (a - b).abs() <= tolerance
}

#[inline]
fn size_is_finite(s: Size) -> bool {
    s.width.is_finite() && s.height.is_finite()
}

#[inline]
fn point_is_finite(p: Point) -> bool {
    p.x.is_finite() && p.y.is_finite()
}

#[inline]
fn rect_is_finite(r: Rect) -> bool {
    point_is_finite(r.origin) && size_is_finite(r.size)
}

#[inline]
fn rect_positive(r: Rect) -> Rect {
    r.get_positive()
}

#[inline]
fn rect_center(r: Rect) -> Point {
    Point::new(
        r.origin.x + r.size.width * 0.5,
        r.origin.y + r.size.height * 0.5,
    )
}

#[inline]
fn rect_shift(r: Rect, dx: Scalar, dy: Scalar) -> Rect {
    Rect::from_xywh(r.origin.x + dx, r.origin.y + dy, r.size.width, r.size.height)
}

#[inline]
fn rect_expand_ltrb(r: Rect, left: Scalar, top: Scalar, right: Scalar, bottom: Scalar) -> Rect {
    Rect::from_xywh(
        r.origin.x - left,
        r.origin.y - top,
        r.size.width + left + right,
        r.size.height + top + bottom,
    )
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum FillType {
    #[default]
    NonZero,
    Odd,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Cap {
    #[default]
    Butt,
    Round,
    Square,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Join {
    #[default]
    Miter,
    Round,
    Bevel,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Convexity {
    #[default]
    Unknown,
    Convex,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RoundingRadii {
    pub top_left: Size,
    pub bottom_left: Size,
    pub top_right: Size,
    pub bottom_right: Size,
}

impl RoundingRadii {
    pub const ZERO: Self = Self {
        top_left: Size::ZERO,
        bottom_left: Size::ZERO,
        top_right: Size::ZERO,
        bottom_right: Size::ZERO,
    };

    pub const fn from_single_radius(radius: Scalar) -> Self {
        let s = Size::new(radius, radius);
        Self {
            top_left: s,
            bottom_left: s,
            top_right: s,
            bottom_right: s,
        }
    }

    pub const fn make_radius(radius: Scalar) -> Self {
        Self::from_single_radius(radius)
    }

    pub const fn make_radii(radii: Size) -> Self {
        Self {
            top_left: radii,
            bottom_left: radii,
            top_right: radii,
            bottom_right: radii,
        }
    }

    pub const fn from_size(radii: Size) -> Self {
        Self::make_radii(radii)
    }

    pub const fn make_nine_patch(
        left: Scalar,
        top: Scalar,
        right: Scalar,
        bottom: Scalar,
    ) -> Self {
        Self {
            top_left: Size::new(left, top),
            top_right: Size::new(right, top),
            bottom_left: Size::new(left, bottom),
            bottom_right: Size::new(right, bottom),
        }
    }

    pub const fn from_radii(
        top_left: Size,
        top_right: Size,
        bottom_left: Size,
        bottom_right: Size,
    ) -> Self {
        Self {
            top_left,
            bottom_left,
            top_right,
            bottom_right,
        }
    }

    pub fn are_all_corners_zero(&self) -> bool {
        self.top_left.is_empty()
            && self.bottom_left.is_empty()
            && self.top_right.is_empty()
            && self.bottom_right.is_empty()
    }

    pub fn are_all_corners_empty(&self) -> bool {
        self.are_all_corners_zero()
    }

    pub fn is_finite(&self) -> bool {
        size_is_finite(self.top_left)
            && size_is_finite(self.top_right)
            && size_is_finite(self.bottom_left)
            && size_is_finite(self.bottom_right)
    }

    pub fn are_all_corners_same(&self) -> bool {
        self.are_all_corners_same_with_tolerance(K_EH_CLOSE_ENOUGH)
    }

    pub fn are_all_corners_same_with_tolerance(&self, tolerance: Scalar) -> bool {
        scalar_nearly_equal_tolerance(self.top_left.width, self.top_right.width, tolerance)
            && scalar_nearly_equal_tolerance(
                self.top_left.width,
                self.bottom_right.width,
                tolerance,
            )
            && scalar_nearly_equal_tolerance(self.top_left.width, self.bottom_left.width, tolerance)
            && scalar_nearly_equal_tolerance(self.top_left.height, self.top_right.height, tolerance)
            && scalar_nearly_equal_tolerance(
                self.top_left.height,
                self.bottom_right.height,
                tolerance,
            )
            && scalar_nearly_equal_tolerance(
                self.top_left.height,
                self.bottom_left.height,
                tolerance,
            )
    }

    pub fn are_all_corners_circular(&self) -> bool {
        scalar_nearly_equal(self.top_left.width, self.top_left.height)
            && scalar_nearly_equal(self.top_right.width, self.top_right.height)
            && scalar_nearly_equal(self.bottom_left.width, self.bottom_left.height)
            && scalar_nearly_equal(self.bottom_right.width, self.bottom_right.height)
    }

    pub fn scaled(&self, in_bounds: Rect) -> Self {
        if !rect_is_finite(in_bounds) {
            return Self::ZERO;
        }
        let bounds = rect_positive(in_bounds);
        if bounds.is_empty() || self.are_all_corners_empty() || !self.is_finite() {
            return Self::ZERO;
        }

        let normalize_empty = |s: Size| -> Size {
            if s.is_empty() {
                Size::ZERO
            } else {
                s
            }
        };

        let mut radii = Self {
            top_left: normalize_empty(self.top_left),
            top_right: normalize_empty(self.top_right),
            bottom_left: normalize_empty(self.bottom_left),
            bottom_right: normalize_empty(self.bottom_right),
        };

        let adjust_scale = |r1: Scalar, r2: Scalar, dimension: Scalar, scale: &mut Scalar| {
            if r1 + r2 > dimension {
                *scale = (*scale).min(dimension / (r1 + r2));
            }
        };

        let mut scale: Scalar = 1.0;
        adjust_scale(
            radii.top_left.width,
            radii.top_right.width,
            bounds.size.width,
            &mut scale,
        );
        adjust_scale(
            radii.bottom_left.width,
            radii.bottom_right.width,
            bounds.size.width,
            &mut scale,
        );
        adjust_scale(
            radii.top_left.height,
            radii.bottom_left.height,
            bounds.size.height,
            &mut scale,
        );
        adjust_scale(
            radii.top_right.height,
            radii.bottom_right.height,
            bounds.size.height,
            &mut scale,
        );

        if scale < 1.0 {
            radii = radii * scale;
        }

        radii
    }
}

impl Mul<Scalar> for RoundingRadii {
    type Output = Self;

    fn mul(self, scale: Scalar) -> Self {
        Self {
            top_left: Size::new(self.top_left.width * scale, self.top_left.height * scale),
            top_right: Size::new(self.top_right.width * scale, self.top_right.height * scale),
            bottom_left: Size::new(
                self.bottom_left.width * scale,
                self.bottom_left.height * scale,
            ),
            bottom_right: Size::new(
                self.bottom_right.width * scale,
                self.bottom_right.height * scale,
            ),
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RoundRect {
    pub bounds: Rect,
    pub radii: RoundingRadii,
}

impl RoundRect {
    pub const fn new() -> Self {
        Self {
            bounds: Rect::ZERO,
            radii: RoundingRadii::ZERO,
        }
    }

    pub fn from_rect(rect: Rect) -> Self {
        Self::from_rect_radii(rect, RoundingRadii::ZERO)
    }

    pub fn make_rect(rect: Rect) -> Self {
        Self::from_rect(rect)
    }

    pub fn from_oval(rect: Rect) -> Self {
        let bounds = rect.get_positive();
        Self::from_rect_radii(
            bounds,
            RoundingRadii::make_radii(Size::new(bounds.size.width * 0.5, bounds.size.height * 0.5)),
        )
    }

    pub fn make_oval(rect: Rect) -> Self {
        Self::from_oval(rect)
    }

    pub fn from_rect_radius(rect: Rect, radius: Scalar) -> Self {
        Self::from_rect_radii(rect, RoundingRadii::from_single_radius(radius))
    }

    pub fn make_rect_radius(rect: Rect, radius: Scalar) -> Self {
        Self::from_rect_radius(rect, radius)
    }

    pub fn from_rect_xy(rect: Rect, x_radius: Scalar, y_radius: Scalar) -> Self {
        Self::from_rect_radii(rect, RoundingRadii::make_radii(Size::new(x_radius, y_radius)))
    }

    pub fn make_rect_xy(rect: Rect, x_radius: Scalar, y_radius: Scalar) -> Self {
        Self::from_rect_xy(rect, x_radius, y_radius)
    }

    pub fn from_rect_size(rect: Rect, corner_radii: Size) -> Self {
        Self::from_rect_radii(rect, RoundingRadii::make_radii(corner_radii))
    }

    pub fn make_rect_size(rect: Rect, corner_radii: Size) -> Self {
        Self::from_rect_size(rect, corner_radii)
    }

    pub fn from_nine_patch(
        rect: Rect,
        left: Scalar,
        top: Scalar,
        right: Scalar,
        bottom: Scalar,
    ) -> Self {
        Self::from_rect_radii(
            rect,
            RoundingRadii::make_nine_patch(left, top, right, bottom),
        )
    }

    pub fn make_nine_patch(
        rect: Rect,
        left: Scalar,
        top: Scalar,
        right: Scalar,
        bottom: Scalar,
    ) -> Self {
        Self::from_nine_patch(rect, left, top, right, bottom)
    }

    pub fn from_rect_radii(in_bounds: Rect, in_radii: RoundingRadii) -> Self {
        if !rect_is_finite(in_bounds) {
            return Self::default();
        }
        let bounds = rect_positive(in_bounds);
        Self {
            bounds,
            radii: in_radii.scaled(bounds),
        }
    }

    pub fn make_rect_radii(in_bounds: Rect, in_radii: RoundingRadii) -> Self {
        Self::from_rect_radii(in_bounds, in_radii)
    }

    pub const fn get_bounds(&self) -> Rect {
        self.bounds
    }

    pub const fn get_radii(&self) -> RoundingRadii {
        self.radii
    }

    pub fn is_finite(&self) -> bool {
        rect_is_finite(self.bounds) && self.radii.is_finite()
    }

    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }

    pub fn is_rect(&self) -> bool {
        !self.bounds.is_empty() && self.radii.are_all_corners_empty()
    }

    pub fn is_oval(&self) -> bool {
        !self.bounds.is_empty()
            && self.radii.are_all_corners_same()
            && scalar_nearly_equal(self.radii.top_left.width, self.bounds.get_width() * 0.5)
            && scalar_nearly_equal(self.radii.top_left.height, self.bounds.get_height() * 0.5)
    }

    pub fn contains(&self, p: Point) -> bool {
        if !self.bounds.contains(p) {
            return false;
        }
        let left = self.bounds.get_left();
        let top = self.bounds.get_top();
        let right = self.bounds.get_right();
        let bottom = self.bounds.get_bottom();

        corner_contains(
            p,
            Point::new(left, top),
            Point::new(-1.0, -1.0),
            self.radii.top_left,
        ) && corner_contains(
            p,
            Point::new(right, top),
            Point::new(1.0, -1.0),
            self.radii.top_right,
        ) && corner_contains(
            p,
            Point::new(left, bottom),
            Point::new(-1.0, 1.0),
            self.radii.bottom_left,
        ) && corner_contains(
            p,
            Point::new(right, bottom),
            Point::new(1.0, 1.0),
            self.radii.bottom_right,
        )
    }

    pub fn shift(&self, dx: Scalar, dy: Scalar) -> Self {
        Self::from_rect_radii(rect_shift(self.bounds, dx, dy), self.radii)
    }

    pub fn expand(&self, amount: Scalar) -> Self {
        self.expand_ltrb(amount, amount, amount, amount)
    }

    pub fn expand_xy(&self, horizontal: Scalar, vertical: Scalar) -> Self {
        self.expand_ltrb(horizontal, vertical, horizontal, vertical)
    }

    pub fn expand_ltrb(
        &self,
        left: Scalar,
        top: Scalar,
        right: Scalar,
        bottom: Scalar,
    ) -> Self {
        Self::from_rect_radii(
            rect_expand_ltrb(self.bounds, left, top, right, bottom),
            self.radii,
        )
    }
}

fn corner_contains(p: Point, corner: Point, direction: Point, radii: Size) -> bool {
    if radii.is_empty() {
        return true;
    }
    let corner_relative = Point::new(
        (corner.x - p.x) * direction.x,
        (corner.y - p.y) * direction.y,
    );
    let quadrant_relative = Point::new(
        radii.width - corner_relative.x,
        radii.height - corner_relative.y,
    );
    if quadrant_relative.x <= 0.0 || quadrant_relative.y <= 0.0 {
        return true;
    }
    let unit = Point::new(
        quadrant_relative.x / radii.width,
        quadrant_relative.y / radii.height,
    );
    unit.get_length_squared() <= 1.0
}

pub trait VertexWriter {
    fn end_contour(&mut self);
    fn write(&mut self, point: Point);
}

pub struct FanVertexWriter<'a> {
    count: usize,
    index_count: usize,
    point_buffer: &'a mut [Point],
    index_buffer: &'a mut [u16],
}

impl<'a> FanVertexWriter<'a> {
    pub fn new(point_buffer: &'a mut [Point], index_buffer: &'a mut [u16]) -> Self {
        Self {
            count: 0,
            index_count: 0,
            point_buffer,
            index_buffer,
        }
    }

    pub const fn get_index_count(&self) -> usize {
        self.index_count
    }
}

impl VertexWriter for FanVertexWriter<'_> {
    fn end_contour(&mut self) {
        if self.count == 0 {
            return;
        }
        if self.index_count < self.index_buffer.len() {
            self.index_buffer[self.index_count] = 0xFFFF;
        }
        self.index_count += 1;
    }

    fn write(&mut self, point: Point) {
        if self.index_count < self.index_buffer.len() {
            self.index_buffer[self.index_count] = self.count as u16;
        }
        self.index_count += 1;
        if self.count < self.point_buffer.len() {
            self.point_buffer[self.count] = point;
        }
        self.count += 1;
    }
}

pub struct StripVertexWriter<'a> {
    count: usize,
    index_count: usize,
    contour_start: usize,
    point_buffer: &'a mut [Point],
    index_buffer: &'a mut [u16],
}

impl<'a> StripVertexWriter<'a> {
    pub fn new(point_buffer: &'a mut [Point], index_buffer: &'a mut [u16]) -> Self {
        Self {
            count: 0,
            index_count: 0,
            contour_start: 0,
            point_buffer,
            index_buffer,
        }
    }

    pub const fn get_index_count(&self) -> usize {
        self.index_count
    }
}

impl VertexWriter for StripVertexWriter<'_> {
    fn end_contour(&mut self) {
        if self.count == 0 || self.contour_start == self.count - 1 {
            return;
        }

        let start = self.contour_start;
        let end = self.count - 1;

        if self.index_count < self.index_buffer.len() {
            self.index_buffer[self.index_count] = start as u16;
        }
        self.index_count += 1;

        let mut a = start + 1;
        let mut b = end;
        while a < b {
            if self.index_count < self.index_buffer.len() {
                self.index_buffer[self.index_count] = a as u16;
            }
            self.index_count += 1;
            if self.index_count < self.index_buffer.len() {
                self.index_buffer[self.index_count] = b as u16;
            }
            self.index_count += 1;
            a += 1;
            b -= 1;
        }
        if a == b {
            if self.index_count < self.index_buffer.len() {
                self.index_buffer[self.index_count] = a as u16;
            }
            self.index_count += 1;
        }

        self.contour_start = self.count;
        if self.index_count < self.index_buffer.len() {
            self.index_buffer[self.index_count] = 0xFFFF;
        }
        self.index_count += 1;
    }

    fn write(&mut self, point: Point) {
        if self.count < self.point_buffer.len() {
            self.point_buffer[self.count] = point;
        }
        self.count += 1;
    }
}

#[inline]
fn linear_solve(t: Scalar, p0: Scalar, p1: Scalar) -> Scalar {
    p0 + t * (p1 - p0)
}

#[inline]
fn quadratic_solve(t: Scalar, p0: Scalar, p1: Scalar, p2: Scalar) -> Scalar {
    let u = 1.0 - t;
    u * u * p0 + 2.0 * u * t * p1 + t * t * p2
}

#[inline]
fn quadratic_solve_derivative(t: Scalar, p0: Scalar, p1: Scalar, p2: Scalar) -> Scalar {
    2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1)
}

#[inline]
fn conic_solve(t: Scalar, p0: Scalar, p1: Scalar, p2: Scalar, w: Scalar) -> Scalar {
    let u = 1.0 - t;
    let c0 = u * u;
    let c1 = 2.0 * t * u * w;
    let c2 = t * t;
    (p0 * c0 + p1 * c1 + p2 * c2) / (c0 + c1 + c2)
}

#[inline]
fn cubic_solve(t: Scalar, p0: Scalar, p1: Scalar, p2: Scalar, p3: Scalar) -> Scalar {
    let u = 1.0 - t;
    u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3
}

#[inline]
fn cubic_solve_derivative(t: Scalar, p0: Scalar, p1: Scalar, p2: Scalar, p3: Scalar) -> Scalar {
    let u = 1.0 - t;
    -3.0 * p0 * u * u
        + p1 * (3.0 * u * u - 6.0 * u * t)
        + p2 * (6.0 * u * t - 3.0 * t * t)
        + 3.0 * p3 * t * t
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct LinearPathComponent {
    pub p1: Point,
    pub p2: Point,
}

impl LinearPathComponent {
    pub const fn new(p1: Point, p2: Point) -> Self {
        Self { p1, p2 }
    }

    pub fn solve(&self, time: Scalar) -> Point {
        Point::new(
            linear_solve(time, self.p1.x, self.p2.x),
            linear_solve(time, self.p1.y, self.p2.y),
        )
    }

    pub fn solve_derivative(&self, _time: Scalar) -> Point {
        self.p2 - self.p1
    }

    pub fn append_polyline_points(&self, points: &mut Vec<Point>) {
        if points.last().copied() != Some(self.p2) {
            points.push(self.p2);
        }
    }

    pub fn to_linear_path_components(&self) -> Vec<Point> {
        vec![self.p2]
    }

    pub fn extrema(&self) -> Vec<Point> {
        vec![self.p1, self.p2]
    }

    pub fn get_start_direction(&self) -> Option<Point> {
        if self.p1 == self.p2 {
            None
        } else {
            Some((self.p1 - self.p2).normalize())
        }
    }

    pub fn get_end_direction(&self) -> Option<Point> {
        if self.p1 == self.p2 {
            None
        } else {
            Some((self.p2 - self.p1).normalize())
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct QuadPathComponent {
    pub p1: Point,
    pub cp: Point,
    pub p2: Point,
}

pub type QuadraticPathComponent = QuadPathComponent;

fn quadratic_path_bounding_populate_values(
    values: &mut Vec<Scalar>,
    p1: Scalar,
    cp: Scalar,
    p2: Scalar,
) {
    // Derivative of (1-t)^2*p1 + 2*(1-t)*t*cp + t^2*p2 is:
    // 2 * ((p1 - 2*cp + p2)*t + (cp - p1)) = 0
    // => t = (p1 - cp) / (p1 - 2*cp + p2)
    let denom = p1 - 2.0 * cp + p2;
    if denom.abs() < 1e-12 {
        return;
    }
    let t = (p1 - cp) / denom;
    if (0.0..=1.0).contains(&t) {
        values.push(t);
    }
}

impl QuadPathComponent {
    pub const fn new(p1: Point, cp: Point, p2: Point) -> Self {
        Self { p1, cp, p2 }
    }

    pub fn solve(&self, time: Scalar) -> Point {
        Point::new(
            quadratic_solve(time, self.p1.x, self.cp.x, self.p2.x),
            quadratic_solve(time, self.p1.y, self.cp.y, self.p2.y),
        )
    }

    pub fn solve_derivative(&self, time: Scalar) -> Point {
        Point::new(
            quadratic_solve_derivative(time, self.p1.x, self.cp.x, self.p2.x),
            quadratic_solve_derivative(time, self.p1.y, self.cp.y, self.p2.y),
        )
    }

    fn subdivision_line_count(&self, scale_factor: Scalar) -> Scalar {
        let raw = compute_quadratic_subdivisions(scale_factor.max(0.0), self.p1, self.cp, self.p2);
        if raw.is_finite() && raw > 0.0 {
            raw.ceil()
        } else {
            0.0
        }
    }

    pub fn to_linear_path_components(&self, scale_factor: Scalar) -> Vec<Point> {
        let mut out = Vec::new();
        self.to_linear_path_components_with(scale_factor, |p| out.push(p));
        out
    }

    pub fn to_linear_path_components_with<F: FnMut(Point)>(
        &self,
        scale_factor: Scalar,
        mut proc: F,
    ) {
        let line_count = self.subdivision_line_count(scale_factor) as usize;
        for i in 1..line_count {
            proc(self.solve(i as Scalar / line_count as Scalar));
        }
        proc(self.p2);
    }

    pub fn to_linear_path_components_writer(&self, scale: Scalar, writer: &mut dyn VertexWriter) {
        let line_count = self.subdivision_line_count(scale) as usize;
        for i in 1..line_count {
            writer.write(self.solve(i as Scalar / line_count as Scalar));
        }
        writer.write(self.p2);
    }

    pub fn append_polyline_points(&self, scale_factor: Scalar, points: &mut Vec<Point>) {
        self.to_linear_path_components_with(scale_factor, |p| points.push(p));
    }

    pub fn count_linear_path_components(&self, scale: Scalar) -> usize {
        (self.subdivision_line_count(scale) as usize) + 2
    }

    pub fn extrema(&self) -> Vec<Point> {
        let mut values = Vec::with_capacity(2);
        quadratic_path_bounding_populate_values(&mut values, self.p1.x, self.cp.x, self.p2.x);
        quadratic_path_bounding_populate_values(&mut values, self.p1.y, self.cp.y, self.p2.y);

        let mut points = Vec::with_capacity(2 + values.len());
        points.push(self.p1);
        points.push(self.p2);
        for t in values {
            points.push(self.solve(t));
        }
        points
    }

    pub fn get_start_direction(&self) -> Option<Point> {
        if self.p1 != self.cp {
            Some((self.p1 - self.cp).normalize())
        } else if self.p1 != self.p2 {
            Some((self.p1 - self.p2).normalize())
        } else {
            None
        }
    }

    pub fn get_end_direction(&self) -> Option<Point> {
        if self.p2 != self.cp {
            Some((self.p2 - self.cp).normalize())
        } else if self.p2 != self.p1 {
            Some((self.p2 - self.p1).normalize())
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ConicPathComponent {
    pub p1: Point,
    pub cp: Point,
    pub p2: Point,
    pub weight: Point,
}

impl ConicPathComponent {
    pub const fn new(p1: Point, cp: Point, p2: Point, weight: Scalar) -> Self {
        Self {
            p1,
            cp,
            p2,
            weight: Point::new(weight, weight),
        }
    }

    pub fn solve(&self, time: Scalar) -> Point {
        Point::new(
            conic_solve(time, self.p1.x, self.cp.x, self.p2.x, self.weight.x),
            conic_solve(time, self.p1.y, self.cp.y, self.p2.y, self.weight.y),
        )
    }

    pub fn subdivide_to_quadratic_points(&self) -> [Point; 5] {
        let w = self.weight.x;
        let scale = 1.0 / (1.0 + w);
        let tp1 = self.p1 * scale;
        let tcp = self.cp * (w * scale);
        let tp2 = self.p2 * scale;

        let sub_cp1 = tp1 + tcp;
        let sub_cp2 = tcp + tp2;
        let sub_mid = (tp1 + tcp + tcp + tp2) * 0.5;

        [self.p1, sub_cp1, sub_mid, sub_cp2, self.p2]
    }

    pub fn to_quadratic_path_components(&self) -> [QuadraticPathComponent; 2] {
        let pts = self.subdivide_to_quadratic_points();
        [
            QuadraticPathComponent::new(pts[0], pts[1], pts[2]),
            QuadraticPathComponent::new(pts[2], pts[3], pts[4]),
        ]
    }

    pub fn extrema(&self) -> Vec<Point> {
        let mut points = Vec::new();
        for quad in self.to_quadratic_path_components() {
            points.extend(quad.extrema());
        }
        points
    }

    pub fn to_linear_path_components(&self, scale_factor: Scalar) -> Vec<Point> {
        let mut out = Vec::new();
        for quad in self.to_quadratic_path_components() {
            quad.to_linear_path_components_with(scale_factor, |p| {
                if out.last().copied() != Some(p) {
                    out.push(p);
                }
            });
        }
        out
    }

    pub fn append_polyline_points(&self, scale_factor: Scalar, points: &mut Vec<Point>) {
        for p in self.to_linear_path_components(scale_factor) {
            if points.last().copied() != Some(p) {
                points.push(p);
            }
        }
    }

    pub fn get_start_direction(&self) -> Option<Point> {
        if self.p1 != self.cp {
            Some((self.p1 - self.cp).normalize())
        } else if self.p1 != self.p2 {
            Some((self.p1 - self.p2).normalize())
        } else {
            None
        }
    }

    pub fn get_end_direction(&self) -> Option<Point> {
        if self.p2 != self.cp {
            Some((self.p2 - self.cp).normalize())
        } else if self.p2 != self.p1 {
            Some((self.p2 - self.p1).normalize())
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct CubicPathComponent {
    pub p1: Point,
    pub cp1: Point,
    pub cp2: Point,
    pub p2: Point,
}

fn cubic_path_bounding_populate_values(
    values: &mut Vec<Scalar>,
    p1: Scalar,
    p2: Scalar,
    p3: Scalar,
    p4: Scalar,
) {
    let a = 3.0 * (-p1 + 3.0 * p2 - 3.0 * p3 + p4);
    let b = 6.0 * (p1 - 2.0 * p2 + p3);
    let c = 3.0 * (p2 - p1);

    if a.abs() < 1e-12 {
        if b.abs() < 1e-12 {
            return;
        }
        let t = -c / b;
        if (0.0..=1.0).contains(&t) {
            values.push(t);
        }
        return;
    }

    let b2_minus_4ac = (b * b) - (4.0 * a * c);
    if b2_minus_4ac < 0.0 {
        return;
    }

    let root_b2_minus_4ac = b2_minus_4ac.sqrt();
    let q = if b < 0.0 {
        -(b - root_b2_minus_4ac) * 0.5
    } else {
        -(b + root_b2_minus_4ac) * 0.5
    };

    {
        let t = q / a;
        if (0.0..=1.0).contains(&t) {
            values.push(t);
        }
    }

    if q.abs() >= 1e-12 {
        let t = c / q;
        if (0.0..=1.0).contains(&t) {
            values.push(t);
        }
    }
}

impl CubicPathComponent {
    pub const fn new(p1: Point, cp1: Point, cp2: Point, p2: Point) -> Self {
        Self { p1, cp1, cp2, p2 }
    }

    pub fn from_quadratic(q: QuadraticPathComponent) -> Self {
        Self {
            p1: q.p1,
            cp1: q.p1 + (q.cp - q.p1) * (2.0 / 3.0),
            cp2: q.p2 + (q.cp - q.p2) * (2.0 / 3.0),
            p2: q.p2,
        }
    }

    pub fn solve(&self, time: Scalar) -> Point {
        Point::new(
            cubic_solve(time, self.p1.x, self.cp1.x, self.cp2.x, self.p2.x),
            cubic_solve(time, self.p1.y, self.cp1.y, self.cp2.y, self.p2.y),
        )
    }

    pub fn solve_derivative(&self, time: Scalar) -> Point {
        Point::new(
            cubic_solve_derivative(time, self.p1.x, self.cp1.x, self.cp2.x, self.p2.x),
            cubic_solve_derivative(time, self.p1.y, self.cp1.y, self.cp2.y, self.p2.y),
        )
    }

    fn subdivision_line_count(&self, scale: Scalar) -> Scalar {
        let raw = compute_cubic_subdivisions(scale.max(0.0), self.p1, self.cp1, self.cp2, self.p2);
        if raw.is_finite() && raw > 0.0 {
            raw.ceil()
        } else {
            0.0
        }
    }

    pub fn to_linear_path_components(&self, scale: Scalar) -> Vec<Point> {
        let mut out = Vec::new();
        self.to_linear_path_components_with(scale, |p| out.push(p));
        out
    }

    pub fn to_linear_path_components_with<F: FnMut(Point)>(&self, scale: Scalar, mut proc: F) {
        let line_count = self.subdivision_line_count(scale) as usize;
        for i in 1..line_count {
            proc(self.solve(i as Scalar / line_count as Scalar));
        }
        proc(self.p2);
    }

    pub fn to_linear_path_components_writer(&self, scale: Scalar, writer: &mut dyn VertexWriter) {
        let line_count = self.subdivision_line_count(scale) as usize;
        for i in 1..line_count {
            writer.write(self.solve(i as Scalar / line_count as Scalar));
        }
        writer.write(self.p2);
    }

    pub fn append_polyline_points(&self, scale: Scalar, points: &mut Vec<Point>) {
        self.to_linear_path_components_with(scale, |p| points.push(p));
    }

    pub fn count_linear_path_components(&self, scale: Scalar) -> usize {
        (self.subdivision_line_count(scale) as usize) + 2
    }

    fn lower(&self) -> QuadraticPathComponent {
        QuadraticPathComponent::new(
            (self.cp1 - self.p1) * 3.0,
            (self.cp2 - self.cp1) * 3.0,
            (self.p2 - self.cp2) * 3.0,
        )
    }

    pub fn subsegment(&self, t0: Scalar, t1: Scalar) -> CubicPathComponent {
        let p0 = self.solve(t0);
        let p3 = self.solve(t1);
        let d = self.lower();
        let scale = (t1 - t0) * (1.0 / 3.0);
        let p1 = p0 + d.solve(t0) * scale;
        let p2 = p3 - d.solve(t1) * scale;
        CubicPathComponent::new(p0, p1, p2, p3)
    }

    pub fn extrema(&self) -> Vec<Point> {
        let mut values = Vec::with_capacity(4);
        cubic_path_bounding_populate_values(
            &mut values,
            self.p1.x,
            self.cp1.x,
            self.cp2.x,
            self.p2.x,
        );
        cubic_path_bounding_populate_values(
            &mut values,
            self.p1.y,
            self.cp1.y,
            self.cp2.y,
            self.p2.y,
        );

        let mut points = Vec::with_capacity(2 + values.len());
        points.push(self.p1);
        points.push(self.p2);
        for t in values {
            points.push(self.solve(t));
        }
        points
    }

    pub fn get_start_direction(&self) -> Option<Point> {
        if self.p1 != self.cp1 {
            Some((self.p1 - self.cp1).normalize())
        } else if self.p1 != self.cp2 {
            Some((self.p1 - self.cp2).normalize())
        } else if self.p1 != self.p2 {
            Some((self.p1 - self.p2).normalize())
        } else {
            None
        }
    }

    pub fn get_end_direction(&self) -> Option<Point> {
        if self.p2 != self.cp2 {
            Some((self.p2 - self.cp2).normalize())
        } else if self.p2 != self.cp1 {
            Some((self.p2 - self.cp1).normalize())
        } else if self.p2 != self.p1 {
            Some((self.p2 - self.p1).normalize())
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ContourComponent {
    pub destination: Point,
    pub is_closed: bool,
}

impl ContourComponent {
    pub const fn new(destination: Point, is_closed: bool) -> Self {
        Self {
            destination,
            is_closed,
        }
    }

    pub const fn is_closed(&self) -> bool {
        self.is_closed
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PathComponent {
    Contour(ContourComponent),
    Linear(LinearPathComponent),
    Quad(QuadPathComponent),
    Cubic(CubicPathComponent),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolylineContour {
    pub start_index: usize,
    pub end_index: usize,
    pub is_closed: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathPolyline {
    pub points: Vec<Point>,
    pub contours: Vec<PolylineContour>,
}

pub type Polyline = PathPolyline;

impl PathPolyline {
    pub fn get_point(&self, index: usize) -> Point {
        self.points[index]
    }

    pub fn get_contour_point_bounds(&self, contour_index: usize) -> (usize, usize) {
        if contour_index >= self.contours.len() {
            return (self.points.len(), self.points.len());
        }
        let c = &self.contours[contour_index];
        (c.start_index, c.end_index)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Path {
    pub components: Vec<PathComponent>,
    pub fill_type: FillType,
    pub convexity: Convexity,
}

impl Path {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_fill_type(&self) -> FillType {
        self.fill_type
    }

    pub fn get_convexity(&self) -> Convexity {
        self.convexity
    }

    pub fn set_convexity(&mut self, convexity: Convexity) {
        self.convexity = convexity;
    }

    pub fn is_convex(&self) -> bool {
        self.convexity == Convexity::Convex
    }

    pub fn is_empty(&self) -> bool {
        !self
            .components
            .iter()
            .any(|c| !matches!(c, PathComponent::Contour(_)))
    }

    pub fn is_single_contour(&self) -> bool {
        let contour_count = self
            .components
            .iter()
            .filter(|c| matches!(c, PathComponent::Contour(_)))
            .count();
        contour_count <= 1
            || (contour_count == 2
                && matches!(self.components.last(), Some(PathComponent::Contour(_))))
    }

    pub fn get_component_count(&self) -> usize {
        self.components.len()
    }

    pub fn count_storage(&self, scale: Scalar) -> (usize, usize) {
        let mut points = 0usize;
        let mut contours = 0usize;
        for comp in &self.components {
            match comp {
                PathComponent::Contour(_) => contours += 1,
                PathComponent::Linear(_) => points += 2,
                PathComponent::Quad(q) => points += q.count_linear_path_components(scale),
                PathComponent::Cubic(c) => points += c.count_linear_path_components(scale),
            }
        }
        (points, contours)
    }

    pub fn write_polyline_to(&self, scale: Scalar, writer: &mut dyn VertexWriter) {
        let mut started_contour = false;
        let mut first_point = true;

        for (i, comp) in self.components.iter().enumerate() {
            match comp {
                PathComponent::Linear(linear) => {
                    if first_point {
                        writer.write(linear.p1);
                        first_point = false;
                    }
                    writer.write(linear.p2);
                }
                PathComponent::Quad(quad) => {
                    if first_point {
                        writer.write(quad.p1);
                        first_point = false;
                    }
                    quad.to_linear_path_components_writer(scale, writer);
                }
                PathComponent::Cubic(cubic) => {
                    if first_point {
                        writer.write(cubic.p1);
                        first_point = false;
                    }
                    cubic.to_linear_path_components_writer(scale, writer);
                }
                PathComponent::Contour(_) => {
                    if i == self.components.len() - 1 {
                        continue;
                    }
                    if started_contour {
                        writer.end_contour();
                    }
                    started_contour = true;
                    first_point = true;
                }
            }
        }
        if started_contour {
            writer.end_contour();
        }
    }

    pub fn write_polyline(&self, scale: Scalar) -> PathPolyline {
        self.create_polyline(scale)
    }

    pub fn create_polyline(&self, scale: Scalar) -> PathPolyline {
        let mut points = Vec::new();
        let mut contours: Vec<PolylineContour> = Vec::new();
        let mut current_start = 0usize;
        let mut current_closed = false;
        let mut in_contour = false;
        let mut pending_contour: Option<ContourComponent> = None;

        for (i, comp) in self.components.iter().enumerate() {
            match *comp {
                PathComponent::Contour(c) => {
                    if i == self.components.len() - 1 {
                        break;
                    }
                    if in_contour && points.len() > current_start {
                        contours.push(PolylineContour {
                            start_index: current_start,
                            end_index: points.len(),
                            is_closed: current_closed,
                        });
                        in_contour = false;
                    }
                    pending_contour = Some(c);
                }
                PathComponent::Linear(l) => {
                    if let Some(c) = pending_contour.take() {
                        current_start = points.len();
                        current_closed = c.is_closed;
                        in_contour = true;
                        points.push(c.destination);
                    } else if !in_contour {
                        current_start = points.len();
                        current_closed = false;
                        in_contour = true;
                        points.push(l.p1);
                    }
                    l.append_polyline_points(&mut points);
                }
                PathComponent::Quad(q) => {
                    if let Some(c) = pending_contour.take() {
                        current_start = points.len();
                        current_closed = c.is_closed;
                        in_contour = true;
                        points.push(c.destination);
                    } else if !in_contour {
                        current_start = points.len();
                        current_closed = false;
                        in_contour = true;
                        points.push(q.p1);
                    }
                    q.append_polyline_points(scale, &mut points);
                }
                PathComponent::Cubic(c) => {
                    if let Some(cont) = pending_contour.take() {
                        current_start = points.len();
                        current_closed = cont.is_closed;
                        in_contour = true;
                        points.push(cont.destination);
                    } else if !in_contour {
                        current_start = points.len();
                        current_closed = false;
                        in_contour = true;
                        points.push(c.p1);
                    }
                    c.append_polyline_points(scale, &mut points);
                }
            }
        }
        if in_contour && points.len() > current_start {
            contours.push(PolylineContour {
                start_index: current_start,
                end_index: points.len(),
                is_closed: current_closed,
            });
        }
        PathPolyline { points, contours }
    }

    pub fn get_bounding_box(&self) -> Option<Rect> {
        let mut min_x = Scalar::INFINITY;
        let mut min_y = Scalar::INFINITY;
        let mut max_x = Scalar::NEG_INFINITY;
        let mut max_y = Scalar::NEG_INFINITY;
        let mut found = false;

        let mut clamp = |p: Point| {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
            found = true;
        };

        for comp in &self.components {
            match comp {
                PathComponent::Contour(_) => {}
                PathComponent::Linear(l) => {
                    clamp(l.p1);
                    clamp(l.p2);
                }
                PathComponent::Quad(q) => {
                    for p in q.extrema() {
                        clamp(p);
                    }
                }
                PathComponent::Cubic(c) => {
                    for p in c.extrema() {
                        clamp(p);
                    }
                }
            }
        }

        if !found {
            None
        } else {
            Some(Rect::from_xywh(min_x, min_y, max_x - min_x, max_y - min_y))
        }
    }

    pub fn get_transformed_bounding_box(&self, transform: &Matrix) -> Option<Rect> {
        let b = self.get_bounding_box()?;
        let corners = [
            transform.transform_point(Point::new(b.get_left(), b.get_top())),
            transform.transform_point(Point::new(b.get_right(), b.get_top())),
            transform.transform_point(Point::new(b.get_left(), b.get_bottom())),
            transform.transform_point(Point::new(b.get_right(), b.get_bottom())),
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
        Some(Rect::from_ltrb(min_x, min_y, max_x, max_y))
    }

    pub fn shift(&self, offset: Point) -> Self {
        let components = self
            .components
            .iter()
            .map(|comp| match *comp {
                PathComponent::Contour(c) => PathComponent::Contour(ContourComponent {
                    destination: c.destination + offset,
                    is_closed: c.is_closed,
                }),
                PathComponent::Linear(l) => PathComponent::Linear(LinearPathComponent {
                    p1: l.p1 + offset,
                    p2: l.p2 + offset,
                }),
                PathComponent::Quad(q) => PathComponent::Quad(QuadPathComponent {
                    p1: q.p1 + offset,
                    cp: q.cp + offset,
                    p2: q.p2 + offset,
                }),
                PathComponent::Cubic(c) => PathComponent::Cubic(CubicPathComponent {
                    p1: c.p1 + offset,
                    cp1: c.cp1 + offset,
                    cp2: c.cp2 + offset,
                    p2: c.p2 + offset,
                }),
            })
            .collect();
        Self {
            components,
            fill_type: self.fill_type,
            convexity: self.convexity,
        }
    }

    pub fn transform(&self, matrix: &Matrix) -> Self {
        let components = self
            .components
            .iter()
            .map(|comp| match *comp {
                PathComponent::Contour(c) => PathComponent::Contour(ContourComponent {
                    destination: matrix.transform_point(c.destination),
                    is_closed: c.is_closed,
                }),
                PathComponent::Linear(l) => PathComponent::Linear(LinearPathComponent {
                    p1: matrix.transform_point(l.p1),
                    p2: matrix.transform_point(l.p2),
                }),
                PathComponent::Quad(q) => PathComponent::Quad(QuadPathComponent {
                    p1: matrix.transform_point(q.p1),
                    cp: matrix.transform_point(q.cp),
                    p2: matrix.transform_point(q.p2),
                }),
                PathComponent::Cubic(c) => PathComponent::Cubic(CubicPathComponent {
                    p1: matrix.transform_point(c.p1),
                    cp1: matrix.transform_point(c.cp1),
                    cp2: matrix.transform_point(c.cp2),
                    p2: matrix.transform_point(c.p2),
                }),
            })
            .collect();
        Self {
            components,
            fill_type: self.fill_type,
            convexity: self.convexity,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PathBuilder {
    components: Vec<PathComponent>,
    current: Point,
    subpath_start: Point,
    current_contour_location: usize,
    contour_count: usize,
    convexity: Convexity,
}

impl Default for PathBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PathBuilder {
    pub fn new() -> Self {
        let mut pb = Self {
            components: Vec::new(),
            current: Point::ZERO,
            subpath_start: Point::ZERO,
            current_contour_location: 0,
            contour_count: 0,
            convexity: Convexity::Unknown,
        };
        pb.add_contour_component(Point::ZERO, false);
        pb
    }

    pub fn reserve(&mut self, _point_size: usize, verb_size: usize) -> &mut Self {
        self.components.reserve(verb_size);
        self
    }

    pub fn set_convexity(&mut self, value: Convexity) -> &mut Self {
        self.convexity = value;
        self
    }

    pub const fn get_convexity(&self) -> Convexity {
        self.convexity
    }

    fn add_contour_component(&mut self, destination: Point, is_closed: bool) {
        if matches!(self.components.last(), Some(PathComponent::Contour(_))) {
            self.components[self.current_contour_location] =
                PathComponent::Contour(ContourComponent {
                    destination,
                    is_closed,
                });
        } else {
            self.current_contour_location = self.components.len();
            self.components
                .push(PathComponent::Contour(ContourComponent {
                    destination,
                    is_closed,
                }));
            self.contour_count += 1;
        }
    }

    fn set_contour_closed(&mut self, is_closed: bool) {
        if let Some(PathComponent::Contour(ref mut c)) =
            self.components.get_mut(self.current_contour_location)
        {
            c.is_closed = is_closed;
        }
    }

    fn add_linear_component(&mut self, p1: Point, p2: Point) {
        self.components
            .push(PathComponent::Linear(LinearPathComponent { p1, p2 }));
    }

    fn add_linear_component_if_needed(&mut self, p1: Point, p2: Point) {
        if scalar_nearly_equal_tolerance(p1.x, p2.x, 1e-4)
            && scalar_nearly_equal_tolerance(p1.y, p2.y, 1e-4)
        {
            return;
        }
        self.add_linear_component(p1, p2);
    }

    fn add_quadratic_component(&mut self, p1: Point, cp: Point, p2: Point) {
        self.components
            .push(PathComponent::Quad(QuadPathComponent { p1, cp, p2 }));
    }

    fn add_conic_component(&mut self, p1: Point, cp: Point, p2: Point, weight: Scalar) {
        if !weight.is_finite() {
            self.add_linear_component(p1, cp);
            self.add_linear_component(cp, p2);
        } else if weight <= 0.0 {
            self.add_linear_component(p1, p2);
        } else if (weight - 1.0).abs() <= Scalar::EPSILON {
            self.add_quadratic_component(p1, cp, p2);
        } else {
            let conic = ConicPathComponent::new(p1, cp, p2, weight);
            for quad in conic.to_quadratic_path_components() {
                self.add_quadratic_component(quad.p1, quad.cp, quad.p2);
            }
        }
    }

    fn add_cubic_component(&mut self, p1: Point, cp1: Point, cp2: Point, p2: Point) {
        self.components
            .push(PathComponent::Cubic(CubicPathComponent {
                p1,
                cp1,
                cp2,
                p2,
            }));
    }

    pub fn move_to(&mut self, p: Point) -> &mut Self {
        self.move_to_relative(p, false)
    }

    pub fn move_to_relative(&mut self, point: Point, relative: bool) -> &mut Self {
        self.current = if relative {
            self.current + point
        } else {
            point
        };
        self.subpath_start = self.current;
        self.add_contour_component(self.current, false);
        self
    }

    pub fn close(&mut self) -> &mut Self {
        if self.subpath_start != self.current {
            self.line_to(self.subpath_start);
        }
        self.set_contour_closed(true);
        self.add_contour_component(self.current, false);
        self
    }

    pub fn line_to(&mut self, p: Point) -> &mut Self {
        self.line_to_relative(p, false)
    }

    pub fn line_to_relative(&mut self, point: Point, relative: bool) -> &mut Self {
        let p = if relative {
            self.current + point
        } else {
            point
        };
        self.add_linear_component(self.current, p);
        self.current = p;
        self
    }

    pub fn horizontal_line_to(&mut self, x: Scalar) -> &mut Self {
        self.horizontal_line_to_relative(x, false)
    }

    pub fn horizontal_line_to_relative(&mut self, x: Scalar, relative: bool) -> &mut Self {
        let endpoint = if relative {
            Point::new(self.current.x + x, self.current.y)
        } else {
            Point::new(x, self.current.y)
        };
        self.add_linear_component(self.current, endpoint);
        self.current = endpoint;
        self
    }

    pub fn vertical_line_to(&mut self, y: Scalar) -> &mut Self {
        self.vertical_line_to_relative(y, false)
    }

    pub fn vertical_line_to_relative(&mut self, y: Scalar, relative: bool) -> &mut Self {
        let endpoint = if relative {
            Point::new(self.current.x, self.current.y + y)
        } else {
            Point::new(self.current.x, y)
        };
        self.add_linear_component(self.current, endpoint);
        self.current = endpoint;
        self
    }

    pub fn quad_to(&mut self, cp: Point, p2: Point) -> &mut Self {
        self.quadratic_curve_to_relative(cp, p2, false)
    }

    pub fn quadratic_curve_to(&mut self, cp: Point, p2: Point) -> &mut Self {
        self.quadratic_curve_to_relative(cp, p2, false)
    }

    pub fn quadratic_curve_to_relative(
        &mut self,
        control_point: Point,
        point: Point,
        relative: bool,
    ) -> &mut Self {
        let cp = if relative {
            self.current + control_point
        } else {
            control_point
        };
        let p2 = if relative {
            self.current + point
        } else {
            point
        };
        self.add_quadratic_component(self.current, cp, p2);
        self.current = p2;
        self
    }

    pub fn conic_curve_to(&mut self, cp: Point, p2: Point, weight: Scalar) -> &mut Self {
        self.conic_curve_to_relative(cp, p2, weight, false)
    }

    pub fn conic_curve_to_relative(
        &mut self,
        control_point: Point,
        point: Point,
        weight: Scalar,
        relative: bool,
    ) -> &mut Self {
        let cp = if relative {
            self.current + control_point
        } else {
            control_point
        };
        let p2 = if relative {
            self.current + point
        } else {
            point
        };
        self.add_conic_component(self.current, cp, p2, weight);
        self.current = p2;
        self
    }

    pub fn cubic_to(&mut self, cp1: Point, cp2: Point, p2: Point) -> &mut Self {
        self.cubic_curve_to_relative(cp1, cp2, p2, false)
    }

    pub fn cubic_curve_to(&mut self, cp1: Point, cp2: Point, p2: Point) -> &mut Self {
        self.cubic_curve_to_relative(cp1, cp2, p2, false)
    }

    pub fn cubic_curve_to_relative(
        &mut self,
        control_point_1: Point,
        control_point_2: Point,
        point: Point,
        relative: bool,
    ) -> &mut Self {
        let cp1 = if relative {
            self.current + control_point_1
        } else {
            control_point_1
        };
        let cp2 = if relative {
            self.current + control_point_2
        } else {
            control_point_2
        };
        let p2 = if relative {
            self.current + point
        } else {
            point
        };
        self.add_cubic_component(self.current, cp1, cp2, p2);
        self.current = p2;
        self
    }

    pub fn add_line(&mut self, p1: Point, p2: Point) -> &mut Self {
        self.move_to(p1);
        self.add_linear_component(p1, p2);
        self.current = p2;
        self
    }

    pub fn add_quadratic_curve(&mut self, p1: Point, cp: Point, p2: Point) -> &mut Self {
        self.move_to(p1);
        self.add_quadratic_component(p1, cp, p2);
        self.current = p2;
        self
    }

    pub fn add_conic_curve(
        &mut self,
        p1: Point,
        cp: Point,
        p2: Point,
        weight: Scalar,
    ) -> &mut Self {
        self.move_to(p1);
        self.add_conic_component(p1, cp, p2, weight);
        self.current = p2;
        self
    }

    pub fn add_cubic_curve(&mut self, p1: Point, cp1: Point, cp2: Point, p2: Point) -> &mut Self {
        self.move_to(p1);
        self.add_cubic_component(p1, cp1, cp2, p2);
        self.current = p2;
        self
    }

    pub fn add_rect(&mut self, rect: Rect) -> &mut Self {
        let origin = rect.origin;
        let size = rect.size;
        let tl = origin;
        let tr = Point::new(origin.x + size.width, origin.y);
        let br = Point::new(origin.x + size.width, origin.y + size.height);
        let bl = Point::new(origin.x, origin.y + size.height);

        self.move_to(tl).line_to(tr).line_to(br).line_to(bl).close()
    }

    pub fn add_circle(&mut self, center: Point, radius: Scalar) -> &mut Self {
        self.add_oval(Rect::from_xywh(
            center.x - radius,
            center.y - radius,
            radius * 2.0,
            radius * 2.0,
        ))
    }

    pub fn add_oval(&mut self, container: Rect) -> &mut Self {
        let c = rect_center(container);
        let r = c - container.origin;
        let m = r * K_ARC_APPROXIMATION_MAGIC;

        self.move_to(Point::new(c.x, c.y - r.y));

        // Top right arc.
        self.add_cubic_component(
            Point::new(c.x, c.y - r.y),
            Point::new(c.x + m.x, c.y - r.y),
            Point::new(c.x + r.x, c.y - m.y),
            Point::new(c.x + r.x, c.y),
        );

        // Bottom right arc.
        self.add_cubic_component(
            Point::new(c.x + r.x, c.y),
            Point::new(c.x + r.x, c.y + m.y),
            Point::new(c.x + m.x, c.y + r.y),
            Point::new(c.x, c.y + r.y),
        );

        // Bottom left arc.
        self.add_cubic_component(
            Point::new(c.x, c.y + r.y),
            Point::new(c.x - m.x, c.y + r.y),
            Point::new(c.x - r.x, c.y + m.y),
            Point::new(c.x - r.x, c.y),
        );

        // Top left arc.
        self.add_cubic_component(
            Point::new(c.x - r.x, c.y),
            Point::new(c.x - r.x, c.y - m.y),
            Point::new(c.x - m.x, c.y - r.y),
            Point::new(c.x, c.y - r.y),
        );

        self.current = Point::new(c.x, c.y - r.y);
        self.close()
    }

    pub fn add_rounded_rect(&mut self, rect: Rect, radii: RoundingRadii) -> &mut Self {
        self.add_round_rect(RoundRect::from_rect_radii(rect, radii))
    }

    pub fn add_round_rect(&mut self, round_rect: RoundRect) -> &mut Self {
        let rect = round_rect.get_bounds();
        let radii = round_rect.get_radii();
        if radii.are_all_corners_empty() {
            return self.add_rect(rect);
        }

        let rect_origin = rect.origin;
        let rect_size = rect.size;

        self.current = Point::new(rect_origin.x + radii.top_left.width, rect_origin.y);
        self.move_to(self.current);

        // Top line.
        self.add_linear_component_if_needed(
            Point::new(rect_origin.x + radii.top_left.width, rect_origin.y),
            Point::new(
                rect_origin.x + rect_size.width - radii.top_right.width,
                rect_origin.y,
            ),
        );

        // Top right arc.
        self.add_rounded_rect_top_right(rect, radii);

        // Right line.
        self.add_linear_component_if_needed(
            Point::new(
                rect_origin.x + rect_size.width,
                rect_origin.y + radii.top_right.height,
            ),
            Point::new(
                rect_origin.x + rect_size.width,
                rect_origin.y + rect_size.height - radii.bottom_right.height,
            ),
        );

        // Bottom right arc.
        self.add_rounded_rect_bottom_right(rect, radii);

        // Bottom line.
        self.add_linear_component_if_needed(
            Point::new(
                rect_origin.x + rect_size.width - radii.bottom_right.width,
                rect_origin.y + rect_size.height,
            ),
            Point::new(
                rect_origin.x + radii.bottom_left.width,
                rect_origin.y + rect_size.height,
            ),
        );

        // Bottom left arc.
        self.add_rounded_rect_bottom_left(rect, radii);

        // Left line.
        self.add_linear_component_if_needed(
            Point::new(
                rect_origin.x,
                rect_origin.y + rect_size.height - radii.bottom_left.height,
            ),
            Point::new(rect_origin.x, rect_origin.y + radii.top_left.height),
        );

        // Top left arc.
        self.add_rounded_rect_top_left(rect, radii);

        self.current = Point::new(rect_origin.x + radii.top_left.width, rect_origin.y);
        self.close()
    }

    fn add_rounded_rect_top_left(&mut self, rect: Rect, radii: RoundingRadii) {
        let magic_tl = radii.top_left.width * K_ARC_APPROXIMATION_MAGIC;
        let magic_tl_h = radii.top_left.height * K_ARC_APPROXIMATION_MAGIC;
        let corner = rect.origin;
        self.add_cubic_component(
            Point::new(corner.x, corner.y + radii.top_left.height),
            Point::new(corner.x, corner.y + radii.top_left.height - magic_tl_h),
            Point::new(corner.x + radii.top_left.width - magic_tl, corner.y),
            Point::new(corner.x + radii.top_left.width, corner.y),
        );
    }

    fn add_rounded_rect_top_right(&mut self, rect: Rect, radii: RoundingRadii) {
        let magic_tr_w = radii.top_right.width * K_ARC_APPROXIMATION_MAGIC;
        let magic_tr_h = radii.top_right.height * K_ARC_APPROXIMATION_MAGIC;
        let corner = Point::new(rect.origin.x + rect.get_width(), rect.origin.y);
        self.add_cubic_component(
            Point::new(corner.x - radii.top_right.width, corner.y),
            Point::new(corner.x - radii.top_right.width + magic_tr_w, corner.y),
            Point::new(corner.x, corner.y + radii.top_right.height - magic_tr_h),
            Point::new(corner.x, corner.y + radii.top_right.height),
        );
    }

    fn add_rounded_rect_bottom_right(&mut self, rect: Rect, radii: RoundingRadii) {
        let magic_br_w = radii.bottom_right.width * K_ARC_APPROXIMATION_MAGIC;
        let magic_br_h = radii.bottom_right.height * K_ARC_APPROXIMATION_MAGIC;
        let corner = Point::new(
            rect.origin.x + rect.size.width,
            rect.origin.y + rect.size.height,
        );
        self.add_cubic_component(
            Point::new(corner.x, corner.y - radii.bottom_right.height),
            Point::new(corner.x, corner.y - radii.bottom_right.height + magic_br_h),
            Point::new(corner.x - radii.bottom_right.width + magic_br_w, corner.y),
            Point::new(corner.x - radii.bottom_right.width, corner.y),
        );
    }

    fn add_rounded_rect_bottom_left(&mut self, rect: Rect, radii: RoundingRadii) {
        let magic_bl_w = radii.bottom_left.width * K_ARC_APPROXIMATION_MAGIC;
        let magic_bl_h = radii.bottom_left.height * K_ARC_APPROXIMATION_MAGIC;
        let corner = Point::new(rect.origin.x, rect.origin.y + rect.get_height());
        self.add_cubic_component(
            Point::new(corner.x + radii.bottom_left.width, corner.y),
            Point::new(corner.x + radii.bottom_left.width - magic_bl_w, corner.y),
            Point::new(corner.x, corner.y - radii.bottom_left.height + magic_bl_h),
            Point::new(corner.x, corner.y - radii.bottom_left.height),
        );
    }

    pub fn add_arc(
        &mut self,
        oval_bounds: Rect,
        start: Radians,
        sweep: Radians,
        use_center: bool,
    ) -> &mut Self {
        let mut start_rad = start.0;
        let mut sweep_rad = sweep.0;
        if sweep_rad < 0.0 {
            start_rad += sweep_rad;
            sweep_rad = -sweep_rad;
        }
        let two_pi = std::f32::consts::TAU;
        let pi_over_2 = std::f32::consts::FRAC_PI_2;
        sweep_rad = sweep_rad.min(two_pi);
        start_rad %= two_pi;

        let center = rect_center(oval_bounds);
        let radius = center - oval_bounds.origin;

        let mut p1_unit = Point::new(start_rad.cos(), start_rad.sin());
        let first_pt = Point::new(
            center.x + p1_unit.x * radius.x,
            center.y + p1_unit.y * radius.y,
        );

        if use_center {
            self.move_to(center);
            self.line_to(first_pt);
        } else {
            self.move_to(first_pt);
        }

        while sweep_rad > 0.0 {
            let (quadrant_angle, p2_unit) = if sweep_rad < pi_over_2 {
                let qa = sweep_rad;
                (
                    qa,
                    Point::new((start_rad + qa).cos(), (start_rad + qa).sin()),
                )
            } else {
                (pi_over_2, Point::new(-p1_unit.y, p1_unit.x))
            };

            let scale = (quadrant_angle / pi_over_2) * K_ARC_APPROXIMATION_MAGIC;
            let arc_cp_lengths = Point::new(radius.x * scale, radius.y * scale);

            let p1 = Point::new(
                center.x + p1_unit.x * radius.x,
                center.y + p1_unit.y * radius.y,
            );
            let p2 = Point::new(
                center.x + p2_unit.x * radius.x,
                center.y + p2_unit.y * radius.y,
            );
            let cp1 = Point::new(
                p1.x - p1_unit.y * arc_cp_lengths.x,
                p1.y + p1_unit.x * arc_cp_lengths.y,
            );
            let cp2 = Point::new(
                p2.x + p2_unit.y * arc_cp_lengths.x,
                p2.y - p2_unit.x * arc_cp_lengths.y,
            );

            self.add_cubic_component(p1, cp1, cp2, p2);
            self.current = p2;

            start_rad += quadrant_angle;
            sweep_rad -= quadrant_angle;
            p1_unit = p2_unit;
        }

        if use_center {
            self.close();
        }

        self
    }

    pub fn add_path(&mut self, path: &Path) -> &mut Self {
        for comp in &path.components {
            if matches!(comp, PathComponent::Contour(_)) {
                self.current_contour_location = self.components.len();
                self.contour_count += 1;
            }
            self.components.push(*comp);
        }
        self
    }

    pub fn shift(&mut self, offset: Point) -> &mut Self {
        for comp in &mut self.components {
            match comp {
                PathComponent::Contour(c) => {
                    c.destination += offset;
                }
                PathComponent::Linear(l) => {
                    l.p1 += offset;
                    l.p2 += offset;
                }
                PathComponent::Quad(q) => {
                    q.p1 += offset;
                    q.cp += offset;
                    q.p2 += offset;
                }
                PathComponent::Cubic(c) => {
                    c.p1 += offset;
                    c.cp1 += offset;
                    c.cp2 += offset;
                    c.p2 += offset;
                }
            }
        }
        self.current += offset;
        self.subpath_start += offset;
        self
    }

    pub fn transform(&mut self, matrix: &Matrix) -> &mut Self {
        for comp in &mut self.components {
            match comp {
                PathComponent::Contour(c) => {
                    c.destination = matrix.transform_point(c.destination);
                }
                PathComponent::Linear(l) => {
                    l.p1 = matrix.transform_point(l.p1);
                    l.p2 = matrix.transform_point(l.p2);
                }
                PathComponent::Quad(q) => {
                    q.p1 = matrix.transform_point(q.p1);
                    q.cp = matrix.transform_point(q.cp);
                    q.p2 = matrix.transform_point(q.p2);
                }
                PathComponent::Cubic(c) => {
                    c.p1 = matrix.transform_point(c.p1);
                    c.cp1 = matrix.transform_point(c.cp1);
                    c.cp2 = matrix.transform_point(c.cp2);
                    c.p2 = matrix.transform_point(c.p2);
                }
            }
        }
        self.current = matrix.transform_point(self.current);
        self.subpath_start = matrix.transform_point(self.subpath_start);
        self
    }

    pub fn copy_path(&self, fill_type: FillType) -> Path {
        self.build(fill_type)
    }

    pub fn take_path(&mut self, fill_type: FillType) -> Path {
        let path = Path {
            components: std::mem::take(&mut self.components),
            fill_type,
            convexity: self.convexity,
        };
        self.current = Point::ZERO;
        self.subpath_start = Point::ZERO;
        self.current_contour_location = 0;
        self.contour_count = 0;
        self.convexity = Convexity::Unknown;
        self.add_contour_component(Point::ZERO, false);
        path
    }

    pub fn build(&self, fill_type: FillType) -> Path {
        Path {
            components: self.components.clone(),
            fill_type,
            convexity: self.convexity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use impeller_geometry::Degrees;

    const EPS: Scalar = 1e-3;

    fn assert_point_near(a: Point, b: Point) {
        assert_abs_diff_eq!(a.x, b.x, epsilon = EPS);
        assert_abs_diff_eq!(a.y, b.y, epsilon = EPS);
    }

    fn assert_rect_near(a: Rect, b: Rect) {
        assert_abs_diff_eq!(a.get_left(), b.get_left(), epsilon = 1e-2);
        assert_abs_diff_eq!(a.get_top(), b.get_top(), epsilon = 1e-2);
        assert_abs_diff_eq!(a.get_right(), b.get_right(), epsilon = 1e-2);
        assert_abs_diff_eq!(a.get_bottom(), b.get_bottom(), epsilon = 1e-2);
    }

    /// C++ source: path_unittests.cc:CubicPathComponentPolylineDoesNotIncludePointOne
    #[test]
    fn test_cubic_path_component_polyline_does_not_include_point_one() {
        let component = CubicPathComponent::new(
            Point::new(10.0, 10.0),
            Point::new(20.0, 35.0),
            Point::new(35.0, 20.0),
            Point::new(40.0, 40.0),
        );
        let mut polyline = Vec::new();
        component.append_polyline_points(1.0, &mut polyline);
        assert_ne!(polyline.first().unwrap().x, 10.0);
        assert_ne!(polyline.first().unwrap().y, 10.0);
        assert_eq!(polyline.last().unwrap().x, 40.0);
        assert_eq!(polyline.last().unwrap().y, 40.0);
    }

    /// C++ source: path_unittests.cc:EmptyPathWithContour
    #[test]
    fn test_empty_path_with_contour() {
        let mut builder = PathBuilder::new();
        let path = builder.take_path(FillType::NonZero);
        assert!(path.is_empty());
    }

    /// C++ source: path_unittests.cc:PathCreatePolyLineDoesNotDuplicatePoints
    #[test]
    fn test_path_create_polyline_does_not_duplicate_points() {
        let mut builder = PathBuilder::new();
        builder
            .move_to(Point::new(10.0, 10.0))
            .line_to(Point::new(20.0, 20.0))
            .line_to(Point::new(30.0, 30.0))
            .move_to(Point::new(40.0, 40.0))
            .line_to(Point::new(50.0, 50.0));

        let polyline = builder.take_path(FillType::NonZero).create_polyline(1.0);
        assert_eq!(polyline.contours.len(), 2);
        assert_eq!(polyline.points.len(), 5);
        assert_eq!(polyline.get_point(0).x, 10.0);
        assert_eq!(polyline.get_point(1).x, 20.0);
        assert_eq!(polyline.get_point(2).x, 30.0);
        assert_eq!(polyline.get_point(3).x, 40.0);
        assert_eq!(polyline.get_point(4).x, 50.0);
    }

    /// C++ source: path_unittests.cc:PathSingleContour
    #[test]
    fn test_path_single_contour() {
        assert!(PathBuilder::new()
            .add_circle(Point::new(100.0, 100.0), 50.0)
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(PathBuilder::new()
            .add_oval(Rect::from_xywh(100.0, 100.0, 100.0, 100.0))
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(PathBuilder::new()
            .add_rect(Rect::from_xywh(100.0, 100.0, 100.0, 100.0))
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(PathBuilder::new()
            .add_round_rect(RoundRect::from_rect_radius(
                Rect::from_xywh(100.0, 100.0, 100.0, 100.0),
                10.0,
            ))
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(PathBuilder::new()
            .add_line(Point::new(100.0, 100.0), Point::new(200.0, 100.0))
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(PathBuilder::new()
            .add_cubic_curve(
                Point::new(100.0, 100.0),
                Point::new(100.0, 50.0),
                Point::new(100.0, 150.0),
                Point::new(200.0, 100.0),
            )
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(PathBuilder::new()
            .add_quadratic_curve(
                Point::new(100.0, 100.0),
                Point::new(100.0, 50.0),
                Point::new(200.0, 100.0),
            )
            .take_path(FillType::NonZero)
            .is_single_contour());
    }

    /// C++ source: path_unittests.cc:PathSingleContourDoubleShapes
    #[test]
    fn test_path_single_contour_double_shapes() {
        assert!(!PathBuilder::new()
            .add_circle(Point::new(100.0, 100.0), 50.0)
            .add_circle(Point::new(100.0, 100.0), 50.0)
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(!PathBuilder::new()
            .add_oval(Rect::from_xywh(100.0, 100.0, 100.0, 100.0))
            .add_oval(Rect::from_xywh(100.0, 100.0, 100.0, 100.0))
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(!PathBuilder::new()
            .add_rect(Rect::from_xywh(100.0, 100.0, 100.0, 100.0))
            .add_rect(Rect::from_xywh(100.0, 100.0, 100.0, 100.0))
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(!PathBuilder::new()
            .add_round_rect(RoundRect::from_rect_radius(
                Rect::from_xywh(100.0, 100.0, 100.0, 100.0),
                10.0,
            ))
            .add_round_rect(RoundRect::from_rect_radius(
                Rect::from_xywh(100.0, 100.0, 100.0, 100.0),
                10.0,
            ))
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(!PathBuilder::new()
            .add_line(Point::new(100.0, 100.0), Point::new(200.0, 100.0))
            .add_line(Point::new(100.0, 100.0), Point::new(200.0, 100.0))
            .take_path(FillType::NonZero)
            .is_single_contour());
        assert!(!PathBuilder::new()
            .add_quadratic_curve(
                Point::new(100.0, 100.0),
                Point::new(100.0, 50.0),
                Point::new(200.0, 100.0),
            )
            .close()
            .add_quadratic_curve(
                Point::new(100.0, 100.0),
                Point::new(100.0, 50.0),
                Point::new(200.0, 100.0),
            )
            .take_path(FillType::NonZero)
            .is_single_contour());
    }

    /// C++ source: path_unittests.cc:PathBuilderSetsCorrectContourPropertiesForAddCommands
    #[test]
    fn test_path_builder_sets_correct_contour_properties_for_add_commands() {
        let p_circle = PathBuilder::new()
            .add_circle(Point::new(100.0, 100.0), 50.0)
            .take_path(FillType::NonZero);
        assert_eq!(
            p_circle.components[0],
            PathComponent::Contour(ContourComponent::new(Point::new(100.0, 50.0), true))
        );

        let p_oval = PathBuilder::new()
            .add_oval(Rect::from_xywh(100.0, 100.0, 100.0, 100.0))
            .take_path(FillType::NonZero);
        assert_eq!(
            p_oval.components[0],
            PathComponent::Contour(ContourComponent::new(Point::new(150.0, 100.0), true))
        );

        let p_rect = PathBuilder::new()
            .add_rect(Rect::from_xywh(100.0, 100.0, 100.0, 100.0))
            .take_path(FillType::NonZero);
        assert_eq!(
            p_rect.components[0],
            PathComponent::Contour(ContourComponent::new(Point::new(100.0, 100.0), true))
        );

        let p_rrect = PathBuilder::new()
            .add_round_rect(RoundRect::from_rect_radius(
                Rect::from_xywh(100.0, 100.0, 100.0, 100.0),
                10.0,
            ))
            .take_path(FillType::NonZero);
        assert_eq!(
            p_rrect.components[0],
            PathComponent::Contour(ContourComponent::new(Point::new(110.0, 100.0), true))
        );

        let p_line = PathBuilder::new()
            .add_line(Point::new(100.0, 100.0), Point::new(200.0, 100.0))
            .take_path(FillType::NonZero);
        assert_eq!(
            p_line.components[0],
            PathComponent::Contour(ContourComponent::new(Point::new(100.0, 100.0), false))
        );
    }

    /// C++ source: path_unittests.cc:PathCreatePolylineGeneratesCorrectContourData
    #[test]
    fn test_path_create_polyline_generates_correct_contour_data() {
        let polyline = PathBuilder::new()
            .add_line(Point::new(100.0, 100.0), Point::new(200.0, 100.0))
            .move_to(Point::new(100.0, 200.0))
            .line_to(Point::new(150.0, 250.0))
            .line_to(Point::new(200.0, 200.0))
            .close()
            .take_path(FillType::NonZero)
            .create_polyline(1.0);
        assert_eq!(polyline.points.len(), 6);
        assert_eq!(polyline.contours.len(), 2);
        assert!(!polyline.contours[0].is_closed);
        assert_eq!(polyline.contours[0].start_index, 0);
        assert!(polyline.contours[1].is_closed);
        assert_eq!(polyline.contours[1].start_index, 2);
    }

    /// C++ source: path_unittests.cc:PolylineGetContourPointBoundsReturnsCorrectRanges
    #[test]
    fn test_polyline_get_contour_point_bounds_returns_correct_ranges() {
        let polyline = PathBuilder::new()
            .add_line(Point::new(100.0, 100.0), Point::new(200.0, 100.0))
            .move_to(Point::new(100.0, 200.0))
            .line_to(Point::new(150.0, 250.0))
            .line_to(Point::new(200.0, 200.0))
            .close()
            .take_path(FillType::NonZero)
            .create_polyline(1.0);
        assert_eq!(polyline.get_contour_point_bounds(0), (0, 2));
        assert_eq!(polyline.get_contour_point_bounds(1), (2, 6));
    }

    /// C++ source: path_unittests.cc:PathAddRectPolylineHasCorrectContourData
    #[test]
    fn test_path_add_rect_polyline_has_correct_contour_data() {
        let polyline = PathBuilder::new()
            .add_rect(Rect::from_ltrb(50.0, 60.0, 70.0, 80.0))
            .take_path(FillType::NonZero)
            .create_polyline(1.0);
        assert_eq!(polyline.contours.len(), 1);
        assert!(polyline.contours[0].is_closed);
        assert_eq!(polyline.contours[0].start_index, 0);
        assert_eq!(polyline.points.len(), 5);
        assert_eq!(polyline.get_point(0), Point::new(50.0, 60.0));
        assert_eq!(polyline.get_point(1), Point::new(70.0, 60.0));
        assert_eq!(polyline.get_point(2), Point::new(70.0, 80.0));
        assert_eq!(polyline.get_point(3), Point::new(50.0, 80.0));
        assert_eq!(polyline.get_point(4), Point::new(50.0, 60.0));
    }

    /// C++ source: path_unittests.cc:PathPolylineDuplicatesAreRemovedForSameContour
    #[test]
    fn test_path_polyline_duplicates_are_removed_for_same_contour() {
        let polyline = PathBuilder::new()
            .move_to(Point::new(50.0, 50.0))
            .line_to(Point::new(50.0, 50.0))
            .line_to(Point::new(100.0, 50.0))
            .line_to(Point::new(100.0, 50.0))
            .line_to(Point::new(100.0, 100.0))
            .close()
            .line_to(Point::new(0.0, 50.0))
            .line_to(Point::new(0.0, 100.0))
            .line_to(Point::new(0.0, 100.0))
            .take_path(FillType::NonZero)
            .create_polyline(1.0);
        assert_eq!(polyline.contours.len(), 2);
        assert_eq!(polyline.contours[0].start_index, 0);
        assert!(polyline.contours[0].is_closed);
        assert_eq!(polyline.contours[1].start_index, 4);
        assert!(!polyline.contours[1].is_closed);
        assert_eq!(polyline.points.len(), 7);
        assert_eq!(polyline.get_point(0), Point::new(50.0, 50.0));
        assert_eq!(polyline.get_point(1), Point::new(100.0, 50.0));
        assert_eq!(polyline.get_point(2), Point::new(100.0, 100.0));
        assert_eq!(polyline.get_point(3), Point::new(50.0, 50.0));
        assert_eq!(polyline.get_point(4), Point::new(50.0, 50.0));
        assert_eq!(polyline.get_point(5), Point::new(0.0, 50.0));
        assert_eq!(polyline.get_point(6), Point::new(0.0, 100.0));
    }

    /// C++ source: path_unittests.cc:PathShifting
    #[test]
    fn test_path_shifting() {
        let mut builder = PathBuilder::new();
        let path = builder
            .add_line(Point::new(0.0, 0.0), Point::new(10.0, 10.0))
            .add_quadratic_curve(
                Point::new(10.0, 10.0),
                Point::new(15.0, 15.0),
                Point::new(20.0, 20.0),
            )
            .add_cubic_curve(
                Point::new(20.0, 20.0),
                Point::new(25.0, 25.0),
                Point::new(-5.0, -5.0),
                Point::new(30.0, 30.0),
            )
            .close()
            .shift(Point::new(1.0, 1.0))
            .take_path(FillType::NonZero);

        assert_eq!(
            path.components[0],
            PathComponent::Contour(ContourComponent::new(Point::new(1.0, 1.0), false))
        );
        assert_eq!(
            path.components[1],
            PathComponent::Linear(LinearPathComponent::new(
                Point::new(1.0, 1.0),
                Point::new(11.0, 11.0),
            ))
        );
        assert_eq!(
            path.components[3],
            PathComponent::Quad(QuadPathComponent::new(
                Point::new(11.0, 11.0),
                Point::new(16.0, 16.0),
                Point::new(21.0, 21.0),
            ))
        );
        assert_eq!(
            path.components[5],
            PathComponent::Cubic(CubicPathComponent::new(
                Point::new(21.0, 21.0),
                Point::new(26.0, 26.0),
                Point::new(-4.0, -4.0),
                Point::new(31.0, 31.0),
            ))
        );
    }

    /// C++ source: path_unittests.cc:PathBuilderWillComputeBounds
    #[test]
    fn test_path_builder_will_compute_bounds() {
        let mut builder = PathBuilder::new();
        let path_1 = builder
            .add_line(Point::new(0.0, 0.0), Point::new(1.0, 1.0))
            .take_path(FillType::NonZero);
        assert_eq!(
            path_1.get_bounding_box(),
            Some(Rect::from_ltrb(0.0, 0.0, 1.0, 1.0))
        );

        let path_2 = builder
            .add_line(Point::new(-1.0, -1.0), Point::new(1.0, 1.0))
            .take_path(FillType::NonZero);
        assert_eq!(
            path_2.get_bounding_box(),
            Some(Rect::from_ltrb(-1.0, -1.0, 1.0, 1.0))
        );
    }

    /// C++ source: path_unittests.cc:PathHorizontalLine
    #[test]
    fn test_path_horizontal_line() {
        let mut builder = PathBuilder::new();
        let path = builder.horizontal_line_to(10.0).take_path(FillType::NonZero);
        assert_eq!(
            path.components[1],
            PathComponent::Linear(LinearPathComponent::new(
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
            ))
        );
    }

    /// C++ source: path_unittests.cc:PathVerticalLine
    #[test]
    fn test_path_vertical_line() {
        let mut builder = PathBuilder::new();
        let path = builder.vertical_line_to(10.0).take_path(FillType::NonZero);
        assert_eq!(
            path.components[1],
            PathComponent::Linear(LinearPathComponent::new(
                Point::new(0.0, 0.0),
                Point::new(0.0, 10.0),
            ))
        );
    }

    /// C++ source: path_unittests.cc:QuadradicPath
    #[test]
    fn test_quadratic_path() {
        let mut builder = PathBuilder::new();
        let path = builder
            .quadratic_curve_to(Point::new(10.0, 10.0), Point::new(20.0, 20.0))
            .take_path(FillType::NonZero);
        assert_eq!(
            path.components[1],
            PathComponent::Quad(QuadPathComponent::new(
                Point::new(0.0, 0.0),
                Point::new(10.0, 10.0),
                Point::new(20.0, 20.0),
            ))
        );
    }

    /// C++ source: path_unittests.cc:CubicPath
    #[test]
    fn test_cubic_path() {
        let mut builder = PathBuilder::new();
        let path = builder
            .cubic_curve_to(
                Point::new(10.0, 10.0),
                Point::new(-10.0, -10.0),
                Point::new(20.0, 20.0),
            )
            .take_path(FillType::NonZero);
        assert_eq!(
            path.components[1],
            PathComponent::Cubic(CubicPathComponent::new(
                Point::new(0.0, 0.0),
                Point::new(10.0, 10.0),
                Point::new(-10.0, -10.0),
                Point::new(20.0, 20.0),
            ))
        );
    }

    /// C++ source: path_unittests.cc:BoundingBoxCubic
    #[test]
    fn test_bounding_box_cubic() {
        let mut builder = PathBuilder::new();
        let path = builder
            .add_cubic_curve(
                Point::new(120.0, 160.0),
                Point::new(25.0, 200.0),
                Point::new(220.0, 260.0),
                Point::new(220.0, 40.0),
            )
            .take_path(FillType::NonZero);
        let box_opt = path.get_bounding_box();
        let expected = Rect::from_xywh(93.9101, 40.0, 126.09, 158.862);
        assert!(box_opt.is_some());
        assert_rect_near(box_opt.unwrap(), expected);
    }

    /// C++ source: path_unittests.cc:BoundingBoxOfCompositePathIsCorrect
    #[test]
    fn test_bounding_box_of_composite_path_is_correct() {
        let mut builder = PathBuilder::new();
        builder.add_round_rect(RoundRect::from_rect_radius(
            Rect::from_xywh(10.0, 10.0, 300.0, 300.0),
            50.0,
        ));
        let path = builder.take_path(FillType::NonZero);
        let actual = path.get_bounding_box();
        let expected = Rect::from_xywh(10.0, 10.0, 300.0, 300.0);
        assert!(actual.is_some());
        assert_rect_near(actual.unwrap(), expected);
    }

    /// C++ source: path_unittests.cc:ExtremaOfCubicPathComponentIsCorrect
    #[test]
    fn test_extrema_of_cubic_path_component_is_correct() {
        let cubic = CubicPathComponent::new(
            Point::new(11.769_268, 252.883_15),
            Point::new(-6.285_793_3, 204.356_46),
            Point::new(-4.539_972_3, 156.552_9),
            Point::new(17.006_73, 109.472_49),
        );
        let points = cubic.extrema();
        assert_eq!(points.len(), 3);
        assert_point_near(points[2], cubic.solve(0.455_916));
    }

    /// C++ source: path_unittests.cc:PathGetBoundingBoxForCubicWithNoDerivativeRootsIsCorrect
    #[test]
    fn test_path_get_bounding_box_for_cubic_with_no_derivative_roots_is_correct() {
        let mut builder = PathBuilder::new();
        builder.add_cubic_curve(
            Point::new(0.0, 1.0),
            Point::new(2.0, 3.0),
            Point::new(4.0, 5.0),
            Point::new(6.0, 7.0),
        );
        let path = builder.take_path(FillType::NonZero);
        let actual = path.get_bounding_box();
        let expected = Rect::from_ltrb(0.0, 1.0, 6.0, 7.0);
        assert!(actual.is_some());
        assert_rect_near(actual.unwrap(), expected);
    }

    /// C++ source: path_unittests.cc:EmptyPath
    #[test]
    fn test_empty_path() {
        let path = PathBuilder::new().take_path(FillType::NonZero);
        assert_eq!(path.get_component_count(), 1);
        assert_eq!(
            path.components[0],
            PathComponent::Contour(ContourComponent::new(Point::ZERO, false))
        );
        let polyline = path.create_polyline(1.0);
        assert!(polyline.points.is_empty());
        assert!(polyline.contours.is_empty());
    }

    /// C++ source: path_unittests.cc:SimplePath
    #[test]
    fn test_simple_path() {
        let mut builder = PathBuilder::new();
        let path = builder
            .add_line(Point::new(0.0, 0.0), Point::new(100.0, 100.0))
            .add_quadratic_curve(
                Point::new(100.0, 100.0),
                Point::new(200.0, 200.0),
                Point::new(300.0, 300.0),
            )
            .add_cubic_curve(
                Point::new(300.0, 300.0),
                Point::new(400.0, 400.0),
                Point::new(500.0, 500.0),
                Point::new(600.0, 600.0),
            )
            .take_path(FillType::NonZero);

        assert_eq!(path.get_component_count(), 6);
    }

    /// C++ source: path_unittests.cc:RepeatCloseDoesNotAddNewLines
    #[test]
    fn test_repeat_close_does_not_add_new_lines() {
        let mut builder = PathBuilder::new();
        let path = builder
            .line_to(Point::new(0.0, 10.0))
            .line_to(Point::new(10.0, 10.0))
            .close()
            .close()
            .close()
            .take_path(FillType::NonZero);

        assert_eq!(path.get_component_count(), 5);
        let linear_count = path
            .components
            .iter()
            .filter(|c| matches!(c, PathComponent::Linear(_)))
            .count();
        let contour_count = path
            .components
            .iter()
            .filter(|c| matches!(c, PathComponent::Contour(_)))
            .count();
        assert_eq!(linear_count, 3);
        assert_eq!(contour_count, 2);
    }

    /// C++ source: path_unittests.cc:CloseAfterMoveDoesNotAddNewLines
    #[test]
    fn test_close_after_move_does_not_add_new_lines() {
        let mut builder = PathBuilder::new();
        let path = builder
            .line_to(Point::new(0.0, 10.0))
            .line_to(Point::new(10.0, 10.0))
            .move_to(Point::new(30.0, 30.0))
            .close()
            .close()
            .take_path(FillType::NonZero);

        assert_eq!(path.get_component_count(), 4);
        let linear_count = path
            .components
            .iter()
            .filter(|c| matches!(c, PathComponent::Linear(_)))
            .count();
        let contour_count = path
            .components
            .iter()
            .filter(|c| matches!(c, PathComponent::Contour(_)))
            .count();
        assert_eq!(linear_count, 2);
        assert_eq!(contour_count, 2);
    }

    /// C++ source: path_unittests.cc:CloseAtOriginDoesNotAddNewLineSegment
    #[test]
    fn test_close_at_origin_does_not_add_new_line_segment() {
        let mut builder = PathBuilder::new();
        let path = builder
            .line_to(Point::new(10.0, 0.0))
            .line_to(Point::new(10.0, 10.0))
            .line_to(Point::new(0.0, 10.0))
            .line_to(Point::new(0.0, 0.0))
            .close()
            .take_path(FillType::NonZero);

        assert_eq!(path.get_component_count(), 6);
        let linear_count = path
            .components
            .iter()
            .filter(|c| matches!(c, PathComponent::Linear(_)))
            .count();
        let contour_count = path
            .components
            .iter()
            .filter(|c| matches!(c, PathComponent::Contour(_)))
            .count();
        assert_eq!(linear_count, 4);
        assert_eq!(contour_count, 2);
    }

    /// C++ source: path_unittests.cc:CanBeCloned
    #[test]
    fn test_can_be_cloned() {
        let mut builder = PathBuilder::new();
        builder
            .move_to(Point::new(10.0, 10.0))
            .line_to(Point::new(20.0, 20.0))
            .set_convexity(Convexity::Convex);

        let path_a = builder.take_path(FillType::Odd);
        let path_b = path_a.clone();

        assert_eq!(path_a.get_bounding_box(), path_b.get_bounding_box());
        assert_eq!(path_a.get_fill_type(), path_b.get_fill_type());
        assert_eq!(path_a.is_convex(), path_b.is_convex());
        assert!(path_a.is_convex());

        let poly_a = path_a.create_polyline(1.0);
        let poly_b = path_b.create_polyline(1.0);
        assert_eq!(poly_a, poly_b);
    }

    /// C++ source: path_unittests.cc:FanTessellation
    #[test]
    fn test_fan_tessellation() {
        let path = PathBuilder::new()
            .add_round_rect(RoundRect::from_rect_radius(
                Rect::from_ltrb(0.0, 0.0, 100.0, 100.0),
                10.0,
            ))
            .take_path(FillType::NonZero);
        let (points, contours) = path.count_storage(1.0);

        let mut point_storage = vec![Point::ZERO; points];
        let mut index_storage = vec![0u16; points + (contours - 1)];

        let mut writer = FanVertexWriter::new(&mut point_storage, &mut index_storage);
        path.write_polyline_to(1.0, &mut writer);

        assert!(writer.get_index_count() <= index_storage.len());
        assert_eq!(point_storage[0], Point::new(10.0, 0.0));
    }

    /// C++ source: path_unittests.cc:FanTessellationUnclosedPath
    #[test]
    fn test_fan_tessellation_unclosed_path() {
        let path = PathBuilder::new()
            .line_to(Point::new(100.0, 0.0))
            .line_to(Point::new(100.0, 100.0))
            .line_to(Point::new(0.0, 100.0))
            .take_path(FillType::NonZero);

        let expected = vec![
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 100.0),
            Point::new(0.0, 100.0),
            Point::new(0.0, 0.0),
            Point::new(0.0, 0.0),
        ];
        let expected_indices: Vec<u16> = vec![0, 1, 2, 3, 0xFFFF, 0];

        let (points, contours) = path.count_storage(1.0);
        let mut point_storage = vec![Point::ZERO; points];
        let mut index_storage = vec![0u16; points + (contours - 1)];

        let mut writer = FanVertexWriter::new(&mut point_storage, &mut index_storage);
        path.write_polyline_to(1.0, &mut writer);

        assert!(index_storage <= expected_indices);
        assert_eq!(point_storage, expected);
    }

    /// C++ source: path_unittests.cc:StripTessellationUnclosedPath
    #[test]
    fn test_strip_tessellation_unclosed_path() {
        let path = PathBuilder::new()
            .line_to(Point::new(100.0, 0.0))
            .line_to(Point::new(100.0, 100.0))
            .line_to(Point::new(0.0, 100.0))
            .take_path(FillType::NonZero);

        let expected = vec![
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 100.0),
            Point::new(0.0, 100.0),
            Point::new(0.0, 0.0),
            Point::new(0.0, 0.0),
        ];
        let expected_indices: Vec<u16> = vec![0, 1, 3, 2, 0xFFFF, 0];

        let (points, contours) = path.count_storage(1.0);
        let mut point_storage = vec![Point::ZERO; points];
        let mut index_storage = vec![0u16; points + (contours - 1)];

        let mut writer = StripVertexWriter::new(&mut point_storage, &mut index_storage);
        path.write_polyline_to(1.0, &mut writer);

        assert!(index_storage <= expected_indices);
        assert_eq!(point_storage, expected);
    }

    /// C++ source: path_unittests.cc:FanTessellationMultiContour
    #[test]
    fn test_fan_tessellation_multi_contour() {
        let mut builder = PathBuilder::new();
        for i in 0..10 {
            builder.add_round_rect(RoundRect::from_rect_radius(
                Rect::from_ltrb(i as Scalar, i as Scalar, 100.0, 100.0),
                10.0,
            ));
        }
        let path = builder.take_path(FillType::NonZero);
        let (points, contours) = path.count_storage(1.0);

        let mut point_storage = vec![Point::ZERO; points];
        let mut index_storage = vec![0u16; points + (contours - 1)];

        let mut writer = FanVertexWriter::new(&mut point_storage, &mut index_storage);
        path.write_polyline_to(1.0, &mut writer);

        assert!(writer.get_index_count() <= index_storage.len());
        assert_eq!(point_storage[0], Point::new(10.0, 0.0));
    }

    /// C++ source: path_unittests.cc:StripTessellation
    #[test]
    fn test_strip_tessellation() {
        let path = PathBuilder::new()
            .add_round_rect(RoundRect::from_rect_radius(
                Rect::from_ltrb(0.0, 0.0, 100.0, 100.0),
                10.0,
            ))
            .take_path(FillType::NonZero);
        let (points, contours) = path.count_storage(1.0);

        let mut point_storage = vec![Point::ZERO; points];
        let mut index_storage = vec![0u16; points + (contours - 1)];

        let mut writer = StripVertexWriter::new(&mut point_storage, &mut index_storage);
        path.write_polyline_to(1.0, &mut writer);

        assert!(writer.get_index_count() <= index_storage.len());
        assert_eq!(point_storage[0], Point::new(10.0, 0.0));
    }

    /// C++ source: path_unittests.cc:StripTessellationMultiContour
    #[test]
    fn test_strip_tessellation_multi_contour() {
        let mut builder = PathBuilder::new();
        for i in 0..10 {
            builder.add_round_rect(RoundRect::from_rect_radius(
                Rect::from_ltrb(i as Scalar, i as Scalar, 100.0, 100.0),
                10.0,
            ));
        }
        let path = builder.take_path(FillType::NonZero);
        let (points, contours) = path.count_storage(1.0);

        let mut point_storage = vec![Point::ZERO; points];
        let mut index_storage = vec![0u16; points + (contours - 1)];

        let mut writer = StripVertexWriter::new(&mut point_storage, &mut index_storage);
        path.write_polyline_to(1.0, &mut writer);

        assert!(writer.get_index_count() <= index_storage.len());
        assert_eq!(point_storage[0], Point::new(10.0, 0.0));
    }

    /// C++ source: path_unittests.cc:PathBuilderAddPathBasher
    #[test]
    fn test_path_builder_add_path_basher() {
        let mut test_path_builder = PathBuilder::new();
        test_path_builder.add_oval(Rect::from_ltrb(10.0, 10.0, 50.0, 50.0));
        let test_path = test_path_builder.take_path(FillType::NonZero);

        for _ in 0..25 {
            let mut path_builder = PathBuilder::new();
            for _ in 0..4 {
                path_builder.add_circle(Point::new(50.0, 50.0), 25.0);
                path_builder.add_oval(Rect::from_ltrb(100.0, 100.0, 200.0, 200.0));
                path_builder.add_path(&test_path);
                path_builder.add_rect(Rect::from_ltrb(50.0, 50.0, 75.0, 57.0));
                path_builder.add_line(Point::new(80.0, 70.0), Point::new(110.0, 95.0));
                path_builder.add_arc(
                    Rect::from_ltrb(50.0, 50.0, 100.0, 100.0),
                    Radians::from(Degrees(20.0)),
                    Radians::from(Degrees(100.0)),
                    false,
                );
                path_builder.add_round_rect(RoundRect::from_rect_size(
                    Rect::from_ltrb(70.0, 70.0, 130.0, 130.0),
                    Size::new(10.0, 10.0),
                ));
            }
            let combined = path_builder.take_path(FillType::NonZero);
            let bounds = combined.get_bounding_box();
            assert!(bounds.is_some());
            assert_rect_near(bounds.unwrap(), Rect::from_ltrb(10.0, 10.0, 200.0, 200.0));
        }
    }

    /// C++ source: path_unittests.cc:PathBuilderDoesNotMutateCopiedPaths
    #[test]
    fn test_path_builder_does_not_mutate_copied_paths() {
        let mut builder = PathBuilder::new();
        builder
            .move_to(Point::new(10.0, 10.0))
            .line_to(Point::new(20.0, 20.0))
            .line_to(Point::new(20.0, 10.0));

        let path1 = builder.copy_path(FillType::NonZero);
        assert_eq!(path1.get_component_count(), 3);

        builder.close();
        builder.shift(Point::new(23.0, 42.0));

        assert_eq!(path1.get_component_count(), 3);
        assert_eq!(
            path1.components[0],
            PathComponent::Contour(ContourComponent::new(Point::new(10.0, 10.0), false))
        );
    }

    /// C++ source: round_rect_unittests.cc:EmptyDeclaration
    #[test]
    fn test_round_rect_empty_declaration() {
        let round_rect = RoundRect::new();
        assert!(round_rect.is_empty());
        assert!(!round_rect.is_rect());
        assert!(!round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert!(round_rect.get_bounds().is_empty());
        assert_eq!(round_rect.get_bounds(), Rect::ZERO);
        assert_eq!(round_rect.get_radii().top_left, Size::ZERO);
        assert_eq!(round_rect.get_radii().top_right, Size::ZERO);
        assert_eq!(round_rect.get_radii().bottom_left, Size::ZERO);
        assert_eq!(round_rect.get_radii().bottom_right, Size::ZERO);
    }

    /// C++ source: round_rect_unittests.cc:DefaultConstructor
    #[test]
    fn test_round_rect_default_constructor() {
        let round_rect = RoundRect::default();
        assert!(round_rect.is_empty());
        assert!(!round_rect.is_rect());
        assert!(!round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(round_rect.get_bounds(), Rect::ZERO);
    }

    /// C++ source: round_rect_unittests.cc:EmptyRectConstruction
    #[test]
    fn test_round_rect_empty_rect_construction() {
        let round_rect = RoundRect::from_rect(Rect::from_ltrb(20.0, 20.0, 20.0, 20.0));
        assert!(round_rect.is_empty());
        assert!(!round_rect.is_rect());
        assert!(!round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(
            round_rect.get_bounds(),
            Rect::from_ltrb(20.0, 20.0, 20.0, 20.0)
        );
        assert_eq!(round_rect.get_radii(), RoundingRadii::ZERO);
    }

    /// C++ source: round_rect_unittests.cc:RectConstructor
    #[test]
    fn test_round_rect_rect_constructor() {
        let round_rect = RoundRect::from_rect(Rect::from_ltrb(10.0, 10.0, 20.0, 20.0));
        assert!(!round_rect.is_empty());
        assert!(round_rect.is_rect());
        assert!(!round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(
            round_rect.get_bounds(),
            Rect::from_ltrb(10.0, 10.0, 20.0, 20.0)
        );
        assert_eq!(round_rect.get_radii(), RoundingRadii::ZERO);
    }

    /// C++ source: round_rect_unittests.cc:InvertedRectConstruction
    #[test]
    fn test_round_rect_inverted_rect_construction() {
        let round_rect = RoundRect::from_rect(Rect::from_ltrb(20.0, 20.0, 10.0, 10.0));
        assert!(!round_rect.is_empty());
        assert!(round_rect.is_rect());
        assert!(!round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(
            round_rect.get_bounds(),
            Rect::from_ltrb(10.0, 10.0, 20.0, 20.0)
        );
        assert_eq!(round_rect.get_radii(), RoundingRadii::ZERO);
    }

    /// C++ source: round_rect_unittests.cc:EmptyOvalConstruction
    #[test]
    fn test_round_rect_empty_oval_construction() {
        let round_rect =
            RoundRect::from_rect_xy(Rect::from_ltrb(20.0, 20.0, 20.0, 20.0), 10.0, 10.0);
        assert!(round_rect.is_empty());
        assert!(!round_rect.is_rect());
        assert!(!round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(
            round_rect.get_bounds(),
            Rect::from_ltrb(20.0, 20.0, 20.0, 20.0)
        );
        assert_eq!(round_rect.get_radii(), RoundingRadii::ZERO);
    }

    /// C++ source: round_rect_unittests.cc:OvalConstructor
    #[test]
    fn test_round_rect_oval_constructor() {
        let round_rect = RoundRect::from_oval(Rect::from_ltrb(10.0, 10.0, 20.0, 20.0));
        assert!(!round_rect.is_empty());
        assert!(!round_rect.is_rect());
        assert!(round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(
            round_rect.get_bounds(),
            Rect::from_ltrb(10.0, 10.0, 20.0, 20.0)
        );
        assert_eq!(round_rect.get_radii().top_left, Size::new(5.0, 5.0));
        assert_eq!(round_rect.get_radii().top_right, Size::new(5.0, 5.0));
        assert_eq!(round_rect.get_radii().bottom_left, Size::new(5.0, 5.0));
        assert_eq!(round_rect.get_radii().bottom_right, Size::new(5.0, 5.0));
    }

    /// C++ source: round_rect_unittests.cc:InvertedOvalConstruction
    #[test]
    fn test_round_rect_inverted_oval_construction() {
        let round_rect =
            RoundRect::from_rect_xy(Rect::from_ltrb(20.0, 20.0, 10.0, 10.0), 10.0, 10.0);
        assert!(!round_rect.is_empty());
        assert!(!round_rect.is_rect());
        assert!(round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(
            round_rect.get_bounds(),
            Rect::from_ltrb(10.0, 10.0, 20.0, 20.0)
        );
        assert_eq!(round_rect.get_radii().top_left, Size::new(5.0, 5.0));
        assert_eq!(round_rect.get_radii().top_right, Size::new(5.0, 5.0));
        assert_eq!(round_rect.get_radii().bottom_left, Size::new(5.0, 5.0));
        assert_eq!(round_rect.get_radii().bottom_right, Size::new(5.0, 5.0));
    }

    /// C++ source: round_rect_unittests.cc:RectRadiusConstructor
    #[test]
    fn test_round_rect_rect_radius_constructor() {
        let round_rect = RoundRect::from_rect_radius(Rect::from_ltrb(10.0, 10.0, 20.0, 20.0), 2.0);
        assert!(!round_rect.is_empty());
        assert!(!round_rect.is_rect());
        assert!(!round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(round_rect.get_radii().top_left, Size::new(2.0, 2.0));
        assert_eq!(round_rect.get_radii().top_right, Size::new(2.0, 2.0));
        assert_eq!(round_rect.get_radii().bottom_left, Size::new(2.0, 2.0));
        assert_eq!(round_rect.get_radii().bottom_right, Size::new(2.0, 2.0));
    }

    /// C++ source: round_rect_unittests.cc:RectXYConstructor
    #[test]
    fn test_round_rect_rect_xy_constructor() {
        let round_rect =
            RoundRect::from_rect_xy(Rect::from_ltrb(10.0, 10.0, 20.0, 20.0), 2.0, 3.0);
        assert!(!round_rect.is_empty());
        assert!(!round_rect.is_rect());
        assert!(!round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(round_rect.get_radii().top_left, Size::new(2.0, 3.0));
        assert_eq!(round_rect.get_radii().top_right, Size::new(2.0, 3.0));
        assert_eq!(round_rect.get_radii().bottom_left, Size::new(2.0, 3.0));
        assert_eq!(round_rect.get_radii().bottom_right, Size::new(2.0, 3.0));
    }

    /// C++ source: round_rect_unittests.cc:RectSizeConstructor
    #[test]
    fn test_round_rect_rect_size_constructor() {
        let round_rect = RoundRect::from_rect_size(
            Rect::from_ltrb(10.0, 10.0, 20.0, 20.0),
            Size::new(2.0, 3.0),
        );
        assert!(!round_rect.is_empty());
        assert!(!round_rect.is_rect());
        assert!(!round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(round_rect.get_radii().top_left, Size::new(2.0, 3.0));
    }

    /// C++ source: round_rect_unittests.cc:RectRadiiConstructor
    #[test]
    fn test_round_rect_rect_radii_constructor() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_ltrb(10.0, 10.0, 20.0, 20.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 1.5),
                Size::new(2.0, 2.5),
                Size::new(3.0, 3.5),
                Size::new(4.0, 4.5),
            ),
        );
        assert!(!round_rect.is_empty());
        assert!(!round_rect.is_rect());
        assert!(!round_rect.is_oval());
        assert!(round_rect.is_finite());
        assert_eq!(round_rect.get_radii().top_left, Size::new(1.0, 1.5));
        assert_eq!(round_rect.get_radii().top_right, Size::new(2.0, 2.5));
        assert_eq!(round_rect.get_radii().bottom_left, Size::new(3.0, 3.5));
        assert_eq!(round_rect.get_radii().bottom_right, Size::new(4.0, 4.5));
    }

    /// C++ source: round_rect_unittests.cc:RectRadiiOverflowWidthConstructor
    #[test]
    fn test_round_rect_rect_radii_overflow_width_constructor() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_xywh(10.0, 10.0, 6.0, 30.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 2.0),
                Size::new(3.0, 4.0),
                Size::new(5.0, 6.0),
                Size::new(7.0, 8.0),
            ),
        );
        assert_eq!(
            round_rect.get_bounds(),
            Rect::from_ltrb(10.0, 10.0, 16.0, 40.0)
        );
        assert_eq!(round_rect.get_radii().top_left, Size::new(0.5, 1.0));
        assert_eq!(round_rect.get_radii().top_right, Size::new(1.5, 2.0));
        assert_eq!(round_rect.get_radii().bottom_left, Size::new(2.5, 3.0));
        assert_eq!(round_rect.get_radii().bottom_right, Size::new(3.5, 4.0));
    }

    /// C++ source: round_rect_unittests.cc:RectRadiiOverflowHeightConstructor
    #[test]
    fn test_round_rect_rect_radii_overflow_height_constructor() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_xywh(10.0, 10.0, 30.0, 6.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 2.0),
                Size::new(3.0, 4.0),
                Size::new(5.0, 6.0),
                Size::new(7.0, 8.0),
            ),
        );
        assert_eq!(
            round_rect.get_bounds(),
            Rect::from_ltrb(10.0, 10.0, 40.0, 16.0)
        );
        assert_eq!(round_rect.get_radii().top_left, Size::new(0.5, 1.0));
        assert_eq!(round_rect.get_radii().top_right, Size::new(1.5, 2.0));
        assert_eq!(round_rect.get_radii().bottom_left, Size::new(2.5, 3.0));
        assert_eq!(round_rect.get_radii().bottom_right, Size::new(3.5, 4.0));
    }

    /// C++ source: round_rect_unittests.cc:Shift
    #[test]
    fn test_round_rect_shift() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_xywh(10.0, 10.0, 30.0, 30.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 2.0),
                Size::new(3.0, 4.0),
                Size::new(5.0, 6.0),
                Size::new(7.0, 8.0),
            ),
        );
        let shifted = round_rect.shift(5.0, 6.0);
        assert_eq!(
            shifted.get_bounds(),
            Rect::from_ltrb(15.0, 16.0, 45.0, 46.0)
        );
        assert_eq!(shifted.get_radii(), round_rect.get_radii());
    }

    /// C++ source: round_rect_unittests.cc:ExpandScalar
    #[test]
    fn test_round_rect_expand_scalar() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_xywh(10.0, 10.0, 30.0, 30.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 2.0),
                Size::new(3.0, 4.0),
                Size::new(5.0, 6.0),
                Size::new(7.0, 8.0),
            ),
        );
        let expanded = round_rect.expand(5.0);
        assert_eq!(expanded.get_bounds(), Rect::from_ltrb(5.0, 5.0, 45.0, 45.0));
        assert_eq!(expanded.get_radii(), round_rect.get_radii());
    }

    /// C++ source: round_rect_unittests.cc:ExpandTwoScalars
    #[test]
    fn test_round_rect_expand_two_scalars() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_xywh(10.0, 10.0, 30.0, 30.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 2.0),
                Size::new(3.0, 4.0),
                Size::new(5.0, 6.0),
                Size::new(7.0, 8.0),
            ),
        );
        let expanded = round_rect.expand_xy(5.0, 6.0);
        assert_eq!(expanded.get_bounds(), Rect::from_ltrb(5.0, 4.0, 45.0, 46.0));
    }

    /// C++ source: round_rect_unittests.cc:ExpandFourScalars
    #[test]
    fn test_round_rect_expand_four_scalars() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_xywh(10.0, 10.0, 30.0, 30.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 2.0),
                Size::new(3.0, 4.0),
                Size::new(5.0, 6.0),
                Size::new(7.0, 8.0),
            ),
        );
        let expanded = round_rect.expand_ltrb(5.0, 6.0, 7.0, 8.0);
        assert_eq!(expanded.get_bounds(), Rect::from_ltrb(5.0, 4.0, 47.0, 48.0));
    }

    /// C++ source: round_rect_unittests.cc:ContractScalar
    #[test]
    fn test_round_rect_contract_scalar() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_xywh(10.0, 10.0, 30.0, 30.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 2.0),
                Size::new(3.0, 4.0),
                Size::new(5.0, 6.0),
                Size::new(7.0, 8.0),
            ),
        );
        let contracted = round_rect.expand(-2.0);
        assert_eq!(
            contracted.get_bounds(),
            Rect::from_ltrb(12.0, 12.0, 38.0, 38.0)
        );
    }

    /// C++ source: round_rect_unittests.cc:ContractTwoScalars
    #[test]
    fn test_round_rect_contract_two_scalars() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_xywh(10.0, 10.0, 30.0, 30.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 2.0),
                Size::new(3.0, 4.0),
                Size::new(5.0, 6.0),
                Size::new(7.0, 8.0),
            ),
        );
        let contracted = round_rect.expand_xy(-1.0, -2.0);
        assert_eq!(
            contracted.get_bounds(),
            Rect::from_ltrb(11.0, 12.0, 39.0, 38.0)
        );
    }

    /// C++ source: round_rect_unittests.cc:ContractFourScalars
    #[test]
    fn test_round_rect_contract_four_scalars() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_xywh(10.0, 10.0, 30.0, 30.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 2.0),
                Size::new(3.0, 4.0),
                Size::new(5.0, 6.0),
                Size::new(7.0, 8.0),
            ),
        );
        let contracted = round_rect.expand_ltrb(-1.0, -1.5, -2.0, -2.5);
        assert_eq!(
            contracted.get_bounds(),
            Rect::from_ltrb(11.0, 11.5, 38.0, 37.5)
        );
    }

    /// C++ source: round_rect_unittests.cc:ContractAndRequireRadiiAdjustment
    #[test]
    fn test_round_rect_contract_and_require_radii_adjustment() {
        let round_rect = RoundRect::from_rect_radii(
            Rect::from_xywh(10.0, 10.0, 30.0, 30.0),
            RoundingRadii::from_radii(
                Size::new(1.0, 2.0),
                Size::new(3.0, 4.0),
                Size::new(5.0, 6.0),
                Size::new(7.0, 8.0),
            ),
        );
        let contracted = round_rect.expand(-12.0);
        assert_eq!(
            contracted.get_bounds(),
            Rect::from_ltrb(22.0, 22.0, 28.0, 28.0)
        );
        assert_eq!(contracted.get_radii().top_left, Size::new(0.5, 1.0));
        assert_eq!(contracted.get_radii().top_right, Size::new(1.5, 2.0));
        assert_eq!(contracted.get_radii().bottom_left, Size::new(2.5, 3.0));
        assert_eq!(contracted.get_radii().bottom_right, Size::new(3.5, 4.0));
    }

    /// C++ source: round_rect_unittests.cc:NoCornerRoundRectContains
    #[test]
    fn test_no_corner_round_rect_contains() {
        let bounds = Rect::from_ltrb(-50.0, -50.0, 50.0, 50.0);
        let no_corners = RoundRect::from_rect_xy(bounds, 0.0, 0.0);

        assert!(no_corners.contains(Point::new(-50.0, -50.0)));
        assert!(no_corners.contains(Point::new(-50.0, 49.99)));
        assert!(no_corners.contains(Point::new(49.99, -50.0)));
        assert!(no_corners.contains(Point::new(49.99, 49.99)));
        assert!(!no_corners.contains(Point::new(-50.01, -50.0)));
        assert!(!no_corners.contains(Point::new(-50.0, -50.01)));
        assert!(!no_corners.contains(Point::new(-50.01, 50.0)));
        assert!(!no_corners.contains(Point::new(-50.0, 50.01)));
        assert!(!no_corners.contains(Point::new(50.01, -50.0)));
        assert!(!no_corners.contains(Point::new(50.0, -50.01)));
        assert!(!no_corners.contains(Point::new(50.01, 50.0)));
        assert!(!no_corners.contains(Point::new(50.0, 50.01)));
    }

    /// C++ source: round_rect_unittests.cc:TinyCornerRoundRectContains
    #[test]
    fn test_tiny_corner_round_rect_contains() {
        let bounds = Rect::from_ltrb(-50.0, -50.0, 50.0, 50.0);
        let tiny_corners = RoundRect::from_rect_xy(bounds, 0.01, 0.01);

        assert!(!tiny_corners.contains(Point::new(-50.0, -50.0)));
        assert!(!tiny_corners.contains(Point::new(-50.0, 50.0)));
        assert!(!tiny_corners.contains(Point::new(50.0, -50.0)));
        assert!(!tiny_corners.contains(Point::new(50.0, 50.0)));
    }

    /// C++ source: round_rect_unittests.cc:UniformCircularRoundRectContains
    #[test]
    fn test_uniform_circular_round_rect_contains() {
        let bounds = Rect::from_ltrb(-50.0, -50.0, 50.0, 50.0);
        let expanded_2_r_2 =
            RoundRect::from_rect_xy(rect_expand_ltrb(bounds, 2.0, 2.0, 2.0, 2.0), 2.0, 2.0);

        assert!(expanded_2_r_2.contains(Point::new(-50.0, -50.0)));
        assert!(expanded_2_r_2.contains(Point::new(-50.0, 50.0)));
        assert!(expanded_2_r_2.contains(Point::new(50.0, -50.0)));
        assert!(expanded_2_r_2.contains(Point::new(50.0, 50.0)));

        let sqrt2 = std::f32::consts::SQRT_2;
        let coord_out = 50.0 + sqrt2 + K_EH_CLOSE_ENOUGH;
        let coord_in = 50.0 + sqrt2 - K_EH_CLOSE_ENOUGH;

        assert!(expanded_2_r_2.contains(Point::new(-coord_in, -coord_in)));
        assert!(!expanded_2_r_2.contains(Point::new(-coord_out, -coord_out)));
        assert!(expanded_2_r_2.contains(Point::new(coord_in, -coord_in)));
        assert!(!expanded_2_r_2.contains(Point::new(coord_out, -coord_out)));
        assert!(expanded_2_r_2.contains(Point::new(-coord_in, coord_in)));
        assert!(!expanded_2_r_2.contains(Point::new(-coord_out, coord_out)));
        assert!(expanded_2_r_2.contains(Point::new(coord_in, coord_in)));
        assert!(!expanded_2_r_2.contains(Point::new(coord_out, coord_out)));
    }

    /// C++ source: round_rect_unittests.cc:UniformEllipticalRoundRectContains
    #[test]
    fn test_uniform_elliptical_round_rect_contains() {
        let bounds = Rect::from_ltrb(-50.0, -50.0, 50.0, 50.0);
        let expanded_2_r_2 =
            RoundRect::from_rect_xy(rect_expand_ltrb(bounds, 2.0, 2.0, 2.0, 2.0), 2.0, 3.0);

        assert!(expanded_2_r_2.contains(Point::new(-50.0, -50.0)));
        assert!(expanded_2_r_2.contains(Point::new(-50.0, 50.0)));
        assert!(expanded_2_r_2.contains(Point::new(50.0, -50.0)));
        assert!(expanded_2_r_2.contains(Point::new(50.0, 50.0)));

        let sqrt2 = std::f32::consts::SQRT_2;
        let x_coord_out = 50.0 + sqrt2 + K_EH_CLOSE_ENOUGH;
        let x_coord_in = 50.0 + sqrt2 - K_EH_CLOSE_ENOUGH;
        let y_coord_out = 49.0 + 3.0 * sqrt2 / 2.0 + K_EH_CLOSE_ENOUGH;
        let y_coord_in = 49.0 + 3.0 * sqrt2 / 2.0 - K_EH_CLOSE_ENOUGH;

        assert!(expanded_2_r_2.contains(Point::new(-x_coord_in, -y_coord_in)));
        assert!(!expanded_2_r_2.contains(Point::new(-x_coord_out, -y_coord_out)));
        assert!(expanded_2_r_2.contains(Point::new(x_coord_in, -y_coord_in)));
        assert!(!expanded_2_r_2.contains(Point::new(x_coord_out, -y_coord_out)));
        assert!(expanded_2_r_2.contains(Point::new(-x_coord_in, y_coord_in)));
        assert!(!expanded_2_r_2.contains(Point::new(-x_coord_out, y_coord_out)));
        assert!(expanded_2_r_2.contains(Point::new(x_coord_in, y_coord_in)));
        assert!(!expanded_2_r_2.contains(Point::new(x_coord_out, y_coord_out)));
    }

    /// C++ source: round_rect_unittests.cc:DifferingCornersRoundRectContains
    #[test]
    fn test_differing_corners_round_rect_contains() {
        let bounds = Rect::from_ltrb(-50.0, -50.0, 50.0, 50.0);
        let round_rect = RoundRect::from_rect_radii(
            bounds,
            RoundingRadii::from_radii(
                Size::new(2.0, 3.0),
                Size::new(4.0, 5.0),
                Size::new(6.0, 7.0),
                Size::new(8.0, 9.0),
            ),
        );

        let sqrt2 = std::f32::consts::SQRT_2;
        let coord = |radius: Scalar| -> Scalar {
            50.0 - radius + radius * sqrt2 / 2.0 - K_EH_CLOSE_ENOUGH
        };
        let coord_in = |radius: Scalar| -> Scalar { coord(radius) - K_EH_CLOSE_ENOUGH };
        let coord_out = |radius: Scalar| -> Scalar { coord(radius) + 2.0 * K_EH_CLOSE_ENOUGH };

        assert!(round_rect.contains(Point::new(-coord_in(2.0), -coord_in(3.0))));
        assert!(!round_rect.contains(Point::new(-coord_out(2.0), -coord_out(3.0))));
        assert!(round_rect.contains(Point::new(coord_in(4.0), -coord_in(5.0))));
        assert!(!round_rect.contains(Point::new(coord_out(4.0), -coord_out(5.0))));
        assert!(round_rect.contains(Point::new(-coord_in(6.0), coord_in(7.0))));
        assert!(!round_rect.contains(Point::new(-coord_out(6.0), coord_out(7.0))));
        assert!(round_rect.contains(Point::new(coord_in(8.0), coord_in(9.0))));
        assert!(!round_rect.contains(Point::new(coord_out(8.0), coord_out(9.0))));
    }
}
