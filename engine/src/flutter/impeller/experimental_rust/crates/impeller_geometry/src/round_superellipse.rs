#![forbid(unsafe_code)]

use crate::matrix::Matrix;
use crate::point::Point;
use crate::rect::Rect;
use crate::scalar::{scalar_nearly_equal, Radians, Scalar, K_EH_CLOSE_ENOUGH, K_SQRT_2};
use crate::size::Size;
use crate::vector::Vector3;
use std::ops::Mul;

#[inline]
fn rect_is_finite(r: Rect) -> bool {
    r.origin.is_finite() && r.size.is_finite()
}

#[inline]
fn normalize_point_cpp(p: Point) -> Point {
    let length = p.get_length();
    if length == 0.0 {
        Point::new(1.0, 0.0)
    } else {
        Point::new(p.x / length, p.y / length)
    }
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

    pub const fn make_nine_patch(left: Scalar, top: Scalar, right: Scalar, bottom: Scalar) -> Self {
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
        self.top_left.is_finite()
            && self.top_right.is_finite()
            && self.bottom_left.is_finite()
            && self.bottom_right.is_finite()
    }

    pub fn are_all_corners_same(&self) -> bool {
        self.are_all_corners_same_with_tolerance(K_EH_CLOSE_ENOUGH)
    }

    pub fn are_all_corners_same_with_tolerance(&self, tolerance: Scalar) -> bool {
        scalar_nearly_equal(self.top_left.width, self.top_right.width, tolerance)
            && scalar_nearly_equal(self.top_left.width, self.bottom_right.width, tolerance)
            && scalar_nearly_equal(self.top_left.width, self.bottom_left.width, tolerance)
            && scalar_nearly_equal(self.top_left.height, self.top_right.height, tolerance)
            && scalar_nearly_equal(self.top_left.height, self.bottom_right.height, tolerance)
            && scalar_nearly_equal(self.top_left.height, self.bottom_left.height, tolerance)
    }

    pub fn are_all_corners_circular(&self) -> bool {
        scalar_nearly_equal(self.top_left.width, self.top_left.height, K_EH_CLOSE_ENOUGH)
            && scalar_nearly_equal(
                self.top_right.width,
                self.top_right.height,
                K_EH_CLOSE_ENOUGH,
            )
            && scalar_nearly_equal(
                self.bottom_left.width,
                self.bottom_left.height,
                K_EH_CLOSE_ENOUGH,
            )
            && scalar_nearly_equal(
                self.bottom_right.width,
                self.bottom_right.height,
                K_EH_CLOSE_ENOUGH,
            )
    }

    pub fn scaled(&self, in_bounds: Rect) -> Self {
        if !rect_is_finite(in_bounds) {
            return Self::ZERO;
        }
        let bounds = in_bounds.get_positive();
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
        Self::from_rect_radii(
            rect,
            RoundingRadii::make_radii(Size::new(x_radius, y_radius)),
        )
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
        let bounds = in_bounds.get_positive();
        Self {
            bounds,
            radii: in_radii.scaled(bounds),
        }
    }

    pub fn make_rect_radii(in_bounds: Rect, in_radii: RoundingRadii) -> Self {
        Self::from_rect_radii(in_bounds, in_radii)
    }

    pub const fn bounds(&self) -> Rect {
        self.bounds
    }

    pub const fn get_bounds(&self) -> Rect {
        self.bounds
    }

    pub const fn radii(&self) -> RoundingRadii {
        self.radii
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
            && scalar_nearly_equal(
                self.radii.top_left.width,
                self.bounds.get_width() * 0.5,
                K_EH_CLOSE_ENOUGH,
            )
            && scalar_nearly_equal(
                self.radii.top_left.height,
                self.bounds.get_height() * 0.5,
                K_EH_CLOSE_ENOUGH,
            )
    }

    pub fn contains(&self, p: Point) -> bool {
        if !self.bounds.contains(p) {
            return false;
        }
        let left = self.bounds.get_left();
        let top = self.bounds.get_top();
        let right = self.bounds.get_right();
        let bottom = self.bounds.get_bottom();

        rrect_corner_contains(
            p,
            Point::new(left, top),
            Point::new(-1.0, -1.0),
            self.radii.top_left,
        ) && rrect_corner_contains(
            p,
            Point::new(right, top),
            Point::new(1.0, -1.0),
            self.radii.top_right,
        ) && rrect_corner_contains(
            p,
            Point::new(left, bottom),
            Point::new(-1.0, 1.0),
            self.radii.bottom_left,
        ) && rrect_corner_contains(
            p,
            Point::new(right, bottom),
            Point::new(1.0, 1.0),
            self.radii.bottom_right,
        )
    }

    pub fn shift(&self, dx: Scalar, dy: Scalar) -> Self {
        Self::from_rect_radii(self.bounds.shift(dx, dy), self.radii)
    }

    pub fn expand(&self, amount: Scalar) -> Self {
        self.expand_ltrb(amount, amount, amount, amount)
    }

    pub fn expand_xy(&self, horizontal: Scalar, vertical: Scalar) -> Self {
        self.expand_ltrb(horizontal, vertical, horizontal, vertical)
    }

    pub fn expand_ltrb(&self, left: Scalar, top: Scalar, right: Scalar, bottom: Scalar) -> Self {
        Self::from_rect_radii(self.bounds.expand(left, top, right, bottom), self.radii)
    }
}

fn rrect_corner_contains(p: Point, corner: Point, direction: Point, radii: Size) -> bool {
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

/// Ported from `impeller::RoundSuperellipse` (`round_superellipse.h` / `round_superellipse.cc`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RoundSuperellipse {
    bounds: Rect,
    radii: RoundingRadii,
}

impl RoundSuperellipse {
    pub const fn new() -> Self {
        Self {
            bounds: Rect::ZERO,
            radii: RoundingRadii::ZERO,
        }
    }

    pub fn from_rect(rect: Rect) -> Self {
        Self::make_rect_radii(rect, RoundingRadii::ZERO)
    }

    pub fn make_rect(rect: Rect) -> Self {
        Self::from_rect(rect)
    }

    pub fn from_oval(rect: Rect) -> Self {
        let bounds = rect.get_positive();
        Self::make_rect_radii(
            bounds,
            RoundingRadii::make_radii(Size::new(bounds.size.width * 0.5, bounds.size.height * 0.5)),
        )
    }

    pub fn make_oval(rect: Rect) -> Self {
        Self::from_oval(rect)
    }

    pub fn from_rect_radius(rect: Rect, radius: Scalar) -> Self {
        Self::make_rect_radii(rect, RoundingRadii::make_radius(radius))
    }

    pub fn make_rect_radius(rect: Rect, radius: Scalar) -> Self {
        Self::from_rect_radius(rect, radius)
    }

    pub fn from_rect_xy(rect: Rect, x_radius: Scalar, y_radius: Scalar) -> Self {
        Self::make_rect_radii(
            rect,
            RoundingRadii::make_radii(Size::new(x_radius, y_radius)),
        )
    }

    pub fn make_rect_xy(rect: Rect, x_radius: Scalar, y_radius: Scalar) -> Self {
        Self::from_rect_xy(rect, x_radius, y_radius)
    }

    pub fn from_rect_size(rect: Rect, corner_radii: Size) -> Self {
        Self::make_rect_radii(rect, RoundingRadii::make_radii(corner_radii))
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
        Self::make_rect_radii(
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
        Self::make_rect_radii(in_bounds, in_radii)
    }

    pub fn make_rect_radii(in_bounds: Rect, in_radii: RoundingRadii) -> Self {
        if !rect_is_finite(in_bounds) {
            return Self::default();
        }
        let bounds = in_bounds.get_positive();
        Self {
            bounds,
            radii: in_radii.scaled(bounds),
        }
    }

    #[inline]
    pub const fn bounds(&self) -> Rect {
        self.bounds
    }

    #[inline]
    pub const fn get_bounds(&self) -> Rect {
        self.bounds
    }

    #[inline]
    pub const fn radii(&self) -> RoundingRadii {
        self.radii
    }

    #[inline]
    pub const fn get_radii(&self) -> RoundingRadii {
        self.radii
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        rect_is_finite(self.bounds) && self.radii.is_finite()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }

    #[inline]
    pub fn is_rect(&self) -> bool {
        !self.bounds.is_empty() && self.radii.are_all_corners_empty()
    }

    #[inline]
    pub fn is_oval(&self) -> bool {
        !self.bounds.is_empty()
            && self.radii.are_all_corners_same()
            && scalar_nearly_equal(
                self.radii.top_left.width,
                self.bounds.get_width() * 0.5,
                K_EH_CLOSE_ENOUGH,
            )
            && scalar_nearly_equal(
                self.radii.top_left.height,
                self.bounds.get_height() * 0.5,
                K_EH_CLOSE_ENOUGH,
            )
    }

    pub fn contains(&self, p: Point) -> bool {
        if !self.bounds.contains(p) {
            return false;
        }
        let param = RoundSuperellipseParam::make_bounds_radii(self.bounds, self.radii);
        param.contains(p)
    }

    #[inline]
    pub fn shift(&self, dx: Scalar, dy: Scalar) -> Self {
        Self::make_rect_radii(self.bounds.shift(dx, dy), self.radii)
    }

    #[inline]
    pub fn expand(&self, amount: Scalar) -> Self {
        Self::make_rect_radii(self.bounds.expand_all(amount), self.radii)
    }

    #[inline]
    pub fn expand_xy(&self, horizontal: Scalar, vertical: Scalar) -> Self {
        Self::make_rect_radii(self.bounds.expand_xy(horizontal, vertical), self.radii)
    }

    #[inline]
    pub fn expand_ltrb(&self, left: Scalar, top: Scalar, right: Scalar, bottom: Scalar) -> Self {
        Self::make_rect_radii(self.bounds.expand(left, top, right, bottom), self.radii)
    }

    #[inline]
    pub fn to_approximate_round_rect(&self) -> RoundRect {
        RoundRect::make_rect_radii(self.bounds, self.radii)
    }

    #[inline]
    pub fn to_param(&self) -> RoundSuperellipseParam {
        RoundSuperellipseParam::make_bounds_radii(self.bounds, self.radii)
    }
}

/// Receiver trait for contour segments emitted by `RoundSuperellipseParam::dispatch`
/// (matching `impeller::PathReceiver` in `path_source.h`).
pub trait PathReceiver {
    fn move_to(&mut self, p2: Point, will_be_closed: bool);
    fn line_to(&mut self, p2: Point);
    fn quad_to(&mut self, cp: Point, p2: Point) {
        let _ = (cp, p2);
    }
    fn conic_to(&mut self, cp: Point, p2: Point, weight: Scalar) -> bool {
        let _ = (cp, p2, weight);
        false
    }
    fn cubic_to(&mut self, cp1: Point, cp2: Point, p2: Point);
    fn close(&mut self);
}

/// Parameters for drawing a square-like rounded superellipse octant
/// (`RoundSuperellipseParam::Octant` in `round_superellipse_param.h`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RoundSuperellipseOctant {
    pub offset: Point,
    pub se_a: Scalar,
    pub se_n: Scalar,
    pub se_max_theta: Scalar,
    pub circle_start: Point,
    pub circle_center: Point,
    pub circle_max_angle: Radians,
    pub circle_radius: Scalar,
}

impl RoundSuperellipseOctant {
    /// Evaluates the superellipse (Lamé curve) point in octant-local coordinates for implicit
    /// parameter `theta` in `0.0..=self.se_max_theta`.
    pub fn evaluate_lame(&self, theta: Scalar) -> Point {
        if self.se_n <= 0.0 || self.se_a == 0.0 {
            return Point::new(0.0, self.se_a);
        }
        let (s, c) = theta.sin_cos();
        let exp = 2.0 / self.se_n;
        Point::new(
            self.se_a * s.abs().powf(exp).copysign(s),
            self.se_a * c.abs().powf(exp).copysign(c),
        )
    }
}

/// Parameters for drawing a quadrant of a rounded superellipse
/// (`RoundSuperellipseParam::Quadrant` in `round_superellipse_param.h`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RoundSuperellipseQuadrant {
    pub offset: Point,
    pub signed_scale: Point,
    pub top: RoundSuperellipseOctant,
    pub right: RoundSuperellipseOctant,
}

/// Expanded drawing & hit-testing parameters for a rounded superellipse
/// (`RoundSuperellipseParam` in `round_superellipse_param.h` / `round_superellipse_param.cc`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RoundSuperellipseParam {
    pub top_right: RoundSuperellipseQuadrant,
    pub bottom_right: RoundSuperellipseQuadrant,
    pub bottom_left: RoundSuperellipseQuadrant,
    pub top_left: RoundSuperellipseQuadrant,
    pub all_corners_same: bool,
    pub top_split: Scalar,
    pub bottom_split: Scalar,
    pub left_split: Scalar,
    pub right_split: Scalar,
}

impl RoundSuperellipseParam {
    pub const K_GAP_FACTOR: Scalar = 0.292_893_22; // 1.0 - cos(pi/4)

    pub fn make_bounds_radius(bounds: Rect, radius: Scalar) -> Self {
        Self {
            top_right: compute_quadrant(
                bounds.get_center(),
                bounds.get_right_top(),
                Size::new(radius, radius),
                Size::new(-1.0, 1.0),
            ),
            all_corners_same: true,
            ..Self::default()
        }
    }

    pub fn from_bounds_radius(bounds: Rect, radius: Scalar) -> Self {
        Self::make_bounds_radius(bounds, radius)
    }

    pub fn make_bounds_radii(bounds: Rect, radii: RoundingRadii) -> Self {
        if radii.are_all_corners_same() && !radii.top_left.is_empty() {
            // Having four empty corners indicates a rectangle, which needs special
            // treatment on border containment and therefore is not `all_corners_same`.
            let center = bounds.get_center();
            return Self {
                top_right: compute_quadrant(
                    center,
                    bounds.get_right_top(),
                    radii.top_right,
                    Size::new(-1.0, 1.0),
                ),
                all_corners_same: true,
                top_split: center.x,
                bottom_split: center.x,
                left_split: center.y,
                right_split: center.y,
                ..Self::default()
            };
        }

        let top_split = split(
            bounds.get_left(),
            bounds.get_right(),
            radii.top_left.width,
            radii.top_right.width,
        );
        let right_split = split(
            bounds.get_top(),
            bounds.get_bottom(),
            radii.top_right.height,
            radii.bottom_right.height,
        );
        let bottom_split = split(
            bounds.get_left(),
            bounds.get_right(),
            radii.bottom_left.width,
            radii.bottom_right.width,
        );
        let left_split = split(
            bounds.get_top(),
            bounds.get_bottom(),
            radii.top_left.height,
            radii.bottom_left.height,
        );

        Self {
            top_right: compute_quadrant(
                Point::new(top_split, right_split),
                bounds.get_right_top(),
                radii.top_right,
                Size::new(1.0, -1.0),
            ),
            bottom_right: compute_quadrant(
                Point::new(bottom_split, right_split),
                bounds.get_right_bottom(),
                radii.bottom_right,
                Size::new(1.0, 1.0),
            ),
            bottom_left: compute_quadrant(
                Point::new(bottom_split, left_split),
                bounds.get_left_bottom(),
                radii.bottom_left,
                Size::new(-1.0, 1.0),
            ),
            top_left: compute_quadrant(
                Point::new(top_split, left_split),
                bounds.get_left_top(),
                radii.top_left,
                Size::new(-1.0, -1.0),
            ),
            all_corners_same: false,
            top_split,
            bottom_split,
            left_split,
            right_split,
        }
    }

    pub fn from_bounds_radii(bounds: Rect, radii: RoundingRadii) -> Self {
        Self::make_bounds_radii(bounds, radii)
    }

    pub fn contains(&self, point: Point) -> bool {
        if self.all_corners_same {
            return corner_contains(&self.top_right, point, false);
        }
        corner_contains(&self.top_right, point, true)
            && corner_contains(&self.bottom_right, point, true)
            && corner_contains(&self.bottom_left, point, true)
            && corner_contains(&self.top_left, point, true)
    }

    pub fn dispatch(&self, path_receiver: &mut impl PathReceiver) {
        let start = self.top_right.offset
            + self.top_right.signed_scale
                * (self.top_right.top.offset + Point::new(0.0, self.top_right.top.se_a));
        path_receiver.move_to(start, true);

        if self.all_corners_same {
            add_quadrant(path_receiver, &self.top_right, false, Point::new(1.0, 1.0));
            add_quadrant(path_receiver, &self.top_right, true, Point::new(1.0, -1.0));
            add_quadrant(
                path_receiver,
                &self.top_right,
                false,
                Point::new(-1.0, -1.0),
            );
            add_quadrant(path_receiver, &self.top_right, true, Point::new(-1.0, 1.0));
        } else {
            add_quadrant(path_receiver, &self.top_right, false, Point::new(1.0, 1.0));
            add_quadrant(
                path_receiver,
                &self.bottom_right,
                true,
                Point::new(1.0, 1.0),
            );
            add_quadrant(
                path_receiver,
                &self.bottom_left,
                false,
                Point::new(1.0, 1.0),
            );
            add_quadrant(path_receiver, &self.top_left, true, Point::new(1.0, 1.0));
        }

        path_receiver.line_to(start);
        path_receiver.close();
    }
}

#[inline]
fn split(left: Scalar, right: Scalar, ratio_left: Scalar, ratio_right: Scalar) -> Scalar {
    if ratio_left == 0.0 && ratio_right == 0.0 {
        return (left + right) * 0.5;
    }
    (left * ratio_right + right * ratio_left) / (ratio_left + ratio_right)
}

#[inline]
fn replace_nan_with_default(in_pt: Point, default_size: Size) -> Point {
    Point::new(
        if in_pt.x.is_nan() {
            default_size.width
        } else {
            in_pt.x
        },
        if in_pt.y.is_nan() {
            default_size.height
        } else {
            in_pt.y
        },
    )
}

#[inline]
fn flip_point(a: Point) -> Point {
    Point::new(a.y, a.x)
}

const PRECOMPUTED_OCTANT_VARIABLES: [[Scalar; 2]; 11] = [
    /*ratio=2.00*/ [2.000_000_0, 1.132_766_7],
    /*ratio=2.10*/ [2.183_498_1, 1.203_119_2],
    /*ratio=2.20*/ [2.338_886_7, 1.286_987_9],
    /*ratio=2.30*/ [2.486_605_6, 1.363_519_4],
    /*ratio=2.40*/ [2.622_266, 1.447_179_8],
    /*ratio=2.50*/ [2.751_489_9, 1.533_858_2],
    /*ratio=3.00*/ [3.362_982_7, 1.982_882_9],
    /*ratio=3.50*/ [4.086_499, 2.238_118_4],
    /*ratio=4.00*/ [4.854_811, 2.475_634_6],
    /*ratio=4.50*/ [5.629_455_6, 2.729_486],
    /*ratio=5.00*/ [6.430_238, 2.980_204],
];

const MIN_RATIO: Scalar = 2.00;
const FIRST_STEP_INVERSE: Scalar = 10.0;
const FIRST_MAX_RATIO: Scalar = 2.50;
const FIRST_NUM_RECORDS: Scalar = 6.0;
const SECOND_STEP_INVERSE: Scalar = 2.0;
const SECOND_MAX_RATIO: Scalar = 5.00;
const THIRD_N_SLOPE: Scalar = 1.559_599_4;
const THIRD_KXJ_SLOPE: Scalar = 0.522_807_2;

fn compute_n_and_xj(mut ratio: Scalar) -> [Scalar; 2] {
    let num_records = PRECOMPUTED_OCTANT_VARIABLES.len();
    if ratio > SECOND_MAX_RATIO {
        let n = THIRD_N_SLOPE * (ratio - SECOND_MAX_RATIO)
            + PRECOMPUTED_OCTANT_VARIABLES[num_records - 1][0];
        let k_xj = THIRD_KXJ_SLOPE * (ratio - SECOND_MAX_RATIO)
            + PRECOMPUTED_OCTANT_VARIABLES[num_records - 1][1];
        return [n, 1.0 - 1.0 / k_xj];
    }

    ratio = ratio.clamp(MIN_RATIO, SECOND_MAX_RATIO);
    let steps = if ratio < FIRST_MAX_RATIO {
        (ratio - MIN_RATIO) * FIRST_STEP_INVERSE
    } else {
        (ratio - FIRST_MAX_RATIO) * SECOND_STEP_INVERSE + FIRST_NUM_RECORDS - 1.0
    };

    let left = (steps.floor().max(0.0) as usize).min(num_records - 2);
    let frac = steps - (left as Scalar);

    let n = (1.0 - frac) * PRECOMPUTED_OCTANT_VARIABLES[left][0]
        + frac * PRECOMPUTED_OCTANT_VARIABLES[left + 1][0];
    let k_xj = (1.0 - frac) * PRECOMPUTED_OCTANT_VARIABLES[left][1]
        + frac * PRECOMPUTED_OCTANT_VARIABLES[left + 1][1];
    [n, 1.0 - 1.0 / k_xj]
}

fn find_circle_center(a: Point, b: Point, r: Scalar) -> Point {
    let a_to_b = b - a;
    let m = (a + b) * 0.5;
    let c_to_m = Point::new(-a_to_b.y, a_to_b.x);
    let distance_am = a_to_b.get_length() * 0.5;
    let distance_cm = (r * r - distance_am * distance_am).max(0.0).sqrt();
    m - distance_cm * normalize_point_cpp(c_to_m)
}

fn compute_octant(center: Point, a: Scalar, radius: Scalar) -> RoundSuperellipseOctant {
    if radius <= K_EH_CLOSE_ENOUGH {
        return RoundSuperellipseOctant {
            offset: center,
            se_a: a,
            se_n: 0.0,
            circle_start: Point::new(a, a),
            ..RoundSuperellipseOctant::default()
        };
    }

    let ratio = a * 2.0 / radius;
    let g = RoundSuperellipseParam::K_GAP_FACTOR * radius;

    let precomputed_vars = compute_n_and_xj(ratio);
    let n = precomputed_vars[0];
    let xj = precomputed_vars[1] * a;
    let yj = (1.0 - precomputed_vars[1].powf(n)).powf(1.0 / n) * a;
    let max_theta = precomputed_vars[1].powf(n * 0.5).clamp(-1.0, 1.0).asin();

    let tan_phij = (xj / yj).powf(n - 1.0);
    let d = (xj - tan_phij * yj) / (1.0 - tan_phij);
    let r = (a - d - g) * K_SQRT_2;

    let point_m = Point::new(a - g, a - g);
    let point_j = Point::new(xj, yj);
    let circle_center = if radius == 0.0 {
        point_m
    } else {
        find_circle_center(point_j, point_m, r)
    };
    let circle_max_angle = if radius == 0.0 {
        Radians(0.0)
    } else {
        (point_m - circle_center).angle_to(point_j - circle_center)
    };

    RoundSuperellipseOctant {
        offset: center,
        se_a: a,
        se_n: n,
        se_max_theta: max_theta,
        circle_start: point_j,
        circle_center,
        circle_max_angle,
        circle_radius: r,
    }
}

fn compute_quadrant(
    center: Point,
    corner: Point,
    in_radii: Size,
    sign: Size,
) -> RoundSuperellipseQuadrant {
    let corner_vector = corner - center;
    let radii = Size::new(
        in_radii.width.abs().min(corner_vector.x.abs()),
        in_radii.height.abs().min(corner_vector.y.abs()),
    );

    let norm_radius = radii.min_dimension();
    let forward_scale = if norm_radius == 0.0 {
        Size::new(1.0, 1.0)
    } else {
        radii / norm_radius
    };
    let norm_half_size = corner_vector.abs() / forward_scale;
    let signed_scale = replace_nan_with_default(corner_vector / norm_half_size, sign);

    let c = norm_half_size.x - norm_half_size.y;

    RoundSuperellipseQuadrant {
        offset: center,
        signed_scale,
        top: compute_octant(Point::new(0.0, -c), norm_half_size.x, norm_radius),
        right: compute_octant(Point::new(c, 0.0), norm_half_size.y, norm_radius),
    }
}

fn octant_contains(param: &RoundSuperellipseOctant, p: Point) -> bool {
    if p.x < 0.0 || p.y < 0.0 || p.y < p.x {
        return true;
    }
    if p.x <= param.circle_start.x {
        let p_se = p / param.se_a;
        return p_se.x.powf(param.se_n) + p_se.y.powf(param.se_n) <= 1.0;
    }
    let circle_radius_sq = param.circle_start.get_distance_squared(param.circle_center);
    let p_circle = p - param.circle_center;
    p_circle.get_distance_squared(Point::ZERO) < circle_radius_sq
}

fn corner_contains(param: &RoundSuperellipseQuadrant, p: Point, check_quadrant: bool) -> bool {
    let mut norm_point = (p - param.offset) / param.signed_scale;
    if check_quadrant {
        if norm_point.x < 0.0 || norm_point.y < 0.0 {
            return true;
        }
    } else {
        norm_point = norm_point.abs();
    }
    if param.top.se_n < 2.0 || param.right.se_n < 2.0 {
        let x_delta = param.right.offset.x + param.right.se_a - norm_point.x;
        let y_delta = param.top.offset.y + param.top.se_a - norm_point.y;
        let x_within = x_delta > 0.0 || (x_delta == 0.0 && param.signed_scale.x < 0.0);
        let y_within = y_delta > 0.0 || (y_delta == 0.0 && param.signed_scale.y < 0.0);
        return x_within && y_within;
    }
    octant_contains(&param.top, norm_point - param.top.offset)
        && octant_contains(&param.right, flip_point(norm_point - param.right.offset))
}

#[derive(Copy, Clone, Debug)]
struct ConicParam {
    p1: Point,
    c: Point,
    p2: Point,
    weight: Scalar,
}

const K_FLIP: Matrix = Matrix::make_column(
    0.0, 1.0, 0.0, 0.0, //
    1.0, 0.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
);

fn add_quadrant(
    receiver: &mut impl PathReceiver,
    param: &RoundSuperellipseQuadrant,
    reverse: bool,
    scale_sign: Point,
) {
    let s = param.signed_scale * scale_sign;
    let transform = Matrix::make_translate_scale(
        Vector3::new(s.x, s.y, 1.0),
        Vector3::new(param.offset.x, param.offset.y, 0.0),
    );
    if param.top.se_n < 2.0 || param.right.se_n < 2.0 {
        receiver
            .line_to(transform * (param.top.offset + Point::new(param.top.se_a, param.top.se_a)));
        if !reverse {
            receiver.line_to(transform * (param.right.offset + Point::new(param.right.se_a, 0.0)));
        } else {
            receiver.line_to(transform * (param.top.offset + Point::new(0.0, param.top.se_a)));
        }
        return;
    }
    if !reverse {
        add_octant(receiver, &param.top, false, false, &transform);
        add_octant(receiver, &param.right, true, true, &transform);
    } else {
        add_octant(receiver, &param.right, false, true, &transform);
        add_octant(receiver, &param.top, true, false, &transform);
    }
}

fn add_octant(
    receiver: &mut impl PathReceiver,
    param: &RoundSuperellipseOctant,
    reverse: bool,
    flip: bool,
    external_transform: &Matrix,
) {
    let mut transform =
        *external_transform * Matrix::make_translation(param.offset.x, param.offset.y, 0.0);
    if flip {
        transform = transform * K_FLIP;
    }

    let circle_points = circular_arc_points(param);
    let (se_conic_0, se_conic_1) = superellipse_arc_points(param);

    if !reverse {
        receiver.conic_to(
            transform * se_conic_0.c,
            transform * se_conic_0.p2,
            se_conic_0.weight,
        );
        receiver.conic_to(
            transform * se_conic_1.c,
            transform * se_conic_1.p2,
            se_conic_1.weight,
        );
        receiver.cubic_to(
            transform * circle_points[1],
            transform * circle_points[2],
            transform * circle_points[3],
        );
    } else {
        receiver.cubic_to(
            transform * circle_points[2],
            transform * circle_points[1],
            transform * circle_points[0],
        );
        receiver.conic_to(
            transform * se_conic_1.c,
            transform * se_conic_1.p1,
            se_conic_1.weight,
        );
        receiver.conic_to(
            transform * se_conic_0.c,
            transform * se_conic_0.p1,
            se_conic_0.weight,
        );
    }
}

fn superellipse_arc_points(param: &RoundSuperellipseOctant) -> (ConicParam, ConicParam) {
    let a_pt = Point::new(0.0, param.se_a);
    let j_pt = param.circle_start;

    let (weight1, weight2, yh_over_a) =
        superellipse_bezier_factors(param.se_n, j_pt.x / param.se_a, j_pt.y / param.se_a);

    let h_pt = Point::new(
        (1.0 - yh_over_a.powf(param.se_n))
            .max(0.0)
            .powf(1.0 / param.se_n)
            * param.se_a,
        yh_over_a * param.se_a,
    );

    let k_a: Scalar = 0.0;
    let k_j = -(j_pt.x / j_pt.y).powf(param.se_n - 1.0);
    let k_h = -(h_pt.x / h_pt.y).powf(param.se_n - 1.0);

    (
        ConicParam {
            p1: a_pt,
            c: intersection(a_pt, k_a, h_pt, k_h),
            p2: h_pt,
            weight: weight1,
        },
        ConicParam {
            p1: h_pt,
            c: intersection(h_pt, k_h, j_pt, k_j),
            p2: j_pt,
            weight: weight2,
        },
    )
}

fn circular_arc_points(param: &RoundSuperellipseOctant) -> [Point; 4] {
    let start_vector = param.circle_start - param.circle_center;
    let end_vector = start_vector.rotate(Radians(-param.circle_max_angle.0));
    let circle_end = param.circle_center + end_vector;
    let start_tangent = normalize_point_cpp(Point::new(start_vector.y, -start_vector.x));
    let end_tangent = normalize_point_cpp(Point::new(-end_vector.y, end_vector.x));
    let bezier_factor = (param.circle_max_angle.0 / 4.0).tan() * (4.0 / 3.0);
    let radius = start_vector.get_length();

    [
        param.circle_start,
        param.circle_start + start_tangent * (bezier_factor * radius),
        circle_end + end_tangent * (bezier_factor * radius),
        circle_end,
    ]
}

fn superellipse_bezier_factors(
    mut n: Scalar,
    xj_over_a: Scalar,
    yj_over_a: Scalar,
) -> (Scalar, Scalar, Scalar) {
    const PRECOMPUTED_BEZIER_VARIABLES: [[Scalar; 2]; 13] = [
        /*n= 2.0*/ [0.7078, 8.3194],
        /*n= 3.0*/ [0.7895, 2.4523],
        /*n= 4.0*/ [0.8379, 1.8528],
        /*n= 5.0*/ [0.8701, 1.6891],
        /*n= 6.0*/ [0.8932, 1.5806],
        /*n= 7.0*/ [0.9107, 1.5043],
        /*n= 8.0*/ [0.9244, 1.4470],
        /*n= 9.0*/ [0.9355, 1.4037],
        /*n=10.0*/ [0.9448, 1.3701],
        /*n=11.0*/ [0.9526, 1.3431],
        /*n=12.0*/ [0.9594, 1.3212],
        /*n=13.0*/ [0.9653, 1.3032],
        /*n=14.0*/ [0.9705, 1.2880],
    ];
    let num_records = PRECOMPUTED_BEZIER_VARIABLES.len();
    const STEP: Scalar = 1.00;
    const MIN_N: Scalar = 2.00;
    let max_n = MIN_N + ((num_records - 1) as Scalar) * STEP;

    if n >= max_n {
        n = max_n;
    }

    let steps = ((n - MIN_N) / STEP).clamp(0.0, (num_records - 1) as Scalar);
    let left = (steps.floor().max(0.0) as usize).min(num_records - 2);
    let frac = steps - (left as Scalar);

    // Matches C++ operator precedence in round_superellipse_param.cc:513-516.
    let weight1 = (1.0 - frac) * PRECOMPUTED_BEZIER_VARIABLES[left][0]
        + frac * PRECOMPUTED_BEZIER_VARIABLES[left + 1][0] * n.sqrt();
    let weight2 = (1.0 - frac) * PRECOMPUTED_BEZIER_VARIABLES[left][1]
        + frac * PRECOMPUTED_BEZIER_VARIABLES[left + 1][1] * xj_over_a;

    let yh_proportion = n.sqrt();
    const YA_OVER_A: Scalar = 1.0;
    let yh_over_a = (YA_OVER_A * yh_proportion + yj_over_a) / (yh_proportion + 1.0);

    (weight1, weight2, yh_over_a)
}

fn intersection(p1: Point, k1: Scalar, p2: Point, k2: Scalar) -> Point {
    if (k1 - k2).abs() < K_EH_CLOSE_ENOUGH {
        return (p1 + p2) * 0.5;
    }
    let x = (k1 * p1.x - k2 * p2.x + p2.y - p1.y) / (k1 - k2);
    let y = k1 * (x - p1.x) + p1.y;
    Point::new(x, y)
}
