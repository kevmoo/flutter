use crate::matrix::Matrix;
use crate::point::{IPoint, Point};
use crate::scalar::Scalar;
use crate::size::{ISize, Size};
use crate::vector::Vector3;
use bytemuck::{Pod, Zeroable};

#[inline]
const fn saturated_add_i64(location: i64, distance: i64) -> i64 {
    if location >= 0 {
        if distance > i64::MAX - location {
            return i64::MAX;
        }
    } else if distance < i64::MIN - location {
        return i64::MIN;
    }
    location + distance
}

#[inline]
const fn saturated_sub_i64(upper: i64, lower: i64) -> i64 {
    if upper >= 0 {
        if lower < 0 && upper > i64::MAX + lower {
            return i64::MAX;
        }
    } else if lower > 0 && upper < i64::MIN + lower {
        return i64::MIN;
    }
    upper - lower
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Rect {
    pub origin: Point,
    pub size: Size,
}

impl Rect {
    pub const ZERO: Self = Self {
        origin: Point::ZERO,
        size: Size::ZERO,
    };

    const MINIMUM_HOMOGENOUS: Scalar = 1.0 / ((1 << 14) as Scalar);

    #[inline]
    pub const fn from_xywh(x: Scalar, y: Scalar, width: Scalar, height: Scalar) -> Self {
        Self {
            origin: Point::new(x, y),
            size: Size::new(width, height),
        }
    }

    #[inline]
    pub const fn from_ltrb(left: Scalar, top: Scalar, right: Scalar, bottom: Scalar) -> Self {
        Self::from_xywh(left, top, right - left, bottom - top)
    }

    #[inline]
    pub const fn from_origin_size(origin: Point, size: Size) -> Self {
        Self { origin, size }
    }

    #[inline]
    pub const fn make_wh(width: Scalar, height: Scalar) -> Self {
        Self::from_xywh(0.0, 0.0, width, height)
    }

    #[inline]
    pub const fn make_size(size: Size) -> Self {
        Self::from_origin_size(Point::ZERO, size)
    }

    #[inline]
    pub const fn make_circle_bounds(center: Point, radius: Scalar) -> Self {
        Self::from_ltrb(
            center.x - radius,
            center.y - radius,
            center.x + radius,
            center.y + radius,
        )
    }

    #[inline]
    pub const fn make_ellipse_bounds(center: Point, radii: Size) -> Self {
        Self::from_ltrb(
            center.x - radii.width,
            center.y - radii.height,
            center.x + radii.width,
            center.y + radii.height,
        )
    }

    #[inline]
    pub const fn make_ellipse_bounds_point(center: Point, radii: Point) -> Self {
        Self::from_ltrb(
            center.x - radii.x,
            center.y - radii.y,
            center.x + radii.x,
            center.y + radii.y,
        )
    }

    #[inline]
    pub fn make_point_bounds(points: &[Point]) -> Option<Self> {
        let first = points.first()?;
        let mut left = first.x;
        let mut top = first.y;
        let mut right = first.x;
        let mut bottom = first.y;
        for pt in &points[1..] {
            left = left.min(pt.x);
            top = top.min(pt.y);
            right = right.max(pt.x);
            bottom = bottom.max(pt.y);
        }
        Some(Self::from_ltrb(left, top, right, bottom))
    }

    #[inline]
    pub const fn from_irect(rect: IRect) -> Self {
        Self::from_ltrb(
            rect.get_left() as Scalar,
            rect.get_top() as Scalar,
            rect.get_right() as Scalar,
            rect.get_bottom() as Scalar,
        )
    }

    #[inline]
    pub const fn get_left(&self) -> Scalar {
        self.origin.x
    }

    #[inline]
    pub const fn get_top(&self) -> Scalar {
        self.origin.y
    }

    #[inline]
    pub const fn get_right(&self) -> Scalar {
        self.origin.x + self.size.width
    }

    #[inline]
    pub const fn get_bottom(&self) -> Scalar {
        self.origin.y + self.size.height
    }

    #[inline]
    pub const fn get_x(&self) -> Scalar {
        self.origin.x
    }

    #[inline]
    pub const fn get_y(&self) -> Scalar {
        self.origin.y
    }

    #[inline]
    pub const fn get_width(&self) -> Scalar {
        self.size.width
    }

    #[inline]
    pub const fn get_height(&self) -> Scalar {
        self.size.height
    }

    #[inline]
    pub const fn get_origin(&self) -> Point {
        self.origin
    }

    #[inline]
    pub const fn get_size(&self) -> Size {
        self.size
    }

    #[inline]
    pub const fn get_left_top(&self) -> Point {
        self.origin
    }

    #[inline]
    pub const fn get_right_top(&self) -> Point {
        Point::new(self.get_right(), self.origin.y)
    }

    #[inline]
    pub const fn get_left_bottom(&self) -> Point {
        Point::new(self.origin.x, self.get_bottom())
    }

    #[inline]
    pub const fn get_right_bottom(&self) -> Point {
        Point::new(self.get_right(), self.get_bottom())
    }

    #[inline]
    pub const fn get_ltrb(&self) -> [Scalar; 4] {
        [
            self.get_left(),
            self.get_top(),
            self.get_right(),
            self.get_bottom(),
        ]
    }

    #[inline]
    pub const fn get_xywh(&self) -> [Scalar; 4] {
        [
            self.origin.x,
            self.origin.y,
            self.size.width,
            self.size.height,
        ]
    }

    #[inline]
    pub fn get_center(&self) -> Point {
        Point::new(
            self.get_left() * 0.5 + self.get_right() * 0.5,
            self.get_top() * 0.5 + self.get_bottom() * 0.5,
        )
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        !(self.get_left() < self.get_right() && self.get_top() < self.get_bottom())
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.origin.is_finite() && self.get_right().is_finite() && self.get_bottom().is_finite()
    }

    #[inline]
    pub fn is_square(&self) -> bool {
        !self.is_empty() && self.size.width == self.size.height
    }

    #[inline]
    pub fn area(&self) -> Scalar {
        if self.is_empty() {
            0.0
        } else {
            self.size.width * self.size.height
        }
    }

    #[inline]
    pub fn scale(&self, scale: Scalar) -> Self {
        Self::from_ltrb(
            self.get_left() * scale,
            self.get_top() * scale,
            self.get_right() * scale,
            self.get_bottom() * scale,
        )
    }

    #[inline]
    pub fn scale_xy(&self, scale_x: Scalar, scale_y: Scalar) -> Self {
        Self::from_ltrb(
            self.get_left() * scale_x,
            self.get_top() * scale_y,
            self.get_right() * scale_x,
            self.get_bottom() * scale_y,
        )
    }

    #[inline]
    pub fn contains(&self, p: Point) -> bool {
        !self.is_empty()
            && p.x >= self.get_left()
            && p.y >= self.get_top()
            && p.x < self.get_right()
            && p.y < self.get_bottom()
    }

    #[inline]
    pub fn contains_inclusive(&self, p: Point) -> bool {
        !self.is_empty()
            && p.x >= self.get_left()
            && p.y >= self.get_top()
            && p.x <= self.get_right()
            && p.y <= self.get_bottom()
    }

    #[inline]
    pub fn contains_rect(&self, o: Self) -> bool {
        !self.is_empty()
            && (o.is_empty()
                || (o.get_left() >= self.get_left()
                    && o.get_top() >= self.get_top()
                    && o.get_right() <= self.get_right()
                    && o.get_bottom() <= self.get_bottom()))
    }

    #[inline]
    pub fn get_positive(&self) -> Self {
        if !self.is_empty() {
            return *self;
        }
        let l = self.get_left();
        let t = self.get_top();
        let r = self.get_right();
        let b = self.get_bottom();
        Self::from_ltrb(l.min(r), t.min(b), l.max(r), t.max(b))
    }

    #[inline]
    pub fn get_points(&self) -> [Point; 4] {
        if self.is_empty() {
            return [Point::ZERO; 4];
        }
        [
            self.get_left_top(),
            self.get_right_top(),
            self.get_left_bottom(),
            self.get_right_bottom(),
        ]
    }

    #[inline]
    pub fn get_transformed_points(&self, transform: &Matrix) -> [Point; 4] {
        let pts = self.get_points();
        [
            *transform * pts[0],
            *transform * pts[1],
            *transform * pts[2],
            *transform * pts[3],
        ]
    }

    #[inline]
    pub fn transform_bounds(&self, transform: &Matrix) -> Self {
        if self.is_empty() {
            return Self::ZERO;
        }
        let pts = self.get_transformed_points(transform);
        Self::make_point_bounds(&pts).unwrap_or(Self::ZERO)
    }

    fn interpolate_and_insert(
        clipped: &mut [Point; 8],
        mut index: usize,
        p: Vector3,
        neighbor: Vector3,
    ) -> usize {
        if neighbor.z >= Self::MINIMUM_HOMOGENOUS {
            let t = (Self::MINIMUM_HOMOGENOUS - p.z) / (neighbor.z - p.z);
            clipped[index] = Point::new(
                ((1.0 - t) * p.x + t * neighbor.x) / Self::MINIMUM_HOMOGENOUS,
                ((1.0 - t) * p.y + t * neighbor.y) / Self::MINIMUM_HOMOGENOUS,
            );
            index += 1;
        }
        index
    }

    fn clip_and_insert(
        clipped: &mut [Point; 8],
        mut index: usize,
        left: Vector3,
        p: Vector3,
        right: Vector3,
    ) -> usize {
        if p.z >= Self::MINIMUM_HOMOGENOUS {
            clipped[index] = Point::new(p.x / p.z, p.y / p.z);
            index += 1;
        } else {
            index = Self::interpolate_and_insert(clipped, index, p, left);
            index = Self::interpolate_and_insert(clipped, index, p, right);
        }
        index
    }

    pub fn transform_and_clip_bounds(&self, transform: &Matrix) -> Self {
        if !transform.has_perspective_2d() {
            return self.transform_bounds(transform);
        }
        if self.is_empty() {
            return Self::ZERO;
        }

        let ul = transform.transform_homogenous(self.get_left_top());
        let ur = transform.transform_homogenous(self.get_right_top());
        let ll = transform.transform_homogenous(self.get_left_bottom());
        let lr = transform.transform_homogenous(self.get_right_bottom());

        let mut points = [Point::ZERO; 8];
        let mut index = 0;
        index = Self::clip_and_insert(&mut points, index, ll, ul, ur);
        index = Self::clip_and_insert(&mut points, index, ul, ur, lr);
        index = Self::clip_and_insert(&mut points, index, ur, lr, ll);
        index = Self::clip_and_insert(&mut points, index, lr, ll, ul);

        Self::make_point_bounds(&points[..index]).unwrap_or(Self::ZERO)
    }

    pub fn get_normalizing_transform(&self) -> Matrix {
        if !self.is_empty() {
            let sx = 1.0 / self.get_width();
            let sy = 1.0 / self.get_height();
            let tx = self.get_left() * -sx;
            let ty = self.get_top() * -sy;

            if sx != 0.0 && sy != 0.0 && 0.0 * sx * sy * tx * ty == 0.0 {
                return Matrix::make_column(
                    sx, 0.0, 0.0, 0.0, //
                    0.0, sy, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, //
                    tx, ty, 0.0, 1.0,
                );
            }
        }
        Matrix::make_scale(0.0, 0.0, 1.0)
    }

    #[inline]
    pub fn union(&self, o: Self) -> Self {
        if self.is_empty() {
            return o;
        }
        if o.is_empty() {
            return *self;
        }
        Self::from_ltrb(
            self.get_left().min(o.get_left()),
            self.get_top().min(o.get_top()),
            self.get_right().max(o.get_right()),
            self.get_bottom().max(o.get_bottom()),
        )
    }

    #[inline]
    pub fn intersects_with_rect(&self, o: Self) -> bool {
        !self.is_empty()
            && !o.is_empty()
            && self.get_left() < o.get_right()
            && self.get_top() < o.get_bottom()
            && self.get_right() > o.get_left()
            && self.get_bottom() > o.get_top()
    }

    #[inline]
    pub fn intersection(&self, o: Self) -> Option<Self> {
        if self.intersects_with_rect(o) {
            Some(Self::from_ltrb(
                self.get_left().max(o.get_left()),
                self.get_top().max(o.get_top()),
                self.get_right().min(o.get_right()),
                self.get_bottom().min(o.get_bottom()),
            ))
        } else {
            None
        }
    }

    #[inline]
    pub fn intersection_or_empty(&self, o: Self) -> Self {
        self.intersection(o).unwrap_or(Self::ZERO)
    }

    pub fn cutout(&self, o: Self) -> Option<Self> {
        if self.is_empty() {
            return None;
        }
        let [a_left, a_top, a_right, a_bottom] = self.get_ltrb();
        let [b_left, b_top, b_right, b_bottom] = o.get_ltrb();

        if b_left <= a_left && b_right >= a_right {
            if b_top <= a_top && b_bottom >= a_bottom {
                return None;
            }
            if b_top <= a_top && b_bottom > a_top {
                return Some(Self::from_ltrb(a_left, b_bottom, a_right, a_bottom));
            }
            if b_bottom >= a_bottom && b_top < a_bottom {
                return Some(Self::from_ltrb(a_left, a_top, a_right, b_top));
            }
        }
        if b_top <= a_top && b_bottom >= a_bottom {
            if b_left <= a_left && b_right > a_left {
                return Some(Self::from_ltrb(b_right, a_top, a_right, a_bottom));
            }
            if b_right >= a_right && b_left < a_right {
                return Some(Self::from_ltrb(a_left, a_top, b_left, a_bottom));
            }
        }
        Some(*self)
    }

    #[inline]
    pub fn cutout_or_empty(&self, o: Self) -> Self {
        self.cutout(o).unwrap_or(Self::ZERO)
    }

    #[inline]
    pub const fn shift(&self, dx: Scalar, dy: Scalar) -> Self {
        Self::from_xywh(
            self.origin.x + dx,
            self.origin.y + dy,
            self.size.width,
            self.size.height,
        )
    }

    #[inline]
    pub const fn shift_point(&self, offset: Point) -> Self {
        self.shift(offset.x, offset.y)
    }

    #[inline]
    pub const fn expand(
        &self,
        left: Scalar,
        top: Scalar,
        right: Scalar,
        bottom: Scalar,
    ) -> Self {
        Self::from_ltrb(
            self.get_left() - left,
            self.get_top() - top,
            self.get_right() + right,
            self.get_bottom() + bottom,
        )
    }

    #[inline]
    pub const fn expand_all(&self, amount: Scalar) -> Self {
        self.expand(amount, amount, amount, amount)
    }

    #[inline]
    pub const fn expand_xy(&self, horizontal: Scalar, vertical: Scalar) -> Self {
        self.expand(horizontal, vertical, horizontal, vertical)
    }

    pub fn expand_to_min_transformed_size(
        &self,
        min_transformed_size: Size,
        transform: &Matrix,
    ) -> Option<Self> {
        debug_assert!(!transform.has_perspective_2d());

        let transform_scaling = transform.get_basis_scale_xy();
        if transform_scaling.x == 0.0 || transform_scaling.y == 0.0 {
            return None;
        }

        let min_local_size = Size::new(
            min_transformed_size.width / transform_scaling.x,
            min_transformed_size.height / transform_scaling.y,
        );
        if !min_local_size.is_finite() {
            return None;
        }

        let current_size = self.get_size();
        if current_size.width >= min_local_size.width
            && current_size.height >= min_local_size.height
        {
            return Some(*self);
        }

        let expanded_size = current_size.max(min_local_size);
        Some(Self::make_ellipse_bounds_point(
            self.get_center(),
            Point::new(expanded_size.width * 0.5, expanded_size.height * 0.5),
        ))
    }

    #[inline]
    pub fn project(&self, source: Self) -> Self {
        if self.is_empty() {
            return Self::ZERO;
        }
        source
            .shift(-self.get_left(), -self.get_top())
            .scale_xy(1.0 / self.get_width(), 1.0 / self.get_height())
    }

    #[inline]
    pub fn round_out(&self) -> Self {
        Self::from_ltrb(
            self.get_left().floor(),
            self.get_top().floor(),
            self.get_right().ceil(),
            self.get_bottom().ceil(),
        )
    }

    #[inline]
    pub fn round_in(&self) -> Self {
        Self::from_ltrb(
            self.get_left().ceil(),
            self.get_top().ceil(),
            self.get_right().floor(),
            self.get_bottom().floor(),
        )
    }

    #[inline]
    pub fn round(&self) -> Self {
        Self::from_ltrb(
            self.get_left().round(),
            self.get_top().round(),
            self.get_right().round(),
            self.get_bottom().round(),
        )
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct IRect {
    left: i64,
    top: i64,
    right: i64,
    bottom: i64,
}

impl IRect {
    pub const ZERO: Self = Self {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };

    #[inline]
    pub const fn from_ltrb(left: i64, top: i64, right: i64, bottom: i64) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    #[inline]
    pub const fn from_xywh(x: i64, y: i64, width: i64, height: i64) -> Self {
        Self {
            left: x,
            top: y,
            right: saturated_add_i64(x, width),
            bottom: saturated_add_i64(y, height),
        }
    }

    #[inline]
    pub const fn make_wh(width: i64, height: i64) -> Self {
        Self::from_ltrb(0, 0, width, height)
    }

    #[inline]
    pub const fn make_size(size: ISize) -> Self {
        Self::from_ltrb(0, 0, size.width, size.height)
    }

    #[inline]
    pub const fn make_circle_bounds(center: IPoint, radius: i64) -> Self {
        Self::from_ltrb(
            saturated_sub_i64(center.x, radius),
            saturated_sub_i64(center.y, radius),
            saturated_add_i64(center.x, radius),
            saturated_add_i64(center.y, radius),
        )
    }

    #[inline]
    pub const fn make_maximum() -> Self {
        Self::from_ltrb(i64::MIN, i64::MIN, i64::MAX, i64::MAX)
    }

    #[inline]
    pub const fn get_left(&self) -> i64 {
        self.left
    }

    #[inline]
    pub const fn get_top(&self) -> i64 {
        self.top
    }

    #[inline]
    pub const fn get_right(&self) -> i64 {
        self.right
    }

    #[inline]
    pub const fn get_bottom(&self) -> i64 {
        self.bottom
    }

    #[inline]
    pub const fn get_x(&self) -> i64 {
        self.left
    }

    #[inline]
    pub const fn get_y(&self) -> i64 {
        self.top
    }

    #[inline]
    pub const fn get_width(&self) -> i64 {
        saturated_sub_i64(self.right, self.left)
    }

    #[inline]
    pub const fn get_height(&self) -> i64 {
        saturated_sub_i64(self.bottom, self.top)
    }

    #[inline]
    pub const fn get_origin(&self) -> IPoint {
        IPoint::new(self.left, self.top)
    }

    #[inline]
    pub const fn get_size(&self) -> ISize {
        ISize::new(self.get_width(), self.get_height())
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        !(self.left < self.right && self.top < self.bottom)
    }

    #[inline]
    pub const fn contains(&self, p: IPoint) -> bool {
        !self.is_empty()
            && p.x >= self.left
            && p.y >= self.top
            && p.x < self.right
            && p.y < self.bottom
    }

    #[inline]
    pub fn union(&self, o: Self) -> Self {
        if self.is_empty() {
            return o;
        }
        if o.is_empty() {
            return *self;
        }
        Self::from_ltrb(
            self.left.min(o.left),
            self.top.min(o.top),
            self.right.max(o.right),
            self.bottom.max(o.bottom),
        )
    }

    #[inline]
    pub const fn intersects_with_rect(&self, o: Self) -> bool {
        !self.is_empty()
            && !o.is_empty()
            && self.left < o.right
            && self.top < o.bottom
            && self.right > o.left
            && self.bottom > o.top
    }

    #[inline]
    pub fn intersection(&self, o: Self) -> Option<Self> {
        if self.intersects_with_rect(o) {
            Some(Self::from_ltrb(
                self.left.max(o.left),
                self.top.max(o.top),
                self.right.min(o.right),
                self.bottom.min(o.bottom),
            ))
        } else {
            None
        }
    }
}
