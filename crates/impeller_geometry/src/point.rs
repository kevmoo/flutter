use crate::scalar::{Radians, Scalar};
use crate::size::{ISize, Size};
use bytemuck::{Pod, Zeroable};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Point {
    pub x: Scalar,
    pub y: Scalar,
}

pub type Vector2 = Point;
pub type Quad = [Point; 4];

impl Point {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    #[inline]
    pub const fn new(x: Scalar, y: Scalar) -> Self {
        Self { x, y }
    }

    #[inline]
    pub const fn make_xy(x: Scalar, y: Scalar) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn get_length(&self) -> Scalar {
        self.get_length_squared().sqrt()
    }

    #[inline]
    pub fn get_length_squared(&self) -> Scalar {
        ((self.x as f64) * (self.x as f64) + (self.y as f64) * (self.y as f64)) as Scalar
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        let len = self.get_length();
        if len <= Scalar::EPSILON {
            Self::ZERO
        } else {
            Self::new(self.x / len, self.y / len)
        }
    }

    #[inline]
    pub fn dot(&self, other: Self) -> Scalar {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn cross(&self, other: Self) -> Scalar {
        self.x * other.y - self.y * other.x
    }

    #[inline]
    pub fn cross_three_points(p0: Self, p1: Self, p2: Self) -> Scalar {
        (p1 - p0).cross(p2 - p0)
    }

    #[inline]
    pub fn lerp(self, other: Self, t: Scalar) -> Self {
        Self::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )
    }

    #[inline]
    pub fn get_distance_squared(&self, other: Self) -> Scalar {
        let dx = (other.x - self.x) as f64;
        let dy = (other.y - self.y) as f64;
        (dx * dx + dy * dy) as Scalar
    }

    #[inline]
    pub fn get_distance(&self, other: Self) -> Scalar {
        self.get_distance_squared(other).sqrt()
    }

    #[inline]
    pub fn get_distance_to_segment_squared(&self, p0: Self, p1: Self) -> Scalar {
        let u = p1 - p0;
        let v = *self - p0;

        let dot = u.dot(v);
        if dot <= 0.0 {
            return v.get_length_squared();
        }

        let u_length_squared = u.get_length_squared();
        if dot >= u_length_squared {
            return self.get_distance_squared(p1);
        }

        let cross = u.cross(v);
        (cross / u_length_squared) * cross
    }

    #[inline]
    pub fn get_distance_to_segment(&self, p0: Self, p1: Self) -> Scalar {
        self.get_distance_to_segment_squared(p0, p1).sqrt()
    }

    #[inline]
    pub fn min(&self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    #[inline]
    pub fn max(&self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }

    #[inline]
    pub fn floor(&self) -> Self {
        Self::new(self.x.floor(), self.y.floor())
    }

    #[inline]
    pub fn ceil(&self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil())
    }

    #[inline]
    pub fn round(&self) -> Self {
        Self::new(self.x.round(), self.y.round())
    }

    #[inline]
    pub fn abs(&self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }

    #[inline]
    pub fn reflect(&self, axis: Self) -> Self {
        *self - axis * (self.dot(axis) * 2.0)
    }

    #[inline]
    pub fn rotate(&self, angle: Radians) -> Self {
        let (sin_a, cos_a) = angle.0.sin_cos();
        Self::new(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )
    }

    #[inline]
    pub const fn perpendicular_right(&self) -> Self {
        Self::new(-self.y, self.x)
    }

    #[inline]
    pub const fn perpendicular_left(&self) -> Self {
        Self::new(self.y, -self.x)
    }

    #[inline]
    pub fn angle_to(&self, other: Self) -> Radians {
        Radians(self.cross(other).atan2(self.dot(other)))
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

impl From<Size> for Point {
    #[inline]
    fn from(s: Size) -> Self {
        Self::new(s.width, s.height)
    }
}

impl Add for Point {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Add<Size> for Point {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Size) -> Self {
        Self::new(self.x + rhs.width, self.y + rhs.height)
    }
}

impl Sub for Point {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Sub<Size> for Point {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Size) -> Self {
        Self::new(self.x - rhs.width, self.y - rhs.height)
    }
}

impl Mul<Scalar> for Point {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Scalar) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Point> for Scalar {
    type Output = Point;
    #[inline]
    fn mul(self, rhs: Point) -> Point {
        Point::new(self * rhs.x, self * rhs.y)
    }
}

impl Mul for Point {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl Mul<Size> for Point {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Size) -> Self {
        Self::new(self.x * rhs.width, self.y * rhs.height)
    }
}

impl Div<Scalar> for Point {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Scalar) -> Self {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl Div<Point> for Scalar {
    type Output = Point;
    #[inline]
    fn div(self, rhs: Point) -> Point {
        Point::new(self / rhs.x, self / rhs.y)
    }
}

impl Div for Point {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::new(self.x / rhs.x, self.y / rhs.y)
    }
}

impl Div<Size> for Point {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Size) -> Self {
        Self::new(self.x / rhs.width, self.y / rhs.height)
    }
}

impl Neg for Point {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

impl AddAssign for Point {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl SubAssign for Point {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl MulAssign<Scalar> for Point {
    #[inline]
    fn mul_assign(&mut self, rhs: Scalar) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl DivAssign<Scalar> for Point {
    #[inline]
    fn div_assign(&mut self, rhs: Scalar) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct IPoint {
    pub x: i64,
    pub y: i64,
}

impl IPoint {
    pub const ZERO: Self = Self { x: 0, y: 0 };

    #[inline]
    pub const fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }
}

impl Add for IPoint {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Add<ISize> for IPoint {
    type Output = Self;
    #[inline]
    fn add(self, rhs: ISize) -> Self {
        Self::new(self.x + rhs.width, self.y + rhs.height)
    }
}

impl Sub for IPoint {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Sub<ISize> for IPoint {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: ISize) -> Self {
        Self::new(self.x - rhs.width, self.y - rhs.height)
    }
}

impl Mul for IPoint {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl Mul<i64> for IPoint {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: i64) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Div for IPoint {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::new(self.x / rhs.x, self.y / rhs.y)
    }
}

impl Div<i64> for IPoint {
    type Output = Self;
    #[inline]
    fn div(self, rhs: i64) -> Self {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl Neg for IPoint {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}
