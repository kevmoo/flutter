use crate::point::Point;
use crate::scalar::Scalar;
use crate::size::Size;
use bytemuck::{Pod, Zeroable};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vector3 {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
}

impl Vector3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub const fn new(x: Scalar, y: Scalar, z: Scalar) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn splat(v: Scalar) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline]
    pub const fn to_array(self) -> [Scalar; 3] {
        [self.x, self.y, self.z]
    }

    #[inline]
    pub const fn from_array(a: [Scalar; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }

    #[inline]
    pub fn get_length(&self) -> Scalar {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        let len = self.get_length();
        Self::new(self.x / len, self.y / len, self.z / len)
    }

    #[inline]
    pub fn dot(&self, other: Self) -> Scalar {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn abs(&self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    #[inline]
    pub fn cross(&self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    #[inline]
    pub fn min(&self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    #[inline]
    pub fn max(&self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    #[inline]
    pub fn floor(&self) -> Self {
        Self::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    #[inline]
    pub fn ceil(&self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    #[inline]
    pub fn round(&self) -> Self {
        Self::new(self.x.round(), self.y.round(), self.z.round())
    }

    #[inline]
    pub fn lerp(&self, other: Self, t: Scalar) -> Self {
        *self + (other - *self) * t
    }

    #[inline]
    pub fn combine(a: Self, a_scale: Scalar, b: Self, b_scale: Scalar) -> Self {
        Self::new(
            a_scale * a.x + b_scale * b.x,
            a_scale * a.y + b_scale * b.y,
            a_scale * a.z + b_scale * b.z,
        )
    }
}

impl From<Point> for Vector3 {
    #[inline]
    fn from(p: Point) -> Self {
        Self::new(p.x, p.y, 0.0)
    }
}

impl From<Size> for Vector3 {
    #[inline]
    fn from(s: Size) -> Self {
        Self::new(s.width, s.height, 0.0)
    }
}

impl From<[Scalar; 3]> for Vector3 {
    #[inline]
    fn from(a: [Scalar; 3]) -> Self {
        Self::from_array(a)
    }
}

impl From<Vector3> for [Scalar; 3] {
    #[inline]
    fn from(v: Vector3) -> Self {
        v.to_array()
    }
}

impl Neg for Vector3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Add for Vector3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<Scalar> for Vector3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Scalar) -> Self {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl Sub for Vector3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Sub<Scalar> for Vector3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Scalar) -> Self {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl Mul for Vector3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Mul<Scalar> for Vector3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Scalar) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vector3> for Scalar {
    type Output = Vector3;
    #[inline]
    fn mul(self, rhs: Vector3) -> Vector3 {
        rhs * self
    }
}

impl Div for Vector3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl Div<Scalar> for Vector3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Scalar) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl AddAssign for Vector3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl SubAssign for Vector3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl MulAssign<Scalar> for Vector3 {
    #[inline]
    fn mul_assign(&mut self, rhs: Scalar) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl DivAssign<Scalar> for Vector3 {
    #[inline]
    fn div_assign(&mut self, rhs: Scalar) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Vector4 {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
    pub w: Scalar,
}

impl Default for Vector4 {
    #[inline]
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

impl Vector4 {
    #[inline]
    pub const fn new(x: Scalar, y: Scalar, z: Scalar, w: Scalar) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub const fn to_array(self) -> [Scalar; 4] {
        [self.x, self.y, self.z, self.w]
    }

    #[inline]
    pub const fn from_array(a: [Scalar; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        let inverse = 1.0 / (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        Self::new(
            self.x * inverse,
            self.y * inverse,
            self.z * inverse,
            self.w * inverse,
        )
    }

    #[inline]
    pub fn min(&self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
            self.w.min(other.w),
        )
    }

    #[inline]
    pub fn max(&self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
            self.w.max(other.w),
        )
    }

    #[inline]
    pub fn floor(&self) -> Self {
        Self::new(self.x.floor(), self.y.floor(), self.z.floor(), self.w.floor())
    }

    #[inline]
    pub fn ceil(&self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil(), self.w.ceil())
    }

    #[inline]
    pub fn round(&self) -> Self {
        Self::new(self.x.round(), self.y.round(), self.z.round(), self.w.round())
    }

    #[inline]
    pub fn lerp(&self, other: Self, t: Scalar) -> Self {
        *self + (other - *self) * t
    }

    #[inline]
    pub const fn xy(&self) -> Point {
        Point::new(self.x, self.y)
    }
}

impl From<Vector3> for Vector4 {
    #[inline]
    fn from(v: Vector3) -> Self {
        Self::new(v.x, v.y, v.z, 1.0)
    }
}

impl From<Point> for Vector4 {
    #[inline]
    fn from(p: Point) -> Self {
        Self::new(p.x, p.y, 0.0, 1.0)
    }
}

impl From<[Scalar; 4]> for Vector4 {
    #[inline]
    fn from(a: [Scalar; 4]) -> Self {
        Self::from_array(a)
    }
}

impl From<Vector4> for [Scalar; 4] {
    #[inline]
    fn from(v: Vector4) -> Self {
        v.to_array()
    }
}

impl Add for Vector4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w)
    }
}

impl Sub for Vector4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)
    }
}

impl Mul<Scalar> for Vector4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Scalar) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl Mul for Vector4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z, self.w * rhs.w)
    }
}
