use crate::scalar::Scalar;
use bytemuck::{Pod, Zeroable};
use std::ops::{Add, Div, Mul, MulAssign, Neg, Sub};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Size {
    pub width: Scalar,
    pub height: Scalar,
}

impl Size {
    pub const ZERO: Self = Self {
        width: 0.0,
        height: 0.0,
    };

    #[inline]
    pub const fn new(width: Scalar, height: Scalar) -> Self {
        Self { width, height }
    }

    #[inline]
    pub const fn make_wh(width: Scalar, height: Scalar) -> Self {
        Self { width, height }
    }

    #[inline]
    pub const fn square(dimension: Scalar) -> Self {
        Self {
            width: dimension,
            height: dimension,
        }
    }

    #[inline]
    pub const fn infinite() -> Self {
        Self {
            width: Scalar::MAX,
            height: Scalar::MAX,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        !(self.width > 0.0 && self.height > 0.0)
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.width.is_finite() && self.height.is_finite()
    }

    #[inline]
    pub fn is_square(&self) -> bool {
        self.width == self.height
    }

    #[inline]
    pub fn min(&self, other: Self) -> Self {
        Self::new(self.width.min(other.width), self.height.min(other.height))
    }

    #[inline]
    pub fn max(&self, other: Self) -> Self {
        Self::new(self.width.max(other.width), self.height.max(other.height))
    }

    #[inline]
    pub fn min_dimension(&self) -> Scalar {
        self.width.min(self.height)
    }

    #[inline]
    pub fn max_dimension(&self) -> Scalar {
        self.width.max(self.height)
    }

    #[inline]
    pub fn abs(&self) -> Self {
        Self::new(self.width.abs(), self.height.abs())
    }

    #[inline]
    pub fn floor(&self) -> Self {
        Self::new(self.width.floor(), self.height.floor())
    }

    #[inline]
    pub fn ceil(&self) -> Self {
        Self::new(self.width.ceil(), self.height.ceil())
    }

    #[inline]
    pub fn round(&self) -> Self {
        Self::new(self.width.round(), self.height.round())
    }

    #[inline]
    pub fn area(&self) -> Scalar {
        self.width * self.height
    }

    #[inline]
    pub fn mip_count(&self) -> usize {
        if self.is_empty() || self.width <= 0.0 || self.height <= 0.0 {
            return 1;
        }
        let result = (self.width.log2().min(self.height.log2())) as usize;
        result.max(1)
    }
}

impl Add for Size {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.width + rhs.width, self.height + rhs.height)
    }
}

impl Sub for Size {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.width - rhs.width, self.height - rhs.height)
    }
}

impl Neg for Size {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.width, -self.height)
    }
}

impl Mul<Scalar> for Size {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Scalar) -> Self {
        Self::new(self.width * rhs, self.height * rhs)
    }
}

impl Mul<Size> for Scalar {
    type Output = Size;
    #[inline]
    fn mul(self, rhs: Size) -> Size {
        rhs * self
    }
}

impl MulAssign<Scalar> for Size {
    #[inline]
    fn mul_assign(&mut self, rhs: Scalar) {
        self.width *= rhs;
        self.height *= rhs;
    }
}

impl Div<Scalar> for Size {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Scalar) -> Self {
        Self::new(self.width / rhs, self.height / rhs)
    }
}

impl Div for Size {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::new(self.width / rhs.width, self.height / rhs.height)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct ISize {
    pub width: i64,
    pub height: i64,
}

impl ISize {
    pub const ZERO: Self = Self {
        width: 0,
        height: 0,
    };

    #[inline]
    pub const fn new(width: i64, height: i64) -> Self {
        Self { width, height }
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        !(self.width > 0 && self.height > 0)
    }

    #[inline]
    pub const fn is_square(&self) -> bool {
        self.width == self.height
    }

    #[inline]
    pub fn min(&self, other: Self) -> Self {
        Self::new(self.width.min(other.width), self.height.min(other.height))
    }

    #[inline]
    pub fn max(&self, other: Self) -> Self {
        Self::new(self.width.max(other.width), self.height.max(other.height))
    }

    #[inline]
    pub fn min_dimension(&self) -> i64 {
        self.width.min(self.height)
    }

    #[inline]
    pub fn max_dimension(&self) -> i64 {
        self.width.max(self.height)
    }

    #[inline]
    pub const fn area(&self) -> i64 {
        self.width * self.height
    }

    #[inline]
    pub fn mip_count(&self) -> usize {
        if self.is_empty() || self.width <= 0 || self.height <= 0 {
            return 1;
        }
        let w = (self.width as f64).log2();
        let h = (self.height as f64).log2();
        let result = w.min(h) as usize;
        result.max(1)
    }
}

impl Add for ISize {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.width + rhs.width, self.height + rhs.height)
    }
}

impl Sub for ISize {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.width - rhs.width, self.height - rhs.height)
    }
}

impl Neg for ISize {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.width, -self.height)
    }
}
