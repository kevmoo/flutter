use bytemuck::{Pod, Zeroable};
use std::ops::{Add, Neg, Sub};

pub type Scalar = f32;

pub const K_E: Scalar = std::f32::consts::E;
pub const K_LOG2_E: Scalar = std::f32::consts::LOG2_E;
pub const K_LOG10_E: Scalar = std::f32::consts::LOG10_E;
pub const K_LOG_E2: Scalar = std::f32::consts::LN_2;
pub const K_LOG_E10: Scalar = std::f32::consts::LN_10;
pub const K_PI: Scalar = std::f32::consts::PI;
pub const K_2_PI: Scalar = std::f32::consts::TAU;
pub const K_PI_OVER_2: Scalar = std::f32::consts::FRAC_PI_2;
pub const K_PI_OVER_4: Scalar = std::f32::consts::FRAC_PI_4;
pub const K_1_OVER_PI: Scalar = std::f32::consts::FRAC_1_PI;
pub const K_2_OVER_PI: Scalar = std::f32::consts::FRAC_2_PI;
pub const K_2_OVER_SQRT_PI: Scalar = std::f32::consts::FRAC_2_SQRT_PI;
pub const K_SQRT_2: Scalar = std::f32::consts::SQRT_2;
pub const K_1_OVER_SQRT_2: Scalar = std::f32::consts::FRAC_1_SQRT_2;
pub const K_SQRT_2_OVER_2: Scalar = std::f32::consts::FRAC_1_SQRT_2;
pub const K_PHI: Scalar = (1.0 + 2.236_068) * 0.5;
pub const K_EH_CLOSE_ENOUGH: Scalar = 1.0e-3;

#[inline]
pub fn scalar_nearly_zero(x: Scalar, tolerance: Scalar) -> bool {
    x.abs() <= tolerance
}

#[inline]
pub fn scalar_nearly_equal(x: Scalar, y: Scalar, tolerance: Scalar) -> bool {
    scalar_nearly_zero(x - y, tolerance)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd, Pod, Zeroable)]
pub struct Radians(pub Scalar);

impl Radians {
    #[inline]
    pub const fn new(radians: Scalar) -> Self {
        Self(radians)
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

impl Neg for Radians {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Add for Radians {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Radians {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd, Pod, Zeroable)]
pub struct Degrees(pub Scalar);

impl Degrees {
    #[inline]
    pub const fn new(degrees: Scalar) -> Self {
        Self(degrees)
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    #[inline]
    pub fn get_positive(self) -> Self {
        let mut deg = self.0 % 360.0;
        if deg < 0.0 {
            deg += 360.0;
        }
        Self(deg)
    }
}

impl Neg for Degrees {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Add for Degrees {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Degrees {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl From<Degrees> for Radians {
    #[inline]
    fn from(d: Degrees) -> Self {
        Radians(d.0.to_radians())
    }
}

impl From<Radians> for Degrees {
    #[inline]
    fn from(r: Radians) -> Self {
        Degrees(r.0.to_degrees())
    }
}
