use crate::scalar::Scalar;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Shear {
    pub xy: Scalar,
    pub xz: Scalar,
    pub yz: Scalar,
}

impl Shear {
    pub const ZERO: Self = Self {
        xy: 0.0,
        xz: 0.0,
        yz: 0.0,
    };

    #[inline]
    pub const fn new(xy: Scalar, xz: Scalar, yz: Scalar) -> Self {
        Self { xy, xz, yz }
    }

    #[inline]
    pub const fn to_array(self) -> [Scalar; 3] {
        [self.xy, self.xz, self.yz]
    }

    #[inline]
    pub const fn from_array(a: [Scalar; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
}

impl From<[Scalar; 3]> for Shear {
    #[inline]
    fn from(a: [Scalar; 3]) -> Self {
        Self::from_array(a)
    }
}

impl From<Shear> for [Scalar; 3] {
    #[inline]
    fn from(s: Shear) -> Self {
        s.to_array()
    }
}
