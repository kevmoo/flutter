use crate::scalar::Scalar;
use crate::vector::Vector3;
use bytemuck::{Pod, Zeroable};
use std::ops::{Add, Mul, Sub};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Quaternion {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
    pub w: Scalar,
}

impl Quaternion {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    #[inline]
    pub const fn new(x: Scalar, y: Scalar, z: Scalar, w: Scalar) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn from_axis_angle(axis: Vector3, angle: Scalar) -> Self {
        let (sine, cosine) = (angle * 0.5).sin_cos();
        Self {
            x: sine * axis.x,
            y: sine * axis.y,
            z: sine * axis.z,
            w: cosine,
        }
    }

    #[inline]
    pub fn dot(&self, other: Self) -> Scalar {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    #[inline]
    pub fn length(&self) -> Scalar {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len == 0.0 {
            Self::IDENTITY
        } else {
            let m = 1.0 / len;
            Self::new(self.x * m, self.y * m, self.z * m, self.w * m)
        }
    }

    #[inline]
    pub const fn invert(&self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    #[inline]
    pub fn slerp(&self, to: Self, time: Scalar) -> Self {
        let t = time as f64;
        let cosine = self.dot(to) as f64;
        if cosine.abs() < 1.0 - 1.0e-3 {
            let sine = (1.0 - cosine * cosine).sqrt();
            let angle = sine.atan2(cosine);
            let sine_inverse = 1.0 / sine;
            let c0 = (((1.0 - t) * angle).sin() * sine_inverse) as Scalar;
            let c1 = ((t * angle).sin() * sine_inverse) as Scalar;
            *self * c0 + to * c1
        } else {
            (*self * (1.0 - time) + to * time).normalize()
        }
    }
}

impl Mul for Quaternion {
    type Output = Self;

    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y + self.y * o.w + self.z * o.x - self.x * o.z,
            self.w * o.z + self.z * o.w + self.x * o.y - self.y * o.x,
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
        )
    }
}

impl Mul<Scalar> for Quaternion {
    type Output = Self;

    #[inline]
    fn mul(self, scale: Scalar) -> Self {
        Self::new(
            scale * self.x,
            scale * self.y,
            scale * self.z,
            scale * self.w,
        )
    }
}

impl Mul<Quaternion> for Scalar {
    type Output = Quaternion;

    #[inline]
    fn mul(self, q: Quaternion) -> Quaternion {
        q * self
    }
}

impl Mul<Vector3> for Quaternion {
    type Output = Vector3;

    #[inline]
    fn mul(self, vector: Vector3) -> Vector3 {
        let v = Vector3::new(self.x, self.y, self.z);
        v * (v.dot(vector) * 2.0)
            + vector * (self.w * self.w - v.dot(v))
            + v.cross(vector) * (2.0 * self.w)
    }
}

impl Add for Quaternion {
    type Output = Self;

    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z, self.w + o.w)
    }
}

impl Sub for Quaternion {
    type Output = Self;

    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z, self.w - o.w)
    }
}
