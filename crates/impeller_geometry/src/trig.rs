use crate::point::Point;
use crate::scalar::{Degrees, Radians, Scalar};
use crate::size::Size;
use std::ops::{Mul, Neg};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Trig {
    pub cos: f64,
    pub sin: f64,
}

impl Default for Trig {
    #[inline]
    fn default() -> Self {
        Self { cos: 1.0, sin: 0.0 }
    }
}

impl Trig {
    #[inline]
    pub fn new(r: Radians) -> Self {
        Self {
            cos: (r.0 as f64).cos(),
            sin: (r.0 as f64).sin(),
        }
    }

    #[inline]
    pub fn from_degrees(d: Degrees) -> Self {
        Self::new(Radians::from(d))
    }

    #[inline]
    pub const fn from_cos_sin(cos: f64, sin: f64) -> Self {
        Self { cos, sin }
    }
}

impl From<Radians> for Trig {
    #[inline]
    fn from(r: Radians) -> Self {
        Self::new(r)
    }
}

impl From<Degrees> for Trig {
    #[inline]
    fn from(d: Degrees) -> Self {
        Self::from_degrees(d)
    }
}

impl Neg for Trig {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            cos: self.cos,
            sin: -self.sin,
        }
    }
}

impl Mul<Point> for Trig {
    type Output = Point;

    #[inline]
    fn mul(self, vector: Point) -> Point {
        Point::new(
            (vector.x as f64 * self.cos - vector.y as f64 * self.sin) as Scalar,
            (vector.x as f64 * self.sin + vector.y as f64 * self.cos) as Scalar,
        )
    }
}

impl Mul<Scalar> for Trig {
    type Output = Point;

    #[inline]
    fn mul(self, radius: Scalar) -> Point {
        Point::new(
            (self.cos * radius as f64) as Scalar,
            (self.sin * radius as f64) as Scalar,
        )
    }
}

impl Mul<f64> for Trig {
    type Output = Point;

    #[inline]
    fn mul(self, radius: f64) -> Point {
        Point::new((self.cos * radius) as Scalar, (self.sin * radius) as Scalar)
    }
}

impl Mul<Size> for Trig {
    type Output = Point;

    #[inline]
    fn mul(self, ellipse_radii: Size) -> Point {
        Point::new(
            (self.cos * ellipse_radii.width as f64) as Scalar,
            (self.sin * ellipse_radii.height as f64) as Scalar,
        )
    }
}
