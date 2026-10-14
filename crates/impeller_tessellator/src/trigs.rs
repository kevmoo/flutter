use impeller_geometry::{Point, Scalar, Size};
use std::ops::Neg;

/// Pixel tolerance used by Impeller to determine circle/arc divisions.
pub const K_CIRCLE_TOLERANCE: Scalar = 0.1;

const K_PRECOMPUTED_DIVISION_COUNT: usize = 1024;

#[rustfmt::skip]
const K_PRECOMPUTED_DIVISIONS: [usize; K_PRECOMPUTED_DIVISION_COUNT] = [
     1,  2,  3,  4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,  7,
     8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10,
    10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13,
    13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25,
    25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26,
    26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28,
    28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
    31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
    33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35,
    35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36,
    36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
    36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38,
    38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
    38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
    39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
    45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
    45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
    46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47,
    47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
    47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49,
    49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49,
    49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51,
    51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
    51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52,
    52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
    52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53,
    53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53,
    53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54,
    54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
    54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
    54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
    55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
    55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57,
];

/// Computes the number of equal angular divisions per quarter-circle needed to
/// approximate a circle of `pixel_radius` within `K_CIRCLE_TOLERANCE`.
///
/// Ported from `ComputeQuadrantDivisions` in `tessellator.cc`.
pub fn compute_quadrant_divisions(pixel_radius: Scalar) -> usize {
    if pixel_radius <= 0.0 || !pixel_radius.is_finite() {
        return 1;
    }
    let radius_index = pixel_radius.ceil() as usize;
    if radius_index < K_PRECOMPUTED_DIVISION_COUNT {
        return K_PRECOMPUTED_DIVISIONS[radius_index];
    }

    let k = (K_CIRCLE_TOLERANCE as f64) / (pixel_radius as f64);
    let acos_val = (1.0 - k).clamp(-1.0, 1.0).acos();
    if acos_val <= f64::EPSILON {
        return 1;
    }
    (std::f64::consts::FRAC_PI_4 / acos_val).ceil() as usize
}

/// Precomputed cosine and sine pair for an angle, matching `impeller::Trig`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Trig {
    pub cos: f64,
    pub sin: f64,
}

impl Trig {
    #[inline]
    pub const fn new(cos: f64, sin: f64) -> Self {
        Self { cos, sin }
    }

    #[inline]
    pub fn from_radians(radians: f64) -> Self {
        Self {
            cos: radians.cos(),
            sin: radians.sin(),
        }
    }

    /// Rotates `vector` by the angle represented by this `Trig` (`Trig::operator*(Vector2)`).
    #[inline]
    pub fn rotate_vector(self, vector: Point) -> Point {
        Point::new(
            (vector.x as f64 * self.cos - vector.y as f64 * self.sin) as Scalar,
            (vector.x as f64 * self.sin + vector.y as f64 * self.cos) as Scalar,
        )
    }

    /// Scales unit circle point by `radius` (`Trig::operator*(double)`).
    #[inline]
    pub fn scale_radius(self, radius: Scalar) -> Point {
        Point::new(
            (self.cos * radius as f64) as Scalar,
            (self.sin * radius as f64) as Scalar,
        )
    }

    /// Scales unit circle point by `radii` (`Trig::operator*(Size)`).
    #[inline]
    pub fn scale_radii(self, radii: Size) -> Point {
        Point::new(
            (self.cos * radii.width as f64) as Scalar,
            (self.sin * radii.height as f64) as Scalar,
        )
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

/// Quarter-circle sample table matching `Tessellator::Trigs` in `tessellator.h`.
/// Always contains `divisions + 1 >= 2` entries from `(1, 0)` at 0 deg to `(0, 1)` at 90 deg.
#[derive(Clone, Debug, PartialEq)]
pub struct Trigs {
    entries: Vec<Trig>,
}

impl Trigs {
    pub fn from_divisions(divisions: usize) -> Self {
        let divisions = divisions.max(1);
        let mut entries = Vec::with_capacity(divisions + 1);
        let angle_scale = std::f64::consts::FRAC_PI_2 / (divisions as f64);
        entries.push(Trig::new(1.0, 0.0));
        for i in 1..divisions {
            entries.push(Trig::from_radians((i as f64) * angle_scale));
        }
        entries.push(Trig::new(0.0, 1.0));
        Self { entries }
    }

    pub fn from_pixel_radius(pixel_radius: Scalar) -> Self {
        Self::from_divisions(compute_quadrant_divisions(pixel_radius))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[inline]
    pub fn steps(&self) -> usize {
        self.entries.len().saturating_sub(1)
    }

    #[inline]
    pub fn as_slice(&self) -> &[Trig] {
        &self.entries
    }
}

impl std::ops::Index<usize> for Trigs {
    type Output = Trig;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}
