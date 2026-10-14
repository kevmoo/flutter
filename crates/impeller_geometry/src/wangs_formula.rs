use crate::point::Point;
use crate::scalar::Scalar;

const K_PRECISION: Scalar = 4.0;

/// Returns the minimum number of evenly spaced (in the parametric sense) line
/// segments that the cubic must be chopped into in order to guarantee all lines
/// stay within a distance of "1/precision" pixels from the true curve.
#[inline]
pub fn compute_cubic_subdivisions(
    scale_factor: Scalar,
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
) -> Scalar {
    let k = scale_factor * 0.75 * K_PRECISION;
    let a = p0 - p1 * 2.0 + p2;
    let b = p1 - p2 * 2.0 + p3;
    let max_len_sq = a.get_length_squared().max(b.get_length_squared());
    (k * max_len_sq.sqrt()).sqrt()
}

/// Returns the minimum number of evenly spaced (in the parametric sense) line
/// segments that the quadratic must be chopped into in order to guarantee all
/// lines stay within a distance of "1/precision" pixels from the true curve.
#[inline]
pub fn compute_quadratic_subdivisions(
    scale_factor: Scalar,
    p0: Point,
    p1: Point,
    p2: Point,
) -> Scalar {
    let k = scale_factor * 0.25 * K_PRECISION;
    (k * (p0 - p1 * 2.0 + p2).get_length()).sqrt()
}

/// Alias for [`compute_quadratic_subdivisions`] matching `ComputeQuadratureSubdivisions`.
#[inline]
pub fn compute_quadrature_subdivisions(
    scale_factor: Scalar,
    p0: Point,
    p1: Point,
    p2: Point,
) -> Scalar {
    compute_quadratic_subdivisions(scale_factor, p0, p1, p2)
}

/// Returns Wang's formula specialized for a conic curve (Zheng-Sederberg).
pub fn compute_conic_subdivisions(
    scale_factor: Scalar,
    mut p0: Point,
    mut p1: Point,
    mut p2: Point,
    w: Scalar,
) -> Scalar {
    let c = (p0.min(p1).min(p2) + p0.max(p1).max(p2)) * 0.5;
    p0 -= c;
    p1 -= c;
    p2 -= c;

    let max_len = p0.dot(p0).max(p1.dot(p1).max(p2.dot(p2))).sqrt();

    let dp = p1 * (-2.0 * w) + p0 + p2;
    let dw = (-2.0 * w + 2.0).abs();

    let k = scale_factor * K_PRECISION;
    let rp_minus_1 = (max_len * k - 1.0).max(0.0);
    let numer = dp.dot(dp).sqrt() * k + rp_minus_1 * dw;
    let denom = 4.0 * w.min(1.0);

    (numer / denom).sqrt()
}
