// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hex lattice geometry.
//!
//! Hexes are addressed by axial coordinates `(q, r)`. Hex centers sit at
//! `q·a₀ + r·a₁` with `a₀ = (1, 0)` and `a₁ = (½, √3⁄2)`, so neighboring centers are
//! exactly one unit apart (pointy-top hexes).

/// √3, exact to f64 precision.
pub const SQRT3: f64 = 1.732_050_807_568_877_2;

/// The direct lattice basis `[a₀, a₁]` (columns), as `[[a₀x, a₁x], [a₀y, a₁y]]`.
pub const BASIS: [[f64; 2]; 2] = [[1.0, 0.5], [0.0, SQRT3 / 2.0]];

/// Area of one hex cell for unit center spacing.
pub const CELL_AREA: f64 = SQRT3 / 2.0;

/// Cartesian position of the hex center (or any fractional axial point).
#[inline]
pub const fn to_cartesian(q: f64, r: f64) -> (f64, f64) {
    (q + 0.5 * r, (SQRT3 / 2.0) * r)
}

/// Fractional axial coordinates of a cartesian point.
#[inline]
pub const fn to_axial(x: f64, y: f64) -> (f64, f64) {
    let r = y * (2.0 / SQRT3);
    (x - 0.5 * r, r)
}

/// Squared cartesian length of an axial offset: `dq² + dq·dr + dr²`.
#[inline]
pub const fn len2(dq: f64, dr: f64) -> f64 {
    dq * dq + dq * dr + dr * dr
}

/// Rounds fractional axial coordinates to the hex that contains the point.
pub fn round(q: f64, r: f64) -> (i64, i64) {
    let s = -q - r;
    let (mut rq, mut rr, rs) = (q.round(), r.round(), s.round());
    let (dq, dr, ds) = ((rq - q).abs(), (rr - r).abs(), (rs - s).abs());
    if dq > dr && dq > ds {
        rq = -rr - rs;
    } else if dr > ds {
        rr = -rq - rs;
    }
    (rq as i64, rr as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_rounding() {
        for q in -5..5 {
            for r in -5..5 {
                let (x, y) = to_cartesian(q as f64, r as f64);
                // Points within the inscribed circle (radius 1/2) round to the center.
                for k in 0..12 {
                    let a = k as f64 * 0.523;
                    let (fq, fr) = to_axial(x + 0.49 * a.cos(), y + 0.49 * a.sin());
                    assert_eq!(round(fq, fr), (q, r));
                }
            }
        }
        assert!((len2(1.0, -1.0) - 1.0).abs() < 1e-12);
    }
}
