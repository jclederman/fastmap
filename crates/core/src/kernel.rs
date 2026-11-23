// SPDX-License-Identifier: AGPL-3.0-or-later
//! Kernel mathematics: the power law, the scale-split filter, tapers and the fine kernel.
//!
//! The target field has the spectrum of the power law `k∞(r) = −r^α/α` (`−log r` when α = 0),
//! with `α = β − 2`, high-passed by the smooth filter [`split`] at the scale `σ_R = 1.15·R`:
//!
//! ```text
//! amplitude(κ) ∝ κ^(−β) · F(κ σ_R),     F(x) = 1 − e^{−x²/2}(1 + x²/2)
//! ```
//!
//! `F` vanishes as κ⁴ and tends to 1 above `κ ≈ 2/σ`. It puts the half-power point at the same
//! wavelength as a hard cutoff at `R` would, so `R` is the largest feature size. The field is
//! the sum of a fine level and three coarse levels (see [`crate::levels`]). This module builds
//! the fine kernel: the power law high-passed at `σ₀ ≈ 10` hexes and smoothly truncated within
//! [`crate::tile::MAX_RADIUS`].

use crate::fft::Fft2;
use crate::hex::SQRT3;
use crate::tile::WINDOW;
use rustfft::num_complex::Complex32;

/// Generation parameters that affect the elevation field (not just its display).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Spectral exponent β: the amplitude spectrum falls off as `|k|^(−β)`.
    /// Larger values give smoother terrain. Valid range: [`Params::BETA_RANGE`].
    pub beta: f32,
    /// Largest feature size in hexes. Valid range: [`Params::RADIUS_RANGE`]. The upper limit
    /// leaves the outermost coarse level room for the spectrum's roll-off below `1/σ_R`.
    pub radius: f32,
}

impl Params {
    pub const BETA_RANGE: (f32, f32) = (0.5, 3.0);
    pub const RADIUS_RANGE: (f32, f32) = (4.0, 2500.0);
    pub const DEFAULT: Params = Params { beta: 1.5, radius: 1000.0 };

    /// Clamps both parameters into their valid ranges.
    pub fn clamped(self) -> Self {
        Self {
            beta: self.beta.clamp(Self::BETA_RANGE.0, Self::BETA_RANGE.1),
            radius: self.radius.clamp(Self::RADIUS_RANGE.0, Self::RADIUS_RANGE.1),
        }
    }

    /// The outermost split scale `σ_R`, in hexes.
    pub fn sigma_r(self) -> f64 {
        SIGMA_PER_RADIUS * self.clamped().radius as f64
    }

    /// The fine level's split scale `σ₀`, in hexes (never more than `σ_R`).
    pub fn sigma_fine(self) -> f64 {
        let beta = self.clamped().beta as f64;
        (SIGMA_FINE + 4.0 * (beta - 2.5).max(0.0)).min(self.sigma_r())
    }
}

impl Default for Params {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// `σ_R / R`. With this ratio the spectrum reaches half power at the same wavelength as the
/// spectrum of a kernel cut off sharply at `R`.
pub const SIGMA_PER_RADIUS: f64 = 1.15;
/// The fine level's split scale for β ≤ 2.5. It is raised slightly above that, where the
/// steeper spectrum needs a smoother fine kernel.
pub const SIGMA_FINE: f64 = 10.0;
/// The fine kernel is tapered smoothly to zero over this range of radii.
pub const FINE_TAPER: (f64, f64) = (20.0, 54.5);
/// Before high-passing, the power law is tapered over this range to fit the FFT window.
const PRE_TAPER: (f64, f64) = (95.0, 110.0);
/// Hex cell area for unit center spacing.
pub const CELL_AREA: f64 = SQRT3 / 2.0;

/// Radius of the disk with the same area as one hex cell. It is used to average the
/// singular kernel center over the cell.
const CELL_DISK_RADIUS: f64 = 0.525_037_567_904_332; // sqrt(CELL_AREA / π)
const _: () = assert!((CELL_DISK_RADIUS * CELL_DISK_RADIUS * std::f64::consts::PI - CELL_AREA).abs() < 1e-12);

/// The power law `−r^α/α` (`−log r` for α = 0) at `r > 0`.
pub fn power_law(r: f64, alpha: f64) -> f64 {
    if alpha.abs() < 1e-9 { -r.ln() } else { -r.powf(alpha) / alpha }
}

/// The power law's mean over the center cell, approximated by the equal-area disk of radius
/// ρ: the mean of `r^α` over that disk is `2ρ^α / (α + 2)`.
pub fn power_law_center(alpha: f64) -> f64 {
    let rho = CELL_DISK_RADIUS;
    if alpha.abs() < 1e-9 { -rho.ln() + 0.5 } else { -2.0 * rho.powf(alpha) / ((alpha + 2.0) * alpha) }
}

/// The constant `C` in the 2D Fourier transform of the power law, `C·κ^(−β)`:
/// `C = 2^(α+1) π Γ(1 + α/2) / Γ(1 − α/2)`. It is positive for every β in range.
pub fn power_law_constant(beta: f64) -> f64 {
    let a = beta - 2.0;
    2f64.powf(a + 1.0) * std::f64::consts::PI * gamma(1.0 + a / 2.0) / gamma(1.0 - a / 2.0)
}

/// The scale-split high-pass filter `F(κσ) = 1 − e^{−x}(1 + x)` with `x = (κσ)²/2`.
pub fn split(kappa_sigma: f64) -> f64 {
    let x = 0.5 * kappa_sigma * kappa_sigma;
    if x < 1e-3 {
        x * x * (0.5 - x * (1.0 / 3.0 - x * (0.125 - x / 30.0)))
    } else {
        1.0 - (-x).exp() * (1.0 + x)
    }
}

/// A smooth (C∞, "Planck") taper: 1 for `r ≤ a`, 0 for `r ≥ b`.
pub fn taper(r: f64, a: f64, b: f64) -> f64 {
    if r <= a {
        1.0
    } else if r >= b {
        0.0
    } else {
        // g(t) / (g(t) + g(1 − t)) with g(x) = e^{−1/x}, written with a single exponential.
        let t = (b - r) / (b - a);
        1.0 / (1.0 + (1.0 / t - 1.0 / (1.0 - t)).exp())
    }
}

/// Largest squared hex distance on the circular `WINDOW × WINDOW` grid.
pub const MAX_NORM: usize = 3 * (WINDOW / 2) * (WINDOW / 2);

/// Squared hex distance `dq² + dq·dr + dr²` of every point of the circular `WINDOW × WINDOW`
/// grid (row = `dr`, column = `dq`, both wrapped). Built once.
pub fn grid_norms() -> &'static [u32] {
    static NORMS: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    NORMS.get_or_init(|| {
        let n = WINDOW;
        (0..n * n)
            .map(|i| {
                let (q, r) = (wrap(i % n, n), wrap(i / n, n));
                (q * q + q * r + r * r) as u32
            })
            .collect()
    })
}

/// `f(m)` at every point of the grid, where `m` is the point's squared hex distance (see
/// [`grid_norms`]). Only about one point in ten has a distance not seen before, so `f` is
/// evaluated once per distinct value.
pub fn radial(f: impl Fn(u32) -> f64) -> Vec<f64> {
    let mut memo = vec![f64::NAN; MAX_NORM + 1];
    grid_norms()
        .iter()
        .map(|&m| {
            let v = &mut memo[m as usize];
            if v.is_nan() {
                *v = f(m);
            }
            *v
        })
        .collect()
}

/// `K² − K·J + J²`, minimized over the hex lattice's periodic images: the squared length,
/// in units of `(2π/n)²·4/3`, of the shortest wave vector equivalent to FFT bin `(K, J)`
/// (`K` along `q`, `J` along `r`, `J` wrapped).
pub fn folded_norm(k: i64, j: i64, n: i64) -> i64 {
    let mut best = i64::MAX;
    for a in -1..=1 {
        for b in -1..=1 {
            let (x, y) = (k + a * n, j + b * n);
            best = best.min(x * x - x * y + y * y);
        }
    }
    best
}

/// Gamma function (Lanczos approximation, g = 7; relative error below 1e-13 for x > 0).
pub fn gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let t = x + 7.5;
        let s = G[1..].iter().enumerate().fold(G[0], |s, (i, g)| s + g / (x + i as f64 + 1.0));
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * s
    }
}

/// Length of a wave vector given in cycles per hex along the axial `q` and `r` directions.
pub fn kappa(fq: f64, fr: f64) -> f64 {
    // Dual basis of [[1, ½], [0, √3/2]]: b₀ = (1, −1/√3), b₁ = (0, 2/√3).
    let ky = (-fq + 2.0 * fr) / SQRT3;
    2.0 * std::f64::consts::PI * (fq * fq + ky * ky).sqrt()
}

/// Length of the shortest wave vector equivalent to `(fq, fr)` on the hex lattice.
pub fn kappa_folded(fq: f64, fr: f64) -> f64 {
    let mut best = f64::MAX;
    for a in -1..=1 {
        for b in -1..=1 {
            best = best.min(kappa(fq + a as f64, fr + b as f64));
        }
    }
    best
}

/// Wrapped offset of FFT index `i` in an `n`-point window.
#[inline]
pub fn wrap(i: usize, n: usize) -> i64 {
    if i < n / 2 { i as i64 } else { i as i64 - n as i64 }
}

/// The fine kernel sampled on the `WINDOW × WINDOW` circular grid (row = `dr`, column = `dq`,
/// wrapped), unnormalized: the power law high-passed by `F(κσ₀)` and tapered to zero over
/// [`FINE_TAPER`], with its sum removed.
pub fn fine_spatial(params: Params, fft: &mut Fft2) -> Vec<f64> {
    let p = params.clamped();
    let alpha = p.beta as f64 - 2.0;
    let sigma = p.sigma_fine();
    let n = WINDOW;
    // Everything radial is evaluated once per distinct distance (see `radial`).
    let pre = radial(|m| {
        let r = (m as f64).sqrt();
        let k = if m == 0 { power_law_center(alpha) } else { power_law(r, alpha) };
        k * taper(r, PRE_TAPER.0, PRE_TAPER.1)
    });
    let real: Vec<f32> = pre.iter().map(|&v| v as f32).collect();
    let mut spec = vec![Complex32::default(); fft.spectrum_len()];
    fft.forward(&real, &mut spec);
    // Spectrum layout: row k = q-frequency 0..=n/2, column j = r-frequency. The filter depends
    // only on the folded wave vector's length, an integer norm, so it is also memoized.
    let half = n / 2 + 1;
    let unit = 2.0 * std::f64::consts::PI / n as f64 * (4.0f64 / 3.0).sqrt();
    let mut memo = vec![f64::NAN; MAX_NORM + 1];
    for k in 0..half {
        for j in 0..n {
            let m = folded_norm(k as i64, wrap(j, n), n as i64) as usize;
            if memo[m].is_nan() {
                memo[m] = split(sigma * unit * (m as f64).sqrt());
            }
            spec[k * n + j] *= memo[m] as f32;
        }
    }
    let mut out = vec![0.0f32; n * n];
    fft.inverse(&mut spec, 0..n, 0..n, &mut out);
    let scale = 1.0 / (n * n) as f64;
    let window = radial(|m| taper((m as f64).sqrt(), FINE_TAPER.0, FINE_TAPER.1));
    let mut k: Vec<f64> = out.iter().zip(&window).map(|(&v, &w)| v as f64 * scale * w).collect();
    // Truncation leaves a small nonzero sum, i.e. a white plateau at the largest scales. Remove
    // it with a smooth bump, so that the fine level carries nothing far above its split scale.
    let bump = radial(|m| taper((m as f64).sqrt(), 0.0, FINE_TAPER.1));
    let c = k.iter().sum::<f64>() / bump.iter().sum::<f64>();
    k.iter_mut().zip(&bump).for_each(|(v, b)| *v -= c * b);
    k
}

/// The spectrum, in the transposed layout used by [`Fft2`], of a real, even kernel sampled on
/// the circular grid, scaled by `scale`. The kernel is even, so its spectrum is real. The real
/// part is taken, which discards only rounding noise; the *signed* spectrum is kept.
pub fn spectrum_of(kernel: &[f64], scale: f64, fft: &mut Fft2) -> Vec<f32> {
    let real: Vec<f32> = kernel.iter().map(|&v| v as f32).collect();
    let mut spec = vec![Complex32::default(); fft.spectrum_len()];
    fft.forward(&real, &mut spec);
    spec.into_iter().map(|c| (c.re as f64 * scale) as f32).collect()
}

/// Projection of a kernel on the circular grid onto the x axis: `v[i]` is the sum of all taps
/// with `2x = 2·dq + dr = m0 + i` (half-hex bins). Its cosine transform is the kernel's exact
/// lattice spectrum along the x axis (the projection-slice theorem).
pub struct Projection {
    m0: i64,
    v: Vec<f64>,
}

pub fn ray_projection(kernel: &[f64], n: usize) -> Projection {
    let offset = 3 * n as i64 / 2;
    let mut p = vec![0.0; 3 * n + 1];
    for (i, &v) in kernel.iter().enumerate() {
        if v != 0.0 {
            let m = 2 * wrap(i % n, n) + wrap(i / n, n);
            p[(m + offset) as usize] += v;
        }
    }
    let first = p.iter().position(|v| *v != 0.0).unwrap_or(0);
    let last = p.iter().rposition(|v| *v != 0.0).unwrap_or(0);
    Projection { m0: first as i64 - offset, v: p[first..=last].to_vec() }
}

impl Projection {
    /// Scales every tap.
    pub fn scaled(mut self, k: f64) -> Self {
        self.v.iter_mut().for_each(|v| *v *= k);
        self
    }
}

/// Evaluates a projection from [`ray_projection`] at wavenumber `kappa` (spacing `unit` hexes
/// between lattice points): `Σᵢ v[i] cos((m0 + i)·w)` with `w = κ·unit/2`, by Horner's rule on
/// `e^{iw}`.
pub fn ray_spectrum(projection: &Projection, kappa: f64, unit: f64) -> f64 {
    let w = 0.5 * kappa * unit;
    let (c, s) = (w.cos(), w.sin());
    let (mut hr, mut hi) = (0.0f64, 0.0f64);
    for &v in projection.v.iter().rev() {
        (hr, hi) = (hr * c - hi * s + v, hr * s + hi * c);
    }
    let a = w * projection.m0 as f64;
    hr * a.cos() - hi * a.sin()
}

/// [`ray_spectrum`] at `len` evenly spaced wavenumbers `κᵢ = kmax·i/(len − 1)`, with the
/// same operations per point (so the same bits). Four points are evaluated together, which
/// hides the latency of the Horner recurrence, and in WebAssembly they run two per SIMD
/// instruction.
pub fn ray_table(projection: &Projection, kmax: f64, unit: f64, len: usize) -> Vec<f64> {
    assert!(len >= 2);
    let mut out = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        let m = (len - i).min(4);
        let (mut w, mut c, mut s) = ([0.0f64; 4], [1.0f64; 4], [0.0f64; 4]);
        for l in 0..m {
            let kappa = kmax * (i + l) as f64 / (len - 1) as f64;
            w[l] = 0.5 * kappa * unit;
            (c[l], s[l]) = (w[l].cos(), w[l].sin());
        }
        let (hr, hi) = horner4(&projection.v, c, s);
        for l in 0..m {
            let a = w[l] * projection.m0 as f64;
            out.push(hr[l] * a.cos() - hi[l] * a.sin());
        }
        i += m;
    }
    out
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn horner4(v: &[f64], c: [f64; 4], s: [f64; 4]) -> ([f64; 4], [f64; 4]) {
    let (mut hr, mut hi) = ([0.0f64; 4], [0.0f64; 4]);
    for &x in v.iter().rev() {
        for l in 0..4 {
            (hr[l], hi[l]) = (hr[l] * c[l] - hi[l] * s[l] + x, hr[l] * s[l] + hi[l] * c[l]);
        }
    }
    (hr, hi)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn horner4(v: &[f64], c: [f64; 4], s: [f64; 4]) -> ([f64; 4], [f64; 4]) {
    use core::arch::wasm32::*;
    let (c0, c1) = (f64x2(c[0], c[1]), f64x2(c[2], c[3]));
    let (s0, s1) = (f64x2(s[0], s[1]), f64x2(s[2], s[3]));
    let z = f64x2_splat(0.0);
    let (mut r0, mut i0, mut r1, mut i1) = (z, z, z, z);
    for &x in v.iter().rev() {
        let xv = f64x2_splat(x);
        (r0, i0) = (
            f64x2_add(f64x2_sub(f64x2_mul(r0, c0), f64x2_mul(i0, s0)), xv),
            f64x2_add(f64x2_mul(r0, s0), f64x2_mul(i0, c0)),
        );
        (r1, i1) = (
            f64x2_add(f64x2_sub(f64x2_mul(r1, c1), f64x2_mul(i1, s1)), xv),
            f64x2_add(f64x2_mul(r1, s1), f64x2_mul(i1, c1)),
        );
    }
    let get = |a: v128, b: v128| {
        [
            f64x2_extract_lane::<0>(a),
            f64x2_extract_lane::<1>(a),
            f64x2_extract_lane::<0>(b),
            f64x2_extract_lane::<1>(b),
        ]
    };
    (get(r0, r1), get(i0, i1))
}

/// Inverse of the standard normal CDF (Acklam's rational approximation, relative error
/// below 1.2e-9). The field is very nearly Gaussian with unit variance, so
/// `inverse_normal_cdf(1 − land)` is the sea level that leaves a fraction `land` above water.
#[allow(clippy::excessive_precision)]
pub fn inverse_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] =
        [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// Sea level, in units of the field's standard deviation, that leaves `land` (0–1) of the
/// map above water.
pub fn sea_level(land: f64) -> f32 {
    inverse_normal_cdf(1.0 - land) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_quantiles() {
        assert!(inverse_normal_cdf(0.5).abs() < 1e-9);
        assert!((inverse_normal_cdf(0.975) - 1.959_963_985).abs() < 1e-6);
        assert!((inverse_normal_cdf(0.001) + 3.090_232_306).abs() < 1e-6);
    }

    #[test]
    fn gamma_values() {
        assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-12);
        assert!((gamma(1.0) - 1.0).abs() < 1e-12);
        assert!((gamma(4.5) - 11.631_728_396_567_448).abs() < 1e-9);
        // β = 2 is the logarithm, whose transform is 2π/κ².
        assert!((power_law_constant(2.0) - 2.0 * std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn split_filter() {
        assert!(split(0.0) == 0.0);
        assert!((split(1e-2) - 1.0 + (-(5e-5f64)).exp() * (1.0 + 5e-5)).abs() < 1e-15);
        assert!(split(10.0) > 0.999_999);
        // Continuity at the series switch-over, x = 1e-3.
        let edge = 2e-3f64.sqrt();
        let (lo, hi) = (split(edge * (1.0 - 1e-9)), split(edge * (1.0 + 1e-9)));
        // The inputs differ by 2e-9 relative, so F (∝ κ⁴σ⁴ here) by 8e-9.
        assert!((lo - hi).abs() < 1.2e-8 * hi, "{lo} vs {hi}");
    }

    #[test]
    fn fine_kernel_support() {
        // Every nonzero tap must have |dq|, |dr| ≤ MARGIN − APRON.
        let mut fft = Fft2::new(WINDOW);
        let k = fine_spatial(Params::DEFAULT, &mut fft);
        let n = WINDOW;
        let limit = (crate::tile::MARGIN - crate::tile::APRON) as i64;
        for (i, v) in k.iter().enumerate() {
            if *v != 0.0 {
                assert!(wrap(i / n, n).abs() <= limit && wrap(i % n, n).abs() <= limit, "tap at {i}");
            }
        }
        assert!(FINE_TAPER.1 <= crate::tile::MAX_RADIUS);
    }

    /// Along the x axis, the fine kernel's lattice spectrum follows `κ^(−β)` between the split
    /// scale and the lattice scale, for every β (the smooth taper leaves no cutoff ripple).
    #[test]
    fn fine_spectral_slope() {
        let mut fft = Fft2::new(WINDOW);
        for beta in [1.0f32, 1.5, 2.0, 2.5, 3.0] {
            let k = fine_spatial(Params { beta, radius: 1000.0 }, &mut fft);
            let proj = ray_projection(&k, WINDOW);
            let pts: Vec<(f64, f64)> = (0..24)
                .map(|i| 0.25 * 3f64.powf(i as f64 / 23.0)) // κ ∈ [0.25, 0.75]
                .map(|kap| (kap.ln(), ray_spectrum(&proj, kap, 1.0).abs().ln()))
                .collect();
            let (mx, my) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
            let (mx, my) = (mx / pts.len() as f64, my / pts.len() as f64);
            let slope = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
                / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
            assert!((slope + beta as f64).abs() < 0.12, "β={beta}: slope {slope}");
        }
    }
}
