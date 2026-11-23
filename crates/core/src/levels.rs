// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coarse levels: the large scales, on coarser hex lattices.
//!
//! A fine tile's window only has room for a kernel of radius ≈ 54.6 hexes. Larger scales come
//! from three coarse levels with grid spacings of 8, 64 and 512 hexes. Each level is an
//! independent random field: white noise on its own lattice, convolved with its own compact
//! kernel by overlap-save (exactly as for fine tiles, on the same 256² window), then
//! interpolated onto the hexes with a cubic B-spline.
//!
//! The levels split the target power spectrum between them, so the sum has the target
//! spectrum. Level `j` carries `|K̂ⱼ|² − |K̂ⱼ₋₁|²`, where `K̂ⱼ` is the cumulative spectrum: the
//! fine kernel's exact lattice spectrum plus the power law band between the fine split scale and
//! the level's outer split scale. Its kernel is designed in the spectrum (dividing by the
//! B-spline's response) and then truncated smoothly in real space, so it is compact and
//! overlap-save is exact: the levels are seamless whatever the design accuracy.

use crate::fft::Fft2;
use crate::hex::SQRT3;
use crate::kernel::{self, CELL_AREA, Params, wrap};
use crate::noise;
use crate::tile::{APRON, STORED, Spectrum, TILE, TileCoord, WINDOW};
use rustfft::num_complex::Complex32;

/// Number of coarse levels.
pub const LEVELS: usize = 3;
/// Grid spacing of each coarse level, in hexes.
pub const SPACING: [i64; LEVELS] = [8, 64, 512];
/// Outer split scale of each level's cumulative spectrum (capped at `σ_R`); the top level
/// always reaches `σ_R`.
pub const SPLIT: [f64; LEVELS - 1] = [80.0, 640.0];
/// Coarse FFT window side length, in nodes (the fine window size, so the FFT plan is shared).
pub const CWINDOW: usize = WINDOW;
/// Nodes of neighboring noise on each side of a coarse tile.
pub const CMARGIN: usize = 80;
/// Coarse tile side length, in nodes.
pub const CTILE: usize = CWINDOW - 2 * CMARGIN;
/// Extra nodes stored around each coarse tile, so that each fine tile's B-spline support lies
/// inside a single coarse tile.
pub const CAPRON: usize = 3;
/// Side length of a stored coarse tile.
pub const CSTORED: usize = CTILE + 2 * CAPRON;
/// Largest coarse kernel radius, in nodes, that fits inside the coarse margin.
pub const MAX_REACH: f64 = (CMARGIN - CAPRON) as f64 * SQRT3 / 2.0;
/// The coarse kernel is truncated over `[5.5 σ_outer, 5.5 σ_outer + 3 σ_inner]`.
const TRUNCATE_OUTER: f64 = 5.5;
const TRUNCATE_WIDTH: f64 = 3.0;
/// Seed salts that give each level its own noise.
const SALT: [u64; LEVELS] = [0x6a09_e667_f3bc_c909, 0xbb67_ae85_84ca_a73b, 0x3c6e_f372_fe94_f82b];

const _: () =
    assert!((SPACING[0] as usize * CTILE).is_multiple_of(TILE) && TILE.is_multiple_of(SPACING[0] as usize));

/// Address of a coarse tile: level `level` nodes `[CTILE·cx, CTILE·(cx+1)) × [CTILE·cy, …)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoarseCoord {
    pub level: u8,
    pub cx: i32,
    pub cy: i32,
}

impl CoarseCoord {
    /// The coarse tile of `level` whose stored region contains every node that fine tile
    /// `coord` needs.
    pub fn parent(level: usize, coord: TileCoord) -> Self {
        let s = SPACING[level];
        let (q0, r0) = coord.origin();
        let c = |x: i64| x.div_euclid(s).div_euclid(CTILE as i64) as i32;
        Self { level: level as u8, cx: c(q0), cy: c(r0) }
    }
    /// Index of the tile's first node.
    pub fn origin(self) -> (i64, i64) {
        (self.cx as i64 * CTILE as i64, self.cy as i64 * CTILE as i64)
    }
}

/// One designed level kernel.
pub struct Level {
    /// Grid spacing in hexes.
    pub spacing: i64,
    /// The kernel on the `CWINDOW²` circular node grid (row = `dR`, column = `dQ`), in units
    /// of the total σ. Zero if the level carries no power at these parameters.
    pub spatial: Vec<f64>,
    /// Its spectrum, transposed layout, with the `1/N²` inverse-FFT normalization folded in.
    pub spectrum: Vec<f32>,
    /// Share of the total variance (spatial average).
    pub variance: f64,
    /// Truncation range of the kernel, in hexes.
    pub truncation: (f64, f64),
}

impl Level {
    pub fn active(&self) -> bool {
        self.variance > 0.0
    }
}

/// All kernels for one parameter set.
pub struct Design {
    pub params: Params,
    /// The fine kernel on the `WINDOW²` circular grid, in units of the total σ.
    pub fine_spatial: Vec<f64>,
    /// Its spectrum, transposed layout, `1/N²` folded in.
    pub fine_spectrum: Vec<f32>,
    pub levels: Vec<Level>,
    /// Standard deviation of the unnormalized field (the normalization divisor).
    pub sigma: f64,
    projection: kernel::Projection,
}

/// Cubic B-spline.
#[inline]
pub fn bspline(t: f64) -> f64 {
    let t = t.abs();
    if t < 1.0 {
        2.0 / 3.0 - t * t + 0.5 * t * t * t
    } else if t < 2.0 {
        (2.0 - t).powi(3) / 6.0
    } else {
        0.0
    }
}

/// Lattice transform of the B-spline sampled every `1/s`: `Σₙ β(n/s) cos(ω n)`, which is `s` at
/// 0. By the Poisson summation formula it is `s Σₖ sinc⁴(s(ω + 2πk)/2)`. The terms fall off as
/// `k⁻⁴`, so five terms leave a relative error below 1e-6 for `s ≥ 8`.
pub fn bspline_response(s: i64, omega: f64) -> f64 {
    let sinc4 = |x: f64| if x.abs() < 1e-8 { 1.0 } else { (x.sin() / x).powi(4) };
    let tau = 2.0 * std::f64::consts::PI;
    s as f64 * (-2..=2).map(|k| sinc4(0.5 * s as f64 * (omega + tau * k as f64))).sum::<f64>()
}

/// Rows in the tables of realized and target power along the x axis.
const TABLE: usize = 1024;

/// The parts of a level's design that do not depend on (β, R), built once.
struct LevelTables {
    /// Top wavenumber of the power tables.
    kmax: f64,
    /// B-spline response in the passband at each q-frequency `0..=n/2`.
    phi: Vec<f64>,
    /// Sum of the squared B-spline response over all aliases, at each frequency `0..n`.
    alias: Vec<f64>,
    /// For each lower level `l`, the squared response of its interpolation (B-spline, then
    /// the half-resolution spline) at each power-table wavenumber.
    lower_response: Vec<Vec<f64>>,
}

fn level_tables() -> &'static [LevelTables; LEVELS] {
    static TABLES: std::sync::OnceLock<[LevelTables; LEVELS]> = std::sync::OnceLock::new();
    TABLES.get_or_init(|| {
        let n = CWINDOW;
        std::array::from_fn(|j| {
            let s = SPACING[j];
            let kmax = 2.0 * std::f64::consts::PI * 1.01 / s as f64;
            let half = n / 2 + 1;
            let phi = (0..half)
                .map(|k| bspline_response(s, 2.0 * std::f64::consts::PI * k as f64 / (n as f64 * s as f64)))
                .collect();
            // Σₐ φ(ω + 2πa/s)² = s Σₘ ρ(m) cos(2π f m), ρ the autocorrelation of the samples.
            let rho: Vec<f64> = (0..4)
                .map(|m| {
                    (1 - 2 * s..2 * s)
                        .map(|q| bspline(q as f64 / s as f64) * bspline(q as f64 / s as f64 + m as f64))
                        .sum()
                })
                .collect();
            let alias = (0..n)
                .map(|k| {
                    let f = k as f64 / n as f64;
                    s as f64
                        * (rho[0]
                            + 2.0
                                * (1..4)
                                    .map(|m| rho[m] * (2.0 * std::f64::consts::PI * f * m as f64).cos())
                                    .sum::<f64>())
                })
                .collect();
            let lower_response = SPACING[..j]
                .iter()
                .map(|&sl| {
                    (0..TABLE)
                        .map(|i| {
                            let kap = kmax * i as f64 / (TABLE - 1) as f64;
                            let phi = bspline_response(sl, kap) * bspline_response(sl, kap / 2.0);
                            phi * phi
                        })
                        .collect()
                })
                .collect();
            LevelTables { kmax, phi, alias, lower_response }
        })
    })
}

impl Design {
    pub fn new(params: Params, fft: &mut Fft2) -> Self {
        let p = params.clamped();
        let beta = p.beta as f64;
        let (sigma_r, sigma0) = (p.sigma_r(), p.sigma_fine());
        let n = CWINDOW;
        let fine = kernel::fine_spatial(p, fft);
        let fine_var: f64 = fine.iter().map(|v| v * v).sum();
        let projection = kernel::ray_projection(&fine, WINDOW);
        let c_over_a = kernel::power_law_constant(beta) / CELL_AREA;
        // Power law band between σ₀ and `hi` (zero at κ = 0, where F vanishes as κ⁴).
        let band = move |kap: f64, hi: f64| {
            if kap <= 0.0 {
                0.0
            } else {
                c_over_a * kap.powf(-beta) * (kernel::split(kap * hi) - kernel::split(kap * sigma0))
            }
        };
        let outer = |j: usize| if j + 1 == LEVELS { sigma_r } else { SPLIT[j].min(sigma_r) };

        let mut levels: Vec<Level> = Vec::with_capacity(LEVELS);
        let mut lower: Vec<(usize, kernel::Projection)> = Vec::new(); // level, ray projection
        let mut prev_reach = kernel::FINE_TAPER.1;
        for (j, &s) in SPACING.iter().enumerate() {
            let (lo, hi) = (if j == 0 { sigma0 } else { outer(j - 1) }, outer(j));
            let a = (TRUNCATE_OUTER * hi).max(prev_reach);
            // A level above the outermost split scale only fills the deficit the truncation of the
            // level below leaves at its largest scales, so it gets a wide kernel.
            let filler = hi <= lo * (1.0 + 1e-9);
            // The level that reaches σ_R holds the field's slow roll-off below 1/σ_R, so it gets
            // as much room as its window allows, up to twice its taper start.
            let outermost = hi >= sigma_r * (1.0 - 1e-9);
            let b = if filler {
                4.0 * a
            } else if outermost {
                (2.0 * a).max(a + TRUNCATE_WIDTH * lo)
            } else {
                a + TRUNCATE_WIDTH * lo
            }
            .min(MAX_REACH * s as f64);
            let truncation = (a.min(0.98 * b), b);
            prev_reach = truncation.1;
            // Power already realized below this level (fine kernel plus lower levels, main
            // spectral image), tabulated along the x axis over this level's wavenumbers.
            let tables = &level_tables()[j];
            let kmax = tables.kmax;
            let fine_table = kernel::ray_table(&projection, kmax, 1.0, TABLE);
            let mut below_table: Vec<f64> = fine_table.iter().map(|f| f * f).collect();
            for (l, proj) in &lower {
                let sl = SPACING[*l];
                let g = kernel::ray_table(proj, kmax, sl as f64, TABLE);
                for ((p, g), phi2) in below_table.iter_mut().zip(g).zip(&tables.lower_response[*l]) {
                    *p += phi2 * g * g / (sl * sl) as f64;
                }
            }
            let lookup = |t: &[f64], kap: f64| {
                let x = (kap / kmax * (TABLE - 1) as f64).min((TABLE - 1) as f64 - 1e-9);
                let i = x as usize;
                t[i] + (t[i + 1] - t[i]) * (x - i as f64)
            };
            // Cumulative target up to this level's outer scale, minus what is already realized.
            let need: Vec<f64> = (0..TABLE)
                .map(|i| {
                    let kap = kmax * i as f64 / (TABLE - 1) as f64;
                    (fine_table[i] + band(kap, hi)).powi(2) - below_table[i]
                })
                .collect();
            let power = |kap: f64| lookup(&need, kap);
            // B-spline response in the passband, and its sum over all aliases (for the variance).
            let half = n / 2 + 1;
            let (phi, alias) = (&tables.phi, &tables.alias);
            let phi_at = |j: usize| phi[j.min(n - j)];
            // Design: Ĝ = s·√P / Φ̂, on the coarse window's spectrum grid. √P depends only on
            // the wave vector's length, whose square is an integer norm in units of
            // (2π/(n·s))²·4/3, so it is evaluated once per distinct norm.
            let unit = 2.0 * std::f64::consts::PI / (n as f64 * s as f64) * (4.0f64 / 3.0).sqrt();
            let mut root_power = vec![f64::NAN; kernel::MAX_NORM + 1];
            let mut spec = vec![Complex32::default(); fft.spectrum_len()];
            for kq in 0..half {
                for jr in 0..n {
                    let ph = phi[kq] * phi_at(jr);
                    if ph > 1e-4 * (s * s) as f64 {
                        let (k, j) = (kq as i64, wrap(jr, n));
                        let m = (k * k - k * j + j * j) as usize;
                        if root_power[m].is_nan() {
                            root_power[m] = power(unit * (m as f64).sqrt()).max(0.0).sqrt();
                        }
                        let g = s as f64 * root_power[m] / ph;
                        spec[kq * n + jr] = Complex32::new(g as f32, 0.0);
                    }
                }
            }
            let mut real = vec![0.0f32; n * n];
            fft.inverse(&mut spec, 0..n, 0..n, &mut real);
            let scale = 1.0 / (n * n) as f64;
            let radius = |m: u32| s as f64 * (m as f64).sqrt();
            let window = kernel::radial(|m| kernel::taper(radius(m), truncation.0, truncation.1));
            let mut spatial: Vec<f64> =
                real.iter().zip(&window).map(|(&v, &w)| v as f64 * scale * w).collect();
            // Every band vanishes at κ = 0, but truncating the kernel's tail leaves a small sum: a
            // white plateau at scales far above R. Remove it with a smooth bump.
            let bump = kernel::radial(|m| kernel::taper(radius(m), 0.0, truncation.1));
            let c = spatial.iter().sum::<f64>() / bump.iter().sum::<f64>();
            spatial.iter_mut().zip(&bump).for_each(|(v, b)| *v -= c * b);
            let g_t = kernel::spectrum_of(&spatial, 1.0, fft);
            let mut variance = 0.0;
            for kq in 0..half {
                let w = if kq == 0 || kq == n / 2 { 1.0 } else { 2.0 };
                let aq = alias[kq];
                for jr in 0..n {
                    let g = g_t[kq * n + jr] as f64;
                    variance += w * g * g * aq * alias[jr];
                }
            }
            variance /= (s * s * s * s) as f64 * (n * n) as f64;
            if variance < 1e-9 * fine_var {
                // Nothing left for this level to carry.
                spatial.iter_mut().for_each(|v| *v = 0.0);
                levels.push(Level {
                    spacing: s,
                    spatial,
                    spectrum: vec![0.0; g_t.len()],
                    variance: 0.0,
                    truncation,
                });
                continue;
            }
            lower.push((j, kernel::ray_projection(&spatial, n)));
            levels.push(Level { spacing: s, spatial, spectrum: g_t, variance, truncation });
        }

        let sigma = (fine_var + levels.iter().map(|l| l.variance).sum::<f64>()).sqrt();
        let norm = 1.0 / ((n * n) as f64 * sigma);
        let fine_spectrum = kernel::spectrum_of(&fine, norm, fft);
        for l in &mut levels {
            l.spatial.iter_mut().for_each(|v| *v /= sigma);
            l.spectrum.iter_mut().for_each(|v| *v = (*v as f64 * norm) as f32);
            l.variance /= sigma * sigma;
        }
        let fine_spatial = fine.into_iter().map(|v| v / sigma).collect();
        let projection = projection.scaled(1.0 / sigma);
        Self { params: p, fine_spatial, fine_spectrum, levels, sigma, projection }
    }

    /// Target power at wavenumber `kappa` along the x axis, in units of σ²: the fine kernel's
    /// exact lattice spectrum plus the power-law band up to `σ_R`.
    pub fn target_power(&self, kappa: f64) -> f64 {
        let p = self.params;
        let beta = p.beta as f64;
        let fine = kernel::ray_spectrum(&self.projection, kappa, 1.0);
        let band = if kappa <= 0.0 {
            0.0
        } else {
            kernel::power_law_constant(beta) / CELL_AREA
                * kappa.powf(-beta)
                * (kernel::split(kappa * p.sigma_r()) - kernel::split(kappa * p.sigma_fine()))
                / self.sigma
        };
        (fine + band).powi(2)
    }

    /// Power actually realized at wavenumber `kappa` along the x axis (main spectral image),
    /// from the truncated kernels and the B-spline interpolation, in units of σ².
    pub fn realized_power(&self, kappa: f64) -> f64 {
        let fine = kernel::ray_spectrum(&self.projection, kappa, 1.0);
        let mut total = fine * fine;
        for l in self.levels.iter().filter(|l| l.active()) {
            let s = l.spacing;
            let proj = kernel::ray_projection(&l.spatial, CWINDOW);
            let g = kernel::ray_spectrum(&proj, kappa, s as f64);
            // Along x, the axial frequencies are (κ, κ/2) radians per hex.
            let phi = bspline_response(s, kappa) * bspline_response(s, kappa / 2.0);
            total += phi * phi * g * g / (s * s) as f64;
        }
        total
    }
}

/// Noise key of a level.
pub fn level_key(seed: u64, level: usize) -> u64 {
    noise::seed_key(seed ^ SALT[level])
}

/// Fills `window` with the noise of coarse tile `c`'s window.
pub fn coarse_noise(seed: u64, c: CoarseCoord, window: &mut [f32]) {
    let (q0, r0) = c.origin();
    noise::fill(level_key(seed, c.level as usize), q0 - CMARGIN as i64, r0 - CMARGIN as i64, CWINDOW, window);
}

/// Noise spectrum of a coarse tile's window.
pub fn coarse_noise_spectrum(seed: u64, c: CoarseCoord, window: &mut [f32], fft: &mut Fft2) -> Spectrum {
    coarse_noise(seed, c, window);
    let mut spec = vec![Complex32::default(); fft.spectrum_len()].into_boxed_slice();
    fft.forward(window, &mut spec);
    Spectrum(spec)
}

/// A coarse tile from its noise spectrum: `CSTORED²` node values, row-major (row = R), with
/// index `[CAPRON + R_local][CAPRON + Q_local]` holding node `origin + (Q_local, R_local)`.
pub fn coarse_tile(
    spectrum: &Spectrum,
    kernel: &[f32],
    work: &mut [Complex32],
    fft: &mut Fft2,
) -> Box<[f32]> {
    for ((w, s), k) in work.iter_mut().zip(spectrum.0.iter()).zip(kernel.iter()) {
        *w = *s * *k;
    }
    let keep = (CMARGIN - CAPRON)..(CMARGIN + CTILE + CAPRON);
    let mut out = vec![0.0; CSTORED * CSTORED].into_boxed_slice();
    fft.inverse(work, keep.clone(), keep, &mut out);
    out
}

/// Cubic B-spline weights for the `s` positions within a cell: hex `q = s·n + i` takes
/// `Σₐ w[i][a] · node(n − 1 + a)`.
pub fn weights(s: i64) -> Vec<[f32; 4]> {
    (0..s)
        .map(|i| {
            let t = i as f64 / s as f64;
            [bspline(t + 1.0), bspline(t), bspline(t - 1.0), bspline(t - 2.0)].map(|w| w as f32)
        })
        .collect()
}

/// Adds the B-spline interpolation of coarse tile `c` (values `nodes`) to fine tile `coord`'s
/// stored elevations `out` (`STORED²`, apron included). Separable: first along `q` for each
/// coarse row the tile needs, then along `r`.
pub fn add_interpolated(
    out: &mut [f32],
    coord: TileCoord,
    c: CoarseCoord,
    nodes: &[f32],
    w: &[[f32; 4]],
    tmp: &mut Vec<f32>,
) {
    let s = SPACING[c.level as usize];
    let (q0, r0) = coord.origin();
    let (qa, ra) = (q0 - APRON as i64, r0 - APRON as i64);
    let (nq0, nr0) = c.origin();
    let (nq0, nr0) = (nq0 - CAPRON as i64, nr0 - CAPRON as i64); // node of stored index 0
    let node = |x: i64| x.div_euclid(s);
    let (rn0, rn1) = (node(ra) - 1, node(ra + STORED as i64 - 1) + 2);
    debug_assert!(rn0 >= nr0 && rn1 < nr0 + CSTORED as i64);
    debug_assert!(node(qa) > nq0 && node(qa + STORED as i64 - 1) + 2 < nq0 + CSTORED as i64);
    let rows = (rn1 - rn0 + 1) as usize;
    tmp.resize(rows * STORED, 0.0);
    // Node and weight index of the first column; both advance by counting, not division.
    let (qn_first, off_first) = (node(qa), qa.rem_euclid(s) as usize);
    let s = s as usize;
    for (k, rn) in (rn0..=rn1).enumerate() {
        let src = &nodes[(rn - nr0) as usize * CSTORED..][..CSTORED];
        let dst = &mut tmp[k * STORED..(k + 1) * STORED];
        let (mut b, mut off) = ((qn_first - 1 - nq0) as usize, off_first);
        for d in dst.iter_mut() {
            let wi = &w[off];
            *d = wi[0] * src[b] + wi[1] * src[b + 1] + wi[2] * src[b + 2] + wi[3] * src[b + 3];
            off += 1;
            if off == s {
                off = 0;
                b += 1;
            }
        }
    }
    for (row, o) in out.chunks_exact_mut(STORED).enumerate() {
        let r = ra + row as i64;
        let rn = node(r);
        let wi = w[(r - rn * s as i64) as usize];
        let k = (rn - 1 - rn0) as usize;
        let t = &tmp[k * STORED..(k + 4) * STORED];
        let (a, b, c2, d) =
            (&t[..STORED], &t[STORED..2 * STORED], &t[2 * STORED..3 * STORED], &t[3 * STORED..]);
        add_weighted4(o, wi, [a, b, c2, d]);
    }
}

/// `o[i] += w₀·a[i] + w₁·b[i] + w₂·c[i] + w₃·d[i]`, summed left to right.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn add_weighted4(o: &mut [f32], w: [f32; 4], [a, b, c, d]: [&[f32]; 4]) {
    for i in 0..o.len() {
        o[i] += w[0] * a[i] + w[1] * b[i] + w[2] * c[i] + w[3] * d[i];
    }
}

/// Four outputs per instruction, with the same operations in the same order as the scalar
/// version, so the same bits.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn add_weighted4(o: &mut [f32], w: [f32; 4], [a, b, c, d]: [&[f32]; 4]) {
    use core::arch::wasm32::*;
    let n = o.len();
    assert!(a.len() >= n && b.len() >= n && c.len() >= n && d.len() >= n);
    let wv = w.map(|v| f32x4_splat(v));
    let body = n & !3;
    for i in (0..body).step_by(4) {
        // SAFETY: i + 4 ≤ n, and every slice holds at least n values (asserted above).
        unsafe {
            let ld = |s: &[f32]| v128_load(s.as_ptr().add(i) as *const v128);
            let mut x = f32x4_mul(wv[0], ld(a));
            x = f32x4_add(x, f32x4_mul(wv[1], ld(b)));
            x = f32x4_add(x, f32x4_mul(wv[2], ld(c)));
            x = f32x4_add(x, f32x4_mul(wv[3], ld(d)));
            let p = o.as_mut_ptr().add(i) as *mut v128;
            v128_store(p, f32x4_add(v128_load(p), x));
        }
    }
    for i in body..n {
        o[i] += w[0] * a[i] + w[1] * b[i] + w[2] * c[i] + w[3] * d[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_fit_inside_coarse_margin() {
        let mut fft = Fft2::new(WINDOW);
        for radius in [30.0f32, 300.0, 1000.0, Params::RADIUS_RANGE.1] {
            let d = Design::new(Params { beta: 1.5, radius }, &mut fft);
            let limit = (CMARGIN - CAPRON) as i64;
            for l in &d.levels {
                for (i, v) in l.spatial.iter().enumerate() {
                    if *v != 0.0 {
                        let (dq, dr) = (wrap(i % CWINDOW, CWINDOW), wrap(i / CWINDOW, CWINDOW));
                        assert!(
                            dq.abs() <= limit && dr.abs() <= limit,
                            "R={radius} s={}: tap {dq},{dr}",
                            l.spacing
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bspline_response_matches_direct_sum() {
        for s in SPACING {
            for i in 0..20 {
                let w = std::f64::consts::PI * i as f64 / 19.0;
                let direct: f64 =
                    (1 - 2 * s..2 * s).map(|n| bspline(n as f64 / s as f64) * (w * n as f64).cos()).sum();
                assert!((bspline_response(s, w) - direct).abs() < 1e-6 * s as f64, "s={s} ω={w}");
            }
        }
    }

    #[test]
    fn weights_partition_unity() {
        for s in SPACING {
            for w in weights(s) {
                assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            }
        }
    }

    /// The realized power spectrum (fine kernel + truncated coarse kernels + interpolation)
    /// matches the target along the x axis, wherever the target holds non-negligible power.
    #[test]
    fn design_closure() {
        let mut fft = Fft2::new(WINDOW);
        for (beta, radius) in [
            (0.5f32, 1000.0f32),
            (1.0, 300.0),
            (1.5, 8.0),
            (1.5, 30.0),
            (1.5, 300.0),
            (1.5, 1000.0),
            (1.5, 2500.0),
            (2.5, 2500.0),
            (2.0, 1000.0),
            (2.5, 1000.0),
            (3.0, 300.0),
            (3.0, 2500.0),
        ] {
            let d = Design::new(Params { beta, radius }, &mut fft);
            let ks: Vec<f64> = (0..120).map(|i| 1e-4 * 10f64.powf(i as f64 / 119.0 * 3.9)).collect();
            let peak = ks.iter().map(|&k| d.target_power(k) * k * k).fold(0.0, f64::max);
            let mut worst = (0.0, 0.0);
            for &k in &ks {
                let t = d.target_power(k);
                // Judge only where the power per log-wavenumber is at least 1% of its peak, and at
                // wavelengths up to about 8 R: the deliberate roll-off beyond that is suppressed
                // slightly more than the target asks for.
                if t * k * k < 1e-2 * peak || k * d.params.sigma_r() < 0.7 {
                    continue;
                }
                let e = (d.realized_power(k) / t - 1.0).abs();
                if e > worst.0 {
                    worst = (e, k);
                }
            }
            assert!(worst.0 < 0.04, "β={beta} R={radius}: {:.3} at κ={:.5}", worst.0, worst.1);
        }
    }

    #[test]
    fn variance_is_normalized() {
        let mut fft = Fft2::new(WINDOW);
        let d = Design::new(Params::DEFAULT, &mut fft);
        let total = d.fine_spatial.iter().map(|v| v * v).sum::<f64>()
            + d.levels.iter().map(|l| l.variance).sum::<f64>();
        assert!((total - 1.0).abs() < 1e-9);
        assert!(d.levels.iter().all(|l| l.active()));
    }
}
