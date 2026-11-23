// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tiles and the overlap-save generator.
//!
//! The elevation field is the sum of a fine level and three coarse levels
//! ([`crate::levels`]). The fine level is white noise over the whole (unbounded) hex lattice,
//! convolved with a compact kernel of radius at most [`MAX_RADIUS`]. Each tile is produced
//! independently:
//!
//! 1. Evaluate noise over a `WINDOW × WINDOW` patch: the tile plus a `MARGIN`-hex border.
//! 2. FFT, multiply by the kernel spectrum, inverse FFT. This is a *circular* convolution.
//! 3. Keep the center. Kernel taps never reach further than `MARGIN − APRON` hexes in index
//!    space, so the circular wrap-around never touches the kept region, and every kept value
//!    equals the true, infinite linear convolution.
//!
//! 4. Add each coarse level: B-spline interpolation of the one coarse tile (made the same way,
//!    on a coarser lattice, and cached) whose stored nodes cover this tile.
//!
//! Every tile is therefore complete and exact on its own. Adjacent tiles agree to rounding
//! error, and the result does not depend on which tiles exist or in what order they were made.

use crate::fft::Fft2;
use crate::hex::SQRT3;
use crate::kernel::Params;
use crate::levels::{self, CoarseCoord, Design, LEVELS};
use crate::noise;
use rustfft::num_complex::Complex32;
use std::collections::HashMap;

/// FFT window side length (power of two).
pub const WINDOW: usize = 256;
/// Hexes of neighboring noise on each side of the tile inside the window.
pub const MARGIN: usize = 64;
/// Tile side length in hexes.
pub const TILE: usize = WINDOW - 2 * MARGIN;
/// Extra ring of exact values around each tile, used for seamless shading at tile edges.
pub const APRON: usize = 1;
/// Side length of the stored tile, including the apron.
pub const STORED: usize = TILE + 2 * APRON;
/// Number of complex values in a noise spectrum.
pub const SPECTRUM_LEN: usize = WINDOW * (WINDOW / 2 + 1);

/// Largest fine-kernel radius whose support fits inside the margin. An index offset `(dq, dr)` has
/// cartesian length ≥ (√3/2)·max(|dq|, |dr|), so radius `R` reaches at most `2R/√3` hexes
/// along either index axis.
pub const MAX_RADIUS: f64 = (MARGIN - APRON) as f64 * SQRT3 / 2.0;

const _: () = assert!(WINDOW.is_power_of_two() && TILE > 0 && APRON < MARGIN);

/// Integer tile address. Tile `(tx, ty)` covers hexes
/// `q ∈ [tx·TILE, (tx+1)·TILE)`, `r ∈ [ty·TILE, (ty+1)·TILE)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileCoord {
    pub tx: i32,
    pub ty: i32,
}

impl TileCoord {
    pub const fn new(tx: i32, ty: i32) -> Self {
        Self { tx, ty }
    }
    /// Axial coordinates of the tile's first hex.
    pub const fn origin(self) -> (i64, i64) {
        (self.tx as i64 * TILE as i64, self.ty as i64 * TILE as i64)
    }
}

/// The noise spectrum of one tile window. It depends only on the seed and the tile, so it
/// can be cached and reused when the kernel parameters change.
#[derive(Clone)]
pub struct Spectrum(pub Box<[Complex32]>);

/// A finished tile: `STORED × STORED` elevations, row-major (row = r, column = q), in units of
/// the field's standard deviation. Index `[APRON + r_local][APRON + q_local]` holds hex
/// `origin + (q_local, r_local)`, and the outer ring is the apron.
#[derive(Clone)]
pub struct Tile {
    pub coord: TileCoord,
    pub elevation: Box<[f32]>,
}

impl Tile {
    /// Elevation of local hex `(q, r)`, where each coordinate is in `-APRON..TILE + APRON`.
    #[inline]
    pub fn get(&self, q: isize, r: isize) -> f32 {
        let a = APRON as isize;
        self.elevation[((r + a) as usize) * STORED + (q + a) as usize]
    }
}

/// Reusable generator. It holds the FFT plans, the scratch buffers, the kernel designs of the
/// last few parameter sets and a small cache of coarse tiles. Creating one is cheap. Reuse it: every call after the first
/// allocates little more than the output.
pub struct Generator {
    fft: Fft2,
    /// Recent kernel designs, the current one first.
    designs: Vec<Design>,
    window: Vec<f32>,
    work: Vec<Complex32>,
    weights: Vec<Vec<[f32; 4]>>,
    interp: Vec<f32>,
    /// Coarse tiles for `coarse_key`, with a last-use stamp.
    coarse: HashMap<CoarseCoord, (u64, Box<[f32]>)>,
    coarse_key: Option<(u64, Params)>,
    /// Coarse noise spectra for `noise_seed`, with a last-use stamp.
    coarse_noise: HashMap<CoarseCoord, (u64, Spectrum)>,
    noise_seed: Option<u64>,
    clock: u64,
}

impl Default for Generator {
    fn default() -> Self {
        Self::new()
    }
}

/// Kernel designs kept (about 2.6 MB each), so returning to recent (β, R) values, as when a
/// slider is dragged back and forth, needs no redesign.
const DESIGN_CAP: usize = 4;

/// Cached coarse tiles (about 40 KB each) and coarse noise spectra (about 260 KB each).
const COARSE_CAP: usize = 64;
const COARSE_NOISE_CAP: usize = 24;

impl Generator {
    pub fn new() -> Self {
        Self {
            fft: Fft2::new(WINDOW),
            designs: Vec::new(),
            window: vec![0.0; WINDOW * WINDOW],
            work: vec![Complex32::default(); SPECTRUM_LEN],
            weights: levels::SPACING.iter().map(|&s| levels::weights(s)).collect(),
            interp: Vec::new(),
            coarse: HashMap::new(),
            coarse_key: None,
            coarse_noise: HashMap::new(),
            noise_seed: None,
            clock: 0,
        }
    }

    /// Makes the kernels for `params` current, designing them unless they are cached.
    /// Returns true if they had to be designed.
    pub fn set_params(&mut self, params: Params) -> bool {
        let params = params.clamped();
        if let Some(i) = self.designs.iter().position(|d| d.params == params) {
            let d = self.designs.remove(i);
            self.designs.insert(0, d);
            return false;
        }
        let d = Design::new(params, &mut self.fft);
        self.designs.insert(0, d);
        self.designs.truncate(DESIGN_CAP);
        true
    }

    /// The current kernel design, if any.
    pub fn design(&self) -> Option<&Design> {
        self.designs.first()
    }

    /// Step 1 and the forward half of step 2: the noise spectrum of a tile's window.
    pub fn noise_spectrum(&mut self, seed: u64, coord: TileCoord) -> Spectrum {
        let (q0, r0) = coord.origin();
        let (q0, r0) = (q0 - MARGIN as i64, r0 - MARGIN as i64);
        noise::fill(noise::seed_key(seed), q0, r0, WINDOW, &mut self.window);
        let mut spec = vec![Complex32::default(); SPECTRUM_LEN].into_boxed_slice();
        self.fft.forward(&self.window, &mut spec);
        Spectrum(spec)
    }

    /// The fine level only: multiply by the fine kernel, inverse FFT, keep the center.
    fn fine_from_spectrum(&mut self, params: Params, coord: TileCoord, spectrum: &Spectrum) -> Tile {
        self.set_params(params);
        let kernel = &self.designs.first().expect("design").fine_spectrum;
        for ((w, s), k) in self.work.iter_mut().zip(spectrum.0.iter()).zip(kernel.iter()) {
            *w = *s * *k;
        }
        let keep = (MARGIN - APRON)..(MARGIN + TILE + APRON);
        let mut elevation = vec![0.0; STORED * STORED].into_boxed_slice();
        self.fft.inverse(&mut self.work, keep.clone(), keep, &mut elevation);
        Tile { coord, elevation }
    }

    /// The rest of step 2, step 3 and step 4: the complete tile from its cached noise spectrum.
    pub fn tile_from_spectrum(
        &mut self,
        seed: u64,
        params: Params,
        coord: TileCoord,
        spectrum: &Spectrum,
    ) -> Tile {
        let mut tile = self.fine_from_spectrum(params, coord, spectrum);
        for level in 0..LEVELS {
            if !self.design().expect("design").levels[level].active() {
                continue;
            }
            let c = CoarseCoord::parent(level, coord);
            self.coarse_tile(seed, params, c);
            let nodes = &self.coarse[&c].1;
            levels::add_interpolated(
                &mut tile.elevation,
                coord,
                c,
                nodes,
                &self.weights[level],
                &mut self.interp,
            );
        }
        tile
    }

    /// Generates one complete tile: `tile(seed, params, coord)` is a pure function.
    pub fn tile(&mut self, seed: u64, params: Params, coord: TileCoord) -> Tile {
        let spec = self.noise_spectrum(seed, coord);
        self.tile_from_spectrum(seed, params, coord, &spec)
    }

    /// The node values of coarse tile `c`, generating (and caching) them if needed.
    pub fn coarse_tile(&mut self, seed: u64, params: Params, c: CoarseCoord) -> &[f32] {
        self.set_params(params);
        let params = params.clamped();
        if self.coarse_key != Some((seed, params)) {
            self.coarse.clear();
            self.coarse_key = Some((seed, params));
        }
        if self.noise_seed != Some(seed) {
            self.coarse_noise.clear();
            self.noise_seed = Some(seed);
        }
        self.clock += 1;
        let now = self.clock;
        if !self.coarse.contains_key(&c) {
            if !self.coarse_noise.contains_key(&c) {
                evict_oldest(&mut self.coarse_noise, COARSE_NOISE_CAP);
                let spec = levels::coarse_noise_spectrum(seed, c, &mut self.window, &mut self.fft);
                self.coarse_noise.insert(c, (now, spec));
            }
            let noise = &mut self.coarse_noise.get_mut(&c).expect("noise").0;
            *noise = now;
            let spec = &self.coarse_noise[&c].1;
            let kernel = &self.designs.first().expect("design").levels[c.level as usize].spectrum;
            let nodes = levels::coarse_tile(spec, kernel, &mut self.work, &mut self.fft);
            evict_oldest(&mut self.coarse, COARSE_CAP);
            self.coarse.insert(c, (now, nodes));
        }
        let entry = self.coarse.get_mut(&c).expect("coarse tile");
        entry.0 = now;
        &entry.1
    }
}

/// Removes the least recently used entry if the map is full.
fn evict_oldest<V>(map: &mut HashMap<CoarseCoord, (u64, V)>, cap: usize) {
    if map.len() >= cap
        && let Some(old) = map.iter().min_by_key(|(_, (t, _))| *t).map(|(k, _)| *k)
    {
        map.remove(&old);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::wrap;
    use crate::levels::{CAPRON, CSTORED, CTILE, CWINDOW, SPACING};

    const P: Params = Params::DEFAULT;

    #[test]
    fn deterministic_and_order_independent() {
        let mut g1 = Generator::new();
        let a1 = g1.tile(42, P, TileCoord::new(0, 0));
        let b1 = g1.tile(42, P, TileCoord::new(3, -2));
        let mut g2 = Generator::new();
        let b2 = g2.tile(42, P, TileCoord::new(3, -2));
        let a2 = g2.tile(42, P, TileCoord::new(0, 0));
        assert_eq!(a1.elevation, a2.elevation);
        assert_eq!(b1.elevation, b2.elevation);
        let c = g1.tile(43, P, TileCoord::new(0, 0));
        assert_ne!(a1.elevation, c.elevation);
    }

    /// Neighboring tiles agree on shared hexes, including across the boundaries of every
    /// coarse level's tiles (fine tiles 5|6 for level 1, 47|48 for level 2, 383|384 for level 3).
    #[test]
    fn neighbors_agree_on_shared_hexes() {
        let mut g = Generator::new();
        let t = TILE as isize;
        for tx in [-1, 5, 47, 383] {
            for ty in [-1, 5, 383] {
                let a = g.tile(9, P, TileCoord::new(tx, ty));
                let right = g.tile(9, P, TileCoord::new(tx + 1, ty));
                let up = g.tile(9, P, TileCoord::new(tx, ty + 1));
                let diag = g.tile(9, P, TileCoord::new(tx + 1, ty + 1));
                for i in 0..t {
                    assert!((a.get(t, i) - right.get(0, i)).abs() < 1e-4, "{tx},{ty} right {i}");
                    assert!((a.get(t - 1, i) - right.get(-1, i)).abs() < 1e-4);
                    assert!((a.get(i, t) - up.get(i, 0)).abs() < 1e-4, "{tx},{ty} up {i}");
                }
                assert!((a.get(t, t) - diag.get(0, 0)).abs() < 1e-4);
            }
        }
    }

    /// Each level's overlap-save output equals the direct linear convolution of its infinite
    /// noise field, and each tile equals the fine level plus the B-spline sum of every level.
    #[test]
    fn matches_direct_convolution() {
        for params in [P, Params { beta: 0.8, radius: 20.0 }, Params { beta: 2.6, radius: 3000.0 }] {
            let mut g = Generator::new();
            let seed = 5;
            let coord = TileCoord::new(-1, 2);
            let spec = g.noise_spectrum(seed, coord);
            let fine = g.fine_from_spectrum(params, coord, &spec);
            let tile = g.tile_from_spectrum(seed, params, coord, &spec);
            let design = Design::new(params, &mut Fft2::new(WINDOW));
            let (q0, r0) = coord.origin();
            let reach = (MARGIN - APRON) as i64;
            let fine_direct = |q: i64, r: i64| {
                let mut sum = 0.0f64;
                for dr in -reach..=reach {
                    for dq in -reach..=reach {
                        let kv = design.fine_spatial[(dr.rem_euclid(WINDOW as i64) as usize) * WINDOW
                            + dq.rem_euclid(WINDOW as i64) as usize];
                        if kv != 0.0 {
                            sum += kv * noise::noise(seed, q - dq, r - dr) as f64;
                        }
                    }
                }
                sum
            };
            // Direct coarse node value of level `l` at node (Q, R).
            let creach = (levels::CMARGIN - CAPRON) as i64;
            let node_direct = |l: usize, nq: i64, nr: i64| {
                let key = levels::level_key(seed, l);
                let k = &design.levels[l].spatial;
                let mut sum = 0.0f64;
                for dr in -creach..=creach {
                    for dq in -creach..=creach {
                        let kv = k[(dr.rem_euclid(CWINDOW as i64) as usize) * CWINDOW
                            + dq.rem_euclid(CWINDOW as i64) as usize];
                        if kv != 0.0 {
                            sum += kv * noise::noise_from_key(key, nq - dq, nr - dr) as f64;
                        }
                    }
                }
                sum
            };
            for &(lq, lr) in &[(-1i64, -1i64), (0, 0), (17, 99), (TILE as i64, 5), (127, 127)] {
                let (q, r) = (q0 + lq, r0 + lr);
                let want_fine = fine_direct(q, r);
                let got_fine = fine.get(lq as isize, lr as isize) as f64;
                assert!(
                    (got_fine - want_fine).abs() < 2e-4,
                    "{params:?} fine ({lq},{lr}): {got_fine} vs {want_fine}"
                );
                let mut want = want_fine;
                for (l, &s) in SPACING.iter().enumerate() {
                    if !design.levels[l].active() {
                        continue;
                    }
                    let (nq, nr) = (q.div_euclid(s), r.div_euclid(s));
                    let w = levels::weights(s);
                    let (wq, wr) = (w[(q - nq * s) as usize], w[(r - nr * s) as usize]);
                    for (b, wb) in wr.iter().enumerate() {
                        for (a, wa) in wq.iter().enumerate() {
                            want += (*wa * *wb) as f64 * node_direct(l, nq - 1 + a as i64, nr - 1 + b as i64);
                        }
                    }
                }
                let got = tile.get(lq as isize, lr as isize) as f64;
                assert!((got - want).abs() < 1e-3, "{params:?} total ({lq},{lr}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn coarse_tiles_cover_their_fine_tiles() {
        // Every fine tile's B-spline support must lie inside its parent coarse tile.
        for (l, &s) in SPACING.iter().enumerate() {
            for tx in [-400i32, -385, -49, -7, -1, 0, 1, 5, 6, 47, 48, 383, 384, 1000] {
                let coord = TileCoord::new(tx, 0);
                let c = CoarseCoord::parent(l, coord);
                let (q0, _) = coord.origin();
                let lo = (q0 - APRON as i64).div_euclid(s) - 1;
                let hi = (q0 + TILE as i64 + APRON as i64 - 1).div_euclid(s) + 2;
                let (n0, _) = c.origin();
                assert!(
                    lo >= n0 - CAPRON as i64 && hi < n0 - CAPRON as i64 + CSTORED as i64,
                    "level {l} tile {tx}"
                );
            }
        }
        assert_eq!(CTILE, 96);
        let _ = wrap(0, 2);
    }

    /// Interpolating the coarse levels must not imprint their grids: the variance of the
    /// hex-to-hex slope (what hillshade shows) is the same at every position within a level-1
    /// cell. This matters most for large β, where the coarse levels carry most of the slope.
    #[test]
    fn no_coarse_grid_pattern() {
        let mut g = Generator::new();
        for beta in [1.5f32, 2.5] {
            let p = Params { beta, radius: 1000.0 };
            let s = SPACING[0] as usize;
            let mut sum = vec![0.0f64; s];
            let mut count = vec![0usize; s];
            for tx in 0..6 {
                for ty in 0..6 {
                    let t = g.tile(11, p, TileCoord::new(tx, ty));
                    for r in 0..TILE as isize {
                        for q in 0..TILE as isize {
                            let d = (t.get(q + 1, r) - t.get(q, r)) as f64;
                            let k = (q as usize + TILE * tx as usize) % s;
                            sum[k] += d * d;
                            count[k] += 1;
                        }
                    }
                }
            }
            let v: Vec<f64> = sum.iter().zip(&count).map(|(a, n)| a / *n as f64).collect();
            let mean = v.iter().sum::<f64>() / s as f64;
            for (k, x) in v.iter().enumerate() {
                assert!(
                    (x / mean - 1.0).abs() < 0.03,
                    "β={beta}: slope variance at q mod {s} = {k} is {:.3}× the mean",
                    x / mean
                );
            }
        }
    }

    #[test]
    fn unit_variance() {
        let mut g = Generator::new();
        let p = Params { beta: 1.5, radius: 20.0 };
        let (mut s, mut s2, mut n) = (0.0f64, 0.0f64, 0usize);
        for tx in 0..4 {
            for ty in 0..4 {
                let t = g.tile(1, p, TileCoord::new(tx * 3, ty * 3));
                for v in t.elevation.iter() {
                    s += *v as f64;
                    s2 += (*v as f64).powi(2);
                    n += 1;
                }
            }
        }
        let mean = s / n as f64;
        let sd = (s2 / n as f64 - mean * mean).sqrt();
        assert!(mean.abs() < 0.15, "mean {mean}");
        assert!((sd - 1.0).abs() < 0.15, "sd {sd}");
    }

    #[test]
    fn changing_params_reuses_noise_spectrum() {
        let mut g = Generator::new();
        let c = TileCoord::new(2, 2);
        let spec = g.noise_spectrum(3, c);
        let p2 = Params { beta: 2.0, radius: 300.0 };
        let via_cache = g.tile_from_spectrum(3, p2, c, &spec);
        let direct = g.tile(3, p2, c);
        assert_eq!(via_cache.elevation, direct.elevation);
    }
}
