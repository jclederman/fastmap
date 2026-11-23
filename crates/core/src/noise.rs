// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stateless, counter-based white noise.
//!
//! Every hex's noise value is a pure function of `(seed, q, r)`. Tiles can therefore be
//! generated in any order, in parallel, or regenerated later, and always agree. Only integer
//! operations and exactly rounded single float operations are used, so every device produces
//! the same bits.
//!
//! The value at `(q, r)` is `lowbias32(column_key(q) ^ row_key(r))`, mapped to a float. The
//! column and row keys are 64-bit hashes of the coordinate, folded to 32 bits, computed once per
//! column and once per row of a window. That leaves one cheap 32-bit hash per hex, which runs
//! four hexes at a time with SIMD. The final mixer breaks up the row × column structure of the
//! XOR (checked by the four-corner correlation in the tests).

/// SplitMix64 finalizer: a fast bijective 64-bit mixer with good avalanche.
#[inline(always)]
pub const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Chris Wellons' `lowbias32`: a bijective 32-bit mixer with two multiplies.
#[inline(always)]
pub const fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// Pre-mixes a user seed so that nearby seeds give unrelated maps.
#[inline(always)]
pub const fn seed_key(seed: u64) -> u64 {
    mix64(seed ^ 0x9e37_79b9_7f4a_7c15)
}

const ROW_SALT: u64 = 0xa54f_f53a_5f1d_36f1;

/// 32-bit key of column `q`.
#[inline(always)]
pub const fn column_key(seed_key: u64, q: i64) -> u32 {
    (mix64(seed_key.wrapping_add(q as u64)) >> 32) as u32
}

/// 32-bit key of row `r`, from a stream independent of the column keys.
#[inline(always)]
pub const fn row_key(seed_key: u64, r: i64) -> u32 {
    (mix64(mix64(seed_key ^ ROW_SALT).wrapping_add(r as u64)) >> 32) as u32
}

/// Step between the 2²⁴ output levels: uniform on `[−√3, √3)`, so zero mean and unit variance.
const SCALE: f32 = 2.0 * 1.732_050_8 / 16_777_216.0;
/// Offset that centers each level in its step.
const OFFSET: f32 = 0.5 * SCALE - 1.732_050_8;

/// Noise value from a column key and a row key.
#[inline(always)]
pub const fn sample(column_key: u32, row_key: u32) -> f32 {
    (lowbias32(column_key ^ row_key) >> 8) as f32 * SCALE + OFFSET
}

/// Noise value of hex `(q, r)` for a pre-mixed key (see [`seed_key`]).
#[inline]
pub const fn noise_from_key(seed_key: u64, q: i64, r: i64) -> f32 {
    sample(column_key(seed_key, q), row_key(seed_key, r))
}

/// Noise value of hex `(q, r)` for `seed`.
#[inline]
pub const fn noise(seed: u64, q: i64, r: i64) -> f32 {
    noise_from_key(seed_key(seed), q, r)
}

/// Fills the row-major `n × n` window whose first hex is `(q0, r0)` (row = `r`):
/// `out[i·n + j] = noise_from_key(seed_key, q0 + j, r0 + i)`.
pub fn fill(seed_key: u64, q0: i64, r0: i64, n: usize, out: &mut [f32]) {
    assert_eq!(out.len(), n * n);
    let cols: Vec<u32> = (0..n).map(|j| column_key(seed_key, q0 + j as i64)).collect();
    for (i, row) in out.chunks_exact_mut(n).enumerate() {
        fill_row(&cols, row_key(seed_key, r0 + i as i64), row);
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
fn fill_row(cols: &[u32], rk: u32, out: &mut [f32]) {
    for (v, &ck) in out.iter_mut().zip(cols) {
        *v = sample(ck, rk);
    }
}

/// Four hexes per instruction. The same operations as [`sample`], so the same bits.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn fill_row(cols: &[u32], rk: u32, out: &mut [f32]) {
    use core::arch::wasm32::*;
    let (c4, ct) = cols.as_chunks::<4>();
    let (o4, ot) = out.as_chunks_mut::<4>();
    let rkv = u32x4_splat(rk);
    let (m1, m2) = (u32x4_splat(0x7feb_352d), u32x4_splat(0x846c_a68b));
    let (scale, offset) = (f32x4_splat(SCALE), f32x4_splat(OFFSET));
    for (c, o) in c4.iter().zip(o4.iter_mut()) {
        // SAFETY: `c` and `o` are 16-byte arrays; the loads and stores are unaligned-safe.
        unsafe {
            let mut x = v128_xor(v128_load(c.as_ptr() as *const v128), rkv);
            x = v128_xor(x, u32x4_shr(x, 16));
            x = i32x4_mul(x, m1);
            x = v128_xor(x, u32x4_shr(x, 15));
            x = i32x4_mul(x, m2);
            x = v128_xor(x, u32x4_shr(x, 16));
            // < 2²⁴ after the shift, so the signed conversion is exact.
            let f = f32x4_convert_i32x4(u32x4_shr(x, 8));
            v128_store(o.as_mut_ptr() as *mut v128, f32x4_add(f32x4_mul(f, scale), offset));
        }
    }
    for (v, &ck) in ot.iter_mut().zip(ct) {
        *v = sample(ck, rk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments() {
        let n = 200_000;
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        for i in 0..n {
            let v = noise(7, i % 500, i / 500) as f64;
            s += v;
            s2 += v * v;
        }
        let mean = s / n as f64;
        let var = s2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.01, "var {var}");
    }

    #[test]
    fn seeds_differ() {
        assert_ne!(noise(1, 3, 4), noise(2, 3, 4));
        assert_eq!(noise(1, 3, 4), noise(1, 3, 4));
    }

    /// The window fill (SIMD where available) matches the per-hex definition bit for bit,
    /// including a width that leaves a scalar tail.
    #[test]
    fn fill_matches_definition() {
        for (n, q0, r0) in [(256usize, -5_000_000_000i64, 77i64), (13, 3, -9)] {
            let key = seed_key(42);
            let mut w = vec![0.0f32; n * n];
            fill(key, q0, r0, n, &mut w);
            for i in 0..n {
                for j in 0..n {
                    let want = noise_from_key(key, q0 + j as i64, r0 + i as i64);
                    assert_eq!(w[i * n + j].to_bits(), want.to_bits(), "({i}, {j})");
                }
            }
        }
    }

    /// Pins the exact values, so any change to the map for a given seed is deliberate.
    #[test]
    fn known_values() {
        let got = [noise(1, 0, 0), noise(1, 1, 0), noise(1, 0, 1), noise(12345, -7, 1 << 40)];
        let bits = got.map(f32::to_bits);
        assert_eq!(bits, KNOWN, "values {got:?}");
    }
    const KNOWN: [u32; 4] = [0xbfd7_f2ac, 0x3ea5_b780, 0xbf2f_0e20, 0x3f24_1024];

    /// No row × column structure: the product of the four corners of random rectangles has
    /// zero mean (it is about 0.6 if the mixer is left out).
    #[test]
    fn no_rectangle_correlation() {
        let key = seed_key(9);
        let mut state = 1u64;
        let mut next = |m: i64| {
            state = mix64(state);
            (state % m as u64) as i64
        };
        let n = 400_000;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let (q1, q2, r1, r2) = (next(1 << 20), next(1 << 20), next(1 << 20), next(1 << 20));
            let v = |q, r| noise_from_key(key, q, r) as f64;
            sum += v(q1, r1) * v(q2, r1) * v(q1, r2) * v(q2, r2);
        }
        let mean = sum / n as f64;
        assert!(mean.abs() < 5.0 / (n as f64).sqrt(), "four-corner correlation {mean}");
    }
}
