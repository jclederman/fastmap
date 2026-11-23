// SPDX-License-Identifier: AGPL-3.0-or-later
//! 2D real FFT over an `N × N` window. Plans and scratch buffers are built once and reused.
//!
//! Spectra are stored *transposed*: `HALF = N/2 + 1` rows (one per frequency along `q`), each
//! holding `N` complex values (frequencies along `r`). This keeps both FFT passes on
//! contiguous memory.
//!
//! The pass along `q` handles two real rows with one complex FFT: row `a` goes in the real
//! part and row `b` in the imaginary part, and the two spectra are separated afterwards using
//! Hermitian symmetry. All rows then go through a single batched FFT call, which is much faster
//! than many separate small real FFTs (especially in WebAssembly). The separation writes
//! straight into the transposed layout, so no separate transpose pass is needed.

use std::ops::Range;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

pub struct Fft2 {
    n: usize,
    half: usize,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    /// Packed rows: `N/2` complex rows of length `N`.
    pack: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl Fft2 {
    /// `n` must be a multiple of 4.
    pub fn new(n: usize) -> Self {
        assert!(n >= 4 && n.is_multiple_of(4), "FFT size {n} must be a multiple of 4");
        let mut planner = FftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let scratch_len = fwd.get_inplace_scratch_len().max(inv.get_inplace_scratch_len());
        Self {
            n,
            half: n / 2 + 1,
            fwd,
            inv,
            pack: vec![Complex32::default(); n * n / 2],
            scratch: vec![Complex32::default(); scratch_len],
        }
    }

    /// Number of complex values in a spectrum (`HALF × N`).
    pub fn spectrum_len(&self) -> usize {
        self.n * self.half
    }

    /// Forward FFT of a row-major `N × N` real array into a transposed spectrum.
    pub fn forward(&mut self, input: &[f32], out: &mut [Complex32]) {
        let n = self.n;
        assert_eq!(input.len(), n * n);
        assert_eq!(out.len(), n * self.half);
        // Pass 1: along q, two rows per complex FFT, all in one batched call.
        for (pair, z) in input.chunks_exact(2 * n).zip(self.pack.chunks_exact_mut(n)) {
            let (a, b) = pair.split_at(n);
            simd::interleave(a, b, z);
        }
        self.fwd.process_with_scratch(&mut self.pack, &mut self.scratch);
        for (j, z) in self.pack.chunks_exact(n).enumerate() {
            simd::split_transposed(z, out, 2 * j);
        }
        // Pass 2: along r, HALF FFTs of length N, all in place.
        self.fwd.process_with_scratch(out, &mut self.scratch);
    }

    /// Inverse FFT of a transposed spectrum. Only rows `rows.start..rows.end` and columns
    /// `cols.start..cols.end` of the real result are written, row-major, to `out`.
    /// `spec` is used as scratch space and its contents are destroyed. No normalization is
    /// applied (fold `1/N²` into the kernel).
    pub fn inverse(
        &mut self,
        spec: &mut [Complex32],
        rows: Range<usize>,
        cols: Range<usize>,
        out: &mut [f32],
    ) {
        let n = self.n;
        assert_eq!(spec.len(), n * self.half);
        assert!(rows.end <= n && cols.end <= n);
        let width = cols.len();
        assert_eq!(out.len(), rows.len() * width);
        // Pass 1: along r.
        self.inv.process_with_scratch(spec, &mut self.scratch);
        // Pass 2: along q, two wanted rows per complex FFT. An odd last row is paired with zeros.
        let pairs = rows.len().div_ceil(2);
        let pack = &mut self.pack[..pairs * n];
        for (j, z) in pack.chunks_exact_mut(n).enumerate() {
            let r = rows.start + 2 * j;
            simd::hermitian_pair(spec, n, r, r + 1 < rows.end, z);
        }
        self.inv.process_with_scratch(pack, &mut self.scratch);
        for (j, z) in pack.chunks_exact(n).enumerate() {
            let z = &z[cols.clone()];
            let (a, rest) = out[2 * j * width..].split_at_mut(width);
            if 2 * j + 1 < rows.len() {
                simd::deinterleave(z, a, &mut rest[..width]);
            } else {
                for (o, v) in a.iter_mut().zip(z) {
                    *o = v.re;
                }
            }
        }
    }
}

/// The per-element loops around the batched FFTs. The SIMD versions perform exactly the same
/// float operations as the scalar ones, so results do not depend on the path taken.
mod simd {
    use super::Complex32;

    /// `z[i] = a[i] + i·b[i]`.
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    pub fn interleave(a: &[f32], b: &[f32], z: &mut [Complex32]) {
        for ((z, &a), &b) in z.iter_mut().zip(a).zip(b) {
            *z = Complex32::new(a, b);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub fn interleave(a: &[f32], b: &[f32], z: &mut [Complex32]) {
        use core::arch::wasm32::*;
        let (a4, _) = a.as_chunks::<4>();
        let (b4, _) = b.as_chunks::<4>();
        let (z4, _) = z.as_chunks_mut::<4>();
        // N is a multiple of 4, so there is no tail.
        for ((a, b), z) in a4.iter().zip(b4).zip(z4.iter_mut()) {
            // SAFETY: 16-byte and 32-byte arrays; unaligned loads and stores are allowed.
            unsafe {
                let va = v128_load(a.as_ptr() as *const v128);
                let vb = v128_load(b.as_ptr() as *const v128);
                let p = z.as_mut_ptr() as *mut v128;
                v128_store(p, i32x4_shuffle::<0, 4, 1, 5>(va, vb));
                v128_store(p.add(1), i32x4_shuffle::<2, 6, 3, 7>(va, vb));
            }
        }
    }

    /// Separates the spectra of the two real rows packed in `z` and writes them to columns
    /// `c` and `c + 1` of the transposed spectrum: `A[k] = (Z[k] + conj Z[N−k]) / 2`,
    /// `B[k] = (Z[k] − conj Z[N−k]) / 2i`, for `k = 0..=N/2`.
    #[inline(always)]
    fn split_one(z: &[Complex32], out: &mut [Complex32], c: usize, k: usize) {
        let n = z.len();
        let (zk, zm) = (z[k], z[(n - k) % n].conj());
        let d = (zk - zm) * 0.5;
        out[k * n + c] = (zk + zm) * 0.5;
        out[k * n + c + 1] = Complex32::new(d.im, -d.re);
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    pub fn split_transposed(z: &[Complex32], out: &mut [Complex32], c: usize) {
        for k in 0..=z.len() / 2 {
            split_one(z, out, c, k);
        }
    }

    /// Two frequencies at a time: each vector holds two complex values.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub fn split_transposed(z: &[Complex32], out: &mut [Complex32], c: usize) {
        use core::arch::wasm32::*;
        let n = z.len();
        split_one(z, out, c, 0);
        let half = f32x4_splat(0.5);
        let conj = f32x4(1.0, -1.0, 1.0, -1.0);
        assert!(out.len() >= (n / 2 + 1) * n && c + 1 < n);
        let (zp, op) = (z.as_ptr(), out.as_mut_ptr());
        // k = 1..=N/2 in pairs (k, k+1); N/2 is even.
        for k in (1..n / 2).step_by(2) {
            // SAFETY: k + 1 ≤ N/2 < N and N − k − 1 ≥ N/2, so both loads are in `z`; the
            // stores hit (k·N + c, +1) and ((k+1)·N + c, +1), inside `out` by the assert above.
            unsafe {
                let vk = v128_load(zp.add(k) as *const v128); // Z[k], Z[k+1]
                let vm = v128_load(zp.add(n - k - 1) as *const v128); // Z[N−k−1], Z[N−k]
                let vm = f32x4_mul(i64x2_shuffle::<1, 0>(vm, vm), conj); // conj Z[N−k], conj Z[N−k−1]
                let a = f32x4_mul(f32x4_add(vk, vm), half);
                let d = f32x4_mul(f32x4_sub(vk, vm), half);
                let b = f32x4_mul(i32x4_shuffle::<1, 0, 3, 2>(d, d), conj); // (d.im, −d.re)
                v128_store(op.add(k * n + c) as *mut v128, i64x2_shuffle::<0, 2>(a, b));
                v128_store(op.add((k + 1) * n + c) as *mut v128, i64x2_shuffle::<1, 3>(a, b));
            }
        }
    }

    /// Builds the full complex spectrum `Z = A + i·B` of rows `r` and `r + 1` (or `B = 0`)
    /// from the half spectra in the transposed `spec`, using `A[N−k] = conj A[k]`.
    #[inline(always)]
    fn pair_one(spec: &[Complex32], n: usize, r: usize, two: bool, z: &mut [Complex32], k: usize) {
        let mut a = spec[k * n + r];
        let mut b = if two { spec[k * n + r + 1] } else { Complex32::default() };
        if k == 0 || k == n / 2 {
            // The input is the spectrum of real data, so DC and Nyquist are real.
            a.im = 0.0;
            b.im = 0.0;
        }
        z[k] = Complex32::new(a.re - b.im, a.im + b.re);
        if k != 0 && k != n / 2 {
            z[n - k] = Complex32::new(a.re + b.im, b.re - a.im);
        }
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    pub fn hermitian_pair(spec: &[Complex32], n: usize, r: usize, two: bool, z: &mut [Complex32]) {
        for k in 0..=n / 2 {
            pair_one(spec, n, r, two, z, k);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub fn hermitian_pair(spec: &[Complex32], n: usize, r: usize, two: bool, z: &mut [Complex32]) {
        use core::arch::wasm32::*;
        pair_one(spec, n, r, two, z, 0);
        pair_one(spec, n, r, two, z, n / 2);
        if !two {
            for k in 1..n / 2 {
                pair_one(spec, n, r, two, z, k);
            }
            return;
        }
        assert!(z.len() == n && spec.len() >= (n / 2) * n + r + 2);
        let sign = f32x4(-1.0, 1.0, 1.0, -1.0);
        let (sp, zp) = (spec.as_ptr(), z.as_mut_ptr());
        for k in 1..n / 2 {
            // SAFETY: k < N/2, so the load (k·N + r, +1) is inside `spec` by the assert, and
            // both stores (k and N − k) are inside `z`.
            unsafe {
                let v = v128_load(sp.add(k * n + r) as *const v128); // a.re a.im b.re b.im
                let lo = i32x4_shuffle::<0, 1, 0, 2>(v, v); // a.re a.im a.re b.re
                let hi = i32x4_shuffle::<3, 2, 3, 1>(v, v); // b.im b.re b.im a.im
                let s = f32x4_add(lo, f32x4_mul(hi, sign));
                v128_store64_lane::<0>(s, zp.add(k) as *mut u64);
                v128_store64_lane::<1>(s, zp.add(n - k) as *mut u64);
            }
        }
    }

    /// `a[i] = z[i].re`, `b[i] = z[i].im`.
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    pub fn deinterleave(z: &[Complex32], a: &mut [f32], b: &mut [f32]) {
        for ((v, a), b) in z.iter().zip(a).zip(b) {
            *a = v.re;
            *b = v.im;
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub fn deinterleave(z: &[Complex32], a: &mut [f32], b: &mut [f32]) {
        use core::arch::wasm32::*;
        let (z4, zt) = z.as_chunks::<4>();
        let (a4, at) = a.as_chunks_mut::<4>();
        let (b4, bt) = b.as_chunks_mut::<4>();
        for ((z, a), b) in z4.iter().zip(a4.iter_mut()).zip(b4.iter_mut()) {
            // SAFETY: 32-byte and 16-byte arrays; unaligned loads and stores are allowed.
            unsafe {
                let p = z.as_ptr() as *const v128;
                let (lo, hi) = (v128_load(p), v128_load(p.add(1)));
                v128_store(a.as_mut_ptr() as *mut v128, i32x4_shuffle::<0, 2, 4, 6>(lo, hi));
                v128_store(b.as_mut_ptr() as *mut v128, i32x4_shuffle::<1, 3, 5, 7>(lo, hi));
            }
        }
        for ((v, a), b) in zt.iter().zip(at).zip(bt) {
            *a = v.re;
            *b = v.im;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let n = 32;
        let mut f = Fft2::new(n);
        let orig: Vec<f32> = (0..n * n).map(|i| ((i * 7919) % 101) as f32 - 50.0).collect();
        let mut spec = vec![Complex32::default(); f.spectrum_len()];
        // An even and an odd number of rows, and a column range with a tail.
        for (rows, cols) in [(2..n - 2, 3..n - 3), (1..n - 4, 0..n), (5..6, 7..20)] {
            f.forward(&orig, &mut spec);
            let mut out = vec![0.0; rows.len() * cols.len()];
            f.inverse(&mut spec, rows.clone(), cols.clone(), &mut out);
            let scale = 1.0 / (n * n) as f32;
            for (i, r) in rows.clone().enumerate() {
                for (j, c) in cols.clone().enumerate() {
                    let got = out[i * cols.len() + j] * scale;
                    assert!((got - orig[r * n + c]).abs() < 1e-3, "{got} vs {}", orig[r * n + c]);
                }
            }
        }
    }

    /// The forward transform matches a direct DFT, in the transposed layout.
    #[test]
    fn matches_direct_dft() {
        let n = 16;
        let mut f = Fft2::new(n);
        let x: Vec<f32> = (0..n * n).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
        let mut spec = vec![Complex32::default(); f.spectrum_len()];
        f.forward(&x, &mut spec);
        for kq in 0..=n / 2 {
            for kr in 0..n {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for r in 0..n {
                    for q in 0..n {
                        let ph = -2.0 * std::f64::consts::PI * ((kq * q + kr * r) as f64) / n as f64;
                        re += x[r * n + q] as f64 * ph.cos();
                        im += x[r * n + q] as f64 * ph.sin();
                    }
                }
                let got = spec[kq * n + kr];
                assert!(
                    (got.re as f64 - re).abs() < 1e-3 && (got.im as f64 - im).abs() < 1e-3,
                    "({kq}, {kr}): {got} vs {re} {im}"
                );
            }
        }
    }
}
