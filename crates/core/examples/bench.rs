// SPDX-License-Identifier: AGPL-3.0-or-later
//! Times tile generation: `cargo run -p fastmap-core --release --example bench`.

use fastmap_core::{CoarseCoord, Generator, Params, TileCoord};
use std::time::Instant;

fn main() {
    let mut g = Generator::new();
    let p = Params::default();
    let t = Instant::now();
    g.set_params(p);
    println!("kernel design (β, R):   {:7.3} ms", t.elapsed().as_secs_f64() * 1e3);

    // A coarse tile of each level, from scratch.
    for level in 0..3 {
        let t = Instant::now();
        std::hint::black_box(g.coarse_tile(1, p, CoarseCoord { level: level as u8, cx: 7, cy: -3 }));
        println!("coarse tile, level {}:   {:7.3} ms", level + 1, t.elapsed().as_secs_f64() * 1e3);
    }

    // 64 fine tiles inside one level-1 coarse tile (6 × 6 fine tiles each), so the coarse cost
    // is amortized as it is when panning.
    let n = 64;
    let coords: Vec<_> = (0..n).map(|i| TileCoord::new(i % 8, i / 8)).collect();
    for &c in &coords {
        std::hint::black_box(g.tile(1, p, c)); // warm the coarse cache
    }
    let t = Instant::now();
    let specs: Vec<_> = coords.iter().map(|&c| g.noise_spectrum(1, c)).collect();
    let fwd = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    let t = Instant::now();
    for (c, s) in coords.iter().zip(&specs) {
        std::hint::black_box(g.tile_from_spectrum(1, p, *c, s));
    }
    let inv = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("noise + forward FFT:    {fwd:7.3} ms/tile");
    println!("kernel + inverse FFT + coarse interpolation: {inv:7.3} ms/tile");
    println!(
        "total (coarse cached):  {:7.3} ms/tile ({:.1} Mhex/s)",
        fwd + inv,
        (fastmap_core::TILE * fastmap_core::TILE) as f64 / (fwd + inv) / 1e3
    );

    // Cold: a fresh generator over a 6 × 6 block, coarse tiles included.
    let mut g = Generator::new();
    g.set_params(p);
    let t = Instant::now();
    for i in 0..36 {
        std::hint::black_box(g.tile(2, p, TileCoord::new(i % 6, i / 6)));
    }
    println!("cold 6×6 block:         {:7.3} ms/tile", t.elapsed().as_secs_f64() * 1e3 / 36.0);
}
