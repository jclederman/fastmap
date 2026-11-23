// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prints the target and realized power spectra along the x axis for one parameter set, and
//! each level's share of the variance:
//! `cargo run -p fastmap-core --release --example spectrum -- 1.5 1000` (β, R).

use fastmap_core::{Design, Params, fft::Fft2, tile::WINDOW};

fn main() {
    let args: Vec<f32> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let d = Params::DEFAULT;
    let params = Params { beta: *args.first().unwrap_or(&d.beta), radius: *args.get(1).unwrap_or(&d.radius) };
    let design = Design::new(params, &mut Fft2::new(WINDOW));
    let coarse: Vec<f64> = design.levels.iter().map(|l| l.variance).collect();
    println!(
        "β = {}, R = {}: variance shares fine {:.4}, coarse {:?}",
        params.beta,
        params.radius,
        1.0 - coarse.iter().sum::<f64>(),
        coarse.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>()
    );
    println!("{:>10} {:>12} {:>12} {:>8}", "κ", "target", "realized", "ratio");
    for i in 0..40 {
        let k = 1e-4 * 10f64.powf(i as f64 / 39.0 * 3.9);
        let (t, r) = (design.target_power(k), design.realized_power(k));
        println!("{k:>10.5} {t:>12.4e} {r:>12.4e} {:>8.3}", r / t);
    }
}
