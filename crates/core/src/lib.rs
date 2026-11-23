// SPDX-License-Identifier: AGPL-3.0-or-later
//! # fastmap-core
//!
//! Seamless, unbounded fractal elevation on a hex lattice, generated tile by tile with FFT
//! convolution, with features up to thousands of hexes across.
//!
//! ```
//! use fastmap_core::{Generator, Params, TileCoord};
//! let mut generator = Generator::new();
//! let tile = generator.tile(42, Params::default(), TileCoord::new(0, 0));
//! let elevation_of_first_hex = tile.get(0, 0); // in units of σ
//! # assert!(elevation_of_first_hex.is_finite());
//! ```
//!
//! See `docs/method.md` in the repository for how and why it works.

pub mod fft;
pub mod hex;
pub mod kernel;
pub mod levels;
pub mod noise;
pub mod tile;

pub use kernel::{Params, sea_level};
pub use levels::{CoarseCoord, Design};
pub use tile::{APRON, Generator, MARGIN, STORED, Spectrum, TILE, Tile, TileCoord, WINDOW};
