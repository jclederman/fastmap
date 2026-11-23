// SPDX-License-Identifier: AGPL-3.0-or-later
//! Color schemes, built entirely at compile time into one 256-entry lookup row each.
//!
//! A *sea-level* scheme splits its row in two: entries 0–127 are water (deep → shallow) and
//! entries 128–255 are land (shore → peaks). A *continuous* scheme maps the raw elevation
//! over the whole row and ignores sea level.

type Rgb = [f32; 3];

/// How the map is colored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Palette {
    /// Blue water, green lowlands, brown and gray mountains, snow.
    #[default]
    Terrain,
    /// Atlas-style elevation tints: pale bathymetry, then green, yellow, tan and brown land.
    Atlas,
    /// An old paper map: faded blue-green water, cream land darkening to sepia.
    Parchment,
    /// Raw elevation, black to white.
    Grayscale,
    /// Raw elevation on the perceptually uniform viridis scale.
    Viridis,
}

impl Palette {
    pub const ALL: [Palette; 5] =
        [Palette::Terrain, Palette::Atlas, Palette::Parchment, Palette::Grayscale, Palette::Viridis];

    /// Short lowercase name, used in URLs and on the command line.
    pub const fn name(self) -> &'static str {
        match self {
            Palette::Terrain => "terrain",
            Palette::Atlas => "atlas",
            Palette::Parchment => "parchment",
            Palette::Grayscale => "gray",
            Palette::Viridis => "viridis",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    /// The next scheme, wrapping around.
    pub fn next(self) -> Self {
        Self::ALL[(self as usize + 1) % Self::ALL.len()]
    }

    /// True if the scheme ignores sea level and colors the raw elevation.
    pub const fn continuous(self) -> bool {
        matches!(self, Palette::Grayscale | Palette::Viridis)
    }
}

const TERRAIN_WATER: [(f32, Rgb); 4] = [
    (0.0, [10.0, 24.0, 58.0]),
    (0.55, [24.0, 70.0, 128.0]),
    (0.9, [52.0, 120.0, 176.0]),
    (1.0, [92.0, 162.0, 204.0]),
];
const TERRAIN_LAND: [(f32, Rgb); 7] = [
    (0.0, [200.0, 190.0, 142.0]),
    (0.04, [128.0, 168.0, 96.0]),
    (0.25, [82.0, 132.0, 70.0]),
    (0.5, [58.0, 98.0, 56.0]),
    (0.68, [128.0, 112.0, 86.0]),
    (0.84, [150.0, 145.0, 138.0]),
    (1.0, [246.0, 246.0, 250.0]),
];

const ATLAS_WATER: [(f32, Rgb); 4] = [
    (0.0, [44.0, 88.0, 156.0]),
    (0.5, [92.0, 146.0, 204.0]),
    (0.88, [150.0, 196.0, 232.0]),
    (1.0, [186.0, 222.0, 244.0]),
];
const ATLAS_LAND: [(f32, Rgb); 7] = [
    (0.0, [104.0, 160.0, 92.0]),
    (0.12, [146.0, 186.0, 108.0]),
    (0.3, [214.0, 210.0, 136.0]),
    (0.48, [222.0, 178.0, 108.0]),
    (0.66, [184.0, 124.0, 72.0]),
    (0.84, [140.0, 92.0, 62.0]),
    (1.0, [236.0, 232.0, 228.0]),
];

const PARCHMENT_WATER: [(f32, Rgb); 3] =
    [(0.0, [96.0, 124.0, 130.0]), (0.7, [140.0, 164.0, 160.0]), (1.0, [176.0, 194.0, 180.0])];
const PARCHMENT_LAND: [(f32, Rgb); 5] = [
    (0.0, [236.0, 222.0, 182.0]),
    (0.3, [222.0, 202.0, 156.0]),
    (0.6, [196.0, 166.0, 120.0]),
    (0.85, [156.0, 122.0, 86.0]),
    (1.0, [118.0, 88.0, 62.0]),
];

const GRAY: [(f32, Rgb); 2] = [(0.0, [0.0, 0.0, 0.0]), (1.0, [255.0, 255.0, 255.0])];

const VIRIDIS: [(f32, Rgb); 9] = [
    (0.0, [68.0, 1.0, 84.0]),
    (0.125, [72.0, 40.0, 120.0]),
    (0.25, [62.0, 73.0, 137.0]),
    (0.375, [49.0, 104.0, 142.0]),
    (0.5, [38.0, 130.0, 142.0]),
    (0.625, [31.0, 158.0, 137.0]),
    (0.75, [53.0, 183.0, 121.0]),
    (0.875, [110.0, 206.0, 88.0]),
    (1.0, [253.0, 231.0, 37.0]),
];

const fn ramp<const N: usize>(stops: &[(f32, Rgb); N], t: f32) -> [u8; 4] {
    let mut i = 0;
    while i + 2 < N && t > stops[i + 1].0 {
        i += 1;
    }
    let (t0, c0) = stops[i];
    let (t1, c1) = stops[i + 1];
    let mut w = (t - t0) / (t1 - t0);
    if w < 0.0 {
        w = 0.0;
    }
    if w > 1.0 {
        w = 1.0;
    }
    let mut out = [255u8; 4];
    let mut k = 0;
    while k < 3 {
        out[k] = (c0[k] + (c1[k] - c0[k]) * w + 0.5) as u8;
        k += 1;
    }
    out
}

type Row = [[u8; 4]; 256];

const fn split<const A: usize, const B: usize>(water: &[(f32, Rgb); A], land: &[(f32, Rgb); B]) -> Row {
    let mut row = [[0u8; 4]; 256];
    let mut i = 0;
    while i < 128 {
        row[i] = ramp(water, i as f32 / 127.0);
        row[128 + i] = ramp(land, i as f32 / 127.0);
        i += 1;
    }
    row
}

const fn continuous<const N: usize>(stops: &[(f32, Rgb); N]) -> Row {
    let mut row = [[0u8; 4]; 256];
    let mut i = 0;
    while i < 256 {
        row[i] = ramp(stops, i as f32 / 255.0);
        i += 1;
    }
    row
}

/// One row per [`Palette`], in declaration order.
pub const PALETTES: [Row; Palette::ALL.len()] = [
    split(&TERRAIN_WATER, &TERRAIN_LAND),
    split(&ATLAS_WATER, &ATLAS_LAND),
    split(&PARCHMENT_WATER, &PARCHMENT_LAND),
    continuous(&GRAY),
    continuous(&VIRIDIS),
];
