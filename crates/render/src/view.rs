// SPDX-License-Identifier: AGPL-3.0-or-later
//! Camera: maps between screen pixels, world (cartesian) coordinates and tiles.
//!
//! World coordinates are kept in f64 and only converted to f32 relative to the camera
//! center, so precision holds far from the origin.

use fastmap_core::{TILE, TileCoord, hex};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// World position (cartesian, hex spacing = 1) at the center of the screen.
    pub center: [f64; 2],
    /// Pixels per hex spacing.
    pub zoom: f64,
    /// Viewport size in physical pixels.
    pub width: u32,
    pub height: u32,
}

impl View {
    pub const MIN_ZOOM: f64 = 1.0;
    pub const MAX_ZOOM: f64 = 256.0;
    /// Safety cap on the number of tiles a view may request.
    pub const MAX_VISIBLE: usize = 1024;

    pub fn new(width: u32, height: u32) -> Self {
        Self { center: [0.0, 0.0], zoom: 3.0, width: width.max(1), height: height.max(1) }
    }

    /// World units → clip space scale.
    pub fn clip_scale(&self) -> [f32; 2] {
        [(2.0 * self.zoom / self.width as f64) as f32, (2.0 * self.zoom / self.height as f64) as f32]
    }

    /// Cartesian offset of a tile's origin hex from the camera center.
    pub fn tile_offset(&self, c: TileCoord) -> [f32; 2] {
        let (q, r) = c.origin();
        let (x, y) = hex::to_cartesian(q as f64, r as f64);
        [(x - self.center[0]) as f32, (y - self.center[1]) as f32]
    }

    /// World position of the center of a tile.
    pub fn tile_center(c: TileCoord) -> [f64; 2] {
        let (q, r) = c.origin();
        let h = (TILE as f64 - 1.0) / 2.0;
        let (x, y) = hex::to_cartesian(q as f64 + h, r as f64 + h);
        [x, y]
    }

    pub fn tile_distance2(&self, c: TileCoord) -> f64 {
        let p = Self::tile_center(c);
        (p[0] - self.center[0]).powi(2) + (p[1] - self.center[1]).powi(2)
    }

    /// Pixel (origin at the top-left, y down) → world.
    pub fn screen_to_world(&self, px: f64, py: f64) -> [f64; 2] {
        [
            self.center[0] + (px - self.width as f64 / 2.0) / self.zoom,
            self.center[1] - (py - self.height as f64 / 2.0) / self.zoom,
        ]
    }

    /// Moves the camera by a drag of `(dx, dy)` pixels.
    pub fn pan_pixels(&mut self, dx: f64, dy: f64) {
        self.center[0] -= dx / self.zoom;
        self.center[1] += dy / self.zoom;
    }

    /// Multiplies the zoom by `factor`, keeping the world point under pixel `(px, py)` fixed.
    pub fn zoom_at(&mut self, px: f64, py: f64, factor: f64) {
        let before = self.screen_to_world(px, py);
        self.zoom = (self.zoom * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        let after = self.screen_to_world(px, py);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
    }

    /// Range of tile coordinates `(tx_min..=tx_max, ty_min..=ty_max)` that intersect the
    /// screen, widened by `pad` tiles.
    pub fn tile_range(&self, pad: i32) -> ([i32; 2], [i32; 2]) {
        let corners = [
            self.screen_to_world(0.0, 0.0),
            self.screen_to_world(self.width as f64, 0.0),
            self.screen_to_world(0.0, self.height as f64),
            self.screen_to_world(self.width as f64, self.height as f64),
        ];
        let (mut qmin, mut qmax, mut rmin, mut rmax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for [x, y] in corners {
            let (q, r) = hex::to_axial(x, y);
            qmin = qmin.min(q);
            qmax = qmax.max(q);
            rmin = rmin.min(r);
            rmax = rmax.max(r);
        }
        let t = TILE as f64;
        let to_tile = |v: f64| (v / t).floor().clamp(i32::MIN as f64 / 2.0, i32::MAX as f64 / 2.0) as i32;
        (
            [to_tile(qmin - 1.0) - pad, to_tile(qmax + 1.0) + pad],
            [to_tile(rmin - 1.0) - pad, to_tile(rmax + 1.0) + pad],
        )
    }

    /// Tiles that intersect the screen (plus `pad` tiles around it), nearest the center first.
    pub fn tiles_near(&self, pad: i32) -> Vec<TileCoord> {
        let ([x0, x1], [y0, y1]) = self.tile_range(pad);
        let mut v: Vec<TileCoord> = Vec::new();
        'outer: for ty in y0..=y1 {
            for tx in x0..=x1 {
                v.push(TileCoord::new(tx, ty));
                if v.len() >= 4 * Self::MAX_VISIBLE {
                    break 'outer;
                }
            }
        }
        v.sort_by(|a, b| self.tile_distance2(*a).total_cmp(&self.tile_distance2(*b)));
        v.truncate(Self::MAX_VISIBLE);
        v
    }

    /// Tiles that intersect the screen.
    pub fn visible_tiles(&self) -> Vec<TileCoord> {
        self.tiles_near(0)
    }
}
