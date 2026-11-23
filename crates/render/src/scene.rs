// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scene: tile scheduling shared by the native viewer and the web demo.
//!
//! The scene decides which tiles the view needs and generates them nearest-first within a
//! per-frame time budget, so the UI never blocks. Once the visible tiles are done it keeps
//! going on a ring of tiles around the view, reaching further ahead in the direction the view
//! is moving, so a quick pan rarely reveals an empty tile. It caches noise spectra so that kernel
//! changes (β, R) only redo the cheap half of the work.
//!
//! After a seed or kernel change, the old tiles stay on screen until *every* visible tile has its
//! replacement. The replacements are uploaded to spare GPU layers and shown together in one
//! frame, so tiles of different versions are never drawn side by side. Further changes made
//! while a swap is under way (say, while a slider is dragged) wait for it to finish, and then
//! the latest one is applied.

use std::collections::{HashMap, HashSet};

use fastmap_core::{Generator, Params, Spectrum, TileCoord, hex};

use crate::{Renderer, Style, View};

/// Map extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extent {
    /// Tiles are generated on demand wherever the camera goes.
    Unbounded,
    /// Exactly `w × h` tiles, `(0, 0)..(w, h)`.
    Bounded { w: u32, h: u32 },
}

/// Timing and counters for the performance panel.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Tiles generated since start.
    pub tiles_generated: u64,
    /// Mean milliseconds per tile over the last batch: full generation (noise + both FFTs).
    pub ms_per_tile_full: f64,
    /// Mean milliseconds per tile when only the kernel changed (cached noise spectrum).
    pub ms_per_tile_cached: f64,
    /// Total compute time, in ms, for the current map once complete (bounded mode, or the current view).
    pub last_complete_ms: f64,
    /// Number of tiles in that completed set.
    pub last_complete_tiles: usize,
    /// Visible tiles still waiting to be generated.
    pub pending: usize,
    /// Off-screen tiles around the view still waiting to be generated.
    pub prefetch_pending: usize,
    /// Milliseconds the last kernel design (after a β or R change) took.
    pub ms_design: f64,
}

/// Tiles generated beyond each edge of the screen.
const PREFETCH_PAD: i32 = 2;
/// How far ahead of a moving view to prefetch, in ms of the current velocity.
const LEAD_MS: f64 = 400.0;

pub struct Scene {
    pub generator: Generator,
    pub renderer: Renderer,
    pub view: View,
    pub style: Style,
    /// Generate off-screen tiles around the view once the visible ones are done (default on).
    pub prefetch: bool,
    /// Seed and parameters last requested; adopted once no swap is under way.
    seed: u64,
    params: Params,
    /// Seed and kernel parameters of the tiles being generated.
    gen_seed: u64,
    gen_params: Params,
    extent: Extent,
    /// Version of (seed, params). A drawn tile is stale if its version differs.
    version: u64,
    tile_version: HashMap<TileCoord, u64>,
    spectra: HashMap<TileCoord, (Spectrum, u64)>,
    spectra_seed: u64,
    spectra_cap: usize,
    clock: u64,
    stats: Stats,
    batch_ms: f64,
    batch_tiles: usize,
    dirty: bool,
    /// View center and clock at the last step, and the smoothed view velocity (world units/ms).
    last_pos: Option<([f64; 2], f64)>,
    velocity: [f64; 2],
    // Exponential moving averages.
    full_ms: Ema,
    cached_ms: Ema,
}

#[derive(Clone, Copy, Debug, Default)]
struct Ema(Option<f64>);
impl Ema {
    fn push(&mut self, v: f64) -> f64 {
        let n = match self.0 {
            Some(o) => o + 0.1 * (v - o),
            None => v,
        };
        self.0 = Some(n);
        n
    }
}

impl Scene {
    pub fn new(renderer: Renderer, view: View, seed: u64, params: Params, spectra_cap: usize) -> Self {
        Self {
            generator: Generator::new(),
            renderer,
            view,
            style: Style::default(),
            prefetch: true,
            seed,
            params: params.clamped(),
            gen_seed: seed,
            gen_params: params.clamped(),
            extent: Extent::Unbounded,
            version: 1,
            tile_version: HashMap::new(),
            spectra: HashMap::new(),
            spectra_seed: seed,
            spectra_cap,
            clock: 0,
            stats: Stats::default(),
            batch_ms: 0.0,
            batch_tiles: 0,
            dirty: true,
            last_pos: None,
            velocity: [0.0, 0.0],
            full_ms: Ema::default(),
            cached_ms: Ema::default(),
        }
    }

    /// The requested seed (it may not be on screen yet).
    pub fn seed(&self) -> u64 {
        self.seed
    }
    /// The requested kernel parameters (they may not be on screen yet).
    pub fn params(&self) -> Params {
        self.params
    }
    pub fn extent(&self) -> Extent {
        self.extent
    }
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Requests a new seed. It takes effect at the next [`Scene::step`] with no swap under way.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.dirty = true;
    }

    /// Requests new kernel parameters, like [`Scene::set_seed`].
    pub fn set_params(&mut self, params: Params) {
        self.params = params.clamped();
        self.dirty = true;
    }

    /// Starts generating the requested seed and parameters, designing the kernels up front so
    /// their cost does not land on one tile.
    fn adopt_request(&mut self, now_ms: &dyn Fn() -> f64) {
        if (self.seed, self.params) == (self.gen_seed, self.gen_params) {
            return;
        }
        if self.seed != self.gen_seed {
            self.gen_seed = self.seed;
            self.spectra.clear();
            self.spectra_seed = self.gen_seed;
        }
        if self.params != self.gen_params {
            self.gen_params = self.params;
            let t0 = now_ms();
            self.generator.set_params(self.gen_params);
            self.stats.ms_design = now_ms() - t0;
        }
        self.bump();
    }

    /// Visible tiles that are drawn at an old version and have no staged replacement yet.
    fn stale_visible(&self, visible: &[TileCoord]) -> usize {
        visible
            .iter()
            .filter(|c| {
                matches!(self.tile_version.get(c), Some(&v) if v != self.version)
                    && !self.renderer.is_staged(**c)
            })
            .count()
    }

    /// Shows every staged tile at once.
    fn commit(&mut self) {
        for c in self.renderer.commit_staged() {
            self.tile_version.insert(c, self.version);
        }
        self.dirty = true;
    }

    pub fn set_extent(&mut self, extent: Extent) {
        if extent != self.extent {
            self.extent = extent;
            if let Extent::Bounded { .. } = extent {
                // Tiles outside the bounds must disappear.
                let outside: Vec<_> =
                    self.tile_version.keys().copied().filter(|c| !self.in_bounds(*c)).collect();
                for c in outside {
                    self.renderer.evict(c);
                    self.tile_version.remove(&c);
                }
                let staged_outside: Vec<_> =
                    self.renderer.resident_tiles().filter(|c| !self.in_bounds(*c)).collect();
                for c in staged_outside {
                    self.renderer.evict(c);
                }
                self.clamp_view();
            }
            self.start_batch();
            self.dirty = true;
        }
    }

    pub fn set_style(&mut self, style: Style) {
        if style != self.style {
            self.style = style;
            self.dirty = true;
        }
    }

    /// Call after changing `view` directly (pan, zoom, resize).
    pub fn view_changed(&mut self) {
        self.clamp_view();
        self.dirty = true;
    }

    /// True if a redraw is needed (the view or style changed, or new tiles arrived).
    pub fn needs_redraw(&self) -> bool {
        self.dirty
    }

    /// True while tiles (visible or prefetched) are still waiting to be generated.
    pub fn busy(&self) -> bool {
        self.stats.pending + self.stats.prefetch_pending > 0
    }

    /// True once every visible tile is up to date.
    pub fn view_ready(&self) -> bool {
        self.stats.pending == 0
    }

    fn bump(&mut self) {
        self.version += 1;
        self.start_batch();
        self.dirty = true;
    }

    fn start_batch(&mut self) {
        self.batch_ms = 0.0;
        self.batch_tiles = 0;
    }

    fn in_bounds(&self, c: TileCoord) -> bool {
        match self.extent {
            Extent::Unbounded => true,
            Extent::Bounded { w, h } => c.tx >= 0 && c.ty >= 0 && (c.tx as u32) < w && (c.ty as u32) < h,
        }
    }

    /// World-space bounding box of the bounded map, if any: `[xmin, ymin, xmax, ymax]`.
    pub fn bounds(&self) -> Option<[f64; 4]> {
        let Extent::Bounded { w, h } = self.extent else { return None };
        let t = fastmap_core::TILE as f64;
        let (qmax, rmax) = (w as f64 * t - 1.0, h as f64 * t - 1.0);
        let pts = [(0.0, 0.0), (qmax, 0.0), (0.0, rmax), (qmax, rmax)].map(|(q, r)| hex::to_cartesian(q, r));
        let xs = pts.map(|p| p.0);
        let ys = pts.map(|p| p.1);
        Some([
            xs.iter().copied().fold(f64::MAX, f64::min) - 1.0,
            ys.iter().copied().fold(f64::MAX, f64::min) - 1.0,
            xs.iter().copied().fold(f64::MIN, f64::max) + 1.0,
            ys.iter().copied().fold(f64::MIN, f64::max) + 1.0,
        ])
    }

    /// Centers the camera on the bounded map and zooms to fit it (or resets to the origin).
    pub fn fit_view(&mut self) {
        if let Some([x0, y0, x1, y1]) = self.bounds() {
            self.view.center = [(x0 + x1) / 2.0, (y0 + y1) / 2.0];
            let zx = self.view.width as f64 / (x1 - x0);
            let zy = self.view.height as f64 / (y1 - y0);
            self.view.zoom = (zx.min(zy) * 0.95).clamp(View::MIN_ZOOM, View::MAX_ZOOM);
        } else {
            // Unbounded: center on tile (0, 0) and show about 1600 hexes across (or as many as
            // the zoom limit allows), so the largest features are in view.
            self.view.center = View::tile_center(TileCoord::new(0, 0));
            let span = 1600.0;
            self.view.zoom = (self.view.width.min(self.view.height * 2) as f64 / span)
                .clamp(View::MIN_ZOOM, View::MAX_ZOOM);
        }
        self.view_changed();
    }

    fn clamp_view(&mut self) {
        // Never let the view need more tiles than the GPU can hold.
        while self.view.zoom < View::MAX_ZOOM
            && self.view.visible_tiles().len() > self.renderer.capacity() * 3 / 4
        {
            self.view.zoom *= 1.1;
        }
        if let Some([x0, y0, x1, y1]) = self.bounds() {
            self.view.center[0] = self.view.center[0].clamp(x0, x1);
            self.view.center[1] = self.view.center[1].clamp(y0, y1);
        }
    }

    /// Tiles the current view wants: the visible ones first, then the prefetch ring. Each is
    /// paired with whether it is visible. Within each group, tiles nearest the point the view
    /// is heading for come first.
    fn wanted(&self) -> Vec<(TileCoord, bool)> {
        let lead = self.lead();
        let target = [self.view.center[0] + lead[0], self.view.center[1] + lead[1]];
        let dist = |c: &TileCoord| {
            let p = View::tile_center(*c);
            (p[0] - target[0]).powi(2) + (p[1] - target[1]).powi(2)
        };
        let mut visible = self.view.visible_tiles();
        visible.retain(|c| self.in_bounds(*c));
        visible.sort_by(|a, b| dist(a).total_cmp(&dist(b)));
        let mut out: Vec<(TileCoord, bool)> = visible.iter().map(|&c| (c, true)).collect();
        if self.prefetch {
            let mut seen: HashSet<TileCoord> = visible.into_iter().collect();
            let mut ahead = self.view;
            ahead.center = target;
            let mut extra: Vec<TileCoord> = self.view.tiles_near(PREFETCH_PAD);
            extra.extend(ahead.tiles_near(1));
            extra.retain(|c| self.in_bounds(*c) && seen.insert(*c));
            extra.sort_by(|a, b| dist(a).total_cmp(&dist(b)));
            out.extend(extra.into_iter().map(|c| (c, false)));
        }
        out.truncate(self.renderer.capacity());
        out
    }

    /// World offset the view is expected to move over the next `LEAD_MS`, capped at one
    /// screen.
    fn lead(&self) -> [f64; 2] {
        let [vx, vy] = self.velocity;
        let (dx, dy) = (vx * LEAD_MS, vy * LEAD_MS);
        let len = (dx * dx + dy * dy).sqrt();
        let cap = self.view.width.max(self.view.height) as f64 / self.view.zoom;
        if len > cap { [dx * cap / len, dy * cap / len] } else { [dx, dy] }
    }

    fn track_velocity(&mut self, t: f64) {
        let c = self.view.center;
        let v = match self.last_pos {
            Some((p, t0)) if t > t0 && t - t0 < 250.0 => [(c[0] - p[0]) / (t - t0), (c[1] - p[1]) / (t - t0)],
            _ => [0.0, 0.0],
        };
        // Light smoothing: follows a drag within a few frames, and settles soon after it stops.
        self.velocity = [0.5 * (self.velocity[0] + v[0]), 0.5 * (self.velocity[1] + v[1])];
        self.last_pos = Some((c, t));
    }

    /// Generates tiles until `budget_ms` has elapsed (at least one if any are pending).
    /// `now_ms` is a monotonic clock in milliseconds. Returns the number of tiles generated.
    pub fn step(&mut self, queue: &wgpu::Queue, budget_ms: f64, now_ms: &dyn Fn() -> f64) -> usize {
        let start = now_ms();
        self.track_velocity(start);
        let wanted = self.wanted();
        let visible: Vec<TileCoord> = wanted.iter().filter(|t| t.1).map(|t| t.0).collect();
        // Finish a swap whose stale tiles have all been replaced (or panned out of view), then
        // take up any new request.
        let mut stale = self.stale_visible(&visible);
        if stale == 0 && self.renderer.has_staged() {
            self.commit();
        }
        if stale == 0 {
            self.adopt_request(now_ms);
            stale = self.stale_visible(&visible);
        }
        let mut todo: Vec<(TileCoord, bool)> = wanted
            .iter()
            .copied()
            .filter(|(c, _)| self.tile_version.get(c) != Some(&self.version) && !self.renderer.is_staged(*c))
            .collect();
        // Visible tiles first; within each group, tiles with nothing drawn yet before stale ones.
        // The sort is stable, so the distance order within each class is kept.
        todo.sort_by_key(|&(c, visible)| (!visible, self.tile_version.contains_key(&c)));
        let wanted: HashSet<TileCoord> = wanted.into_iter().map(|(c, _)| c).collect();
        let mut made = 0;
        let mut made_visible = 0;
        for (c, visible) in todo.iter().copied() {
            // During a swap every new tile is staged, so it needs a layer of its own.
            let mut stage = stale > 0;
            if (stage || !self.renderer.is_resident(c)) && self.renderer.is_full() {
                // Make room by evicting the farthest tile the view no longer wants.
                let view = self.view;
                let far = self
                    .renderer
                    .resident_tiles()
                    .filter(|k| !wanted.contains(k))
                    .max_by(|a, b| view.tile_distance2(*a).total_cmp(&view.tile_distance2(*b)));
                match far {
                    Some(far) => {
                        self.renderer.evict(far);
                        self.tile_version.remove(&far);
                    }
                    // A view so large that old and new tiles don't both fit: replace in place.
                    None if stage && self.renderer.is_resident(c) => stage = false,
                    // Cannot happen without staging: `wanted` holds at most `capacity` tiles.
                    None => break,
                }
            }
            let t0 = now_ms();
            self.clock += 1;
            let cached = self.spectra_seed == self.gen_seed && self.spectra.contains_key(&c);
            let uncached = if cached {
                None
            } else {
                let s = self.generator.noise_spectrum(self.gen_seed, c);
                self.insert_spectrum(c, s)
            };
            let tile = match &uncached {
                Some(s) => self.generator.tile_from_spectrum(self.gen_seed, self.gen_params, c, s),
                None => {
                    let entry = self.spectra.get_mut(&c).expect("spectrum present");
                    entry.1 = self.clock;
                    self.generator.tile_from_spectrum(self.gen_seed, self.gen_params, c, &entry.0)
                }
            };
            let was_stale = visible && self.tile_version.contains_key(&c);
            if stage {
                self.renderer.upload_staged(queue, &tile);
            } else {
                self.renderer.upload(queue, &tile, &self.view);
                self.tile_version.insert(c, self.version);
                self.tile_version.retain(|k, _| self.renderer.is_resident(*k));
            }
            if was_stale && stale > 0 {
                stale -= 1;
                if stale == 0 {
                    // The last stale visible tile has its replacement: show them all.
                    self.commit();
                }
            }
            let dt = now_ms() - t0;
            if cached {
                self.stats.ms_per_tile_cached = self.cached_ms.push(dt);
            } else {
                self.stats.ms_per_tile_full = self.full_ms.push(dt);
            }
            if visible {
                // The view-fill statistic counts only the tiles on screen.
                self.batch_ms += dt;
                self.batch_tiles += 1;
                made_visible += 1;
            }
            self.stats.tiles_generated += 1;
            made += 1;
            self.dirty = true;
            if now_ms() - start >= budget_ms {
                break;
            }
        }
        let visible_todo = todo.iter().filter(|t| t.1).count();
        self.stats.pending = visible_todo.saturating_sub(made_visible);
        self.stats.prefetch_pending = (todo.len() - visible_todo) - (made - made_visible);
        if made_visible > 0 && self.stats.pending == 0 {
            self.stats.last_complete_ms = self.batch_ms;
            self.stats.last_complete_tiles = self.batch_tiles;
            self.start_batch();
        }
        made
    }

    /// Caches a tile's noise spectrum. When full, the spectrum farthest from the view is
    /// dropped, so the tiles on screen keep theirs; `s` itself is not kept if it is the
    /// farthest. Returns the spectrum if it was not cached.
    fn insert_spectrum(&mut self, c: TileCoord, s: Spectrum) -> Option<Spectrum> {
        if self.spectra.len() >= self.spectra_cap {
            let view = self.view;
            let far = self
                .spectra
                .keys()
                .copied()
                .max_by(|a, b| view.tile_distance2(*a).total_cmp(&view.tile_distance2(*b)));
            match far {
                Some(old) if view.tile_distance2(old) > view.tile_distance2(c) => {
                    self.spectra.remove(&old);
                }
                _ => return Some(s),
            }
        }
        self.spectra.insert(c, (s, self.clock));
        None
    }

    /// Renders the current frame into `target`.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        self.dirty = false;
        let style = self.style;
        let view = self.view;
        self.renderer.render(device, queue, target, &view, &style)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
            let adapter = instance.request_adapter(&Default::default()).await.ok()?;
            let limits = wgpu::Limits {
                max_texture_array_layers: adapter.limits().max_texture_array_layers.min(1024),
                ..wgpu::Limits::downlevel_webgl2_defaults()
            };
            adapter
                .request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() })
                .await
                .ok()
        })
    }

    /// Versions of the visible tiles that are drawn.
    fn drawn_versions(s: &Scene) -> HashSet<u64> {
        s.view
            .visible_tiles()
            .iter()
            .filter(|c| s.renderer.is_resident(**c))
            .map(|c| s.tile_version[c])
            .collect()
    }

    /// Steps one tile at a time until the view shows the current version, checking after every
    /// step that tiles of different versions are never on screen together. Returns the steps.
    fn swap(s: &mut Scene, queue: &wgpu::Queue, pan_at: Option<usize>) -> usize {
        let t0 = Instant::now();
        let now = move || t0.elapsed().as_secs_f64() * 1e3;
        for step in 1..20_000 {
            s.step(queue, 0.0, &now);
            if pan_at == Some(step) {
                s.view.pan_pixels(250.0, -120.0);
                s.view_changed();
            }
            let v = drawn_versions(s);
            assert!(v.len() <= 1, "versions {v:?} on screen together after {step} steps");
            if v == HashSet::from([s.version])
                && s.view_ready()
                && (s.params, s.seed) == (s.gen_params, s.gen_seed)
            {
                return step;
            }
        }
        panic!("swap did not finish");
    }

    #[test]
    fn changes_swap_all_visible_tiles_at_once() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; test skipped");
            return;
        };
        let renderer = Renderer::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, 1024);
        let mut s = Scene::new(renderer, View::new(1600, 900), 3, Params::default(), 96);
        s.view.zoom = 1.5;
        s.view_changed();
        swap(&mut s, &queue, None);
        let visible = s.view.visible_tiles().len();
        assert!(visible > 20, "{visible} visible tiles");

        for (i, beta) in [2.5f32, 1.0, 1.7].into_iter().enumerate() {
            s.set_params(Params { beta, ..Params::default() });
            let pan = (i == 1).then_some(visible / 3);
            let steps = swap(&mut s, &queue, pan);
            // One tile per step, so the swap really spanned many frames.
            assert!(steps >= visible / 2, "{steps} steps for {visible} tiles");
        }

        // A new seed is swapped in the same way.
        s.set_seed(4);
        swap(&mut s, &queue, None);

        // Requests made during a swap wait for it; then only the latest is applied.
        s.set_params(Params { beta: 2.0, ..Params::default() });
        let t0 = Instant::now();
        let now = move || t0.elapsed().as_secs_f64() * 1e3;
        s.step(&queue, 0.0, &now);
        let v = s.version;
        s.set_params(Params { beta: 2.2, ..Params::default() });
        s.step(&queue, 0.0, &now);
        s.set_params(Params { beta: 2.4, ..Params::default() });
        assert_eq!(s.version, v, "a request was adopted mid-swap");
        swap(&mut s, &queue, None);
        assert_eq!(s.gen_params.beta, 2.4);
        assert_eq!(s.version, v + 1, "the intermediate request should have been skipped");
    }
}
