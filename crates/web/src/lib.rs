// SPDX-License-Identifier: AGPL-3.0-or-later
//! FastMap browser demo.
//!
//! The page (`web/index.html`) is static HTML/CSS. Its small loader imports either the
//! `webgpu` or the `webgl` build of this crate and calls [`start`]. Everything else is here:
//! the GPU setup, the controls, pointer and wheel input, URL state, the frame loop and the
//! performance panel.

use std::cell::RefCell;
use std::rc::Rc;

use fastmap_core::{Params, sea_level};
use fastmap_render::{Extent, Palette, Renderer, Scene, View, preferred_format};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{
    AddEventListenerOptions, Document, Event, HtmlAnchorElement, HtmlCanvasElement, HtmlElement,
    HtmlInputElement, HtmlSelectElement, PointerEvent, UrlSearchParams, WheelEvent, Window,
};

const BACKEND: &str = if cfg!(feature = "webgl") { "WebGL2" } else { "WebGPU" };
/// Per-frame time budget for tile generation, in ms. That leaves headroom for 60 fps.
const BUDGET_MS: f64 = 7.0;
const MAX_BOUNDED: u32 = 16;

fn window() -> Window {
    web_sys::window().expect("window")
}
fn document() -> Document {
    window().document().expect("document")
}
fn el<T: JsCast>(id: &str) -> T {
    document()
        .get_element_by_id(id)
        .unwrap_or_else(|| panic!("missing #{id}"))
        .dyn_into::<T>()
        .unwrap_or_else(|_| panic!("#{id} has the wrong type"))
}
fn input(id: &str) -> HtmlInputElement {
    el(id)
}
fn set_text(id: &str, text: &str) {
    if let Some(e) = document().get_element_by_id(id) {
        e.set_text_content(Some(text));
    }
}
fn now() -> f64 {
    window().performance().expect("performance").now()
}

fn random_seed() -> u64 {
    (js_sys::Math::random() * 1e9) as u64
}

/// R within its valid range.
fn clamp_radius(r: f32) -> f32 {
    r.clamp(Params::RADIUS_RANGE.0, Params::RADIUS_RANGE.1)
}

/// R for display: one decimal below 100 hexes, whole hexes above.
fn format_radius(r: f32) -> String {
    if r < 100.0 { format!("{r:.1}") } else { format!("{r:.0}") }
}

/// Everything the controls and the URL describe.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Settings {
    seed: u64,
    beta: f32,
    radius: f32,
    land: f64,
    bounded: bool,
    w: u32,
    h: u32,
    palette: Palette,
    shade: bool,
    outlines: bool,
    tiles: bool,
    relief: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            seed: 1,
            beta: Params::DEFAULT.beta,
            radius: Params::DEFAULT.radius,
            land: 0.4,
            bounded: false,
            w: 4,
            h: 3,
            palette: Palette::Terrain,
            shade: false,
            outlines: false,
            tiles: false,
            relief: 6.0,
        }
    }
}

impl Settings {
    fn from_url(q: &UrlSearchParams) -> (Self, Option<[f64; 3]>) {
        let d = Self::default();
        let get = |k: &str| q.get(k);
        let num = |k: &str| get(k).and_then(|v| v.parse::<f64>().ok());
        let flag = |k: &str, def: bool| get(k).map(|v| v == "1").unwrap_or(def);
        let s = Self {
            // Without a seed in the URL, every visit starts on a new map.
            seed: get("seed").and_then(|v| v.parse().ok()).unwrap_or_else(random_seed),
            beta: num("beta").map(|v| v as f32).unwrap_or(d.beta),
            radius: num("r").map(|v| clamp_radius(v as f32)).unwrap_or(d.radius),
            land: num("land").map(|v| (v / 100.0).clamp(0.0, 1.0)).unwrap_or(d.land),
            bounded: get("mode").as_deref() == Some("bounded"),
            w: num("w").map(|v| (v as u32).clamp(1, MAX_BOUNDED)).unwrap_or(d.w),
            h: num("h").map(|v| (v as u32).clamp(1, MAX_BOUNDED)).unwrap_or(d.h),
            palette: get("color").and_then(|v| Palette::from_name(&v)).unwrap_or(d.palette),
            shade: flag("shade", d.shade),
            outlines: flag("outlines", d.outlines),
            tiles: flag("tiles", d.tiles),
            relief: num("relief").map(|v| v as f32).unwrap_or(d.relief),
        };
        let cam = match (num("x"), num("y"), num("z")) {
            (Some(x), Some(y), Some(z)) => Some([x, y, z]),
            _ => None,
        };
        (s, cam)
    }

    fn to_query(self, view: &View) -> String {
        let b = |v: bool| if v { "1" } else { "0" };
        let mut q = format!(
            "?seed={}&beta={:.2}&r={}&land={:.0}&mode={}",
            self.seed,
            self.beta,
            format_radius(self.radius),
            self.land * 100.0,
            if self.bounded { "bounded" } else { "unbounded" }
        );
        if self.bounded {
            q += &format!("&w={}&h={}", self.w, self.h);
        }
        q += &format!(
            "&color={}&shade={}&outlines={}&tiles={}&relief={:.1}&x={:.1}&y={:.1}&z={:.3}",
            self.palette.name(),
            b(self.shade),
            b(self.outlines),
            b(self.tiles),
            self.relief,
            view.center[0],
            view.center[1],
            view.zoom
        );
        q
    }

    /// Pushes these settings into the HTML controls.
    fn to_ui(self) {
        input("seed").set_value(&self.seed.to_string());
        input("beta").set_value(&format!("{:.2}", self.beta));
        input("radius").set_value(&format!("{:.3}", (self.radius as f64).log10()));
        input("land").set_value(&format!("{:.0}", self.land * 100.0));
        input("mode-bounded").set_checked(self.bounded);
        input("mode-unbounded").set_checked(!self.bounded);
        input("bw").set_value(&self.w.to_string());
        input("bh").set_value(&self.h.to_string());
        el::<HtmlSelectElement>("color").set_value(self.palette.name());
        input("shade").set_checked(self.shade);
        input("outlines").set_checked(self.outlines);
        input("tiles").set_checked(self.tiles);
        input("relief").set_value(&format!("{:.1}", self.relief));
        self.show_values();
    }

    /// Reads the HTML controls.
    fn from_ui(prev: Self) -> Self {
        let f = |id: &str, d: f64| input(id).value().parse::<f64>().unwrap_or(d);
        Self {
            seed: input("seed").value().trim().parse().unwrap_or(prev.seed),
            beta: f("beta", prev.beta as f64) as f32,
            // The R slider is logarithmic: its value is log₁₀ R.
            radius: clamp_radius(10f64.powf(f("radius", (prev.radius as f64).log10())) as f32),
            land: f("land", prev.land * 100.0) / 100.0,
            bounded: input("mode-bounded").checked(),
            w: (f("bw", prev.w as f64) as u32).clamp(1, MAX_BOUNDED),
            h: (f("bh", prev.h as f64) as u32).clamp(1, MAX_BOUNDED),
            palette: Palette::from_name(&el::<HtmlSelectElement>("color").value()).unwrap_or(prev.palette),
            shade: input("shade").checked(),
            outlines: input("outlines").checked(),
            tiles: input("tiles").checked(),
            relief: f("relief", prev.relief as f64) as f32,
        }
    }

    fn show_values(self) {
        set_text("beta-val", &format!("{:.2}", self.beta));
        set_text("radius-val", &format_radius(self.radius));
        set_text("land-val", &format!("{:.0}%", self.land * 100.0));
        set_text("relief-val", &format!("{:.1}", self.relief));
        let dims = document().get_element_by_id("bounded-dims");
        if let Some(d) = dims.and_then(|d| d.dyn_into::<HtmlElement>().ok()) {
            d.set_hidden(!self.bounded);
        }
    }
}

struct App {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    max_dim: u32,
    canvas: HtmlCanvasElement,
    scene: Scene,
    settings: Settings,
    pointers: Vec<(i32, f64, f64)>,
    last_frame: f64,
    frame_ms: f64,
    last_panel: f64,
    url_dirty: bool,
    first_complete: Option<f64>,
}

impl App {
    fn apply(&mut self, s: Settings) {
        let prev = self.settings;
        self.settings = s;
        self.scene.set_seed(s.seed);
        self.scene.set_params(Params { beta: s.beta, radius: s.radius });
        let mut style = self.scene.style;
        style.sea_level = sea_level(s.land);
        style.palette = s.palette;
        style.hillshade = s.shade;
        style.hex_outlines = s.outlines;
        style.tile_boundaries = s.tiles;
        style.relief = s.relief;
        self.scene.set_style(style);
        let extent = if s.bounded { Extent::Bounded { w: s.w, h: s.h } } else { Extent::Unbounded };
        if extent != self.scene.extent() {
            self.scene.set_extent(extent);
            if s.bounded && (!prev.bounded || prev.w != s.w || prev.h != s.h) {
                self.scene.fit_view();
            }
        }
        s.show_values();
        self.url_dirty = true;
    }

    fn device_pixel_ratio() -> f64 {
        window().device_pixel_ratio().clamp(0.5, 4.0)
    }

    /// Matches the drawing buffer to the canvas's CSS size × devicePixelRatio.
    fn resize(&mut self) {
        let dpr = Self::device_pixel_ratio();
        let w = ((self.canvas.client_width() as f64 * dpr).round() as u32).clamp(1, self.max_dim);
        let h = ((self.canvas.client_height() as f64 * dpr).round() as u32).clamp(1, self.max_dim);
        if w != self.config.width || h != self.config.height {
            self.canvas.set_width(w);
            self.canvas.set_height(h);
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
            self.scene.view.width = w;
            self.scene.view.height = h;
            self.scene.view_changed();
        }
    }

    fn draw(&mut self) {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                let view = frame.texture.create_view(&Default::default());
                let cmd = self.scene.render(&self.device, &self.queue, &view);
                self.queue.submit([cmd]);
                self.queue.present(frame);
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
            }
            _ => {}
        }
    }

    fn frame(&mut self) {
        let t = now();
        if self.last_frame > 0.0 {
            let dt = t - self.last_frame;
            self.frame_ms += 0.1 * (dt - self.frame_ms);
        }
        self.last_frame = t;
        self.resize();
        self.scene.step(&self.queue, BUDGET_MS, &now);
        if self.scene.needs_redraw() {
            self.draw();
        }
        if self.first_complete.is_none() && self.scene.view_ready() && self.scene.stats().tiles_generated > 0
        {
            self.first_complete = Some(now());
        }
        if t - self.last_panel > 250.0 {
            self.last_panel = t;
            self.update_panel();
            if self.url_dirty {
                self.url_dirty = false;
                let q = self.settings.to_query(&self.scene.view);
                let _ =
                    window().history().and_then(|h| h.replace_state_with_url(&JsValue::NULL, "", Some(&q)));
            }
        }
    }

    fn update_panel(&self) {
        let s = self.scene.stats();
        let ms = |v: f64| if v > 0.0 { format!("{v:.2} ms") } else { "–".into() };
        set_text("st-backend", BACKEND);
        set_text("st-tile", &ms(s.ms_per_tile_full));
        set_text("st-cached", &ms(s.ms_per_tile_cached));
        set_text(
            "st-map",
            &if s.last_complete_tiles > 0 {
                format!("{} tiles in {:.0} ms", s.last_complete_tiles, s.last_complete_ms)
            } else {
                "–".into()
            },
        );
        set_text("st-pending", &s.pending.to_string());
        set_text("st-frame", &format!("{:.0} fps", 1000.0 / self.frame_ms.max(0.1)));
        set_text("st-drawn", &self.scene.renderer.drawn_count().to_string());
        set_text("st-resident", &self.scene.renderer.resident_count().to_string());
        if let Some(t) = self.first_complete {
            set_text("st-load", &format!("{:.2} s", t / 1000.0));
        }
        set_text("st-size", &wasm_transfer_size());
    }
}

/// Bytes transferred for the wasm binary, from the Resource Timing API.
fn wasm_transfer_size() -> String {
    let entries = window().performance().map(|p| p.get_entries_by_type("resource")).unwrap_or_default();
    for e in entries.iter() {
        let name =
            js_sys::Reflect::get(&e, &"name".into()).ok().and_then(|v| v.as_string()).unwrap_or_default();
        if name.contains("_bg.wasm") {
            let get =
                |k: &str| js_sys::Reflect::get(&e, &k.into()).ok().and_then(|v| v.as_f64()).unwrap_or(0.0);
            let (wire, raw) = (get("encodedBodySize"), get("decodedBodySize"));
            if raw > 0.0 {
                // Served uncompressed (or from cache) there is only one size to show.
                if wire <= 0.0 || wire >= raw {
                    return format!("{:.0} KB", raw / 1024.0);
                }
                return format!("{:.0} KB ({:.0} KB uncompressed)", wire / 1024.0, raw / 1024.0);
            }
        }
    }
    "–".into()
}

type FrameCallback = Rc<RefCell<Option<Closure<dyn FnMut()>>>>;

fn listen<E: JsCast + 'static>(
    target: &web_sys::EventTarget,
    kind: &str,
    passive: bool,
    f: impl FnMut(E) + 'static,
) {
    let mut f = f;
    let cb = Closure::<dyn FnMut(Event)>::new(move |e: Event| f(e.unchecked_into::<E>()));
    let opts = AddEventListenerOptions::new();
    opts.set_passive(passive);
    target
        .add_event_listener_with_callback_and_add_event_listener_options(
            kind,
            cb.as_ref().unchecked_ref(),
            &opts,
        )
        .expect("listener");
    cb.forget();
}

/// Entry point, called by the loader in `index.html`. It fails (and the loader falls back to
/// the WebGL2 build) if no adapter or device can be created.
#[wasm_bindgen]
pub async fn start() -> Result<(), JsValue> {
    let canvas: HtmlCanvasElement = el("map");
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = if cfg!(feature = "webgl") { wgpu::Backends::GL } else { wgpu::Backends::BROWSER_WEBGPU };
    let instance = wgpu::Instance::new(desc);
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .map_err(|e| JsValue::from_str(&format!("surface: {e}")))?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("adapter: {e}")))?;
    let al = adapter.limits();
    let limits = wgpu::Limits {
        max_texture_array_layers: al.max_texture_array_layers.min(1024),
        max_texture_dimension_2d: al.max_texture_dimension_2d,
        ..wgpu::Limits::downlevel_webgl2_defaults()
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("fastmap"),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("device: {e}")))?;

    let caps = surface.get_capabilities(&adapter);
    if caps.formats.is_empty() {
        return Err("surface has no formats".into());
    }
    let format = preferred_format(&caps.formats);
    let max_dim = device.limits().max_texture_dimension_2d;
    let dpr = App::device_pixel_ratio();
    let w = ((canvas.client_width() as f64 * dpr) as u32).clamp(1, max_dim);
    let h = ((canvas.client_height() as f64 * dpr) as u32).clamp(1, max_dim);
    canvas.set_width(w);
    canvas.set_height(h);
    let mut config = surface.get_default_config(&adapter, w, h).ok_or("surface config")?;
    config.format = format;
    surface.configure(&device, &config);

    let renderer = Renderer::new(&device, &queue, format, 1024);
    let query = UrlSearchParams::new_with_str(&window().location().search().unwrap_or_default())?;
    let (settings, cam) = Settings::from_url(&query);
    let scene = Scene::new(renderer, View::new(w, h), settings.seed, Params::default(), 96);
    let app = Rc::new(RefCell::new(App {
        surface,
        device,
        queue,
        config,
        max_dim,
        canvas: canvas.clone(),
        scene,
        settings: Settings { bounded: !settings.bounded, ..settings },
        pointers: Vec::new(),
        last_frame: 0.0,
        frame_ms: 16.7,
        last_panel: 0.0,
        url_dirty: false,
        first_complete: None,
    }));
    {
        let mut a = app.borrow_mut();
        settings.to_ui();
        a.apply(settings);
        a.scene.fit_view();
        if let Some([x, y, z]) = cam {
            a.scene.view.center = [x, y];
            a.scene.view.zoom = z.clamp(View::MIN_ZOOM, View::MAX_ZOOM);
            a.scene.view_changed();
        }
        a.url_dirty = false;
    }

    wire_controls(&app);
    wire_pointer(&app, &canvas);

    // Frame loop.
    let tick: FrameCallback = Rc::new(RefCell::new(None));
    let tick2 = tick.clone();
    let app2 = app.clone();
    *tick.borrow_mut() = Some(Closure::new(move || {
        app2.borrow_mut().frame();
        let _ = window().request_animation_frame(tick2.borrow().as_ref().unwrap().as_ref().unchecked_ref());
    }));
    window().request_animation_frame(tick.borrow().as_ref().unwrap().as_ref().unchecked_ref())?;

    if let Some(body) = document().body() {
        let _ = body.class_list().add_1("ready");
    }
    Ok(())
}

fn wire_controls(app: &Rc<RefCell<App>>) {
    let ids = [
        "seed",
        "beta",
        "radius",
        "land",
        "mode-unbounded",
        "mode-bounded",
        "bw",
        "bh",
        "color",
        "shade",
        "outlines",
        "tiles",
        "relief",
    ];
    for id in ids {
        let a1 = app.clone();
        let target: web_sys::EventTarget = el(id);
        let kind = if id == "seed" { "change" } else { "input" };
        listen(&target, kind, true, move |_: Event| {
            let mut a = a1.borrow_mut();
            let s = Settings::from_ui(a.settings);
            a.apply(s);
        });
        if id == "mode-unbounded" || id == "mode-bounded" || id == "color" {
            let app = app.clone();
            listen(&target, "change", true, move |_: Event| {
                let mut a = app.borrow_mut();
                let s = Settings::from_ui(a.settings);
                a.apply(s);
            });
        }
    }

    let a2 = app.clone();
    listen(&el::<web_sys::EventTarget>("regenerate"), "click", true, move |_: Event| {
        let mut a = a2.borrow_mut();
        let seed = random_seed();
        let s = Settings { seed, ..a.settings };
        input("seed").set_value(&seed.to_string());
        a.apply(s);
    });

    let a4 = app.clone();
    listen(&el::<web_sys::EventTarget>("save"), "click", true, move |_: Event| {
        let mut a = a4.borrow_mut();
        // Draw and read back in the same task, while the drawing buffer is still valid.
        a.scene.view_changed();
        a.draw();
        if let Ok(url) = a.canvas.to_data_url() {
            let link: HtmlAnchorElement = document().create_element("a").expect("a").unchecked_into();
            link.set_href(&url);
            link.set_download(&format!("fastmap-{}.png", a.settings.seed));
            link.click();
        }
    });
}

fn wire_pointer(app: &Rc<RefCell<App>>, canvas: &HtmlCanvasElement) {
    let target: &web_sys::EventTarget = canvas.as_ref();
    let dpr = App::device_pixel_ratio;

    let a = app.clone();
    let c = canvas.clone();
    listen(target, "pointerdown", true, move |e: PointerEvent| {
        let _ = c.set_pointer_capture(e.pointer_id());
        a.borrow_mut().pointers.push((e.pointer_id(), e.offset_x() as f64, e.offset_y() as f64));
    });

    let a = app.clone();
    listen(target, "pointermove", true, move |e: PointerEvent| {
        let mut a = a.borrow_mut();
        let id = e.pointer_id();
        let (x, y) = (e.offset_x() as f64, e.offset_y() as f64);
        let Some(i) = a.pointers.iter().position(|p| p.0 == id) else { return };
        let k = dpr();
        match a.pointers.len() {
            1 => {
                let (_, px, py) = a.pointers[0];
                a.scene.view.pan_pixels((x - px) * k, (y - py) * k);
            }
            2 => {
                // Pinch: zoom about the midpoint by the change in finger distance, and pan with the midpoint.
                let other = a.pointers[1 - i];
                let old = a.pointers[i];
                let d_old = ((old.1 - other.1).powi(2) + (old.2 - other.2).powi(2)).sqrt().max(1.0);
                let d_new = ((x - other.1).powi(2) + (y - other.2).powi(2)).sqrt().max(1.0);
                let (mx0, my0) = ((old.1 + other.1) / 2.0, (old.2 + other.2) / 2.0);
                let (mx1, my1) = ((x + other.1) / 2.0, (y + other.2) / 2.0);
                a.scene.view.pan_pixels((mx1 - mx0) * k, (my1 - my0) * k);
                a.scene.view.zoom_at(mx1 * k, my1 * k, d_new / d_old);
            }
            _ => {}
        }
        a.pointers[i] = (id, x, y);
        a.scene.view_changed();
        a.url_dirty = true;
    });

    for kind in ["pointerup", "pointercancel", "pointerleave"] {
        let a = app.clone();
        listen(target, kind, true, move |e: PointerEvent| {
            a.borrow_mut().pointers.retain(|p| p.0 != e.pointer_id());
        });
    }

    let a = app.clone();
    listen(target, "wheel", false, move |e: WheelEvent| {
        e.prevent_default();
        let mut a = a.borrow_mut();
        let dy = match e.delta_mode() {
            1 => e.delta_y() * 16.0,
            2 => e.delta_y() * 400.0,
            _ => e.delta_y(),
        };
        let k = dpr();
        a.scene.view.zoom_at(e.offset_x() as f64 * k, e.offset_y() as f64 * k, (-dy * 0.0015).exp());
        a.scene.view_changed();
        a.url_dirty = true;
    });

    let a = app.clone();
    listen(target, "dblclick", true, move |e: web_sys::MouseEvent| {
        let mut a = a.borrow_mut();
        let k = dpr();
        a.scene.view.zoom_at(e.offset_x() as f64 * k, e.offset_y() as f64 * k, 2.0);
        a.scene.view_changed();
    });
}
