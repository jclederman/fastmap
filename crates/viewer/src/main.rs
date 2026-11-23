// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native FastMap viewer.
//!
//! Interactive: `cargo run -p fastmap-viewer --release`
//!
//! Headless screenshot: `cargo run -p fastmap-viewer --release -- --screenshot out.png [options]`
//!
//! Run with `--help` for the controls and options.

use std::sync::Arc;
use std::time::Instant;

use fastmap_core::{Params, sea_level};
use fastmap_render::{Extent, Palette, Renderer, Scene, View, preferred_format};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const HELP: &str = "\
fastmap-viewer: native FastMap viewer

Controls:
  drag            pan                 wheel        zoom
  space           new random seed     b            toggle bounded 4×3 map
  [ / ]           β −/+ 0.1           - / =        radius ÷/× 1.25
  , / .           land −/+ 5%         h o t g      hillshade / outlines / tile boundaries / next palette
  f               fit view            esc          quit

Options (interactive and screenshot):
  --seed N  --beta B  --radius R  --land F (0-1)  --zoom Z (px/hex)  --center X,Y
  --bounded WxH  --palette terrain|atlas|parchment|gray|viridis  --grayscale  --no-hillshade  --no-outlines  --boundaries
  --screenshot FILE.png  --size WxH (default 1600x900)
";

struct Options {
    seed: u64,
    params: Params,
    land: f64,
    zoom: Option<f64>,
    center: Option<[f64; 2]>,
    extent: Extent,
    palette: Palette,
    hillshade: bool,
    outlines: bool,
    boundaries: bool,
    screenshot: Option<String>,
    size: (u32, u32),
}

fn parse_pair<T: std::str::FromStr>(s: &str, sep: char) -> Option<(T, T)> {
    let (a, b) = s.split_once(sep)?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

fn parse_args() -> Options {
    let mut o = Options {
        seed: 1,
        params: Params::default(),
        land: 0.4,
        zoom: None,
        center: None,
        extent: Extent::Unbounded,
        palette: Palette::Terrain,
        hillshade: true,
        outlines: true,
        boundaries: false,
        screenshot: None,
        size: (1600, 900),
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--help" | "-h" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--seed" => o.seed = val().parse().expect("seed"),
            "--beta" => o.params.beta = val().parse().expect("beta"),
            "--radius" => o.params.radius = val().parse().expect("radius"),
            "--land" => o.land = val().parse().expect("land"),
            "--zoom" => o.zoom = Some(val().parse().expect("zoom")),
            "--center" => o.center = parse_pair(&val(), ',').map(|(x, y)| [x, y]),
            "--bounded" => {
                let (w, h) = parse_pair(&val(), 'x').expect("WxH");
                o.extent = Extent::Bounded { w, h };
            }
            "--grayscale" => o.palette = Palette::Grayscale,
            "--palette" => o.palette = Palette::from_name(&val()).expect("palette name; see --help"),
            "--no-hillshade" => o.hillshade = false,
            "--no-outlines" => o.outlines = false,
            "--boundaries" => o.boundaries = true,
            "--screenshot" => o.screenshot = Some(val()),
            "--size" => o.size = parse_pair(&val(), 'x').expect("WxH"),
            other => panic!("unknown option {other}; see --help"),
        }
    }
    o
}

fn configure_scene(scene: &mut Scene, o: &Options) {
    scene.set_extent(o.extent);
    let mut style = scene.style;
    style.sea_level = sea_level(o.land);
    style.palette = o.palette;
    style.hillshade = o.hillshade;
    style.hex_outlines = o.outlines;
    style.tile_boundaries = o.boundaries;
    scene.set_style(style);
    scene.fit_view();
    if let Some(z) = o.zoom {
        scene.view.zoom = z;
    }
    if let Some(c) = o.center {
        scene.view.center = c;
    }
    scene.view_changed();
}

async fn device_for(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
            ..Default::default()
        })
        .await
        .expect("no suitable GPU adapter");
    let limits = wgpu::Limits {
        max_texture_array_layers: adapter.limits().max_texture_array_layers.min(1024),
        ..wgpu::Limits::downlevel_webgl2_defaults()
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("fastmap"),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .expect("device");
    (adapter, device, queue)
}

fn clock() -> impl Fn() -> f64 {
    let t0 = Instant::now();
    move || t0.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let o = parse_args();
    if let Some(path) = o.screenshot.clone() {
        screenshot(&o, &path);
        return;
    }
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App { opts: o, state: None };
    event_loop.run_app(&mut app).expect("run");
}

/// Renders one frame offscreen, after generating every tile it needs, and writes a PNG.
fn screenshot(o: &Options, path: &str) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let (_adapter, device, queue) = pollster::block_on(device_for(&instance, None));
    let (w, h) = o.size;
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let renderer = Renderer::new(&device, &queue, format, 1024);
    let mut scene = Scene::new(renderer, View::new(w, h), o.seed, o.params, 64);
    configure_scene(&mut scene, o);
    scene.prefetch = false;
    let now = clock();
    let t0 = now();
    while {
        scene.step(&queue, f64::INFINITY, &now);
        scene.busy()
    } {}
    let gen_ms = now() - t0;

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let draw = scene.render(&device, &queue, &view);
    let row = (w * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([draw, enc.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = buffer.slice(..).get_mapped_range().expect("mapped");
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let start = (y * row) as usize;
        pixels.extend_from_slice(&data[start..start + (w * 4) as usize]);
    }
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().expect("png header").write_image_data(&pixels).expect("png data");
    let s = scene.stats();
    println!(
        "{path}: {w}x{h}, {} tiles in {gen_ms:.1} ms ({:.3} ms/tile)",
        s.tiles_generated,
        gen_ms / s.tiles_generated.max(1) as f64
    );
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    scene: Scene,
    land: f64,
    drag: Option<(f64, f64)>,
    cursor: (f64, f64),
    now: Box<dyn Fn() -> f64>,
    rng: u64,
    last_title: f64,
}

struct App {
    opts: Options,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("FastMap")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0)),
            )
            .expect("window"),
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let (adapter, device, queue) = pollster::block_on(device_for(&instance, Some(&surface)));
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = preferred_format(&caps.formats);
        let mut config =
            surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).expect("config");
        config.format = format;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        println!(
            "adapter: {:?} ({:?}), format {format:?}",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let renderer = Renderer::new(&device, &queue, format, 1024);
        let mut scene = Scene::new(
            renderer,
            View::new(config.width, config.height),
            self.opts.seed,
            self.opts.params,
            512,
        );
        configure_scene(&mut scene, &self.opts);
        self.state = Some(State {
            window,
            surface,
            device,
            queue,
            config,
            scene,
            land: self.opts.land,
            drag: None,
            cursor: (0.0, 0.0),
            now: Box::new(clock()),
            rng: self.opts.seed,
            last_title: 0.0,
        });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(st) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                st.config.width = size.width.max(1);
                st.config.height = size.height.max(1);
                st.surface.configure(&st.device, &st.config);
                st.scene.view.width = st.config.width;
                st.scene.view.height = st.config.height;
                st.scene.view_changed();
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                st.drag = (state == ElementState::Pressed).then_some(st.cursor);
            }
            WindowEvent::CursorMoved { position, .. } => {
                st.cursor = (position.x, position.y);
                if let Some((x, y)) = st.drag {
                    st.scene.view.pan_pixels(position.x - x, position.y - y);
                    st.scene.view_changed();
                    st.drag = Some(st.cursor);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 60.0,
                };
                st.scene.view.zoom_at(st.cursor.0, st.cursor.1, 1.2f64.powf(lines));
                st.scene.view_changed();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let mut p = st.scene.params();
                let mut style = st.scene.style;
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => el.exit(),
                    Key::Named(NamedKey::Space) => {
                        st.rng = fastmap_core::noise::mix64(st.rng);
                        st.scene.set_seed(st.rng % 1_000_000);
                    }
                    Key::Character(c) => match c {
                        "[" => p.beta -= 0.1,
                        "]" => p.beta += 0.1,
                        "-" => p.radius /= 1.25,
                        "=" => p.radius *= 1.25,
                        "," => st.land = (st.land - 0.05).max(0.0),
                        "." => st.land = (st.land + 0.05).min(1.0),
                        "h" => style.hillshade ^= true,
                        "o" => style.hex_outlines ^= true,
                        "t" => style.tile_boundaries ^= true,
                        "g" => style.palette = style.palette.next(),
                        "f" => st.scene.fit_view(),
                        "b" => {
                            let e = match st.scene.extent() {
                                Extent::Unbounded => Extent::Bounded { w: 4, h: 3 },
                                Extent::Bounded { .. } => Extent::Unbounded,
                            };
                            st.scene.set_extent(e);
                            st.scene.fit_view();
                        }
                        _ => {}
                    },
                    _ => {}
                }
                style.sea_level = sea_level(st.land);
                st.scene.set_params(p);
                st.scene.set_style(style);
            }
            WindowEvent::RedrawRequested => {
                let now = &st.now;
                st.scene.step(&st.queue, 8.0, now);
                match st.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                        let view = frame.texture.create_view(&Default::default());
                        let cmd = st.scene.render(&st.device, &st.queue, &view);
                        st.queue.submit([cmd]);
                        st.queue.present(frame);
                    }
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        st.surface.configure(&st.device, &st.config);
                    }
                    _ => {}
                }
                let t = now();
                if t - st.last_title > 250.0 {
                    st.last_title = t;
                    let s = st.scene.stats();
                    let p = st.scene.params();
                    st.window.set_title(&format!(
                        "FastMap  seed {}  β {:.1}  R {:.0}  land {:.0}%  | {:.3} ms/tile ({:.3} cached)  {} drawn  {} pending",
                        st.scene.seed(),
                        p.beta,
                        p.radius,
                        st.land * 100.0,
                        s.ms_per_tile_full,
                        s.ms_per_tile_cached,
                        st.scene.renderer.drawn_count(),
                        s.pending
                    ));
                }
            }
            _ => {}
        }
        if st.scene.needs_redraw() || st.scene.busy() {
            st.window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(st) = &self.state
            && (st.scene.needs_redraw() || st.scene.busy())
        {
            st.window.request_redraw();
        }
    }
}
