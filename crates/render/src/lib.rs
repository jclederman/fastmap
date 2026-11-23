// SPDX-License-Identifier: AGPL-3.0-or-later
//! # fastmap-render
//!
//! A platform-agnostic wgpu renderer for FastMap tiles.
//!
//! - Each resident tile is one layer of an `R32Float` texture array, uploaded once.
//! - Each tile is drawn as one instanced rhombus. The fragment shader rounds each pixel to its
//!   hex and shades it, so hex edges stay sharp at every zoom level, and panning or zooming
//!   only rewrites a small uniform and instance buffer.

mod palette;
pub mod scene;
pub mod view;

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use fastmap_core::{APRON, STORED, TILE, Tile, TileCoord};
pub use palette::Palette;
pub use scene::{Extent, Scene, Stats};
pub use view::View;

const SHADER: &str = include_str!("map.wgsl");

/// Compile-time check that the shader's constants match the core crate.
const fn contains(hay: &str, needle: &str) -> bool {
    let (h, n) = (hay.as_bytes(), needle.as_bytes());
    let mut i = 0;
    while i + n.len() <= h.len() {
        let mut j = 0;
        while j < n.len() && h[i + j] == n[j] {
            j += 1;
        }
        if j == n.len() {
            return true;
        }
        i += 1;
    }
    false
}
const _: () = assert!(TILE == 128 && contains(SHADER, "const TILE: i32 = 128;"));
const _: () = assert!(APRON == 1 && contains(SHADER, "const APRON: i32 = 1;"));

/// How the map is colored and shaded. Changing any of these is instant; nothing is
/// regenerated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    /// Sea level in units of σ (see [`fastmap_core::sea_level`]).
    pub sea_level: f32,
    /// Hillshade vertical exaggeration.
    pub relief: f32,
    pub palette: Palette,
    pub hillshade: bool,
    pub hex_outlines: bool,
    pub tile_boundaries: bool,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            sea_level: fastmap_core::sea_level(0.4),
            relief: 6.0,
            palette: Palette::Terrain,
            hillshade: true,
            hex_outlines: true,
            tile_boundaries: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    scale: [f32; 2],
    sea_level: f32,
    relief: f32,
    mode: u32,
    flags: u32,
    px_per_hex: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    offset: [f32; 2],
    layer: u32,
    _pad: u32,
}

/// Background color behind tiles that are not generated yet (sRGB).
pub const BACKGROUND: [f64; 3] = [0.075, 0.085, 0.10];

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    instances: wgpu::Buffer,
    instance_capacity: usize,
    elevation: wgpu::Texture,
    layers: u32,
    /// Tiles that are drawn, by layer.
    resident: HashMap<TileCoord, u32>,
    /// Replacement tiles that are uploaded but not drawn until [`Renderer::commit_staged`].
    staged: HashMap<TileCoord, u32>,
    free: Vec<u32>,
    srgb_target: bool,
    drawn: usize,
}

impl Renderer {
    /// Creates the renderer for color targets of `format`. `max_tiles` is capped by the
    /// device's texture-array layer limit.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        max_tiles: u32,
    ) -> Self {
        const ROWS: u32 = Palette::ALL.len() as u32;
        let layers = max_tiles.min(device.limits().max_texture_array_layers).max(1);
        let elevation = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("elevation"),
            size: wgpu::Extent3d {
                width: STORED as u32,
                height: STORED as u32,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let palette = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("palette"),
            size: wgpu::Extent3d { width: 256, height: ROWS, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            palette.as_image_copy(),
            bytemuck::cast_slice(&palette::PALETTES),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256 * 4), rows_per_image: None },
            wgpu::Extent3d { width: 256, height: ROWS, depth_or_array_layers: 1 },
        );

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_capacity = layers as usize;
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: (instance_capacity * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture_entry = |binding, dim| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: dim,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("map"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(1, wgpu::TextureViewDimension::D2Array),
                texture_entry(2, wgpu::TextureViewDimension::D2),
            ],
        });
        let elevation_view = elevation.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let palette_view = palette.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("map"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&elevation_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&palette_view),
                },
            ],
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("map.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("map"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("map"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Uint32],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            uniforms,
            instances,
            instance_capacity,
            elevation,
            layers,
            resident: HashMap::new(),
            staged: HashMap::new(),
            free: (0..layers).rev().collect(),
            srgb_target: format.is_srgb(),
            drawn: 0,
        }
    }

    /// Maximum number of tiles that can be resident at once.
    pub fn capacity(&self) -> usize {
        self.layers as usize
    }
    /// Number of layers in use, drawn or staged.
    pub fn resident_count(&self) -> usize {
        self.layers as usize - self.free.len()
    }
    /// Number of tiles drawn in the last frame.
    pub fn drawn_count(&self) -> usize {
        self.drawn
    }
    /// True if every layer holds a tile, so the next new tile needs an eviction.
    pub fn is_full(&self) -> bool {
        self.free.is_empty()
    }
    /// Tiles holding a layer, drawn or staged (a tile may appear twice).
    pub fn resident_tiles(&self) -> impl Iterator<Item = TileCoord> + '_ {
        self.resident.keys().chain(self.staged.keys()).copied()
    }
    pub fn is_staged(&self, coord: TileCoord) -> bool {
        self.staged.contains_key(&coord)
    }
    pub fn has_staged(&self) -> bool {
        !self.staged.is_empty()
    }
    pub fn is_resident(&self, coord: TileCoord) -> bool {
        self.resident.contains_key(&coord)
    }
    /// Drops every resident tile, for example after the seed or kernel changes.
    pub fn clear(&mut self) {
        self.resident.clear();
        self.staged.clear();
        self.free = (0..self.layers).rev().collect();
    }
    /// Frees a tile's layers, drawn and staged.
    pub fn evict(&mut self, coord: TileCoord) {
        self.free.extend(self.resident.remove(&coord));
        self.free.extend(self.staged.remove(&coord));
    }

    /// Uploads a replacement tile into a free layer without drawing it. The caller must make
    /// sure a layer is free (see [`Renderer::is_full`]) unless the tile is already staged.
    pub fn upload_staged(&mut self, queue: &wgpu::Queue, tile: &Tile) {
        let layer = match self.staged.get(&tile.coord) {
            Some(&l) => l,
            None => {
                let l = self.free.pop().expect("a free layer for a staged tile");
                self.staged.insert(tile.coord, l);
                l
            }
        };
        self.write_layer(queue, layer, tile);
    }

    /// Makes every staged tile visible at once, freeing the layers they replace. Returns the
    /// tiles that changed.
    pub fn commit_staged(&mut self) -> Vec<TileCoord> {
        let mut changed = Vec::with_capacity(self.staged.len());
        for (c, layer) in self.staged.drain() {
            self.free.extend(self.resident.insert(c, layer));
            changed.push(c);
        }
        changed
    }

    /// Uploads a finished tile. If the tile array is full, the resident tile farthest from
    /// `view` is evicted first.
    pub fn upload(&mut self, queue: &wgpu::Queue, tile: &Tile, view: &View) {
        let layer = match self.resident.get(&tile.coord) {
            Some(&l) => l,
            None => {
                if self.free.is_empty() {
                    let far = *self
                        .resident
                        .keys()
                        .max_by(|a, b| view.tile_distance2(**a).total_cmp(&view.tile_distance2(**b)))
                        .expect("non-empty");
                    self.evict(far);
                }
                let l = self.free.pop().expect("free layer");
                self.resident.insert(tile.coord, l);
                l
            }
        };
        self.write_layer(queue, layer, tile);
    }

    fn write_layer(&self, queue: &wgpu::Queue, layer: u32, tile: &Tile) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.elevation,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&tile.elevation),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(STORED as u32 * 4),
                rows_per_image: Some(STORED as u32),
            },
            wgpu::Extent3d { width: STORED as u32, height: STORED as u32, depth_or_array_layers: 1 },
        );
    }

    /// Records the draw of all resident tiles that intersect `view` into `target`.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        view: &View,
        style: &Style,
    ) -> wgpu::CommandBuffer {
        let _ = device;
        let visible = view.visible_tiles();
        let mut inst: Vec<Instance> = Vec::with_capacity(visible.len().min(self.instance_capacity));
        for c in visible {
            if let Some(&layer) = self.resident.get(&c) {
                if inst.len() == self.instance_capacity {
                    break;
                }
                inst.push(Instance { offset: view.tile_offset(c), layer, _pad: 0 });
            }
        }
        self.drawn = inst.len();
        if !inst.is_empty() {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&inst));
        }
        let flags = (style.hillshade as u32)
            | (style.hex_outlines as u32) << 1
            | (style.tile_boundaries as u32) << 2
            | (self.srgb_target as u32) << 3;
        let u = Uniforms {
            scale: view.clip_scale(),
            sea_level: style.sea_level,
            relief: style.relief,
            mode: style.palette as u32 | (style.palette.continuous() as u32) << 8,
            flags,
            px_per_hex: view.zoom as f32,
            _pad: 0.0,
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        let bg = if self.srgb_target { BACKGROUND.map(srgb_to_linear) } else { BACKGROUND };
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("map") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("map"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: bg[0], g: bg[1], b: bg[2], a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            if !inst.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.instances.slice(..));
                pass.draw(0..6, 0..inst.len() as u32);
            }
        }
        encoder.finish()
    }
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Picks the preferred surface format: non-sRGB 8-bit, since the shader outputs sRGB values
/// directly. Falls back to the first available format.
pub fn preferred_format(formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    formats
        .iter()
        .copied()
        .find(|f| matches!(f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
        .unwrap_or(formats[0])
}
