// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Hex-per-pixel terrain shader. Each instance is one tile: a rhombus in axial space that
// covers its hexes. For every screen pixel, the fragment shader rounds to the nearest hex,
// reads that hex's elevation, and shades it. Hex edges are therefore resolved exactly at any
// zoom level.

// Must match fastmap_core::TILE / APRON (checked at compile time on the Rust side).
const TILE: i32 = 128;
const APRON: i32 = 1;
const SQRT3: f32 = 1.7320508;

struct Uniforms {
    // World units → clip space.
    scale: vec2<f32>,
    sea_level: f32,
    relief: f32,
    // bits 0-7: palette row; bit 8: continuous (ignores sea level).
    mode: u32,
    // bit 0: hillshade, bit 1: hex outlines, bit 2: tile boundaries, bit 3: sRGB target.
    flags: u32,
    px_per_hex: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var elevation: texture_2d_array<f32>;
@group(0) @binding(2) var palette: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // Fractional axial coordinates relative to the tile origin.
    @location(0) axial: vec2<f32>,
    @location(1) @interpolate(flat) layer: u32,
};

fn to_cartesian(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x + 0.5 * a.y, 0.5 * SQRT3 * a.y);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    // Cartesian offset of the tile origin from the camera center.
    @location(0) offset: vec2<f32>,
    @location(1) layer: u32,
) -> VsOut {
    // Two triangles covering local axial [-1.5, TILE + 0.5]² (hexes poke out of the rhombus of centers).
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let lo = -1.5;
    let hi = f32(TILE) + 0.5;
    let a = vec2<f32>(lo) + corners[vi] * (hi - lo);
    let p = (to_cartesian(a) + offset) * u.scale;
    var out: VsOut;
    out.clip = vec4<f32>(p, 0.0, 1.0);
    out.axial = a;
    out.layer = layer;
    return out;
}

fn hex_round(a: vec2<f32>) -> vec2<i32> {
    let s = -a.x - a.y;
    var q = round(a.x);
    var r = round(a.y);
    let rs = round(s);
    let dq = abs(q - a.x);
    let dr = abs(r - a.y);
    let ds = abs(rs - s);
    if (dq > dr && dq > ds) {
        q = -r - rs;
    } else if (dr > ds) {
        r = -q - rs;
    }
    return vec2<i32>(i32(q), i32(r));
}

fn elev(h: vec2<i32>, layer: u32) -> f32 {
    return textureLoad(elevation, h + vec2<i32>(APRON), i32(layer), 0).r;
}

fn pal(i: f32) -> vec3<f32> {
    return textureLoad(palette, vec2<i32>(clamp(i32(i), 0, 255), i32(u.mode & 0xffu)), 0).rgb;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let h = hex_round(in.axial);
    if (h.x < 0 || h.y < 0 || h.x >= TILE || h.y >= TILE) {
        discard;
    }
    let e = elev(h, in.layer);
    let sea = u.sea_level;

    var color: vec3<f32>;
    let continuous = (u.mode & 0x100u) != 0u;
    if (continuous) {
        color = pal(clamp(e / 7.0 + 0.5, 0.0, 1.0) * 255.0 + 0.5);
    } else if (e < sea) {
        let t = clamp((sea - e) / 2.5, 0.0, 1.0);
        color = pal(127.0 - t * 127.0);
    } else {
        let t = clamp((e - sea) / max(3.5 - sea, 0.75), 0.0, 1.0);
        color = pal(128.0 + t * 127.0);
    }

    if ((u.flags & 1u) != 0u) {
        // Central differences along the two lattice axes. Neighbors at the tile edge come from the apron.
        let dq = 0.5 * (elev(h + vec2<i32>(1, 0), in.layer) - elev(h - vec2<i32>(1, 0), in.layer));
        let dr = 0.5 * (elev(h + vec2<i32>(0, 1), in.layer) - elev(h - vec2<i32>(0, 1), in.layer));
        // Convert to a cartesian gradient: g·a0 = dq, g·a1 = dr.
        let g = vec2<f32>(dq, (dr - 0.5 * dq) / (0.5 * SQRT3));
        let land = select(0.15, 1.0, e >= sea || continuous);
        let n = normalize(vec3<f32>(-g * u.relief * land, 1.0));
        let l = normalize(vec3<f32>(-1.0, 1.0, 1.4));
        let shade = clamp(dot(n, l), 0.0, 1.0);
        let flat_shade = l.z;
        color = color * clamp(0.35 + 0.65 * shade / flat_shade, 0.0, 1.6);
    }

    let f = in.axial - vec2<f32>(h);
    if ((u.flags & 2u) != 0u && u.px_per_hex > 10.0) {
        // Distance to the hex boundary: the inradius is ½ along the three neighbor directions.
        let v = to_cartesian(f);
        let d = max(abs(v.x), max(abs(dot(v, vec2<f32>(0.5, 0.5 * SQRT3))), abs(dot(v, vec2<f32>(-0.5, 0.5 * SQRT3)))));
        let px = (0.5 - d) * u.px_per_hex;
        let fade = clamp((u.px_per_hex - 10.0) / 20.0, 0.0, 1.0) * 0.35;
        color = mix(color, vec3<f32>(0.0), fade * (1.0 - smoothstep(0.0, 1.2, px)));
    }

    if ((u.flags & 4u) != 0u && (h.x == 0 || h.y == 0)) {
        color = mix(color, vec3<f32>(1.0, 0.25, 0.2), 0.55);
    }

    if ((u.flags & 8u) != 0u) {
        color = srgb_to_linear(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)));
    }
    return vec4<f32>(color, 1.0);
}
