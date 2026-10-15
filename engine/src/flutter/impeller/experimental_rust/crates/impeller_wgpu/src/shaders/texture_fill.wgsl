// Ported from:
// - impeller/entity/shaders/texture_fill.vert
// - impeller/entity/shaders/texture_fill.frag
// - impeller/compiler/shader_lib/impeller/texture.glsl
// - impeller/compiler/shader_lib/impeller/tile_mode.glsl

const TILE_MODE_CLAMP: u32 = 0u;
const TILE_MODE_REPEAT: u32 = 1u;
const TILE_MODE_MIRROR: u32 = 2u;
const TILE_MODE_DECAL: u32 = 3u;

struct FrameInfo {
    mvp: mat4x4<f32>,
};

struct TextureFragInfo {
    alpha: f32,
    tile_mode_x: u32,
    tile_mode_y: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<uniform> frame_info: FrameInfo;

@group(0) @binding(1)
var<uniform> frag_info: TextureFragInfo;

@group(0) @binding(2)
var tex: texture_2d<f32>;

@group(0) @binding(3)
var samp: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
};

fn ip_float_tile(t_in: f32, tile_mode: u32) -> f32 {
    var t = t_in;
    if tile_mode == TILE_MODE_CLAMP || tile_mode == TILE_MODE_DECAL {
        t = clamp(t, 0.0, 1.0);
    } else if tile_mode == TILE_MODE_REPEAT {
        t = t - floor(t);
    } else if tile_mode == TILE_MODE_MIRROR {
        let t1 = t - 1.0;
        let t2 = t1 - 2.0 * floor(t1 * 0.5) - 1.0;
        t = abs(t2);
    }
    return t;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = frame_info.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.v_uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(
        ip_float_tile(in.v_uv.x, frag_info.tile_mode_x),
        ip_float_tile(in.v_uv.y, frag_info.tile_mode_y),
    );
    let c = textureSample(tex, samp, uv);

    let out_of_bounds_x =
        frag_info.tile_mode_x == TILE_MODE_DECAL && (in.v_uv.x < 0.0 || in.v_uv.x > 1.0);
    let out_of_bounds_y =
        frag_info.tile_mode_y == TILE_MODE_DECAL && (in.v_uv.y < 0.0 || in.v_uv.y > 1.0);
    if out_of_bounds_x || out_of_bounds_y {
        return vec4<f32>(0.0);
    }
    return c * frag_info.alpha;
}
