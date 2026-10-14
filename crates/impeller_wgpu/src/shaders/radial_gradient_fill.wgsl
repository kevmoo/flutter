// Ported from:
// - impeller/entity/shaders/gradients/gradient_fill.vert
// - impeller/entity/shaders/gradients/radial_gradient_ssbo_fill.frag
// - impeller/entity/shaders/gradients/radial_gradient_uniform_fill.frag
// - impeller/compiler/shader_lib/impeller/gradient.glsl
// - impeller/compiler/shader_lib/impeller/texture.glsl
// - impeller/compiler/shader_lib/impeller/tile_mode.glsl
// - impeller/compiler/shader_lib/impeller/color.glsl

const TILE_MODE_CLAMP: u32 = 0u;
const TILE_MODE_REPEAT: u32 = 1u;
const TILE_MODE_MIRROR: u32 = 2u;
const TILE_MODE_DECAL: u32 = 3u;

struct FrameInfo {
    mvp: mat4x4<f32>,
    matrix: mat4x4<f32>,
};

struct FragInfo {
    center: vec2<f32>,
    radius: f32,
    alpha: f32,
    decal_border_color: vec4<f32>,
    fast_color_0: vec4<f32>,
    fast_color_1: vec4<f32>,
    tile_mode: u32,
    colors_length: u32,
    is_fast_two_stop: u32,
    _pad: u32,
};

struct ColorPoint {
    color: vec4<f32>,
    stop: f32,
    inverse_delta: f32,
    _pad0: f32,
    _pad1: f32,
};

struct ColorStopsUniform {
    points: array<ColorPoint, 64>,
};

@group(0) @binding(0)
var<uniform> frame_info: FrameInfo;

@group(0) @binding(1)
var<uniform> frag_info: FragInfo;

@group(0) @binding(2)
var<uniform> color_data: ColorStopsUniform;

struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_position: vec2<f32>,
};

fn ip_vec2_transform_position(matrix: mat4x4<f32>, point: vec2<f32>) -> vec2<f32> {
    let transformed = matrix * vec4<f32>(point, 0.0, 1.0);
    if abs(transformed.w) > 1.0e-6 {
        return transformed.xy / transformed.w;
    }
    return transformed.xy;
}

fn ip_premultiply(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * color.a, color.a);
}

fn ip_float_tile(t_in: f32, tile_mode: u32) -> f32 {
    var t = t_in;
    if tile_mode == TILE_MODE_CLAMP {
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
    out.v_position = ip_vec2_transform_position(frame_info.matrix, in.position);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let len = length(in.v_position - frag_info.center);
    var t = select(0.0, len / frag_info.radius, frag_info.radius > 0.0);

    var result_color = vec4<f32>(0.0);
    if (t < 0.0 || t > 1.0) && frag_info.tile_mode == TILE_MODE_DECAL {
        result_color = frag_info.decal_border_color;
    } else {
        t = ip_float_tile(t, frag_info.tile_mode);

        if frag_info.is_fast_two_stop != 0u {
            result_color = mix(frag_info.fast_color_0, frag_info.fast_color_1, clamp(t, 0.0, 1.0));
        } else if frag_info.colors_length == 1u {
            result_color = color_data.points[0u].color;
        } else if frag_info.colors_length > 1u {
            let clamped_len = min(frag_info.colors_length, 64u);
            let last_idx = clamped_len - 1u;
            if t <= color_data.points[0u].stop {
                result_color = color_data.points[0u].color;
            } else if t >= color_data.points[last_idx].stop {
                result_color = color_data.points[last_idx].color;
            } else {
                for (var i: u32 = 1u; i < clamped_len; i = i + 1u) {
                    let prev_point = color_data.points[i - 1u];
                    let current_point = color_data.points[i];
                    if t >= prev_point.stop && t <= current_point.stop {
                        if current_point.inverse_delta > 1000.0 {
                            result_color = current_point.color;
                        } else {
                            let ratio = (t - prev_point.stop) * current_point.inverse_delta;
                            result_color = mix(prev_point.color, current_point.color, ratio);
                        }
                        break;
                    }
                }
            }
        }
    }

    return ip_premultiply(result_color) * frag_info.alpha;
}
