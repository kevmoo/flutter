// Ported from:
// - impeller/entity/shaders/gradients/gradient_fill.vert
// - impeller/entity/shaders/gradients/conical_gradient_ssbo_fill.frag
// - impeller/compiler/shader_lib/impeller/gradient.glsl
// - impeller/compiler/shader_lib/impeller/transform.glsl
// - impeller/compiler/shader_lib/impeller/tile_mode.glsl
// - impeller/compiler/shader_lib/impeller/color.glsl

const TILE_MODE_CLAMP: u32 = 0u;
const TILE_MODE_REPEAT: u32 = 1u;
const TILE_MODE_MIRROR: u32 = 2u;
const TILE_MODE_DECAL: u32 = 3u;

const CONICAL_KIND_DEGENERATE: u32 = 0u;
const CONICAL_KIND_RADIAL: u32 = 1u;
const CONICAL_KIND_STRIP: u32 = 2u;
const CONICAL_KIND_CONICAL: u32 = 3u;

struct FrameInfo {
    mvp: mat4x4<f32>,
    matrix: mat4x4<f32>,
};

struct FragInfo {
    start_center: vec2<f32>,
    start_radius: f32,
    end_radius: f32,
    end_center: vec2<f32>,
    alpha: f32,
    kind: u32,
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

fn ip_mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let a00 = m[0][0];
    let a01 = m[0][1];
    let a02 = m[0][2];
    let a10 = m[1][0];
    let a11 = m[1][1];
    let a12 = m[1][2];
    let a20 = m[2][0];
    let a21 = m[2][1];
    let a22 = m[2][2];

    let b01 = a22 * a11 - a12 * a21;
    let b11 = -a22 * a10 + a12 * a20;
    let b21 = a21 * a10 - a11 * a20;

    let det = a00 * b01 + a01 * b11 + a02 * b21;
    let inv_det = select(0.0, 1.0 / det, abs(det) > 1.0e-12);

    return mat3x3<f32>(
        vec3<f32>(b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11)) * inv_det,
        vec3<f32>(b11, (a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10)) * inv_det,
        vec3<f32>(b21, (-a21 * a00 + a01 * a20), (a11 * a00 - a01 * a10)) * inv_det,
    );
}

fn ip_map_to_unit_x(p0: vec2<f32>, p1: vec2<f32>) -> mat3x3<f32> {
    let rot = mat3x3<f32>(
        vec3<f32>(0.0, -1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    let basis = mat3x3<f32>(
        vec3<f32>(p1.y - p0.y, p0.x - p1.x, 0.0),
        vec3<f32>(p1.x - p0.x, p1.y - p0.y, 0.0),
        vec3<f32>(p0.x, p0.y, 1.0),
    );
    return rot * ip_mat3_inverse(basis);
}

fn ip_compute_conical_t_radial(c0: vec2<f32>, r0: f32, r1: f32, pos: vec2<f32>) -> vec2<f32> {
    let d_radius = r1 - r0;
    if abs(d_radius) <= 1.0e-6 {
        return vec2<f32>(0.0, -1.0);
    }
    let scale = 1.0 / d_radius;
    let scale_sign = sign(d_radius);
    let bias = r0 / d_radius;

    let pt = (pos - c0) * scale;
    let t = length(pt) * scale_sign - bias;
    return vec2<f32>(t, 1.0);
}

fn ip_compute_conical_t_strip(c0: vec2<f32>, r0: f32, c1: vec2<f32>, pos: vec2<f32>) -> vec2<f32> {
    let d_center = distance(c0, c1);
    if d_center <= 1.0e-6 {
        return vec2<f32>(0.0, -1.0);
    }
    let transform = ip_map_to_unit_x(c0, c1);
    let r = r0 / d_center;
    let r_2 = r * r;

    let pt = (transform * vec3<f32>(pos, 1.0)).xy;
    let disc = r_2 - pt.y * pt.y;
    if disc < 0.0 {
        return vec2<f32>(0.0, -1.0);
    }
    let t = pt.x + sqrt(disc);
    return vec2<f32>(t, 1.0);
}

fn ip_compute_conical_t_conical(
    c0_in: vec2<f32>,
    r0: f32,
    c1_in: vec2<f32>,
    r1: f32,
    pos: vec2<f32>,
) -> vec2<f32> {
    let scalar_nearly_zero: f32 = 1.0 / 4096.0;
    var c0 = c0_in;
    var c1 = c1_in;
    let d_center = distance(c0, c1);
    if d_center <= 1.0e-6 || abs(r0 - r1) <= 1.0e-6 {
        return vec2<f32>(0.0, -1.0);
    }

    var f = r0 / (r0 - r1);
    let is_swapped = abs(f - 1.0) < scalar_nearly_zero;
    if is_swapped {
        let tmp_pt = c0;
        c0 = c1;
        c1 = tmp_pt;
        f = 0.0;
    }

    let cf = c0 * (1.0 - f) + c1 * f;
    var transform = ip_map_to_unit_x(cf, c1);

    var scale_x = abs(1.0 - f);
    var scale_y = scale_x;
    let local_r1 = abs(r1 - r0) / d_center;
    let is_focal_on_circle = abs(local_r1 - 1.0) < scalar_nearly_zero;
    if is_focal_on_circle {
        scale_x = scale_x * 0.5;
        scale_y = scale_y * 0.5;
    } else {
        scale_x = scale_x * (local_r1 / (local_r1 * local_r1 - 1.0));
        scale_y = scale_y / sqrt(abs(local_r1 * local_r1 - 1.0));
    }
    let scale_mat = mat3x3<f32>(
        vec3<f32>(scale_x, 0.0, 0.0),
        vec3<f32>(0.0, scale_y, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    transform = scale_mat * transform;

    let pt = (transform * vec3<f32>(pos, 1.0)).xy;

    let inv_r1 = 1.0 / local_r1;
    let d_radius_sign = sign(1.0 - f);
    let is_well_behaved = !is_focal_on_circle && local_r1 > 1.0;

    var x_t: f32 = -1.0;
    if is_focal_on_circle {
        if abs(pt.x) > 1.0e-6 {
            x_t = dot(pt, pt) / pt.x;
        }
    } else if is_well_behaved {
        x_t = length(pt) - pt.x * inv_r1;
    } else {
        let temp = pt.x * pt.x - pt.y * pt.y;
        if temp >= 0.0 {
            if is_swapped || d_radius_sign < 0.0 {
                x_t = -sqrt(temp) - pt.x * inv_r1;
            } else {
                x_t = sqrt(temp) - pt.x * inv_r1;
            }
        }
    }

    if !is_well_behaved && x_t < 0.0 {
        return vec2<f32>(0.0, -1.0);
    }

    var t = f + d_radius_sign * x_t;
    if is_swapped {
        t = 1.0 - t;
    }
    return vec2<f32>(t, 1.0);
}

fn ip_compute_conical_t(
    kind: u32,
    c0: vec2<f32>,
    r0: f32,
    c1: vec2<f32>,
    r1: f32,
    pos: vec2<f32>,
) -> vec2<f32> {
    if kind == CONICAL_KIND_DEGENERATE {
        return vec2<f32>(0.0, -1.0);
    } else if kind == CONICAL_KIND_RADIAL {
        return ip_compute_conical_t_radial(c0, r0, r1, pos);
    } else if kind == CONICAL_KIND_STRIP {
        return ip_compute_conical_t_strip(c0, r0, c1, pos);
    } else {
        return ip_compute_conical_t_conical(c0, r0, c1, r1, pos);
    }
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
    let res = ip_compute_conical_t(
        frag_info.kind,
        frag_info.start_center,
        frag_info.start_radius,
        frag_info.end_center,
        frag_info.end_radius,
        in.v_position,
    );

    var t = res.x;
    var result_color = vec4<f32>(0.0);
    if res.y < 0.0 || ((t < 0.0 || t > 1.0) && frag_info.tile_mode == TILE_MODE_DECAL) {
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
