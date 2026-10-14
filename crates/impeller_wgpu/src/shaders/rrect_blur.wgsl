// Ported from:
// - impeller/entity/shaders/rrect_like_blur.vert
// - impeller/entity/shaders/rrect_blur.frag
// - impeller/compiler/shader_lib/impeller/rrect.glsl
// - impeller/compiler/shader_lib/impeller/gaussian.glsl
// - impeller/compiler/shader_lib/impeller/constants.glsl

const K_SQRT_TWO_PI: f32 = 2.50662827463;
const K_HALF_SQRT_TWO: f32 = 0.70710678118;
const K_SQRT_THREE: f32 = 1.73205080757;
const K_TWO_OVER_SQRT_PI: f32 = 1.1283791670955126;

struct FrameInfo {
    mvp: mat4x4<f32>,
};

struct FragInfo {
    color: vec4<f32>,
    center_adjust: vec4<f32>,
    r1_exponent_exponent_inv: vec4<f32>,
    s_inv_min_edge_scale: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> frame_info: FrameInfo;

@group(0) @binding(1)
var<uniform> frag_info: FragInfo;

struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_position: vec2<f32>,
};

// Gaussian distribution function from impeller/gaussian.glsl
fn ip_gaussian(x: f32, sigma: f32) -> f32 {
    let variance = sigma * sigma;
    return exp(-0.5 * x * x / variance) / (K_SQRT_TWO_PI * sigma);
}

// Abramowitz and Stegun erf approximation from impeller/gaussian.glsl
fn ip_erf(x: f32) -> f32 {
    let a = abs(x);
    let b = (0.278393 + (0.230389 + 0.078108 * a * a) * a) * a + 1.0;
    let b2 = b * b;
    let b4 = b2 * b2;
    return sign(x) * (1.0 - 1.0 / b4);
}

// Vec2 variation of Abramowitz and Stegun erf approximation from impeller/gaussian.glsl
fn ip_vec2_erf(x: vec2<f32>) -> vec2<f32> {
    let a = abs(x);
    let b = (0.278393 + (0.230389 + 0.078108 * a * a) * a) * a + 1.0;
    let b2 = b * b;
    let b4 = b2 * b2;
    return sign(x) * (1.0 - 1.0 / b4);
}

// Indefinite Gaussian integral from impeller/gaussian.glsl
fn ip_gaussian_integral(x: f32, sigma: f32) -> f32 {
    return (1.0 + ip_erf(x * (K_HALF_SQRT_TWO / sigma))) * 0.5;
}

// Vec2 indefinite Gaussian integral from impeller/gaussian.glsl
fn ip_vec2_gaussian_integral(x: vec2<f32>, sigma: f32) -> vec2<f32> {
    return (1.0 + ip_vec2_erf(x * (K_HALF_SQRT_TWO / sigma))) * 0.5;
}

fn max_xy(v: vec2<f32>) -> f32 {
    return max(v.x, v.y);
}

// Levien analytical erf7 approximation from impeller/rrect.glsl
fn compute_erf7(x_in: f32) -> f32 {
    var x = x_in * K_TWO_OVER_SQRT_PI;
    let xx = x * x;
    x = x + (0.24295 + (0.03395 + 0.0104 * xx) * xx) * (x * xx);
    return x / sqrt(1.0 + x * x);
}

fn safe_pow(base: f32, exponent: f32) -> f32 {
    if base <= 0.0 {
        return 0.0;
    }
    return pow(base, exponent);
}

fn power_distance(p: vec2<f32>, exponent: f32, exponent_inv: f32) -> f32 {
    let xp = safe_pow(p.x, exponent);
    let yp = safe_pow(p.y, exponent);
    return safe_pow(xp + yp, exponent_inv);
}

fn compute_rrect_distance(
    position: vec2<f32>,
    adjust: vec2<f32>,
    r1_exponent_exponent_inv: vec3<f32>,
) -> f32 {
    let r1 = r1_exponent_exponent_inv.x;
    let exponent = r1_exponent_exponent_inv.y;
    let exponent_inv = r1_exponent_exponent_inv.z;

    let adjusted = position - adjust;
    let d_pos = power_distance(max(adjusted, vec2<f32>(0.0, 0.0)), exponent, exponent_inv);
    let d_neg = min(max_xy(adjusted), 0.0);
    return d_pos + d_neg - r1;
}

fn compute_rrect_fade(d: f32, s_inv_min_edge_scale: vec3<f32>) -> f32 {
    let s_inv = s_inv_min_edge_scale.x;
    let min_edge = s_inv_min_edge_scale.y;
    let scale = s_inv_min_edge_scale.z;
    return scale * (compute_erf7(s_inv * (min_edge + d)) - compute_erf7(s_inv * d));
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = frame_info.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.v_position = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center = frag_info.center_adjust.xy;
    let adjust = frag_info.center_adjust.zw;

    let centered = abs(in.v_position - center);
    let d = compute_rrect_distance(
        centered,
        adjust,
        frag_info.r1_exponent_exponent_inv.xyz,
    );
    let z = clamp(compute_rrect_fade(d, frag_info.s_inv_min_edge_scale.xyz), 0.0, 1.0);

    return frag_info.color * z;
}
