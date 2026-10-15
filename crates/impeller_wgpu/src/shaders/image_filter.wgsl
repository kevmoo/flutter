// Ported from:
// - impeller/entity/shaders/gaussian_blur/gaussian_blur.frag
// - impeller/entity/shaders/morphology_filter.frag
// - impeller/entity/shaders/filters/color_matrix_color_filter.frag
// - impeller/entity/shaders/filters/srgb_to_linear_filter.frag
// - impeller/entity/shaders/filters/linear_to_srgb_filter.frag
// - impeller/compiler/shader_lib/impeller/blending.glsl

const TILE_MODE_CLAMP: u32 = 0u;
const TILE_MODE_REPEAT: u32 = 1u;
const TILE_MODE_MIRROR: u32 = 2u;
const TILE_MODE_DECAL: u32 = 3u;

const FILTER_KIND_BLUR_1D: u32 = 0u;
const FILTER_KIND_DILATE_1D: u32 = 1u;
const FILTER_KIND_ERODE_1D: u32 = 2u;
const FILTER_KIND_COLOR_MATRIX: u32 = 3u;
const FILTER_KIND_COLOR_MODE: u32 = 4u;
const FILTER_KIND_SRGB_TO_LINEAR: u32 = 5u;
const FILTER_KIND_LINEAR_TO_SRGB: u32 = 6u;

struct FrameInfo {
    mvp: mat4x4<f32>,
};

struct FragInfo {
    color_matrix_row0: vec4<f32>,
    color_matrix_row1: vec4<f32>,
    color_matrix_row2: vec4<f32>,
    color_matrix_row3: vec4<f32>,
    color_matrix_offset: vec4<f32>,
    mode_color_pm: vec4<f32>,
    step_uv: vec2<f32>,
    tex_size: vec2<f32>,
    sigma: f32,
    alpha: f32,
    filter_kind: u32,
    radius: i32,
    tile_mode: u32,
    blend_mode: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<uniform> frame_info: FrameInfo;

@group(0) @binding(1)
var<uniform> frag_info: FragInfo;

@group(0) @binding(2)
var filter_texture: texture_2d<f32>;

@group(0) @binding(3)
var filter_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = frame_info.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fn ip_premultiply(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(c.rgb * c.a, c.a);
}

fn ip_unpremultiply(pm: vec4<f32>) -> vec4<f32> {
    if pm.a > 1.0e-6 {
        return vec4<f32>(pm.rgb / pm.a, pm.a);
    }
    return vec4<f32>(0.0);
}

fn sample_tiled(uv: vec2<f32>, tile_mode: u32, tex_size: vec2<f32>) -> vec4<f32> {
    if tile_mode == TILE_MODE_DECAL {
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            return vec4<f32>(0.0);
        }
        return textureSampleLevel(filter_texture, filter_sampler, uv, 0.0);
    }
    if tile_mode == TILE_MODE_REPEAT {
        let rep_uv = uv - floor(uv);
        return textureSampleLevel(filter_texture, filter_sampler, rep_uv, 0.0);
    }
    if tile_mode == TILE_MODE_MIRROR {
        let t1 = uv - vec2<f32>(1.0);
        let t2 = t1 - 2.0 * floor(t1 * 0.5) - vec2<f32>(1.0);
        let mir_uv = abs(t2);
        return textureSampleLevel(filter_texture, filter_sampler, mir_uv, 0.0);
    }
    // TILE_MODE_CLAMP: clamp to half-texel centers to prevent border bleed
    let safe_size = max(tex_size, vec2<f32>(1.0));
    let half_texel = vec2<f32>(0.5) / safe_size;
    let clamped_uv = clamp(uv, half_texel, vec2<f32>(1.0) - half_texel);
    return textureSampleLevel(filter_texture, filter_sampler, clamped_uv, 0.0);
}

fn srgb_channel_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn linear_channel_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn ip_do_color_blend(dst_pm: vec4<f32>, src_pm: vec4<f32>, blended_rgb: vec3<f32>) -> vec4<f32> {
    let dst_u = ip_unpremultiply(dst_pm);
    let src_u = ip_unpremultiply(src_pm);
    let out_a = src_u.a + dst_u.a * (1.0 - src_u.a);
    let out_rgb = (1.0 - dst_u.a) * src_pm.rgb + (1.0 - src_u.a) * dst_pm.rgb + dst_u.a * src_u.a * blended_rgb;
    return vec4<f32>(out_rgb, out_a);
}

fn ip_blend_premultiplied(dst: vec4<f32>, src: vec4<f32>, mode: u32) -> vec4<f32> {
    switch mode {
        case 0u: { // Clear
            return vec4<f32>(0.0);
        }
        case 1u: { // Src
            return src;
        }
        case 2u: { // Dst
            return dst;
        }
        case 3u: { // SrcOver
            return src + dst * (1.0 - src.a);
        }
        case 4u: { // DstOver
            return dst + src * (1.0 - dst.a);
        }
        case 5u: { // SrcIn
            return src * dst.a;
        }
        case 6u: { // DstIn
            return dst * src.a;
        }
        case 7u: { // SrcOut
            return src * (1.0 - dst.a);
        }
        case 8u: { // DstOut
            return dst * (1.0 - src.a);
        }
        case 9u: { // SrcATop
            return src * dst.a + dst * (1.0 - src.a);
        }
        case 10u: { // DstATop
            return dst * src.a + src * (1.0 - dst.a);
        }
        case 11u: { // Xor
            return src * (1.0 - dst.a) + dst * (1.0 - src.a);
        }
        case 12u: { // Plus
            return min(src + dst, vec4<f32>(1.0));
        }
        case 13u: { // Modulate
            return src * dst;
        }
        case 14u: { // Screen
            return src + dst - src * dst;
        }
        case 16u: { // Darken
            let d = ip_unpremultiply(dst).rgb;
            let s = ip_unpremultiply(src).rgb;
            return ip_do_color_blend(dst, src, min(d, s));
        }
        case 17u: { // Lighten
            let d = ip_unpremultiply(dst).rgb;
            let s = ip_unpremultiply(src).rgb;
            return ip_do_color_blend(dst, src, max(d, s));
        }
        case 22u: { // Difference
            let d = ip_unpremultiply(dst).rgb;
            let s = ip_unpremultiply(src).rgb;
            return ip_do_color_blend(dst, src, abs(d - s));
        }
        case 23u: { // Exclusion
            let d = ip_unpremultiply(dst).rgb;
            let s = ip_unpremultiply(src).rgb;
            return ip_do_color_blend(dst, src, d + s - 2.0 * d * s);
        }
        case 24u: { // Multiply
            let d = ip_unpremultiply(dst).rgb;
            let s = ip_unpremultiply(src).rgb;
            return ip_do_color_blend(dst, src, d * s);
        }
        default: {
            return src + dst * (1.0 - src.a);
        }
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let kind = frag_info.filter_kind;
    let r = clamp(frag_info.radius, 0, 32);

    if kind == FILTER_KIND_BLUR_1D {
        if r == 0 || frag_info.sigma <= 1.0e-4 {
            return sample_tiled(in.uv, frag_info.tile_mode, frag_info.tex_size) * frag_info.alpha;
        }
        let inv_two_sigma_sq = 0.5 / (frag_info.sigma * frag_info.sigma);
        var acc_color = vec4<f32>(0.0);
        var acc_weight: f32 = 0.0;
        for (var i: i32 = -r; i <= r; i = i + 1) {
            let fi = f32(i);
            let w = exp(-(fi * fi) * inv_two_sigma_sq);
            let sample_uv = in.uv + fi * frag_info.step_uv;
            acc_color = acc_color + sample_tiled(sample_uv, frag_info.tile_mode, frag_info.tex_size) * w;
            acc_weight = acc_weight + w;
        }
        return (acc_color / max(acc_weight, 1.0e-6)) * frag_info.alpha;
    }

    if kind == FILTER_KIND_DILATE_1D {
        var max_c = vec4<f32>(0.0);
        for (var i: i32 = -r; i <= r; i = i + 1) {
            let sample_uv = in.uv + f32(i) * frag_info.step_uv;
            let c = sample_tiled(sample_uv, TILE_MODE_DECAL, frag_info.tex_size);
            max_c = max(max_c, c);
        }
        return max_c * frag_info.alpha;
    }

    if kind == FILTER_KIND_ERODE_1D {
        var min_c = vec4<f32>(1.0);
        for (var i: i32 = -r; i <= r; i = i + 1) {
            let sample_uv = in.uv + f32(i) * frag_info.step_uv;
            let c = sample_tiled(sample_uv, TILE_MODE_DECAL, frag_info.tex_size);
            min_c = min(min_c, c);
        }
        return min_c * frag_info.alpha;
    }

    let pm = textureSampleLevel(filter_texture, filter_sampler, in.uv, 0.0);

    if kind == FILTER_KIND_COLOR_MATRIX {
        let unpm = ip_unpremultiply(pm);
        let out_r = dot(frag_info.color_matrix_row0, unpm) + frag_info.color_matrix_offset.x;
        let out_g = dot(frag_info.color_matrix_row1, unpm) + frag_info.color_matrix_offset.y;
        let out_b = dot(frag_info.color_matrix_row2, unpm) + frag_info.color_matrix_offset.z;
        let out_a = dot(frag_info.color_matrix_row3, unpm) + frag_info.color_matrix_offset.w;
        let clamped = clamp(vec4<f32>(out_r, out_g, out_b, out_a), vec4<f32>(0.0), vec4<f32>(1.0));
        return ip_premultiply(clamped) * frag_info.alpha;
    }

    if kind == FILTER_KIND_COLOR_MODE {
        return ip_blend_premultiplied(pm, frag_info.mode_color_pm, frag_info.blend_mode) * frag_info.alpha;
    }

    if kind == FILTER_KIND_SRGB_TO_LINEAR {
        let u = ip_unpremultiply(pm);
        let lin = vec4<f32>(
            srgb_channel_to_linear(clamp(u.r, 0.0, 1.0)),
            srgb_channel_to_linear(clamp(u.g, 0.0, 1.0)),
            srgb_channel_to_linear(clamp(u.b, 0.0, 1.0)),
            u.a,
        );
        return ip_premultiply(lin) * frag_info.alpha;
    }

    if kind == FILTER_KIND_LINEAR_TO_SRGB {
        let u = ip_unpremultiply(pm);
        let srgb = vec4<f32>(
            linear_channel_to_srgb(clamp(u.r, 0.0, 1.0)),
            linear_channel_to_srgb(clamp(u.g, 0.0, 1.0)),
            linear_channel_to_srgb(clamp(u.b, 0.0, 1.0)),
            u.a,
        );
        return ip_premultiply(srgb) * frag_info.alpha;
    }

    return pm * frag_info.alpha;
}
