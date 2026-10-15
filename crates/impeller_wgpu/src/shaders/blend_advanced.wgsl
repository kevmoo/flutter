// Ported from:
// - impeller/compiler/shader_lib/impeller/blending.glsl
// - impeller/compiler/shader_lib/impeller/branching.glsl
// - impeller/entity/shaders/blending/advanced_blend.frag

const TILE_MODE_CLAMP: u32 = 0u;
const TILE_MODE_REPEAT: u32 = 1u;
const TILE_MODE_MIRROR: u32 = 2u;
const TILE_MODE_DECAL: u32 = 3u;

const K_EH_CLOSE_ENOUGH: f32 = 0.001;

struct FrameInfo {
    mvp: mat4x4<f32>,
};

struct BlendAdvancedFragInfo {
    dst_uv_offset: vec2<f32>,
    dst_uv_scale: vec2<f32>,
    alpha: f32,
    blend_mode: u32,
    tile_mode_x: u32,
    tile_mode_y: u32,
};

@group(0) @binding(0)
var<uniform> frame_info: FrameInfo;

@group(0) @binding(1)
var<uniform> frag_info: BlendAdvancedFragInfo;

@group(0) @binding(2)
var src_tex: texture_2d<f32>;

@group(0) @binding(3)
var dst_tex: texture_2d<f32>;

@group(0) @binding(4)
var samp: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_clip_pos: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let clip = frame_info.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.clip_position = clip;
    out.v_uv = in.uv;
    out.v_clip_pos = clip;
    return out;
}

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

fn ip_unpremultiply(pm: vec4<f32>) -> vec4<f32> {
    if abs(pm.a) <= K_EH_CLOSE_ENOUGH {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(pm.rgb / pm.a, pm.a);
}

fn ip_component_choose(a: vec3<f32>, b: vec3<f32>, value: vec3<f32>, cutoff: f32) -> vec3<f32> {
    return select(a, b, value > vec3<f32>(cutoff));
}

fn ip_luminosity(color: vec3<f32>) -> f32 {
    return color.r * 0.3 + color.g * 0.59 + color.b * 0.11;
}

fn ip_clip_color(color_in: vec3<f32>) -> vec3<f32> {
    var color = color_in;
    let lum = ip_luminosity(color);
    let mn = min(min(color.r, color.g), color.b);
    let mx = max(max(color.r, color.g), color.b);
    if mn < 0.0 {
        color = vec3<f32>(lum)
            + (((color - vec3<f32>(lum)) * lum) / (lum - mn + K_EH_CLOSE_ENOUGH));
    }
    if mx > 1.0 {
        color = vec3<f32>(lum)
            + (((color - vec3<f32>(lum)) * (1.0 - lum)) / (mx - lum + K_EH_CLOSE_ENOUGH));
    }
    return color;
}

fn ip_set_luminosity(color: vec3<f32>, lum: f32) -> vec3<f32> {
    let relative_lum = lum - ip_luminosity(color);
    return ip_clip_color(color + vec3<f32>(relative_lum));
}

fn ip_saturation(color: vec3<f32>) -> f32 {
    return max(max(color.r, color.g), color.b) - min(min(color.r, color.g), color.b);
}

fn ip_set_saturation(color: vec3<f32>, sat: f32) -> vec3<f32> {
    let mn = min(min(color.r, color.g), color.b);
    let mx = max(max(color.r, color.g), color.b);
    if mn < mx {
        return ((color - vec3<f32>(mn)) * sat) / (mx - mn);
    }
    return vec3<f32>(0.0);
}

fn ip_blend_screen_rgb(dst: vec3<f32>, src: vec3<f32>) -> vec3<f32> {
    return dst + src - (dst * src);
}

fn ip_blend_hard_light_rgb(dst: vec3<f32>, src: vec3<f32>) -> vec3<f32> {
    return ip_component_choose(
        dst * (2.0 * src),
        ip_blend_screen_rgb(dst, 2.0 * src - vec3<f32>(1.0)),
        src,
        0.5,
    );
}

fn ip_color_dodge_channel(d: f32, s: f32) -> f32 {
    if d < K_EH_CLOSE_ENOUGH {
        return 0.0;
    }
    if 1.0 - s < K_EH_CLOSE_ENOUGH {
        return 1.0;
    }
    return min(1.0, d / (1.0 - s));
}

fn ip_blend_color_dodge_rgb(dst: vec3<f32>, src: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        ip_color_dodge_channel(dst.r, src.r),
        ip_color_dodge_channel(dst.g, src.g),
        ip_color_dodge_channel(dst.b, src.b),
    );
}

fn ip_color_burn_channel(d: f32, s: f32) -> f32 {
    if 1.0 - d < K_EH_CLOSE_ENOUGH {
        return 1.0;
    }
    if s < K_EH_CLOSE_ENOUGH {
        return 0.0;
    }
    return 1.0 - min(1.0, (1.0 - d) / s);
}

fn ip_blend_color_burn_rgb(dst: vec3<f32>, src: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        ip_color_burn_channel(dst.r, src.r),
        ip_color_burn_channel(dst.g, src.g),
        ip_color_burn_channel(dst.b, src.b),
    );
}

fn ip_blend_soft_light_rgb(dst: vec3<f32>, src: vec3<f32>) -> vec3<f32> {
    let d_cubic = ((16.0 * dst - vec3<f32>(12.0)) * dst + vec3<f32>(4.0)) * dst;
    let big_d = ip_component_choose(d_cubic, sqrt(max(dst, vec3<f32>(0.0))), dst, 0.25);
    return ip_component_choose(
        dst - (vec3<f32>(1.0) - 2.0 * src) * dst * (vec3<f32>(1.0) - dst),
        dst + (2.0 * src - vec3<f32>(1.0)) * (big_d - dst),
        src,
        0.5,
    );
}

fn ip_apply_blended_color(dst_pm: vec4<f32>, src_pm: vec4<f32>, blend_result: vec3<f32>) -> vec4<f32> {
    let overlap_a = src_pm.a * dst_pm.a;
    let blended_src_pm = vec4<f32>(blend_result * overlap_a, overlap_a) + src_pm * (1.0 - dst_pm.a);
    return blended_src_pm + dst_pm * (1.0 - blended_src_pm.a);
}

fn ip_blend_advanced(dst_pm: vec4<f32>, src_pm: vec4<f32>, mode: u32) -> vec4<f32> {
    switch mode {
        case 0u: { // Clear
            return vec4<f32>(0.0);
        }
        case 1u: { // Src
            return src_pm;
        }
        case 2u: { // Dst
            return dst_pm;
        }
        case 3u: { // SrcOver
            return src_pm + dst_pm * (1.0 - src_pm.a);
        }
        case 4u: { // DstOver
            return dst_pm + src_pm * (1.0 - dst_pm.a);
        }
        case 5u: { // SrcIn
            return src_pm * dst_pm.a;
        }
        case 6u: { // DstIn
            return dst_pm * src_pm.a;
        }
        case 7u: { // SrcOut
            return src_pm * (1.0 - dst_pm.a);
        }
        case 8u: { // DstOut
            return dst_pm * (1.0 - src_pm.a);
        }
        case 9u: { // SrcATop
            return src_pm * dst_pm.a + dst_pm * (1.0 - src_pm.a);
        }
        case 10u: { // DstATop
            return dst_pm * src_pm.a + src_pm * (1.0 - dst_pm.a);
        }
        case 11u: { // Xor
            return src_pm * (1.0 - dst_pm.a) + dst_pm * (1.0 - src_pm.a);
        }
        case 12u: { // Plus
            return min(src_pm + dst_pm, vec4<f32>(1.0));
        }
        case 13u: { // Modulate
            return src_pm * dst_pm;
        }
        case 14u: { // Screen
            return src_pm + dst_pm - src_pm * dst_pm;
        }
        case 15u: { // Overlay
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            return ip_apply_blended_color(dst_pm, src_pm, ip_blend_hard_light_rgb(s, d));
        }
        case 16u: { // Darken
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            return ip_apply_blended_color(dst_pm, src_pm, min(d, s));
        }
        case 17u: { // Lighten
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            return ip_apply_blended_color(dst_pm, src_pm, max(d, s));
        }
        case 18u: { // ColorDodge
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            return ip_apply_blended_color(dst_pm, src_pm, ip_blend_color_dodge_rgb(d, s));
        }
        case 19u: { // ColorBurn
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            return ip_apply_blended_color(dst_pm, src_pm, ip_blend_color_burn_rgb(d, s));
        }
        case 20u: { // HardLight
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            return ip_apply_blended_color(dst_pm, src_pm, ip_blend_hard_light_rgb(d, s));
        }
        case 21u: { // SoftLight
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            return ip_apply_blended_color(dst_pm, src_pm, ip_blend_soft_light_rgb(d, s));
        }
        case 22u: { // Difference
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            return ip_apply_blended_color(dst_pm, src_pm, abs(d - s));
        }
        case 23u: { // Exclusion
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            return ip_apply_blended_color(dst_pm, src_pm, d + s - 2.0 * d * s);
        }
        case 24u: { // Multiply
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            return ip_apply_blended_color(dst_pm, src_pm, d * s);
        }
        case 25u: { // Hue
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            let res = ip_set_luminosity(ip_set_saturation(s, ip_saturation(d)), ip_luminosity(d));
            return ip_apply_blended_color(dst_pm, src_pm, res);
        }
        case 26u: { // Saturation
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            let res = ip_set_luminosity(ip_set_saturation(d, ip_saturation(s)), ip_luminosity(d));
            return ip_apply_blended_color(dst_pm, src_pm, res);
        }
        case 27u: { // Color
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            let res = ip_set_luminosity(s, ip_luminosity(d));
            return ip_apply_blended_color(dst_pm, src_pm, res);
        }
        case 28u: { // Luminosity
            let d = ip_unpremultiply(dst_pm).rgb;
            let s = ip_unpremultiply(src_pm).rgb;
            let res = ip_set_luminosity(d, ip_luminosity(s));
            return ip_apply_blended_color(dst_pm, src_pm, res);
        }
        default: {
            return src_pm + dst_pm * (1.0 - src_pm.a);
        }
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let src_uv = vec2<f32>(
        ip_float_tile(in.v_uv.x, frag_info.tile_mode_x),
        ip_float_tile(in.v_uv.y, frag_info.tile_mode_y),
    );
    var src_pm = textureSampleLevel(src_tex, samp, src_uv, 0.0) * clamp(frag_info.alpha, 0.0, 1.0);
    let out_of_bounds_x =
        frag_info.tile_mode_x == TILE_MODE_DECAL && (in.v_uv.x < 0.0 || in.v_uv.x > 1.0);
    let out_of_bounds_y =
        frag_info.tile_mode_y == TILE_MODE_DECAL && (in.v_uv.y < 0.0 || in.v_uv.y > 1.0);
    if out_of_bounds_x || out_of_bounds_y {
        src_pm = vec4<f32>(0.0);
    }

    let ndc = in.v_clip_pos.xy / in.v_clip_pos.w;
    let parent_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let dst_uv = (parent_uv - frag_info.dst_uv_offset) * frag_info.dst_uv_scale;
    let dst_dims = vec2<f32>(textureDimensions(dst_tex));
    let dst_px = clamp(
        vec2<i32>(floor(dst_uv * dst_dims)),
        vec2<i32>(0),
        max(vec2<i32>(dst_dims) - vec2<i32>(1), vec2<i32>(0)),
    );
    let dst_pm = textureLoad(dst_tex, dst_px, 0);

    return clamp(
        ip_blend_advanced(dst_pm, src_pm, frag_info.blend_mode),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
}
