// Ported from:
// - impeller/entity/shaders/glyph_atlas.vert
// - impeller/entity/shaders/glyph_atlas.frag

struct FrameInfo {
    mvp: mat4x4<f32>,
};

struct FragInfo {
    use_text_color: f32,
    text_contrast: f32,
    alpha: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> frame_info: FrameInfo;

@group(0) @binding(1)
var<uniform> frag_info: FragInfo;

@group(0) @binding(2)
var alpha_atlas_tex: texture_2d<f32>;

@group(0) @binding(3)
var color_atlas_tex: texture_2d<f32>;

@group(0) @binding(4)
var atlas_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) is_color: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_color: vec4<f32>,
    @location(2) @interpolate(flat) is_color: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = frame_info.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.v_uv = in.uv;
    out.v_color = in.color;
    out.is_color = in.is_color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let alpha_sample = textureSampleLevel(alpha_atlas_tex, atlas_sampler, in.v_uv, 0.0);
    let color_sample = textureSampleLevel(color_atlas_tex, atlas_sampler, in.v_uv, 0.0);

    if in.is_color != 0u {
        if frag_info.use_text_color != 0.0 {
            return color_sample.aaaa * in.v_color * frag_info.alpha;
        }
        return color_sample * in.v_color.a * frag_info.alpha;
    }

    var coverage = alpha_sample.r;
    if frag_info.text_contrast != 1.0 && coverage > 0.0 && coverage < 1.0 {
        coverage = 1.0 - pow(1.0 - coverage, frag_info.text_contrast);
    }
    return in.v_color * coverage * frag_info.alpha;
}
