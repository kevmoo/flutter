// Ported from:
// - impeller/entity/shaders/solid_fill.vert
// - impeller/entity/shaders/solid_fill.frag

struct FrameInfo {
    mvp: mat4x4<f32>,
};

struct FragInfo {
    color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> frame_info: FrameInfo;

@group(0) @binding(1)
var<uniform> frag_info: FragInfo;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = frame_info.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.v_color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.v_color * frag_info.color;
}
