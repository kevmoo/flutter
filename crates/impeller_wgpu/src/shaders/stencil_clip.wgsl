// Ported from:
// - impeller/entity/shaders/clip.vert
// - impeller/entity/shaders/clip.frag

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
};

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return frame_info.mvp * vec4<f32>(in.position, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return frag_info.color;
}
