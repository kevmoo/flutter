#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use impeller_core::{BlendFactor, BlendOperation, ColorAttachmentDescriptor, PixelFormat};
use impeller_geometry::{
    BlendMode, Color, Matrix, Point, Rect, Scalar, Size, TileMode, K_EH_CLOSE_ENOUGH,
};
use impeller_tessellator::{TessellatedMesh, Vertex2D};
pub use impeller_typographer::{AtlasGlyphQuad, GlyphAtlas, GlyphAtlasType};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path as FsPath;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

const SOLID_FILL_WGSL: &str = include_str!("shaders/solid_fill.wgsl");
const LINEAR_GRADIENT_FILL_WGSL: &str = include_str!("shaders/linear_gradient_fill.wgsl");
const RADIAL_GRADIENT_FILL_WGSL: &str = include_str!("shaders/radial_gradient_fill.wgsl");
const RRECT_BLUR_WGSL: &str = include_str!("shaders/rrect_blur.wgsl");
const GLYPH_ATLAS_WGSL: &str = include_str!("shaders/glyph_atlas.wgsl");

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SolidVertex {
    pub position: [Scalar; 2],
}

impl SolidVertex {
    pub const fn new(x: Scalar, y: Scalar) -> Self {
        Self { position: [x, y] }
    }
}

impl From<Vertex2D> for SolidVertex {
    fn from(v: Vertex2D) -> Self {
        Self {
            position: v.position,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct UvVertex {
    pub position: [Scalar; 2],
    pub uv: [Scalar; 2],
}

impl UvVertex {
    pub const fn new(x: Scalar, y: Scalar, u: Scalar, v: Scalar) -> Self {
        Self {
            position: [x, y],
            uv: [u, v],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GlyphVertex {
    pub position: [Scalar; 2],
    pub uv: [Scalar; 2],
    pub color: [Scalar; 4],
    pub is_color: u32,
}

impl GlyphVertex {
    pub const fn new(
        position: [Scalar; 2],
        uv: [Scalar; 2],
        color: [Scalar; 4],
        is_color: u32,
    ) -> Self {
        Self {
            position,
            uv,
            color,
            is_color,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GradientStop {
    pub offset: Scalar,
    pub color: Color,
}

impl GradientStop {
    pub const fn new(offset: Scalar, color: Color) -> Self {
        Self { offset, color }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearGradientParams {
    pub start_point: Point,
    pub end_point: Point,
    pub stops: Vec<GradientStop>,
    pub tile_mode: TileMode,
    pub alpha: Scalar,
}

impl LinearGradientParams {
    pub fn new(
        start_point: Point,
        end_point: Point,
        stops: Vec<GradientStop>,
        tile_mode: TileMode,
        alpha: Scalar,
    ) -> Self {
        Self {
            start_point,
            end_point,
            stops,
            tile_mode,
            alpha,
        }
    }

    pub fn two_stop(
        start_point: Point,
        end_point: Point,
        start_color: Color,
        end_color: Color,
        tile_mode: TileMode,
    ) -> Self {
        Self {
            start_point,
            end_point,
            stops: vec![
                GradientStop::new(0.0, start_color),
                GradientStop::new(1.0, end_color),
            ],
            tile_mode,
            alpha: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RadialGradientParams {
    pub center: Point,
    pub radius: Scalar,
    pub stops: Vec<GradientStop>,
    pub tile_mode: TileMode,
    pub alpha: Scalar,
}

impl RadialGradientParams {
    pub fn new(
        center: Point,
        radius: Scalar,
        stops: Vec<GradientStop>,
        tile_mode: TileMode,
        alpha: Scalar,
    ) -> Self {
        Self {
            center,
            radius,
            stops,
            tile_mode,
            alpha,
        }
    }

    pub fn two_stop(
        center: Point,
        radius: Scalar,
        center_color: Color,
        edge_color: Color,
        tile_mode: TileMode,
    ) -> Self {
        Self {
            center,
            radius,
            stops: vec![
                GradientStop::new(0.0, center_color),
                GradientStop::new(1.0, edge_color),
            ],
            tile_mode,
            alpha: 1.0,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RRectBlurParams {
    pub rect: Rect,
    pub corner_radius: Scalar,
    pub blur_sigma: Scalar,
    pub color: Color,
}

impl RRectBlurParams {
    pub const fn new(
        rect: Rect,
        corner_radius: Scalar,
        blur_sigma: Scalar,
        color: Color,
    ) -> Self {
        Self {
            rect,
            corner_radius,
            blur_sigma,
            color,
        }
    }

    pub fn pad_for_sigma(sigma: Scalar) -> Scalar {
        let s = sigma.max(0.0);
        let factor = ((1.0 / 47.6) * s + 2.5).min(3.5);
        s * factor
    }

    pub fn coverage_rect(&self) -> Rect {
        let pad = Self::pad_for_sigma(self.blur_sigma.max(K_EH_CLOSE_ENOUGH));
        let left = self.rect.get_left() - pad;
        let top = self.rect.get_top() - pad;
        let right = self.rect.get_right() + pad;
        let bottom = self.rect.get_bottom() + pad;
        Rect::from_ltrb(left, top, right, bottom)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolidFillBatch {
    pub mesh: TessellatedMesh,
    pub transform: Matrix,
    pub color: Color,
    pub blend_mode: BlendMode,
    pub clip_rect: Option<Rect>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearGradientBatch {
    pub mesh: TessellatedMesh,
    pub transform: Matrix,
    pub gradient: LinearGradientParams,
    pub blend_mode: BlendMode,
    pub clip_rect: Option<Rect>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RadialGradientBatch {
    pub mesh: TessellatedMesh,
    pub transform: Matrix,
    pub gradient: RadialGradientParams,
    pub blend_mode: BlendMode,
    pub clip_rect: Option<Rect>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RRectBlurBatch {
    pub transform: Matrix,
    pub params: RRectBlurParams,
    pub blend_mode: BlendMode,
    pub clip_rect: Option<Rect>,
}

#[derive(Clone, Debug)]
pub struct GlyphAtlasBatch {
    pub quads: Vec<AtlasGlyphQuad>,
    pub transform: Matrix,
    pub alpha_atlas: Option<Arc<GlyphAtlas>>,
    pub color_atlas: Option<Arc<GlyphAtlas>>,
    pub blend_mode: BlendMode,
    pub clip_rect: Option<Rect>,
    pub linear_filter: Option<bool>,
    pub use_text_color: bool,
    pub text_contrast: Scalar,
    pub alpha: Scalar,
}

impl PartialEq for GlyphAtlasBatch {
    fn eq(&self, other: &Self) -> bool {
        let same_atlas = |a: &Option<Arc<GlyphAtlas>>, b: &Option<Arc<GlyphAtlas>>| match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => {
                Arc::ptr_eq(x, y)
                    || (x.atlas_type == y.atlas_type
                        && x.size == y.size
                        && x.generation == y.generation
                        && x.pixels == y.pixels)
            }
            _ => false,
        };
        self.quads == other.quads
            && self.transform == other.transform
            && same_atlas(&self.alpha_atlas, &other.alpha_atlas)
            && same_atlas(&self.color_atlas, &other.color_atlas)
            && self.blend_mode == other.blend_mode
            && self.clip_rect == other.clip_rect
            && self.linear_filter == other.linear_filter
            && self.use_text_color == other.use_text_color
            && self.text_contrast == other.text_contrast
            && self.alpha == other.alpha
    }
}

pub const DEFAULT_TEXT_CONTRAST: Scalar = 1.38;

impl GlyphAtlasBatch {
    pub fn new(
        quads: Vec<AtlasGlyphQuad>,
        transform: Matrix,
        alpha_atlas: Option<Arc<GlyphAtlas>>,
        color_atlas: Option<Arc<GlyphAtlas>>,
        blend_mode: BlendMode,
        clip_rect: Option<Rect>,
    ) -> Self {
        Self {
            quads,
            transform,
            alpha_atlas,
            color_atlas,
            blend_mode,
            clip_rect,
            linear_filter: None,
            use_text_color: false,
            text_contrast: DEFAULT_TEXT_CONTRAST,
            alpha: 1.0,
        }
    }

    pub fn from_atlas(
        quads: Vec<AtlasGlyphQuad>,
        transform: Matrix,
        atlas: Arc<GlyphAtlas>,
        blend_mode: BlendMode,
        clip_rect: Option<Rect>,
    ) -> Self {
        let (alpha_atlas, color_atlas) = match atlas.atlas_type {
            GlyphAtlasType::AlphaBitmap => (Some(atlas), None),
            GlyphAtlasType::ColorBitmap => (None, Some(atlas)),
        };
        Self::new(
            quads,
            transform,
            alpha_atlas,
            color_atlas,
            blend_mode,
            clip_rect,
        )
    }

    pub fn from_typographer(
        quads: Vec<AtlasGlyphQuad>,
        transform: Matrix,
        alpha_atlas: &GlyphAtlas,
        color_atlas: &GlyphAtlas,
        blend_mode: BlendMode,
        clip_rect: Option<Rect>,
    ) -> Self {
        let needs_alpha = quads.iter().any(|q| !q.is_color);
        let needs_color = quads.iter().any(|q| q.is_color);
        Self::new(
            quads,
            transform,
            if needs_alpha {
                Some(Arc::new(alpha_atlas.clone()))
            } else {
                None
            },
            if needs_color {
                Some(Arc::new(color_atlas.clone()))
            } else {
                None
            },
            blend_mode,
            clip_rect,
        )
    }

    pub const fn mvp(&self) -> Matrix {
        self.transform
    }

    pub fn with_mvp(mut self, transform: Matrix) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_linear_filter(mut self, linear: bool) -> Self {
        self.linear_filter = Some(linear);
        self
    }

    pub fn with_use_text_color(mut self, use_text_color: bool) -> Self {
        self.use_text_color = use_text_color;
        self
    }

    pub fn with_text_contrast(mut self, text_contrast: Scalar) -> Self {
        self.text_contrast = text_contrast;
        self
    }

    pub fn with_alpha(mut self, alpha: Scalar) -> Self {
        self.alpha = alpha;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DrawCommand {
    SolidFill {
        mesh: TessellatedMesh,
        transform: Matrix,
        color: Color,
        blend_mode: BlendMode,
        clip_rect: Option<Rect>,
    },
    LinearGradientFill {
        mesh: TessellatedMesh,
        transform: Matrix,
        gradient: LinearGradientParams,
        blend_mode: BlendMode,
        clip_rect: Option<Rect>,
    },
    RadialGradientFill {
        mesh: TessellatedMesh,
        transform: Matrix,
        gradient: RadialGradientParams,
        blend_mode: BlendMode,
        clip_rect: Option<Rect>,
    },
    RRectBlur {
        transform: Matrix,
        params: RRectBlurParams,
        blend_mode: BlendMode,
        clip_rect: Option<Rect>,
    },
    GlyphAtlas(GlyphAtlasBatch),
}

impl DrawCommand {
    pub fn clip_rect(&self) -> Option<Rect> {
        match self {
            DrawCommand::SolidFill { clip_rect, .. }
            | DrawCommand::LinearGradientFill { clip_rect, .. }
            | DrawCommand::RadialGradientFill { clip_rect, .. }
            | DrawCommand::RRectBlur { clip_rect, .. } => *clip_rect,
            DrawCommand::GlyphAtlas(batch) => batch.clip_rect,
        }
    }

    pub fn blend_mode(&self) -> BlendMode {
        match self {
            DrawCommand::SolidFill { blend_mode, .. }
            | DrawCommand::LinearGradientFill { blend_mode, .. }
            | DrawCommand::RadialGradientFill { blend_mode, .. }
            | DrawCommand::RRectBlur { blend_mode, .. } => *blend_mode,
            DrawCommand::GlyphAtlas(batch) => batch.blend_mode,
        }
    }

    pub fn with_clip_rect(mut self, new_clip: Option<Rect>) -> Self {
        match &mut self {
            DrawCommand::SolidFill { clip_rect, .. }
            | DrawCommand::LinearGradientFill { clip_rect, .. }
            | DrawCommand::RadialGradientFill { clip_rect, .. }
            | DrawCommand::RRectBlur { clip_rect, .. } => {
                *clip_rect = new_clip;
            }
            DrawCommand::GlyphAtlas(batch) => {
                batch.clip_rect = new_clip;
            }
        }
        self
    }
}

impl From<SolidFillBatch> for DrawCommand {
    fn from(b: SolidFillBatch) -> Self {
        DrawCommand::SolidFill {
            mesh: b.mesh,
            transform: b.transform,
            color: b.color,
            blend_mode: b.blend_mode,
            clip_rect: b.clip_rect,
        }
    }
}

impl From<LinearGradientBatch> for DrawCommand {
    fn from(b: LinearGradientBatch) -> Self {
        DrawCommand::LinearGradientFill {
            mesh: b.mesh,
            transform: b.transform,
            gradient: b.gradient,
            blend_mode: b.blend_mode,
            clip_rect: b.clip_rect,
        }
    }
}

impl From<RadialGradientBatch> for DrawCommand {
    fn from(b: RadialGradientBatch) -> Self {
        DrawCommand::RadialGradientFill {
            mesh: b.mesh,
            transform: b.transform,
            gradient: b.gradient,
            blend_mode: b.blend_mode,
            clip_rect: b.clip_rect,
        }
    }
}

impl From<RRectBlurBatch> for DrawCommand {
    fn from(b: RRectBlurBatch) -> Self {
        DrawCommand::RRectBlur {
            transform: b.transform,
            params: b.params,
            blend_mode: b.blend_mode,
            clip_rect: b.clip_rect,
        }
    }
}

impl From<GlyphAtlasBatch> for DrawCommand {
    fn from(b: GlyphAtlasBatch) -> Self {
        DrawCommand::GlyphAtlas(b)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PipelineKind {
    SolidFill,
    LinearGradientFill,
    RadialGradientFill,
    RRectBlur,
    GlyphAtlas,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct SolidFrameUniform {
    mvp: [f32; 16],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct SolidFragUniform {
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GradientFrameUniform {
    mvp: [f32; 16],
    matrix: [f32; 16],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct LinearGradientFragUniform {
    start_point: [f32; 2],
    end_point: [f32; 2],
    start_to_end: [f32; 2],
    alpha: f32,
    inverse_dot_start_to_end: f32,
    decal_border_color: [f32; 4],
    fast_color_0: [f32; 4],
    fast_color_1: [f32; 4],
    tile_mode: u32,
    colors_length: u32,
    is_fast_two_stop: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct RadialGradientFragUniform {
    center: [f32; 2],
    radius: f32,
    alpha: f32,
    decal_border_color: [f32; 4],
    fast_color_0: [f32; 4],
    fast_color_1: [f32; 4],
    tile_mode: u32,
    colors_length: u32,
    is_fast_two_stop: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GpuColorPoint {
    color: [f32; 4],
    stop: f32,
    inverse_delta: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct RRectBlurFragUniform {
    color: [f32; 4],
    center_adjust: [f32; 4],
    r1_exponent_exponent_inv: [f32; 4],
    s_inv_min_edge_scale: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GlyphAtlasFragUniform {
    use_text_color: f32,
    text_contrast: f32,
    alpha: f32,
    _pad: f32,
}

fn map_blend_factor(factor: BlendFactor) -> wgpu::BlendFactor {
    match factor {
        BlendFactor::Zero => wgpu::BlendFactor::Zero,
        BlendFactor::One => wgpu::BlendFactor::One,
        BlendFactor::SourceAlpha => wgpu::BlendFactor::SrcAlpha,
        BlendFactor::OneMinusSourceAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        BlendFactor::DestinationAlpha => wgpu::BlendFactor::DstAlpha,
        BlendFactor::OneMinusDestinationAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        BlendFactor::SourceColor => wgpu::BlendFactor::Src,
        BlendFactor::OneMinusSourceColor => wgpu::BlendFactor::OneMinusSrc,
        BlendFactor::DestinationColor => wgpu::BlendFactor::Dst,
        BlendFactor::OneMinusDestinationColor => wgpu::BlendFactor::OneMinusDst,
        BlendFactor::SourceAlphaSaturated => wgpu::BlendFactor::SrcAlphaSaturated,
        BlendFactor::BlendColor | BlendFactor::BlendAlpha => wgpu::BlendFactor::Constant,
        BlendFactor::OneMinusBlendColor | BlendFactor::OneMinusBlendAlpha => {
            wgpu::BlendFactor::OneMinusConstant
        }
    }
}

fn map_blend_operation(op: BlendOperation) -> wgpu::BlendOperation {
    match op {
        BlendOperation::Add => wgpu::BlendOperation::Add,
        BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
        BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
    }
}

pub fn blend_state_for_mode(mode: BlendMode) -> wgpu::BlendState {
    match mode {
        BlendMode::Multiply => wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        },
        BlendMode::Darken => wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Min,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        },
        BlendMode::Lighten => wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        },
        _ => {
            let desc =
                ColorAttachmentDescriptor::for_blend_mode(PixelFormat::R8G8B8A8UNormInt, mode);
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: map_blend_factor(desc.src_color_blend_factor),
                    dst_factor: map_blend_factor(desc.dst_color_blend_factor),
                    operation: map_blend_operation(desc.color_blend_op),
                },
                alpha: wgpu::BlendComponent {
                    src_factor: map_blend_factor(desc.src_alpha_blend_factor),
                    dst_factor: map_blend_factor(desc.dst_alpha_blend_factor),
                    operation: map_blend_operation(desc.alpha_blend_op),
                },
            }
        }
    }
}

pub fn compute_erf7(x_in: Scalar) -> Scalar {
    let mut x = x_in * std::f32::consts::FRAC_2_SQRT_PI;
    let xx = x * x;
    x += (0.24295 + (0.03395 + 0.0104 * xx) * xx) * (x * xx);
    x / (1.0 + x * x).sqrt()
}

pub fn ip_erf(x: Scalar) -> Scalar {
    let a = x.abs();
    let b = (0.278393 + (0.230389 + 0.078108 * a * a) * a) * a + 1.0;
    let b2 = b * b;
    let b4 = b2 * b2;
    x.signum() * (1.0 - 1.0 / b4)
}

pub fn ip_gaussian_integral(x: Scalar, sigma: Scalar) -> Scalar {
    const HALF_SQRT_TWO: Scalar = std::f32::consts::FRAC_1_SQRT_2;
    let s = sigma.max(K_EH_CLOSE_ENOUGH);
    (1.0 + ip_erf(x * (HALF_SQRT_TWO / s))) * 0.5
}

fn build_rrect_blur_frag_uniform(
    params: &RRectBlurParams,
    blend_mode: BlendMode,
) -> RRectBlurFragUniform {
    let width = (params.rect.get_right() - params.rect.get_left())
        .abs()
        .max(K_EH_CLOSE_ENOUGH);
    let height = (params.rect.get_bottom() - params.rect.get_top())
        .abs()
        .max(K_EH_CLOSE_ENOUGH);
    let blur_sigma = params.blur_sigma.clamp(K_EH_CLOSE_ENOUGH, 500.0);
    let radius = params
        .corner_radius
        .clamp(K_EH_CLOSE_ENOUGH, (width * 0.5).max(K_EH_CLOSE_ENOUGH))
        .min(
            params
                .corner_radius
                .clamp(K_EH_CLOSE_ENOUGH, (height * 0.5).max(K_EH_CLOSE_ENOUGH)),
        );

    let sigma = (blur_sigma * std::f32::consts::SQRT_2).max(1.0);
    let mut r_size = Point::new(width, height);
    let center = r_size * 0.5;
    let min_edge = r_size.x.min(r_size.y);
    let r_max = (0.5 * min_edge).max(K_EH_CLOSE_ENOUGH);
    let r0 = radius.hypot(sigma * 1.15).min(r_max).max(K_EH_CLOSE_ENOUGH);
    let r1 = radius.hypot(sigma * 2.0).min(r_max).max(K_EH_CLOSE_ENOUGH);

    let exponent = (2.0 * r1 / r0).max(K_EH_CLOSE_ENOUGH);
    let s_inv = 1.0 / sigma;

    let vx = r_size.x * s_inv * 0.5;
    let vy = r_size.y * s_inv * 0.5;
    let eccentric_x = (-vx * vx).exp();
    let eccentric_y = (-vy * vy).exp();
    let delta = 1.25 * sigma * (eccentric_x - eccentric_y);
    r_size.x += delta.min(0.0);
    r_size.y += delta.max(0.0);

    let adjust = Point::new(r_size.x * 0.5 - r1, r_size.y * 0.5 - r1);
    let exponent_inv = 1.0 / exponent;
    let scale = 0.5 * compute_erf7(s_inv * 0.5 * (r_size.x.max(r_size.y) - 0.5 * radius));

    let effective_color = if blend_mode == BlendMode::Clear {
        Color::WHITE.to_array_premultiplied()
    } else {
        params.color.to_array_premultiplied()
    };

    RRectBlurFragUniform {
        color: effective_color,
        center_adjust: [center.x, center.y, adjust.x, adjust.y],
        r1_exponent_exponent_inv: [r1, exponent, exponent_inv, sigma],
        s_inv_min_edge_scale: [s_inv, min_edge, scale, 0.0],
    }
}

fn build_gpu_color_points(
    stops: &[GradientStop],
) -> (Vec<GpuColorPoint>, bool, [f32; 4], [f32; 4]) {
    if stops.is_empty() {
        return (
            vec![GpuColorPoint::default()],
            true,
            [0.0; 4],
            [0.0; 4],
        );
    }

    let mut sorted = stops.to_vec();
    sorted.sort_by(|a, b| {
        a.offset
            .partial_cmp(&b.offset)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let is_fast_two_stop = sorted.len() == 2
        && (sorted[0].offset - 0.0).abs() <= K_EH_CLOSE_ENOUGH
        && (sorted[1].offset - 1.0).abs() <= K_EH_CLOSE_ENOUGH;

    let c0 = sorted.first().map(|s| s.color).unwrap_or(Color::TRANSPARENT);
    let c1 = sorted.last().map(|s| s.color).unwrap_or(c0);
    let fast_0 = [c0.red, c0.green, c0.blue, c0.alpha];
    let fast_1 = [c1.red, c1.green, c1.blue, c1.alpha];

    let mut gpu_stops = Vec::with_capacity(sorted.len());
    let mut last_stop = sorted[0].offset;
    for (i, s) in sorted.iter().enumerate() {
        let cur_stop = s.offset.clamp(0.0, 1.0);
        let delta = if i == 0 { 0.0 } else { cur_stop - last_stop };
        let inverse_delta = if delta.abs() <= 1.0e-6 {
            10000.0
        } else {
            1.0 / delta
        };
        gpu_stops.push(GpuColorPoint {
            color: [s.color.red, s.color.green, s.color.blue, s.color.alpha],
            stop: cur_stop,
            inverse_delta,
            _pad: [0.0; 2],
        });
        last_stop = cur_stop;
    }

    (gpu_stops, is_fast_two_stop, fast_0, fast_1)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct AtlasCacheKey {
    atlas_type: GlyphAtlasType,
    generation: u64,
    width: u32,
    height: u32,
    pixel_hash: u64,
}

impl AtlasCacheKey {
    fn from_atlas(atlas: &GlyphAtlas) -> Self {
        let mut hasher = DefaultHasher::new();
        atlas.pixels.hash(&mut hasher);
        Self {
            atlas_type: atlas.atlas_type,
            generation: atlas.generation,
            width: atlas.size.width.max(1) as u32,
            height: atlas.size.height.max(1) as u32,
            pixel_hash: hasher.finish(),
        }
    }
}

struct CachedGpuAtlas {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

pub const DEFAULT_SAMPLE_COUNT: u32 = 4;

fn normalize_sample_count(sample_count: u32) -> u32 {
    if sample_count >= 4 {
        4
    } else {
        1
    }
}

pub struct OffscreenTarget {
    width: u32,
    height: u32,
    internal_width: u32,
    internal_height: u32,
    ssaa_scale: u32,
    sample_count: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    _msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
}

impl OffscreenTarget {
    pub fn new(ctx: &WgpuRenderer, width: u32, height: u32) -> Self {
        Self::new_with_sample_count(ctx, width, height, DEFAULT_SAMPLE_COUNT)
    }

    pub fn new_with_sample_count(
        ctx: &WgpuRenderer,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let samples = normalize_sample_count(sample_count);
        let ssaa_scale = if samples > 1 { 2 } else { 1 };
        let internal_width = w * ssaa_scale;
        let internal_height = h * ssaa_scale;
        let extent = wgpu::Extent3d {
            width: internal_width,
            height: internal_height,
            depth_or_array_layers: 1,
        };
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Impeller OffscreenTarget Resolve Texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (_msaa_texture, msaa_view) = if samples > 1 {
            let msaa_tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Impeller OffscreenTarget 4x MSAA Texture"),
                size: extent,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: TARGET_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let msaa_v = msaa_tex.create_view(&wgpu::TextureViewDescriptor::default());
            (Some(msaa_tex), Some(msaa_v))
        } else {
            (None, None)
        };

        Self {
            width: w,
            height: h,
            internal_width,
            internal_height,
            ssaa_scale,
            sample_count: samples,
            texture,
            view,
            _msaa_texture,
            msaa_view,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn internal_width(&self) -> u32 {
        self.internal_width
    }

    pub fn internal_height(&self) -> u32 {
        self.internal_height
    }

    pub fn ssaa_scale(&self) -> u32 {
        self.ssaa_scale
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn msaa_view(&self) -> Option<&wgpu::TextureView> {
        self.msaa_view.as_ref()
    }

    pub fn render(
        &self,
        ctx: &WgpuRenderer,
        clear_color: Color,
        commands: &[DrawCommand],
    ) -> Result<(), String> {
        ctx.encode_and_submit(self, clear_color, commands)
    }

    pub fn readback_rgba(&self, ctx: &WgpuRenderer) -> Result<Vec<u8>, String> {
        ctx.readback_target(self)
    }

    pub async fn readback_rgba_async(&self, ctx: &WgpuRenderer) -> Result<Vec<u8>, String> {
        ctx.readback_target_async(self).await
    }

    pub fn render_and_readback(
        &self,
        ctx: &WgpuRenderer,
        clear_color: Color,
        commands: &[DrawCommand],
    ) -> Result<Vec<u8>, String> {
        self.render(ctx, clear_color, commands)?;
        self.readback_rgba(ctx)
    }
}

struct PreparedDrawCall {
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    scissor: [u32; 4],
}

pub struct WgpuRenderer {
    pub adapter_name: String,
    pub backend_name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    solid_shader: wgpu::ShaderModule,
    linear_gradient_shader: wgpu::ShaderModule,
    radial_gradient_shader: wgpu::ShaderModule,
    rrect_blur_shader: wgpu::ShaderModule,
    glyph_atlas_shader: wgpu::ShaderModule,
    uniform_pair_layout: wgpu::BindGroupLayout,
    gradient_layout: wgpu::BindGroupLayout,
    glyph_atlas_layout: wgpu::BindGroupLayout,
    linear_sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,
    fallback_alpha_atlas: Arc<CachedGpuAtlas>,
    fallback_color_atlas: Arc<CachedGpuAtlas>,
    pipeline_cache: Mutex<HashMap<(PipelineKind, u32, u32), Arc<wgpu::RenderPipeline>>>,
    atlas_cache: Mutex<HashMap<AtlasCacheKey, Arc<CachedGpuAtlas>>>,
}

pub type WgpuContext = WgpuRenderer;

impl WgpuRenderer {
    pub fn new() -> Result<Self, String> {
        Self::new_headless()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_headless() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_headless() -> Result<Self, String> {
        Err(
            "Synchronous WgpuRenderer::new_headless is unavailable on wasm32; use new_async or new_for_canvas"
                .to_string(),
        )
    }

    pub async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .or_else(|| {
                // Retry with fallback adapter on platforms where only software rasterizer is available.
                None
            });

        let adapter = match adapter {
            Some(a) => a,
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::None,
                    force_fallback_adapter: true,
                    compatible_surface: None,
                })
                .await
                .ok_or_else(|| "Failed to find a suitable wgpu Adapter".to_string())?,
        };

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| format!("Failed to request wgpu Device: {e}"))?;

        Ok(Self::from_device_and_queue(
            adapter_name,
            backend_name,
            device,
            queue,
        ))
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn new_for_canvas(
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(Self, wgpu::Surface<'static>, wgpu::SurfaceConfiguration), String> {
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("Failed to create canvas surface: {e}"))?;
        Self::init_with_surface(instance, surface, width, height).await
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn new_for_offscreen_canvas(
        canvas: web_sys::OffscreenCanvas,
    ) -> Result<(Self, wgpu::Surface<'static>, wgpu::SurfaceConfiguration), String> {
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas))
            .map_err(|e| format!("Failed to create offscreen canvas surface: {e}"))?;
        Self::init_with_surface(instance, surface, width, height).await
    }

    #[cfg(target_arch = "wasm32")]
    async fn init_with_surface(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<(Self, wgpu::Surface<'static>, wgpu::SurfaceConfiguration), String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or_else(|| "Failed to request WebGPU/WebGL2 adapter for canvas".to_string())?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to request WebGPU/WebGL2 device: {e}"))?;

        let config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| "Surface incompatible with adapter".to_string())?;
        surface.configure(&device, &config);

        let ctx = Self::from_device_and_queue(adapter_name, backend_name, device, queue);
        Ok((ctx, surface, config))
    }

    fn from_device_and_queue(
        adapter_name: String,
        backend_name: String,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        let solid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller solid_fill.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SOLID_FILL_WGSL.into()),
        });
        let linear_gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller linear_gradient_fill.wgsl"),
            source: wgpu::ShaderSource::Wgsl(LINEAR_GRADIENT_FILL_WGSL.into()),
        });
        let radial_gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller radial_gradient_fill.wgsl"),
            source: wgpu::ShaderSource::Wgsl(RADIAL_GRADIENT_FILL_WGSL.into()),
        });
        let rrect_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller rrect_blur.wgsl"),
            source: wgpu::ShaderSource::Wgsl(RRECT_BLUR_WGSL.into()),
        });
        let glyph_atlas_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller glyph_atlas.wgsl"),
            source: wgpu::ShaderSource::Wgsl(GLYPH_ATLAS_WGSL.into()),
        });

        let uniform_pair_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Impeller Uniform Pair BindGroupLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let gradient_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Impeller Gradient BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let glyph_atlas_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Impeller GlyphAtlas BindGroupLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Impeller GlyphAtlas Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Impeller GlyphAtlas Nearest Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let fallback_alpha_atlas = Arc::new(Self::upload_atlas_raw(
            &device,
            &queue,
            GlyphAtlasType::AlphaBitmap,
            1,
            1,
            &[255u8],
        ));
        let fallback_color_atlas = Arc::new(Self::upload_atlas_raw(
            &device,
            &queue,
            GlyphAtlasType::ColorBitmap,
            1,
            1,
            &[255u8, 255, 255, 255],
        ));

        Self {
            adapter_name,
            backend_name,
            device,
            queue,
            solid_shader,
            linear_gradient_shader,
            radial_gradient_shader,
            rrect_blur_shader,
            glyph_atlas_shader,
            uniform_pair_layout,
            gradient_layout,
            glyph_atlas_layout,
            linear_sampler,
            nearest_sampler,
            fallback_alpha_atlas,
            fallback_color_atlas,
            pipeline_cache: Mutex::new(HashMap::new()),
            atlas_cache: Mutex::new(HashMap::new()),
        }
    }

    fn upload_atlas_raw(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas_type: GlyphAtlasType,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> CachedGpuAtlas {
        let w = width.max(1);
        let h = height.max(1);
        let (format, bpp, label) = match atlas_type {
            GlyphAtlasType::AlphaBitmap => (
                wgpu::TextureFormat::R8Unorm,
                1u32,
                "Impeller Alpha GlyphAtlas R8Unorm",
            ),
            GlyphAtlasType::ColorBitmap => (
                wgpu::TextureFormat::Rgba8Unorm,
                4u32,
                "Impeller Color GlyphAtlas Rgba8Unorm",
            ),
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let expected_len = (w * h * bpp) as usize;
        let data_slice: std::borrow::Cow<'_, [u8]> = if pixels.len() >= expected_len {
            std::borrow::Cow::Borrowed(&pixels[..expected_len])
        } else {
            let mut padded = vec![0u8; expected_len];
            padded[..pixels.len()].copy_from_slice(pixels);
            std::borrow::Cow::Owned(padded)
        };

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data_slice,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * bpp),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        CachedGpuAtlas {
            _texture: texture,
            view,
            width: w,
            height: h,
        }
    }

    fn get_or_upload_atlas(&self, atlas: &GlyphAtlas) -> Arc<CachedGpuAtlas> {
        let key = AtlasCacheKey::from_atlas(atlas);
        if let Ok(cache) = self.atlas_cache.lock() {
            if let Some(existing) = cache.get(&key) {
                return Arc::clone(existing);
            }
        }

        let uploaded = Arc::new(Self::upload_atlas_raw(
            &self.device,
            &self.queue,
            atlas.atlas_type,
            key.width,
            key.height,
            &atlas.pixels,
        ));

        if let Ok(mut cache) = self.atlas_cache.lock() {
            cache.insert(key, Arc::clone(&uploaded));
        }
        uploaded
    }

    pub fn cached_atlas_count(&self) -> usize {
        self.atlas_cache
            .lock()
            .map(|cache| cache.len())
            .unwrap_or(0)
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn create_offscreen_target(&self, width: u32, height: u32) -> OffscreenTarget {
        OffscreenTarget::new(self, width, height)
    }

    pub fn create_offscreen_target_with_sample_count(
        &self,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> OffscreenTarget {
        OffscreenTarget::new_with_sample_count(self, width, height, sample_count)
    }

    pub fn cached_pipeline_count(&self) -> usize {
        self.pipeline_cache
            .lock()
            .map(|cache| cache.len())
            .unwrap_or(0)
    }

    pub fn get_or_create_pipeline(
        &self,
        kind: PipelineKind,
        blend_mode: BlendMode,
    ) -> Arc<wgpu::RenderPipeline> {
        self.get_or_create_pipeline_with_samples(kind, blend_mode, DEFAULT_SAMPLE_COUNT)
    }

    pub fn get_or_create_pipeline_with_samples(
        &self,
        kind: PipelineKind,
        blend_mode: BlendMode,
        sample_count: u32,
    ) -> Arc<wgpu::RenderPipeline> {
        let samples = normalize_sample_count(sample_count);
        let key = (kind, blend_mode as u32, samples);
        if let Ok(cache) = self.pipeline_cache.lock() {
            if let Some(existing) = cache.get(&key) {
                return Arc::clone(existing);
            }
        }

        let (shader, bgl, label) = match kind {
            PipelineKind::SolidFill => (
                &self.solid_shader,
                &self.uniform_pair_layout,
                "Impeller SolidFill Pipeline",
            ),
            PipelineKind::LinearGradientFill => (
                &self.linear_gradient_shader,
                &self.gradient_layout,
                "Impeller LinearGradientFill Pipeline",
            ),
            PipelineKind::RadialGradientFill => (
                &self.radial_gradient_shader,
                &self.gradient_layout,
                "Impeller RadialGradientFill Pipeline",
            ),
            PipelineKind::RRectBlur => (
                &self.rrect_blur_shader,
                &self.uniform_pair_layout,
                "Impeller RRectBlur Pipeline",
            ),
            PipelineKind::GlyphAtlas => (
                &self.glyph_atlas_shader,
                &self.glyph_atlas_layout,
                "Impeller GlyphAtlas Pipeline",
            ),
        };

        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[bgl],
                    push_constant_ranges: &[],
                });

        let solid_attrs = [wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        }];
        let glyph_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 32,
                shader_location: 3,
            },
        ];

        let vertex_layout = if kind == PipelineKind::GlyphAtlas {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GlyphVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &glyph_attrs,
            }
        } else {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SolidVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &solid_attrs,
            }
        };

        let blend_state = blend_state_for_mode(blend_mode);

        let pipeline = Arc::new(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: TARGET_FORMAT,
                        blend: Some(blend_state),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            },
        ));

        if let Ok(mut cache) = self.pipeline_cache.lock() {
            cache.insert(key, Arc::clone(&pipeline));
        }
        pipeline
    }

    fn compute_scissor_rect(
        clip_rect: Option<Rect>,
        internal_width: u32,
        internal_height: u32,
        ssaa_scale: u32,
    ) -> Option<[u32; 4]> {
        match clip_rect {
            None => Some([0, 0, internal_width, internal_height]),
            Some(r) => {
                if r.is_empty() {
                    return None;
                }
                let s = ssaa_scale.max(1) as Scalar;
                let min_x = (r.get_left() * s).floor().max(0.0) as u32;
                let min_y = (r.get_top() * s).floor().max(0.0) as u32;
                let max_x = (r.get_right() * s)
                    .ceil()
                    .max(0.0)
                    .min(internal_width as Scalar) as u32;
                let max_y = (r.get_bottom() * s)
                    .ceil()
                    .max(0.0)
                    .min(internal_height as Scalar) as u32;
                if max_x <= min_x || max_y <= min_y {
                    None
                } else {
                    Some([min_x, min_y, max_x - min_x, max_y - min_y])
                }
            }
        }
    }

    fn prepare_draw_call(
        &self,
        cmd: &DrawCommand,
        ortho: Matrix,
        internal_width: u32,
        internal_height: u32,
        ssaa_scale: u32,
        sample_count: u32,
    ) -> Option<PreparedDrawCall> {
        let scissor = Self::compute_scissor_rect(
            cmd.clip_rect(),
            internal_width,
            internal_height,
            ssaa_scale,
        )?;

        match cmd {
            DrawCommand::SolidFill {
                mesh,
                transform,
                color,
                blend_mode,
                ..
            } => {
                if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                    return None;
                }
                let pipeline = self.get_or_create_pipeline_with_samples(
                    PipelineKind::SolidFill,
                    *blend_mode,
                    sample_count,
                );
                let mvp = ortho * *transform;
                let frame_uniform = SolidFrameUniform { mvp: mvp.m };
                let frag_uniform = SolidFragUniform {
                    color: color.to_array_premultiplied(),
                };

                let frame_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("SolidFill FrameInfo"),
                        contents: bytemuck::bytes_of(&frame_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let frag_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("SolidFill FragInfo"),
                        contents: bytemuck::bytes_of(&frag_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SolidFill BindGroup"),
                    layout: &self.uniform_pair_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: frame_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: frag_buf.as_entire_binding(),
                        },
                    ],
                });

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("SolidFill VertexBuffer"),
                            contents: bytemuck::cast_slice(&mesh.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("SolidFill IndexBuffer"),
                            contents: bytemuck::cast_slice(&mesh.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                Some(PreparedDrawCall {
                    pipeline,
                    bind_group,
                    vertex_buffer,
                    index_buffer,
                    index_count: mesh.indices.len() as u32,
                    scissor,
                })
            }
            DrawCommand::LinearGradientFill {
                mesh,
                transform,
                gradient,
                blend_mode,
                ..
            } => {
                if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                    return None;
                }
                let pipeline = self.get_or_create_pipeline_with_samples(
                    PipelineKind::LinearGradientFill,
                    *blend_mode,
                    sample_count,
                );
                let mvp = ortho * *transform;
                let frame_uniform = GradientFrameUniform {
                    mvp: mvp.m,
                    matrix: Matrix::IDENTITY.m,
                };

                let start_to_end = gradient.end_point - gradient.start_point;
                let dot_se = start_to_end.x * start_to_end.x + start_to_end.y * start_to_end.y;
                let inv_dot = if dot_se <= 1.0e-8 { 0.0 } else { 1.0 / dot_se };
                let (gpu_stops, is_fast_two_stop, fast_color_0, fast_color_1) =
                    build_gpu_color_points(&gradient.stops);

                let frag_uniform = LinearGradientFragUniform {
                    start_point: [gradient.start_point.x, gradient.start_point.y],
                    end_point: [gradient.end_point.x, gradient.end_point.y],
                    start_to_end: [start_to_end.x, start_to_end.y],
                    alpha: gradient.alpha.clamp(0.0, 1.0),
                    inverse_dot_start_to_end: inv_dot,
                    decal_border_color: [0.0; 4],
                    fast_color_0,
                    fast_color_1,
                    tile_mode: gradient.tile_mode as u32,
                    colors_length: if gradient.stops.is_empty() {
                        0
                    } else {
                        gpu_stops.len() as u32
                    },
                    is_fast_two_stop: u32::from(is_fast_two_stop),
                    _pad: 0,
                };

                let frame_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("LinearGradient FrameInfo"),
                        contents: bytemuck::bytes_of(&frame_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let frag_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("LinearGradient FragInfo"),
                        contents: bytemuck::bytes_of(&frag_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let stops_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("LinearGradient Stops SSBO"),
                        contents: bytemuck::cast_slice(&gpu_stops),
                        usage: wgpu::BufferUsages::STORAGE,
                    });

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("LinearGradient BindGroup"),
                    layout: &self.gradient_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: frame_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: frag_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: stops_buf.as_entire_binding(),
                        },
                    ],
                });

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("LinearGradient VertexBuffer"),
                            contents: bytemuck::cast_slice(&mesh.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("LinearGradient IndexBuffer"),
                            contents: bytemuck::cast_slice(&mesh.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                Some(PreparedDrawCall {
                    pipeline,
                    bind_group,
                    vertex_buffer,
                    index_buffer,
                    index_count: mesh.indices.len() as u32,
                    scissor,
                })
            }
            DrawCommand::RadialGradientFill {
                mesh,
                transform,
                gradient,
                blend_mode,
                ..
            } => {
                if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                    return None;
                }
                let pipeline = self.get_or_create_pipeline_with_samples(
                    PipelineKind::RadialGradientFill,
                    *blend_mode,
                    sample_count,
                );
                let mvp = ortho * *transform;
                let frame_uniform = GradientFrameUniform {
                    mvp: mvp.m,
                    matrix: Matrix::IDENTITY.m,
                };

                let (gpu_stops, is_fast_two_stop, fast_color_0, fast_color_1) =
                    build_gpu_color_points(&gradient.stops);

                let frag_uniform = RadialGradientFragUniform {
                    center: [gradient.center.x, gradient.center.y],
                    radius: gradient.radius.max(K_EH_CLOSE_ENOUGH),
                    alpha: gradient.alpha.clamp(0.0, 1.0),
                    decal_border_color: [0.0; 4],
                    fast_color_0,
                    fast_color_1,
                    tile_mode: gradient.tile_mode as u32,
                    colors_length: if gradient.stops.is_empty() {
                        0
                    } else {
                        gpu_stops.len() as u32
                    },
                    is_fast_two_stop: u32::from(is_fast_two_stop),
                    _pad: 0,
                };

                let frame_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("RadialGradient FrameInfo"),
                        contents: bytemuck::bytes_of(&frame_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let frag_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("RadialGradient FragInfo"),
                        contents: bytemuck::bytes_of(&frag_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let stops_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("RadialGradient Stops SSBO"),
                        contents: bytemuck::cast_slice(&gpu_stops),
                        usage: wgpu::BufferUsages::STORAGE,
                    });

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RadialGradient BindGroup"),
                    layout: &self.gradient_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: frame_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: frag_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: stops_buf.as_entire_binding(),
                        },
                    ],
                });

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("RadialGradient VertexBuffer"),
                            contents: bytemuck::cast_slice(&mesh.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("RadialGradient IndexBuffer"),
                            contents: bytemuck::cast_slice(&mesh.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                Some(PreparedDrawCall {
                    pipeline,
                    bind_group,
                    vertex_buffer,
                    index_buffer,
                    index_count: mesh.indices.len() as u32,
                    scissor,
                })
            }
            DrawCommand::RRectBlur {
                transform,
                params,
                blend_mode,
                ..
            } => {
                if params.rect.is_empty() {
                    return None;
                }
                let pipeline = self.get_or_create_pipeline_with_samples(
                    PipelineKind::RRectBlur,
                    *blend_mode,
                    sample_count,
                );

                let pos_left = params.rect.get_left();
                let pos_top = params.rect.get_top();
                let width = (params.rect.get_right() - pos_left).abs();
                let height = (params.rect.get_bottom() - pos_top).abs();
                let blur_sigma = params.blur_sigma.clamp(K_EH_CLOSE_ENOUGH, 500.0);
                let blur_radius = RRectBlurParams::pad_for_sigma(blur_sigma);

                let left = -blur_radius;
                let top = -blur_radius;
                let right = width + blur_radius;
                let bottom = height + blur_radius;

                let quad_vertices = [
                    SolidVertex::new(left, top),
                    SolidVertex::new(right, top),
                    SolidVertex::new(left, bottom),
                    SolidVertex::new(right, bottom),
                ];
                let quad_indices: [u32; 6] = [0, 1, 2, 2, 1, 3];

                let local_origin_tx = Matrix::make_translation(pos_left, pos_top, 0.0);
                let mvp = ortho * (*transform * local_origin_tx);
                let frame_uniform = SolidFrameUniform { mvp: mvp.m };
                let frag_uniform = build_rrect_blur_frag_uniform(params, *blend_mode);

                let frame_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("RRectBlur FrameInfo"),
                        contents: bytemuck::bytes_of(&frame_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let frag_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("RRectBlur FragInfo"),
                        contents: bytemuck::bytes_of(&frag_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RRectBlur BindGroup"),
                    layout: &self.uniform_pair_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: frame_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: frag_buf.as_entire_binding(),
                        },
                    ],
                });

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("RRectBlur VertexBuffer"),
                            contents: bytemuck::cast_slice(&quad_vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("RRectBlur IndexBuffer"),
                            contents: bytemuck::cast_slice(&quad_indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                Some(PreparedDrawCall {
                    pipeline,
                    bind_group,
                    vertex_buffer,
                    index_buffer,
                    index_count: 6,
                    scissor,
                })
            }
            DrawCommand::GlyphAtlas(batch) => {
                if batch.quads.is_empty() {
                    return None;
                }
                let pipeline = self.get_or_create_pipeline_with_samples(
                    PipelineKind::GlyphAtlas,
                    batch.blend_mode,
                    sample_count,
                );

                let alpha_gpu = batch
                    .alpha_atlas
                    .as_deref()
                    .map(|a| self.get_or_upload_atlas(a))
                    .unwrap_or_else(|| Arc::clone(&self.fallback_alpha_atlas));
                let color_gpu = batch
                    .color_atlas
                    .as_deref()
                    .map(|c| self.get_or_upload_atlas(c))
                    .unwrap_or_else(|| Arc::clone(&self.fallback_color_atlas));

                let alpha_w = alpha_gpu.width.max(1) as Scalar;
                let alpha_h = alpha_gpu.height.max(1) as Scalar;
                let color_w = color_gpu.width.max(1) as Scalar;
                let color_h = color_gpu.height.max(1) as Scalar;

                let mut vertices = Vec::with_capacity(batch.quads.len() * 4);
                let mut indices = Vec::with_capacity(batch.quads.len() * 6);

                for quad in &batch.quads {
                    if quad.dest_rect.is_empty() || quad.atlas_rect.is_empty() {
                        continue;
                    }
                    let (inv_w, inv_h) = if quad.is_color {
                        (1.0 / color_w, 1.0 / color_h)
                    } else {
                        (1.0 / alpha_w, 1.0 / alpha_h)
                    };

                    let l = quad.dest_rect.get_left();
                    let t = quad.dest_rect.get_top();
                    let r = quad.dest_rect.get_right();
                    let b = quad.dest_rect.get_bottom();

                    let u0 = quad.atlas_rect.get_left() * inv_w;
                    let v0 = quad.atlas_rect.get_top() * inv_h;
                    let u1 = quad.atlas_rect.get_right() * inv_w;
                    let v1 = quad.atlas_rect.get_bottom() * inv_h;

                    let pm_color = quad.color.to_array_premultiplied();
                    let is_color_u32 = u32::from(quad.is_color);
                    let base = vertices.len() as u32;

                    vertices.push(GlyphVertex::new([l, t], [u0, v0], pm_color, is_color_u32));
                    vertices.push(GlyphVertex::new([r, t], [u1, v0], pm_color, is_color_u32));
                    vertices.push(GlyphVertex::new([r, b], [u1, v1], pm_color, is_color_u32));
                    vertices.push(GlyphVertex::new([l, b], [u0, v1], pm_color, is_color_u32));

                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }

                if vertices.is_empty() || indices.is_empty() {
                    return None;
                }

                let mvp = ortho * batch.transform;
                let frame_uniform = SolidFrameUniform { mvp: mvp.m };
                let frag_uniform = GlyphAtlasFragUniform {
                    use_text_color: if batch.use_text_color { 1.0 } else { 0.0 },
                    text_contrast: batch.text_contrast.max(0.1),
                    alpha: batch.alpha.clamp(0.0, 1.0),
                    _pad: 0.0,
                };

                let frame_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("GlyphAtlas FrameInfo"),
                        contents: bytemuck::bytes_of(&frame_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let frag_buf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("GlyphAtlas FragInfo"),
                        contents: bytemuck::bytes_of(&frag_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                let use_linear = batch
                    .linear_filter
                    .unwrap_or_else(|| !batch.transform.is_translation_scale_only());
                let sampler = if use_linear {
                    &self.linear_sampler
                } else {
                    &self.nearest_sampler
                };

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GlyphAtlas BindGroup"),
                    layout: &self.glyph_atlas_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: frame_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: frag_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&alpha_gpu.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&color_gpu.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                });

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("GlyphAtlas VertexBuffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("GlyphAtlas IndexBuffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                Some(PreparedDrawCall {
                    pipeline,
                    bind_group,
                    vertex_buffer,
                    index_buffer,
                    index_count: indices.len() as u32,
                    scissor,
                })
            }
        }
    }

    fn encode_and_submit(
        &self,
        target: &OffscreenTarget,
        clear_color: Color,
        commands: &[DrawCommand],
    ) -> Result<(), String> {
        let ssaa = target.ssaa_scale.max(1) as Scalar;
        let ortho = Matrix::make_orthographic(Size::new(
            target.internal_width as Scalar,
            target.internal_height as Scalar,
        )) * Matrix::make_scale(ssaa, ssaa, 1.0);
        let prepared: Vec<PreparedDrawCall> = commands
            .iter()
            .filter_map(|cmd| {
                self.prepare_draw_call(
                    cmd,
                    ortho,
                    target.internal_width,
                    target.internal_height,
                    target.ssaa_scale,
                    target.sample_count,
                )
            })
            .collect();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Impeller Offscreen CommandEncoder"),
            });

        let pm_clear = clear_color.premultiply();
        let (attachment_view, resolve_target, store_op) = match &target.msaa_view {
            Some(msaa_view) => (msaa_view, Some(&target.view), wgpu::StoreOp::Discard),
            None => (&target.view, None, wgpu::StoreOp::Store),
        };
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Impeller Offscreen RenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: attachment_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(pm_clear.red.clamp(0.0, 1.0)),
                            g: f64::from(pm_clear.green.clamp(0.0, 1.0)),
                            b: f64::from(pm_clear.blue.clamp(0.0, 1.0)),
                            a: f64::from(pm_clear.alpha.clamp(0.0, 1.0)),
                        }),
                        store: store_op,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            for call in &prepared {
                render_pass.set_pipeline(&call.pipeline);
                render_pass.set_bind_group(0, &call.bind_group, &[]);
                render_pass.set_scissor_rect(
                    call.scissor[0],
                    call.scissor[1],
                    call.scissor[2],
                    call.scissor[3],
                );
                render_pass.set_vertex_buffer(0, call.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(call.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..call.index_count, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn enqueue_readback_buffer(&self, target: &OffscreenTarget) -> (wgpu::Buffer, u32, u32) {
        let unpadded_bytes_per_row = target.internal_width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = u64::from(padded_bytes_per_row) * u64::from(target.internal_height);

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Impeller Readback Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Impeller Readback CommandEncoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(target.internal_height),
                },
            },
            wgpu::Extent3d {
                width: target.internal_width,
                height: target.internal_height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        (readback_buffer, unpadded_bytes_per_row, padded_bytes_per_row)
    }

    fn extract_unpadded_rows(
        readback_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        ssaa_scale: u32,
        unpadded_bytes_per_row: u32,
        padded_bytes_per_row: u32,
    ) -> Vec<u8> {
        let buffer_slice = readback_buffer.slice(..);
        let mapped = buffer_slice.get_mapped_range();
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        if ssaa_scale == 2 {
            for y in 0..height {
                let row0 = ((y * 2) * padded_bytes_per_row) as usize;
                let row1 = ((y * 2 + 1) * padded_bytes_per_row) as usize;
                for x in 0..width {
                    let col0 = (x * 2 * 4) as usize;
                    let col1 = col0 + 4;
                    for c in 0..4 {
                        let p00 = u16::from(mapped[row0 + col0 + c]);
                        let p10 = u16::from(mapped[row0 + col1 + c]);
                        let p01 = u16::from(mapped[row1 + col0 + c]);
                        let p11 = u16::from(mapped[row1 + col1 + c]);
                        rgba.push(((p00 + p10 + p01 + p11 + 2) / 4) as u8);
                    }
                }
            }
        } else {
            for row in 0..height {
                let start = (row * padded_bytes_per_row) as usize;
                let end = start + unpadded_bytes_per_row as usize;
                rgba.extend_from_slice(&mapped[start..end]);
            }
        }
        drop(mapped);
        readback_buffer.unmap();
        rgba
    }

    fn readback_target(&self, target: &OffscreenTarget) -> Result<Vec<u8>, String> {
        let (readback_buffer, unpadded_bytes_per_row, padded_bytes_per_row) =
            self.enqueue_readback_buffer(target);

        let buffer_slice = readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("Failed waiting for buffer map: {e}"))?
            .map_err(|e| format!("Failed mapping readback buffer: {e}"))?;

        Ok(Self::extract_unpadded_rows(
            &readback_buffer,
            target.width,
            target.height,
            target.ssaa_scale,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        ))
    }

    async fn readback_target_async(&self, target: &OffscreenTarget) -> Result<Vec<u8>, String> {
        let (readback_buffer, unpadded_bytes_per_row, padded_bytes_per_row) =
            self.enqueue_readback_buffer(target);

        let buffer_slice = readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("Failed waiting for buffer map: {e}"))?
            .map_err(|e| format!("Failed mapping readback buffer: {e}"))?;

        Ok(Self::extract_unpadded_rows(
            &readback_buffer,
            target.width,
            target.height,
            target.ssaa_scale,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        ))
    }

    pub fn render_to_rgba(
        &self,
        width: u32,
        height: u32,
        clear_color: Color,
        commands: &[DrawCommand],
    ) -> Result<Vec<u8>, String> {
        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }
        let target = OffscreenTarget::new(self, width, height);
        target.render_and_readback(self, clear_color, commands)
    }

    pub fn render_to_png(
        &self,
        path: &FsPath,
        width: u32,
        height: u32,
        clear_color: Color,
        commands: &[DrawCommand],
    ) -> Result<Vec<u8>, String> {
        let rgba = self.render_to_rgba(width, height, clear_color, commands)?;
        save_rgba_png(path, width, height, &rgba)?;
        Ok(rgba)
    }
}

pub fn save_rgba_png(path: &FsPath, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    image::save_buffer(path, rgba, width, height, image::ColorType::Rgba8)
        .map_err(|e| e.to_string())
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GoldenDiffMetrics {
    pub width: u32,
    pub height: u32,
    pub rmse: f64,
    pub max_channel_diff: u8,
    pub mismatched_pixels: usize,
    pub mismatch_fraction: f64,
}

pub fn compare_golden_rgba(
    actual: &[u8],
    expected: &[u8],
    width: u32,
    height: u32,
    tolerance: u8,
) -> Result<(GoldenDiffMetrics, Vec<u8>), String> {
    if width == 0 || height == 0 {
        return Err("compare_golden_rgba requires non-zero dimensions".to_string());
    }
    let total_pixels = (width as usize) * (height as usize);
    let expected_len = total_pixels * 4;
    if actual.len() != expected_len || expected.len() != expected_len {
        return Err(format!(
            "RGBA buffer length mismatch: actual={}, expected={}, required={expected_len}",
            actual.len(),
            expected.len()
        ));
    }

    let triptych_width = (width as usize) * 3;
    let mut triptych = vec![0u8; triptych_width * (height as usize) * 4];

    let mut sum_sq_diff: u64 = 0;
    let mut max_channel_diff: u8 = 0;
    let mut mismatched_pixels: usize = 0;

    for y in 0..(height as usize) {
        for x in 0..(width as usize) {
            let src_idx = (y * (width as usize) + x) * 4;
            let a = &actual[src_idx..src_idx + 4];
            let e = &expected[src_idx..src_idx + 4];

            let mut pixel_max_diff: u8 = 0;
            for c in 0..4 {
                let d = a[c].abs_diff(e[c]);
                sum_sq_diff += u64::from(d) * u64::from(d);
                if d > max_channel_diff {
                    max_channel_diff = d;
                }
                if d > pixel_max_diff {
                    pixel_max_diff = d;
                }
            }
            if pixel_max_diff > tolerance {
                mismatched_pixels += 1;
            }

            let left_idx = (y * triptych_width + x) * 4;
            triptych[left_idx..left_idx + 4].copy_from_slice(e);

            let mid_idx = (y * triptych_width + (width as usize) + x) * 4;
            triptych[mid_idx..mid_idx + 4].copy_from_slice(a);

            let right_idx = (y * triptych_width + (width as usize) * 2 + x) * 4;
            let heat = if pixel_max_diff == 0 {
                [e[0] / 6, e[1] / 6, e[2] / 6, 255]
            } else if pixel_max_diff <= tolerance {
                let amp = (u16::from(pixel_max_diff) * 8).min(255) as u8;
                [0, amp, amp.saturating_add(48), 255]
            } else {
                let r_heat = (u16::from(pixel_max_diff) * 8).clamp(128, 255) as u8;
                let g_heat = (u16::from(a[1].abs_diff(e[1])) * 4).min(180) as u8;
                [r_heat, g_heat, 0, 255]
            };
            triptych[right_idx..right_idx + 4].copy_from_slice(&heat);
        }
    }

    let rmse = ((sum_sq_diff as f64) / ((expected_len as f64).max(1.0))).sqrt();
    let mismatch_fraction = (mismatched_pixels as f64) / ((total_pixels as f64).max(1.0));

    Ok((
        GoldenDiffMetrics {
            width,
            height,
            rmse,
            max_channel_diff,
            mismatched_pixels,
            mismatch_fraction,
        },
        triptych,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller_geometry::{Degrees, Radians};

    fn make_rect_mesh(rect: Rect) -> TessellatedMesh {
        let l = rect.get_left();
        let t = rect.get_top();
        let r = rect.get_right();
        let b = rect.get_bottom();
        TessellatedMesh {
            vertices: vec![
                Vertex2D::new(l, t),
                Vertex2D::new(r, t),
                Vertex2D::new(r, b),
                Vertex2D::new(l, b),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    fn pixel_at(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let idx = ((y * width + x) * 4) as usize;
        [rgba[idx], rgba[idx + 1], rgba[idx + 2], rgba[idx + 3]]
    }

    fn assert_pixel_approx(actual: [u8; 4], expected: [u8; 4], tol: u8) {
        for c in 0..4 {
            let diff = actual[c].abs_diff(expected[c]);
            assert!(
                diff <= tol,
                "pixel channel {c} mismatch: actual={actual:?}, expected={expected:?} (tol={tol})"
            );
        }
    }

    /// C++ source: renderer_unittests.cc:CanCreateBoxPrimitive
    #[test]
    fn test_can_create_box_primitive_solid_fill() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        assert!(!ctx.adapter_name.is_empty());

        let cmd = DrawCommand::from(SolidFillBatch {
            mesh: make_rect_mesh(Rect::from_xywh(10.0, 10.0, 44.0, 44.0)),
            transform: Matrix::IDENTITY,
            color: Color::RED,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        });

        let rgba = ctx
            .render_to_rgba(64, 64, Color::BLACK, &[cmd])
            .expect("gpu render");
        assert_eq!(rgba.len(), 64 * 64 * 4);
        assert_pixel_approx(pixel_at(&rgba, 64, 2, 2), [0, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 32, 32), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 60, 60), [0, 0, 0, 255], 1);
    }

    /// C++ source: renderer_unittests.cc:CanRenderMultiplePrimitives
    #[test]
    fn test_can_render_multiple_primitives() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let cmds = vec![
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(4.0, 4.0, 24.0, 24.0)),
                transform: Matrix::IDENTITY,
                color: Color::RED,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(36.0, 4.0, 24.0, 24.0)),
                transform: Matrix::IDENTITY,
                color: Color::GREEN,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(20.0, 36.0, 24.0, 24.0)),
                transform: Matrix::IDENTITY,
                color: Color::BLUE,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
        ];

        let rgba = ctx
            .render_to_rgba(64, 64, Color::WHITE, &cmds)
            .expect("gpu render");
        assert_pixel_approx(pixel_at(&rgba, 64, 16, 16), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 48, 16), [0, 255, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 32, 48), [0, 0, 255, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 2, 60), [255, 255, 255, 255], 1);
    }

    /// C++ source: renderer_unittests.cc:CanRenderToTexture
    #[test]
    fn test_can_render_to_offscreen_texture_unaligned_width() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = ctx.create_offscreen_target(53, 37);
        assert_eq!(target.width(), 53);
        assert_eq!(target.height(), 37);

        let cmd = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(10.0, 5.0, 30.0, 25.0)),
            transform: Matrix::IDENTITY,
            color: Color::new(0.2, 0.6, 1.0, 1.0),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let rgba = target
            .render_and_readback(&ctx, Color::BLACK, &[cmd])
            .expect("offscreen render and readback");
        assert_eq!(rgba.len(), 53 * 37 * 4);
        assert_pixel_approx(pixel_at(&rgba, 53, 2, 2), [0, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 53, 25, 18), [51, 153, 255, 255], 2);
    }

    /// C++ source: renderer_unittests.cc:CanBlitTextureToBuffer
    #[test]
    fn test_can_blit_texture_to_buffer_and_save_png() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let cmd = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 16.0, 16.0)),
            transform: Matrix::IDENTITY,
            color: Color::GREEN,
            blend_mode: BlendMode::Src,
            clip_rect: None,
        };
        let tmp_png = std::env::temp_dir().join("impeller_wgpu_test_blit.png");
        let rgba = ctx
            .render_to_png(&tmp_png, 32, 32, Color::RED, &[cmd])
            .expect("render_to_png");
        assert_pixel_approx(pixel_at(&rgba, 32, 8, 8), [0, 255, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 32, 24, 24), [255, 0, 0, 255], 1);
        let decoded = image::open(&tmp_png).expect("open saved png").to_rgba8();
        assert_eq!(decoded.dimensions(), (32, 32));
        assert_eq!(decoded.as_raw(), &rgba);
        let _ = std::fs::remove_file(&tmp_png);
    }

    /// C++ source: renderer_unittests.cc:PipelineLibraryAllowsConcurrentGetPipeline
    #[test]
    fn test_pipeline_library_allows_concurrent_get_pipeline() {
        let ctx = Arc::new(WgpuContext::new_headless().expect("headless wgpu context"));
        let modes = [
            BlendMode::Clear,
            BlendMode::Src,
            BlendMode::SrcOver,
            BlendMode::DstOver,
            BlendMode::SrcIn,
            BlendMode::Plus,
            BlendMode::Modulate,
            BlendMode::Screen,
            BlendMode::Multiply,
        ];
        let kinds = [
            PipelineKind::SolidFill,
            PipelineKind::LinearGradientFill,
            PipelineKind::RadialGradientFill,
            PipelineKind::RRectBlur,
            PipelineKind::GlyphAtlas,
        ];

        std::thread::scope(|s| {
            for kind in kinds {
                let ctx_clone = Arc::clone(&ctx);
                s.spawn(move || {
                    for mode in modes {
                        let _p = ctx_clone.get_or_create_pipeline(kind, mode);
                    }
                });
            }
        });

        assert_eq!(ctx.cached_pipeline_count(), kinds.len() * modes.len());
    }

    /// C++ source: entity_unittests.cc:CanDrawCorrectlyWithRotatedTransform
    #[test]
    fn test_can_draw_correctly_with_rotated_transform() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let transform = Matrix::make_translation(32.0, 32.0, 0.0)
            * Matrix::make_rotation_z(Radians::from(Degrees(90.0)));
        let cmd = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 20.0, 10.0)),
            transform,
            color: Color::RED,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let rgba = ctx
            .render_to_rgba(64, 64, Color::BLACK, &[cmd])
            .expect("render rotated");
        assert_pixel_approx(pixel_at(&rgba, 64, 27, 42), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 42, 35), [0, 0, 0, 255], 1);
    }

    /// C++ source: entity_unittests.cc:BlendingModeOptions
    #[test]
    fn test_blending_mode_source_over_and_source() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let src_over_cmd = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 32.0, 32.0)),
            transform: Matrix::IDENTITY,
            color: Color::new(1.0, 0.0, 0.0, 0.5),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let src_cmd = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(32.0, 0.0, 32.0, 32.0)),
            transform: Matrix::IDENTITY,
            color: Color::new(1.0, 0.0, 0.0, 0.5),
            blend_mode: BlendMode::Src,
            clip_rect: None,
        };

        let rgba = ctx
            .render_to_rgba(64, 32, Color::BLUE, &[src_over_cmd, src_cmd])
            .expect("render blend modes");
        assert_pixel_approx(pixel_at(&rgba, 64, 16, 16), [128, 0, 128, 255], 2);
        assert_pixel_approx(pixel_at(&rgba, 64, 48, 16), [128, 0, 0, 128], 2);
    }

    /// C++ source: entity_unittests.cc:ColorFilterWithForegroundColorClearBlend
    #[test]
    fn test_blending_mode_clear() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let clear_hole = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(16.0, 16.0, 32.0, 32.0)),
            transform: Matrix::IDENTITY,
            color: Color::RED,
            blend_mode: BlendMode::Clear,
            clip_rect: None,
        };

        let rgba = ctx
            .render_to_rgba(64, 64, Color::GREEN, &[clear_hole])
            .expect("render clear blend");
        assert_pixel_approx(pixel_at(&rgba, 64, 32, 32), [0, 0, 0, 0], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 4, 4), [0, 255, 0, 255], 1);
    }

    /// C++ source: entity_unittests.cc:ColorFilterWithForegroundColorDstBlend
    #[test]
    fn test_blending_mode_dst_and_dst_over() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let dst_cmd = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 32.0, 32.0)),
            transform: Matrix::IDENTITY,
            color: Color::RED,
            blend_mode: BlendMode::Dst,
            clip_rect: None,
        };
        let dst_over_cmd = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(32.0, 0.0, 32.0, 32.0)),
            transform: Matrix::IDENTITY,
            color: Color::RED,
            blend_mode: BlendMode::DstOver,
            clip_rect: None,
        };

        let rgba = ctx
            .render_to_rgba(
                64,
                32,
                Color::new(0.0, 1.0, 0.0, 0.5),
                &[dst_cmd, dst_over_cmd],
            )
            .expect("render dst and dst_over");
        assert_pixel_approx(pixel_at(&rgba, 64, 16, 16), [0, 128, 0, 128], 2);
        assert_pixel_approx(pixel_at(&rgba, 64, 48, 16), [128, 128, 0, 255], 2);
    }

    /// C++ source: entity_unittests.cc:ColorFilterWithForegroundColorSrcInBlend
    #[test]
    fn test_blending_mode_porter_duff_in_out_atop_xor() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let cmds = vec![
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 32.0, 32.0)),
                transform: Matrix::IDENTITY,
                color: Color::RED,
                blend_mode: BlendMode::SrcIn,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(32.0, 0.0, 32.0, 32.0)),
                transform: Matrix::IDENTITY,
                color: Color::RED,
                blend_mode: BlendMode::SrcOut,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(0.0, 32.0, 32.0, 32.0)),
                transform: Matrix::IDENTITY,
                color: Color::new(1.0, 0.0, 0.0, 0.5),
                blend_mode: BlendMode::DstOut,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(32.0, 32.0, 32.0, 32.0)),
                transform: Matrix::IDENTITY,
                color: Color::new(1.0, 0.0, 0.0, 0.5),
                blend_mode: BlendMode::Xor,
                clip_rect: None,
            },
        ];

        let rgba = ctx
            .render_to_rgba(64, 64, Color::new(0.0, 0.0, 1.0, 0.5), &cmds)
            .expect("render porter-duff modes");
        assert_pixel_approx(pixel_at(&rgba, 64, 16, 16), [128, 0, 0, 128], 2);
        assert_pixel_approx(pixel_at(&rgba, 64, 48, 16), [128, 0, 0, 128], 2);
        assert_pixel_approx(pixel_at(&rgba, 64, 16, 48), [0, 0, 64, 64], 2);
        assert_pixel_approx(pixel_at(&rgba, 64, 48, 48), [64, 0, 64, 128], 2);
    }

    /// C++ source: entity_unittests.cc:SetBlendMode
    #[test]
    fn test_blending_mode_plus_modulate_screen_multiply() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let src_color = Color::new(0.5, 0.4, 0.5, 1.0);
        let cmds = vec![
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 32.0, 32.0)),
                transform: Matrix::IDENTITY,
                color: src_color,
                blend_mode: BlendMode::Plus,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(32.0, 0.0, 32.0, 32.0)),
                transform: Matrix::IDENTITY,
                color: src_color,
                blend_mode: BlendMode::Modulate,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(0.0, 32.0, 32.0, 32.0)),
                transform: Matrix::IDENTITY,
                color: src_color,
                blend_mode: BlendMode::Screen,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(32.0, 32.0, 32.0, 32.0)),
                transform: Matrix::IDENTITY,
                color: src_color,
                blend_mode: BlendMode::Multiply,
                clip_rect: None,
            },
        ];

        let rgba = ctx
            .render_to_rgba(64, 64, Color::new(0.4, 0.5, 0.8, 1.0), &cmds)
            .expect("render separable blend modes");
        assert_pixel_approx(pixel_at(&rgba, 64, 16, 16), [230, 230, 255, 255], 3);
        assert_pixel_approx(pixel_at(&rgba, 64, 48, 16), [51, 51, 102, 255], 3);
        assert_pixel_approx(pixel_at(&rgba, 64, 16, 48), [179, 179, 230, 255], 3);
        assert_pixel_approx(pixel_at(&rgba, 64, 48, 48), [51, 51, 102, 255], 3);
    }

    /// C++ source: entity_unittests.cc:FilterCoverageRespectsCropRect
    #[test]
    fn test_scissor_clip_rect_restricts_rendering() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let clipped_cmd = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
            transform: Matrix::IDENTITY,
            color: Color::RED,
            blend_mode: BlendMode::SrcOver,
            clip_rect: Some(Rect::from_xywh(16.0, 16.0, 32.0, 32.0)),
        };

        let rgba = ctx
            .render_to_rgba(64, 64, Color::BLACK, &[clipped_cmd])
            .expect("render clipped");
        assert_pixel_approx(pixel_at(&rgba, 64, 8, 8), [0, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 15, 32), [0, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 16, 16), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 32, 32), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 47, 47), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 48, 32), [0, 0, 0, 255], 1);
    }

    /// C++ source: entity_unittests.cc:LinearGradientContentsIsOpaque
    #[test]
    fn test_linear_gradient_two_stop_clamp() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let grad = LinearGradientParams::two_stop(
            Point::new(16.0, 0.0),
            Point::new(48.0, 0.0),
            Color::RED,
            Color::BLUE,
            TileMode::Clamp,
        );
        let cmd = DrawCommand::LinearGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 32.0)),
            transform: Matrix::IDENTITY,
            gradient: grad,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let rgba = ctx
            .render_to_rgba(64, 32, Color::BLACK, &[cmd])
            .expect("render linear gradient clamp");
        assert_pixel_approx(pixel_at(&rgba, 64, 4, 16), [255, 0, 0, 255], 2);
        let mid = pixel_at(&rgba, 64, 32, 16);
        assert!(mid[0] > 110 && mid[0] < 145, "mid red={}", mid[0]);
        assert!(mid[2] > 110 && mid[2] < 145, "mid blue={}", mid[2]);
        assert_pixel_approx(pixel_at(&rgba, 64, 60, 16), [0, 0, 255, 255], 2);
    }

    /// C++ source: entity_unittests.cc:LinearGradientApplyColorFilter
    #[test]
    fn test_linear_gradient_multi_stop_ssbo_16_stops() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let mut stops = Vec::with_capacity(16);
        for i in 0..16 {
            let t = i as Scalar / 15.0;
            let color = Color::new(t, 1.0 - t, (t * 2.0 - 1.0).abs(), 1.0);
            stops.push(GradientStop::new(t, color));
        }

        let grad = LinearGradientParams::new(
            Point::new(0.0, 0.0),
            Point::new(128.0, 0.0),
            stops,
            TileMode::Clamp,
            1.0,
        );
        let cmd = DrawCommand::LinearGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 128.0, 16.0)),
            transform: Matrix::IDENTITY,
            gradient: grad,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let rgba = ctx
            .render_to_rgba(128, 16, Color::BLACK, &[cmd])
            .expect("render 16-stop linear gradient");
        let left = pixel_at(&rgba, 128, 1, 8);
        let center = pixel_at(&rgba, 128, 64, 8);
        let right = pixel_at(&rgba, 128, 126, 8);
        assert!(left[0] < 15 && left[1] > 240, "left={left:?}");
        assert!(
            center[0] > 110 && center[0] < 145 && center[2] < 25,
            "center={center:?}"
        );
        assert!(right[0] > 240 && right[1] < 15, "right={right:?}");
    }

    /// C++ source: entity_unittests.cc:TiledTextureContentsIsOpaque
    #[test]
    fn test_linear_gradient_tile_modes_repeat_mirror_decal() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let make_cmd = |tile_mode: TileMode, y_offset: Scalar| DrawCommand::LinearGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, y_offset, 96.0, 16.0)),
            transform: Matrix::IDENTITY,
            gradient: LinearGradientParams::two_stop(
                Point::new(16.0, y_offset),
                Point::new(48.0, y_offset),
                Color::RED,
                Color::GREEN,
                tile_mode,
            ),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let cmds = vec![
            make_cmd(TileMode::Repeat, 0.0),
            make_cmd(TileMode::Mirror, 16.0),
            make_cmd(TileMode::Decal, 32.0),
        ];

        let rgba = ctx
            .render_to_rgba(96, 48, Color::BLACK, &cmds)
            .expect("render linear tile modes");

        let rep_a = pixel_at(&rgba, 96, 24, 8);
        let rep_b = pixel_at(&rgba, 96, 56, 8);
        assert_pixel_approx(rep_a, rep_b, 3);

        let mir_a = pixel_at(&rgba, 96, 44, 24);
        let mir_b = pixel_at(&rgba, 96, 51, 24);
        assert_pixel_approx(mir_a, mir_b, 2);

        assert_pixel_approx(pixel_at(&rgba, 96, 4, 40), [0, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 96, 60, 40), [0, 0, 0, 255], 1);
        let dec_mid = pixel_at(&rgba, 96, 32, 40);
        assert!(dec_mid[0] > 110 && dec_mid[1] > 110, "dec_mid={dec_mid:?}");
    }

    /// C++ source: entity_unittests.cc:RadialGradientContentsIsOpaque
    #[test]
    fn test_radial_gradient_two_stop_and_multi_stop_clamp() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let grad = RadialGradientParams::new(
            Point::new(32.0, 32.0),
            24.0,
            vec![
                GradientStop::new(0.0, Color::RED),
                GradientStop::new(0.5, Color::GREEN),
                GradientStop::new(1.0, Color::BLUE),
            ],
            TileMode::Clamp,
            1.0,
        );
        let cmd = DrawCommand::RadialGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
            transform: Matrix::IDENTITY,
            gradient: grad,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let rgba = ctx
            .render_to_rgba(64, 64, Color::BLACK, &[cmd])
            .expect("render radial gradient");
        let center = pixel_at(&rgba, 64, 32, 32);
        assert!(center[0] > 235 && center[1] < 25, "center={center:?}");
        let mid = pixel_at(&rgba, 64, 44, 32);
        assert!(mid[1] > 220 && mid[0] < 35 && mid[2] < 35, "mid={mid:?}");
        assert_pixel_approx(pixel_at(&rgba, 64, 60, 60), [0, 0, 255, 255], 2);
    }

    /// C++ source: entity_unittests.cc:RadialGradientApplyColorFilter
    #[test]
    fn test_radial_gradient_tile_modes_repeat_mirror_decal() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let decal_cmd = DrawCommand::RadialGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
            transform: Matrix::IDENTITY,
            gradient: RadialGradientParams::two_stop(
                Point::new(32.0, 32.0),
                16.0,
                Color::WHITE,
                Color::RED,
                TileMode::Decal,
            ),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let rgba_decal = ctx
            .render_to_rgba(64, 64, Color::TRANSPARENT, &[decal_cmd])
            .expect("render radial decal");
        assert_eq!(pixel_at(&rgba_decal, 64, 32, 32)[3], 255);
        assert_pixel_approx(pixel_at(&rgba_decal, 64, 4, 4), [0, 0, 0, 0], 1);

        let mirror_cmd = DrawCommand::RadialGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
            transform: Matrix::IDENTITY,
            gradient: RadialGradientParams::two_stop(
                Point::new(32.0, 32.5),
                16.0,
                Color::RED,
                Color::BLUE,
                TileMode::Mirror,
            ),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let rgba_mirror = ctx
            .render_to_rgba(64, 64, Color::BLACK, &[mirror_cmd])
            .expect("render radial mirror");
        assert_pixel_approx(
            pixel_at(&rgba_mirror, 64, 36, 32),
            pixel_at(&rgba_mirror, 64, 59, 32),
            2,
        );
    }

    /// C++ source: entity_unittests.cc:RRectShadowTest
    #[test]
    fn test_rrect_blur_analytical_gaussian_shadow() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let params = RRectBlurParams::new(
            Rect::from_xywh(30.0, 30.0, 68.0, 68.0),
            12.0,
            6.0,
            Color::WHITE,
        );
        let cmd = DrawCommand::RRectBlur {
            transform: Matrix::IDENTITY,
            params,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let rgba = ctx
            .render_to_rgba(128, 128, Color::TRANSPARENT, &[cmd])
            .expect("render rrect blur");

        let center_a = pixel_at(&rgba, 128, 64, 64)[3];
        let edge_a = pixel_at(&rgba, 128, 30, 64)[3];
        let outside_near_a = pixel_at(&rgba, 128, 22, 64)[3];
        let far_corner_a = pixel_at(&rgba, 128, 2, 2)[3];

        assert!(center_a > 245, "center_a={center_a}");
        assert!(
            edge_a > 95 && edge_a < 160,
            "edge_a={edge_a} (expected ~128 at boundary)"
        );
        assert!(
            outside_near_a > 5 && outside_near_a < edge_a,
            "outside_near_a={outside_near_a}"
        );
        assert_eq!(far_corner_a, 0);

        let q1 = pixel_at(&rgba, 128, 34, 34)[3];
        let q2 = pixel_at(&rgba, 128, 93, 34)[3];
        let q3 = pixel_at(&rgba, 128, 34, 93)[3];
        let q4 = pixel_at(&rgba, 128, 93, 93)[3];
        assert!(q1.abs_diff(q2) <= 3);
        assert!(q1.abs_diff(q3) <= 3);
        assert!(q1.abs_diff(q4) <= 3);
    }

    /// C++ source: entity_unittests.cc:GaussianBlurFilter
    #[test]
    fn test_rrect_blur_sigma_variation_and_erf_accuracy() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let rect = Rect::from_xywh(32.0, 32.0, 64.0, 64.0);

        let tight = DrawCommand::RRectBlur {
            transform: Matrix::IDENTITY,
            params: RRectBlurParams::new(rect, 8.0, 2.0, Color::RED),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let wide = DrawCommand::RRectBlur {
            transform: Matrix::IDENTITY,
            params: RRectBlurParams::new(rect, 8.0, 12.0, Color::RED),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let rgba_tight = ctx
            .render_to_rgba(128, 128, Color::TRANSPARENT, &[tight])
            .expect("render tight blur");
        let rgba_wide = ctx
            .render_to_rgba(128, 128, Color::TRANSPARENT, &[wide])
            .expect("render wide blur");

        let a_tight = pixel_at(&rgba_tight, 128, 22, 64)[3];
        let a_wide = pixel_at(&rgba_wide, 128, 22, 64)[3];
        assert!(
            a_wide > a_tight + 30,
            "expected wider blur falloff at x=22: a_wide={a_wide}, a_tight={a_tight}"
        );

        assert!((compute_erf7(0.0)).abs() < 1.0e-6);
        assert!((compute_erf7(3.0) - 1.0).abs() < 1.0e-3);
        assert!((compute_erf7(-3.0) + 1.0).abs() < 1.0e-3);
        assert!((ip_gaussian_integral(0.0, 1.0) - 0.5).abs() < 1.0e-5);
    }

    /// C++ source: entity_unittests.cc:CanRenderEmptyPathsWithoutCrashing
    #[test]
    fn test_can_render_empty_meshes_and_degenerate_gradients_without_crashing() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let empty_mesh = TessellatedMesh::default();
        let cmds = vec![
            DrawCommand::SolidFill {
                mesh: empty_mesh,
                transform: Matrix::IDENTITY,
                color: Color::RED,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::LinearGradientFill {
                mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 16.0, 16.0)),
                transform: Matrix::IDENTITY,
                gradient: LinearGradientParams::new(
                    Point::new(5.0, 5.0),
                    Point::new(5.0, 5.0),
                    vec![],
                    TileMode::Clamp,
                    1.0,
                ),
                blend_mode: BlendMode::SrcOver,
                clip_rect: Some(Rect::from_xywh(100.0, 100.0, 10.0, 10.0)),
            },
            DrawCommand::RRectBlur {
                transform: Matrix::IDENTITY,
                params: RRectBlurParams::new(Rect::ZERO, 0.0, 0.0, Color::RED),
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
        ];

        let rgba = ctx
            .render_to_rgba(32, 32, Color::BLUE, &cmds)
            .expect("render degenerate commands");
        assert_pixel_approx(pixel_at(&rgba, 32, 16, 16), [0, 0, 255, 255], 1);
    }

    /// C++ source: entity_unittests.cc:TextContentsCeilsGlyphScaleToDecimal
    #[test]
    fn test_glyph_atlas_r8_unorm_alpha_mask_tinting() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 32, 32);
        // Fill [4..20, 4..20] with 255 (opaque mask), and [20..28, 4..20] with 128 (50% mask)
        for y in 4..20 {
            for x in 4..20 {
                atlas.pixels[y * 32 + x] = 255;
            }
            for x in 20..28 {
                atlas.pixels[y * 32 + x] = 128;
            }
        }

        let quad = AtlasGlyphQuad {
            dest_rect: Rect::from_xywh(8.0, 8.0, 24.0, 16.0),
            atlas_rect: Rect::from_xywh(4.0, 4.0, 24.0, 16.0),
            color: Color::new(0.0, 1.0, 0.5, 1.0),
            is_color: false,
        };
        let batch = GlyphAtlasBatch::from_atlas(
            vec![quad],
            Matrix::IDENTITY,
            Arc::new(atlas),
            BlendMode::SrcOver,
            None,
        )
        .with_linear_filter(false)
        .with_text_contrast(1.0);

        let rgba = ctx
            .render_to_rgba(48, 32, Color::BLACK, &[DrawCommand::GlyphAtlas(batch)])
            .expect("render r8 glyph atlas");

        // Full mask region: (8..24, 8..24) -> (0, 255, 128, 255)
        assert_pixel_approx(pixel_at(&rgba, 48, 12, 12), [0, 255, 128, 255], 2);
        // 50% mask region: (24..32, 8..24) -> (0, 128, 64, 255) over black
        assert_pixel_approx(pixel_at(&rgba, 48, 26, 12), [0, 128, 64, 255], 3);
        // Outside glyph: black
        assert_pixel_approx(pixel_at(&rgba, 48, 2, 2), [0, 0, 0, 255], 1);
    }

    /// C++ source: typographer_unittests.cc:CanCreateGlyphAtlas
    #[test]
    fn test_glyph_atlas_rgba8_unorm_color_glyph_sampling() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let mut color_atlas = GlyphAtlas::new(GlyphAtlasType::ColorBitmap, 32, 32);
        // Left half of [0..16, 0..16] is Red, right half [8..16, 0..16] is Blue
        for y in 0..16 {
            for x in 0..8 {
                let idx = (y * 32 + x) * 4;
                color_atlas.pixels[idx..idx + 4].copy_from_slice(&[255, 0, 0, 255]);
            }
            for x in 8..16 {
                let idx = (y * 32 + x) * 4;
                color_atlas.pixels[idx..idx + 4].copy_from_slice(&[0, 64, 255, 255]);
            }
        }

        let quad_full = AtlasGlyphQuad {
            dest_rect: Rect::from_xywh(4.0, 4.0, 16.0, 16.0),
            atlas_rect: Rect::from_xywh(0.0, 0.0, 16.0, 16.0),
            color: Color::WHITE,
            is_color: true,
        };
        let quad_half_alpha = AtlasGlyphQuad {
            dest_rect: Rect::from_xywh(24.0, 4.0, 16.0, 16.0),
            atlas_rect: Rect::from_xywh(0.0, 0.0, 16.0, 16.0),
            color: Color::new(1.0, 1.0, 1.0, 0.5),
            is_color: true,
        };

        let batch = GlyphAtlasBatch::from_atlas(
            vec![quad_full, quad_half_alpha],
            Matrix::IDENTITY,
            Arc::new(color_atlas),
            BlendMode::SrcOver,
            None,
        )
        .with_linear_filter(false);

        let rgba = ctx
            .render_to_rgba(48, 32, Color::BLACK, &[DrawCommand::GlyphAtlas(batch)])
            .expect("render rgba8 color glyph atlas");

        assert_pixel_approx(pixel_at(&rgba, 48, 6, 10), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 48, 14, 10), [0, 64, 255, 255], 1);
        // Modulated by 0.5 alpha over black
        assert_pixel_approx(pixel_at(&rgba, 48, 26, 10), [128, 0, 0, 255], 2);
        assert_pixel_approx(pixel_at(&rgba, 48, 34, 10), [0, 32, 128, 255], 2);
    }

    /// C++ source: entity_unittests.cc:TextContentsCeilsGlyphScaleToDecimal
    #[test]
    fn test_glyph_atlas_mvp_rotation_and_scale() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        atlas.pixels.fill(255);

        // Local quad is [0..10, 0..6]. Apply scale(2, 2) -> [0..20, 0..12],
        // then rotate +90 deg -> [-12..0, 0..20], then translate(32, 16) -> [20..32, 16..36].
        let transform = Matrix::make_translation(32.0, 16.0, 0.0)
            * Matrix::make_rotation_z(Radians::from(Degrees(90.0)))
            * Matrix::make_scale(2.0, 2.0, 1.0);

        let quad = AtlasGlyphQuad {
            dest_rect: Rect::from_xywh(0.0, 0.0, 10.0, 6.0),
            atlas_rect: Rect::from_xywh(2.0, 2.0, 10.0, 6.0),
            color: Color::RED,
            is_color: false,
        };
        let batch = GlyphAtlasBatch::from_atlas(
            vec![quad],
            transform,
            Arc::new(atlas),
            BlendMode::SrcOver,
            None,
        );
        assert_eq!(batch.mvp(), transform);

        let rgba = ctx
            .render_to_rgba(64, 64, Color::BLACK, &[DrawCommand::GlyphAtlas(batch)])
            .expect("render transformed glyph quad");

        // Inside transformed region [20..32, 16..36]: e.g. (26, 26) -> Red
        assert_pixel_approx(pixel_at(&rgba, 64, 26, 26), [255, 0, 0, 255], 1);
        // Unrotated region (40, 18) -> Black
        assert_pixel_approx(pixel_at(&rgba, 64, 40, 18), [0, 0, 0, 255], 1);
    }

    /// C++ source: aiks_unittests.cc:CanRenderTextFrame
    #[test]
    fn test_glyph_atlas_z_order_interleaving_with_rects() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        atlas.pixels.fill(255);
        let atlas = Arc::new(atlas);

        // 1. Background rect: opaque Blue over [0..64, 0..64]
        let bg_rect = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
            transform: Matrix::IDENTITY,
            color: Color::BLUE,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        // 2. Background glyph: opaque White over [8..28, 8..56]
        let bg_text = DrawCommand::GlyphAtlas(
            GlyphAtlasBatch::from_atlas(
                vec![AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(8.0, 8.0, 20.0, 48.0),
                    atlas_rect: Rect::from_xywh(1.0, 1.0, 14.0, 14.0),
                    color: Color::WHITE,
                    is_color: false,
                }],
                Matrix::IDENTITY,
                Arc::clone(&atlas),
                BlendMode::SrcOver,
                None,
            )
            .with_linear_filter(false),
        );
        // 3. Translucent Red overlay (alpha=0.5) over [0..64, 16..48]
        let overlay_rect = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 16.0, 64.0, 32.0)),
            transform: Matrix::IDENTITY,
            color: Color::new(1.0, 0.0, 0.0, 0.5),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        // 4. Foreground glyph: opaque Green over [36..56, 8..56] (on top of the red overlay)
        let fg_text = DrawCommand::GlyphAtlas(
            GlyphAtlasBatch::from_atlas(
                vec![AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(36.0, 8.0, 20.0, 48.0),
                    atlas_rect: Rect::from_xywh(1.0, 1.0, 14.0, 14.0),
                    color: Color::GREEN,
                    is_color: false,
                }],
                Matrix::IDENTITY,
                Arc::clone(&atlas),
                BlendMode::SrcOver,
                None,
            )
            .with_linear_filter(false),
        );

        let rgba = ctx
            .render_to_rgba(
                64,
                64,
                Color::BLACK,
                &[bg_rect, bg_text, overlay_rect, fg_text],
            )
            .expect("render interleaved z-order");

        // Background glyph under red overlay at (18, 32): White (255,255,255) + 50% Red -> (255, 128, 128, 255)
        assert_pixel_approx(pixel_at(&rgba, 64, 18, 32), [255, 128, 128, 255], 2);
        // Background glyph outside overlay at (18, 10): Pure White (255, 255, 255, 255)
        assert_pixel_approx(pixel_at(&rgba, 64, 18, 10), [255, 255, 255, 255], 1);
        // Foreground glyph on top of red overlay at (46, 32): Pure Green (0, 255, 0, 255)
        assert_pixel_approx(pixel_at(&rgba, 64, 46, 32), [0, 255, 0, 255], 1);
    }

    /// C++ source: entity_unittests.cc:FilterCoverageRespectsCropRect
    #[test]
    fn test_glyph_atlas_hardware_scissor_clip_rect() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        atlas.pixels.fill(255);

        let batch = GlyphAtlasBatch::from_atlas(
            vec![AtlasGlyphQuad {
                dest_rect: Rect::from_xywh(8.0, 8.0, 48.0, 48.0),
                atlas_rect: Rect::from_xywh(1.0, 1.0, 14.0, 14.0),
                color: Color::RED,
                is_color: false,
            }],
            Matrix::IDENTITY,
            Arc::new(atlas),
            BlendMode::SrcOver,
            Some(Rect::from_xywh(16.0, 16.0, 24.0, 24.0)),
        );

        let rgba = ctx
            .render_to_rgba(64, 64, Color::BLACK, &[DrawCommand::GlyphAtlas(batch)])
            .expect("render scissor-clipped glyph quad");

        // Inside glyph quad but outside scissor [16..40, 16..40] -> Black
        assert_pixel_approx(pixel_at(&rgba, 64, 12, 24), [0, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 64, 44, 24), [0, 0, 0, 255], 1);
        // Inside scissor -> Red
        assert_pixel_approx(pixel_at(&rgba, 64, 24, 24), [255, 0, 0, 255], 1);
    }

    /// C++ source: typographer_unittests.cc:GlyphAtlasWithOddUniqueGlyphSize
    #[test]
    fn test_glyph_atlas_subpixel_uv_sampling() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        // Create 16x16 R8Unorm atlas where columns x < 8 are 0 and columns x >= 8 are 255
        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        for y in 0..16 {
            for x in 8..16 {
                atlas.pixels[y * 16 + x] = 255;
            }
        }
        let atlas = Arc::new(atlas);

        // Quad 1: 1x1 pixel at (4, 4) sampling interior texel (9.0..10.0) -> 255
        // Quad 2: 1x1 pixel at (12, 4) sampling half-texel boundary (7.5..8.5) -> ~128 bilinear average
        let batch = GlyphAtlasBatch::from_atlas(
            vec![
                AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(4.0, 4.0, 1.0, 1.0),
                    atlas_rect: Rect::from_xywh(9.0, 4.0, 1.0, 1.0),
                    color: Color::WHITE,
                    is_color: false,
                },
                AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(12.0, 4.0, 1.0, 1.0),
                    atlas_rect: Rect::from_xywh(7.5, 4.0, 1.0, 1.0),
                    color: Color::WHITE,
                    is_color: false,
                },
            ],
            Matrix::IDENTITY,
            atlas,
            BlendMode::SrcOver,
            None,
        )
        .with_linear_filter(true)
        .with_text_contrast(1.0);

        let rgba = ctx
            .render_to_rgba(32, 16, Color::BLACK, &[DrawCommand::GlyphAtlas(batch)])
            .expect("render subpixel uv");

        let exact = pixel_at(&rgba, 32, 4, 4)[0];
        let subpixel = pixel_at(&rgba, 32, 12, 4)[0];
        assert!(exact > 245, "exact={exact}");
        assert!(
            subpixel > 95 && subpixel < 160,
            "subpixel={subpixel} (expected ~128 bilinear interpolation)"
        );
    }

    /// C++ source: typographer_unittests.cc:LazyAtlasTracksColor
    #[test]
    fn test_glyph_atlas_mixed_alpha_and_color_and_generation_cache() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let mut alpha_atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        alpha_atlas.pixels.fill(255);
        let mut color_atlas = GlyphAtlas::new(GlyphAtlasType::ColorBitmap, 16, 16);
        for px in color_atlas.pixels.as_chunks_mut::<4>().0 {
            *px = [0, 200, 255, 255];
        }

        let batch = GlyphAtlasBatch::from_typographer(
            vec![
                AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(4.0, 4.0, 8.0, 8.0),
                    atlas_rect: Rect::from_xywh(1.0, 1.0, 8.0, 8.0),
                    color: Color::RED,
                    is_color: false,
                },
                AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(20.0, 4.0, 8.0, 8.0),
                    atlas_rect: Rect::from_xywh(1.0, 1.0, 8.0, 8.0),
                    color: Color::WHITE,
                    is_color: true,
                },
            ],
            Matrix::IDENTITY,
            &alpha_atlas,
            &color_atlas,
            BlendMode::SrcOver,
            None,
        );

        let rgba = ctx
            .render_to_rgba(32, 16, Color::BLACK, &[DrawCommand::GlyphAtlas(batch)])
            .expect("render mixed glyph batch");
        assert_pixel_approx(pixel_at(&rgba, 32, 8, 8), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 32, 24, 8), [0, 200, 255, 255], 1);
        assert_eq!(ctx.cached_atlas_count(), 2);
    }

    /// C++ source: entity_unittests.cc:CanRenderMSAADiagonalEdge
    #[test]
    fn test_msaa_4x_diagonal_edge_antialiasing() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let diag_triangle = TessellatedMesh {
            vertices: vec![
                Vertex2D::new(0.0, 0.0),
                Vertex2D::new(32.0, 0.0),
                Vertex2D::new(32.0, 24.0),
            ],
            indices: vec![0, 1, 2],
        };
        let cmd = DrawCommand::SolidFill {
            mesh: diag_triangle,
            transform: Matrix::IDENTITY,
            color: Color::WHITE,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let target_1x = OffscreenTarget::new_with_sample_count(&ctx, 32, 32, 1);
        assert_eq!(target_1x.sample_count(), 1);
        let rgba_1x = target_1x
            .render_and_readback(&ctx, Color::BLACK, std::slice::from_ref(&cmd))
            .expect("render 1x");

        let target_4x = OffscreenTarget::new(&ctx, 32, 32);
        assert_eq!(target_4x.sample_count(), 4);
        let rgba_4x = target_4x
            .render_and_readback(&ctx, Color::BLACK, &[cmd])
            .expect("render 4x");

        let fractional_1x = rgba_1x
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] > 0 && px[0] < 255)
            .count();
        let fractional_4x = rgba_4x
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] > 0 && px[0] < 255)
            .count();

        assert_eq!(
            fractional_1x, 0,
            "1x non-MSAA solid triangle should have 0 intermediate coverage pixels, got {fractional_1x}"
        );
        assert!(
            fractional_4x >= 16,
            "4x MSAA diagonal edge should produce fractional intermediate coverage pixels, got {fractional_4x}"
        );

        let (self_metrics, triptych) =
            compare_golden_rgba(&rgba_4x, &rgba_4x, 32, 32, 0).expect("self golden diff");
        assert_eq!(self_metrics.mismatched_pixels, 0);
        assert_eq!(self_metrics.max_channel_diff, 0);
        assert!(self_metrics.rmse.abs() < 1e-9);
        assert_eq!(triptych.len(), 32 * 3 * 32 * 4);

        let (vs_1x_metrics, _) =
            compare_golden_rgba(&rgba_4x, &rgba_1x, 32, 32, 2).expect("4x vs 1x golden diff");
        assert!(vs_1x_metrics.mismatched_pixels >= 16);
        assert!(vs_1x_metrics.rmse > 1.0);
    }
}
