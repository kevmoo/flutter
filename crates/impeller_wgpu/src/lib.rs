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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const SOLID_FILL_WGSL: &str = include_str!("shaders/solid_fill.wgsl");
const LINEAR_GRADIENT_FILL_WGSL: &str = include_str!("shaders/linear_gradient_fill.wgsl");
const RADIAL_GRADIENT_FILL_WGSL: &str = include_str!("shaders/radial_gradient_fill.wgsl");
const SWEEP_GRADIENT_FILL_WGSL: &str = include_str!("shaders/sweep_gradient_fill.wgsl");
const CONICAL_GRADIENT_FILL_WGSL: &str = include_str!("shaders/conical_gradient_fill.wgsl");
const RRECT_BLUR_WGSL: &str = include_str!("shaders/rrect_blur.wgsl");
const GLYPH_ATLAS_WGSL: &str = include_str!("shaders/glyph_atlas.wgsl");
const TEXTURE_FILL_WGSL: &str = include_str!("shaders/texture_fill.wgsl");
const STENCIL_CLIP_WGSL: &str = include_str!("shaders/stencil_clip.wgsl");
const IMAGE_FILTER_WGSL: &str = include_str!("shaders/image_filter.wgsl");

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const STENCIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;

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
pub struct ColoredSolidVertex {
    pub position: [Scalar; 2],
    pub color: [f32; 4],
}

impl ColoredSolidVertex {
    pub const fn new(position: [Scalar; 2], color: [f32; 4]) -> Self {
        Self { position, color }
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

#[derive(Clone, Debug, PartialEq)]
pub struct SweepGradientParams {
    pub center: Point,
    pub start_angle_deg: Scalar,
    pub end_angle_deg: Scalar,
    pub stops: Vec<GradientStop>,
    pub tile_mode: TileMode,
    pub alpha: Scalar,
}

impl SweepGradientParams {
    pub fn new(
        center: Point,
        start_angle_deg: Scalar,
        end_angle_deg: Scalar,
        stops: Vec<GradientStop>,
        tile_mode: TileMode,
        alpha: Scalar,
    ) -> Self {
        Self {
            center,
            start_angle_deg,
            end_angle_deg,
            stops,
            tile_mode,
            alpha,
        }
    }

    pub fn two_stop(
        center: Point,
        start_angle_deg: Scalar,
        end_angle_deg: Scalar,
        start_color: Color,
        end_color: Color,
        tile_mode: TileMode,
    ) -> Self {
        Self {
            center,
            start_angle_deg,
            end_angle_deg,
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
pub struct ConicalGradientParams {
    pub start_center: Point,
    pub start_radius: Scalar,
    pub end_center: Point,
    pub end_radius: Scalar,
    pub stops: Vec<GradientStop>,
    pub tile_mode: TileMode,
    pub alpha: Scalar,
}

impl ConicalGradientParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        start_center: Point,
        start_radius: Scalar,
        end_center: Point,
        end_radius: Scalar,
        stops: Vec<GradientStop>,
        tile_mode: TileMode,
        alpha: Scalar,
    ) -> Self {
        Self {
            start_center,
            start_radius,
            end_center,
            end_radius,
            stops,
            tile_mode,
            alpha,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn two_stop(
        start_center: Point,
        start_radius: Scalar,
        end_center: Point,
        end_radius: Scalar,
        start_color: Color,
        end_color: Color,
        tile_mode: TileMode,
    ) -> Self {
        Self {
            start_center,
            start_radius,
            end_center,
            end_radius,
            stops: vec![
                GradientStop::new(0.0, start_color),
                GradientStop::new(1.0, end_color),
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
    pub const fn new(rect: Rect, corner_radius: Scalar, blur_sigma: Scalar, color: Color) -> Self {
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
pub struct SweepGradientBatch {
    pub mesh: TessellatedMesh,
    pub transform: Matrix,
    pub gradient: SweepGradientParams,
    pub blend_mode: BlendMode,
    pub clip_rect: Option<Rect>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConicalGradientBatch {
    pub mesh: TessellatedMesh,
    pub transform: Matrix,
    pub gradient: ConicalGradientParams,
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
                    || (x.atlas_id == y.atlas_id
                        && x.atlas_type == y.atlas_type
                        && x.size == y.size
                        && x.generation == y.generation
                        && x.resize_epoch == y.resize_epoch
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
pub struct ImageTexture {
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub rgba_premultiplied: Arc<Vec<u8>>,
}

impl ImageTexture {
    pub fn new(id: u64, width: u32, height: u32, rgba_premultiplied: Vec<u8>) -> Self {
        Self {
            id,
            width,
            height,
            rgba_premultiplied: Arc::new(rgba_premultiplied),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextureFillBatch {
    pub vertices: Vec<UvVertex>,
    pub indices: Vec<u32>,
    pub transform: Matrix,
    pub texture: ImageTexture,
    pub alpha: Scalar,
    pub tile_mode_x: TileMode,
    pub tile_mode_y: TileMode,
    pub linear_filter: bool,
    pub blend_mode: BlendMode,
    pub clip_rect: Option<Rect>,
}

impl TextureFillBatch {
    #[allow(clippy::too_many_arguments)]
    pub fn from_rect(
        src_rect: Rect,
        dst_rect: Rect,
        transform: Matrix,
        texture: ImageTexture,
        alpha: Scalar,
        linear_filter: bool,
        blend_mode: BlendMode,
        clip_rect: Option<Rect>,
    ) -> Self {
        let inv_w = 1.0 / (texture.width.max(1) as Scalar);
        let inv_h = 1.0 / (texture.height.max(1) as Scalar);
        let u0 = src_rect.get_left() * inv_w;
        let v0 = src_rect.get_top() * inv_h;
        let u1 = src_rect.get_right() * inv_w;
        let v1 = src_rect.get_bottom() * inv_h;

        let l = dst_rect.get_left();
        let t = dst_rect.get_top();
        let r = dst_rect.get_right();
        let b = dst_rect.get_bottom();

        Self {
            vertices: vec![
                UvVertex::new(l, t, u0, v0),
                UvVertex::new(r, t, u1, v0),
                UvVertex::new(r, b, u1, v1),
                UvVertex::new(l, b, u0, v1),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            transform,
            texture,
            alpha,
            tile_mode_x: TileMode::Clamp,
            tile_mode_y: TileMode::Clamp,
            linear_filter,
            blend_mode,
            clip_rect,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerColorFilter {
    Mode(Color, BlendMode),
    Matrix([Scalar; 20]),
    SrgbToLinear,
    LinearToSrgb,
    Compose(Box<LayerColorFilter>, Box<LayerColorFilter>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerImageFilter {
    Blur {
        sigma_x: Scalar,
        sigma_y: Scalar,
        tile_mode: TileMode,
    },
    Dilate {
        radius_x: Scalar,
        radius_y: Scalar,
    },
    Erode {
        radius_x: Scalar,
        radius_y: Scalar,
    },
    Matrix(Matrix),
    ColorFilter(LayerColorFilter),
    Compose(Box<LayerImageFilter>, Box<LayerImageFilter>),
}

impl LayerImageFilter {
    pub fn halo_padding(&self) -> (Scalar, Scalar) {
        match self {
            Self::Blur {
                sigma_x, sigma_y, ..
            } => (
                (sigma_x.max(0.0) * 3.0).ceil(),
                (sigma_y.max(0.0) * 3.0).ceil(),
            ),
            Self::Dilate { radius_x, radius_y } | Self::Erode { radius_x, radius_y } => {
                (radius_x.max(0.0).ceil(), radius_y.max(0.0).ceil())
            }
            Self::Matrix(_) | Self::ColorFilter(_) => (0.0, 0.0),
            Self::Compose(outer, inner) => {
                let (ix, iy) = inner.halo_padding();
                let (ox, oy) = outer.halo_padding();
                (ix + ox, iy + oy)
            }
        }
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
    SweepGradientFill {
        mesh: TessellatedMesh,
        transform: Matrix,
        gradient: SweepGradientParams,
        blend_mode: BlendMode,
        clip_rect: Option<Rect>,
    },
    ConicalGradientFill {
        mesh: TessellatedMesh,
        transform: Matrix,
        gradient: ConicalGradientParams,
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
    TextureFill(TextureFillBatch),
    PushClip {
        mesh: TessellatedMesh,
        transform: Matrix,
        is_difference: bool,
        clip_rect: Option<Rect>,
    },
    PopClip {
        clip_rect: Option<Rect>,
    },
    SaveLayer {
        bounds: Rect,
        commands: Vec<DrawCommand>,
        alpha: Scalar,
        blend_mode: BlendMode,
        color_filter: Option<LayerColorFilter>,
        image_filter: Option<LayerImageFilter>,
        clip_rect: Option<Rect>,
    },
}

impl DrawCommand {
    pub fn clip_rect(&self) -> Option<Rect> {
        match self {
            DrawCommand::SolidFill { clip_rect, .. }
            | DrawCommand::LinearGradientFill { clip_rect, .. }
            | DrawCommand::RadialGradientFill { clip_rect, .. }
            | DrawCommand::SweepGradientFill { clip_rect, .. }
            | DrawCommand::ConicalGradientFill { clip_rect, .. }
            | DrawCommand::RRectBlur { clip_rect, .. }
            | DrawCommand::PushClip { clip_rect, .. }
            | DrawCommand::PopClip { clip_rect }
            | DrawCommand::SaveLayer { clip_rect, .. } => *clip_rect,
            DrawCommand::GlyphAtlas(batch) => batch.clip_rect,
            DrawCommand::TextureFill(batch) => batch.clip_rect,
        }
    }

    pub fn blend_mode(&self) -> BlendMode {
        match self {
            DrawCommand::SolidFill { blend_mode, .. }
            | DrawCommand::LinearGradientFill { blend_mode, .. }
            | DrawCommand::RadialGradientFill { blend_mode, .. }
            | DrawCommand::SweepGradientFill { blend_mode, .. }
            | DrawCommand::ConicalGradientFill { blend_mode, .. }
            | DrawCommand::RRectBlur { blend_mode, .. }
            | DrawCommand::SaveLayer { blend_mode, .. } => *blend_mode,
            DrawCommand::GlyphAtlas(batch) => batch.blend_mode,
            DrawCommand::TextureFill(batch) => batch.blend_mode,
            DrawCommand::PushClip { .. } | DrawCommand::PopClip { .. } => BlendMode::SrcOver,
        }
    }

    pub fn with_clip_rect(mut self, new_clip: Option<Rect>) -> Self {
        match &mut self {
            DrawCommand::SolidFill { clip_rect, .. }
            | DrawCommand::LinearGradientFill { clip_rect, .. }
            | DrawCommand::RadialGradientFill { clip_rect, .. }
            | DrawCommand::SweepGradientFill { clip_rect, .. }
            | DrawCommand::ConicalGradientFill { clip_rect, .. }
            | DrawCommand::RRectBlur { clip_rect, .. }
            | DrawCommand::PushClip { clip_rect, .. }
            | DrawCommand::PopClip { clip_rect }
            | DrawCommand::SaveLayer { clip_rect, .. } => {
                *clip_rect = new_clip;
            }
            DrawCommand::GlyphAtlas(batch) => {
                batch.clip_rect = new_clip;
            }
            DrawCommand::TextureFill(batch) => {
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

impl From<SweepGradientBatch> for DrawCommand {
    fn from(b: SweepGradientBatch) -> Self {
        DrawCommand::SweepGradientFill {
            mesh: b.mesh,
            transform: b.transform,
            gradient: b.gradient,
            blend_mode: b.blend_mode,
            clip_rect: b.clip_rect,
        }
    }
}

impl From<ConicalGradientBatch> for DrawCommand {
    fn from(b: ConicalGradientBatch) -> Self {
        DrawCommand::ConicalGradientFill {
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

impl From<TextureFillBatch> for DrawCommand {
    fn from(b: TextureFillBatch) -> Self {
        DrawCommand::TextureFill(b)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PipelineKind {
    SolidFill,
    LinearGradientFill,
    RadialGradientFill,
    SweepGradientFill,
    ConicalGradientFill,
    RRectBlur,
    GlyphAtlas,
    TextureFill,
    StencilIncrement,
    StencilDecrement,
    ImageFilter,
}

const FILTER_KIND_BLUR_1D: u32 = 0;
const FILTER_KIND_DILATE_1D: u32 = 1;
const FILTER_KIND_ERODE_1D: u32 = 2;
const FILTER_KIND_COLOR_MATRIX: u32 = 3;
const FILTER_KIND_COLOR_MODE: u32 = 4;
const FILTER_KIND_SRGB_TO_LINEAR: u32 = 5;
const FILTER_KIND_LINEAR_TO_SRGB: u32 = 6;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct ImageFilterFragUniform {
    color_matrix_row0: [f32; 4],
    color_matrix_row1: [f32; 4],
    color_matrix_row2: [f32; 4],
    color_matrix_row3: [f32; 4],
    color_matrix_offset: [f32; 4],
    mode_color_pm: [f32; 4],
    step_uv: [f32; 2],
    tex_size: [f32; 2],
    sigma: f32,
    alpha: f32,
    filter_kind: u32,
    radius: i32,
    tile_mode: u32,
    blend_mode: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct TextureFragUniform {
    alpha: f32,
    tile_mode_x: u32,
    tile_mode_y: u32,
    _pad: u32,
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
struct SweepGradientFragUniform {
    center: [f32; 2],
    bias: f32,
    scale: f32,
    decal_border_color: [f32; 4],
    fast_color_0: [f32; 4],
    fast_color_1: [f32; 4],
    tile_mode: u32,
    colors_length: u32,
    is_fast_two_stop: u32,
    alpha: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct ConicalGradientFragUniform {
    start_center: [f32; 2],
    start_radius: f32,
    end_radius: f32,
    end_center: [f32; 2],
    alpha: f32,
    kind: u32,
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

pub const MAX_GRADIENT_STOPS: usize = 64;

fn build_gpu_color_points(
    stops: &[GradientStop],
) -> (
    [GpuColorPoint; MAX_GRADIENT_STOPS],
    u32,
    bool,
    [f32; 4],
    [f32; 4],
) {
    let mut gpu_stops = [GpuColorPoint::default(); MAX_GRADIENT_STOPS];
    if stops.is_empty() {
        return (gpu_stops, 0, true, [0.0; 4], [0.0; 4]);
    }

    let mut sorted = stops.to_vec();
    sorted.sort_by(|a, b| {
        a.offset
            .partial_cmp(&b.offset)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if sorted.len() > MAX_GRADIENT_STOPS {
        sorted.truncate(MAX_GRADIENT_STOPS);
    }

    let is_fast_two_stop = sorted.len() == 2
        && (sorted[0].offset - 0.0).abs() <= K_EH_CLOSE_ENOUGH
        && (sorted[1].offset - 1.0).abs() <= K_EH_CLOSE_ENOUGH;

    let c0 = sorted
        .first()
        .map(|s| s.color)
        .unwrap_or(Color::TRANSPARENT);
    let c1 = sorted.last().map(|s| s.color).unwrap_or(c0);
    let fast_0 = [c0.red, c0.green, c0.blue, c0.alpha];
    let fast_1 = [c1.red, c1.green, c1.blue, c1.alpha];

    let mut last_stop = sorted[0].offset;
    for (i, s) in sorted.iter().enumerate() {
        let cur_stop = s.offset.clamp(0.0, 1.0);
        let delta = if i == 0 { 0.0 } else { cur_stop - last_stop };
        let inverse_delta = if delta.abs() <= 1.0e-6 {
            10000.0
        } else {
            1.0 / delta
        };
        gpu_stops[i] = GpuColorPoint {
            color: [s.color.red, s.color.green, s.color.blue, s.color.alpha],
            stop: cur_stop,
            inverse_delta,
            _pad: [0.0; 2],
        };
        last_stop = cur_stop;
    }

    (
        gpu_stops,
        sorted.len() as u32,
        is_fast_two_stop,
        fast_0,
        fast_1,
    )
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct AtlasCacheKey {
    atlas_type: GlyphAtlasType,
    atlas_id: u64,
    resize_epoch: u64,
    width: u32,
    height: u32,
    pixel_hash: u64,
}

impl AtlasCacheKey {
    fn from_atlas(atlas: &GlyphAtlas) -> Self {
        let pixel_hash = if atlas.generation > 0 || (atlas.atlas_id > 0 && atlas.pixels.is_empty())
        {
            0
        } else {
            let mut hasher = DefaultHasher::new();
            atlas.pixels.hash(&mut hasher);
            hasher.finish()
        };
        Self {
            atlas_type: atlas.atlas_type,
            atlas_id: atlas.atlas_id,
            resize_epoch: atlas.resize_epoch,
            width: atlas.size.width.max(1) as u32,
            height: atlas.size.height.max(1) as u32,
            pixel_hash,
        }
    }
}

fn is_2d_affine_transform(m: &Matrix) -> bool {
    m.m[2].abs() < 1e-6
        && m.m[3].abs() < 1e-6
        && m.m[6].abs() < 1e-6
        && m.m[7].abs() < 1e-6
        && m.m[14].abs() < 1e-6
        && (m.m[15] - 1.0).abs() < 1e-5
}

fn is_2d_translation_only(m: &Matrix) -> bool {
    (m.m[0] - 1.0).abs() < 1e-5
        && m.m[1].abs() < 1e-6
        && m.m[2].abs() < 1e-6
        && m.m[3].abs() < 1e-6
        && m.m[4].abs() < 1e-6
        && (m.m[5] - 1.0).abs() < 1e-5
        && m.m[6].abs() < 1e-6
        && m.m[7].abs() < 1e-6
        && m.m[14].abs() < 1e-6
        && (m.m[15] - 1.0).abs() < 1e-5
}

fn transform_vertex_2d(m: &Matrix, pos: [Scalar; 2]) -> [Scalar; 2] {
    [
        m.m[0] * pos[0] + m.m[4] * pos[1] + m.m[12],
        m.m[1] * pos[0] + m.m[5] * pos[1] + m.m[13],
    ]
}

fn effective_glyph_linear_filter(batch: &GlyphAtlasBatch) -> bool {
    batch
        .linear_filter
        .unwrap_or_else(|| !batch.transform.is_translation_scale_only())
}

fn compatible_atlas_slot(
    prev_atlas: &Option<Arc<GlyphAtlas>>,
    prev_uses_slot: bool,
    cur_atlas: &Option<Arc<GlyphAtlas>>,
    cur_uses_slot: bool,
) -> bool {
    match (prev_atlas, cur_atlas) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            Arc::ptr_eq(x, y) || AtlasCacheKey::from_atlas(x) == AtlasCacheKey::from_atlas(y)
        }
        (None, Some(_)) => !prev_uses_slot,
        (Some(_), None) => !cur_uses_slot,
    }
}

fn shift_rect(r: Rect, dx: Scalar, dy: Scalar) -> Rect {
    Rect::from_xywh(
        r.origin.x + dx,
        r.origin.y + dy,
        r.size.width,
        r.size.height,
    )
}

#[inline]
fn intersect_rects(a: Rect, b: Rect) -> Option<Rect> {
    let left = a.get_left().max(b.get_left());
    let top = a.get_top().max(b.get_top());
    let right = a.get_right().min(b.get_right());
    let bottom = a.get_bottom().min(b.get_bottom());
    if right > left && bottom > top {
        Some(Rect::from_ltrb(left, top, right, bottom))
    } else {
        None
    }
}

fn transform_2d_rect(tx: &Matrix, r: Rect) -> Rect {
    if r.is_empty() {
        return Rect::ZERO;
    }
    if *tx == Matrix::IDENTITY {
        return r;
    }
    let l = r.get_left();
    let t = r.get_top();
    let right = r.get_right();
    let b = r.get_bottom();
    let p0 = transform_vertex_2d(tx, [l, t]);
    let p1 = transform_vertex_2d(tx, [right, t]);
    let p2 = transform_vertex_2d(tx, [right, b]);
    let p3 = transform_vertex_2d(tx, [l, b]);
    let min_x = p0[0].min(p1[0]).min(p2[0]).min(p3[0]);
    let min_y = p0[1].min(p1[1]).min(p2[1]).min(p3[1]);
    let max_x = p0[0].max(p1[0]).max(p2[0]).max(p3[0]);
    let max_y = p0[1].max(p1[1]).max(p2[1]).max(p3[1]);
    Rect::from_ltrb(min_x, min_y, max_x, max_y)
}

fn apply_clip_to_dev_bounds(dev_rect: Option<Rect>, clip_rect: Option<Rect>) -> Option<Rect> {
    match (dev_rect, clip_rect) {
        (Some(r), Some(clip)) => {
            if r.is_empty() || clip.is_empty() {
                Some(Rect::ZERO)
            } else {
                Some(intersect_rects(r, clip).unwrap_or(Rect::ZERO))
            }
        }
        (Some(r), None) => Some(r),
        (None, Some(clip)) => {
            if clip.is_empty() {
                Some(Rect::ZERO)
            } else {
                Some(clip)
            }
        }
        (None, None) => None,
    }
}

fn mesh_local_aabb(mesh: &TessellatedMesh) -> Option<Rect> {
    let first = mesh.vertices.first()?;
    let mut min_x = first.position[0];
    let mut min_y = first.position[1];
    let mut max_x = min_x;
    let mut max_y = min_y;
    for v in &mesh.vertices[1..] {
        min_x = min_x.min(v.position[0]);
        min_y = min_y.min(v.position[1]);
        max_x = max_x.max(v.position[0]);
        max_y = max_y.max(v.position[1]);
    }
    Some(Rect::from_ltrb(min_x, min_y, max_x, max_y))
}

#[inline]
fn union_rects(a: Rect, b: Rect) -> Rect {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    let left = a.get_left().min(b.get_left());
    let top = a.get_top().min(b.get_top());
    let right = a.get_right().max(b.get_right());
    let bottom = a.get_bottom().max(b.get_bottom());
    Rect::from_ltrb(left, top, right, bottom)
}

struct CommandClusterInfo {
    dev_bounds: Option<Rect>,
    clusterable: bool,
    uses_alpha: bool,
    uses_color: bool,
}

fn inspect_command_for_clustering(cmd: &DrawCommand) -> CommandClusterInfo {
    match cmd {
        DrawCommand::SolidFill {
            mesh,
            transform,
            blend_mode,
            clip_rect,
            ..
        } => {
            let is_2d = *transform == Matrix::IDENTITY || is_2d_affine_transform(transform);
            let dev_bounds = if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                Some(Rect::ZERO)
            } else if is_2d {
                let local = mesh_local_aabb(mesh).unwrap_or(Rect::ZERO);
                apply_clip_to_dev_bounds(Some(transform_2d_rect(transform, local)), *clip_rect)
            } else {
                apply_clip_to_dev_bounds(None, *clip_rect)
            };
            CommandClusterInfo {
                dev_bounds,
                clusterable: *blend_mode == BlendMode::SrcOver && is_2d,
                uses_alpha: false,
                uses_color: false,
            }
        }
        DrawCommand::LinearGradientFill {
            mesh,
            transform,
            clip_rect,
            ..
        }
        | DrawCommand::RadialGradientFill {
            mesh,
            transform,
            clip_rect,
            ..
        }
        | DrawCommand::SweepGradientFill {
            mesh,
            transform,
            clip_rect,
            ..
        }
        | DrawCommand::ConicalGradientFill {
            mesh,
            transform,
            clip_rect,
            ..
        } => {
            let is_2d = *transform == Matrix::IDENTITY || is_2d_affine_transform(transform);
            let dev_bounds = if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                Some(Rect::ZERO)
            } else if is_2d {
                let local = mesh_local_aabb(mesh).unwrap_or(Rect::ZERO);
                apply_clip_to_dev_bounds(Some(transform_2d_rect(transform, local)), *clip_rect)
            } else {
                apply_clip_to_dev_bounds(None, *clip_rect)
            };
            CommandClusterInfo {
                dev_bounds,
                clusterable: false,
                uses_alpha: false,
                uses_color: false,
            }
        }
        DrawCommand::PushClip { .. }
        | DrawCommand::PopClip { .. }
        | DrawCommand::SaveLayer { .. } => CommandClusterInfo {
            dev_bounds: None,
            clusterable: false,
            uses_alpha: false,
            uses_color: false,
        },
        DrawCommand::RRectBlur {
            transform,
            params,
            blend_mode,
            clip_rect,
        } => {
            let is_2d = *transform == Matrix::IDENTITY || is_2d_affine_transform(transform);
            let dev_bounds = if params.rect.is_empty() {
                Some(Rect::ZERO)
            } else if is_2d {
                let local = params.coverage_rect();
                apply_clip_to_dev_bounds(Some(transform_2d_rect(transform, local)), *clip_rect)
            } else {
                apply_clip_to_dev_bounds(None, *clip_rect)
            };
            CommandClusterInfo {
                dev_bounds,
                clusterable: *blend_mode == BlendMode::SrcOver && is_2d,
                uses_alpha: false,
                uses_color: false,
            }
        }
        DrawCommand::GlyphAtlas(batch) => {
            let mut min_x = Scalar::INFINITY;
            let mut min_y = Scalar::INFINITY;
            let mut max_x = Scalar::NEG_INFINITY;
            let mut max_y = Scalar::NEG_INFINITY;
            let mut has_quad = false;
            let mut uses_alpha = false;
            let mut uses_color = false;
            for q in &batch.quads {
                if !q.dest_rect.is_empty() && !q.atlas_rect.is_empty() {
                    min_x = min_x.min(q.dest_rect.get_left());
                    min_y = min_y.min(q.dest_rect.get_top());
                    max_x = max_x.max(q.dest_rect.get_right());
                    max_y = max_y.max(q.dest_rect.get_bottom());
                    has_quad = true;
                    if q.is_color {
                        uses_color = true;
                    } else {
                        uses_alpha = true;
                    }
                }
            }
            let is_2d =
                batch.transform == Matrix::IDENTITY || is_2d_affine_transform(&batch.transform);
            let dev_bounds = if !has_quad {
                Some(Rect::ZERO)
            } else if is_2d {
                let local = Rect::from_ltrb(min_x, min_y, max_x, max_y);
                apply_clip_to_dev_bounds(
                    Some(transform_2d_rect(&batch.transform, local)),
                    batch.clip_rect,
                )
            } else {
                apply_clip_to_dev_bounds(None, batch.clip_rect)
            };
            CommandClusterInfo {
                dev_bounds,
                clusterable: batch.blend_mode == BlendMode::SrcOver && is_2d,
                uses_alpha,
                uses_color,
            }
        }
        DrawCommand::TextureFill(batch) => {
            let is_2d =
                batch.transform == Matrix::IDENTITY || is_2d_affine_transform(&batch.transform);
            let dev_bounds = if batch.vertices.is_empty() || batch.indices.is_empty() {
                Some(Rect::ZERO)
            } else if is_2d {
                let first = &batch.vertices[0];
                let mut min_x = first.position[0];
                let mut min_y = first.position[1];
                let mut max_x = min_x;
                let mut max_y = min_y;
                for v in &batch.vertices[1..] {
                    min_x = min_x.min(v.position[0]);
                    min_y = min_y.min(v.position[1]);
                    max_x = max_x.max(v.position[0]);
                    max_y = max_y.max(v.position[1]);
                }
                let local = Rect::from_ltrb(min_x, min_y, max_x, max_y);
                apply_clip_to_dev_bounds(
                    Some(transform_2d_rect(&batch.transform, local)),
                    batch.clip_rect,
                )
            } else {
                apply_clip_to_dev_bounds(None, batch.clip_rect)
            };
            CommandClusterInfo {
                dev_bounds,
                clusterable: false,
                uses_alpha: false,
                uses_color: false,
            }
        }
    }
}

struct ClusterSlot<'a> {
    cmds: Vec<&'a DrawCommand>,
    member_bounds: Vec<Rect>,
    union_bounds: Option<Rect>,
    clusterable: bool,
    uses_alpha: bool,
    uses_color: bool,
}

#[inline]
fn slot_overlaps_bounds(slot: &ClusterSlot<'_>, bounds: Rect) -> bool {
    let Some(slot_union) = slot.union_bounds else {
        return true;
    };
    if slot_union.is_empty() || intersect_rects(bounds, slot_union).is_none() {
        return false;
    }
    slot.member_bounds
        .iter()
        .any(|&b| intersect_rects(bounds, b).is_some())
}

fn is_slot_compatible(
    slot: &ClusterSlot<'_>,
    cur_cmd: &DrawCommand,
    cur_uses_alpha: bool,
    cur_uses_color: bool,
) -> bool {
    let Some(&prev_cmd) = slot.cmds.first() else {
        return false;
    };
    match (prev_cmd, cur_cmd) {
        (
            DrawCommand::SolidFill {
                clip_rect: prev_clip,
                ..
            },
            DrawCommand::SolidFill {
                clip_rect: cur_clip,
                ..
            },
        ) => prev_clip == cur_clip,
        (
            DrawCommand::RRectBlur {
                clip_rect: prev_clip,
                ..
            },
            DrawCommand::RRectBlur {
                clip_rect: cur_clip,
                ..
            },
        ) => prev_clip == cur_clip,
        (DrawCommand::GlyphAtlas(prev_b), DrawCommand::GlyphAtlas(cur_b)) => {
            if prev_b.clip_rect != cur_b.clip_rect
                || prev_b.use_text_color != cur_b.use_text_color
                || prev_b.text_contrast.max(0.1).to_bits() != cur_b.text_contrast.max(0.1).to_bits()
                || prev_b.alpha.clamp(0.0, 1.0).to_bits() != cur_b.alpha.clamp(0.0, 1.0).to_bits()
                || effective_glyph_linear_filter(prev_b) != effective_glyph_linear_filter(cur_b)
            {
                return false;
            }
            let prev_alpha = if prev_b.alpha_atlas.is_some() {
                &prev_b.alpha_atlas
            } else if let Some(DrawCommand::GlyphAtlas(last_b)) = slot.cmds.last() {
                &last_b.alpha_atlas
            } else {
                &prev_b.alpha_atlas
            };
            let prev_color = if prev_b.color_atlas.is_some() {
                &prev_b.color_atlas
            } else if let Some(DrawCommand::GlyphAtlas(last_b)) = slot.cmds.last() {
                &last_b.color_atlas
            } else {
                &prev_b.color_atlas
            };
            compatible_atlas_slot(
                prev_alpha,
                slot.uses_alpha,
                &cur_b.alpha_atlas,
                cur_uses_alpha,
            ) && compatible_atlas_slot(
                prev_color,
                slot.uses_color,
                &cur_b.color_atlas,
                cur_uses_color,
            )
        }
        _ => false,
    }
}

fn cluster_commands_into_slots(commands: &[DrawCommand]) -> Vec<ClusterSlot<'_>> {
    const MAX_LOOKBACK: usize = 64;
    let mut slots: Vec<ClusterSlot<'_>> = Vec::with_capacity(commands.len().min(64));

    for cmd in commands {
        let info = inspect_command_for_clustering(cmd);
        let mut merged = false;

        if info.clusterable {
            if let Some(bounds) = info.dev_bounds {
                if !bounds.is_empty() {
                    let min_idx = slots.len().saturating_sub(MAX_LOOKBACK);
                    for idx in (min_idx..slots.len()).rev() {
                        if slots[idx].clusterable
                            && is_slot_compatible(
                                &slots[idx],
                                cmd,
                                info.uses_alpha,
                                info.uses_color,
                            )
                        {
                            let slot = &mut slots[idx];
                            slot.cmds.push(cmd);
                            slot.member_bounds.push(bounds);
                            slot.union_bounds = Some(match slot.union_bounds {
                                Some(u) if !u.is_empty() => union_rects(u, bounds),
                                _ => bounds,
                            });
                            slot.uses_alpha |= info.uses_alpha;
                            slot.uses_color |= info.uses_color;
                            merged = true;
                            break;
                        }
                        if slot_overlaps_bounds(&slots[idx], bounds) {
                            break;
                        }
                    }
                }
            }
        }

        if !merged {
            let (union_bounds, member_bounds, non_empty_bounds) = match info.dev_bounds {
                Some(b) if b.is_empty() => (Some(Rect::ZERO), Vec::new(), false),
                Some(b) => (Some(b), vec![b], true),
                None => (None, Vec::new(), false),
            };
            slots.push(ClusterSlot {
                cmds: vec![cmd],
                member_bounds,
                union_bounds,
                clusterable: info.clusterable && non_empty_bounds,
                uses_alpha: info.uses_alpha,
                uses_color: info.uses_color,
            });
        }
    }

    slots
}

fn cluster_non_overlapping_commands(commands: &[DrawCommand]) -> Vec<&DrawCommand> {
    if commands.len() <= 2 {
        return commands.iter().collect();
    }
    let slots = cluster_commands_into_slots(commands);
    let mut out = Vec::with_capacity(commands.len());
    for slot in slots {
        out.extend(slot.cmds);
    }
    out
}

pub fn coalesce_draw_commands(commands: &[DrawCommand]) -> Vec<DrawCommand> {
    if commands.len() <= 1 {
        return commands.to_vec();
    }
    let mut out: Vec<DrawCommand> = Vec::with_capacity(commands.len());
    for cmd in commands {
        let mut merged = false;
        if let Some(last) = out.last_mut() {
            match (last, cmd) {
                (
                    DrawCommand::SolidFill {
                        mesh: prev_mesh,
                        transform: prev_tx,
                        color: prev_color,
                        blend_mode: prev_blend,
                        clip_rect: prev_clip,
                    },
                    DrawCommand::SolidFill {
                        mesh,
                        transform,
                        color,
                        blend_mode,
                        clip_rect,
                    },
                ) => {
                    if *prev_color == *color
                        && *prev_blend == *blend_mode
                        && *prev_clip == *clip_rect
                        && is_2d_affine_transform(prev_tx)
                        && is_2d_affine_transform(transform)
                    {
                        if *prev_tx != Matrix::IDENTITY {
                            for v in &mut prev_mesh.vertices {
                                v.position = transform_vertex_2d(prev_tx, v.position);
                            }
                            *prev_tx = Matrix::IDENTITY;
                        }
                        let base_vertex = prev_mesh.vertices.len() as u32;
                        prev_mesh.vertices.reserve(mesh.vertices.len());
                        if *transform == Matrix::IDENTITY {
                            prev_mesh.vertices.extend_from_slice(&mesh.vertices);
                        } else {
                            for v in &mesh.vertices {
                                prev_mesh.vertices.push(Vertex2D {
                                    position: transform_vertex_2d(transform, v.position),
                                });
                            }
                        }
                        prev_mesh.indices.reserve(mesh.indices.len());
                        for &idx in &mesh.indices {
                            prev_mesh.indices.push(base_vertex + idx);
                        }
                        merged = true;
                    }
                }
                (DrawCommand::GlyphAtlas(prev_batch), DrawCommand::GlyphAtlas(batch)) => {
                    let prev_linear = effective_glyph_linear_filter(prev_batch);
                    let cur_linear = effective_glyph_linear_filter(batch);
                    let prev_uses_alpha = prev_batch.quads.iter().any(|q| !q.is_color);
                    let cur_uses_alpha = batch.quads.iter().any(|q| !q.is_color);
                    let prev_uses_color = prev_batch.quads.iter().any(|q| q.is_color);
                    let cur_uses_color = batch.quads.iter().any(|q| q.is_color);
                    if prev_batch.blend_mode == batch.blend_mode
                        && prev_batch.clip_rect == batch.clip_rect
                        && prev_batch.use_text_color == batch.use_text_color
                        && (prev_batch.text_contrast - batch.text_contrast).abs() < 1e-5
                        && (prev_batch.alpha - batch.alpha).abs() < 1e-5
                        && prev_linear == cur_linear
                        && compatible_atlas_slot(
                            &prev_batch.alpha_atlas,
                            prev_uses_alpha,
                            &batch.alpha_atlas,
                            cur_uses_alpha,
                        )
                        && compatible_atlas_slot(
                            &prev_batch.color_atlas,
                            prev_uses_color,
                            &batch.color_atlas,
                            cur_uses_color,
                        )
                    {
                        if prev_batch.transform == batch.transform {
                            if prev_batch.alpha_atlas.is_none() {
                                prev_batch.alpha_atlas = batch.alpha_atlas.clone();
                            }
                            if prev_batch.color_atlas.is_none() {
                                prev_batch.color_atlas = batch.color_atlas.clone();
                            }
                            prev_batch.quads.extend_from_slice(&batch.quads);
                            merged = true;
                        } else if is_2d_translation_only(&prev_batch.transform)
                            && is_2d_translation_only(&batch.transform)
                        {
                            if prev_batch.transform != Matrix::IDENTITY {
                                let (dx, dy) =
                                    (prev_batch.transform.m[12], prev_batch.transform.m[13]);
                                for q in &mut prev_batch.quads {
                                    q.dest_rect = shift_rect(q.dest_rect, dx, dy);
                                }
                                prev_batch.transform = Matrix::IDENTITY;
                                prev_batch.linear_filter = Some(prev_linear);
                            }
                            if prev_batch.alpha_atlas.is_none() {
                                prev_batch.alpha_atlas = batch.alpha_atlas.clone();
                            }
                            if prev_batch.color_atlas.is_none() {
                                prev_batch.color_atlas = batch.color_atlas.clone();
                            }
                            let (bx, by) = (batch.transform.m[12], batch.transform.m[13]);
                            prev_batch.quads.reserve(batch.quads.len());
                            if bx.abs() < 1e-6 && by.abs() < 1e-6 {
                                prev_batch.quads.extend_from_slice(&batch.quads);
                            } else {
                                for q in &batch.quads {
                                    prev_batch.quads.push(AtlasGlyphQuad {
                                        dest_rect: shift_rect(q.dest_rect, bx, by),
                                        ..*q
                                    });
                                }
                            }
                            merged = true;
                        }
                    }
                }
                _ => {}
            }
        }
        if !merged {
            out.push(cmd.clone());
        }
    }
    out
}

pub struct CachedGpuAtlas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    uploaded_generation: AtomicU64,
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

struct SaveLayerTexture {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[derive(Default)]
struct SaveLayerPool {
    available: Vec<Arc<SaveLayerTexture>>,
}

struct PreparedSubpass {
    target_tex: Arc<SaveLayerTexture>,
    has_stencil: bool,
    calls: Vec<PendingArenaCall>,
}

struct ResolvedSubpass {
    target_tex: Arc<SaveLayerTexture>,
    has_stencil: bool,
    calls: Vec<PendingArenaCall>,
    resolved_bgs: Vec<Arc<wgpu::BindGroup>>,
}

#[derive(Clone)]
enum DrawCallBinding {
    SolidFill,
    RRectBlur,
    LinearGradient,
    RadialGradient,
    SweepGradient,
    ConicalGradient,
    GlyphAtlas {
        alpha_gpu: Arc<CachedGpuAtlas>,
        color_gpu: Arc<CachedGpuAtlas>,
        use_linear: bool,
    },
    TextureFill {
        gpu_tex: Arc<CachedGpuAtlas>,
        linear_filter: bool,
    },
    LayerTextureFill {
        layer_tex: Arc<SaveLayerTexture>,
        linear_filter: bool,
    },
    ImageFilter {
        src_tex: Arc<SaveLayerTexture>,
        linear_filter: bool,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CoalesceTag {
    None,
    SolidFill2D {
        blend_mode: BlendMode,
    },
    GlyphAtlas2D {
        blend_mode: BlendMode,
        use_text_color: bool,
        text_contrast_bits: u32,
        alpha_bits: u32,
        use_linear: bool,
    },
}

struct PendingArenaCall {
    pipeline: Arc<wgpu::RenderPipeline>,
    binding: DrawCallBinding,
    glyph_alpha_opt: Option<Arc<CachedGpuAtlas>>,
    glyph_color_opt: Option<Arc<CachedGpuAtlas>>,
    dynamic_offsets: [u32; 3],
    dynamic_offset_count: usize,
    vertex_offset: u64,
    vertex_byte_len: u64,
    vertex_count: u32,
    index_offset: u64,
    index_byte_len: u64,
    index_count: u32,
    scissor: [u32; 4],
    stencil_ref: u32,
    coalesce_tag: CoalesceTag,
}

type CachedGlyphBgValue = (
    Arc<CachedGpuAtlas>,
    Arc<CachedGpuAtlas>,
    Arc<wgpu::BindGroup>,
);
type CachedTextureBgValue = (Arc<CachedGpuAtlas>, Arc<wgpu::BindGroup>);
type CachedLayerBgValue = (Arc<SaveLayerTexture>, Arc<wgpu::BindGroup>);
type CachedSsaaBgValue = (usize, Arc<SsaaScratchTarget>, Arc<wgpu::BindGroup>);

#[derive(Default)]
struct FrameArena {
    vertex_cpu: Vec<u8>,
    index_cpu: Vec<u8>,
    uniform_cpu: Vec<u8>,
    vertex_gpu: Option<Arc<wgpu::Buffer>>,
    vertex_gpu_capacity: u64,
    index_gpu: Option<Arc<wgpu::Buffer>>,
    index_gpu_capacity: u64,
    uniform_gpu: Option<Arc<wgpu::Buffer>>,
    uniform_gpu_capacity: u64,
    solid_fill_bg: Option<Arc<wgpu::BindGroup>>,
    rrect_blur_bg: Option<Arc<wgpu::BindGroup>>,
    linear_gradient_bg: Option<Arc<wgpu::BindGroup>>,
    radial_gradient_bg: Option<Arc<wgpu::BindGroup>>,
    sweep_gradient_bg: Option<Arc<wgpu::BindGroup>>,
    conical_gradient_bg: Option<Arc<wgpu::BindGroup>>,
    glyph_atlas_bgs: HashMap<(usize, usize, bool), CachedGlyphBgValue>,
    texture_fill_bgs: HashMap<(usize, bool), CachedTextureBgValue>,
    layer_texture_bgs: HashMap<(usize, bool), CachedLayerBgValue>,
    image_filter_bgs: HashMap<(usize, bool), CachedLayerBgValue>,
    ssaa_resolve_bg: Option<CachedSsaaBgValue>,
    seen_atlas_eviction_epoch: u64,
    last_frame_upload_count: usize,
    last_frame_draw_call_count: usize,
    last_frame_bind_groups_created: usize,
}

impl FrameArena {
    fn begin_frame(&mut self, current_atlas_eviction_epoch: u64) {
        self.vertex_cpu.clear();
        self.index_cpu.clear();
        self.uniform_cpu.clear();
        self.last_frame_upload_count = 0;
        self.last_frame_draw_call_count = 0;
        self.last_frame_bind_groups_created = 0;
        if self.seen_atlas_eviction_epoch != current_atlas_eviction_epoch {
            self.glyph_atlas_bgs.clear();
            self.seen_atlas_eviction_epoch = current_atlas_eviction_epoch;
        }
    }

    fn clear_bind_groups(&mut self) {
        self.solid_fill_bg = None;
        self.rrect_blur_bg = None;
        self.linear_gradient_bg = None;
        self.radial_gradient_bg = None;
        self.sweep_gradient_bg = None;
        self.conical_gradient_bg = None;
        self.glyph_atlas_bgs.clear();
        self.texture_fill_bgs.clear();
        self.layer_texture_bgs.clear();
        self.image_filter_bgs.clear();
        self.ssaa_resolve_bg = None;
    }

    fn allocated_gpu_buffer_count(&self) -> usize {
        usize::from(self.vertex_gpu.is_some())
            + usize::from(self.index_gpu.is_some())
            + usize::from(self.uniform_gpu.is_some())
    }

    fn cached_bind_group_count(&self) -> usize {
        usize::from(self.solid_fill_bg.is_some())
            + usize::from(self.rrect_blur_bg.is_some())
            + usize::from(self.linear_gradient_bg.is_some())
            + usize::from(self.radial_gradient_bg.is_some())
            + usize::from(self.sweep_gradient_bg.is_some())
            + usize::from(self.conical_gradient_bg.is_some())
            + self.glyph_atlas_bgs.len()
            + self.texture_fill_bgs.len()
            + self.layer_texture_bgs.len()
            + self.image_filter_bgs.len()
            + usize::from(self.ssaa_resolve_bg.is_some())
    }

    fn push_uniform(&mut self, bytes: &[u8], align: usize) -> u32 {
        let aligned_start = self.uniform_cpu.len().div_ceil(align) * align;
        if aligned_start > self.uniform_cpu.len() {
            self.uniform_cpu.resize(aligned_start, 0);
        }
        let offset = self.uniform_cpu.len() as u32;
        self.uniform_cpu.extend_from_slice(bytes);
        let aligned_end = self.uniform_cpu.len().div_ceil(align) * align;
        self.uniform_cpu.resize(aligned_end, 0);
        offset
    }

    fn align_cpu_4(buf: &mut Vec<u8>) {
        let align = wgpu::COPY_BUFFER_ALIGNMENT as usize;
        let aligned = buf.len().div_ceil(align) * align;
        if aligned > buf.len() {
            buf.resize(aligned, 0);
        }
    }

    fn upload_cpu_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniform_align: usize,
    ) {
        if !self.vertex_cpu.is_empty() {
            Self::align_cpu_4(&mut self.vertex_cpu);
            let needed = self.vertex_cpu.len() as u64;
            if self.vertex_gpu.is_none() || self.vertex_gpu_capacity < needed {
                let cap = needed.max(4096).next_power_of_two();
                self.vertex_gpu = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Impeller FrameArena VertexBuffer"),
                    size: cap,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })));
                self.vertex_gpu_capacity = cap;
            }
            if let Some(buf) = self.vertex_gpu.as_ref() {
                queue.write_buffer(buf, 0, &self.vertex_cpu);
                self.last_frame_upload_count += 1;
            }
        }

        if !self.index_cpu.is_empty() {
            Self::align_cpu_4(&mut self.index_cpu);
            let needed = self.index_cpu.len() as u64;
            if self.index_gpu.is_none() || self.index_gpu_capacity < needed {
                let cap = needed.max(4096).next_power_of_two();
                self.index_gpu = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Impeller FrameArena IndexBuffer"),
                    size: cap,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })));
                self.index_gpu_capacity = cap;
            }
            if let Some(buf) = self.index_gpu.as_ref() {
                queue.write_buffer(buf, 0, &self.index_cpu);
                self.last_frame_upload_count += 1;
            }
        }

        if !self.uniform_cpu.is_empty() {
            let min_uniform_len = uniform_align.max(512);
            let padded_len = self
                .uniform_cpu
                .len()
                .max(min_uniform_len)
                .div_ceil(uniform_align)
                * uniform_align;
            if padded_len > self.uniform_cpu.len() {
                self.uniform_cpu.resize(padded_len, 0);
            }
            let needed = self.uniform_cpu.len() as u64;
            if self.uniform_gpu.is_none() || self.uniform_gpu_capacity < needed {
                let cap = needed.max(16384).next_power_of_two();
                self.uniform_gpu = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Impeller FrameArena UniformBuffer"),
                    size: cap,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })));
                self.uniform_gpu_capacity = cap;
                self.clear_bind_groups();
            }
            if let Some(buf) = self.uniform_gpu.as_ref() {
                queue.write_buffer(buf, 0, &self.uniform_cpu);
                self.last_frame_upload_count += 1;
            }
        }
    }
}

type PipelineCacheKey = (PipelineKind, u32, u32, wgpu::TextureFormat, bool);
type ImageCacheKey = (u64, u32, u32);

struct SsaaScratchTarget {
    width: u32,
    height: u32,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct SurfaceMsaaScratch {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    sample_count: u32,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct StencilScratchTarget {
    width: u32,
    height: u32,
    sample_count: u32,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

pub struct WgpuRenderer {
    pub adapter_name: String,
    pub backend_name: String,
    is_gl_backend: bool,
    _instance: Option<wgpu::Instance>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    solid_shader: wgpu::ShaderModule,
    linear_gradient_shader: wgpu::ShaderModule,
    radial_gradient_shader: wgpu::ShaderModule,
    sweep_gradient_shader: wgpu::ShaderModule,
    conical_gradient_shader: wgpu::ShaderModule,
    rrect_blur_shader: wgpu::ShaderModule,
    glyph_atlas_shader: wgpu::ShaderModule,
    texture_fill_shader: wgpu::ShaderModule,
    stencil_clip_shader: wgpu::ShaderModule,
    image_filter_shader: wgpu::ShaderModule,
    solid_fill_layout: wgpu::BindGroupLayout,
    rrect_blur_layout: wgpu::BindGroupLayout,
    linear_gradient_layout: wgpu::BindGroupLayout,
    radial_gradient_layout: wgpu::BindGroupLayout,
    sweep_gradient_layout: wgpu::BindGroupLayout,
    conical_gradient_layout: wgpu::BindGroupLayout,
    glyph_atlas_layout: wgpu::BindGroupLayout,
    texture_fill_layout: wgpu::BindGroupLayout,
    image_filter_layout: wgpu::BindGroupLayout,
    linear_sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,
    fallback_alpha_atlas: Arc<CachedGpuAtlas>,
    fallback_color_atlas: Arc<CachedGpuAtlas>,
    pipeline_cache: Mutex<HashMap<PipelineCacheKey, Arc<wgpu::RenderPipeline>>>,
    atlas_cache: Mutex<HashMap<AtlasCacheKey, Arc<CachedGpuAtlas>>>,
    atlas_eviction_epoch: AtomicU64,
    image_cache: Mutex<HashMap<ImageCacheKey, Arc<CachedGpuAtlas>>>,
    ssaa_scratch: Mutex<Option<Arc<SsaaScratchTarget>>>,
    surface_msaa_scratch: Mutex<Option<Arc<SurfaceMsaaScratch>>>,
    stencil_scratch: Mutex<Option<Arc<StencilScratchTarget>>>,
    save_layer_pool: Mutex<SaveLayerPool>,
    frame_arena: Mutex<FrameArena>,
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
        let is_gl_backend = info.backend == wgpu::Backend::Gl;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| format!("Failed to request wgpu Device: {e}"))?;

        Ok(Self::from_device_and_queue(
            Some(instance),
            adapter_name,
            backend_name,
            is_gl_backend,
            device,
            queue,
        ))
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn new_for_canvas(
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(Self, wgpu::Surface<'static>, wgpu::SurfaceConfiguration), String> {
        Self::new_for_canvas_with_backend(canvas, true).await
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn new_for_canvas_with_backend(
        canvas: web_sys::HtmlCanvasElement,
        prefer_webgpu: bool,
    ) -> Result<(Self, wgpu::Surface<'static>, wgpu::SurfaceConfiguration), String> {
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        if prefer_webgpu {
            let webgpu_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                ..Default::default()
            });
            if let Ok(surface) =
                webgpu_instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            {
                if let Ok(ok) =
                    Self::init_with_surface(webgpu_instance, surface, width, height, false).await
                {
                    return Ok(ok);
                }
            }
        }

        let gl_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = gl_instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("Failed to create WebGL2 canvas surface: {e}"))?;
        Self::init_with_surface(gl_instance, surface, width, height, true).await
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn new_for_offscreen_canvas(
        canvas: web_sys::OffscreenCanvas,
    ) -> Result<(Self, wgpu::Surface<'static>, wgpu::SurfaceConfiguration), String> {
        Self::new_for_offscreen_canvas_with_backend(canvas, true).await
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn new_for_offscreen_canvas_with_backend(
        canvas: web_sys::OffscreenCanvas,
        prefer_webgpu: bool,
    ) -> Result<(Self, wgpu::Surface<'static>, wgpu::SurfaceConfiguration), String> {
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        if prefer_webgpu {
            let webgpu_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                ..Default::default()
            });
            if let Ok(surface) =
                webgpu_instance.create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas.clone()))
            {
                if let Ok(ok) =
                    Self::init_with_surface(webgpu_instance, surface, width, height, false).await
                {
                    return Ok(ok);
                }
            }
        }

        let gl_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = gl_instance
            .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas))
            .map_err(|e| format!("Failed to create WebGL2 offscreen canvas surface: {e}"))?;
        Self::init_with_surface(gl_instance, surface, width, height, true).await
    }

    #[cfg(target_arch = "wasm32")]
    async fn init_with_surface(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
        is_webgl2: bool,
    ) -> Result<(Self, wgpu::Surface<'static>, wgpu::SurfaceConfiguration), String> {
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
        {
            Some(a) => a,
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::None,
                    force_fallback_adapter: true,
                    compatible_surface: Some(&surface),
                })
                .await
                .ok_or_else(|| "Failed to request WebGPU/WebGL2 adapter for canvas".to_string())?,
        };

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);

        let required_limits = if is_webgl2 || info.backend == wgpu::Backend::Gl {
            adapter.limits()
        } else {
            wgpu::Limits::default().using_resolution(adapter.limits())
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to request WebGPU/WebGL2 device: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let mut config = surface
            .get_default_config(&adapter, width.max(1), height.max(1))
            .ok_or_else(|| "Surface incompatible with adapter".to_string())?;
        if caps.formats.contains(&wgpu::TextureFormat::Rgba8Unorm) {
            config.format = wgpu::TextureFormat::Rgba8Unorm;
        } else if caps.formats.contains(&wgpu::TextureFormat::Bgra8Unorm) {
            config.format = wgpu::TextureFormat::Bgra8Unorm;
        }
        surface.configure(&device, &config);

        let is_gl_backend = is_webgl2 || info.backend == wgpu::Backend::Gl;
        let ctx = Self::from_device_and_queue(
            Some(instance),
            adapter_name,
            backend_name,
            is_gl_backend,
            device,
            queue,
        );
        Ok((ctx, surface, config))
    }

    fn from_device_and_queue(
        instance: Option<wgpu::Instance>,
        adapter_name: String,
        backend_name: String,
        is_gl_backend: bool,
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
        let sweep_gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller sweep_gradient_fill.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SWEEP_GRADIENT_FILL_WGSL.into()),
        });
        let conical_gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller conical_gradient_fill.wgsl"),
            source: wgpu::ShaderSource::Wgsl(CONICAL_GRADIENT_FILL_WGSL.into()),
        });
        let rrect_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller rrect_blur.wgsl"),
            source: wgpu::ShaderSource::Wgsl(RRECT_BLUR_WGSL.into()),
        });
        let glyph_atlas_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller glyph_atlas.wgsl"),
            source: wgpu::ShaderSource::Wgsl(GLYPH_ATLAS_WGSL.into()),
        });
        let texture_fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller texture_fill.wgsl"),
            source: wgpu::ShaderSource::Wgsl(TEXTURE_FILL_WGSL.into()),
        });
        let stencil_clip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller stencil_clip.wgsl"),
            source: wgpu::ShaderSource::Wgsl(STENCIL_CLIP_WGSL.into()),
        });
        let image_filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Impeller image_filter.wgsl"),
            source: wgpu::ShaderSource::Wgsl(IMAGE_FILTER_WGSL.into()),
        });

        let dyn_uniform_entry = |binding: u32,
                                 visibility: wgpu::ShaderStages,
                                 byte_size: usize|
         -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(byte_size as u64),
                },
                count: None,
            }
        };

        let solid_fill_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Impeller SolidFill BindGroupLayout"),
            entries: &[
                dyn_uniform_entry(
                    0,
                    wgpu::ShaderStages::VERTEX,
                    std::mem::size_of::<SolidFrameUniform>(),
                ),
                dyn_uniform_entry(
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    std::mem::size_of::<SolidFragUniform>(),
                ),
            ],
        });

        let rrect_blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Impeller RRectBlur BindGroupLayout"),
            entries: &[
                dyn_uniform_entry(
                    0,
                    wgpu::ShaderStages::VERTEX,
                    std::mem::size_of::<SolidFrameUniform>(),
                ),
                dyn_uniform_entry(
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    std::mem::size_of::<RRectBlurFragUniform>(),
                ),
            ],
        });

        let linear_gradient_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Impeller LinearGradient BindGroupLayout"),
                entries: &[
                    dyn_uniform_entry(
                        0,
                        wgpu::ShaderStages::VERTEX,
                        std::mem::size_of::<GradientFrameUniform>(),
                    ),
                    dyn_uniform_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<LinearGradientFragUniform>(),
                    ),
                    dyn_uniform_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<[GpuColorPoint; MAX_GRADIENT_STOPS]>(),
                    ),
                ],
            });

        let radial_gradient_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Impeller RadialGradient BindGroupLayout"),
                entries: &[
                    dyn_uniform_entry(
                        0,
                        wgpu::ShaderStages::VERTEX,
                        std::mem::size_of::<GradientFrameUniform>(),
                    ),
                    dyn_uniform_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<RadialGradientFragUniform>(),
                    ),
                    dyn_uniform_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<[GpuColorPoint; MAX_GRADIENT_STOPS]>(),
                    ),
                ],
            });

        let sweep_gradient_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Impeller SweepGradient BindGroupLayout"),
                entries: &[
                    dyn_uniform_entry(
                        0,
                        wgpu::ShaderStages::VERTEX,
                        std::mem::size_of::<GradientFrameUniform>(),
                    ),
                    dyn_uniform_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<SweepGradientFragUniform>(),
                    ),
                    dyn_uniform_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<[GpuColorPoint; MAX_GRADIENT_STOPS]>(),
                    ),
                ],
            });

        let conical_gradient_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Impeller ConicalGradient BindGroupLayout"),
                entries: &[
                    dyn_uniform_entry(
                        0,
                        wgpu::ShaderStages::VERTEX,
                        std::mem::size_of::<GradientFrameUniform>(),
                    ),
                    dyn_uniform_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<ConicalGradientFragUniform>(),
                    ),
                    dyn_uniform_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<[GpuColorPoint; MAX_GRADIENT_STOPS]>(),
                    ),
                ],
            });

        let glyph_atlas_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Impeller GlyphAtlas BindGroupLayout"),
                entries: &[
                    dyn_uniform_entry(
                        0,
                        wgpu::ShaderStages::VERTEX,
                        std::mem::size_of::<SolidFrameUniform>(),
                    ),
                    dyn_uniform_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<GlyphAtlasFragUniform>(),
                    ),
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

        let texture_fill_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Impeller TextureFill BindGroupLayout"),
                entries: &[
                    dyn_uniform_entry(
                        0,
                        wgpu::ShaderStages::VERTEX,
                        std::mem::size_of::<SolidFrameUniform>(),
                    ),
                    dyn_uniform_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<TextureFragUniform>(),
                    ),
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let image_filter_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Impeller ImageFilter BindGroupLayout"),
                entries: &[
                    dyn_uniform_entry(
                        0,
                        wgpu::ShaderStages::VERTEX,
                        std::mem::size_of::<SolidFrameUniform>(),
                    ),
                    dyn_uniform_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        std::mem::size_of::<ImageFilterFragUniform>(),
                    ),
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
            1,
        ));
        let fallback_color_atlas = Arc::new(Self::upload_atlas_raw(
            &device,
            &queue,
            GlyphAtlasType::ColorBitmap,
            1,
            1,
            &[255u8, 255, 255, 255],
            1,
        ));

        Self {
            adapter_name,
            backend_name,
            is_gl_backend,
            _instance: instance,
            device,
            queue,
            solid_shader,
            linear_gradient_shader,
            radial_gradient_shader,
            sweep_gradient_shader,
            conical_gradient_shader,
            rrect_blur_shader,
            glyph_atlas_shader,
            texture_fill_shader,
            stencil_clip_shader,
            image_filter_shader,
            solid_fill_layout,
            rrect_blur_layout,
            linear_gradient_layout,
            radial_gradient_layout,
            sweep_gradient_layout,
            conical_gradient_layout,
            glyph_atlas_layout,
            texture_fill_layout,
            image_filter_layout,
            linear_sampler,
            nearest_sampler,
            fallback_alpha_atlas,
            fallback_color_atlas,
            pipeline_cache: Mutex::new(HashMap::new()),
            atlas_cache: Mutex::new(HashMap::new()),
            atlas_eviction_epoch: AtomicU64::new(0),
            image_cache: Mutex::new(HashMap::new()),
            ssaa_scratch: Mutex::new(None),
            surface_msaa_scratch: Mutex::new(None),
            stencil_scratch: Mutex::new(None),
            save_layer_pool: Mutex::new(SaveLayerPool::default()),
            frame_arena: Mutex::new(FrameArena::default()),
        }
    }

    pub fn is_gl_backend(&self) -> bool {
        self.is_gl_backend
    }

    pub fn atlas_eviction_epoch(&self) -> u64 {
        self.atlas_eviction_epoch.load(Ordering::Relaxed)
    }

    fn upload_atlas_raw(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas_type: GlyphAtlasType,
        width: u32,
        height: u32,
        pixels: &[u8],
        initial_generation: u64,
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

        if !pixels.is_empty() {
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
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        CachedGpuAtlas {
            texture,
            view,
            width: w,
            height: h,
            uploaded_generation: AtomicU64::new(initial_generation),
        }
    }

    pub fn sync_glyph_atlas(&self, atlas: &GlyphAtlas) -> Arc<CachedGpuAtlas> {
        let key = AtlasCacheKey::from_atlas(atlas);
        let existing_opt = self
            .atlas_cache
            .lock()
            .ok()
            .and_then(|cache| cache.get(&key).cloned());

        if let Some(existing) = existing_opt {
            let prev_gen = existing.uploaded_generation.load(Ordering::Relaxed);
            if atlas.generation > prev_gen && !atlas.pixels.is_empty() {
                let (raw_y_min, raw_y_max) = atlas
                    .dirty_row_range_since(prev_gen)
                    .unwrap_or((0, existing.height));
                let y_min = raw_y_min.min(existing.height);
                let y_max = raw_y_max.min(existing.height);
                if y_max > y_min {
                    let bpp = match atlas.atlas_type {
                        GlyphAtlasType::AlphaBitmap => 1u32,
                        GlyphAtlasType::ColorBitmap => 4u32,
                    };
                    let row_bytes = existing.width * bpp;
                    let byte_start = (y_min * row_bytes) as usize;
                    let byte_end = (y_max * row_bytes) as usize;
                    if byte_end <= atlas.pixels.len() {
                        self.queue.write_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: &existing.texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d {
                                    x: 0,
                                    y: y_min,
                                    z: 0,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            &atlas.pixels[byte_start..byte_end],
                            wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(row_bytes),
                                rows_per_image: Some(y_max - y_min),
                            },
                            wgpu::Extent3d {
                                width: existing.width,
                                height: y_max - y_min,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                }
                existing
                    .uploaded_generation
                    .store(atlas.generation, Ordering::Relaxed);
            }
            return existing;
        }

        let initial_gen = if atlas.pixels.is_empty() {
            0
        } else {
            atlas.generation
        };
        let uploaded = Arc::new(Self::upload_atlas_raw(
            &self.device,
            &self.queue,
            atlas.atlas_type,
            key.width,
            key.height,
            &atlas.pixels,
            initial_gen,
        ));

        if let Ok(mut cache) = self.atlas_cache.lock() {
            if key.atlas_id > 0 && key.resize_epoch > 0 {
                let before = cache.len();
                cache.retain(|k, _| {
                    !(k.atlas_id == key.atlas_id && k.resize_epoch < key.resize_epoch)
                });
                if cache.len() < before {
                    self.atlas_eviction_epoch.fetch_add(1, Ordering::Relaxed);
                }
            }
            cache.insert(key, Arc::clone(&uploaded));
        }
        uploaded
    }

    fn get_or_upload_atlas(&self, atlas: &GlyphAtlas) -> Arc<CachedGpuAtlas> {
        self.sync_glyph_atlas(atlas)
    }

    pub fn cached_atlas_count(&self) -> usize {
        self.atlas_cache
            .lock()
            .map(|cache| cache.len())
            .unwrap_or(0)
    }

    fn get_or_upload_image_texture(&self, image: &ImageTexture) -> Arc<CachedGpuAtlas> {
        let key = (image.id, image.width.max(1), image.height.max(1));
        if let Ok(cache) = self.image_cache.lock() {
            if let Some(existing) = cache.get(&key) {
                return Arc::clone(existing);
            }
        }

        let uploaded = Arc::new(Self::upload_atlas_raw(
            &self.device,
            &self.queue,
            GlyphAtlasType::ColorBitmap,
            key.1,
            key.2,
            &image.rgba_premultiplied,
            1,
        ));

        if let Ok(mut cache) = self.image_cache.lock() {
            cache.insert(key, Arc::clone(&uploaded));
        }
        uploaded
    }

    pub fn cached_image_count(&self) -> usize {
        self.image_cache
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
        self.get_or_create_pipeline_with_samples_and_format(
            kind,
            blend_mode,
            DEFAULT_SAMPLE_COUNT,
            TARGET_FORMAT,
        )
    }

    pub fn get_or_create_pipeline_with_samples(
        &self,
        kind: PipelineKind,
        blend_mode: BlendMode,
        sample_count: u32,
    ) -> Arc<wgpu::RenderPipeline> {
        self.get_or_create_pipeline_with_samples_and_format(
            kind,
            blend_mode,
            sample_count,
            TARGET_FORMAT,
        )
    }

    pub fn get_or_create_pipeline_with_samples_and_format(
        &self,
        kind: PipelineKind,
        blend_mode: BlendMode,
        sample_count: u32,
        format: wgpu::TextureFormat,
    ) -> Arc<wgpu::RenderPipeline> {
        let is_stencil_clip = matches!(
            kind,
            PipelineKind::StencilIncrement | PipelineKind::StencilDecrement
        );
        self.get_or_create_pipeline_full(kind, blend_mode, sample_count, format, is_stencil_clip)
    }

    pub fn get_or_create_pipeline_full(
        &self,
        kind: PipelineKind,
        blend_mode: BlendMode,
        sample_count: u32,
        format: wgpu::TextureFormat,
        has_stencil: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let samples = normalize_sample_count(sample_count);
        let is_stencil_clip = matches!(
            kind,
            PipelineKind::StencilIncrement | PipelineKind::StencilDecrement
        );
        let effective_stencil = has_stencil || is_stencil_clip;
        let effective_blend = if is_stencil_clip {
            BlendMode::SrcOver
        } else {
            blend_mode
        };
        let key = (
            kind,
            effective_blend as u32,
            samples,
            format,
            effective_stencil,
        );
        if let Ok(cache) = self.pipeline_cache.lock() {
            if let Some(existing) = cache.get(&key) {
                return Arc::clone(existing);
            }
        }

        let (shader, bgl, label) = match kind {
            PipelineKind::SolidFill => (
                &self.solid_shader,
                &self.solid_fill_layout,
                "Impeller SolidFill Pipeline",
            ),
            PipelineKind::LinearGradientFill => (
                &self.linear_gradient_shader,
                &self.linear_gradient_layout,
                "Impeller LinearGradientFill Pipeline",
            ),
            PipelineKind::RadialGradientFill => (
                &self.radial_gradient_shader,
                &self.radial_gradient_layout,
                "Impeller RadialGradientFill Pipeline",
            ),
            PipelineKind::SweepGradientFill => (
                &self.sweep_gradient_shader,
                &self.sweep_gradient_layout,
                "Impeller SweepGradientFill Pipeline",
            ),
            PipelineKind::ConicalGradientFill => (
                &self.conical_gradient_shader,
                &self.conical_gradient_layout,
                "Impeller ConicalGradientFill Pipeline",
            ),
            PipelineKind::RRectBlur => (
                &self.rrect_blur_shader,
                &self.rrect_blur_layout,
                "Impeller RRectBlur Pipeline",
            ),
            PipelineKind::GlyphAtlas => (
                &self.glyph_atlas_shader,
                &self.glyph_atlas_layout,
                "Impeller GlyphAtlas Pipeline",
            ),
            PipelineKind::TextureFill => (
                &self.texture_fill_shader,
                &self.texture_fill_layout,
                "Impeller TextureFill Pipeline",
            ),
            PipelineKind::StencilIncrement => (
                &self.stencil_clip_shader,
                &self.solid_fill_layout,
                "Impeller StencilIncrement Pipeline",
            ),
            PipelineKind::StencilDecrement => (
                &self.stencil_clip_shader,
                &self.solid_fill_layout,
                "Impeller StencilDecrement Pipeline",
            ),
            PipelineKind::ImageFilter => (
                &self.image_filter_shader,
                &self.image_filter_layout,
                "Impeller ImageFilter Pipeline",
            ),
        };

        let pipeline_layout = self
            .device
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
        let colored_solid_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 8,
                shader_location: 1,
            },
        ];
        let uv_attrs = [
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
        ];
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

        let vertex_layout = match kind {
            PipelineKind::SolidFill => wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ColoredSolidVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &colored_solid_attrs,
            },
            PipelineKind::GlyphAtlas => wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GlyphVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &glyph_attrs,
            },
            PipelineKind::TextureFill | PipelineKind::ImageFilter => wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<UvVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &uv_attrs,
            },
            _ => wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SolidVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &solid_attrs,
            },
        };

        let color_target = if is_stencil_clip {
            wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            }
        } else {
            wgpu::ColorTargetState {
                format,
                blend: Some(blend_state_for_mode(effective_blend)),
                write_mask: wgpu::ColorWrites::ALL,
            }
        };

        let depth_stencil = if effective_stencil {
            let (pass_op, write_mask) = match kind {
                PipelineKind::StencilIncrement => (wgpu::StencilOperation::IncrementClamp, 0xFF),
                PipelineKind::StencilDecrement => (wgpu::StencilOperation::DecrementClamp, 0xFF),
                _ => (wgpu::StencilOperation::Keep, 0x00),
            };
            let stencil_face = wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op,
            };
            Some(wgpu::DepthStencilState {
                format: STENCIL_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState {
                    front: stencil_face,
                    back: stencil_face,
                    read_mask: 0xFF,
                    write_mask,
                },
                bias: wgpu::DepthBiasState::default(),
            })
        } else {
            None
        };

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
                    targets: &[Some(color_target)],
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
                depth_stencil,
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

    fn get_or_create_stencil_scratch(
        &self,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Arc<StencilScratchTarget> {
        let w = width.max(1);
        let h = height.max(1);
        let samples = normalize_sample_count(sample_count);
        if let Ok(guard) = self.stencil_scratch.lock() {
            if let Some(existing) = guard.as_ref() {
                if existing.width == w && existing.height == h && existing.sample_count == samples {
                    return Arc::clone(existing);
                }
            }
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Impeller Depth24PlusStencil8 Scratch Texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format: STENCIL_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let created = Arc::new(StencilScratchTarget {
            width: w,
            height: h,
            sample_count: samples,
            _texture: texture,
            view,
        });

        if let Ok(mut guard) = self.stencil_scratch.lock() {
            *guard = Some(Arc::clone(&created));
        }
        created
    }

    fn acquire_save_layer_texture(
        &self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Arc<SaveLayerTexture> {
        let w = width.max(1);
        let h = height.max(1);
        if let Ok(mut pool) = self.save_layer_pool.lock() {
            if let Some(pos) = pool
                .available
                .iter()
                .position(|t| t.width == w && t.height == h && t.format == format)
            {
                return pool.available.swap_remove(pos);
            }
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Impeller SaveLayer Offscreen Texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Arc::new(SaveLayerTexture {
            width: w,
            height: h,
            format,
            _texture: texture,
            view,
        })
    }

    fn release_save_layer_textures(&self, textures: Vec<Arc<SaveLayerTexture>>) {
        if textures.is_empty() {
            return;
        }
        if let Ok(mut pool) = self.save_layer_pool.lock() {
            for tex in textures {
                if pool.available.len() >= 32 {
                    break;
                }
                pool.available.push(tex);
            }
        }
    }

    pub fn pooled_save_layer_texture_count(&self) -> usize {
        self.save_layer_pool
            .lock()
            .map(|p| p.available.len())
            .unwrap_or(0)
    }

    fn compute_scissor_rect(
        clip_rect: Option<Rect>,
        internal_width: u32,
        internal_height: u32,
        ssaa_scale: u32,
        layer_origin: Point,
    ) -> Option<[u32; 4]> {
        match clip_rect {
            None => Some([0, 0, internal_width, internal_height]),
            Some(r) => {
                if r.is_empty() {
                    return None;
                }
                let s = ssaa_scale.max(1) as Scalar;
                let min_x = ((r.get_left() - layer_origin.x) * s).floor().max(0.0) as u32;
                let min_y = ((r.get_top() - layer_origin.y) * s).floor().max(0.0) as u32;
                let max_x = ((r.get_right() - layer_origin.x) * s)
                    .ceil()
                    .max(0.0)
                    .min(internal_width as Scalar) as u32;
                let max_y = ((r.get_bottom() - layer_origin.y) * s)
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

    pub fn pooled_buffer_count(&self) -> usize {
        self.frame_arena
            .lock()
            .map(|arena| arena.allocated_gpu_buffer_count())
            .unwrap_or(0)
    }

    pub fn last_frame_draw_call_count(&self) -> usize {
        self.frame_arena
            .lock()
            .map(|arena| arena.last_frame_draw_call_count)
            .unwrap_or(0)
    }

    pub fn last_frame_buffer_upload_count(&self) -> usize {
        self.frame_arena
            .lock()
            .map(|arena| arena.last_frame_upload_count)
            .unwrap_or(0)
    }

    pub fn cached_bind_group_count(&self) -> usize {
        self.frame_arena
            .lock()
            .map(|arena| arena.cached_bind_group_count())
            .unwrap_or(0)
    }

    pub fn last_frame_bind_groups_created(&self) -> usize {
        self.frame_arena
            .lock()
            .map(|arena| arena.last_frame_bind_groups_created)
            .unwrap_or(0)
    }

    fn uniform_offset_alignment(&self) -> usize {
        (self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(256)) as usize
    }

    fn uniform_binding(buffer: &wgpu::Buffer, byte_len: usize) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(byte_len as u64),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn push_stencil_call(
        &self,
        arena: &mut FrameArena,
        calls: &mut Vec<PendingArenaCall>,
        kind: PipelineKind,
        stencil_ref: u32,
        scissor: [u32; 4],
        frame_off: u32,
        frag_off: u32,
        vertex_bytes: &[u8],
        vertex_count: u32,
        indices: &[u32],
        sample_count: u32,
        format: wgpu::TextureFormat,
    ) {
        let pipeline =
            self.get_or_create_pipeline_full(kind, BlendMode::SrcOver, sample_count, format, true);
        FrameArena::align_cpu_4(&mut arena.vertex_cpu);
        FrameArena::align_cpu_4(&mut arena.index_cpu);
        let vertex_offset = arena.vertex_cpu.len() as u64;
        arena.vertex_cpu.extend_from_slice(vertex_bytes);
        let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;

        let index_offset = arena.index_cpu.len() as u64;
        arena
            .index_cpu
            .extend_from_slice(bytemuck::cast_slice(indices));
        let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

        calls.push(PendingArenaCall {
            pipeline,
            binding: DrawCallBinding::SolidFill,
            glyph_alpha_opt: None,
            glyph_color_opt: None,
            dynamic_offsets: [frame_off, frag_off, 0],
            dynamic_offset_count: 2,
            vertex_offset,
            vertex_byte_len,
            vertex_count,
            index_offset,
            index_byte_len,
            index_count: indices.len() as u32,
            scissor,
            stencil_ref,
            coalesce_tag: CoalesceTag::None,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn push_image_filter_subpass(
        &self,
        arena: &mut FrameArena,
        subpasses: &mut Vec<PreparedSubpass>,
        used_layer_textures: &mut Vec<Arc<SaveLayerTexture>>,
        src_tex: Arc<SaveLayerTexture>,
        frag_uniform: ImageFilterFragUniform,
        linear_filter: bool,
        format: wgpu::TextureFormat,
        uniform_align: usize,
    ) -> Arc<SaveLayerTexture> {
        let dst_tex = self.acquire_save_layer_texture(src_tex.width, src_tex.height, format);
        used_layer_textures.push(Arc::clone(&dst_tex));

        let w = dst_tex.width as Scalar;
        let h = dst_tex.height as Scalar;
        let ortho = Matrix::make_orthographic(Size::new(w, h));
        let frame_u = SolidFrameUniform { mvp: ortho.m };
        let frame_off = arena.push_uniform(bytemuck::bytes_of(&frame_u), uniform_align);
        let frag_off = arena.push_uniform(bytemuck::bytes_of(&frag_uniform), uniform_align);

        let verts = [
            UvVertex::new(0.0, 0.0, 0.0, 0.0),
            UvVertex::new(w, 0.0, 1.0, 0.0),
            UvVertex::new(w, h, 1.0, 1.0),
            UvVertex::new(0.0, h, 0.0, 1.0),
        ];
        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
        FrameArena::align_cpu_4(&mut arena.vertex_cpu);
        FrameArena::align_cpu_4(&mut arena.index_cpu);
        let vertex_offset = arena.vertex_cpu.len() as u64;
        arena
            .vertex_cpu
            .extend_from_slice(bytemuck::cast_slice(&verts));
        let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;
        let index_offset = arena.index_cpu.len() as u64;
        arena
            .index_cpu
            .extend_from_slice(bytemuck::cast_slice(&indices));
        let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

        let pipeline = self.get_or_create_pipeline_full(
            PipelineKind::ImageFilter,
            BlendMode::Src,
            1,
            format,
            false,
        );
        let call = PendingArenaCall {
            pipeline,
            binding: DrawCallBinding::ImageFilter {
                src_tex,
                linear_filter,
            },
            glyph_alpha_opt: None,
            glyph_color_opt: None,
            dynamic_offsets: [frame_off, frag_off, 0],
            dynamic_offset_count: 2,
            vertex_offset,
            vertex_byte_len,
            vertex_count: 4,
            index_offset,
            index_byte_len,
            index_count: 6,
            scissor: [0, 0, dst_tex.width, dst_tex.height],
            stencil_ref: 0,
            coalesce_tag: CoalesceTag::None,
        };
        subpasses.push(PreparedSubpass {
            target_tex: Arc::clone(&dst_tex),
            has_stencil: false,
            calls: vec![call],
        });
        dst_tex
    }

    #[allow(clippy::too_many_arguments)]
    fn bake_layer_transform_subpass(
        &self,
        arena: &mut FrameArena,
        subpasses: &mut Vec<PreparedSubpass>,
        used_layer_textures: &mut Vec<Arc<SaveLayerTexture>>,
        src_tex: Arc<SaveLayerTexture>,
        child_origin: Point,
        ssaa_scalar: Scalar,
        transform: Matrix,
        format: wgpu::TextureFormat,
        uniform_align: usize,
    ) -> Arc<SaveLayerTexture> {
        let dst_tex = self.acquire_save_layer_texture(src_tex.width, src_tex.height, format);
        used_layer_textures.push(Arc::clone(&dst_tex));

        let px_w = dst_tex.width as Scalar;
        let px_h = dst_tex.height as Scalar;
        let logical_w = px_w / ssaa_scalar;
        let logical_h = px_h / ssaa_scalar;
        let child_ortho = Matrix::make_orthographic(Size::new(px_w, px_h))
            * Matrix::make_scale(ssaa_scalar, ssaa_scalar, 1.0)
            * Matrix::make_translation(-child_origin.x, -child_origin.y, 0.0);
        let mvp = child_ortho * transform;
        let frame_u = SolidFrameUniform { mvp: mvp.m };
        let frag_u = TextureFragUniform {
            alpha: 1.0,
            tile_mode_x: TileMode::Decal as u32,
            tile_mode_y: TileMode::Decal as u32,
            _pad: 0,
        };
        let frame_off = arena.push_uniform(bytemuck::bytes_of(&frame_u), uniform_align);
        let frag_off = arena.push_uniform(bytemuck::bytes_of(&frag_u), uniform_align);

        let x0 = child_origin.x;
        let y0 = child_origin.y;
        let x1 = x0 + logical_w;
        let y1 = y0 + logical_h;
        let verts = [
            UvVertex::new(x0, y0, 0.0, 0.0),
            UvVertex::new(x1, y0, 1.0, 0.0),
            UvVertex::new(x1, y1, 1.0, 1.0),
            UvVertex::new(x0, y1, 0.0, 1.0),
        ];
        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
        FrameArena::align_cpu_4(&mut arena.vertex_cpu);
        FrameArena::align_cpu_4(&mut arena.index_cpu);
        let vertex_offset = arena.vertex_cpu.len() as u64;
        arena
            .vertex_cpu
            .extend_from_slice(bytemuck::cast_slice(&verts));
        let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;
        let index_offset = arena.index_cpu.len() as u64;
        arena
            .index_cpu
            .extend_from_slice(bytemuck::cast_slice(&indices));
        let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

        let pipeline = self.get_or_create_pipeline_full(
            PipelineKind::TextureFill,
            BlendMode::Src,
            1,
            format,
            false,
        );
        let call = PendingArenaCall {
            pipeline,
            binding: DrawCallBinding::LayerTextureFill {
                layer_tex: src_tex,
                linear_filter: !transform.is_translation_scale_only(),
            },
            glyph_alpha_opt: None,
            glyph_color_opt: None,
            dynamic_offsets: [frame_off, frag_off, 0],
            dynamic_offset_count: 2,
            vertex_offset,
            vertex_byte_len,
            vertex_count: 4,
            index_offset,
            index_byte_len,
            index_count: 6,
            scissor: [0, 0, dst_tex.width, dst_tex.height],
            stencil_ref: 0,
            coalesce_tag: CoalesceTag::None,
        };
        subpasses.push(PreparedSubpass {
            target_tex: Arc::clone(&dst_tex),
            has_stencil: false,
            calls: vec![call],
        });
        dst_tex
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_color_filter_passes(
        &self,
        arena: &mut FrameArena,
        subpasses: &mut Vec<PreparedSubpass>,
        used_layer_textures: &mut Vec<Arc<SaveLayerTexture>>,
        src_tex: Arc<SaveLayerTexture>,
        filter: &LayerColorFilter,
        format: wgpu::TextureFormat,
        uniform_align: usize,
    ) -> Arc<SaveLayerTexture> {
        match filter {
            LayerColorFilter::Mode(color, mode) => {
                let frag = ImageFilterFragUniform {
                    mode_color_pm: color.to_array_premultiplied(),
                    tex_size: [src_tex.width as f32, src_tex.height as f32],
                    alpha: 1.0,
                    filter_kind: FILTER_KIND_COLOR_MODE,
                    tile_mode: TileMode::Clamp as u32,
                    blend_mode: *mode as u32,
                    ..Default::default()
                };
                self.push_image_filter_subpass(
                    arena,
                    subpasses,
                    used_layer_textures,
                    src_tex,
                    frag,
                    false,
                    format,
                    uniform_align,
                )
            }
            LayerColorFilter::Matrix(m) => {
                let frag = ImageFilterFragUniform {
                    color_matrix_row0: [m[0], m[1], m[2], m[3]],
                    color_matrix_row1: [m[5], m[6], m[7], m[8]],
                    color_matrix_row2: [m[10], m[11], m[12], m[13]],
                    color_matrix_row3: [m[15], m[16], m[17], m[18]],
                    color_matrix_offset: [m[4], m[9], m[14], m[19]],
                    tex_size: [src_tex.width as f32, src_tex.height as f32],
                    alpha: 1.0,
                    filter_kind: FILTER_KIND_COLOR_MATRIX,
                    tile_mode: TileMode::Clamp as u32,
                    ..Default::default()
                };
                self.push_image_filter_subpass(
                    arena,
                    subpasses,
                    used_layer_textures,
                    src_tex,
                    frag,
                    false,
                    format,
                    uniform_align,
                )
            }
            LayerColorFilter::SrgbToLinear => {
                let frag = ImageFilterFragUniform {
                    tex_size: [src_tex.width as f32, src_tex.height as f32],
                    alpha: 1.0,
                    filter_kind: FILTER_KIND_SRGB_TO_LINEAR,
                    tile_mode: TileMode::Clamp as u32,
                    ..Default::default()
                };
                self.push_image_filter_subpass(
                    arena,
                    subpasses,
                    used_layer_textures,
                    src_tex,
                    frag,
                    false,
                    format,
                    uniform_align,
                )
            }
            LayerColorFilter::LinearToSrgb => {
                let frag = ImageFilterFragUniform {
                    tex_size: [src_tex.width as f32, src_tex.height as f32],
                    alpha: 1.0,
                    filter_kind: FILTER_KIND_LINEAR_TO_SRGB,
                    tile_mode: TileMode::Clamp as u32,
                    ..Default::default()
                };
                self.push_image_filter_subpass(
                    arena,
                    subpasses,
                    used_layer_textures,
                    src_tex,
                    frag,
                    false,
                    format,
                    uniform_align,
                )
            }
            LayerColorFilter::Compose(outer, inner) => {
                let mid = self.prepare_color_filter_passes(
                    arena,
                    subpasses,
                    used_layer_textures,
                    src_tex,
                    inner,
                    format,
                    uniform_align,
                );
                self.prepare_color_filter_passes(
                    arena,
                    subpasses,
                    used_layer_textures,
                    mid,
                    outer,
                    format,
                    uniform_align,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_image_filter_passes(
        &self,
        arena: &mut FrameArena,
        subpasses: &mut Vec<PreparedSubpass>,
        used_layer_textures: &mut Vec<Arc<SaveLayerTexture>>,
        mut cur_tex: Arc<SaveLayerTexture>,
        filter: &LayerImageFilter,
        child_origin: Point,
        ssaa_scalar: Scalar,
        format: wgpu::TextureFormat,
        uniform_align: usize,
        post_transform: &mut Matrix,
    ) -> Arc<SaveLayerTexture> {
        match filter {
            LayerImageFilter::Matrix(m) => {
                *post_transform = *m * *post_transform;
                cur_tex
            }
            LayerImageFilter::ColorFilter(cf) => {
                if *post_transform != Matrix::IDENTITY {
                    cur_tex = self.bake_layer_transform_subpass(
                        arena,
                        subpasses,
                        used_layer_textures,
                        cur_tex,
                        child_origin,
                        ssaa_scalar,
                        *post_transform,
                        format,
                        uniform_align,
                    );
                    *post_transform = Matrix::IDENTITY;
                }
                self.prepare_color_filter_passes(
                    arena,
                    subpasses,
                    used_layer_textures,
                    cur_tex,
                    cf,
                    format,
                    uniform_align,
                )
            }
            LayerImageFilter::Blur {
                sigma_x,
                sigma_y,
                tile_mode,
            } => {
                if *post_transform != Matrix::IDENTITY {
                    cur_tex = self.bake_layer_transform_subpass(
                        arena,
                        subpasses,
                        used_layer_textures,
                        cur_tex,
                        child_origin,
                        ssaa_scalar,
                        *post_transform,
                        format,
                        uniform_align,
                    );
                    *post_transform = Matrix::IDENTITY;
                }
                let sx = (sigma_x * ssaa_scalar).max(0.0);
                let sy = (sigma_y * ssaa_scalar).max(0.0);
                if sx > 1.0e-4 {
                    let rx = (sx * 3.0).ceil().clamp(1.0, 64.0) as i32;
                    let frag_x = ImageFilterFragUniform {
                        step_uv: [1.0 / (cur_tex.width as f32), 0.0],
                        tex_size: [cur_tex.width as f32, cur_tex.height as f32],
                        sigma: sx,
                        alpha: 1.0,
                        filter_kind: FILTER_KIND_BLUR_1D,
                        radius: rx,
                        tile_mode: *tile_mode as u32,
                        ..Default::default()
                    };
                    cur_tex = self.push_image_filter_subpass(
                        arena,
                        subpasses,
                        used_layer_textures,
                        cur_tex,
                        frag_x,
                        true,
                        format,
                        uniform_align,
                    );
                }
                if sy > 1.0e-4 {
                    let ry = (sy * 3.0).ceil().clamp(1.0, 64.0) as i32;
                    let frag_y = ImageFilterFragUniform {
                        step_uv: [0.0, 1.0 / (cur_tex.height as f32)],
                        tex_size: [cur_tex.width as f32, cur_tex.height as f32],
                        sigma: sy,
                        alpha: 1.0,
                        filter_kind: FILTER_KIND_BLUR_1D,
                        radius: ry,
                        tile_mode: *tile_mode as u32,
                        ..Default::default()
                    };
                    cur_tex = self.push_image_filter_subpass(
                        arena,
                        subpasses,
                        used_layer_textures,
                        cur_tex,
                        frag_y,
                        true,
                        format,
                        uniform_align,
                    );
                }
                cur_tex
            }
            LayerImageFilter::Dilate { radius_x, radius_y }
            | LayerImageFilter::Erode { radius_x, radius_y } => {
                if *post_transform != Matrix::IDENTITY {
                    cur_tex = self.bake_layer_transform_subpass(
                        arena,
                        subpasses,
                        used_layer_textures,
                        cur_tex,
                        child_origin,
                        ssaa_scalar,
                        *post_transform,
                        format,
                        uniform_align,
                    );
                    *post_transform = Matrix::IDENTITY;
                }
                let kind = if matches!(filter, LayerImageFilter::Dilate { .. }) {
                    FILTER_KIND_DILATE_1D
                } else {
                    FILTER_KIND_ERODE_1D
                };
                let rx = (radius_x * ssaa_scalar).ceil().clamp(0.0, 64.0) as i32;
                let ry = (radius_y * ssaa_scalar).ceil().clamp(0.0, 64.0) as i32;
                if rx >= 1 {
                    let frag_x = ImageFilterFragUniform {
                        step_uv: [1.0 / (cur_tex.width as f32), 0.0],
                        tex_size: [cur_tex.width as f32, cur_tex.height as f32],
                        alpha: 1.0,
                        filter_kind: kind,
                        radius: rx,
                        tile_mode: TileMode::Decal as u32,
                        ..Default::default()
                    };
                    cur_tex = self.push_image_filter_subpass(
                        arena,
                        subpasses,
                        used_layer_textures,
                        cur_tex,
                        frag_x,
                        false,
                        format,
                        uniform_align,
                    );
                }
                if ry >= 1 {
                    let frag_y = ImageFilterFragUniform {
                        step_uv: [0.0, 1.0 / (cur_tex.height as f32)],
                        tex_size: [cur_tex.width as f32, cur_tex.height as f32],
                        alpha: 1.0,
                        filter_kind: kind,
                        radius: ry,
                        tile_mode: TileMode::Decal as u32,
                        ..Default::default()
                    };
                    cur_tex = self.push_image_filter_subpass(
                        arena,
                        subpasses,
                        used_layer_textures,
                        cur_tex,
                        frag_y,
                        false,
                        format,
                        uniform_align,
                    );
                }
                cur_tex
            }
            LayerImageFilter::Compose(outer, inner) => {
                let mid = self.prepare_image_filter_passes(
                    arena,
                    subpasses,
                    used_layer_textures,
                    cur_tex,
                    inner,
                    child_origin,
                    ssaa_scalar,
                    format,
                    uniform_align,
                    post_transform,
                );
                self.prepare_image_filter_passes(
                    arena,
                    subpasses,
                    used_layer_textures,
                    mid,
                    outer,
                    child_origin,
                    ssaa_scalar,
                    format,
                    uniform_align,
                    post_transform,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_pass_into_arena(
        &self,
        arena: &mut FrameArena,
        commands: &[DrawCommand],
        subpasses: &mut Vec<PreparedSubpass>,
        used_layer_textures: &mut Vec<Arc<SaveLayerTexture>>,
        ortho: Matrix,
        internal_width: u32,
        internal_height: u32,
        ssaa_scale: u32,
        sample_count: u32,
        format: wgpu::TextureFormat,
        uniform_align: usize,
        has_stencil: bool,
        layer_origin: Point,
    ) -> Vec<PendingArenaCall> {
        let clustered = cluster_non_overlapping_commands(commands);
        let mut calls: Vec<PendingArenaCall> = Vec::with_capacity(clustered.len().min(256));
        let mut ortho_frame_offset: Option<u32> = None;
        let mut white_frag_offset: Option<u32> = None;
        let mut stencil_depth: u32 = 0;

        for cmd in clustered {
            let scissor_opt = Self::compute_scissor_rect(
                cmd.clip_rect(),
                internal_width,
                internal_height,
                ssaa_scale,
                layer_origin,
            );

            if let DrawCommand::PushClip {
                mesh,
                transform,
                is_difference,
                ..
            } = cmd
            {
                if has_stencil && stencil_depth < 255 {
                    let cur_depth = stencil_depth;
                    if let Some(scissor) = scissor_opt {
                        let frag_off = *white_frag_offset.get_or_insert_with(|| {
                            let u = SolidFragUniform {
                                color: [1.0, 1.0, 1.0, 1.0],
                            };
                            arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                        });
                        if *is_difference {
                            let s = ssaa_scale.max(1) as Scalar;
                            let qx0 = layer_origin.x + (scissor[0] as Scalar) / s;
                            let qy0 = layer_origin.y + (scissor[1] as Scalar) / s;
                            let qx1 = layer_origin.x + ((scissor[0] + scissor[2]) as Scalar) / s;
                            let qy1 = layer_origin.y + ((scissor[1] + scissor[3]) as Scalar) / s;
                            let quad_verts = [
                                SolidVertex::new(qx0, qy0),
                                SolidVertex::new(qx1, qy0),
                                SolidVertex::new(qx1, qy1),
                                SolidVertex::new(qx0, qy1),
                            ];
                            let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
                            let ortho_off = *ortho_frame_offset.get_or_insert_with(|| {
                                let u = SolidFrameUniform { mvp: ortho.m };
                                arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                            });
                            self.push_stencil_call(
                                arena,
                                &mut calls,
                                PipelineKind::StencilIncrement,
                                cur_depth,
                                scissor,
                                ortho_off,
                                frag_off,
                                bytemuck::cast_slice(&quad_verts),
                                4,
                                &quad_indices,
                                sample_count,
                                format,
                            );

                            if !mesh.vertices.is_empty() && !mesh.indices.is_empty() {
                                let mvp = ortho * *transform;
                                let frame_off = if *transform == Matrix::IDENTITY {
                                    ortho_off
                                } else {
                                    let u = SolidFrameUniform { mvp: mvp.m };
                                    arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                                };
                                self.push_stencil_call(
                                    arena,
                                    &mut calls,
                                    PipelineKind::StencilDecrement,
                                    cur_depth + 1,
                                    scissor,
                                    frame_off,
                                    frag_off,
                                    bytemuck::cast_slice(&mesh.vertices),
                                    mesh.vertices.len() as u32,
                                    &mesh.indices,
                                    sample_count,
                                    format,
                                );
                            }
                        } else if !mesh.vertices.is_empty() && !mesh.indices.is_empty() {
                            let frame_off = if *transform == Matrix::IDENTITY {
                                *ortho_frame_offset.get_or_insert_with(|| {
                                    let u = SolidFrameUniform { mvp: ortho.m };
                                    arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                                })
                            } else {
                                let mvp = ortho * *transform;
                                let u = SolidFrameUniform { mvp: mvp.m };
                                arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                            };
                            self.push_stencil_call(
                                arena,
                                &mut calls,
                                PipelineKind::StencilIncrement,
                                cur_depth,
                                scissor,
                                frame_off,
                                frag_off,
                                bytemuck::cast_slice(&mesh.vertices),
                                mesh.vertices.len() as u32,
                                &mesh.indices,
                                sample_count,
                                format,
                            );
                        }
                    }
                    stencil_depth = cur_depth + 1;
                }
                continue;
            }

            if let DrawCommand::PopClip { .. } = cmd {
                if has_stencil && stencil_depth > 0 {
                    let cur_depth = stencil_depth;
                    if let Some(scissor) = scissor_opt {
                        let s = ssaa_scale.max(1) as Scalar;
                        let qx0 = layer_origin.x + (scissor[0] as Scalar) / s;
                        let qy0 = layer_origin.y + (scissor[1] as Scalar) / s;
                        let qx1 = layer_origin.x + ((scissor[0] + scissor[2]) as Scalar) / s;
                        let qy1 = layer_origin.y + ((scissor[1] + scissor[3]) as Scalar) / s;
                        let quad_verts = [
                            SolidVertex::new(qx0, qy0),
                            SolidVertex::new(qx1, qy0),
                            SolidVertex::new(qx1, qy1),
                            SolidVertex::new(qx0, qy1),
                        ];
                        let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
                        let ortho_off = *ortho_frame_offset.get_or_insert_with(|| {
                            let u = SolidFrameUniform { mvp: ortho.m };
                            arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                        });
                        let frag_off = *white_frag_offset.get_or_insert_with(|| {
                            let u = SolidFragUniform {
                                color: [1.0, 1.0, 1.0, 1.0],
                            };
                            arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                        });
                        self.push_stencil_call(
                            arena,
                            &mut calls,
                            PipelineKind::StencilDecrement,
                            cur_depth,
                            scissor,
                            ortho_off,
                            frag_off,
                            bytemuck::cast_slice(&quad_verts),
                            4,
                            &quad_indices,
                            sample_count,
                            format,
                        );
                    }
                    stencil_depth = cur_depth - 1;
                }
                continue;
            }

            let Some(scissor) = scissor_opt else {
                continue;
            };

            match cmd {
                DrawCommand::SolidFill {
                    mesh,
                    transform,
                    color,
                    blend_mode,
                    ..
                } => {
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }
                    let pm_color = color.to_array_premultiplied();
                    let is_identity = *transform == Matrix::IDENTITY;

                    if is_identity || is_2d_affine_transform(transform) {
                        let tag = CoalesceTag::SolidFill2D {
                            blend_mode: *blend_mode,
                        };
                        if let Some(last) = calls.last_mut() {
                            if last.coalesce_tag == tag
                                && last.scissor == scissor
                                && last.stencil_ref == stencil_depth
                            {
                                let base_vertex = last.vertex_count;
                                arena.vertex_cpu.reserve(
                                    mesh.vertices.len() * std::mem::size_of::<ColoredSolidVertex>(),
                                );
                                for v in &mesh.vertices {
                                    let pos = if is_identity {
                                        v.position
                                    } else {
                                        transform_vertex_2d(transform, v.position)
                                    };
                                    let cv = ColoredSolidVertex::new(pos, pm_color);
                                    arena.vertex_cpu.extend_from_slice(bytemuck::bytes_of(&cv));
                                }
                                arena
                                    .index_cpu
                                    .reserve(mesh.indices.len() * std::mem::size_of::<u32>());
                                for &idx in &mesh.indices {
                                    let shifted = base_vertex + idx;
                                    arena
                                        .index_cpu
                                        .extend_from_slice(bytemuck::bytes_of(&shifted));
                                }
                                let v_added = (mesh.vertices.len()
                                    * std::mem::size_of::<ColoredSolidVertex>())
                                    as u64;
                                let i_added =
                                    (mesh.indices.len() * std::mem::size_of::<u32>()) as u64;
                                last.vertex_count += mesh.vertices.len() as u32;
                                last.vertex_byte_len += v_added;
                                last.index_count += mesh.indices.len() as u32;
                                last.index_byte_len += i_added;
                                continue;
                            }
                        }

                        let pipeline = self.get_or_create_pipeline_full(
                            PipelineKind::SolidFill,
                            *blend_mode,
                            sample_count,
                            format,
                            has_stencil,
                        );
                        FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                        FrameArena::align_cpu_4(&mut arena.index_cpu);
                        let vertex_offset = arena.vertex_cpu.len() as u64;
                        arena.vertex_cpu.reserve(
                            mesh.vertices.len() * std::mem::size_of::<ColoredSolidVertex>(),
                        );
                        for v in &mesh.vertices {
                            let pos = if is_identity {
                                v.position
                            } else {
                                transform_vertex_2d(transform, v.position)
                            };
                            let cv = ColoredSolidVertex::new(pos, pm_color);
                            arena.vertex_cpu.extend_from_slice(bytemuck::bytes_of(&cv));
                        }
                        let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;

                        let index_offset = arena.index_cpu.len() as u64;
                        arena
                            .index_cpu
                            .extend_from_slice(bytemuck::cast_slice(&mesh.indices));
                        let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                        let frame_off = *ortho_frame_offset.get_or_insert_with(|| {
                            let u = SolidFrameUniform { mvp: ortho.m };
                            arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                        });
                        let frag_off = *white_frag_offset.get_or_insert_with(|| {
                            let u = SolidFragUniform {
                                color: [1.0, 1.0, 1.0, 1.0],
                            };
                            arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                        });

                        calls.push(PendingArenaCall {
                            pipeline,
                            binding: DrawCallBinding::SolidFill,
                            glyph_alpha_opt: None,
                            glyph_color_opt: None,
                            dynamic_offsets: [frame_off, frag_off, 0],
                            dynamic_offset_count: 2,
                            vertex_offset,
                            vertex_byte_len,
                            vertex_count: mesh.vertices.len() as u32,
                            index_offset,
                            index_byte_len,
                            index_count: mesh.indices.len() as u32,
                            scissor,
                            stencil_ref: stencil_depth,
                            coalesce_tag: tag,
                        });
                    } else {
                        let pipeline = self.get_or_create_pipeline_full(
                            PipelineKind::SolidFill,
                            *blend_mode,
                            sample_count,
                            format,
                            has_stencil,
                        );
                        FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                        FrameArena::align_cpu_4(&mut arena.index_cpu);
                        let vertex_offset = arena.vertex_cpu.len() as u64;
                        arena.vertex_cpu.reserve(
                            mesh.vertices.len() * std::mem::size_of::<ColoredSolidVertex>(),
                        );
                        for v in &mesh.vertices {
                            let cv = ColoredSolidVertex::new(v.position, pm_color);
                            arena.vertex_cpu.extend_from_slice(bytemuck::bytes_of(&cv));
                        }
                        let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;

                        let index_offset = arena.index_cpu.len() as u64;
                        arena
                            .index_cpu
                            .extend_from_slice(bytemuck::cast_slice(&mesh.indices));
                        let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                        let mvp = ortho * *transform;
                        let frame_uniform = SolidFrameUniform { mvp: mvp.m };
                        let frame_off =
                            arena.push_uniform(bytemuck::bytes_of(&frame_uniform), uniform_align);
                        let frag_off = *white_frag_offset.get_or_insert_with(|| {
                            let u = SolidFragUniform {
                                color: [1.0, 1.0, 1.0, 1.0],
                            };
                            arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                        });

                        calls.push(PendingArenaCall {
                            pipeline,
                            binding: DrawCallBinding::SolidFill,
                            glyph_alpha_opt: None,
                            glyph_color_opt: None,
                            dynamic_offsets: [frame_off, frag_off, 0],
                            dynamic_offset_count: 2,
                            vertex_offset,
                            vertex_byte_len,
                            vertex_count: mesh.vertices.len() as u32,
                            index_offset,
                            index_byte_len,
                            index_count: mesh.indices.len() as u32,
                            scissor,
                            stencil_ref: stencil_depth,
                            coalesce_tag: CoalesceTag::None,
                        });
                    }
                }
                DrawCommand::LinearGradientFill {
                    mesh,
                    transform,
                    gradient,
                    blend_mode,
                    ..
                } => {
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }
                    let pipeline = self.get_or_create_pipeline_full(
                        PipelineKind::LinearGradientFill,
                        *blend_mode,
                        sample_count,
                        format,
                        has_stencil,
                    );
                    let mvp = ortho * *transform;
                    let frame_uniform = GradientFrameUniform {
                        mvp: mvp.m,
                        matrix: Matrix::IDENTITY.m,
                    };

                    let start_to_end = gradient.end_point - gradient.start_point;
                    let dot_se = start_to_end.x * start_to_end.x + start_to_end.y * start_to_end.y;
                    let inv_dot = if dot_se <= 1.0e-8 { 0.0 } else { 1.0 / dot_se };
                    let (gpu_stops, colors_length, is_fast_two_stop, fast_color_0, fast_color_1) =
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
                        colors_length,
                        is_fast_two_stop: u32::from(is_fast_two_stop),
                        _pad: 0,
                    };

                    let frame_off =
                        arena.push_uniform(bytemuck::bytes_of(&frame_uniform), uniform_align);
                    let frag_off =
                        arena.push_uniform(bytemuck::bytes_of(&frag_uniform), uniform_align);
                    let stops_off =
                        arena.push_uniform(bytemuck::cast_slice(&gpu_stops), uniform_align);

                    FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                    FrameArena::align_cpu_4(&mut arena.index_cpu);
                    let vertex_offset = arena.vertex_cpu.len() as u64;
                    arena
                        .vertex_cpu
                        .extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
                    let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;

                    let index_offset = arena.index_cpu.len() as u64;
                    arena
                        .index_cpu
                        .extend_from_slice(bytemuck::cast_slice(&mesh.indices));
                    let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                    calls.push(PendingArenaCall {
                        pipeline,
                        binding: DrawCallBinding::LinearGradient,
                        glyph_alpha_opt: None,
                        glyph_color_opt: None,
                        dynamic_offsets: [frame_off, frag_off, stops_off],
                        dynamic_offset_count: 3,
                        vertex_offset,
                        vertex_byte_len,
                        vertex_count: mesh.vertices.len() as u32,
                        index_offset,
                        index_byte_len,
                        index_count: mesh.indices.len() as u32,
                        scissor,
                        stencil_ref: stencil_depth,
                        coalesce_tag: CoalesceTag::None,
                    });
                }
                DrawCommand::RadialGradientFill {
                    mesh,
                    transform,
                    gradient,
                    blend_mode,
                    ..
                } => {
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }
                    let pipeline = self.get_or_create_pipeline_full(
                        PipelineKind::RadialGradientFill,
                        *blend_mode,
                        sample_count,
                        format,
                        has_stencil,
                    );
                    let mvp = ortho * *transform;
                    let frame_uniform = GradientFrameUniform {
                        mvp: mvp.m,
                        matrix: Matrix::IDENTITY.m,
                    };

                    let (gpu_stops, colors_length, is_fast_two_stop, fast_color_0, fast_color_1) =
                        build_gpu_color_points(&gradient.stops);

                    let frag_uniform = RadialGradientFragUniform {
                        center: [gradient.center.x, gradient.center.y],
                        radius: gradient.radius.max(K_EH_CLOSE_ENOUGH),
                        alpha: gradient.alpha.clamp(0.0, 1.0),
                        decal_border_color: [0.0; 4],
                        fast_color_0,
                        fast_color_1,
                        tile_mode: gradient.tile_mode as u32,
                        colors_length,
                        is_fast_two_stop: u32::from(is_fast_two_stop),
                        _pad: 0,
                    };

                    let frame_off =
                        arena.push_uniform(bytemuck::bytes_of(&frame_uniform), uniform_align);
                    let frag_off =
                        arena.push_uniform(bytemuck::bytes_of(&frag_uniform), uniform_align);
                    let stops_off =
                        arena.push_uniform(bytemuck::cast_slice(&gpu_stops), uniform_align);

                    FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                    FrameArena::align_cpu_4(&mut arena.index_cpu);
                    let vertex_offset = arena.vertex_cpu.len() as u64;
                    arena
                        .vertex_cpu
                        .extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
                    let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;

                    let index_offset = arena.index_cpu.len() as u64;
                    arena
                        .index_cpu
                        .extend_from_slice(bytemuck::cast_slice(&mesh.indices));
                    let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                    calls.push(PendingArenaCall {
                        pipeline,
                        binding: DrawCallBinding::RadialGradient,
                        glyph_alpha_opt: None,
                        glyph_color_opt: None,
                        dynamic_offsets: [frame_off, frag_off, stops_off],
                        dynamic_offset_count: 3,
                        vertex_offset,
                        vertex_byte_len,
                        vertex_count: mesh.vertices.len() as u32,
                        index_offset,
                        index_byte_len,
                        index_count: mesh.indices.len() as u32,
                        scissor,
                        stencil_ref: stencil_depth,
                        coalesce_tag: CoalesceTag::None,
                    });
                }
                DrawCommand::SweepGradientFill {
                    mesh,
                    transform,
                    gradient,
                    blend_mode,
                    ..
                } => {
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }
                    let pipeline = self.get_or_create_pipeline_full(
                        PipelineKind::SweepGradientFill,
                        *blend_mode,
                        sample_count,
                        format,
                        has_stencil,
                    );
                    let mvp = ortho * *transform;
                    let frame_uniform = GradientFrameUniform {
                        mvp: mvp.m,
                        matrix: Matrix::IDENTITY.m,
                    };

                    let t0 = gradient.start_angle_deg / 360.0;
                    let t1 = gradient.end_angle_deg / 360.0;
                    let span = t1 - t0;
                    let bias = -t0;
                    let scale = if span.abs() <= 1.0e-6 {
                        0.0
                    } else {
                        1.0 / span
                    };
                    let (gpu_stops, colors_length, is_fast_two_stop, fast_color_0, fast_color_1) =
                        build_gpu_color_points(&gradient.stops);

                    let frag_uniform = SweepGradientFragUniform {
                        center: [gradient.center.x, gradient.center.y],
                        bias,
                        scale,
                        decal_border_color: [0.0; 4],
                        fast_color_0,
                        fast_color_1,
                        tile_mode: gradient.tile_mode as u32,
                        colors_length,
                        is_fast_two_stop: u32::from(is_fast_two_stop),
                        alpha: gradient.alpha.clamp(0.0, 1.0),
                    };

                    let frame_off =
                        arena.push_uniform(bytemuck::bytes_of(&frame_uniform), uniform_align);
                    let frag_off =
                        arena.push_uniform(bytemuck::bytes_of(&frag_uniform), uniform_align);
                    let stops_off =
                        arena.push_uniform(bytemuck::cast_slice(&gpu_stops), uniform_align);

                    FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                    FrameArena::align_cpu_4(&mut arena.index_cpu);
                    let vertex_offset = arena.vertex_cpu.len() as u64;
                    arena
                        .vertex_cpu
                        .extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
                    let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;

                    let index_offset = arena.index_cpu.len() as u64;
                    arena
                        .index_cpu
                        .extend_from_slice(bytemuck::cast_slice(&mesh.indices));
                    let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                    calls.push(PendingArenaCall {
                        pipeline,
                        binding: DrawCallBinding::SweepGradient,
                        glyph_alpha_opt: None,
                        glyph_color_opt: None,
                        dynamic_offsets: [frame_off, frag_off, stops_off],
                        dynamic_offset_count: 3,
                        vertex_offset,
                        vertex_byte_len,
                        vertex_count: mesh.vertices.len() as u32,
                        index_offset,
                        index_byte_len,
                        index_count: mesh.indices.len() as u32,
                        scissor,
                        stencil_ref: stencil_depth,
                        coalesce_tag: CoalesceTag::None,
                    });
                }
                DrawCommand::ConicalGradientFill {
                    mesh,
                    transform,
                    gradient,
                    blend_mode,
                    ..
                } => {
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }
                    let pipeline = self.get_or_create_pipeline_full(
                        PipelineKind::ConicalGradientFill,
                        *blend_mode,
                        sample_count,
                        format,
                        has_stencil,
                    );
                    let mvp = ortho * *transform;
                    let frame_uniform = GradientFrameUniform {
                        mvp: mvp.m,
                        matrix: Matrix::IDENTITY.m,
                    };

                    let start_r = gradient.start_radius.max(0.0);
                    let end_r = gradient.end_radius.max(0.0);
                    let same_center =
                        gradient.start_center.get_distance(gradient.end_center) < K_EH_CLOSE_ENOUGH;
                    let same_radius = (start_r - end_r).abs() < K_EH_CLOSE_ENOUGH;
                    let kind = match (same_center, same_radius) {
                        (true, true) => 0u32,
                        (true, false) => 1u32,
                        (false, true) => 2u32,
                        (false, false) => 3u32,
                    };

                    let (gpu_stops, colors_length, is_fast_two_stop, fast_color_0, fast_color_1) =
                        build_gpu_color_points(&gradient.stops);

                    let frag_uniform = ConicalGradientFragUniform {
                        start_center: [gradient.start_center.x, gradient.start_center.y],
                        start_radius: start_r,
                        end_radius: end_r,
                        end_center: [gradient.end_center.x, gradient.end_center.y],
                        alpha: gradient.alpha.clamp(0.0, 1.0),
                        kind,
                        decal_border_color: [0.0; 4],
                        fast_color_0,
                        fast_color_1,
                        tile_mode: gradient.tile_mode as u32,
                        colors_length,
                        is_fast_two_stop: u32::from(is_fast_two_stop),
                        _pad: 0,
                    };

                    let frame_off =
                        arena.push_uniform(bytemuck::bytes_of(&frame_uniform), uniform_align);
                    let frag_off =
                        arena.push_uniform(bytemuck::bytes_of(&frag_uniform), uniform_align);
                    let stops_off =
                        arena.push_uniform(bytemuck::cast_slice(&gpu_stops), uniform_align);

                    FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                    FrameArena::align_cpu_4(&mut arena.index_cpu);
                    let vertex_offset = arena.vertex_cpu.len() as u64;
                    arena
                        .vertex_cpu
                        .extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
                    let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;

                    let index_offset = arena.index_cpu.len() as u64;
                    arena
                        .index_cpu
                        .extend_from_slice(bytemuck::cast_slice(&mesh.indices));
                    let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                    calls.push(PendingArenaCall {
                        pipeline,
                        binding: DrawCallBinding::ConicalGradient,
                        glyph_alpha_opt: None,
                        glyph_color_opt: None,
                        dynamic_offsets: [frame_off, frag_off, stops_off],
                        dynamic_offset_count: 3,
                        vertex_offset,
                        vertex_byte_len,
                        vertex_count: mesh.vertices.len() as u32,
                        index_offset,
                        index_byte_len,
                        index_count: mesh.indices.len() as u32,
                        scissor,
                        stencil_ref: stencil_depth,
                        coalesce_tag: CoalesceTag::None,
                    });
                }
                DrawCommand::RRectBlur {
                    transform,
                    params,
                    blend_mode,
                    ..
                } => {
                    if params.rect.is_empty() {
                        continue;
                    }
                    let pipeline = self.get_or_create_pipeline_full(
                        PipelineKind::RRectBlur,
                        *blend_mode,
                        sample_count,
                        format,
                        has_stencil,
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

                    let frame_off =
                        arena.push_uniform(bytemuck::bytes_of(&frame_uniform), uniform_align);
                    let frag_off =
                        arena.push_uniform(bytemuck::bytes_of(&frag_uniform), uniform_align);

                    FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                    FrameArena::align_cpu_4(&mut arena.index_cpu);
                    let vertex_offset = arena.vertex_cpu.len() as u64;
                    arena
                        .vertex_cpu
                        .extend_from_slice(bytemuck::cast_slice(&quad_vertices));
                    let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;

                    let index_offset = arena.index_cpu.len() as u64;
                    arena
                        .index_cpu
                        .extend_from_slice(bytemuck::cast_slice(&quad_indices));
                    let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                    calls.push(PendingArenaCall {
                        pipeline,
                        binding: DrawCallBinding::RRectBlur,
                        glyph_alpha_opt: None,
                        glyph_color_opt: None,
                        dynamic_offsets: [frame_off, frag_off, 0],
                        dynamic_offset_count: 2,
                        vertex_offset,
                        vertex_byte_len,
                        vertex_count: 4,
                        index_offset,
                        index_byte_len,
                        index_count: 6,
                        scissor,
                        stencil_ref: stencil_depth,
                        coalesce_tag: CoalesceTag::None,
                    });
                }
                DrawCommand::GlyphAtlas(batch) => {
                    let mut valid_quad_count = 0usize;
                    let mut uses_alpha = false;
                    let mut uses_color = false;
                    for q in &batch.quads {
                        if !q.dest_rect.is_empty() && !q.atlas_rect.is_empty() {
                            valid_quad_count += 1;
                            if q.is_color {
                                uses_color = true;
                            } else {
                                uses_alpha = true;
                            }
                        }
                    }
                    if valid_quad_count == 0 {
                        continue;
                    }

                    let alpha_opt = if uses_alpha {
                        Some(
                            batch
                                .alpha_atlas
                                .as_deref()
                                .map(|a| self.get_or_upload_atlas(a))
                                .unwrap_or_else(|| Arc::clone(&self.fallback_alpha_atlas)),
                        )
                    } else {
                        None
                    };
                    let color_opt = if uses_color {
                        Some(
                            batch
                                .color_atlas
                                .as_deref()
                                .map(|c| self.get_or_upload_atlas(c))
                                .unwrap_or_else(|| Arc::clone(&self.fallback_color_atlas)),
                        )
                    } else {
                        None
                    };

                    let alpha_ref = alpha_opt.as_ref().unwrap_or(&self.fallback_alpha_atlas);
                    let color_ref = color_opt.as_ref().unwrap_or(&self.fallback_color_atlas);
                    let inv_alpha_w = 1.0 / (alpha_ref.width.max(1) as Scalar);
                    let inv_alpha_h = 1.0 / (alpha_ref.height.max(1) as Scalar);
                    let inv_color_w = 1.0 / (color_ref.width.max(1) as Scalar);
                    let inv_color_h = 1.0 / (color_ref.height.max(1) as Scalar);

                    let use_linear = effective_glyph_linear_filter(batch);
                    let text_contrast = batch.text_contrast.max(0.1);
                    let alpha = batch.alpha.clamp(0.0, 1.0);
                    let is_identity = batch.transform == Matrix::IDENTITY;
                    let is_2d = is_identity || is_2d_affine_transform(&batch.transform);

                    let append_glyph_quads =
                        |vertex_cpu: &mut Vec<u8>,
                         index_cpu: &mut Vec<u8>,
                         mut base_vertex: u32,
                         apply_2d_tx: bool| {
                            vertex_cpu
                                .reserve(valid_quad_count * 4 * std::mem::size_of::<GlyphVertex>());
                            index_cpu.reserve(valid_quad_count * 6 * std::mem::size_of::<u32>());
                            for quad in &batch.quads {
                                if quad.dest_rect.is_empty() || quad.atlas_rect.is_empty() {
                                    continue;
                                }
                                let (inv_w, inv_h) = if quad.is_color {
                                    (inv_color_w, inv_color_h)
                                } else {
                                    (inv_alpha_w, inv_alpha_h)
                                };
                                let l = quad.dest_rect.get_left();
                                let t = quad.dest_rect.get_top();
                                let r = quad.dest_rect.get_right();
                                let b = quad.dest_rect.get_bottom();

                                let (p0, p1, p2, p3) = if apply_2d_tx && !is_identity {
                                    (
                                        transform_vertex_2d(&batch.transform, [l, t]),
                                        transform_vertex_2d(&batch.transform, [r, t]),
                                        transform_vertex_2d(&batch.transform, [r, b]),
                                        transform_vertex_2d(&batch.transform, [l, b]),
                                    )
                                } else {
                                    ([l, t], [r, t], [r, b], [l, b])
                                };

                                let u0 = quad.atlas_rect.get_left() * inv_w;
                                let v0 = quad.atlas_rect.get_top() * inv_h;
                                let u1 = quad.atlas_rect.get_right() * inv_w;
                                let v1 = quad.atlas_rect.get_bottom() * inv_h;
                                let pm_color = quad.color.to_array_premultiplied();
                                let is_color_u32 = u32::from(quad.is_color);

                                let verts = [
                                    GlyphVertex::new(p0, [u0, v0], pm_color, is_color_u32),
                                    GlyphVertex::new(p1, [u1, v0], pm_color, is_color_u32),
                                    GlyphVertex::new(p2, [u1, v1], pm_color, is_color_u32),
                                    GlyphVertex::new(p3, [u0, v1], pm_color, is_color_u32),
                                ];
                                vertex_cpu.extend_from_slice(bytemuck::cast_slice(&verts));
                                let idxs = [
                                    base_vertex,
                                    base_vertex + 1,
                                    base_vertex + 2,
                                    base_vertex,
                                    base_vertex + 2,
                                    base_vertex + 3,
                                ];
                                index_cpu.extend_from_slice(bytemuck::cast_slice(&idxs));
                                base_vertex += 4;
                            }
                        };

                    if is_2d {
                        let tag = CoalesceTag::GlyphAtlas2D {
                            blend_mode: batch.blend_mode,
                            use_text_color: batch.use_text_color,
                            text_contrast_bits: text_contrast.to_bits(),
                            alpha_bits: alpha.to_bits(),
                            use_linear,
                        };
                        if let Some(last) = calls.last_mut() {
                            let alpha_compat = match (&last.glyph_alpha_opt, &alpha_opt) {
                                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                                _ => true,
                            };
                            let color_compat = match (&last.glyph_color_opt, &color_opt) {
                                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                                _ => true,
                            };
                            if last.coalesce_tag == tag
                                && last.scissor == scissor
                                && last.stencil_ref == stencil_depth
                                && alpha_compat
                                && color_compat
                            {
                                if last.glyph_alpha_opt.is_none() {
                                    last.glyph_alpha_opt = alpha_opt.clone();
                                }
                                if last.glyph_color_opt.is_none() {
                                    last.glyph_color_opt = color_opt.clone();
                                }
                                last.binding =
                                    DrawCallBinding::GlyphAtlas {
                                        alpha_gpu: last.glyph_alpha_opt.clone().unwrap_or_else(
                                            || Arc::clone(&self.fallback_alpha_atlas),
                                        ),
                                        color_gpu: last.glyph_color_opt.clone().unwrap_or_else(
                                            || Arc::clone(&self.fallback_color_atlas),
                                        ),
                                        use_linear,
                                    };
                                append_glyph_quads(
                                    &mut arena.vertex_cpu,
                                    &mut arena.index_cpu,
                                    last.vertex_count,
                                    true,
                                );
                                let v_added =
                                    (valid_quad_count * 4 * std::mem::size_of::<GlyphVertex>())
                                        as u64;
                                let i_added =
                                    (valid_quad_count * 6 * std::mem::size_of::<u32>()) as u64;
                                last.vertex_count += (valid_quad_count * 4) as u32;
                                last.vertex_byte_len += v_added;
                                last.index_count += (valid_quad_count * 6) as u32;
                                last.index_byte_len += i_added;
                                continue;
                            }
                        }

                        let pipeline = self.get_or_create_pipeline_full(
                            PipelineKind::GlyphAtlas,
                            batch.blend_mode,
                            sample_count,
                            format,
                            has_stencil,
                        );
                        FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                        FrameArena::align_cpu_4(&mut arena.index_cpu);
                        let vertex_offset = arena.vertex_cpu.len() as u64;
                        let index_offset = arena.index_cpu.len() as u64;
                        append_glyph_quads(&mut arena.vertex_cpu, &mut arena.index_cpu, 0, true);
                        let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;
                        let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                        let frame_off = *ortho_frame_offset.get_or_insert_with(|| {
                            let u = SolidFrameUniform { mvp: ortho.m };
                            arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                        });
                        let frag_uniform = GlyphAtlasFragUniform {
                            use_text_color: if batch.use_text_color { 1.0 } else { 0.0 },
                            text_contrast,
                            alpha,
                            _pad: 0.0,
                        };
                        let frag_off =
                            arena.push_uniform(bytemuck::bytes_of(&frag_uniform), uniform_align);

                        calls.push(PendingArenaCall {
                            pipeline,
                            binding: DrawCallBinding::GlyphAtlas {
                                alpha_gpu: alpha_opt
                                    .clone()
                                    .unwrap_or_else(|| Arc::clone(&self.fallback_alpha_atlas)),
                                color_gpu: color_opt
                                    .clone()
                                    .unwrap_or_else(|| Arc::clone(&self.fallback_color_atlas)),
                                use_linear,
                            },
                            glyph_alpha_opt: alpha_opt,
                            glyph_color_opt: color_opt,
                            dynamic_offsets: [frame_off, frag_off, 0],
                            dynamic_offset_count: 2,
                            vertex_offset,
                            vertex_byte_len,
                            vertex_count: (valid_quad_count * 4) as u32,
                            index_offset,
                            index_byte_len,
                            index_count: (valid_quad_count * 6) as u32,
                            scissor,
                            stencil_ref: stencil_depth,
                            coalesce_tag: tag,
                        });
                    } else {
                        let pipeline = self.get_or_create_pipeline_full(
                            PipelineKind::GlyphAtlas,
                            batch.blend_mode,
                            sample_count,
                            format,
                            has_stencil,
                        );
                        FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                        FrameArena::align_cpu_4(&mut arena.index_cpu);
                        let vertex_offset = arena.vertex_cpu.len() as u64;
                        let index_offset = arena.index_cpu.len() as u64;
                        append_glyph_quads(&mut arena.vertex_cpu, &mut arena.index_cpu, 0, false);
                        let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;
                        let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                        let mvp = ortho * batch.transform;
                        let frame_uniform = SolidFrameUniform { mvp: mvp.m };
                        let frame_off =
                            arena.push_uniform(bytemuck::bytes_of(&frame_uniform), uniform_align);
                        let frag_uniform = GlyphAtlasFragUniform {
                            use_text_color: if batch.use_text_color { 1.0 } else { 0.0 },
                            text_contrast,
                            alpha,
                            _pad: 0.0,
                        };
                        let frag_off =
                            arena.push_uniform(bytemuck::bytes_of(&frag_uniform), uniform_align);

                        calls.push(PendingArenaCall {
                            pipeline,
                            binding: DrawCallBinding::GlyphAtlas {
                                alpha_gpu: alpha_opt
                                    .clone()
                                    .unwrap_or_else(|| Arc::clone(&self.fallback_alpha_atlas)),
                                color_gpu: color_opt
                                    .clone()
                                    .unwrap_or_else(|| Arc::clone(&self.fallback_color_atlas)),
                                use_linear,
                            },
                            glyph_alpha_opt: None,
                            glyph_color_opt: None,
                            dynamic_offsets: [frame_off, frag_off, 0],
                            dynamic_offset_count: 2,
                            vertex_offset,
                            vertex_byte_len,
                            vertex_count: (valid_quad_count * 4) as u32,
                            index_offset,
                            index_byte_len,
                            index_count: (valid_quad_count * 6) as u32,
                            scissor,
                            stencil_ref: stencil_depth,
                            coalesce_tag: CoalesceTag::None,
                        });
                    }
                }
                DrawCommand::TextureFill(batch) => {
                    if batch.vertices.is_empty()
                        || batch.indices.is_empty()
                        || batch.texture.width == 0
                        || batch.texture.height == 0
                    {
                        continue;
                    }
                    let pipeline = self.get_or_create_pipeline_full(
                        PipelineKind::TextureFill,
                        batch.blend_mode,
                        sample_count,
                        format,
                        has_stencil,
                    );
                    let gpu_tex = self.get_or_upload_image_texture(&batch.texture);
                    let mvp = ortho * batch.transform;
                    let frame_uniform = SolidFrameUniform { mvp: mvp.m };
                    let frag_uniform = TextureFragUniform {
                        alpha: batch.alpha.clamp(0.0, 1.0),
                        tile_mode_x: batch.tile_mode_x as u32,
                        tile_mode_y: batch.tile_mode_y as u32,
                        _pad: 0,
                    };

                    let frame_off =
                        arena.push_uniform(bytemuck::bytes_of(&frame_uniform), uniform_align);
                    let frag_off =
                        arena.push_uniform(bytemuck::bytes_of(&frag_uniform), uniform_align);

                    FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                    FrameArena::align_cpu_4(&mut arena.index_cpu);
                    let vertex_offset = arena.vertex_cpu.len() as u64;
                    arena
                        .vertex_cpu
                        .extend_from_slice(bytemuck::cast_slice(&batch.vertices));
                    let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;

                    let index_offset = arena.index_cpu.len() as u64;
                    arena
                        .index_cpu
                        .extend_from_slice(bytemuck::cast_slice(&batch.indices));
                    let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                    calls.push(PendingArenaCall {
                        pipeline,
                        binding: DrawCallBinding::TextureFill {
                            gpu_tex,
                            linear_filter: batch.linear_filter,
                        },
                        glyph_alpha_opt: None,
                        glyph_color_opt: None,
                        dynamic_offsets: [frame_off, frag_off, 0],
                        dynamic_offset_count: 2,
                        vertex_offset,
                        vertex_byte_len,
                        vertex_count: batch.vertices.len() as u32,
                        index_offset,
                        index_byte_len,
                        index_count: batch.indices.len() as u32,
                        scissor,
                        stencil_ref: stencil_depth,
                        coalesce_tag: CoalesceTag::None,
                    });
                }
                DrawCommand::SaveLayer {
                    bounds,
                    commands: child_cmds,
                    alpha,
                    blend_mode,
                    color_filter,
                    image_filter,
                    ..
                } => {
                    if bounds.is_empty()
                        || *alpha <= 0.0
                        || (child_cmds.is_empty()
                            && color_filter.is_none()
                            && *blend_mode == BlendMode::SrcOver)
                    {
                        continue;
                    }

                    let (halo_x, halo_y) = image_filter
                        .as_ref()
                        .map(|f| f.halo_padding())
                        .unwrap_or((0.0, 0.0));
                    let s = ssaa_scale.max(1) as Scalar;
                    let parent_logical_w = (internal_width as Scalar) / s;
                    let parent_logical_h = (internal_height as Scalar) / s;
                    let clamp_rect = Rect::from_ltrb(
                        layer_origin.x - halo_x,
                        layer_origin.y - halo_y,
                        layer_origin.x + parent_logical_w + halo_x,
                        layer_origin.y + parent_logical_h + halo_y,
                    );
                    let expanded_bounds = Rect::from_ltrb(
                        bounds.get_left() - halo_x,
                        bounds.get_top() - halo_y,
                        bounds.get_right() + halo_x,
                        bounds.get_bottom() + halo_y,
                    );
                    let Some(clipped_layer_bounds) = intersect_rects(expanded_bounds, clamp_rect)
                    else {
                        continue;
                    };

                    let min_px_x = (clipped_layer_bounds.get_left() * s).floor();
                    let min_px_y = (clipped_layer_bounds.get_top() * s).floor();
                    let max_px_x = (clipped_layer_bounds.get_right() * s).ceil();
                    let max_px_y = (clipped_layer_bounds.get_bottom() * s).ceil();
                    let layer_px_w = ((max_px_x - min_px_x).max(0.0) as u32).min(4096);
                    let layer_px_h = ((max_px_y - min_px_y).max(0.0) as u32).min(4096);
                    if layer_px_w == 0 || layer_px_h == 0 {
                        continue;
                    }

                    let child_origin = Point::new(min_px_x / s, min_px_y / s);
                    let child_logical_w = (layer_px_w as Scalar) / s;
                    let child_logical_h = (layer_px_h as Scalar) / s;
                    let child_ortho = Matrix::make_orthographic(Size::new(
                        layer_px_w as Scalar,
                        layer_px_h as Scalar,
                    )) * Matrix::make_scale(s, s, 1.0)
                        * Matrix::make_translation(-child_origin.x, -child_origin.y, 0.0);

                    let child_has_stencil = child_cmds
                        .iter()
                        .any(|c| matches!(c, DrawCommand::PushClip { .. }));
                    let raw_layer_tex =
                        self.acquire_save_layer_texture(layer_px_w, layer_px_h, TARGET_FORMAT);
                    used_layer_textures.push(Arc::clone(&raw_layer_tex));

                    let child_calls = self.prepare_pass_into_arena(
                        arena,
                        child_cmds,
                        subpasses,
                        used_layer_textures,
                        child_ortho,
                        layer_px_w,
                        layer_px_h,
                        ssaa_scale,
                        1,
                        TARGET_FORMAT,
                        uniform_align,
                        child_has_stencil,
                        child_origin,
                    );
                    subpasses.push(PreparedSubpass {
                        target_tex: Arc::clone(&raw_layer_tex),
                        has_stencil: child_has_stencil,
                        calls: child_calls,
                    });

                    let mut cur_tex = raw_layer_tex;
                    let mut post_transform = Matrix::IDENTITY;
                    if let Some(img_filter) = image_filter {
                        cur_tex = self.prepare_image_filter_passes(
                            arena,
                            subpasses,
                            used_layer_textures,
                            cur_tex,
                            img_filter,
                            child_origin,
                            s,
                            TARGET_FORMAT,
                            uniform_align,
                            &mut post_transform,
                        );
                    }
                    if let Some(col_filter) = color_filter {
                        if post_transform != Matrix::IDENTITY {
                            cur_tex = self.bake_layer_transform_subpass(
                                arena,
                                subpasses,
                                used_layer_textures,
                                cur_tex,
                                child_origin,
                                s,
                                post_transform,
                                TARGET_FORMAT,
                                uniform_align,
                            );
                            post_transform = Matrix::IDENTITY;
                        }
                        cur_tex = self.prepare_color_filter_passes(
                            arena,
                            subpasses,
                            used_layer_textures,
                            cur_tex,
                            col_filter,
                            TARGET_FORMAT,
                            uniform_align,
                        );
                    }

                    let pipeline = self.get_or_create_pipeline_full(
                        PipelineKind::TextureFill,
                        *blend_mode,
                        sample_count,
                        format,
                        has_stencil,
                    );
                    let frame_off = if post_transform == Matrix::IDENTITY {
                        *ortho_frame_offset.get_or_insert_with(|| {
                            let u = SolidFrameUniform { mvp: ortho.m };
                            arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                        })
                    } else {
                        let mvp = ortho * post_transform;
                        let u = SolidFrameUniform { mvp: mvp.m };
                        arena.push_uniform(bytemuck::bytes_of(&u), uniform_align)
                    };
                    let frag_u = TextureFragUniform {
                        alpha: alpha.clamp(0.0, 1.0),
                        tile_mode_x: TileMode::Decal as u32,
                        tile_mode_y: TileMode::Decal as u32,
                        _pad: 0,
                    };
                    let frag_off = arena.push_uniform(bytemuck::bytes_of(&frag_u), uniform_align);

                    let qx0 = child_origin.x;
                    let qy0 = child_origin.y;
                    let qx1 = qx0 + child_logical_w;
                    let qy1 = qy0 + child_logical_h;
                    let quad_verts = [
                        UvVertex::new(qx0, qy0, 0.0, 0.0),
                        UvVertex::new(qx1, qy0, 1.0, 0.0),
                        UvVertex::new(qx1, qy1, 1.0, 1.0),
                        UvVertex::new(qx0, qy1, 0.0, 1.0),
                    ];
                    let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
                    FrameArena::align_cpu_4(&mut arena.vertex_cpu);
                    FrameArena::align_cpu_4(&mut arena.index_cpu);
                    let vertex_offset = arena.vertex_cpu.len() as u64;
                    arena
                        .vertex_cpu
                        .extend_from_slice(bytemuck::cast_slice(&quad_verts));
                    let vertex_byte_len = (arena.vertex_cpu.len() as u64) - vertex_offset;
                    let index_offset = arena.index_cpu.len() as u64;
                    arena
                        .index_cpu
                        .extend_from_slice(bytemuck::cast_slice(&quad_indices));
                    let index_byte_len = (arena.index_cpu.len() as u64) - index_offset;

                    calls.push(PendingArenaCall {
                        pipeline,
                        binding: DrawCallBinding::LayerTextureFill {
                            layer_tex: cur_tex,
                            linear_filter: !post_transform.is_translation_scale_only(),
                        },
                        glyph_alpha_opt: None,
                        glyph_color_opt: None,
                        dynamic_offsets: [frame_off, frag_off, 0],
                        dynamic_offset_count: 2,
                        vertex_offset,
                        vertex_byte_len,
                        vertex_count: 4,
                        index_offset,
                        index_byte_len,
                        index_count: 6,
                        scissor,
                        stencil_ref: stencil_depth,
                        coalesce_tag: CoalesceTag::None,
                    });
                }
                DrawCommand::PushClip { .. } | DrawCommand::PopClip { .. } => {}
            }
        }
        calls
    }

    fn resolve_call_bind_group(
        &self,
        arena: &mut FrameArena,
        binding: &DrawCallBinding,
        uniform_gpu: &wgpu::Buffer,
    ) -> Arc<wgpu::BindGroup> {
        match binding {
            DrawCallBinding::SolidFill => {
                if let Some(bg) = arena.solid_fill_bg.as_ref() {
                    return Arc::clone(bg);
                }
                let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SolidFill Arena BindGroup"),
                    layout: &self.solid_fill_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<SolidFrameUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<SolidFragUniform>(),
                            ),
                        },
                    ],
                }));
                arena.last_frame_bind_groups_created += 1;
                arena.solid_fill_bg = Some(Arc::clone(&created));
                created
            }
            DrawCallBinding::RRectBlur => {
                if let Some(bg) = arena.rrect_blur_bg.as_ref() {
                    return Arc::clone(bg);
                }
                let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RRectBlur Arena BindGroup"),
                    layout: &self.rrect_blur_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<SolidFrameUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<RRectBlurFragUniform>(),
                            ),
                        },
                    ],
                }));
                arena.last_frame_bind_groups_created += 1;
                arena.rrect_blur_bg = Some(Arc::clone(&created));
                created
            }
            DrawCallBinding::LinearGradient => {
                if let Some(bg) = arena.linear_gradient_bg.as_ref() {
                    return Arc::clone(bg);
                }
                let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("LinearGradient Arena BindGroup"),
                    layout: &self.linear_gradient_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<GradientFrameUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<LinearGradientFragUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<[GpuColorPoint; MAX_GRADIENT_STOPS]>(),
                            ),
                        },
                    ],
                }));
                arena.last_frame_bind_groups_created += 1;
                arena.linear_gradient_bg = Some(Arc::clone(&created));
                created
            }
            DrawCallBinding::RadialGradient => {
                if let Some(bg) = arena.radial_gradient_bg.as_ref() {
                    return Arc::clone(bg);
                }
                let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RadialGradient Arena BindGroup"),
                    layout: &self.radial_gradient_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<GradientFrameUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<RadialGradientFragUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<[GpuColorPoint; MAX_GRADIENT_STOPS]>(),
                            ),
                        },
                    ],
                }));
                arena.last_frame_bind_groups_created += 1;
                arena.radial_gradient_bg = Some(Arc::clone(&created));
                created
            }
            DrawCallBinding::SweepGradient => {
                if let Some(bg) = arena.sweep_gradient_bg.as_ref() {
                    return Arc::clone(bg);
                }
                let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SweepGradient Arena BindGroup"),
                    layout: &self.sweep_gradient_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<GradientFrameUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<SweepGradientFragUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<[GpuColorPoint; MAX_GRADIENT_STOPS]>(),
                            ),
                        },
                    ],
                }));
                arena.last_frame_bind_groups_created += 1;
                arena.sweep_gradient_bg = Some(Arc::clone(&created));
                created
            }
            DrawCallBinding::ConicalGradient => {
                if let Some(bg) = arena.conical_gradient_bg.as_ref() {
                    return Arc::clone(bg);
                }
                let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ConicalGradient Arena BindGroup"),
                    layout: &self.conical_gradient_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<GradientFrameUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<ConicalGradientFragUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<[GpuColorPoint; MAX_GRADIENT_STOPS]>(),
                            ),
                        },
                    ],
                }));
                arena.last_frame_bind_groups_created += 1;
                arena.conical_gradient_bg = Some(Arc::clone(&created));
                created
            }
            DrawCallBinding::GlyphAtlas {
                alpha_gpu,
                color_gpu,
                use_linear,
            } => {
                let key = (
                    Arc::as_ptr(alpha_gpu) as usize,
                    Arc::as_ptr(color_gpu) as usize,
                    *use_linear,
                );
                if let Some((_, _, bg)) = arena.glyph_atlas_bgs.get(&key) {
                    return Arc::clone(bg);
                }
                if arena.glyph_atlas_bgs.len() >= 32 {
                    arena.glyph_atlas_bgs.clear();
                }
                let sampler = if *use_linear {
                    &self.linear_sampler
                } else {
                    &self.nearest_sampler
                };
                let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GlyphAtlas Arena BindGroup"),
                    layout: &self.glyph_atlas_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<SolidFrameUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<GlyphAtlasFragUniform>(),
                            ),
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
                }));
                arena.last_frame_bind_groups_created += 1;
                arena.glyph_atlas_bgs.insert(
                    key,
                    (
                        Arc::clone(alpha_gpu),
                        Arc::clone(color_gpu),
                        Arc::clone(&created),
                    ),
                );
                created
            }
            DrawCallBinding::TextureFill {
                gpu_tex,
                linear_filter,
            } => {
                let key = (Arc::as_ptr(gpu_tex) as usize, *linear_filter);
                if let Some((_, bg)) = arena.texture_fill_bgs.get(&key) {
                    return Arc::clone(bg);
                }
                if arena.texture_fill_bgs.len() >= 64 {
                    arena.texture_fill_bgs.clear();
                }
                let sampler = if *linear_filter {
                    &self.linear_sampler
                } else {
                    &self.nearest_sampler
                };
                let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("TextureFill Arena BindGroup"),
                    layout: &self.texture_fill_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<SolidFrameUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<TextureFragUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&gpu_tex.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                }));
                arena.last_frame_bind_groups_created += 1;
                arena
                    .texture_fill_bgs
                    .insert(key, (Arc::clone(gpu_tex), Arc::clone(&created)));
                created
            }
            DrawCallBinding::LayerTextureFill {
                layer_tex,
                linear_filter,
            } => {
                let key = (Arc::as_ptr(layer_tex) as usize, *linear_filter);
                if let Some((_, bg)) = arena.layer_texture_bgs.get(&key) {
                    return Arc::clone(bg);
                }
                if arena.layer_texture_bgs.len() >= 64 {
                    arena.layer_texture_bgs.clear();
                }
                let sampler = if *linear_filter {
                    &self.linear_sampler
                } else {
                    &self.nearest_sampler
                };
                let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("LayerTextureFill Arena BindGroup"),
                    layout: &self.texture_fill_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<SolidFrameUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<TextureFragUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&layer_tex.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                }));
                arena.last_frame_bind_groups_created += 1;
                arena
                    .layer_texture_bgs
                    .insert(key, (Arc::clone(layer_tex), Arc::clone(&created)));
                created
            }
            DrawCallBinding::ImageFilter {
                src_tex,
                linear_filter,
            } => {
                let key = (Arc::as_ptr(src_tex) as usize, *linear_filter);
                if let Some((_, bg)) = arena.image_filter_bgs.get(&key) {
                    return Arc::clone(bg);
                }
                if arena.image_filter_bgs.len() >= 64 {
                    arena.image_filter_bgs.clear();
                }
                let sampler = if *linear_filter {
                    &self.linear_sampler
                } else {
                    &self.nearest_sampler
                };
                let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ImageFilter Arena BindGroup"),
                    layout: &self.image_filter_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<SolidFrameUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: Self::uniform_binding(
                                uniform_gpu,
                                std::mem::size_of::<ImageFilterFragUniform>(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&src_tex.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                }));
                arena.last_frame_bind_groups_created += 1;
                arena
                    .image_filter_bgs
                    .insert(key, (Arc::clone(src_tex), Arc::clone(&created)));
                created
            }
        }
    }

    fn resolve_ssaa_bind_group(
        &self,
        arena: &mut FrameArena,
        scratch: &Arc<SsaaScratchTarget>,
        uniform_gpu: &wgpu::Buffer,
    ) -> Arc<wgpu::BindGroup> {
        let scratch_id = Arc::as_ptr(scratch) as usize;
        if let Some((cached_id, _, bg)) = arena.ssaa_resolve_bg.as_ref() {
            if *cached_id == scratch_id {
                return Arc::clone(bg);
            }
        }
        let created = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSAA Resolve Arena BindGroup"),
            layout: &self.texture_fill_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: Self::uniform_binding(
                        uniform_gpu,
                        std::mem::size_of::<SolidFrameUniform>(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: Self::uniform_binding(
                        uniform_gpu,
                        std::mem::size_of::<TextureFragUniform>(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&scratch.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        }));
        arena.last_frame_bind_groups_created += 1;
        arena.ssaa_resolve_bg = Some((scratch_id, Arc::clone(scratch), Arc::clone(&created)));
        created
    }

    fn resolve_subpasses(
        &self,
        arena: &mut FrameArena,
        subpasses: Vec<PreparedSubpass>,
    ) -> Vec<ResolvedSubpass> {
        if subpasses.is_empty() {
            return Vec::new();
        }
        let Some(uniform_gpu) = arena.uniform_gpu.as_ref().map(Arc::clone) else {
            return Vec::new();
        };
        subpasses
            .into_iter()
            .map(|sub| {
                let resolved_bgs = sub
                    .calls
                    .iter()
                    .map(|c| self.resolve_call_bind_group(arena, &c.binding, &uniform_gpu))
                    .collect();
                ResolvedSubpass {
                    target_tex: sub.target_tex,
                    has_stencil: sub.has_stencil,
                    calls: sub.calls,
                    resolved_bgs,
                }
            })
            .collect()
    }

    fn record_subpasses(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        subpasses: &[ResolvedSubpass],
        vertex_gpu: Option<&Arc<wgpu::Buffer>>,
        index_gpu: Option<&Arc<wgpu::Buffer>>,
    ) {
        for sub in subpasses {
            let stencil_scratch = sub.has_stencil.then(|| {
                self.get_or_create_stencil_scratch(sub.target_tex.width, sub.target_tex.height, 1)
            });
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Impeller SaveLayer Offscreen Subpass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &sub.target_tex.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: stencil_scratch.as_ref().map(|s| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &s.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Discard,
                        }),
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if !sub.calls.is_empty() {
                if let (Some(vb), Some(ib)) = (vertex_gpu, index_gpu) {
                    Self::record_arena_calls(
                        &mut rpass,
                        &sub.calls,
                        &sub.resolved_bgs,
                        vb,
                        ib,
                        sub.has_stencil,
                    );
                }
            }
        }
    }

    fn record_arena_calls<'a>(
        render_pass: &mut wgpu::RenderPass<'a>,
        calls: &'a [PendingArenaCall],
        resolved_bgs: &'a [Arc<wgpu::BindGroup>],
        vertex_gpu: &'a wgpu::Buffer,
        index_gpu: &'a wgpu::Buffer,
        has_stencil: bool,
    ) {
        let mut last_pipeline_id: Option<usize> = None;
        let mut last_scissor: Option<[u32; 4]> = None;
        let mut last_stencil_ref: Option<u32> = None;

        for (call, bg) in calls.iter().zip(resolved_bgs.iter()) {
            let p_id = Arc::as_ptr(&call.pipeline) as usize;
            if last_pipeline_id != Some(p_id) {
                render_pass.set_pipeline(&call.pipeline);
                last_pipeline_id = Some(p_id);
            }
            if has_stencil && last_stencil_ref != Some(call.stencil_ref) {
                render_pass.set_stencil_reference(call.stencil_ref);
                last_stencil_ref = Some(call.stencil_ref);
            }
            render_pass.set_bind_group(
                0,
                bg.as_ref(),
                &call.dynamic_offsets[..call.dynamic_offset_count],
            );
            if last_scissor != Some(call.scissor) {
                render_pass.set_scissor_rect(
                    call.scissor[0],
                    call.scissor[1],
                    call.scissor[2],
                    call.scissor[3],
                );
                last_scissor = Some(call.scissor);
            }
            render_pass.set_vertex_buffer(
                0,
                vertex_gpu.slice(call.vertex_offset..call.vertex_offset + call.vertex_byte_len),
            );
            render_pass.set_index_buffer(
                index_gpu.slice(call.index_offset..call.index_offset + call.index_byte_len),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..call.index_count, 0, 0..1);
        }
    }

    fn encode_and_submit(
        &self,
        target: &OffscreenTarget,
        clear_color: Color,
        commands: &[DrawCommand],
    ) -> Result<(), String> {
        let uniform_align = self.uniform_offset_alignment();
        let ssaa = target.ssaa_scale.max(1) as Scalar;
        let ortho = Matrix::make_orthographic(Size::new(
            target.internal_width as Scalar,
            target.internal_height as Scalar,
        )) * Matrix::make_scale(ssaa, ssaa, 1.0);
        let has_stencil = commands
            .iter()
            .any(|c| matches!(c, DrawCommand::PushClip { .. }));

        let mut used_layer_textures: Vec<Arc<SaveLayerTexture>> = Vec::new();
        let (resolved_subpasses, calls, resolved_bgs, vertex_gpu, index_gpu) = {
            let mut arena = self
                .frame_arena
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            arena.begin_frame(self.atlas_eviction_epoch.load(Ordering::Relaxed));
            let mut subpasses: Vec<PreparedSubpass> = Vec::new();
            let calls = self.prepare_pass_into_arena(
                &mut arena,
                commands,
                &mut subpasses,
                &mut used_layer_textures,
                ortho,
                target.internal_width,
                target.internal_height,
                target.ssaa_scale,
                target.sample_count,
                TARGET_FORMAT,
                uniform_align,
                has_stencil,
                Point::ZERO,
            );
            let subpass_draw_count: usize = subpasses.iter().map(|s| s.calls.len()).sum();
            arena.last_frame_draw_call_count = subpass_draw_count + calls.len();
            arena.upload_cpu_buffers(&self.device, &self.queue, uniform_align);

            let resolved_subpasses = self.resolve_subpasses(&mut arena, subpasses);
            let resolved_bgs: Vec<Arc<wgpu::BindGroup>> = if calls.is_empty() {
                Vec::new()
            } else {
                let uniform_gpu = Arc::clone(arena.uniform_gpu.as_ref().expect("uniform_gpu"));
                calls
                    .iter()
                    .map(|c| self.resolve_call_bind_group(&mut arena, &c.binding, &uniform_gpu))
                    .collect()
            };
            (
                resolved_subpasses,
                calls,
                resolved_bgs,
                arena.vertex_gpu.clone(),
                arena.index_gpu.clone(),
            )
        };

        let stencil_scratch = has_stencil.then(|| {
            self.get_or_create_stencil_scratch(
                target.internal_width,
                target.internal_height,
                target.sample_count,
            )
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Impeller Offscreen CommandEncoder"),
            });

        self.record_subpasses(
            &mut encoder,
            &resolved_subpasses,
            vertex_gpu.as_ref(),
            index_gpu.as_ref(),
        );

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
                depth_stencil_attachment: stencil_scratch.as_ref().map(|s| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &s.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Discard,
                        }),
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if !calls.is_empty() {
                if let (Some(vb), Some(ib)) = (vertex_gpu.as_ref(), index_gpu.as_ref()) {
                    Self::record_arena_calls(
                        &mut render_pass,
                        &calls,
                        &resolved_bgs,
                        vb,
                        ib,
                        has_stencil,
                    );
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.release_save_layer_textures(used_layer_textures);
        Ok(())
    }

    fn get_or_create_ssaa_scratch(&self, width: u32, height: u32) -> Arc<SsaaScratchTarget> {
        let w = width.max(1);
        let h = height.max(1);
        if let Ok(guard) = self.ssaa_scratch.lock() {
            if let Some(existing) = guard.as_ref() {
                if existing.width == w && existing.height == h {
                    return Arc::clone(existing);
                }
            }
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Impeller SSAA 2x Scratch Texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let created = Arc::new(SsaaScratchTarget {
            width: w,
            height: h,
            _texture: texture,
            view,
        });

        if let Ok(mut guard) = self.ssaa_scratch.lock() {
            *guard = Some(Arc::clone(&created));
        }
        created
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_to_texture_view_ssaa(
        &self,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        ssaa_scale: u32,
        clear_color: Color,
        commands: &[DrawCommand],
    ) -> Result<(), String> {
        let w = width.max(1);
        let h = height.max(1);
        if ssaa_scale <= 1 {
            return self.render_to_texture_view(
                target_view,
                None,
                w,
                h,
                format,
                1,
                clear_color,
                commands,
            );
        }

        let uniform_align = self.uniform_offset_alignment();
        let scale = 2u32;
        let ssaa_w = w * scale;
        let ssaa_h = h * scale;
        let scratch = self.get_or_create_ssaa_scratch(ssaa_w, ssaa_h);

        let ssaa_scalar = scale as Scalar;
        let pass1_ortho = Matrix::make_orthographic(Size::new(ssaa_w as Scalar, ssaa_h as Scalar))
            * Matrix::make_scale(ssaa_scalar, ssaa_scalar, 1.0);
        let resolve_pipeline = self.get_or_create_pipeline_with_samples_and_format(
            PipelineKind::TextureFill,
            BlendMode::SrcOver,
            1,
            format,
        );
        let pass2_ortho = Matrix::make_orthographic(Size::new(w as Scalar, h as Scalar));
        let resolve_frame = SolidFrameUniform { mvp: pass2_ortho.m };
        let resolve_frag = TextureFragUniform {
            alpha: 1.0,
            tile_mode_x: TileMode::Clamp as u32,
            tile_mode_y: TileMode::Clamp as u32,
            _pad: 0,
        };
        let resolve_vertices = [
            UvVertex::new(0.0, 0.0, 0.0, 0.0),
            UvVertex::new(w as Scalar, 0.0, 1.0, 0.0),
            UvVertex::new(w as Scalar, h as Scalar, 1.0, 1.0),
            UvVertex::new(0.0, h as Scalar, 0.0, 1.0),
        ];
        let resolve_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
        let has_stencil = commands
            .iter()
            .any(|c| matches!(c, DrawCommand::PushClip { .. }));

        let mut used_layer_textures: Vec<Arc<SaveLayerTexture>> = Vec::new();
        let (
            resolved_subpasses,
            calls,
            resolved_bgs,
            resolve_bg,
            resolve_dyn_offsets,
            resolve_vb_range,
            resolve_ib_range,
            vertex_gpu,
            index_gpu,
        ) = {
            let mut arena = self
                .frame_arena
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            arena.begin_frame(self.atlas_eviction_epoch.load(Ordering::Relaxed));
            let mut subpasses: Vec<PreparedSubpass> = Vec::new();
            let calls = self.prepare_pass_into_arena(
                &mut arena,
                commands,
                &mut subpasses,
                &mut used_layer_textures,
                pass1_ortho,
                ssaa_w,
                ssaa_h,
                scale,
                1,
                TARGET_FORMAT,
                uniform_align,
                has_stencil,
                Point::ZERO,
            );
            let subpass_draw_count: usize = subpasses.iter().map(|s| s.calls.len()).sum();
            arena.last_frame_draw_call_count = subpass_draw_count + calls.len() + 1;

            let resolve_frame_off =
                arena.push_uniform(bytemuck::bytes_of(&resolve_frame), uniform_align);
            let resolve_frag_off =
                arena.push_uniform(bytemuck::bytes_of(&resolve_frag), uniform_align);

            FrameArena::align_cpu_4(&mut arena.vertex_cpu);
            FrameArena::align_cpu_4(&mut arena.index_cpu);
            let r_v_off = arena.vertex_cpu.len() as u64;
            arena
                .vertex_cpu
                .extend_from_slice(bytemuck::cast_slice(&resolve_vertices));
            let r_v_end = arena.vertex_cpu.len() as u64;

            let r_i_off = arena.index_cpu.len() as u64;
            arena
                .index_cpu
                .extend_from_slice(bytemuck::cast_slice(&resolve_indices));
            let r_i_end = arena.index_cpu.len() as u64;

            arena.upload_cpu_buffers(&self.device, &self.queue, uniform_align);

            let resolved_subpasses = self.resolve_subpasses(&mut arena, subpasses);
            let uniform_gpu = Arc::clone(arena.uniform_gpu.as_ref().expect("uniform_gpu"));
            let resolved_bgs: Vec<Arc<wgpu::BindGroup>> = calls
                .iter()
                .map(|c| self.resolve_call_bind_group(&mut arena, &c.binding, &uniform_gpu))
                .collect();
            let resolve_bg = self.resolve_ssaa_bind_group(&mut arena, &scratch, &uniform_gpu);

            (
                resolved_subpasses,
                calls,
                resolved_bgs,
                resolve_bg,
                [resolve_frame_off, resolve_frag_off],
                r_v_off..r_v_end,
                r_i_off..r_i_end,
                arena.vertex_gpu.clone().expect("vertex_gpu"),
                arena.index_gpu.clone().expect("index_gpu"),
            )
        };

        let stencil_scratch =
            has_stencil.then(|| self.get_or_create_stencil_scratch(ssaa_w, ssaa_h, 1));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Impeller Surface SSAA CommandEncoder"),
            });

        self.record_subpasses(
            &mut encoder,
            &resolved_subpasses,
            Some(&vertex_gpu),
            Some(&index_gpu),
        );

        let pm_clear = clear_color.premultiply();
        {
            let mut pass1 = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Impeller Surface SSAA Pass 1 (2x Supersample)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &scratch.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(pm_clear.red.clamp(0.0, 1.0)),
                            g: f64::from(pm_clear.green.clamp(0.0, 1.0)),
                            b: f64::from(pm_clear.blue.clamp(0.0, 1.0)),
                            a: f64::from(pm_clear.alpha.clamp(0.0, 1.0)),
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: stencil_scratch.as_ref().map(|s| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &s.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Discard,
                        }),
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if !calls.is_empty() {
                Self::record_arena_calls(
                    &mut pass1,
                    &calls,
                    &resolved_bgs,
                    &vertex_gpu,
                    &index_gpu,
                    has_stencil,
                );
            }
        }

        {
            let mut pass2 = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Impeller Surface SSAA Pass 2 (Bilinear 2x2 Downsample)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass2.set_pipeline(&resolve_pipeline);
            pass2.set_bind_group(0, resolve_bg.as_ref(), &resolve_dyn_offsets);
            pass2.set_scissor_rect(0, 0, w, h);
            pass2.set_vertex_buffer(0, vertex_gpu.slice(resolve_vb_range));
            pass2.set_index_buffer(index_gpu.slice(resolve_ib_range), wgpu::IndexFormat::Uint32);
            pass2.draw_indexed(0..6, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.release_save_layer_textures(used_layer_textures);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_to_texture_view(
        &self,
        target_view: &wgpu::TextureView,
        msaa_view: Option<&wgpu::TextureView>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
        clear_color: Color,
        commands: &[DrawCommand],
    ) -> Result<(), String> {
        let w = width.max(1);
        let h = height.max(1);
        if msaa_view.is_none() && sample_count > 1 {
            return self.render_to_texture_view_ssaa(
                target_view,
                w,
                h,
                format,
                2,
                clear_color,
                commands,
            );
        }

        let uniform_align = self.uniform_offset_alignment();
        let effective_samples = if msaa_view.is_some() {
            normalize_sample_count(sample_count)
        } else {
            1
        };
        let ortho = Matrix::make_orthographic(Size::new(w as Scalar, h as Scalar));
        let has_stencil = commands
            .iter()
            .any(|c| matches!(c, DrawCommand::PushClip { .. }));

        let mut used_layer_textures: Vec<Arc<SaveLayerTexture>> = Vec::new();
        let (resolved_subpasses, calls, resolved_bgs, vertex_gpu, index_gpu) = {
            let mut arena = self
                .frame_arena
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            arena.begin_frame(self.atlas_eviction_epoch.load(Ordering::Relaxed));
            let mut subpasses: Vec<PreparedSubpass> = Vec::new();
            let calls = self.prepare_pass_into_arena(
                &mut arena,
                commands,
                &mut subpasses,
                &mut used_layer_textures,
                ortho,
                w,
                h,
                1,
                effective_samples,
                format,
                uniform_align,
                has_stencil,
                Point::ZERO,
            );
            let subpass_draw_count: usize = subpasses.iter().map(|s| s.calls.len()).sum();
            arena.last_frame_draw_call_count = subpass_draw_count + calls.len();
            arena.upload_cpu_buffers(&self.device, &self.queue, uniform_align);

            let resolved_subpasses = self.resolve_subpasses(&mut arena, subpasses);
            let resolved_bgs: Vec<Arc<wgpu::BindGroup>> = if calls.is_empty() {
                Vec::new()
            } else {
                let uniform_gpu = Arc::clone(arena.uniform_gpu.as_ref().expect("uniform_gpu"));
                calls
                    .iter()
                    .map(|c| self.resolve_call_bind_group(&mut arena, &c.binding, &uniform_gpu))
                    .collect()
            };
            (
                resolved_subpasses,
                calls,
                resolved_bgs,
                arena.vertex_gpu.clone(),
                arena.index_gpu.clone(),
            )
        };

        let stencil_scratch =
            has_stencil.then(|| self.get_or_create_stencil_scratch(w, h, effective_samples));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Impeller Surface CommandEncoder"),
            });

        self.record_subpasses(
            &mut encoder,
            &resolved_subpasses,
            vertex_gpu.as_ref(),
            index_gpu.as_ref(),
        );

        let pm_clear = clear_color.premultiply();
        let (attachment_view, resolve_target, store_op) = match msaa_view {
            Some(mv) => (mv, Some(target_view), wgpu::StoreOp::Discard),
            None => (target_view, None, wgpu::StoreOp::Store),
        };
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Impeller Surface RenderPass"),
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
                depth_stencil_attachment: stencil_scratch.as_ref().map(|s| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &s.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Discard,
                        }),
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if !calls.is_empty() {
                if let (Some(vb), Some(ib)) = (vertex_gpu.as_ref(), index_gpu.as_ref()) {
                    Self::record_arena_calls(
                        &mut render_pass,
                        &calls,
                        &resolved_bgs,
                        vb,
                        ib,
                        has_stencil,
                    );
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.release_save_layer_textures(used_layer_textures);
        Ok(())
    }

    fn get_or_create_surface_msaa_scratch(
        &self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Arc<SurfaceMsaaScratch> {
        let w = width.max(1);
        let h = height.max(1);
        let samples = normalize_sample_count(sample_count);
        if let Ok(guard) = self.surface_msaa_scratch.lock() {
            if let Some(existing) = guard.as_ref() {
                if existing.width == w
                    && existing.height == h
                    && existing.format == format
                    && existing.sample_count == samples
                {
                    return Arc::clone(existing);
                }
            }
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Impeller Surface 4x MSAA Scratch Texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let created = Arc::new(SurfaceMsaaScratch {
            width: w,
            height: h,
            format,
            sample_count: samples,
            _texture: texture,
            view,
        });

        if let Ok(mut guard) = self.surface_msaa_scratch.lock() {
            *guard = Some(Arc::clone(&created));
        }
        created
    }

    pub fn render_to_surface_view(
        &self,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        clear_color: Color,
        commands: &[DrawCommand],
    ) -> Result<(), String> {
        let w = width.max(1);
        let h = height.max(1);
        if !self.is_gl_backend {
            let msaa = self.get_or_create_surface_msaa_scratch(w, h, format, 4);
            self.render_to_texture_view(
                target_view,
                Some(&msaa.view),
                w,
                h,
                format,
                4,
                clear_color,
                commands,
            )
        } else {
            let ssaa_scale = if (w as u64) * (h as u64) >= 2_000_000 {
                1
            } else {
                2
            };
            self.render_to_texture_view_ssaa(
                target_view,
                w,
                h,
                format,
                ssaa_scale,
                clear_color,
                commands,
            )
        }
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
        (
            readback_buffer,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        )
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

    /// C++ source: entity_unittests.cc:CanDrawTextureFillWithTileModes
    #[test]
    fn test_texture_fill_sampling_and_tile_modes() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let pixels = vec![
            255, 0, 0, 255, // (0,0) Red
            0, 255, 0, 255, // (1,0) Green
            0, 0, 255, 255, // (0,1) Blue
            255, 255, 0, 255, // (1,1) Yellow
        ];
        let tex = ImageTexture::new(42, 2, 2, pixels);

        let batch_clamp = TextureFillBatch::from_rect(
            Rect::from_xywh(0.0, 0.0, 2.0, 2.0),
            Rect::from_xywh(0.0, 0.0, 16.0, 16.0),
            Matrix::IDENTITY,
            tex.clone(),
            1.0,
            false,
            BlendMode::SrcOver,
            None,
        );

        let batch_decal = TextureFillBatch {
            vertices: vec![
                UvVertex::new(16.0, 0.0, -0.5, -0.5),
                UvVertex::new(32.0, 0.0, 1.5, -0.5),
                UvVertex::new(32.0, 16.0, 1.5, 1.5),
                UvVertex::new(16.0, 16.0, -0.5, 1.5),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            transform: Matrix::IDENTITY,
            texture: tex,
            alpha: 0.5,
            tile_mode_x: TileMode::Decal,
            tile_mode_y: TileMode::Decal,
            linear_filter: false,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let rgba = ctx
            .render_to_rgba(
                32,
                16,
                Color::BLACK,
                &[
                    DrawCommand::TextureFill(batch_clamp),
                    DrawCommand::TextureFill(batch_decal),
                ],
            )
            .expect("render texture fill batches");

        assert_pixel_approx(pixel_at(&rgba, 32, 4, 4), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 32, 12, 4), [0, 255, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 32, 4, 12), [0, 0, 255, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 32, 12, 12), [255, 255, 0, 255], 1);

        // Decal outside [0..1] UV (at x=17, y=1 -> u < 0, v < 0) should be transparent over Black
        assert_pixel_approx(pixel_at(&rgba, 32, 17, 1), [0, 0, 0, 255], 1);
        // Decal inside [0..1] UV (at x=22, y=6 -> u ~ 0.25, v ~ 0.25 -> Red * 0.5 alpha)
        assert_pixel_approx(pixel_at(&rgba, 32, 22, 6), [128, 0, 0, 255], 2);
        assert_eq!(ctx.cached_image_count(), 1);
    }

    /// C++ source: renderer_unittests.cc:CanRenderToTextureViewSSAA
    #[test]
    fn test_render_to_texture_view_ssaa_2x_bilinear_resolve() {
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
        ctx.render_to_texture_view_ssaa(
            target_1x.view(),
            32,
            32,
            TARGET_FORMAT,
            2,
            Color::BLACK,
            std::slice::from_ref(&cmd),
        )
        .expect("render_to_texture_view_ssaa 2x");
        let rgba_ssaa = target_1x.readback_rgba(&ctx).expect("readback ssaa");

        let fractional_ssaa = rgba_ssaa
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] > 0 && px[0] < 255)
            .count();
        assert!(
            fractional_ssaa >= 16,
            "2x SSAA bilinear resolve on 1x surface view should produce fractional anti-aliased pixels, got {fractional_ssaa}"
        );
        assert_pixel_approx(pixel_at(&rgba_ssaa, 32, 28, 4), [255, 255, 255, 255], 1);
        assert_pixel_approx(pixel_at(&rgba_ssaa, 32, 4, 28), [0, 0, 0, 255], 1);
    }

    /// C++ source: typographer_unittests.cc:GlyphAtlasGenerationEvictsStaleTextures
    #[test]
    fn test_atlas_cache_key_o1_generation_and_stale_eviction() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");

        let mut gen0_a = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        gen0_a.generation = 0;
        gen0_a.pixels.fill(64);
        let mut gen0_b = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        gen0_b.generation = 0;
        gen0_b.pixels.fill(192);
        assert_ne!(
            AtlasCacheKey::from_atlas(&gen0_a),
            AtlasCacheKey::from_atlas(&gen0_b),
            "generation == 0 synthetic atlases must still distinguish pixel content via pixel_hash"
        );

        let mut gen1 = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        gen1.generation = 1;
        gen1.pixels.fill(80);
        let key1 = AtlasCacheKey::from_atlas(&gen1);
        assert_eq!(
            key1.pixel_hash, 0,
            "generation > 0 must use O(1) zero pixel_hash"
        );

        let mut gen2 = gen1.clone();
        gen2.grow(32, 32);
        gen2.pixels.fill(255);

        let quad = AtlasGlyphQuad {
            dest_rect: Rect::from_xywh(0.0, 0.0, 16.0, 16.0),
            atlas_rect: Rect::from_xywh(0.0, 0.0, 16.0, 16.0),
            color: Color::WHITE,
            is_color: false,
        };

        let batch1 = GlyphAtlasBatch::from_atlas(
            vec![quad],
            Matrix::IDENTITY,
            Arc::new(gen1),
            BlendMode::SrcOver,
            None,
        );
        let _ = ctx
            .render_to_rgba(16, 16, Color::BLACK, &[DrawCommand::GlyphAtlas(batch1)])
            .expect("render gen 1");
        assert_eq!(ctx.cached_atlas_count(), 1);

        let batch2 = GlyphAtlasBatch::from_atlas(
            vec![quad],
            Matrix::IDENTITY,
            Arc::new(gen2),
            BlendMode::SrcOver,
            None,
        );
        let rgba2 = ctx
            .render_to_rgba(16, 16, Color::BLACK, &[DrawCommand::GlyphAtlas(batch2)])
            .expect("render gen 2 after grow");
        assert_eq!(
            ctx.cached_atlas_count(),
            1,
            "advancing resize_epoch on grow must evict stale resize_epoch texture for the same atlas_id"
        );
        assert!(ctx.atlas_eviction_epoch() >= 1);
        assert_pixel_approx(pixel_at(&rgba2, 16, 8, 8), [255, 255, 255, 255], 1);
    }

    /// C++ source: entity_pass_unittests.cc:CoalescesCompatibleDrawCommands
    #[test]
    fn test_coalesce_draw_commands_solid_fill_and_glyph_atlas() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let unit_rect = make_rect_mesh(Rect::from_xywh(0.0, 0.0, 8.0, 8.0));

        let s1 = DrawCommand::SolidFill {
            mesh: unit_rect.clone(),
            transform: Matrix::make_translation(2.0, 2.0, 0.0),
            color: Color::RED,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let s2 = DrawCommand::SolidFill {
            mesh: unit_rect.clone(),
            transform: Matrix::make_translation(12.0, 2.0, 0.0),
            color: Color::RED,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let s3_diff_color = DrawCommand::SolidFill {
            mesh: unit_rect,
            transform: Matrix::make_translation(22.0, 2.0, 0.0),
            color: Color::GREEN,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        atlas.generation = 1;
        atlas.pixels.fill(255);
        let atlas = Arc::new(atlas);

        let g1 = DrawCommand::GlyphAtlas(
            GlyphAtlasBatch::from_atlas(
                vec![AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(0.0, 0.0, 6.0, 6.0),
                    atlas_rect: Rect::from_xywh(0.0, 0.0, 6.0, 6.0),
                    color: Color::WHITE,
                    is_color: false,
                }],
                Matrix::make_translation(2.0, 16.0, 0.0),
                Arc::clone(&atlas),
                BlendMode::SrcOver,
                None,
            )
            .with_linear_filter(false),
        );
        let g2 = DrawCommand::GlyphAtlas(
            GlyphAtlasBatch::from_atlas(
                vec![AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(0.0, 0.0, 6.0, 6.0),
                    atlas_rect: Rect::from_xywh(0.0, 0.0, 6.0, 6.0),
                    color: Color::BLUE,
                    is_color: false,
                }],
                Matrix::make_translation(12.0, 16.0, 0.0),
                atlas,
                BlendMode::SrcOver,
                None,
            )
            .with_linear_filter(false),
        );

        let commands = vec![s1, s2, s3_diff_color, g1, g2];
        let coalesced = coalesce_draw_commands(&commands);
        assert_eq!(
            coalesced.len(),
            3,
            "2 compatible SolidFills + 1 Green SolidFill + 2 compatible GlyphAtlas runs should coalesce to 3 commands"
        );

        let rgba = ctx
            .render_to_rgba(32, 32, Color::BLACK, &commands)
            .expect("render coalesced commands");
        assert_pixel_approx(pixel_at(&rgba, 32, 5, 5), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 32, 15, 5), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 32, 25, 5), [0, 255, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 32, 4, 18), [255, 255, 255, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 32, 14, 18), [0, 0, 255, 255], 1);
    }

    /// C++ source: renderer_unittests.cc:HostBufferPoolReusesAllocationsAcrossFrames
    #[test]
    fn test_frame_buffer_pool_reuses_buffers_in_steady_state() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new(&ctx, 32, 32);

        let build_frame = |color: Color, x_offset: Scalar| {
            vec![DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(x_offset, 4.0, 12.0, 12.0)),
                transform: Matrix::IDENTITY,
                color,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            }]
        };

        assert_eq!(ctx.pooled_buffer_count(), 0);
        let rgba1 = target
            .render_and_readback(&ctx, Color::BLACK, &build_frame(Color::RED, 2.0))
            .expect("frame 1");
        let after_frame_1 = ctx.pooled_buffer_count();
        assert_eq!(
            after_frame_1, 3,
            "frame 1 should populate 3 consolidated FrameArena GPU buffers (vertex, index, uniform)"
        );
        assert_pixel_approx(pixel_at(&rgba1, 32, 6, 8), [255, 0, 0, 255], 1);

        let rgba2 = target
            .render_and_readback(&ctx, Color::BLACK, &build_frame(Color::GREEN, 16.0))
            .expect("frame 2");
        assert_eq!(
            ctx.pooled_buffer_count(),
            after_frame_1,
            "steady-state frame 2 must reuse pooled buffers without allocating new wgpu::Buffers"
        );
        assert_pixel_approx(pixel_at(&rgba2, 32, 6, 8), [0, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba2, 32, 20, 8), [0, 255, 0, 255], 1);

        let rgba3 = target
            .render_and_readback(&ctx, Color::BLACK, &build_frame(Color::BLUE, 8.0))
            .expect("frame 3");
        assert_eq!(ctx.pooled_buffer_count(), after_frame_1);
        assert_pixel_approx(pixel_at(&rgba3, 32, 12, 8), [0, 0, 255, 255], 1);
    }

    /// C++ source: entity_pass_unittests.cc:CoalescesCrossColorSolidFillAndAffineGlyphRuns
    #[test]
    fn test_phase7_cross_color_solid_fill_and_affine_glyph_coalescing() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 48, 32, 1);
        let unit_rect = make_rect_mesh(Rect::from_xywh(0.0, 0.0, 8.0, 8.0));

        let solid_cmds = vec![
            DrawCommand::SolidFill {
                mesh: unit_rect.clone(),
                transform: Matrix::make_translation(2.0, 2.0, 0.0),
                color: Color::RED,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: unit_rect.clone(),
                transform: Matrix::make_translation(12.0, 2.0, 0.0),
                color: Color::GREEN,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: unit_rect.clone(),
                transform: Matrix::make_translation(22.0, 2.0, 0.0)
                    * Matrix::make_scale(1.25, 1.25, 1.0),
                color: Color::BLUE,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: unit_rect,
                transform: Matrix::make_translation(36.0, 2.0, 0.0),
                color: Color::new(1.0, 1.0, 0.0, 1.0),
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
        ];

        let rgba_solid = target
            .render_and_readback(&ctx, Color::BLACK, &solid_cmds)
            .expect("render coalesced cross-color solid fills");
        assert_eq!(
            ctx.last_frame_draw_call_count(),
            1,
            "4 consecutive 2D-affine SolidFill commands with distinct colors must coalesce into 1 GPU draw call"
        );
        assert_pixel_approx(pixel_at(&rgba_solid, 48, 5, 5), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba_solid, 48, 15, 5), [0, 255, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba_solid, 48, 26, 6), [0, 0, 255, 255], 1);
        assert_pixel_approx(pixel_at(&rgba_solid, 48, 39, 5), [255, 255, 0, 255], 1);

        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        atlas.generation = 1;
        atlas.pixels.fill(255);
        let atlas = Arc::new(atlas);

        let glyph_cmds = vec![
            DrawCommand::GlyphAtlas(
                GlyphAtlasBatch::from_atlas(
                    vec![AtlasGlyphQuad {
                        dest_rect: Rect::from_xywh(0.0, 0.0, 8.0, 8.0),
                        atlas_rect: Rect::from_xywh(0.0, 0.0, 8.0, 8.0),
                        color: Color::RED,
                        is_color: false,
                    }],
                    Matrix::make_translation(2.0, 16.0, 0.0),
                    Arc::clone(&atlas),
                    BlendMode::SrcOver,
                    None,
                )
                .with_linear_filter(true),
            ),
            DrawCommand::GlyphAtlas(
                GlyphAtlasBatch::from_atlas(
                    vec![AtlasGlyphQuad {
                        dest_rect: Rect::from_xywh(0.0, 0.0, 4.0, 4.0),
                        atlas_rect: Rect::from_xywh(0.0, 0.0, 8.0, 8.0),
                        color: Color::GREEN,
                        is_color: false,
                    }],
                    Matrix::make_translation(16.0, 16.0, 0.0) * Matrix::make_scale(2.0, 2.0, 1.0),
                    atlas,
                    BlendMode::SrcOver,
                    None,
                )
                .with_linear_filter(true),
            ),
        ];

        let rgba_glyph = target
            .render_and_readback(&ctx, Color::BLACK, &glyph_cmds)
            .expect("render coalesced 2D-affine glyph runs");
        assert_eq!(
            ctx.last_frame_draw_call_count(),
            1,
            "2 consecutive 2D-affine GlyphAtlas commands with translation + scale must coalesce into 1 GPU draw call"
        );
        assert_pixel_approx(pixel_at(&rgba_glyph, 48, 5, 19), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba_glyph, 48, 19, 19), [0, 255, 0, 255], 1);
    }

    /// C++ source: renderer_unittests.cc:FrameArenaSingleUploadAndBindGroupReuseUnder500Commands
    #[test]
    fn test_phase7_frame_arena_500_commands_single_upload_and_bindgroup_reuse() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 64, 64, 1);
        let unit_rect = make_rect_mesh(Rect::from_xywh(0.0, 0.0, 4.0, 4.0));

        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 8, 8);
        atlas.generation = 1;
        atlas.pixels.fill(255);
        let atlas = Arc::new(atlas);
        let tex = ImageTexture::new(99, 2, 2, vec![255; 16]);

        let mut commands = Vec::with_capacity(500);
        for i in 0..495 {
            let x = ((i % 15) * 4) as Scalar;
            let y = (((i / 15) % 15) * 4) as Scalar;
            let color = match i % 4 {
                0 => Color::RED,
                1 => Color::GREEN,
                2 => Color::BLUE,
                _ => Color::WHITE,
            };
            commands.push(DrawCommand::SolidFill {
                mesh: unit_rect.clone(),
                transform: Matrix::make_translation(x, y, 0.0),
                color,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            });
        }
        commands.push(DrawCommand::LinearGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 8.0, 8.0)),
            transform: Matrix::IDENTITY,
            gradient: LinearGradientParams::two_stop(
                Point::new(0.0, 0.0),
                Point::new(8.0, 8.0),
                Color::RED,
                Color::BLUE,
                TileMode::Clamp,
            ),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        });
        commands.push(DrawCommand::RadialGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(8.0, 0.0, 8.0, 8.0)),
            transform: Matrix::IDENTITY,
            gradient: RadialGradientParams::two_stop(
                Point::new(12.0, 4.0),
                4.0,
                Color::GREEN,
                Color::WHITE,
                TileMode::Clamp,
            ),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        });
        commands.push(DrawCommand::RRectBlur {
            transform: Matrix::IDENTITY,
            params: RRectBlurParams::new(
                Rect::from_xywh(16.0, 0.0, 8.0, 8.0),
                2.0,
                1.5,
                Color::WHITE,
            ),
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        });
        commands.push(DrawCommand::GlyphAtlas(GlyphAtlasBatch::from_atlas(
            vec![AtlasGlyphQuad {
                dest_rect: Rect::from_xywh(24.0, 0.0, 8.0, 8.0),
                atlas_rect: Rect::from_xywh(0.0, 0.0, 8.0, 8.0),
                color: Color::WHITE,
                is_color: false,
            }],
            Matrix::IDENTITY,
            atlas,
            BlendMode::SrcOver,
            None,
        )));
        commands.push(DrawCommand::TextureFill(TextureFillBatch::from_rect(
            Rect::from_xywh(0.0, 0.0, 2.0, 2.0),
            Rect::from_xywh(32.0, 0.0, 8.0, 8.0),
            Matrix::IDENTITY,
            tex,
            1.0,
            true,
            BlendMode::SrcOver,
            None,
        )));
        assert_eq!(commands.len(), 500);

        let _ = target
            .render_and_readback(&ctx, Color::BLACK, &commands)
            .expect("render 500-command frame 1");
        assert!(
            ctx.last_frame_buffer_upload_count() <= 3,
            "500-command frame 1 must execute <= 3 queue.write_buffer calls, got {}",
            ctx.last_frame_buffer_upload_count()
        );
        assert_eq!(
            ctx.last_frame_draw_call_count(),
            6,
            "495 multi-color SolidFills + 5 distinct pipeline draws must coalesce to 6 GPU draw calls"
        );
        let cached_after_f1 = ctx.cached_bind_group_count();
        assert!(cached_after_f1 >= 6);

        let _ = target
            .render_and_readback(&ctx, Color::BLACK, &commands)
            .expect("render 500-command frame 2");
        assert!(
            ctx.last_frame_buffer_upload_count() <= 3,
            "500-command frame 2 must execute <= 3 queue.write_buffer calls, got {}",
            ctx.last_frame_buffer_upload_count()
        );
        assert_eq!(
            ctx.last_frame_bind_groups_created(),
            0,
            "steady-state frame 2 must reuse cached BindGroups with zero new allocations"
        );
        assert_eq!(ctx.cached_bind_group_count(), cached_after_f1);

        ctx.render_to_texture_view_ssaa(
            target.view(),
            64,
            64,
            TARGET_FORMAT,
            2,
            Color::BLACK,
            &commands,
        )
        .expect("render 500-command SSAA 2x frame");
        assert!(
            ctx.last_frame_buffer_upload_count() <= 3,
            "2-pass SSAA resolve must also execute <= 3 queue.write_buffer calls per frame, got {}",
            ctx.last_frame_buffer_upload_count()
        );
    }

    /// C++ source: typographer_unittests.cc:GlyphAtlasIncrementalDirtyRowUploadReusesTexture
    #[test]
    fn test_phase8_incremental_sync_glyph_atlas_dirty_row_upload_and_zero_eviction() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 16, 16, 1);

        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        atlas.pixels[0..(8 * 16)].fill(255);
        atlas.generation = 1;
        atlas.row_generations[0..8].fill(1);
        atlas.row_generations[8..16].fill(1);

        let gpu1 = ctx.sync_glyph_atlas(&atlas);
        assert_eq!(ctx.cached_atlas_count(), 1);
        assert_eq!(ctx.atlas_eviction_epoch(), 0);

        let desc1 = Arc::new(atlas.descriptor_without_pixels());
        assert!(desc1.pixels.is_empty());

        let quad = AtlasGlyphQuad {
            dest_rect: Rect::from_xywh(0.0, 0.0, 16.0, 16.0),
            atlas_rect: Rect::from_xywh(0.0, 0.0, 16.0, 16.0),
            color: Color::WHITE,
            is_color: false,
        };
        let cmd1 = DrawCommand::GlyphAtlas(GlyphAtlasBatch::from_atlas(
            vec![quad],
            Matrix::IDENTITY,
            desc1,
            BlendMode::SrcOver,
            None,
        ));
        let rgba1 = target
            .render_and_readback(&ctx, Color::BLACK, &[cmd1])
            .expect("render frame 1 with descriptor_without_pixels");
        assert_pixel_approx(pixel_at(&rgba1, 16, 8, 4), [255, 255, 255, 255], 1);
        assert_pixel_approx(pixel_at(&rgba1, 16, 8, 12), [0, 0, 0, 255], 1);

        // Incremental dirty-row update on rows 8..16 without resizing.
        atlas.pixels[(8 * 16)..(16 * 16)].fill(255);
        atlas.generation = 2;
        atlas.row_generations[8..16].fill(2);
        assert_eq!(atlas.dirty_row_range_since(1), Some((8, 16)));

        let gpu2 = ctx.sync_glyph_atlas(&atlas);
        assert!(
            Arc::ptr_eq(&gpu1, &gpu2),
            "incremental dirty-row sync must reuse the exact same CachedGpuAtlas Arc"
        );
        assert_eq!(
            ctx.atlas_eviction_epoch(),
            0,
            "incremental dirty-row sync must not increment atlas_eviction_epoch"
        );
        assert_eq!(ctx.cached_atlas_count(), 1);

        let desc2 = Arc::new(atlas.descriptor_without_pixels());
        let cmd2 = DrawCommand::GlyphAtlas(GlyphAtlasBatch::from_atlas(
            vec![quad],
            Matrix::IDENTITY,
            desc2,
            BlendMode::SrcOver,
            None,
        ));
        let rgba2 = target
            .render_and_readback(&ctx, Color::BLACK, &[cmd2])
            .expect("render frame 2 after dirty-row sync");
        assert_eq!(
            ctx.last_frame_bind_groups_created(),
            0,
            "reusing CachedGpuAtlas without eviction must keep cached BindGroups intact"
        );
        assert_pixel_approx(pixel_at(&rgba2, 16, 8, 4), [255, 255, 255, 255], 1);
        assert_pixel_approx(pixel_at(&rgba2, 16, 8, 12), [255, 255, 255, 255], 1);
    }

    /// C++ source: entity_pass_unittests.cc:ClustersNonOverlappingInterleavedDrawsPreservingPainterOrder
    #[test]
    fn test_phase8_non_overlapping_lookback_clustering_and_painter_order() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 48, 32, 1);
        let unit_rect = make_rect_mesh(Rect::from_xywh(0.0, 0.0, 6.0, 6.0));

        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 8, 8);
        atlas.generation = 1;
        atlas.pixels.fill(255);
        let _ = ctx.sync_glyph_atlas(&atlas);
        let atlas_desc = Arc::new(atlas.descriptor_without_pixels());

        let non_overlapping = vec![
            DrawCommand::SolidFill {
                mesh: unit_rect.clone(),
                transform: Matrix::make_translation(2.0, 4.0, 0.0),
                color: Color::RED,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::GlyphAtlas(GlyphAtlasBatch::from_atlas(
                vec![AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(0.0, 0.0, 6.0, 6.0),
                    atlas_rect: Rect::from_xywh(0.0, 0.0, 6.0, 6.0),
                    color: Color::GREEN,
                    is_color: false,
                }],
                Matrix::make_translation(12.0, 4.0, 0.0),
                Arc::clone(&atlas_desc),
                BlendMode::SrcOver,
                None,
            )),
            DrawCommand::SolidFill {
                mesh: unit_rect,
                transform: Matrix::make_translation(22.0, 4.0, 0.0),
                color: Color::BLUE,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::GlyphAtlas(GlyphAtlasBatch::from_atlas(
                vec![AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(0.0, 0.0, 6.0, 6.0),
                    atlas_rect: Rect::from_xywh(0.0, 0.0, 6.0, 6.0),
                    color: Color::WHITE,
                    is_color: false,
                }],
                Matrix::make_translation(32.0, 4.0, 0.0),
                Arc::clone(&atlas_desc),
                BlendMode::SrcOver,
                None,
            )),
        ];

        let rgba = target
            .render_and_readback(&ctx, Color::BLACK, &non_overlapping)
            .expect("render non-overlapping interleaved draws");
        assert_eq!(
            ctx.last_frame_draw_call_count(),
            2,
            "alternating non-overlapping [SolidFill, GlyphAtlas, SolidFill, GlyphAtlas] must cluster and coalesce into 2 draw calls"
        );
        assert_pixel_approx(pixel_at(&rgba, 48, 4, 6), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 48, 14, 6), [0, 255, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 48, 24, 6), [0, 0, 255, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 48, 34, 6), [255, 255, 255, 255], 1);

        // Verify single-pass hardware MSAA / DPR-aware surface render path executes cleanly.
        ctx.render_to_surface_view(
            target.view(),
            48,
            32,
            TARGET_FORMAT,
            Color::BLACK,
            &non_overlapping,
        )
        .expect("render_to_surface_view non-overlapping interleaved draws");
        let rgba_surface = target.readback_rgba(&ctx).expect("readback surface view");
        assert_pixel_approx(pixel_at(&rgba_surface, 48, 4, 6), [255, 0, 0, 255], 2);
        assert_pixel_approx(pixel_at(&rgba_surface, 48, 14, 6), [0, 255, 0, 255], 2);

        // Overlapping commands must NOT reorder across each other: strict painter's order preserved.
        let wide_rect = make_rect_mesh(Rect::from_xywh(0.0, 0.0, 16.0, 16.0));
        let overlapping = vec![
            DrawCommand::SolidFill {
                mesh: wide_rect.clone(),
                transform: Matrix::IDENTITY,
                color: Color::RED,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::GlyphAtlas(GlyphAtlasBatch::from_atlas(
                vec![AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(0.0, 0.0, 16.0, 16.0),
                    atlas_rect: Rect::from_xywh(0.0, 0.0, 8.0, 8.0),
                    color: Color::GREEN,
                    is_color: false,
                }],
                Matrix::IDENTITY,
                atlas_desc,
                BlendMode::SrcOver,
                None,
            )),
            DrawCommand::SolidFill {
                mesh: wide_rect,
                transform: Matrix::make_translation(8.0, 0.0, 0.0),
                color: Color::BLUE,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
        ];

        let rgba_overlap = target
            .render_and_readback(&ctx, Color::BLACK, &overlapping)
            .expect("render overlapping interleaved draws");
        assert_eq!(
            ctx.last_frame_draw_call_count(),
            3,
            "overlapping SolidFill must not move before intervening overlapping GlyphAtlas"
        );
        assert_pixel_approx(pixel_at(&rgba_overlap, 48, 4, 8), [0, 255, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba_overlap, 48, 12, 8), [0, 0, 255, 255], 1);
    }

    /// C++ source: entity_pass_unittests.cc:Clusters64InterleavedCellsIntoConstantSlotCount
    #[test]
    fn test_phase9_cluster_64_cells_solid_glyph_rrect_blur_into_3_slots() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 256, 256, 1);
        let unit_rect = make_rect_mesh(Rect::from_xywh(0.0, 0.0, 8.0, 8.0));

        let mut atlas = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 8, 8);
        atlas.generation = 1;
        atlas.pixels.fill(255);
        let _ = ctx.sync_glyph_atlas(&atlas);
        let atlas_desc = Arc::new(atlas.descriptor_without_pixels());

        let mut commands = Vec::with_capacity(64 * 3);
        for cell in 0..64 {
            let col = (cell % 8) as Scalar;
            let row = (cell / 8) as Scalar;
            let base_x = col * 32.0 + 4.0;
            let base_y = row * 32.0 + 4.0;

            commands.push(DrawCommand::SolidFill {
                mesh: unit_rect.clone(),
                transform: Matrix::make_translation(base_x, base_y, 0.0),
                color: Color::RED,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            });
            commands.push(DrawCommand::GlyphAtlas(GlyphAtlasBatch::from_atlas(
                vec![AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(0.0, 0.0, 6.0, 6.0),
                    atlas_rect: Rect::from_xywh(0.0, 0.0, 6.0, 6.0),
                    color: Color::GREEN,
                    is_color: false,
                }],
                Matrix::make_translation(base_x + 10.0, base_y, 0.0),
                Arc::clone(&atlas_desc),
                BlendMode::SrcOver,
                None,
            )));
            commands.push(DrawCommand::RRectBlur {
                transform: Matrix::make_translation(base_x, base_y + 12.0, 0.0),
                params: RRectBlurParams::new(
                    Rect::from_xywh(0.0, 0.0, 8.0, 6.0),
                    2.0,
                    1.0,
                    Color::BLUE,
                ),
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            });
        }
        assert_eq!(commands.len(), 192);

        let slots = cluster_commands_into_slots(&commands);
        assert!(
            slots.len() <= 3,
            "64 non-overlapping cells of [SolidFill, GlyphAtlas, RRectBlur] must cluster into <= 3 ClusterSlots, got {}",
            slots.len()
        );
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].cmds.len(), 64);
        assert_eq!(slots[1].cmds.len(), 64);
        assert_eq!(slots[2].cmds.len(), 64);

        let rgba = target
            .render_and_readback(&ctx, Color::BLACK, &commands)
            .expect("render 64 clustered cells");
        // 64 SolidFills coalesce to 1 draw call + 64 GlyphAtlases coalesce to 1 draw call + 64 RRectBlurs = 66 draw calls (down from 192 pipeline switches to 3!).
        assert_eq!(ctx.last_frame_draw_call_count(), 66);
        assert_pixel_approx(pixel_at(&rgba, 256, 6, 6), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba, 256, 16, 6), [0, 255, 0, 255], 1);
    }

    /// C++ source: entity_unittests.cc:ClipPathIntersect
    #[test]
    fn test_stencil_clip_intersect_triangle_and_msaa() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let tri_mesh = TessellatedMesh {
            vertices: vec![
                Vertex2D::new(32.0, 4.0),
                Vertex2D::new(60.0, 60.0),
                Vertex2D::new(4.0, 60.0),
            ],
            indices: vec![0, 1, 2],
        };

        let commands = vec![
            DrawCommand::PushClip {
                mesh: tri_mesh,
                transform: Matrix::IDENTITY,
                is_difference: false,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
                transform: Matrix::IDENTITY,
                color: Color::RED,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::PopClip { clip_rect: None },
            // After PopClip, drawing outside the triangle must succeed.
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(2.0, 2.0, 10.0, 10.0)),
                transform: Matrix::IDENTITY,
                color: Color::GREEN,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
        ];

        // 1x sample count target
        let target_1x = OffscreenTarget::new_with_sample_count(&ctx, 64, 64, 1);
        let rgba_1x = target_1x
            .render_and_readback(&ctx, Color::BLACK, &commands)
            .expect("stencil clip 1x render");
        // Inside triangle (32, 48) -> RED
        assert_pixel_approx(pixel_at(&rgba_1x, 64, 32, 48), [255, 0, 0, 255], 1);
        // Outside triangle top-right (56, 8) -> clipped during RED draw, stays BLACK
        assert_pixel_approx(pixel_at(&rgba_1x, 64, 56, 8), [0, 0, 0, 255], 1);
        // Outside triangle top-left (6, 6) -> GREEN drawn after PopClip
        assert_pixel_approx(pixel_at(&rgba_1x, 64, 6, 6), [0, 255, 0, 255], 1);

        // 4x MSAA target (default)
        let rgba_msaa = ctx
            .render_to_rgba(64, 64, Color::BLACK, &commands)
            .expect("stencil clip 4x msaa render");
        assert_pixel_approx(pixel_at(&rgba_msaa, 64, 32, 48), [255, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba_msaa, 64, 56, 8), [0, 0, 0, 255], 1);
        assert_pixel_approx(pixel_at(&rgba_msaa, 64, 6, 6), [0, 255, 0, 255], 1);
    }

    /// C++ source: entity_unittests.cc:NestedClipPathIntersect
    #[test]
    fn test_stencil_clip_nested_intersect_and_pop() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 64, 64, 1);

        let outer_mesh = make_rect_mesh(Rect::from_xywh(8.0, 8.0, 36.0, 48.0)); // x: 8..44, y: 8..56
        let inner_mesh = make_rect_mesh(Rect::from_xywh(20.0, 8.0, 36.0, 24.0)); // x: 20..56, y: 8..32

        let commands = vec![
            DrawCommand::PushClip {
                mesh: outer_mesh,
                transform: Matrix::IDENTITY,
                is_difference: false,
                clip_rect: None,
            },
            DrawCommand::PushClip {
                mesh: inner_mesh,
                transform: Matrix::IDENTITY,
                is_difference: false,
                clip_rect: None,
            },
            // Only intersection (x: 20..44, y: 8..32) should receive RED (stencil == 2).
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
                transform: Matrix::IDENTITY,
                color: Color::RED,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::PopClip { clip_rect: None },
            // After popping inner clip, outer clip (x: 8..44, y: 8..56) is active (stencil == 1).
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(0.0, 40.0, 64.0, 12.0)),
                transform: Matrix::IDENTITY,
                color: Color::BLUE,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::PopClip { clip_rect: None },
        ];

        let rgba = target
            .render_and_readback(&ctx, Color::BLACK, &commands)
            .expect("nested stencil clip render");
        // Inside both clips (30, 20) -> RED
        assert_pixel_approx(pixel_at(&rgba, 64, 30, 20), [255, 0, 0, 255], 1);
        // Inside outer only, top-left (14, 20) -> excluded by inner clip -> BLACK
        assert_pixel_approx(pixel_at(&rgba, 64, 14, 20), [0, 0, 0, 255], 1);
        // Inside inner only, top-right (50, 20) -> excluded by outer clip -> BLACK
        assert_pixel_approx(pixel_at(&rgba, 64, 50, 20), [0, 0, 0, 255], 1);
        // Inside outer clip after inner PopClip (16, 46) -> BLUE
        assert_pixel_approx(pixel_at(&rgba, 64, 16, 46), [0, 0, 255, 255], 1);
        // Outside outer clip after inner PopClip (52, 46) -> BLACK
        assert_pixel_approx(pixel_at(&rgba, 64, 52, 46), [0, 0, 0, 255], 1);
    }

    /// C++ source: entity_unittests.cc:ClipPathDifference
    #[test]
    fn test_stencil_clip_difference_punches_hole() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 64, 64, 1);

        let hole_mesh = make_rect_mesh(Rect::from_xywh(20.0, 20.0, 24.0, 24.0));
        let commands = vec![
            DrawCommand::PushClip {
                mesh: hole_mesh,
                transform: Matrix::IDENTITY,
                is_difference: true,
                clip_rect: None,
            },
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(4.0, 4.0, 56.0, 56.0)),
                transform: Matrix::IDENTITY,
                color: Color::GREEN,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::PopClip { clip_rect: None },
            // Verify stencil depth returns to 0 after popping a Difference clip.
            DrawCommand::SolidFill {
                mesh: make_rect_mesh(Rect::from_xywh(28.0, 28.0, 8.0, 8.0)),
                transform: Matrix::IDENTITY,
                color: Color::RED,
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
        ];

        let rgba = target
            .render_and_readback(&ctx, Color::BLACK, &commands)
            .expect("stencil difference clip render");
        // Outer region (10, 10) -> GREEN
        assert_pixel_approx(pixel_at(&rgba, 64, 10, 10), [0, 255, 0, 255], 1);
        // Inside difference hole but outside post-pop RED square (22, 22) -> BLACK
        assert_pixel_approx(pixel_at(&rgba, 64, 22, 22), [0, 0, 0, 255], 1);
        // Inside post-pop RED square (32, 32) -> RED
        assert_pixel_approx(pixel_at(&rgba, 64, 32, 32), [255, 0, 0, 255], 1);
    }

    /// C++ source: entity_unittests.cc:SweepGradientContents
    #[test]
    fn test_sweep_gradient_fill_full_and_sector_decal() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 64, 64, 1);

        let params = SweepGradientParams::new(
            Point::new(32.0, 32.0),
            0.0,
            360.0,
            vec![
                GradientStop::new(0.0, Color::RED),
                GradientStop::new(0.25, Color::GREEN),
                GradientStop::new(0.5, Color::BLUE),
                GradientStop::new(0.75, Color::new(1.0, 1.0, 0.0, 1.0)),
                GradientStop::new(1.0, Color::RED),
            ],
            TileMode::Clamp,
            1.0,
        );
        let cmd = DrawCommand::from(SweepGradientBatch {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
            transform: Matrix::IDENTITY,
            gradient: params,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        });

        let rgba = target
            .render_and_readback(&ctx, Color::BLACK, &[cmd])
            .expect("sweep gradient full render");
        // 90 deg (+Y down from center): (32, 52) -> t = 0.25 -> GREEN
        assert_pixel_approx(pixel_at(&rgba, 64, 32, 52), [0, 255, 0, 255], 6);
        // 180 deg (-X left from center): (12, 32) -> t = 0.50 -> BLUE
        assert_pixel_approx(pixel_at(&rgba, 64, 12, 32), [0, 0, 255, 255], 6);
        // 270 deg (-Y up from center): (32, 12) -> t = 0.75 -> YELLOW
        assert_pixel_approx(pixel_at(&rgba, 64, 32, 12), [255, 255, 0, 255], 6);

        // Sector sweep [90..270] with TileMode::Decal
        let decal_params = SweepGradientParams::two_stop(
            Point::new(32.0, 32.0),
            90.0,
            270.0,
            Color::RED,
            Color::BLUE,
            TileMode::Decal,
        );
        let decal_cmd = DrawCommand::SweepGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
            transform: Matrix::IDENTITY,
            gradient: decal_params,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let rgba_decal = target
            .render_and_readback(&ctx, Color::BLACK, &[decal_cmd])
            .expect("sweep gradient decal render");
        // 180 deg (-X left, (12, 32)) is midpoint of [90..270] -> t = 0.5 -> purple [128, 0, 128, 255]
        assert_pixel_approx(pixel_at(&rgba_decal, 64, 12, 32), [128, 0, 128, 255], 6);
        // ~14 deg (+X right-down, (52, 37)) is outside [90..270] -> Decal transparent -> BLACK
        assert_pixel_approx(pixel_at(&rgba_decal, 64, 52, 37), [0, 0, 0, 255], 1);
    }

    /// C++ source: entity_unittests.cc:ConicalGradientContents
    #[test]
    fn test_conical_gradient_fill_radial_strip_and_conical() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 64, 64, 1);

        // 1. Concentric two-radius (kind = 1: Radial ring from r0=8 to r1=28)
        let radial_params = ConicalGradientParams::two_stop(
            Point::new(32.0, 32.0),
            8.0,
            Point::new(32.0, 32.0),
            28.0,
            Color::RED,
            Color::BLUE,
            TileMode::Clamp,
        );
        let radial_cmd = DrawCommand::from(ConicalGradientBatch {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
            transform: Matrix::IDENTITY,
            gradient: radial_params,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        });
        let rgba_radial = target
            .render_and_readback(&ctx, Color::BLACK, &[radial_cmd])
            .expect("conical concentric radial render");
        // Center (d = 0 <= 8) -> clamped t = 0 -> RED
        assert_pixel_approx(pixel_at(&rgba_radial, 64, 32, 32), [255, 0, 0, 255], 2);
        // Midpoint of ring (d = 18 -> (50, 32)) -> t = 0.5 -> purple [128, 0, 128, 255]
        assert_pixel_approx(pixel_at(&rgba_radial, 64, 50, 32), [128, 0, 128, 255], 8);
        // Outside outer radius (d = 30 -> (62, 32)) -> clamped t = 1 -> BLUE
        assert_pixel_approx(pixel_at(&rgba_radial, 64, 62, 32), [0, 0, 255, 255], 2);

        // 2. Equal-radius strip (kind = 2: Strip between (16, 32) and (48, 32) with r = 16, Decal)
        let strip_params = ConicalGradientParams::two_stop(
            Point::new(16.0, 32.0),
            16.0,
            Point::new(48.0, 32.0),
            16.0,
            Color::RED,
            Color::GREEN,
            TileMode::Decal,
        );
        let strip_cmd = DrawCommand::ConicalGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
            transform: Matrix::IDENTITY,
            gradient: strip_params,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let rgba_strip = target
            .render_and_readback(&ctx, Color::BLACK, &[strip_cmd])
            .expect("conical strip render");
        // Outside strip cylinder (|y - 32| = 24 > 16 at (32, 8)) -> invalid t -> BLACK
        assert_pixel_approx(pixel_at(&rgba_strip, 64, 32, 8), [0, 0, 0, 255], 1);
        // Inside strip at start_center (16, 32) -> pt = (0, 0), t = 0 + sqrt(0.25) = 0.5 -> [128, 128, 0, 255]
        assert_pixel_approx(pixel_at(&rgba_strip, 64, 16, 32), [128, 128, 0, 255], 8);

        // 3. Eccentric two-point conical (kind = 3: focal (20, 32) r0=0 -> center (36, 32) r1=24)
        let conical_params = ConicalGradientParams::new(
            Point::new(20.0, 32.0),
            0.0,
            Point::new(36.0, 32.0),
            24.0,
            vec![
                GradientStop::new(0.0, Color::RED),
                GradientStop::new(0.5, Color::GREEN),
                GradientStop::new(1.0, Color::BLUE),
            ],
            TileMode::Clamp,
            1.0,
        );
        let conical_cmd = DrawCommand::ConicalGradientFill {
            mesh: make_rect_mesh(Rect::from_xywh(0.0, 0.0, 64.0, 64.0)),
            transform: Matrix::IDENTITY,
            gradient: conical_params,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let rgba_conical = target
            .render_and_readback(&ctx, Color::BLACK, &[conical_cmd])
            .expect("conical two-point render");
        // Near focal point (20, 32) -> t ~ 0.0 -> RED-dominant
        let focal_px = pixel_at(&rgba_conical, 64, 20, 32);
        assert!(
            focal_px[0] > 230 && focal_px[2] < 20,
            "focal pixel should be near RED, got {focal_px:?}"
        );
        // At outer edge +X (36 + 24 = 60, 32) -> t = 1.0 -> BLUE
        assert_pixel_approx(pixel_at(&rgba_conical, 64, 60, 32), [0, 0, 255, 255], 8);
    }

    /// C++ source: entity_unittests.cc:SubpassSaveLayerGroupOpacity
    #[test]
    fn test_save_layer_group_opacity_and_pool_reuse() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 64, 64, 1);

        let rect1 = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(10.0, 10.0, 25.0, 25.0)),
            transform: Matrix::IDENTITY,
            color: Color::WHITE,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let rect2 = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(20.0, 20.0, 25.0, 25.0)),
            transform: Matrix::IDENTITY,
            color: Color::WHITE,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let save_layer = DrawCommand::SaveLayer {
            bounds: Rect::from_xywh(10.0, 10.0, 35.0, 35.0),
            commands: vec![rect1, rect2],
            alpha: 0.5,
            blend_mode: BlendMode::SrcOver,
            color_filter: None,
            image_filter: None,
            clip_rect: None,
        };

        let rgba = target
            .render_and_readback(&ctx, Color::BLACK, std::slice::from_ref(&save_layer))
            .expect("savelayer group opacity render");
        // Non-overlapping region (15, 15) and overlapping region (25, 25) must both be ~128 gray,
        // NOT ~191 (0.75) from naive per-command opacity double-blending.
        assert_pixel_approx(pixel_at(&rgba, 64, 15, 15), [128, 128, 128, 255], 3);
        assert_pixel_approx(pixel_at(&rgba, 64, 25, 25), [128, 128, 128, 255], 3);
        assert_pixel_approx(pixel_at(&rgba, 64, 40, 40), [128, 128, 128, 255], 3);
        assert_pixel_approx(pixel_at(&rgba, 64, 5, 5), [0, 0, 0, 255], 1);

        let pooled_after_first = ctx.pooled_save_layer_texture_count();
        assert_eq!(pooled_after_first, 1);

        // Render a second frame with identical SaveLayer dimensions and verify zero new allocations.
        let _ = target
            .render_and_readback(&ctx, Color::BLACK, &[save_layer])
            .expect("savelayer second frame render");
        assert_eq!(ctx.pooled_save_layer_texture_count(), pooled_after_first);
    }

    /// C++ source: entity_unittests.cc:GaussianBlurFilterContents
    #[test]
    fn test_save_layer_gaussian_blur_image_filter() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 64, 64, 1);

        let inner_rect = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(24.0, 24.0, 16.0, 16.0)),
            transform: Matrix::IDENTITY,
            color: Color::WHITE,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        let blur_layer = DrawCommand::SaveLayer {
            bounds: Rect::from_xywh(24.0, 24.0, 16.0, 16.0),
            commands: vec![inner_rect],
            alpha: 1.0,
            blend_mode: BlendMode::SrcOver,
            color_filter: None,
            image_filter: Some(LayerImageFilter::Blur {
                sigma_x: 4.0,
                sigma_y: 4.0,
                tile_mode: TileMode::Decal,
            }),
            clip_rect: None,
        };

        let rgba = target
            .render_and_readback(&ctx, Color::BLACK, &[blur_layer])
            .expect("savelayer gaussian blur render");
        let center = pixel_at(&rgba, 64, 32, 32);
        let edge = pixel_at(&rgba, 64, 24, 32);
        let halo = pixel_at(&rgba, 64, 20, 32);
        let far = pixel_at(&rgba, 64, 2, 2);

        assert!(
            center[0] > 180,
            "blur center should remain bright, got {center:?}"
        );
        assert!(
            edge[0] > 60 && edge[0] < center[0],
            "blur edge should be softer than center, got edge={edge:?} center={center:?}"
        );
        assert!(
            halo[0] > 15 && halo[0] < edge[0],
            "blur halo outside original bounds should receive diffused light, got halo={halo:?}"
        );
        assert_pixel_approx(far, [0, 0, 0, 255], 2);
    }

    /// C++ source: entity_unittests.cc:MorphologyFilterContents
    #[test]
    fn test_save_layer_morphology_dilate_and_erode() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 64, 64, 1);

        let base_rect = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(26.0, 26.0, 12.0, 12.0)),
            transform: Matrix::IDENTITY,
            color: Color::WHITE,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };

        // 1. Dilate by 4px: [26..38] expands to [22..42]
        let dilate_layer = DrawCommand::SaveLayer {
            bounds: Rect::from_xywh(26.0, 26.0, 12.0, 12.0),
            commands: vec![base_rect.clone()],
            alpha: 1.0,
            blend_mode: BlendMode::SrcOver,
            color_filter: None,
            image_filter: Some(LayerImageFilter::Dilate {
                radius_x: 4.0,
                radius_y: 4.0,
            }),
            clip_rect: None,
        };
        let rgba_dilate = target
            .render_and_readback(&ctx, Color::BLACK, &[dilate_layer])
            .expect("savelayer dilate render");
        assert_pixel_approx(pixel_at(&rgba_dilate, 64, 23, 32), [255, 255, 255, 255], 2);
        assert_pixel_approx(pixel_at(&rgba_dilate, 64, 18, 32), [0, 0, 0, 255], 2);

        // 2. Erode by 3px: [26..38] shrinks to [29..35]
        let erode_layer = DrawCommand::SaveLayer {
            bounds: Rect::from_xywh(26.0, 26.0, 12.0, 12.0),
            commands: vec![base_rect],
            alpha: 1.0,
            blend_mode: BlendMode::SrcOver,
            color_filter: None,
            image_filter: Some(LayerImageFilter::Erode {
                radius_x: 3.0,
                radius_y: 3.0,
            }),
            clip_rect: None,
        };
        let rgba_erode = target
            .render_and_readback(&ctx, Color::BLACK, &[erode_layer])
            .expect("savelayer erode render");
        assert_pixel_approx(pixel_at(&rgba_erode, 64, 27, 32), [0, 0, 0, 255], 2);
        assert_pixel_approx(pixel_at(&rgba_erode, 64, 32, 32), [255, 255, 255, 255], 2);
    }

    /// C++ source: entity_unittests.cc:ColorMatrixFilterContents
    #[test]
    fn test_save_layer_color_filter_matrix_and_mode() {
        let ctx = WgpuContext::new_headless().expect("headless wgpu context");
        let target = OffscreenTarget::new_with_sample_count(&ctx, 64, 64, 1);

        let red_rect = DrawCommand::SolidFill {
            mesh: make_rect_mesh(Rect::from_xywh(16.0, 16.0, 32.0, 32.0)),
            transform: Matrix::IDENTITY,
            color: Color::RED,
            blend_mode: BlendMode::SrcOver,
            clip_rect: None,
        };
        // Swap R and G channels: R' = G, G' = R, B' = B, A' = A
        let swap_rg_matrix: [Scalar; 20] = [
            0.0, 1.0, 0.0, 0.0, 0.0, // R'
            1.0, 0.0, 0.0, 0.0, 0.0, // G'
            0.0, 0.0, 1.0, 0.0, 0.0, // B'
            0.0, 0.0, 0.0, 1.0, 0.0, // A'
        ];
        let matrix_layer = DrawCommand::SaveLayer {
            bounds: Rect::from_xywh(16.0, 16.0, 32.0, 32.0),
            commands: vec![red_rect.clone()],
            alpha: 1.0,
            blend_mode: BlendMode::SrcOver,
            color_filter: Some(LayerColorFilter::Matrix(swap_rg_matrix)),
            image_filter: None,
            clip_rect: None,
        };
        let rgba_matrix = target
            .render_and_readback(&ctx, Color::BLACK, &[matrix_layer])
            .expect("savelayer color matrix render");
        assert_pixel_approx(pixel_at(&rgba_matrix, 64, 32, 32), [0, 255, 0, 255], 2);
        assert_pixel_approx(pixel_at(&rgba_matrix, 64, 8, 8), [0, 0, 0, 255], 1);

        // Mode filter: tint opaque red rect with Color::BLUE using BlendMode::SrcIn -> BLUE inside, transparent outside
        let mode_layer = DrawCommand::SaveLayer {
            bounds: Rect::from_xywh(16.0, 16.0, 32.0, 32.0),
            commands: vec![red_rect],
            alpha: 1.0,
            blend_mode: BlendMode::SrcOver,
            color_filter: Some(LayerColorFilter::Mode(Color::BLUE, BlendMode::SrcIn)),
            image_filter: None,
            clip_rect: None,
        };
        let rgba_mode = target
            .render_and_readback(&ctx, Color::BLACK, &[mode_layer])
            .expect("savelayer color mode render");
        assert_pixel_approx(pixel_at(&rgba_mode, 64, 32, 32), [0, 0, 255, 255], 2);
        assert_pixel_approx(pixel_at(&rgba_mode, 64, 8, 8), [0, 0, 0, 255], 1);
    }
}
