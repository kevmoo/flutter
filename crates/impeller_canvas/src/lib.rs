#![forbid(unsafe_code)]

pub use impeller_geometry::{
    BlendMode, Color, ColorMatrix, Matrix, Point, Radians, Rect, Scalar, Size, TileMode,
    K_EH_CLOSE_ENOUGH,
};
pub use impeller_path::{Cap, FillType, Join, Path, PathBuilder, PathComponent, RoundingRadii};
use impeller_tessellator::{
    compute_stroke_alpha_coverage, ArcGeometry, CircleGeometry, DiffRoundRectGeometry,
    EllipseGeometry, FillTessellationKind, LineGeometry, RoundRectGeometry, StrokeParameters,
    TessellatedMesh, Tessellator, Vertex2D,
};
pub use impeller_typographer::{
    AtlasGlyphQuad, BuiltinFont, FontFeature, FontVariation, GlyphAtlas, GlyphAtlasType, GlyphInfo,
    MaterialIcon, Paragraph, ParagraphBuilder, ParagraphLineMetrics, ParagraphStyle,
    PlaceholderSpan, RichTextSpan, RichTextStyle, StrutStyle, SubpixelBin, TextAlign, TextBox,
    TextDirection, TextFrame, TextShadow, TextStyle, TypographerContext,
};
pub use impeller_wgpu::{
    compare_golden_rgba, save_rgba_png, ConicalGradientParams, DrawCommand, GlyphAtlasBatch,
    GoldenDiffMetrics, GradientStop, ImageTexture, LayerColorFilter, LayerImageFilter,
    LinearGradientParams, RRectBlurParams, RadialGradientParams, SweepGradientParams,
    TextureFillBatch, UvVertex, WgpuRenderer, DEFAULT_TEXT_CONTRAST,
};
use std::path::Path as FsPath;
use std::sync::Arc;

pub use impeller_wgpu::WgpuRenderer as WgpuContext;
pub type FontId = BuiltinFont;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum DrawStyle {
    #[default]
    Fill,
    Stroke,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ClipOperation {
    #[default]
    Intersect,
    Difference,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum PointStyle {
    #[default]
    Round,
    Square,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum VertexMode {
    #[default]
    Triangles = 0,
    TriangleStrip = 1,
    TriangleFan = 2,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ColorFilter {
    Mode(Color, BlendMode),
    Matrix([f32; 20]),
    SrgbToLinear,
    LinearToSrgb,
    Compose(Box<ColorFilter>, Box<ColorFilter>),
}

impl ColorFilter {
    pub fn apply(&self, color: Color) -> Color {
        match self {
            Self::Mode(filter_color, blend_mode) => color.blend(*filter_color, *blend_mode),
            Self::Matrix(m) => color.apply_color_matrix(&ColorMatrix { array: *m }),
            Self::SrgbToLinear => color.srgb_to_linear(),
            Self::LinearToSrgb => color.linear_to_srgb(),
            Self::Compose(outer, inner) => outer.apply(inner.apply(color)),
        }
    }

    pub fn to_layer_color_filter(&self) -> LayerColorFilter {
        match self {
            Self::Mode(color, blend_mode) => LayerColorFilter::Mode(*color, *blend_mode),
            Self::Matrix(m) => LayerColorFilter::Matrix(*m),
            Self::SrgbToLinear => LayerColorFilter::SrgbToLinear,
            Self::LinearToSrgb => LayerColorFilter::LinearToSrgb,
            Self::Compose(outer, inner) => LayerColorFilter::Compose(
                Box::new(outer.to_layer_color_filter()),
                Box::new(inner.to_layer_color_filter()),
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ImageFilter {
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
    ColorFilter(ColorFilter),
    Compose(Box<ImageFilter>, Box<ImageFilter>),
}

impl ImageFilter {
    pub fn to_layer_image_filter(&self) -> LayerImageFilter {
        match self {
            Self::Blur {
                sigma_x,
                sigma_y,
                tile_mode,
            } => LayerImageFilter::Blur {
                sigma_x: *sigma_x,
                sigma_y: *sigma_y,
                tile_mode: *tile_mode,
            },
            Self::Dilate { radius_x, radius_y } => LayerImageFilter::Dilate {
                radius_x: *radius_x,
                radius_y: *radius_y,
            },
            Self::Erode { radius_x, radius_y } => LayerImageFilter::Erode {
                radius_x: *radius_x,
                radius_y: *radius_y,
            },
            Self::Matrix(m) => LayerImageFilter::Matrix(*m),
            Self::ColorFilter(cf) => LayerImageFilter::ColorFilter(cf.to_layer_color_filter()),
            Self::Compose(outer, inner) => LayerImageFilter::Compose(
                Box::new(outer.to_layer_image_filter()),
                Box::new(inner.to_layer_image_filter()),
            ),
        }
    }

    pub fn filter_bounds(&self, bounds: Rect) -> Rect {
        match self {
            Self::Blur {
                sigma_x, sigma_y, ..
            } => {
                let pad_x = (sigma_x.max(0.0) * 3.0).ceil();
                let pad_y = (sigma_y.max(0.0) * 3.0).ceil();
                Rect::from_ltrb(
                    bounds.get_left() - pad_x,
                    bounds.get_top() - pad_y,
                    bounds.get_right() + pad_x,
                    bounds.get_bottom() + pad_y,
                )
            }
            Self::Dilate { radius_x, radius_y } => {
                let rx = radius_x.max(0.0);
                let ry = radius_y.max(0.0);
                Rect::from_ltrb(
                    bounds.get_left() - rx,
                    bounds.get_top() - ry,
                    bounds.get_right() + rx,
                    bounds.get_bottom() + ry,
                )
            }
            Self::Erode { radius_x, radius_y } => {
                let rx = radius_x.max(0.0);
                let ry = radius_y.max(0.0);
                Rect::from_ltrb(
                    bounds.get_left() + rx,
                    bounds.get_top() + ry,
                    (bounds.get_right() - rx).max(bounds.get_left() + rx),
                    (bounds.get_bottom() - ry).max(bounds.get_top() + ry),
                )
            }
            Self::Matrix(m) => transform_rect(m, bounds),
            Self::ColorFilter(_) => bounds,
            Self::Compose(outer, inner) => outer.filter_bounds(inner.filter_bounds(bounds)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PaintShader {
    Solid(Color),
    LinearGradient(LinearGradientParams),
    RadialGradient(RadialGradientParams),
    SweepGradient(SweepGradientParams),
    ConicalGradient(ConicalGradientParams),
    Image {
        texture: ImageTexture,
        tile_mode_x: TileMode,
        tile_mode_y: TileMode,
        matrix: Matrix,
        linear_filter: bool,
    },
}

pub type ColorSource = PaintShader;

impl Default for PaintShader {
    fn default() -> Self {
        PaintShader::Solid(Color::BLACK)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Paint {
    pub color: Color,
    pub shader: Option<PaintShader>,
    pub color_source: Option<ColorSource>,
    pub color_filter: Option<ColorFilter>,
    pub image_filter: Option<ImageFilter>,
    pub style: DrawStyle,
    pub stroke_width: Scalar,
    pub stroke_cap: Cap,
    pub stroke_join: Join,
    pub stroke_miter: Scalar,
    pub blend_mode: BlendMode,
    pub mask_blur_sigma: Scalar,
    pub invert_colors: bool,
    pub anti_alias: bool,
}

impl Default for Paint {
    fn default() -> Self {
        Self {
            color: Color::BLACK,
            shader: None,
            color_source: None,
            color_filter: None,
            image_filter: None,
            style: DrawStyle::Fill,
            stroke_width: 0.0,
            stroke_cap: Cap::Butt,
            stroke_join: Join::Miter,
            stroke_miter: 4.0,
            blend_mode: BlendMode::SrcOver,
            mask_blur_sigma: 0.0,
            invert_colors: false,
            anti_alias: true,
        }
    }
}

impl Paint {
    pub fn fill(color: Color) -> Self {
        Self {
            color,
            style: DrawStyle::Fill,
            ..Default::default()
        }
    }

    pub fn stroke(color: Color, stroke_width: Scalar) -> Self {
        Self {
            color,
            style: DrawStyle::Stroke,
            stroke_width,
            ..Default::default()
        }
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn with_style(mut self, style: DrawStyle) -> Self {
        self.style = style;
        self
    }

    pub fn with_stroke_width(mut self, width: Scalar) -> Self {
        self.stroke_width = width;
        self
    }

    pub fn with_stroke_cap(mut self, cap: Cap) -> Self {
        self.stroke_cap = cap;
        self
    }

    pub fn with_stroke_join(mut self, join: Join) -> Self {
        self.stroke_join = join;
        self
    }

    pub fn with_stroke_miter(mut self, miter: Scalar) -> Self {
        self.stroke_miter = miter;
        self
    }

    pub fn with_blend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.blend_mode = blend_mode;
        self
    }

    pub fn with_color_source(mut self, source: ColorSource) -> Self {
        self.color_source = Some(source.clone());
        self.shader = Some(source);
        self
    }

    pub fn with_color_filter(mut self, filter: ColorFilter) -> Self {
        self.color_filter = Some(filter);
        self
    }

    pub fn with_image_filter(mut self, filter: ImageFilter) -> Self {
        self.image_filter = Some(filter);
        self
    }

    pub fn with_mask_blur_sigma(mut self, sigma: Scalar) -> Self {
        self.mask_blur_sigma = sigma;
        self
    }

    pub fn effective_color_source(&self) -> Option<&ColorSource> {
        self.color_source.as_ref().or(self.shader.as_ref())
    }

    pub fn effective_solid_color(&self) -> Color {
        let base = match self.effective_color_source() {
            Some(ColorSource::Solid(c)) => *c,
            _ => self.color,
        };
        let filtered = match &self.color_filter {
            Some(cf) => cf.apply(base),
            None => base,
        };
        if self.invert_colors {
            Color::new(
                (1.0 - filtered.red).clamp(0.0, 1.0),
                (1.0 - filtered.green).clamp(0.0, 1.0),
                (1.0 - filtered.blue).clamp(0.0, 1.0),
                filtered.alpha,
            )
        } else {
            filtered
        }
    }

    pub fn effective_stroke_width(&self) -> Scalar {
        if self.stroke_width <= 0.0 {
            1.0
        } else {
            self.stroke_width
        }
    }

    /// Returns `Some(StrokeParameters)` when `self.style == DrawStyle::Stroke`, else `None`.
    /// Matches C++ `Paint::GetStroke()` in `impeller/display_list/paint.h`.
    pub fn get_stroke(&self) -> Option<StrokeParameters> {
        if self.style == DrawStyle::Stroke {
            Some(StrokeParameters {
                width: self.stroke_width,
                cap: self.stroke_cap,
                join: self.stroke_join,
                miter_limit: self.stroke_miter,
            })
        } else {
            None
        }
    }

    /// Matches C++ `Paint::CanApplyOpacityPeephole` in `impeller/display_list/paint.h`.
    pub fn can_apply_opacity_peephole(&self) -> bool {
        self.blend_mode == BlendMode::SrcOver
            && !self.invert_colors
            && self.color_filter.is_none()
            && self.image_filter.is_none()
            && self.mask_blur_sigma <= K_EH_CLOSE_ENOUGH
    }

    pub fn is_shadow_blur_draw_operation(&self) -> bool {
        if self.style != DrawStyle::Fill || self.image_filter.is_some() {
            return false;
        }
        if matches!(
            self.effective_color_source(),
            Some(
                ColorSource::LinearGradient(_)
                    | ColorSource::RadialGradient(_)
                    | ColorSource::SweepGradient(_)
                    | ColorSource::ConicalGradient(_)
                    | ColorSource::Image { .. }
            )
        ) {
            return false;
        }
        self.mask_blur_sigma > K_EH_CLOSE_ENOUGH
    }

    /// Normalizes gradient stops to match C++ `Paint::ConvertStops` in `paint.cc`:
    /// - Ensures the first stop is at `0.0` (inserting one with the first color if missing).
    /// - Clamps all stop offsets to `[0.0, 1.0]`.
    /// - Ensures the last stop is at `1.0` (appending one with the last color if missing).
    /// - Enforces monotonically non-decreasing stop offsets.
    pub fn convert_stops(stops: &[GradientStop]) -> Vec<GradientStop> {
        if stops.is_empty() {
            return vec![
                GradientStop {
                    offset: 0.0,
                    color: Color::BLACK,
                },
                GradientStop {
                    offset: 1.0,
                    color: Color::BLACK,
                },
            ];
        }

        let mut converted = Vec::with_capacity(stops.len() + 2);
        if stops[0].offset != 0.0 {
            converted.push(GradientStop {
                offset: 0.0,
                color: stops[0].color,
            });
        }
        for s in stops {
            converted.push(GradientStop {
                offset: s.offset.clamp(0.0, 1.0),
                color: s.color,
            });
        }
        if stops[stops.len() - 1].offset != 1.0 {
            let last_color = converted.last().map(|s| s.color).unwrap_or(Color::BLACK);
            converted.push(GradientStop {
                offset: 1.0,
                color: last_color,
            });
        }
        for i in 1..converted.len() {
            let prev = converted[i - 1].offset;
            if converted[i].offset < prev {
                converted[i].offset = prev;
            }
        }
        converted
    }
}

#[derive(Clone, Debug)]
pub enum DisplayListOp {
    Save,
    SaveLayer(Option<Rect>, Option<Paint>),
    Restore,
    RestoreToCount(usize),
    Translate(Scalar, Scalar),
    Scale(Scalar, Scalar),
    Rotate(Radians),
    Skew(Scalar, Scalar),
    Transform(Matrix),
    ResetTransform,
    ClipRect(Rect, ClipOperation),
    ClipRRect(Rect, RoundingRadii, ClipOperation),
    ClipPath(Path, ClipOperation),
    DrawPaint(Paint),
    DrawColor(Color, BlendMode),
    DrawRect(Rect, Paint),
    DrawRRect(Rect, RoundingRadii, Paint),
    DrawDiffRRect(Rect, RoundingRadii, Rect, RoundingRadii, Paint),
    DrawOval(Rect, Paint),
    DrawCircle(Point, Scalar, Paint),
    DrawArc(Rect, Radians, Radians, bool, Paint),
    DrawLine(Point, Point, Paint),
    DrawDashedLine(Point, Point, Scalar, Scalar, Paint),
    DrawDashedPath(Path, Scalar, Vec<Scalar>, Paint),
    DrawPath(Path, Paint),
    DrawRRectShadow(Rect, Scalar, Scalar, Color),
    DrawShadow {
        path: Path,
        color: Color,
        elevation: Scalar,
        transparent_occluder: bool,
        device_pixel_ratio: Scalar,
    },
    DrawTextFrame(TextFrame, Point, Paint),
    DrawParagraph(Arc<Paragraph>, Point),
    DrawText(String, TextStyle, Point),
    DrawIcon(MaterialIcon, Scalar, Color, Point),
    DrawGlyphPath(char, BuiltinFont, Scalar, Point, Paint),
    DrawPolymorphicCard(usize, Scalar, Rect),
    DrawImage(ImageTexture, Point, Paint, bool),
    DrawImageRect(ImageTexture, Rect, Rect, Paint, bool),
    DrawImageNine {
        image: ImageTexture,
        center: Rect,
        dst: Rect,
        paint: Paint,
        linear_filter: bool,
    },
    DrawAtlas {
        atlas: ImageTexture,
        rst_transforms: Vec<[Scalar; 4]>,
        tex_rects: Vec<Rect>,
        colors: Option<Vec<Color>>,
        blend_mode: BlendMode,
        cull_rect: Option<Rect>,
        paint: Option<Paint>,
        linear_filter: bool,
    },
    DrawVertices {
        positions: Vec<Point>,
        tex_coords: Option<Vec<Point>>,
        colors: Option<Vec<Color>>,
        indices: Vec<u16>,
        blend_mode: BlendMode,
        paint: Paint,
    },
    DrawPicture(Arc<DisplayList>),
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CanvasReplayStats {
    pub op_count: u32,
    pub rrect_count: u32,
    pub drrect_count: u32,
    pub path_count: u32,
    pub paragraph_count: u32,
    pub picture_count: u32,
    pub save_layer_count: u32,
    pub path_fill_count: u32,
    pub path_stroke_count: u32,
    pub path_convex_fast_count: u32,
    pub path_pslg_count: u32,
    pub culled_count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct DisplayList {
    ops: Vec<DisplayListOp>,
}

impl DisplayList {
    pub fn ops(&self) -> &[DisplayListOp] {
        &self.ops
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn dispatch(&self, canvas: &mut Canvas) {
        let base_save_count = canvas.get_save_count();
        for op in &self.ops {
            canvas.replay_stats.op_count += 1;
            match op {
                DisplayListOp::Save => canvas.save(),
                DisplayListOp::SaveLayer(bounds, paint) => {
                    canvas.replay_stats.save_layer_count += 1;
                    canvas.save_layer(*bounds, paint.as_ref());
                }
                DisplayListOp::Restore => {
                    canvas.restore();
                }
                DisplayListOp::RestoreToCount(c) => {
                    let adjusted = base_save_count.saturating_sub(1) + (*c).max(1);
                    canvas.restore_to_count(adjusted);
                }
                DisplayListOp::Translate(tx, ty) => canvas.translate(*tx, *ty),
                DisplayListOp::Scale(sx, sy) => canvas.scale(*sx, *sy),
                DisplayListOp::Rotate(r) => canvas.rotate(*r),
                DisplayListOp::Skew(sx, sy) => canvas.skew(*sx, *sy),
                DisplayListOp::Transform(m) => canvas.transform(*m),
                DisplayListOp::ResetTransform => canvas.reset_transform(),
                DisplayListOp::ClipRect(rect, op) => canvas.clip_rect_with_op(*rect, *op),
                DisplayListOp::ClipRRect(rect, radii, op) => canvas.clip_rrect(*rect, *radii, *op),
                DisplayListOp::ClipPath(path, op) => canvas.clip_path(path, *op),
                DisplayListOp::DrawPaint(paint) => canvas.draw_paint(paint),
                DisplayListOp::DrawColor(color, mode) => canvas.draw_color(*color, *mode),
                DisplayListOp::DrawRect(rect, paint) => canvas.draw_rect(*rect, paint),
                DisplayListOp::DrawRRect(rect, radii, paint) => {
                    canvas.replay_stats.rrect_count += 1;
                    canvas.draw_rrect(*rect, *radii, paint);
                }
                DisplayListOp::DrawDiffRRect(or, orad, ir, irad, paint) => {
                    canvas.replay_stats.drrect_count += 1;
                    canvas.draw_diff_rrect(*or, *orad, *ir, *irad, paint);
                }
                DisplayListOp::DrawOval(rect, paint) => canvas.draw_oval(*rect, paint),
                DisplayListOp::DrawCircle(center, radius, paint) => {
                    canvas.draw_circle(*center, *radius, paint)
                }
                DisplayListOp::DrawArc(bounds, start, sweep, use_center, paint) => {
                    canvas.draw_arc(*bounds, *start, *sweep, *use_center, paint)
                }
                DisplayListOp::DrawLine(p0, p1, paint) => canvas.draw_line(*p0, *p1, paint),
                DisplayListOp::DrawDashedLine(p0, p1, on_len, off_len, paint) => {
                    canvas.draw_dashed_line(*p0, *p1, *on_len, *off_len, paint)
                }
                DisplayListOp::DrawDashedPath(path, offset, intervals, paint) => {
                    canvas.draw_dashed_path(path, *offset, intervals, paint)
                }
                DisplayListOp::DrawPath(path, paint) => {
                    canvas.replay_stats.path_count += 1;
                    canvas.draw_path(path, paint);
                }
                DisplayListOp::DrawRRectShadow(rect, radius, sigma, color) => {
                    canvas.draw_rrect_shadow(*rect, *radius, *sigma, *color)
                }
                DisplayListOp::DrawShadow {
                    path,
                    color,
                    elevation,
                    transparent_occluder,
                    device_pixel_ratio,
                } => {
                    canvas.draw_shadow(
                        path,
                        *color,
                        *elevation,
                        *transparent_occluder,
                        *device_pixel_ratio,
                    );
                }
                DisplayListOp::DrawTextFrame(frame, origin, paint) => {
                    canvas.draw_text_frame(frame, *origin, paint)
                }
                DisplayListOp::DrawParagraph(paragraph, offset) => {
                    canvas.replay_stats.paragraph_count += 1;
                    canvas.draw_paragraph(paragraph, *offset);
                }
                DisplayListOp::DrawText(text, style, origin) => {
                    canvas.draw_text(text, *style, *origin)
                }
                DisplayListOp::DrawIcon(icon, size, color, origin) => {
                    canvas.draw_icon(*icon, *size, *color, *origin)
                }
                DisplayListOp::DrawGlyphPath(ch, font, size, origin, paint) => {
                    canvas.draw_glyph_path(*ch, *font, *size, *origin, paint)
                }
                DisplayListOp::DrawPolymorphicCard(index, anim, rect) => {
                    canvas.draw_polymorphic_card(*index, *anim, *rect)
                }
                DisplayListOp::DrawImage(image, offset, paint, linear) => {
                    canvas.draw_image(image, *offset, paint, *linear)
                }
                DisplayListOp::DrawImageRect(image, src, dst, paint, linear) => {
                    canvas.draw_image_rect(image, *src, *dst, paint, *linear)
                }
                DisplayListOp::DrawImageNine {
                    image,
                    center,
                    dst,
                    paint,
                    linear_filter,
                } => canvas.draw_image_nine(image, *center, *dst, paint, *linear_filter),
                DisplayListOp::DrawAtlas {
                    atlas,
                    rst_transforms,
                    tex_rects,
                    colors,
                    blend_mode,
                    cull_rect,
                    paint,
                    linear_filter,
                } => {
                    canvas.draw_atlas(
                        atlas,
                        rst_transforms,
                        tex_rects,
                        colors.as_deref(),
                        *blend_mode,
                        *cull_rect,
                        paint.as_ref(),
                        *linear_filter,
                    );
                }
                DisplayListOp::DrawVertices {
                    positions,
                    tex_coords,
                    colors,
                    indices,
                    blend_mode,
                    paint,
                } => {
                    canvas.draw_vertices(
                        positions,
                        tex_coords.as_deref(),
                        colors.as_deref(),
                        indices,
                        *blend_mode,
                        paint,
                    );
                }
                DisplayListOp::DrawPicture(picture) => {
                    canvas.replay_stats.picture_count += 1;
                    canvas.draw_picture(picture);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DisplayListBuilder {
    ops: Vec<DisplayListOp>,
}

impl DisplayListBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn save(&mut self) -> &mut Self {
        self.ops.push(DisplayListOp::Save);
        self
    }

    pub fn save_layer(&mut self, bounds: Option<Rect>, paint: Option<Paint>) -> &mut Self {
        self.ops.push(DisplayListOp::SaveLayer(bounds, paint));
        self
    }

    pub fn restore(&mut self) -> &mut Self {
        self.ops.push(DisplayListOp::Restore);
        self
    }

    pub fn restore_to_count(&mut self, count: usize) -> &mut Self {
        self.ops.push(DisplayListOp::RestoreToCount(count));
        self
    }

    pub fn translate(&mut self, tx: Scalar, ty: Scalar) -> &mut Self {
        self.ops.push(DisplayListOp::Translate(tx, ty));
        self
    }

    pub fn scale(&mut self, sx: Scalar, sy: Scalar) -> &mut Self {
        self.ops.push(DisplayListOp::Scale(sx, sy));
        self
    }

    pub fn rotate(&mut self, radians: Radians) -> &mut Self {
        self.ops.push(DisplayListOp::Rotate(radians));
        self
    }

    pub fn skew(&mut self, sx: Scalar, sy: Scalar) -> &mut Self {
        self.ops.push(DisplayListOp::Skew(sx, sy));
        self
    }

    pub fn transform(&mut self, m: Matrix) -> &mut Self {
        self.ops.push(DisplayListOp::Transform(m));
        self
    }

    pub fn reset_transform(&mut self) -> &mut Self {
        self.ops.push(DisplayListOp::ResetTransform);
        self
    }

    pub fn clip_rect(&mut self, rect: Rect) -> &mut Self {
        self.ops
            .push(DisplayListOp::ClipRect(rect, ClipOperation::Intersect));
        self
    }

    pub fn clip_rect_with_op(&mut self, rect: Rect, op: ClipOperation) -> &mut Self {
        self.ops.push(DisplayListOp::ClipRect(rect, op));
        self
    }

    pub fn clip_rrect(&mut self, rect: Rect, radii: RoundingRadii, op: ClipOperation) -> &mut Self {
        self.ops.push(DisplayListOp::ClipRRect(rect, radii, op));
        self
    }

    pub fn clip_path(&mut self, path: Path, op: ClipOperation) -> &mut Self {
        self.ops.push(DisplayListOp::ClipPath(path, op));
        self
    }

    pub fn draw_paint(&mut self, paint: Paint) -> &mut Self {
        self.ops.push(DisplayListOp::DrawPaint(paint));
        self
    }

    pub fn draw_color(&mut self, color: Color, mode: BlendMode) -> &mut Self {
        self.ops.push(DisplayListOp::DrawColor(color, mode));
        self
    }

    pub fn draw_rect(&mut self, rect: Rect, paint: Paint) -> &mut Self {
        self.ops.push(DisplayListOp::DrawRect(rect, paint));
        self
    }

    pub fn draw_rrect(&mut self, rect: Rect, radii: RoundingRadii, paint: Paint) -> &mut Self {
        self.ops.push(DisplayListOp::DrawRRect(rect, radii, paint));
        self
    }

    pub fn draw_diff_rrect(
        &mut self,
        outer_rect: Rect,
        outer_radii: RoundingRadii,
        inner_rect: Rect,
        inner_radii: RoundingRadii,
        paint: Paint,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawDiffRRect(
            outer_rect,
            outer_radii,
            inner_rect,
            inner_radii,
            paint,
        ));
        self
    }

    pub fn draw_oval(&mut self, rect: Rect, paint: Paint) -> &mut Self {
        self.ops.push(DisplayListOp::DrawOval(rect, paint));
        self
    }

    pub fn draw_circle(&mut self, center: Point, radius: Scalar, paint: Paint) -> &mut Self {
        self.ops
            .push(DisplayListOp::DrawCircle(center, radius, paint));
        self
    }

    pub fn draw_arc(
        &mut self,
        oval_bounds: Rect,
        start: Radians,
        sweep: Radians,
        use_center: bool,
        paint: Paint,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawArc(
            oval_bounds,
            start,
            sweep,
            use_center,
            paint,
        ));
        self
    }

    pub fn draw_line(&mut self, p0: Point, p1: Point, paint: Paint) -> &mut Self {
        self.ops.push(DisplayListOp::DrawLine(p0, p1, paint));
        self
    }

    pub fn draw_dashed_line(
        &mut self,
        p0: Point,
        p1: Point,
        on_length: Scalar,
        off_length: Scalar,
        paint: Paint,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawDashedLine(
            p0, p1, on_length, off_length, paint,
        ));
        self
    }

    pub fn draw_dashed_path(
        &mut self,
        path: Path,
        dash_offset: Scalar,
        intervals: &[Scalar],
        paint: Paint,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawDashedPath(
            path,
            dash_offset,
            intervals.to_vec(),
            paint,
        ));
        self
    }

    pub fn draw_path(&mut self, path: Path, paint: Paint) -> &mut Self {
        self.ops.push(DisplayListOp::DrawPath(path, paint));
        self
    }

    pub fn draw_rrect_shadow(
        &mut self,
        rect: Rect,
        corner_radius: Scalar,
        blur_sigma: Scalar,
        color: Color,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawRRectShadow(
            rect,
            corner_radius,
            blur_sigma,
            color,
        ));
        self
    }

    pub fn draw_text_frame(
        &mut self,
        text_frame: TextFrame,
        origin: Point,
        paint: Paint,
    ) -> &mut Self {
        self.ops
            .push(DisplayListOp::DrawTextFrame(text_frame, origin, paint));
        self
    }

    pub fn draw_text(
        &mut self,
        text: impl Into<String>,
        style: TextStyle,
        origin: Point,
    ) -> &mut Self {
        self.ops
            .push(DisplayListOp::DrawText(text.into(), style, origin));
        self
    }

    pub fn draw_icon(
        &mut self,
        icon: MaterialIcon,
        size: Scalar,
        color: Color,
        origin: Point,
    ) -> &mut Self {
        self.ops
            .push(DisplayListOp::DrawIcon(icon, size, color, origin));
        self
    }

    pub fn draw_glyph_path(
        &mut self,
        ch: char,
        font: BuiltinFont,
        font_size: Scalar,
        origin: Point,
        paint: Paint,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawGlyphPath(
            ch, font, font_size, origin, paint,
        ));
        self
    }

    pub fn draw_polymorphic_card(
        &mut self,
        index: usize,
        animation_value: Scalar,
        rect: Rect,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawPolymorphicCard(
            index,
            animation_value,
            rect,
        ));
        self
    }

    pub fn draw_paragraph(
        &mut self,
        paragraph: impl Into<Arc<Paragraph>>,
        offset: Point,
    ) -> &mut Self {
        self.ops
            .push(DisplayListOp::DrawParagraph(paragraph.into(), offset));
        self
    }

    pub fn draw_image(
        &mut self,
        image: ImageTexture,
        offset: Point,
        paint: Paint,
        linear_filter: bool,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawImage(
            image,
            offset,
            paint,
            linear_filter,
        ));
        self
    }

    pub fn draw_image_rect(
        &mut self,
        image: ImageTexture,
        src: Rect,
        dst: Rect,
        paint: Paint,
        linear_filter: bool,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawImageRect(
            image,
            src,
            dst,
            paint,
            linear_filter,
        ));
        self
    }

    pub fn draw_image_nine(
        &mut self,
        image: ImageTexture,
        center: Rect,
        dst: Rect,
        paint: Paint,
        linear_filter: bool,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawImageNine {
            image,
            center,
            dst,
            paint,
            linear_filter,
        });
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_atlas(
        &mut self,
        atlas: ImageTexture,
        rst_transforms: Vec<[Scalar; 4]>,
        tex_rects: Vec<Rect>,
        colors: Option<Vec<Color>>,
        blend_mode: BlendMode,
        cull_rect: Option<Rect>,
        paint: Option<Paint>,
        linear_filter: bool,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawAtlas {
            atlas,
            rst_transforms,
            tex_rects,
            colors,
            blend_mode,
            cull_rect,
            paint,
            linear_filter,
        });
        self
    }

    pub fn draw_vertices(
        &mut self,
        positions: Vec<Point>,
        tex_coords: Option<Vec<Point>>,
        colors: Option<Vec<Color>>,
        indices: Vec<u16>,
        blend_mode: BlendMode,
        paint: Paint,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawVertices {
            positions,
            tex_coords,
            colors,
            indices,
            blend_mode,
            paint,
        });
        self
    }

    pub fn draw_shadow(
        &mut self,
        path: Path,
        color: Color,
        elevation: Scalar,
        transparent_occluder: bool,
        device_pixel_ratio: Scalar,
    ) -> &mut Self {
        self.ops.push(DisplayListOp::DrawShadow {
            path,
            color,
            elevation,
            transparent_occluder,
            device_pixel_ratio,
        });
        self
    }

    pub fn draw_picture(&mut self, picture: Arc<DisplayList>) -> &mut Self {
        self.ops.push(DisplayListOp::DrawPicture(picture));
        self
    }

    pub fn build(&self) -> DisplayList {
        DisplayList {
            ops: self.ops.clone(),
        }
    }

    pub fn into_display_list(self) -> DisplayList {
        DisplayList { ops: self.ops }
    }
}

fn make_skew_matrix(sx: Scalar, sy: Scalar) -> Matrix {
    Matrix {
        m: [
            1.0, sy, 0.0, 0.0, // col 0
            sx, 1.0, 0.0, 0.0, // col 1
            0.0, 0.0, 1.0, 0.0, // col 2
            0.0, 0.0, 0.0, 1.0, // col 3
        ],
    }
}

fn invert_matrix(mat: &Matrix) -> Option<Matrix> {
    let m = &mat.m;
    let mut inv = [0.0f32; 16];

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
        + m[9] * m[7] * m[14]
        + m[13] * m[6] * m[11]
        - m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
        - m[8] * m[7] * m[14]
        - m[12] * m[6] * m[11]
        + m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
        + m[8] * m[7] * m[13]
        + m[12] * m[5] * m[11]
        - m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
        - m[8] * m[6] * m[13]
        - m[12] * m[5] * m[10]
        + m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
        - m[9] * m[3] * m[14]
        - m[13] * m[2] * m[11]
        + m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
        + m[8] * m[3] * m[14]
        + m[12] * m[2] * m[11]
        - m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
        - m[8] * m[3] * m[13]
        - m[12] * m[1] * m[11]
        + m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
        + m[8] * m[2] * m[13]
        + m[12] * m[1] * m[10]
        - m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
        + m[5] * m[3] * m[14]
        + m[13] * m[2] * m[7]
        - m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
        - m[4] * m[3] * m[14]
        - m[12] * m[2] * m[7]
        + m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
        + m[4] * m[3] * m[13]
        + m[12] * m[1] * m[7]
        - m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
        - m[4] * m[2] * m[13]
        - m[12] * m[1] * m[6]
        + m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
        - m[5] * m[3] * m[10]
        - m[9] * m[2] * m[7]
        + m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
        + m[4] * m[3] * m[10]
        + m[8] * m[2] * m[7]
        - m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
        - m[4] * m[3] * m[9]
        - m[8] * m[1] * m[7]
        + m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
        + m[4] * m[2] * m[9]
        + m[8] * m[1] * m[6]
        - m[8] * m[2] * m[5];

    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if det.abs() <= Scalar::EPSILON {
        return None;
    }

    let inv_det = 1.0 / det;
    for v in &mut inv {
        *v *= inv_det;
    }
    Some(Matrix { m: inv })
}

fn transform_rect(tx: &Matrix, rect: Rect) -> Rect {
    if rect.is_empty() {
        return Rect::ZERO;
    }
    let corners = [
        tx.transform_point(Point::new(rect.get_left(), rect.get_top())),
        tx.transform_point(Point::new(rect.get_right(), rect.get_top())),
        tx.transform_point(Point::new(rect.get_left(), rect.get_bottom())),
        tx.transform_point(Point::new(rect.get_right(), rect.get_bottom())),
    ];
    let min_x = corners
        .iter()
        .map(|p| p.x)
        .fold(Scalar::INFINITY, Scalar::min);
    let min_y = corners
        .iter()
        .map(|p| p.y)
        .fold(Scalar::INFINITY, Scalar::min);
    let max_x = corners
        .iter()
        .map(|p| p.x)
        .fold(Scalar::NEG_INFINITY, Scalar::max);
    let max_y = corners
        .iter()
        .map(|p| p.y)
        .fold(Scalar::NEG_INFINITY, Scalar::max);
    Rect::from_ltrb(min_x, min_y, max_x, max_y)
}

fn intersect_rects(a: Rect, b: Rect) -> Option<Rect> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
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

fn subtract_rect(outer: Rect, hole: Rect) -> Vec<Rect> {
    let Some(inter) = intersect_rects(outer, hole) else {
        return if outer.is_empty() {
            Vec::new()
        } else {
            vec![outer]
        };
    };

    let ol = outer.get_left();
    let ot = outer.get_top();
    let or = outer.get_right();
    let ob = outer.get_bottom();

    let il = inter.get_left();
    let it = inter.get_top();
    let ir = inter.get_right();
    let ib = inter.get_bottom();

    let mut result = Vec::with_capacity(4);
    // Top strip
    if it > ot {
        result.push(Rect::from_ltrb(ol, ot, or, it));
    }
    // Bottom strip
    if ob > ib {
        result.push(Rect::from_ltrb(ol, ib, or, ob));
    }
    // Left middle strip
    if il > ol && ib > it {
        result.push(Rect::from_ltrb(ol, it, il, ib));
    }
    // Right middle strip
    if or > ir && ib > it {
        result.push(Rect::from_ltrb(ir, it, or, ib));
    }
    result
}

fn union_rect_list(rects: &[Rect]) -> Option<Rect> {
    let mut min_x = Scalar::INFINITY;
    let mut min_y = Scalar::INFINITY;
    let mut max_x = Scalar::NEG_INFINITY;
    let mut max_y = Scalar::NEG_INFINITY;
    let mut any = false;
    for r in rects {
        if !r.is_empty() {
            min_x = min_x.min(r.get_left());
            min_y = min_y.min(r.get_top());
            max_x = max_x.max(r.get_right());
            max_y = max_y.max(r.get_bottom());
            any = true;
        }
    }
    if any && max_x > min_x && max_y > min_y {
        Some(Rect::from_ltrb(min_x, min_y, max_x, max_y))
    } else {
        None
    }
}

#[inline]
fn rect_contains_rect(outer: Rect, inner: Rect) -> bool {
    inner.get_left() >= outer.get_left()
        && inner.get_top() >= outer.get_top()
        && inner.get_right() <= outer.get_right()
        && inner.get_bottom() <= outer.get_bottom()
}

#[inline]
fn expand_rect(rect: Rect, pad: Scalar) -> Rect {
    Rect::from_ltrb(
        rect.get_left() - pad,
        rect.get_top() - pad,
        rect.get_right() + pad,
        rect.get_bottom() + pad,
    )
}

#[inline]
fn union_two_rects(a: Rect, b: Rect) -> Rect {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    Rect::from_ltrb(
        a.get_left().min(b.get_left()),
        a.get_top().min(b.get_top()),
        a.get_right().max(b.get_right()),
        a.get_bottom().max(b.get_bottom()),
    )
}

fn mesh_device_bounds(mesh: &TessellatedMesh, transform: &Matrix) -> Option<Rect> {
    let first = mesh.vertices.first()?;
    let p0 = transform.transform_point(Point::new(first.position[0], first.position[1]));
    let mut min_x = p0.x;
    let mut min_y = p0.y;
    let mut max_x = p0.x;
    let mut max_y = p0.y;
    for v in &mesh.vertices[1..] {
        let pt = transform.transform_point(Point::new(v.position[0], v.position[1]));
        min_x = min_x.min(pt.x);
        min_y = min_y.min(pt.y);
        max_x = max_x.max(pt.x);
        max_y = max_y.max(pt.y);
    }
    let rect = Rect::from_ltrb(min_x, min_y, max_x, max_y);
    if rect.is_empty() {
        None
    } else {
        Some(rect)
    }
}

fn uv_vertices_device_bounds(vertices: &[UvVertex], transform: &Matrix) -> Option<Rect> {
    let first = vertices.first()?;
    let p0 = transform.transform_point(Point::new(first.position[0], first.position[1]));
    let mut min_x = p0.x;
    let mut min_y = p0.y;
    let mut max_x = p0.x;
    let mut max_y = p0.y;
    for v in &vertices[1..] {
        let pt = transform.transform_point(Point::new(v.position[0], v.position[1]));
        min_x = min_x.min(pt.x);
        min_y = min_y.min(pt.y);
        max_x = max_x.max(pt.x);
        max_y = max_y.max(pt.y);
    }
    let rect = Rect::from_ltrb(min_x, min_y, max_x, max_y);
    if rect.is_empty() {
        None
    } else {
        Some(rect)
    }
}

fn command_device_bounds(cmd: &DrawCommand) -> Option<Rect> {
    let (raw, clip_rect) = match cmd {
        DrawCommand::SolidFill {
            mesh,
            transform,
            clip_rect,
            ..
        }
        | DrawCommand::LinearGradientFill {
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
        } => (mesh_device_bounds(mesh, transform)?, *clip_rect),
        DrawCommand::RRectBlur {
            transform,
            params,
            clip_rect,
            ..
        } => (
            transform_rect(transform, params.coverage_rect()),
            *clip_rect,
        ),
        DrawCommand::GlyphAtlas(batch) => {
            let mut acc: Option<Rect> = None;
            for q in &batch.quads {
                let r = transform_rect(&batch.transform, q.dest_rect);
                if !r.is_empty() {
                    acc = Some(match acc {
                        Some(prev) => union_two_rects(prev, r),
                        None => r,
                    });
                }
            }
            (acc?, batch.clip_rect)
        }
        DrawCommand::TextureFill(batch) => (
            uv_vertices_device_bounds(&batch.vertices, &batch.transform)?,
            batch.clip_rect,
        ),
        DrawCommand::SaveLayer {
            bounds,
            image_filter,
            clip_rect,
            ..
        } => {
            let (hx, hy) = image_filter
                .as_ref()
                .map(|f| f.halo_padding())
                .unwrap_or((0.0, 0.0));
            let expanded = Rect::from_ltrb(
                bounds.get_left() - hx,
                bounds.get_top() - hy,
                bounds.get_right() + hx,
                bounds.get_bottom() + hy,
            );
            (expanded, *clip_rect)
        }
        DrawCommand::PushClip { .. } | DrawCommand::PopClip { .. } => return None,
    };
    if raw.is_empty() {
        return None;
    }
    match clip_rect {
        Some(cr) => intersect_rects(raw, cr),
        None => Some(raw),
    }
}

fn commands_device_bounds(commands: &[DrawCommand]) -> Option<Rect> {
    let mut acc: Option<Rect> = None;
    for cmd in commands {
        if let Some(b) = command_device_bounds(cmd) {
            acc = Some(match acc {
                Some(prev) => union_two_rects(prev, b),
                None => b,
            });
        }
    }
    acc
}

fn sync_glyph_atlases_in_slice(
    commands: &mut [DrawCommand],
    alpha_arc: &Arc<GlyphAtlas>,
    color_arc: &Arc<GlyphAtlas>,
) {
    for cmd in commands {
        match cmd {
            DrawCommand::GlyphAtlas(batch) => {
                if batch.alpha_atlas.is_none() && batch.quads.iter().any(|q| !q.is_color) {
                    batch.alpha_atlas = Some(Arc::clone(alpha_arc));
                }
                if batch.color_atlas.is_none() && batch.quads.iter().any(|q| q.is_color) {
                    batch.color_atlas = Some(Arc::clone(color_arc));
                }
            }
            DrawCommand::SaveLayer { commands: sub, .. } => {
                sync_glyph_atlases_in_slice(sub, alpha_arc, color_arc);
            }
            _ => {}
        }
    }
}

const MAX_GEOMETRY_MESH_CACHE_ENTRIES: usize = 512;
const MAX_SEEN_PATH_IDS: usize = 1024;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct CircleMeshCacheKey {
    radius_bits: u32,
    quantized_scale: u32,
    is_stroke: bool,
    stroke_width_bits: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct PathMeshCacheKey {
    path_cache_id: u64,
    fill_type: u8,
    quantized_scale: u32,
    is_stroke: bool,
    stroke_width_bits: u32,
    stroke_cap: u8,
    stroke_join: u8,
    stroke_miter_bits: u32,
}

#[derive(Default)]
struct GeometryMeshCache {
    circles: std::collections::HashMap<CircleMeshCacheKey, Arc<TessellatedMesh>>,
    paths: std::collections::HashMap<
        PathMeshCacheKey,
        (Arc<TessellatedMesh>, Option<FillTessellationKind>),
    >,
    seen_path_ids: std::collections::HashSet<u64>,
}

thread_local! {
    static GEOMETRY_MESH_CACHE: std::cell::RefCell<GeometryMeshCache> =
        std::cell::RefCell::new(GeometryMeshCache::default());
}

#[inline]
fn encode_cap(cap: Cap) -> u8 {
    match cap {
        Cap::Butt => 0,
        Cap::Round => 1,
        Cap::Square => 2,
    }
}

#[inline]
fn encode_join(join: Join) -> u8 {
    match join {
        Join::Miter => 0,
        Join::Round => 1,
        Join::Bevel => 2,
    }
}

#[inline]
fn encode_fill_type(fill_type: FillType) -> u8 {
    match fill_type {
        FillType::NonZero => 0,
        FillType::Odd => 1,
    }
}

#[inline]
fn paint_uses_local_coords_for_shader(paint: &Paint) -> bool {
    matches!(
        paint.effective_color_source(),
        Some(
            ColorSource::LinearGradient(_)
                | ColorSource::RadialGradient(_)
                | ColorSource::SweepGradient(_)
                | ColorSource::ConicalGradient(_)
                | ColorSource::Image { .. }
        )
    )
}

fn translate_cached_mesh(base: &TessellatedMesh, dx: Scalar, dy: Scalar) -> TessellatedMesh {
    if dx == 0.0 && dy == 0.0 {
        return base.clone();
    }
    let vertices = base
        .vertices
        .iter()
        .map(|v| Vertex2D::new(v.position[0] + dx, v.position[1] + dy))
        .collect();
    TessellatedMesh {
        vertices,
        indices: base.indices.clone(),
    }
}

fn cached_circle_fill_mesh_arc(radius: Scalar, scale: Scalar) -> Arc<TessellatedMesh> {
    let eff_scale = scale.max(1.0);
    let q_scale = (eff_scale * 16.0).round().max(1.0) as u32;
    let key = CircleMeshCacheKey {
        radius_bits: radius.to_bits(),
        quantized_scale: q_scale,
        is_stroke: false,
        stroke_width_bits: 0,
    };
    GEOMETRY_MESH_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(found) = cache.circles.get(&key) {
            return Arc::clone(found);
        }
        if cache.circles.len() >= MAX_GEOMETRY_MESH_CACHE_ENTRIES {
            cache.circles.clear();
        }
        let mesh = CircleGeometry::new(Point::ZERO, radius).get_mesh(q_scale as Scalar / 16.0);
        let arc = Arc::new(mesh);
        cache.circles.insert(key, Arc::clone(&arc));
        arc
    })
}

fn cached_circle_stroke_mesh_arc(
    radius: Scalar,
    stroke_w: Scalar,
    scale: Scalar,
) -> Arc<TessellatedMesh> {
    let eff_scale = scale.max(1.0);
    let q_scale = (eff_scale * 16.0).round().max(1.0) as u32;
    let key = CircleMeshCacheKey {
        radius_bits: radius.to_bits(),
        quantized_scale: q_scale,
        is_stroke: true,
        stroke_width_bits: stroke_w.to_bits(),
    };
    GEOMETRY_MESH_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(found) = cache.circles.get(&key) {
            return Arc::clone(found);
        }
        if cache.circles.len() >= MAX_GEOMETRY_MESH_CACHE_ENTRIES {
            cache.circles.clear();
        }
        let mesh = CircleGeometry::new(Point::ZERO, radius)
            .get_stroked_mesh(stroke_w, q_scale as Scalar / 16.0);
        let arc = Arc::new(mesh);
        cache.circles.insert(key, Arc::clone(&arc));
        arc
    })
}

fn cached_path_fill_mesh(path: &Path, scale: Scalar) -> (TessellatedMesh, FillTessellationKind) {
    let eff_scale = scale.max(1.0);
    let cid = path.cache_id();
    if cid == 0 {
        return Tessellator::tessellate_fill_classified(path, eff_scale);
    }
    let q_scale = (eff_scale * 16.0).round().max(1.0) as u32;
    let quantized_tol = (q_scale as Scalar / 16.0).max(1.0);
    let key = PathMeshCacheKey {
        path_cache_id: cid,
        fill_type: encode_fill_type(path.get_fill_type()),
        quantized_scale: q_scale,
        is_stroke: false,
        stroke_width_bits: 0,
        stroke_cap: 0,
        stroke_join: 0,
        stroke_miter_bits: 0,
    };
    GEOMETRY_MESH_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some((found_arc, found_kind)) = cache.paths.get(&key) {
            return (
                (**found_arc).clone(),
                found_kind.unwrap_or(FillTessellationKind::PslgGeneral),
            );
        }
        if !cache.seen_path_ids.contains(&cid) {
            if cache.seen_path_ids.len() >= MAX_SEEN_PATH_IDS {
                cache.seen_path_ids.clear();
            }
            cache.seen_path_ids.insert(cid);
            drop(cache);
            return Tessellator::tessellate_fill_classified(path, quantized_tol);
        }
        drop(cache);
        let (mesh, kind) = Tessellator::tessellate_fill_classified(path, quantized_tol);
        let arc = Arc::new(mesh);
        let mut cache = cell.borrow_mut();
        if cache.paths.len() >= MAX_GEOMETRY_MESH_CACHE_ENTRIES {
            cache.paths.clear();
        }
        cache.paths.insert(key, (Arc::clone(&arc), Some(kind)));
        ((*arc).clone(), kind)
    })
}

fn cached_path_stroke_mesh(
    path: &Path,
    stroke: StrokeParameters,
    scale: Scalar,
) -> TessellatedMesh {
    let cid = path.cache_id();
    if cid == 0 {
        return Tessellator::tessellate_stroke(path, stroke, scale);
    }
    let q_scale = (scale * 16.0).round().max(1.0) as u32;
    let quantized_tol = (q_scale as Scalar / 16.0).max(1e-4);
    let key = PathMeshCacheKey {
        path_cache_id: cid,
        fill_type: encode_fill_type(path.get_fill_type()),
        quantized_scale: q_scale,
        is_stroke: true,
        stroke_width_bits: stroke.width.to_bits(),
        stroke_cap: encode_cap(stroke.cap),
        stroke_join: encode_join(stroke.join),
        stroke_miter_bits: stroke.miter_limit.to_bits(),
    };
    GEOMETRY_MESH_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some((found_arc, _)) = cache.paths.get(&key) {
            return (**found_arc).clone();
        }
        if !cache.seen_path_ids.contains(&cid) {
            if cache.seen_path_ids.len() >= MAX_SEEN_PATH_IDS {
                cache.seen_path_ids.clear();
            }
            cache.seen_path_ids.insert(cid);
            drop(cache);
            return Tessellator::tessellate_stroke(path, stroke, quantized_tol);
        }
        drop(cache);
        let mesh = Tessellator::tessellate_stroke(path, stroke, quantized_tol);
        let arc = Arc::new(mesh);
        let mut cache = cell.borrow_mut();
        if cache.paths.len() >= MAX_GEOMETRY_MESH_CACHE_ENTRIES {
            cache.paths.clear();
        }
        cache.paths.insert(key, (Arc::clone(&arc), None));
        (*arc).clone()
    })
}

/// Converts an `impeller_path::Path` into a dashed `impeller_path::Path` using `kurbo::dash`.
pub fn dash_path(path: &Path, dash_offset: Scalar, intervals: &[Scalar]) -> Path {
    let dashes: Vec<f64> = intervals
        .iter()
        .copied()
        .filter(|&d| d.is_finite() && d > K_EH_CLOSE_ENOUGH)
        .map(|d| d as f64)
        .collect();
    if dashes.is_empty() || path.is_empty() {
        return path.clone();
    }

    let mut bez = kurbo::BezPath::new();
    let mut in_contour = false;
    let mut contour_closed = false;
    let mut pending_contour: Option<impeller_path::ContourComponent> = None;

    for comp in &path.components {
        match *comp {
            PathComponent::Contour(c) => {
                if in_contour && contour_closed {
                    bez.close_path();
                }
                in_contour = false;
                contour_closed = c.is_closed;
                pending_contour = Some(c);
            }
            PathComponent::Linear(l) => {
                if !in_contour {
                    let start = pending_contour
                        .take()
                        .map(|c| c.destination)
                        .unwrap_or(l.p1);
                    bez.move_to(kurbo::Point::new(start.x as f64, start.y as f64));
                    in_contour = true;
                }
                bez.line_to(kurbo::Point::new(l.p2.x as f64, l.p2.y as f64));
            }
            PathComponent::Quad(q) => {
                if !in_contour {
                    let start = pending_contour
                        .take()
                        .map(|c| c.destination)
                        .unwrap_or(q.p1);
                    bez.move_to(kurbo::Point::new(start.x as f64, start.y as f64));
                    in_contour = true;
                }
                bez.quad_to(
                    kurbo::Point::new(q.cp.x as f64, q.cp.y as f64),
                    kurbo::Point::new(q.p2.x as f64, q.p2.y as f64),
                );
            }
            PathComponent::Cubic(c) => {
                if !in_contour {
                    let start = pending_contour
                        .take()
                        .map(|cc| cc.destination)
                        .unwrap_or(c.p1);
                    bez.move_to(kurbo::Point::new(start.x as f64, start.y as f64));
                    in_contour = true;
                }
                bez.curve_to(
                    kurbo::Point::new(c.cp1.x as f64, c.cp1.y as f64),
                    kurbo::Point::new(c.cp2.x as f64, c.cp2.y as f64),
                    kurbo::Point::new(c.p2.x as f64, c.p2.y as f64),
                );
            }
        }
    }
    if in_contour && contour_closed {
        bez.close_path();
    }

    let mut pb = PathBuilder::new();
    for el in kurbo::dash(bez.into_iter(), dash_offset as f64, &dashes) {
        match el {
            kurbo::PathEl::MoveTo(p) => {
                pb.move_to(Point::new(p.x as Scalar, p.y as Scalar));
            }
            kurbo::PathEl::LineTo(p) => {
                pb.line_to(Point::new(p.x as Scalar, p.y as Scalar));
            }
            kurbo::PathEl::QuadTo(p1, p2) => {
                pb.quad_to(
                    Point::new(p1.x as Scalar, p1.y as Scalar),
                    Point::new(p2.x as Scalar, p2.y as Scalar),
                );
            }
            kurbo::PathEl::CurveTo(p1, p2, p3) => {
                pb.cubic_to(
                    Point::new(p1.x as Scalar, p1.y as Scalar),
                    Point::new(p2.x as Scalar, p2.y as Scalar),
                    Point::new(p3.x as Scalar, p3.y as Scalar),
                );
            }
            kurbo::PathEl::ClosePath => {
                pb.close();
            }
        }
    }
    pb.build(path.get_fill_type())
}

#[derive(Clone, Debug)]
struct CanvasState {
    transform: Matrix,
    clip_rects: Vec<Rect>,
    has_explicit_clip: bool,
    layer_start_cmd_index: Option<usize>,
    layer_paint: Option<Paint>,
    layer_bounds_device: Option<Rect>,
    layer_clip_rect: Option<Rect>,
    stencil_clips: Vec<Option<Rect>>,
}

#[derive(Clone, Debug)]
pub struct PendingGlyphBatch {
    pub command_index: usize,
    pub transform: Matrix,
    pub quads: Vec<AtlasGlyphQuad>,
    pub blend_mode: BlendMode,
    pub clip_rect: Option<Rect>,
}

pub struct Canvas {
    width: u32,
    height: u32,
    clear_color: Color,
    initial_cull_rect: Rect,
    stack: Vec<CanvasState>,
    commands: Vec<DrawCommand>,
    typographer: Option<TypographerContext>,
    cached_alpha_atlas: Option<(u64, Arc<GlyphAtlas>)>,
    cached_color_atlas: Option<(u64, Arc<GlyphAtlas>)>,
    default_alpha_atlas: Option<Arc<GlyphAtlas>>,
    default_color_atlas: Option<Arc<GlyphAtlas>>,
    replay_stats: CanvasReplayStats,
}

impl Canvas {
    pub fn new(width: u32, height: u32, clear_color: Color) -> Self {
        let cull_rect = Rect::from_xywh(0.0, 0.0, width as Scalar, height as Scalar);
        Self::with_cull_rect(width, height, clear_color, cull_rect)
    }

    pub fn with_cull_rect(width: u32, height: u32, clear_color: Color, cull_rect: Rect) -> Self {
        let canvas_bounds = Rect::from_xywh(0.0, 0.0, width as Scalar, height as Scalar);
        let initial_clip = intersect_rects(canvas_bounds, cull_rect)
            .into_iter()
            .collect::<Vec<_>>();
        Self {
            width,
            height,
            clear_color,
            initial_cull_rect: cull_rect,
            stack: vec![CanvasState {
                transform: Matrix::IDENTITY,
                clip_rects: initial_clip,
                has_explicit_clip: false,
                layer_start_cmd_index: None,
                layer_paint: None,
                layer_bounds_device: None,
                layer_clip_rect: None,
                stencil_clips: Vec::new(),
            }],
            commands: Vec::new(),
            typographer: None,
            cached_alpha_atlas: None,
            cached_color_atlas: None,
            default_alpha_atlas: None,
            default_color_atlas: None,
            replay_stats: CanvasReplayStats::default(),
        }
    }

    pub const fn replay_stats(&self) -> CanvasReplayStats {
        self.replay_stats
    }

    pub fn set_default_glyph_atlases(
        &mut self,
        alpha: Option<Arc<GlyphAtlas>>,
        color: Option<Arc<GlyphAtlas>>,
    ) {
        self.default_alpha_atlas = alpha;
        self.default_color_atlas = color;
    }

    /// Resets the canvas for a new frame while retaining internal `commands` and `stack` capacities.
    pub fn reset(&mut self, width: u32, height: u32, clear_color: Color) {
        self.width = width;
        self.height = height;
        self.clear_color = clear_color;
        self.replay_stats = CanvasReplayStats::default();
        let canvas_bounds = Rect::from_xywh(0.0, 0.0, width as Scalar, height as Scalar);
        self.initial_cull_rect = canvas_bounds;
        self.commands.clear();
        self.stack.truncate(1);
        if let Some(root) = self.stack.first_mut() {
            root.transform = Matrix::IDENTITY;
            root.clip_rects.clear();
            if let Some(clip) = intersect_rects(canvas_bounds, canvas_bounds) {
                root.clip_rects.push(clip);
            }
            root.has_explicit_clip = false;
            root.layer_start_cmd_index = None;
            root.layer_paint = None;
            root.layer_bounds_device = None;
            root.layer_clip_rect = None;
            root.stencil_clips.clear();
        } else {
            let initial_clip = intersect_rects(canvas_bounds, canvas_bounds)
                .into_iter()
                .collect::<Vec<_>>();
            self.stack.push(CanvasState {
                transform: Matrix::IDENTITY,
                clip_rects: initial_clip,
                has_explicit_clip: false,
                layer_start_cmd_index: None,
                layer_paint: None,
                layer_bounds_device: None,
                layer_clip_rect: None,
                stencil_clips: Vec::new(),
            });
        }
    }

    pub fn with_typographer(mut self, typographer: TypographerContext) -> Self {
        self.typographer = Some(typographer);
        self.sync_glyph_atlases();
        self
    }

    pub fn set_typographer(&mut self, typographer: TypographerContext) {
        self.typographer = Some(typographer);
        self.sync_glyph_atlases();
    }

    pub fn take_typographer(&mut self) -> Option<TypographerContext> {
        self.typographer.take()
    }

    pub fn typographer(&self) -> Option<&TypographerContext> {
        self.typographer.as_ref()
    }

    pub fn typographer_mut(&mut self) -> &mut TypographerContext {
        self.typographer.get_or_insert_with(TypographerContext::new)
    }

    pub fn shape_text(&mut self, text: &str, style: TextStyle, origin: Point) -> TextFrame {
        self.typographer_mut().shape_text(text, style, origin)
    }

    pub fn shape_icon(
        &mut self,
        icon: MaterialIcon,
        size: Scalar,
        color: Color,
        origin: Point,
    ) -> TextFrame {
        let mut buf = [0u8; 4];
        let s = icon.codepoint().encode_utf8(&mut buf);
        let style = TextStyle {
            font: BuiltinFont::MaterialIcons,
            font_size: size,
            line_height: size,
            color,
            max_width: None,
        };
        self.typographer_mut().shape_text(s, style, origin)
    }

    pub fn sync_glyph_atlases_with_arcs(
        &mut self,
        alpha_arc: &Arc<GlyphAtlas>,
        color_arc: &Arc<GlyphAtlas>,
    ) {
        sync_glyph_atlases_in_slice(&mut self.commands, alpha_arc, color_arc);
    }

    pub fn sync_glyph_atlases_with(&mut self, typographer: &TypographerContext) {
        let alpha_gen = typographer.alpha_atlas.generation;
        let alpha_arc = match &self.cached_alpha_atlas {
            Some((g, arc)) if *g == alpha_gen => Arc::clone(arc),
            _ => {
                let arc = Arc::new(typographer.alpha_atlas.clone());
                self.cached_alpha_atlas = Some((alpha_gen, Arc::clone(&arc)));
                arc
            }
        };

        let color_gen = typographer.color_atlas.generation;
        let color_arc = match &self.cached_color_atlas {
            Some((g, arc)) if *g == color_gen => Arc::clone(arc),
            _ => {
                let arc = Arc::new(typographer.color_atlas.clone());
                self.cached_color_atlas = Some((color_gen, Arc::clone(&arc)));
                arc
            }
        };

        self.sync_glyph_atlases_with_arcs(&alpha_arc, &color_arc);
    }

    fn resolve_clip_targets(
        &self,
        local_bounds: Option<Rect>,
        transform: &Matrix,
    ) -> Vec<Option<Rect>> {
        self.resolve_clip_targets_filtered(local_bounds, transform, None)
    }

    fn resolve_clip_targets_filtered(
        &self,
        local_bounds: Option<Rect>,
        transform: &Matrix,
        image_filter: Option<&ImageFilter>,
    ) -> Vec<Option<Rect>> {
        let Some(top) = self.stack.last() else {
            return Vec::new();
        };
        if top.clip_rects.is_empty() {
            return Vec::new();
        }
        let canvas_rect = Rect::from_xywh(0.0, 0.0, self.width as Scalar, self.height as Scalar);
        let Some(canvas_bounds) = intersect_rects(canvas_rect, self.initial_cull_rect) else {
            return Vec::new();
        };

        let Some(lb) = local_bounds else {
            return if top.has_explicit_clip {
                top.clip_rects.iter().copied().map(Some).collect()
            } else {
                vec![None]
            };
        };

        if lb.is_empty() {
            return Vec::new();
        }
        let raw_dev_bounds = transform_rect(transform, lb);
        let dev_bounds = match image_filter {
            Some(imf) => imf.filter_bounds(raw_dev_bounds),
            None => raw_dev_bounds,
        };
        if dev_bounds.is_empty() || intersect_rects(dev_bounds, canvas_bounds).is_none() {
            return Vec::new();
        }
        if !top.has_explicit_clip {
            return vec![None];
        }

        let mut targets = Vec::with_capacity(top.clip_rects.len());
        let mut emitted_unclipped = false;
        for clip in &top.clip_rects {
            if intersect_rects(dev_bounds, *clip).is_none() {
                continue;
            }
            if rect_contains_rect(*clip, dev_bounds) {
                if !emitted_unclipped {
                    targets.push(None);
                    emitted_unclipped = true;
                }
            } else {
                targets.push(Some(*clip));
            }
        }
        targets
    }

    pub fn emit_glyph_quads_slice_with_offset(
        &mut self,
        quads: &[AtlasGlyphQuad],
        offset: Point,
        transform: Matrix,
        blend_mode: BlendMode,
        use_linear_filter: bool,
    ) {
        if quads.is_empty() {
            return;
        }
        let Some(top) = self.stack.last() else {
            return;
        };
        if top.clip_rects.is_empty() {
            return;
        }
        let canvas_rect = Rect::from_xywh(0.0, 0.0, self.width as Scalar, self.height as Scalar);
        let Some(canvas_bounds) = intersect_rects(canvas_rect, self.initial_cull_rect) else {
            return;
        };

        let shift_quad = |q: &AtlasGlyphQuad| -> AtlasGlyphQuad {
            if offset.x == 0.0 && offset.y == 0.0 {
                *q
            } else {
                AtlasGlyphQuad {
                    dest_rect: Rect::from_xywh(
                        q.dest_rect.get_left() + offset.x,
                        q.dest_rect.get_top() + offset.y,
                        q.dest_rect.get_width(),
                        q.dest_rect.get_height(),
                    ),
                    atlas_rect: q.atlas_rect,
                    color: q.color,
                    is_color: q.is_color,
                }
            }
        };

        if !top.has_explicit_clip {
            let mut surviving = Vec::with_capacity(quads.len());
            let mut uses_alpha = false;
            let mut uses_color = false;
            for q in quads {
                let shifted = shift_quad(q);
                let dev_r = transform_rect(&transform, shifted.dest_rect);
                if intersect_rects(dev_r, canvas_bounds).is_some() {
                    if shifted.is_color {
                        uses_color = true;
                    } else {
                        uses_alpha = true;
                    }
                    surviving.push(shifted);
                }
            }
            if surviving.is_empty() {
                return;
            }
            let alpha_atlas = if uses_alpha {
                self.default_alpha_atlas.clone()
            } else {
                None
            };
            let color_atlas = if uses_color {
                self.default_color_atlas.clone()
            } else {
                None
            };
            self.commands.push(DrawCommand::GlyphAtlas(
                GlyphAtlasBatch::new(
                    surviving,
                    transform,
                    alpha_atlas,
                    color_atlas,
                    blend_mode,
                    None,
                )
                .with_linear_filter(use_linear_filter),
            ));
            return;
        }

        if top.clip_rects.len() == 1 {
            let Some(effective_clip) = intersect_rects(top.clip_rects[0], canvas_bounds) else {
                return;
            };
            let mut surviving = Vec::with_capacity(quads.len());
            let mut all_contained = true;
            let mut uses_alpha = false;
            let mut uses_color = false;
            for q in quads {
                let shifted = shift_quad(q);
                let dev_r = transform_rect(&transform, shifted.dest_rect);
                if intersect_rects(dev_r, effective_clip).is_some() {
                    if !rect_contains_rect(effective_clip, dev_r) {
                        all_contained = false;
                    }
                    if shifted.is_color {
                        uses_color = true;
                    } else {
                        uses_alpha = true;
                    }
                    surviving.push(shifted);
                }
            }
            if surviving.is_empty() {
                return;
            }
            let effective_scissor = if all_contained {
                None
            } else {
                Some(effective_clip)
            };
            let alpha_atlas = if uses_alpha {
                self.default_alpha_atlas.clone()
            } else {
                None
            };
            let color_atlas = if uses_color {
                self.default_color_atlas.clone()
            } else {
                None
            };
            self.commands.push(DrawCommand::GlyphAtlas(
                GlyphAtlasBatch::new(
                    surviving,
                    transform,
                    alpha_atlas,
                    color_atlas,
                    blend_mode,
                    effective_scissor,
                )
                .with_linear_filter(use_linear_filter),
            ));
            return;
        }

        for clip in &top.clip_rects {
            let Some(effective_clip) = intersect_rects(*clip, canvas_bounds) else {
                continue;
            };
            let mut surviving = Vec::new();
            let mut all_contained = true;
            let mut uses_alpha = false;
            let mut uses_color = false;
            for q in quads {
                let shifted = shift_quad(q);
                let dev_r = transform_rect(&transform, shifted.dest_rect);
                if intersect_rects(dev_r, effective_clip).is_some() {
                    if !rect_contains_rect(effective_clip, dev_r) {
                        all_contained = false;
                    }
                    if shifted.is_color {
                        uses_color = true;
                    } else {
                        uses_alpha = true;
                    }
                    surviving.push(shifted);
                }
            }
            if surviving.is_empty() {
                continue;
            }
            let effective_scissor = if all_contained {
                None
            } else {
                Some(effective_clip)
            };
            let alpha_atlas = if uses_alpha {
                self.default_alpha_atlas.clone()
            } else {
                None
            };
            let color_atlas = if uses_color {
                self.default_color_atlas.clone()
            } else {
                None
            };
            self.commands.push(DrawCommand::GlyphAtlas(
                GlyphAtlasBatch::new(
                    surviving,
                    transform,
                    alpha_atlas,
                    color_atlas,
                    blend_mode,
                    effective_scissor,
                )
                .with_linear_filter(use_linear_filter),
            ));
        }
    }

    fn emit_glyph_quads_with_blend(
        &mut self,
        quads: Vec<AtlasGlyphQuad>,
        transform: Matrix,
        blend_mode: BlendMode,
        use_linear_filter: bool,
    ) {
        self.emit_glyph_quads_slice_with_offset(
            &quads,
            Point::new(0.0, 0.0),
            transform,
            blend_mode,
            use_linear_filter,
        );
    }

    pub fn emit_glyph_quads_batch(
        &mut self,
        quads: Vec<AtlasGlyphQuad>,
        transform: Matrix,
        use_linear_filter: bool,
    ) {
        self.emit_glyph_quads_with_blend(quads, transform, BlendMode::SrcOver, use_linear_filter);
    }

    fn sync_glyph_atlases(&mut self) {
        let Some(typographer) = self.typographer.take() else {
            return;
        };
        self.sync_glyph_atlases_with(&typographer);
        self.typographer = Some(typographer);
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn clear_color(&self) -> Color {
        self.clear_color
    }

    pub fn get_save_count(&self) -> usize {
        self.stack.len()
    }

    pub fn get_current_transform(&self) -> Matrix {
        self.stack
            .last()
            .map(|s| s.transform)
            .unwrap_or(Matrix::IDENTITY)
    }

    pub fn get_current_clip(&self) -> Option<Rect> {
        let top = self.stack.last()?;
        if !top.has_explicit_clip {
            None
        } else {
            union_rect_list(&top.clip_rects)
        }
    }

    /// Returns the active device-space clip coverage bounded by the canvas target size,
    /// matching C++ `Canvas::GetLocalCoverageLimit()`.
    pub fn get_local_coverage_limit(&self) -> Option<Rect> {
        let top = self.stack.last()?;
        let coverage = union_rect_list(&top.clip_rects)?;
        let canvas_bounds = Rect::from_xywh(0.0, 0.0, self.width as Scalar, self.height as Scalar);
        let bounded = intersect_rects(coverage, canvas_bounds)?;
        intersect_rects(bounded, self.initial_cull_rect)
    }

    /// Returns the current cull/clip bounding box transformed into local canvas coordinates.
    pub fn get_current_local_cull_rect(&self) -> Option<Rect> {
        let device_coverage = self.get_local_coverage_limit()?;
        let inv = invert_matrix(&self.get_current_transform())?;
        let local = transform_rect(&inv, device_coverage);
        if local.is_empty() {
            None
        } else {
            Some(local)
        }
    }

    pub fn save(&mut self) {
        if let Some(top) = self.stack.last() {
            self.stack.push(CanvasState {
                transform: top.transform,
                clip_rects: top.clip_rects.clone(),
                has_explicit_clip: top.has_explicit_clip,
                layer_start_cmd_index: None,
                layer_paint: None,
                layer_bounds_device: None,
                layer_clip_rect: None,
                stencil_clips: Vec::new(),
            });
        }
    }

    pub fn save_layer(&mut self, bounds: Option<Rect>, paint: Option<&Paint>) {
        let start_idx = self.commands.len();
        let tx = self.get_current_transform();
        let layer_clip_rect = self.current_scissor_rect();
        let layer_bounds_device = bounds.and_then(|b| {
            if b.is_empty() {
                None
            } else {
                let dev_b = transform_rect(&tx, b);
                if dev_b.is_empty()
                    || !dev_b.get_left().is_finite()
                    || !dev_b.get_top().is_finite()
                    || !dev_b.get_right().is_finite()
                    || !dev_b.get_bottom().is_finite()
                {
                    None
                } else {
                    Some(dev_b)
                }
            }
        });
        if let Some(top) = self.stack.last() {
            self.stack.push(CanvasState {
                transform: top.transform,
                clip_rects: top.clip_rects.clone(),
                has_explicit_clip: top.has_explicit_clip,
                layer_start_cmd_index: Some(start_idx),
                layer_paint: paint.cloned(),
                layer_bounds_device,
                layer_clip_rect,
                stencil_clips: Vec::new(),
            });
        }
        if let Some(dev_b) = layer_bounds_device {
            let (halo_x, halo_y) = paint
                .and_then(|p| p.image_filter.as_ref())
                .map(|f| f.to_layer_image_filter().halo_padding())
                .unwrap_or((0.0, 0.0));
            let child_clip = if halo_x > 0.0 || halo_y > 0.0 {
                Rect::from_ltrb(
                    dev_b.get_left() - halo_x,
                    dev_b.get_top() - halo_y,
                    dev_b.get_right() + halo_x,
                    dev_b.get_bottom() + halo_y,
                )
            } else {
                dev_b
            };
            self.apply_device_clip_rect(child_clip, ClipOperation::Intersect);
        }
    }

    pub fn restore(&mut self) -> bool {
        if self.stack.len() > 1 {
            if let Some(popped) = self.stack.pop() {
                for sc in popped.stencil_clips.into_iter().rev() {
                    self.commands.push(DrawCommand::PopClip { clip_rect: sc });
                }

                if let Some(start_idx) = popped.layer_start_cmd_index {
                    let layer_paint = popped
                        .layer_paint
                        .unwrap_or_else(|| Paint::fill(Color::WHITE));
                    let layer_alpha = layer_paint.color.alpha.clamp(0.0, 1.0);
                    let layer_blend = layer_paint.blend_mode;
                    let is_identity_layer = layer_paint.image_filter.is_none()
                        && layer_paint.color_filter.is_none()
                        && layer_blend == BlendMode::SrcOver
                        && layer_alpha >= 1.0 - 1e-4;

                    if !is_identity_layer {
                        let mut visual_draw_count = 0usize;
                        let mut has_child_save_layer = false;
                        let mut has_texture_fill = false;
                        for cmd in &self.commands[start_idx..] {
                            match cmd {
                                DrawCommand::PushClip { .. } | DrawCommand::PopClip { .. } => {}
                                DrawCommand::SaveLayer { .. } => {
                                    visual_draw_count += 1;
                                    has_child_save_layer = true;
                                }
                                DrawCommand::TextureFill(_) => {
                                    visual_draw_count += 1;
                                    has_texture_fill = true;
                                }
                                _ => {
                                    visual_draw_count += 1;
                                }
                            }
                        }

                        let can_collapse = layer_paint.image_filter.is_none()
                            && !has_child_save_layer
                            && (visual_draw_count == 0
                                || (visual_draw_count == 1
                                    && (layer_paint.color_filter.is_none() || !has_texture_fill)));

                        if can_collapse {
                            let layer_cf = layer_paint.color_filter.as_ref();
                            let override_blend =
                                (layer_blend != BlendMode::SrcOver).then_some(layer_blend);

                            for cmd in self.commands.iter_mut().skip(start_idx) {
                                match cmd {
                                    DrawCommand::SolidFill {
                                        color, blend_mode, ..
                                    } => {
                                        if let Some(cf) = layer_cf {
                                            *color = cf.apply(*color);
                                        }
                                        color.alpha = (color.alpha * layer_alpha).clamp(0.0, 1.0);
                                        if let Some(bm) = override_blend {
                                            *blend_mode = bm;
                                        }
                                    }
                                    DrawCommand::LinearGradientFill {
                                        gradient,
                                        blend_mode,
                                        ..
                                    } => {
                                        if let Some(cf) = layer_cf {
                                            for stop in &mut gradient.stops {
                                                stop.color = cf.apply(stop.color);
                                            }
                                        }
                                        gradient.alpha =
                                            (gradient.alpha * layer_alpha).clamp(0.0, 1.0);
                                        if let Some(bm) = override_blend {
                                            *blend_mode = bm;
                                        }
                                    }
                                    DrawCommand::RadialGradientFill {
                                        gradient,
                                        blend_mode,
                                        ..
                                    } => {
                                        if let Some(cf) = layer_cf {
                                            for stop in &mut gradient.stops {
                                                stop.color = cf.apply(stop.color);
                                            }
                                        }
                                        gradient.alpha =
                                            (gradient.alpha * layer_alpha).clamp(0.0, 1.0);
                                        if let Some(bm) = override_blend {
                                            *blend_mode = bm;
                                        }
                                    }
                                    DrawCommand::SweepGradientFill {
                                        gradient,
                                        blend_mode,
                                        ..
                                    } => {
                                        if let Some(cf) = layer_cf {
                                            for stop in &mut gradient.stops {
                                                stop.color = cf.apply(stop.color);
                                            }
                                        }
                                        gradient.alpha =
                                            (gradient.alpha * layer_alpha).clamp(0.0, 1.0);
                                        if let Some(bm) = override_blend {
                                            *blend_mode = bm;
                                        }
                                    }
                                    DrawCommand::ConicalGradientFill {
                                        gradient,
                                        blend_mode,
                                        ..
                                    } => {
                                        if let Some(cf) = layer_cf {
                                            for stop in &mut gradient.stops {
                                                stop.color = cf.apply(stop.color);
                                            }
                                        }
                                        gradient.alpha =
                                            (gradient.alpha * layer_alpha).clamp(0.0, 1.0);
                                        if let Some(bm) = override_blend {
                                            *blend_mode = bm;
                                        }
                                    }
                                    DrawCommand::RRectBlur {
                                        params, blend_mode, ..
                                    } => {
                                        if let Some(cf) = layer_cf {
                                            params.color = cf.apply(params.color);
                                        }
                                        params.color.alpha =
                                            (params.color.alpha * layer_alpha).clamp(0.0, 1.0);
                                        if let Some(bm) = override_blend {
                                            *blend_mode = bm;
                                        }
                                    }
                                    DrawCommand::GlyphAtlas(batch) => {
                                        if let Some(cf) = layer_cf {
                                            for q in &mut batch.quads {
                                                q.color = cf.apply(q.color);
                                            }
                                        }
                                        batch.alpha = (batch.alpha * layer_alpha).clamp(0.0, 1.0);
                                        if let Some(bm) = override_blend {
                                            batch.blend_mode = bm;
                                        }
                                    }
                                    DrawCommand::TextureFill(batch) => {
                                        batch.alpha = (batch.alpha * layer_alpha).clamp(0.0, 1.0);
                                        if let Some(bm) = override_blend {
                                            batch.blend_mode = bm;
                                        }
                                    }
                                    DrawCommand::PushClip { .. }
                                    | DrawCommand::PopClip { .. }
                                    | DrawCommand::SaveLayer { .. } => {}
                                }
                            }
                        } else {
                            let sub_commands: Vec<DrawCommand> =
                                self.commands.drain(start_idx..).collect();
                            let bounds = popped
                                .layer_bounds_device
                                .filter(|b| !b.is_empty())
                                .or_else(|| commands_device_bounds(&sub_commands))
                                .or(popped.layer_clip_rect)
                                .unwrap_or_else(|| {
                                    Rect::from_xywh(
                                        0.0,
                                        0.0,
                                        self.width as Scalar,
                                        self.height as Scalar,
                                    )
                                });
                            self.commands.push(DrawCommand::SaveLayer {
                                bounds,
                                commands: sub_commands,
                                alpha: layer_alpha,
                                blend_mode: layer_blend,
                                color_filter: layer_paint
                                    .color_filter
                                    .as_ref()
                                    .map(ColorFilter::to_layer_color_filter),
                                image_filter: layer_paint
                                    .image_filter
                                    .as_ref()
                                    .map(ImageFilter::to_layer_image_filter),
                                clip_rect: popped.layer_clip_rect,
                            });
                        }
                    }
                }
            }
            true
        } else {
            false
        }
    }

    pub fn restore_to_count(&mut self, count: usize) {
        let target = count.max(1);
        while self.stack.len() > target {
            if !self.restore() {
                break;
            }
        }
    }

    pub fn reset_transform(&mut self) {
        if let Some(top) = self.stack.last_mut() {
            top.transform = Matrix::IDENTITY;
        }
    }

    pub fn transform(&mut self, m: Matrix) {
        if let Some(top) = self.stack.last_mut() {
            top.transform = top.transform * m;
        }
    }

    pub fn concat(&mut self, m: Matrix) {
        self.transform(m);
    }

    pub fn pre_concat(&mut self, m: Matrix) {
        if let Some(top) = self.stack.last_mut() {
            top.transform = m * top.transform;
        }
    }

    pub fn translate(&mut self, tx: Scalar, ty: Scalar) {
        self.transform(Matrix::make_translation(tx, ty, 0.0));
    }

    pub fn scale(&mut self, sx: Scalar, sy: Scalar) {
        self.transform(Matrix::make_scale(sx, sy, 1.0));
    }

    pub fn scale_3d(&mut self, sx: Scalar, sy: Scalar, sz: Scalar) {
        self.transform(Matrix::make_scale(sx, sy, sz));
    }

    pub fn rotate(&mut self, radians: Radians) {
        self.transform(Matrix::make_rotation_z(radians));
    }

    pub fn skew(&mut self, sx: Scalar, sy: Scalar) {
        self.transform(make_skew_matrix(sx, sy));
    }

    fn current_scissor_rect(&self) -> Option<Rect> {
        let top = self.stack.last()?;
        if !top.has_explicit_clip {
            None
        } else {
            self.get_local_coverage_limit()
        }
    }

    pub fn clip_rect(&mut self, rect: Rect) {
        self.clip_rect_with_op(rect, ClipOperation::Intersect);
    }

    pub fn clip_rect_with_op(&mut self, rect: Rect, op: ClipOperation) {
        let tx = self.get_current_transform();
        let transformed = transform_rect(&tx, rect);
        self.apply_device_clip_rect(transformed, op);
    }

    pub fn clip_rrect(&mut self, rect: Rect, radii: RoundingRadii, op: ClipOperation) {
        if radii.are_all_corners_zero() || rect.is_empty() {
            self.clip_rect_with_op(rect, op);
            return;
        }
        if self.stack.last().is_none_or(|s| s.clip_rects.is_empty()) {
            return;
        }
        let pre_clip_scissor = self.current_scissor_rect();
        let tx = self.get_current_transform();
        let scale = tx.get_max_basis_length_xy().max(1.0);
        if op == ClipOperation::Intersect {
            let transformed = transform_rect(&tx, rect);
            self.apply_device_clip_rect(transformed, ClipOperation::Intersect);
            if self.stack.last().is_none_or(|s| s.clip_rects.is_empty()) {
                return;
            }
        }
        let mesh = RoundRectGeometry::new(rect, radii).get_mesh(scale);
        if !mesh.vertices.is_empty() && !mesh.indices.is_empty() {
            self.commands.push(DrawCommand::PushClip {
                mesh,
                transform: tx,
                is_difference: op == ClipOperation::Difference,
                clip_rect: pre_clip_scissor,
            });
            if let Some(top) = self.stack.last_mut() {
                top.stencil_clips.push(pre_clip_scissor);
            }
        }
    }

    pub fn clip_path(&mut self, path: &Path, op: ClipOperation) {
        let Some(raw_bbox) = path.get_bounding_box() else {
            if op == ClipOperation::Intersect {
                self.apply_device_clip_rect(Rect::ZERO, ClipOperation::Intersect);
            }
            return;
        };
        if raw_bbox.is_empty() {
            if op == ClipOperation::Intersect {
                self.apply_device_clip_rect(Rect::ZERO, ClipOperation::Intersect);
            }
            return;
        }
        if self.stack.last().is_none_or(|s| s.clip_rects.is_empty()) {
            return;
        }
        let pre_clip_scissor = self.current_scissor_rect();
        let tx = self.get_current_transform();
        let scale = tx.get_max_basis_length_xy().max(1.0);
        if op == ClipOperation::Intersect {
            let transformed = path.get_transformed_bounding_box(&tx).unwrap_or(Rect::ZERO);
            self.apply_device_clip_rect(transformed, ClipOperation::Intersect);
            if self.stack.last().is_none_or(|s| s.clip_rects.is_empty()) {
                return;
            }
        }
        let (mesh, _) = cached_path_fill_mesh(path, scale);
        if !mesh.vertices.is_empty() && !mesh.indices.is_empty() {
            self.commands.push(DrawCommand::PushClip {
                mesh,
                transform: tx,
                is_difference: op == ClipOperation::Difference,
                clip_rect: pre_clip_scissor,
            });
            if let Some(top) = self.stack.last_mut() {
                top.stencil_clips.push(pre_clip_scissor);
            }
        }
    }

    /// Updates conservative device clip bounds for `path` without tessellating stencil clip geometry.
    pub fn clip_path_bounds_only(&mut self, path: &Path, op: ClipOperation) {
        if op == ClipOperation::Intersect {
            let tx = self.get_current_transform();
            let transformed = path
                .get_bounding_box()
                .filter(|b| !b.is_empty())
                .and_then(|_| path.get_transformed_bounding_box(&tx))
                .unwrap_or(Rect::ZERO);
            self.apply_device_clip_rect(transformed, ClipOperation::Intersect);
        }
    }

    fn apply_device_clip_rect(&mut self, device_rect: Rect, op: ClipOperation) {
        let Some(top) = self.stack.last_mut() else {
            return;
        };
        top.has_explicit_clip = true;
        match op {
            ClipOperation::Intersect => {
                if device_rect.is_empty() {
                    top.clip_rects.clear();
                } else {
                    let mut next = Vec::with_capacity(top.clip_rects.len());
                    for existing in &top.clip_rects {
                        if let Some(inter) = intersect_rects(*existing, device_rect) {
                            next.push(inter);
                        }
                    }
                    top.clip_rects = next;
                }
            }
            ClipOperation::Difference => {
                if !device_rect.is_empty() {
                    let mut next = Vec::new();
                    for existing in &top.clip_rects {
                        next.extend(subtract_rect(*existing, device_rect));
                    }
                    top.clip_rects = next;
                }
            }
        }
    }

    fn emit_mesh_with_transform(
        &mut self,
        mesh: TessellatedMesh,
        transform: Matrix,
        paint: &Paint,
    ) {
        self.emit_mesh_with_transform_and_coverage(mesh, transform, paint, 1.0);
    }

    fn emit_mesh_with_transform_and_coverage(
        &mut self,
        mesh: TessellatedMesh,
        transform: Matrix,
        paint: &Paint,
        coverage: Scalar,
    ) {
        let clip_targets = self.resolve_clip_targets(None, &transform);
        self.emit_mesh_with_clip_targets(mesh, transform, paint, coverage, &clip_targets);
    }

    fn emit_mesh_with_clip_targets(
        &mut self,
        mesh: TessellatedMesh,
        transform: Matrix,
        paint: &Paint,
        coverage: Scalar,
        clip_targets: &[Option<Rect>],
    ) {
        if clip_targets.is_empty() || mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return;
        }
        let cov = coverage.clamp(0.0, 1.0);
        if cov <= 0.0 {
            return;
        }

        let is_image_shader = matches!(
            paint.effective_color_source(),
            Some(ColorSource::Image { .. })
        );
        let needs_filter_layer =
            paint.image_filter.is_some() || (is_image_shader && paint.color_filter.is_some());

        if needs_filter_layer {
            let Some(dev_bounds) = mesh_device_bounds(&mesh, &transform) else {
                return;
            };
            let layer_cf = paint
                .color_filter
                .as_ref()
                .map(ColorFilter::to_layer_color_filter);
            let layer_if = paint
                .image_filter
                .as_ref()
                .map(ImageFilter::to_layer_image_filter);

            let mut mesh_opt = Some(mesh);
            let last_idx = clip_targets.len() - 1;
            for (i, &clip_rect) in clip_targets.iter().enumerate() {
                let cur_mesh = if i == last_idx {
                    mesh_opt.take().unwrap_or_default()
                } else {
                    mesh_opt.as_ref().cloned().unwrap_or_default()
                };
                let (inner_cmd, layer_alpha) = match paint.effective_color_source() {
                    Some(ColorSource::LinearGradient(grad)) => {
                        let mut normalized = grad.clone();
                        normalized.stops = Paint::convert_stops(&grad.stops);
                        normalized.alpha = (normalized.alpha * cov).clamp(0.0, 1.0);
                        (
                            DrawCommand::LinearGradientFill {
                                mesh: cur_mesh,
                                transform,
                                gradient: normalized,
                                blend_mode: BlendMode::SrcOver,
                                clip_rect: None,
                            },
                            1.0,
                        )
                    }
                    Some(ColorSource::RadialGradient(grad)) => {
                        let mut normalized = grad.clone();
                        normalized.stops = Paint::convert_stops(&grad.stops);
                        normalized.alpha = (normalized.alpha * cov).clamp(0.0, 1.0);
                        (
                            DrawCommand::RadialGradientFill {
                                mesh: cur_mesh,
                                transform,
                                gradient: normalized,
                                blend_mode: BlendMode::SrcOver,
                                clip_rect: None,
                            },
                            1.0,
                        )
                    }
                    Some(ColorSource::SweepGradient(grad)) => {
                        let mut normalized = grad.clone();
                        normalized.stops = Paint::convert_stops(&grad.stops);
                        normalized.alpha =
                            (normalized.alpha * paint.color.alpha * cov).clamp(0.0, 1.0);
                        (
                            DrawCommand::SweepGradientFill {
                                mesh: cur_mesh,
                                transform,
                                gradient: normalized,
                                blend_mode: BlendMode::SrcOver,
                                clip_rect: None,
                            },
                            1.0,
                        )
                    }
                    Some(ColorSource::ConicalGradient(grad)) => {
                        let mut normalized = grad.clone();
                        normalized.stops = Paint::convert_stops(&grad.stops);
                        normalized.alpha =
                            (normalized.alpha * paint.color.alpha * cov).clamp(0.0, 1.0);
                        (
                            DrawCommand::ConicalGradientFill {
                                mesh: cur_mesh,
                                transform,
                                gradient: normalized,
                                blend_mode: BlendMode::SrcOver,
                                clip_rect: None,
                            },
                            1.0,
                        )
                    }
                    Some(ColorSource::Image {
                        texture,
                        tile_mode_x,
                        tile_mode_y,
                        matrix,
                        linear_filter,
                    }) => {
                        let inv_w = 1.0 / (texture.width.max(1) as Scalar);
                        let inv_h = 1.0 / (texture.height.max(1) as Scalar);
                        let uv_mat = invert_matrix(matrix).unwrap_or(Matrix::IDENTITY);
                        let vertices: Vec<UvVertex> = cur_mesh
                            .vertices
                            .iter()
                            .map(|v| {
                                let pt = uv_mat
                                    .transform_point(Point::new(v.position[0], v.position[1]));
                                UvVertex::new(
                                    v.position[0],
                                    v.position[1],
                                    pt.x * inv_w,
                                    pt.y * inv_h,
                                )
                            })
                            .collect();
                        (
                            DrawCommand::TextureFill(TextureFillBatch {
                                vertices,
                                indices: cur_mesh.indices,
                                transform,
                                texture: texture.clone(),
                                alpha: 1.0,
                                tile_mode_x: *tile_mode_x,
                                tile_mode_y: *tile_mode_y,
                                linear_filter: *linear_filter,
                                blend_mode: BlendMode::SrcOver,
                                clip_rect: None,
                            }),
                            (paint.color.alpha * cov).clamp(0.0, 1.0),
                        )
                    }
                    Some(ColorSource::Solid(c)) => {
                        let mut color = *c;
                        if paint.invert_colors {
                            color = Color::new(
                                (1.0 - color.red).clamp(0.0, 1.0),
                                (1.0 - color.green).clamp(0.0, 1.0),
                                (1.0 - color.blue).clamp(0.0, 1.0),
                                color.alpha,
                            );
                        }
                        color.alpha = (color.alpha * cov).clamp(0.0, 1.0);
                        (
                            DrawCommand::SolidFill {
                                mesh: cur_mesh,
                                transform,
                                color,
                                blend_mode: BlendMode::SrcOver,
                                clip_rect: None,
                            },
                            1.0,
                        )
                    }
                    None => {
                        let mut color = paint.color;
                        if paint.invert_colors {
                            color = Color::new(
                                (1.0 - color.red).clamp(0.0, 1.0),
                                (1.0 - color.green).clamp(0.0, 1.0),
                                (1.0 - color.blue).clamp(0.0, 1.0),
                                color.alpha,
                            );
                        }
                        color.alpha = (color.alpha * cov).clamp(0.0, 1.0);
                        (
                            DrawCommand::SolidFill {
                                mesh: cur_mesh,
                                transform,
                                color,
                                blend_mode: BlendMode::SrcOver,
                                clip_rect: None,
                            },
                            1.0,
                        )
                    }
                };
                self.commands.push(DrawCommand::SaveLayer {
                    bounds: dev_bounds,
                    commands: vec![inner_cmd],
                    alpha: layer_alpha,
                    blend_mode: paint.blend_mode,
                    color_filter: layer_cf.clone(),
                    image_filter: layer_if.clone(),
                    clip_rect,
                });
            }
            return;
        }

        let mut mesh_opt = Some(mesh);
        let last_idx = clip_targets.len() - 1;
        for (i, &clip_rect) in clip_targets.iter().enumerate() {
            let cur_mesh = if i == last_idx {
                mesh_opt.take().unwrap_or_default()
            } else {
                mesh_opt.as_ref().cloned().unwrap_or_default()
            };
            match paint.effective_color_source() {
                Some(ColorSource::LinearGradient(grad)) => {
                    let mut normalized = grad.clone();
                    normalized.stops = Paint::convert_stops(&grad.stops);
                    normalized.alpha = (normalized.alpha * cov).clamp(0.0, 1.0);
                    self.commands.push(DrawCommand::LinearGradientFill {
                        mesh: cur_mesh,
                        transform,
                        gradient: normalized,
                        blend_mode: paint.blend_mode,
                        clip_rect,
                    });
                }
                Some(ColorSource::RadialGradient(grad)) => {
                    let mut normalized = grad.clone();
                    normalized.stops = Paint::convert_stops(&grad.stops);
                    normalized.alpha = (normalized.alpha * cov).clamp(0.0, 1.0);
                    self.commands.push(DrawCommand::RadialGradientFill {
                        mesh: cur_mesh,
                        transform,
                        gradient: normalized,
                        blend_mode: paint.blend_mode,
                        clip_rect,
                    });
                }
                Some(ColorSource::SweepGradient(grad)) => {
                    let mut normalized = grad.clone();
                    normalized.stops = Paint::convert_stops(&grad.stops);
                    if let Some(cf) = &paint.color_filter {
                        for stop in &mut normalized.stops {
                            stop.color = cf.apply(stop.color);
                        }
                    }
                    normalized.alpha = (normalized.alpha * paint.color.alpha * cov).clamp(0.0, 1.0);
                    self.commands.push(DrawCommand::SweepGradientFill {
                        mesh: cur_mesh,
                        transform,
                        gradient: normalized,
                        blend_mode: paint.blend_mode,
                        clip_rect,
                    });
                }
                Some(ColorSource::ConicalGradient(grad)) => {
                    let mut normalized = grad.clone();
                    normalized.stops = Paint::convert_stops(&grad.stops);
                    if let Some(cf) = &paint.color_filter {
                        for stop in &mut normalized.stops {
                            stop.color = cf.apply(stop.color);
                        }
                    }
                    normalized.alpha = (normalized.alpha * paint.color.alpha * cov).clamp(0.0, 1.0);
                    self.commands.push(DrawCommand::ConicalGradientFill {
                        mesh: cur_mesh,
                        transform,
                        gradient: normalized,
                        blend_mode: paint.blend_mode,
                        clip_rect,
                    });
                }
                Some(ColorSource::Image {
                    texture,
                    tile_mode_x,
                    tile_mode_y,
                    matrix,
                    linear_filter,
                }) => {
                    let inv_w = 1.0 / (texture.width.max(1) as Scalar);
                    let inv_h = 1.0 / (texture.height.max(1) as Scalar);
                    let uv_mat = invert_matrix(matrix).unwrap_or(Matrix::IDENTITY);
                    let vertices: Vec<UvVertex> = cur_mesh
                        .vertices
                        .iter()
                        .map(|v| {
                            let pt =
                                uv_mat.transform_point(Point::new(v.position[0], v.position[1]));
                            UvVertex::new(v.position[0], v.position[1], pt.x * inv_w, pt.y * inv_h)
                        })
                        .collect();
                    self.commands
                        .push(DrawCommand::TextureFill(TextureFillBatch {
                            vertices,
                            indices: cur_mesh.indices,
                            transform,
                            texture: texture.clone(),
                            alpha: (paint.color.alpha * cov).clamp(0.0, 1.0),
                            tile_mode_x: *tile_mode_x,
                            tile_mode_y: *tile_mode_y,
                            linear_filter: *linear_filter,
                            blend_mode: paint.blend_mode,
                            clip_rect,
                        }));
                }
                Some(ColorSource::Solid(_)) | None => {
                    let mut color = paint.effective_solid_color();
                    color.alpha = (color.alpha * cov).clamp(0.0, 1.0);
                    self.commands.push(DrawCommand::SolidFill {
                        mesh: cur_mesh,
                        transform,
                        color,
                        blend_mode: paint.blend_mode,
                        clip_rect,
                    });
                }
            }
        }
    }

    fn emit_mesh(&mut self, mesh: TessellatedMesh, paint: &Paint) {
        let transform = self.get_current_transform();
        self.emit_mesh_with_transform(mesh, transform, paint);
    }

    /// Fills the entire canvas (bounded by the current clip) with `paint`,
    /// matching C++ `Canvas::DrawPaint` and `CoverGeometry`.
    pub fn draw_paint(&mut self, paint: &Paint) {
        let w = self.width as Scalar;
        let h = self.height as Scalar;
        let transform = self.get_current_transform();

        if let Some(inv) = invert_matrix(&transform) {
            let p0 = inv.transform_point(Point::new(0.0, 0.0));
            let p1 = inv.transform_point(Point::new(w, 0.0));
            let p2 = inv.transform_point(Point::new(w, h));
            let p3 = inv.transform_point(Point::new(0.0, h));
            let mesh = TessellatedMesh {
                vertices: vec![
                    Vertex2D::from_point(p0),
                    Vertex2D::from_point(p1),
                    Vertex2D::from_point(p2),
                    Vertex2D::from_point(p3),
                ],
                indices: vec![0, 1, 2, 0, 2, 3],
            };
            self.emit_mesh_with_transform(mesh, transform, paint);
        } else {
            let mesh = TessellatedMesh {
                vertices: vec![
                    Vertex2D::new(0.0, 0.0),
                    Vertex2D::new(w, 0.0),
                    Vertex2D::new(w, h),
                    Vertex2D::new(0.0, h),
                ],
                indices: vec![0, 1, 2, 0, 2, 3],
            };
            self.emit_mesh_with_transform(mesh, Matrix::IDENTITY, paint);
        }
    }

    pub fn draw_color(&mut self, color: Color, blend_mode: BlendMode) {
        let paint = Paint::fill(color).with_blend_mode(blend_mode);
        self.draw_paint(&paint);
    }

    pub fn draw_path(&mut self, path: &Path, paint: &Paint) {
        let Some(raw_bbox) = path.get_bounding_box() else {
            return;
        };
        let transform = self.get_current_transform();
        let scale = transform.get_max_basis_length_xy().max(1e-4);
        let imf = paint.image_filter.as_ref();
        match paint.style {
            DrawStyle::Fill => {
                let clip_targets =
                    self.resolve_clip_targets_filtered(Some(raw_bbox), &transform, imf);
                if clip_targets.is_empty() {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                self.replay_stats.path_fill_count += 1;
                let (mesh, kind) = cached_path_fill_mesh(path, scale);
                match kind {
                    FillTessellationKind::ConvexFast
                    | FillTessellationKind::PslgSimpleConvex
                    | FillTessellationKind::PslgDisjointConvex => {
                        self.replay_stats.path_convex_fast_count += 1;
                    }
                    FillTessellationKind::PslgGeneral => {
                        self.replay_stats.path_pslg_count += 1;
                    }
                }
                self.emit_mesh_with_clip_targets(mesh, transform, paint, 1.0, &clip_targets);
            }
            DrawStyle::Stroke => {
                let width = paint.effective_stroke_width();
                let eff_w = width.max(1.0 / scale);
                let miter_mult = if paint.stroke_join == Join::Miter {
                    paint.stroke_miter.max(1.0)
                } else {
                    1.0
                };
                let cap_mult = if paint.stroke_cap == Cap::Square {
                    std::f32::consts::SQRT_2
                } else {
                    1.0
                };
                let pad = eff_w * 0.5 * miter_mult.max(cap_mult);
                let local_bounds = expand_rect(raw_bbox, pad);
                let clip_targets =
                    self.resolve_clip_targets_filtered(Some(local_bounds), &transform, imf);
                if clip_targets.is_empty() {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                self.replay_stats.path_stroke_count += 1;
                let cov = compute_stroke_alpha_coverage(&transform, width);
                let stroke = StrokeParameters {
                    width,
                    cap: paint.stroke_cap,
                    join: paint.stroke_join,
                    miter_limit: paint.stroke_miter,
                };
                let mesh = cached_path_stroke_mesh(path, stroke, scale);
                self.emit_mesh_with_clip_targets(mesh, transform, paint, cov, &clip_targets);
            }
        }
    }

    pub fn draw_rect(&mut self, rect: Rect, paint: &Paint) {
        if paint.style == DrawStyle::Fill && rect.is_empty() {
            return;
        }
        if paint.is_shadow_blur_draw_operation() {
            self.draw_rrect_shadow_with_blend(
                rect,
                0.0,
                paint.mask_blur_sigma,
                paint.effective_solid_color(),
                paint.blend_mode,
            );
            return;
        }
        let transform = self.get_current_transform();
        let imf = paint.image_filter.as_ref();
        match paint.style {
            DrawStyle::Fill => {
                let clip_targets = self.resolve_clip_targets_filtered(Some(rect), &transform, imf);
                if clip_targets.is_empty() {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                let l = rect.get_left();
                let t = rect.get_top();
                let r = rect.get_right();
                let b = rect.get_bottom();
                let mesh = TessellatedMesh {
                    vertices: vec![
                        Vertex2D::new(l, t),
                        Vertex2D::new(r, t),
                        Vertex2D::new(r, b),
                        Vertex2D::new(l, b),
                    ],
                    indices: vec![0, 1, 2, 0, 2, 3],
                };
                self.emit_mesh_with_clip_targets(mesh, transform, paint, 1.0, &clip_targets);
            }
            DrawStyle::Stroke => {
                let scale = transform.get_max_basis_length_xy().max(1e-4);
                let width = paint.effective_stroke_width();
                let stroke_w = width.max(1.0 / scale);
                let half_w = stroke_w * 0.5;
                if paint.stroke_join == Join::Miter
                    && paint.stroke_miter >= std::f32::consts::SQRT_2
                    && rect.get_width() > stroke_w
                    && rect.get_height() > stroke_w
                {
                    let local_bounds = expand_rect(rect, half_w);
                    let clip_targets =
                        self.resolve_clip_targets_filtered(Some(local_bounds), &transform, imf);
                    if clip_targets.is_empty() {
                        self.replay_stats.culled_count += 1;
                        return;
                    }
                    let l = rect.get_left();
                    let t = rect.get_top();
                    let r = rect.get_right();
                    let b = rect.get_bottom();
                    let mesh = TessellatedMesh {
                        vertices: vec![
                            Vertex2D::new(l - half_w, t - half_w),
                            Vertex2D::new(r + half_w, t - half_w),
                            Vertex2D::new(r + half_w, b + half_w),
                            Vertex2D::new(l - half_w, b + half_w),
                            Vertex2D::new(l + half_w, t + half_w),
                            Vertex2D::new(r - half_w, t + half_w),
                            Vertex2D::new(r - half_w, b - half_w),
                            Vertex2D::new(l + half_w, b - half_w),
                        ],
                        indices: vec![
                            0, 1, 5, 0, 5, 4, // Top
                            1, 2, 6, 1, 6, 5, // Right
                            2, 3, 7, 2, 7, 6, // Bottom
                            3, 0, 4, 3, 4, 7, // Left
                        ],
                    };
                    let cov = compute_stroke_alpha_coverage(&transform, width);
                    self.emit_mesh_with_clip_targets(mesh, transform, paint, cov, &clip_targets);
                } else {
                    let mut pb = PathBuilder::new();
                    pb.add_rect(rect);
                    self.draw_path(&pb.build(FillType::NonZero), paint);
                }
            }
        }
    }

    pub fn draw_rrect(&mut self, rect: Rect, radii: RoundingRadii, paint: &Paint) {
        if paint.style == DrawStyle::Fill && rect.is_empty() {
            return;
        }
        if paint.is_shadow_blur_draw_operation() {
            let corner_radius = (radii.top_left.width + radii.top_left.height) * 0.5;
            self.draw_rrect_shadow_with_blend(
                rect,
                corner_radius,
                paint.mask_blur_sigma,
                paint.effective_solid_color(),
                paint.blend_mode,
            );
            return;
        }
        let transform = self.get_current_transform();
        let imf = paint.image_filter.as_ref();
        let scale = transform.get_max_basis_length_xy().max(1e-4);
        let dx = rect.get_left();
        let dy = rect.get_top();
        let w = rect.get_width();
        let h = rect.get_height();
        let needs_local_uv = paint_uses_local_coords_for_shader(paint);
        let (eff_rect, draw_tx) = if needs_local_uv || (dx == 0.0 && dy == 0.0) {
            (rect, transform)
        } else {
            (
                Rect::from_xywh(0.0, 0.0, w, h),
                transform * Matrix::make_translation(dx, dy, 0.0),
            )
        };
        match paint.style {
            DrawStyle::Fill => {
                let clip_targets = self.resolve_clip_targets_filtered(Some(rect), &transform, imf);
                if clip_targets.is_empty() {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                let mesh = RoundRectGeometry::new(eff_rect, radii).get_mesh(scale.max(1.0));
                self.emit_mesh_with_clip_targets(mesh, draw_tx, paint, 1.0, &clip_targets);
            }
            DrawStyle::Stroke => {
                let width = paint.effective_stroke_width();
                let stroke_w = width.max(1.0 / scale);
                let local_bounds = expand_rect(rect, stroke_w * 0.5);
                let clip_targets =
                    self.resolve_clip_targets_filtered(Some(local_bounds), &transform, imf);
                if clip_targets.is_empty() {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                let cov = compute_stroke_alpha_coverage(&transform, width);
                let stroke = StrokeParameters {
                    width: stroke_w,
                    cap: paint.stroke_cap,
                    join: paint.stroke_join,
                    miter_limit: paint.stroke_miter,
                };
                let mesh = RoundRectGeometry::new(eff_rect, radii).get_stroked_mesh(stroke, scale);
                self.emit_mesh_with_clip_targets(mesh, draw_tx, paint, cov, &clip_targets);
            }
        }
    }

    pub fn draw_diff_rrect(
        &mut self,
        outer_rect: Rect,
        outer_radii: RoundingRadii,
        inner_rect: Rect,
        inner_radii: RoundingRadii,
        paint: &Paint,
    ) {
        if paint.style == DrawStyle::Fill {
            if outer_rect.is_empty() {
                return;
            }
            let transform = self.get_current_transform();
            let clip_targets = self.resolve_clip_targets_filtered(
                Some(outer_rect),
                &transform,
                paint.image_filter.as_ref(),
            );
            if clip_targets.is_empty() {
                self.replay_stats.culled_count += 1;
                return;
            }
            let scale = transform.get_max_basis_length_xy().max(1.0);
            let needs_local_uv = paint_uses_local_coords_for_shader(paint);
            let dx = outer_rect.get_left();
            let dy = outer_rect.get_top();
            let (eff_outer, eff_inner, draw_tx) = if needs_local_uv || (dx == 0.0 && dy == 0.0) {
                (outer_rect, inner_rect, transform)
            } else {
                (
                    Rect::from_xywh(0.0, 0.0, outer_rect.get_width(), outer_rect.get_height()),
                    Rect::from_xywh(
                        inner_rect.get_left() - dx,
                        inner_rect.get_top() - dy,
                        inner_rect.get_width(),
                        inner_rect.get_height(),
                    ),
                    transform * Matrix::make_translation(dx, dy, 0.0),
                )
            };
            if let Some(mesh) =
                DiffRoundRectGeometry::new(eff_outer, outer_radii, eff_inner, inner_radii)
                    .get_mesh(scale)
            {
                self.emit_mesh_with_clip_targets(mesh, draw_tx, paint, 1.0, &clip_targets);
                return;
            }
        }
        let mut pb = PathBuilder::new();
        pb.add_rounded_rect(outer_rect, outer_radii);
        pb.add_rounded_rect(inner_rect, inner_radii);
        let fill_type = match paint.style {
            DrawStyle::Fill => FillType::Odd,
            DrawStyle::Stroke => FillType::NonZero,
        };
        self.draw_path(&pb.build(fill_type), paint);
    }

    pub fn draw_oval(&mut self, rect: Rect, paint: &Paint) {
        if paint.style == DrawStyle::Fill && rect.is_empty() {
            return;
        }
        if (rect.get_width() - rect.get_height()).abs() <= K_EH_CLOSE_ENOUGH {
            let radius = rect.get_width() * 0.5;
            let center = Point::new(rect.get_left() + radius, rect.get_top() + radius);
            self.draw_circle(center, radius, paint);
            return;
        }
        let transform = self.get_current_transform();
        let imf = paint.image_filter.as_ref();
        let scale = transform.get_max_basis_length_xy().max(1e-4);
        match paint.style {
            DrawStyle::Fill => {
                let clip_targets = self.resolve_clip_targets_filtered(Some(rect), &transform, imf);
                if clip_targets.is_empty() {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                let mesh = EllipseGeometry::new(rect).get_mesh(scale.max(1.0));
                self.emit_mesh_with_clip_targets(mesh, transform, paint, 1.0, &clip_targets);
            }
            DrawStyle::Stroke => {
                let stroke_w = paint.effective_stroke_width().max(1.0 / scale);
                let local_bounds = expand_rect(rect, stroke_w * 0.5);
                if self
                    .resolve_clip_targets_filtered(Some(local_bounds), &transform, imf)
                    .is_empty()
                {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                let mut pb = PathBuilder::new();
                pb.add_oval(rect);
                self.draw_path(&pb.build(FillType::NonZero), paint);
            }
        }
    }

    pub fn draw_circle(&mut self, center: Point, radius: Scalar, paint: &Paint) {
        if radius <= 0.0 {
            return;
        }
        let circle_rect = Rect::from_xywh(
            center.x - radius,
            center.y - radius,
            radius * 2.0,
            radius * 2.0,
        );
        if paint.is_shadow_blur_draw_operation() {
            self.draw_rrect_shadow_with_blend(
                circle_rect,
                radius,
                paint.mask_blur_sigma,
                paint.effective_solid_color(),
                paint.blend_mode,
            );
            return;
        }
        let transform = self.get_current_transform();
        let imf = paint.image_filter.as_ref();
        let scale = transform.get_max_basis_length_xy().max(1.0);
        let dx = center.x;
        let dy = center.y;
        let needs_local_uv = paint_uses_local_coords_for_shader(paint);
        match paint.style {
            DrawStyle::Fill => {
                let clip_targets =
                    self.resolve_clip_targets_filtered(Some(circle_rect), &transform, imf);
                if clip_targets.is_empty() {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                let base_arc = cached_circle_fill_mesh_arc(radius, scale);
                let (mesh, draw_tx) = if needs_local_uv {
                    (translate_cached_mesh(&base_arc, dx, dy), transform)
                } else if dx == 0.0 && dy == 0.0 {
                    ((*base_arc).clone(), transform)
                } else {
                    (
                        (*base_arc).clone(),
                        transform * Matrix::make_translation(dx, dy, 0.0),
                    )
                };
                self.emit_mesh_with_clip_targets(mesh, draw_tx, paint, 1.0, &clip_targets);
            }
            DrawStyle::Stroke => {
                let width = paint.effective_stroke_width();
                let stroke_w = width.max(1.0 / scale);
                let local_bounds = expand_rect(circle_rect, stroke_w * 0.5);
                let clip_targets =
                    self.resolve_clip_targets_filtered(Some(local_bounds), &transform, imf);
                if clip_targets.is_empty() {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                let cov = compute_stroke_alpha_coverage(&transform, width);
                let base_arc = cached_circle_stroke_mesh_arc(radius, stroke_w, scale);
                let (mesh, draw_tx) = if needs_local_uv {
                    (translate_cached_mesh(&base_arc, dx, dy), transform)
                } else if dx == 0.0 && dy == 0.0 {
                    ((*base_arc).clone(), transform)
                } else {
                    (
                        (*base_arc).clone(),
                        transform * Matrix::make_translation(dx, dy, 0.0),
                    )
                };
                self.emit_mesh_with_clip_targets(mesh, draw_tx, paint, cov, &clip_targets);
            }
        }
    }

    pub fn draw_arc(
        &mut self,
        oval_bounds: Rect,
        start: Radians,
        sweep: Radians,
        use_center: bool,
        paint: &Paint,
    ) {
        if oval_bounds.is_empty() || sweep.0.abs() <= Scalar::EPSILON {
            return;
        }
        let transform = self.get_current_transform();
        let imf = paint.image_filter.as_ref();
        let scale = transform.get_max_basis_length_xy().max(1e-4);
        let arc_geom = ArcGeometry::new(oval_bounds, start, sweep, use_center);
        match paint.style {
            DrawStyle::Fill => {
                let clip_targets = self.resolve_clip_targets_filtered(
                    Some(oval_bounds.get_positive()),
                    &transform,
                    imf,
                );
                if clip_targets.is_empty() {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                let mesh = arc_geom.get_mesh(scale.max(1.0));
                self.emit_mesh_with_clip_targets(mesh, transform, paint, 1.0, &clip_targets);
            }
            DrawStyle::Stroke => {
                let width = paint.effective_stroke_width();
                let stroke_w = width.max(1.0 / scale);
                let local_bounds = expand_rect(
                    oval_bounds.get_positive(),
                    stroke_w * 0.5 * std::f32::consts::SQRT_2,
                );
                let clip_targets =
                    self.resolve_clip_targets_filtered(Some(local_bounds), &transform, imf);
                if clip_targets.is_empty() {
                    self.replay_stats.culled_count += 1;
                    return;
                }
                let cov = compute_stroke_alpha_coverage(&transform, width);
                let mesh = arc_geom.get_stroked_mesh(
                    StrokeParameters {
                        width: stroke_w,
                        cap: paint.stroke_cap,
                        join: paint.stroke_join,
                        miter_limit: paint.stroke_miter,
                    },
                    scale,
                );
                self.emit_mesh_with_clip_targets(mesh, transform, paint, cov, &clip_targets);
            }
        }
    }

    pub fn draw_line(&mut self, p0: Point, p1: Point, paint: &Paint) {
        let transform = self.get_current_transform();
        let scale = transform.get_max_basis_length_xy().max(1e-4);
        let width = paint.effective_stroke_width();
        let pad = width.max(1.0 / scale) * 0.75;
        let local_bounds = Rect::from_ltrb(
            p0.x.min(p1.x) - pad,
            p0.y.min(p1.y) - pad,
            p0.x.max(p1.x) + pad,
            p0.y.max(p1.y) + pad,
        );
        let clip_targets = self.resolve_clip_targets_filtered(
            Some(local_bounds),
            &transform,
            paint.image_filter.as_ref(),
        );
        if clip_targets.is_empty() {
            self.replay_stats.culled_count += 1;
            return;
        }
        let cov = compute_stroke_alpha_coverage(&transform, width);
        let mesh = LineGeometry::new(p0, p1, width, paint.stroke_cap).get_mesh(scale);
        self.emit_mesh_with_clip_targets(mesh, transform, paint, cov, &clip_targets);
    }

    pub fn draw_dashed_line(
        &mut self,
        p0: Point,
        p1: Point,
        on_length: Scalar,
        off_length: Scalar,
        paint: &Paint,
    ) {
        let total_len = p0.get_distance(p1);
        if total_len <= 0.0 || on_length < 0.0 || off_length <= 0.0 {
            self.draw_line(p0, p1, paint);
            return;
        }

        let transform = self.get_current_transform();
        let scale = transform.get_max_basis_length_xy().max(1e-4);
        let width = paint.effective_stroke_width();
        let cov = compute_stroke_alpha_coverage(&transform, width);
        let dir = (p1 - p0) / total_len;
        let step = on_length + off_length;
        let mut combined = TessellatedMesh::default();
        let mut dist = 0.0;
        while dist < total_len {
            let end_dist = (dist + on_length).min(total_len);
            let seg_p0 = p0 + dir * dist;
            let seg_p1 = p0 + dir * end_dist;
            let seg_mesh =
                LineGeometry::new(seg_p0, seg_p1, width, paint.stroke_cap).get_mesh(scale);
            let base = combined.vertices.len() as u32;
            combined.vertices.extend(seg_mesh.vertices);
            combined
                .indices
                .extend(seg_mesh.indices.into_iter().map(|idx| base + idx));
            dist += step;
        }
        self.emit_mesh_with_transform_and_coverage(combined, transform, paint, cov);
    }

    pub fn draw_points(
        &mut self,
        points: &[Point],
        radius: Scalar,
        paint: &Paint,
        point_style: PointStyle,
    ) {
        if radius <= 0.0 {
            return;
        }
        let mut fill_paint = paint.clone();
        fill_paint.style = DrawStyle::Fill;
        for &pt in points {
            match point_style {
                PointStyle::Round => self.draw_circle(pt, radius, &fill_paint),
                PointStyle::Square => {
                    let r =
                        Rect::from_xywh(pt.x - radius, pt.y - radius, radius * 2.0, radius * 2.0);
                    self.draw_rect(r, &fill_paint);
                }
            }
        }
    }

    pub fn draw_rrect_shadow(
        &mut self,
        rect: Rect,
        corner_radius: Scalar,
        blur_sigma: Scalar,
        color: Color,
    ) {
        self.draw_rrect_shadow_with_blend(
            rect,
            corner_radius,
            blur_sigma,
            color,
            BlendMode::SrcOver,
        );
    }

    pub fn draw_rrect_shadow_with_blend(
        &mut self,
        rect: Rect,
        corner_radius: Scalar,
        blur_sigma: Scalar,
        color: Color,
        blend_mode: BlendMode,
    ) {
        let params = RRectBlurParams {
            rect,
            corner_radius,
            blur_sigma,
            color,
        };
        let transform = self.get_current_transform();
        let clip_targets = self.resolve_clip_targets(Some(params.coverage_rect()), &transform);
        if clip_targets.is_empty() {
            self.replay_stats.culled_count += 1;
            return;
        }
        for clip_rect in clip_targets {
            self.commands.push(DrawCommand::RRectBlur {
                transform,
                params,
                blend_mode,
                clip_rect,
            });
        }
    }

    pub fn draw_dashed_path(
        &mut self,
        path: &Path,
        dash_offset: Scalar,
        intervals: &[Scalar],
        paint: &Paint,
    ) {
        let dashed = dash_path(path, dash_offset, intervals);
        let mut stroke_paint = paint.clone();
        stroke_paint.style = DrawStyle::Stroke;
        self.draw_path(&dashed, &stroke_paint);
    }

    pub fn draw_text_frame(&mut self, text_frame: &TextFrame, origin: Point, paint: &Paint) {
        let transform = self.get_current_transform();
        let is_unit_translation = (transform.e(0, 0) - 1.0).abs() <= 1e-3
            && (transform.e(1, 1) - 1.0).abs() <= 1e-3
            && transform.e(0, 1).abs() <= 1e-3
            && transform.e(1, 0).abs() <= 1e-3
            && !transform.has_perspective_2d();
        self.draw_text_frame_with_transform(
            text_frame,
            origin,
            paint,
            transform,
            !is_unit_translation,
        );
    }

    fn draw_text_frame_with_transform(
        &mut self,
        text_frame: &TextFrame,
        origin: Point,
        paint: &Paint,
        transform: Matrix,
        use_linear_filter: bool,
    ) {
        if text_frame.quads.is_empty() {
            return;
        }
        let Some(top) = self.stack.last() else {
            return;
        };
        if top.clip_rects.is_empty() {
            return;
        }

        let paint_color = paint.effective_solid_color();
        let use_paint_override = paint.color != Color::BLACK
            || paint.effective_color_source().is_some()
            || paint.invert_colors;

        let mut adjusted_quads = Vec::with_capacity(text_frame.quads.len());
        for q in &text_frame.quads {
            let dest_rect = Rect::from_xywh(
                q.dest_rect.get_left() + origin.x,
                q.dest_rect.get_top() + origin.y,
                q.dest_rect.get_width(),
                q.dest_rect.get_height(),
            );
            let color = if !use_paint_override && q.color != Color::WHITE {
                q.color
            } else if q.color == Color::WHITE {
                paint_color
            } else {
                Color::new(
                    q.color.red * paint_color.red,
                    q.color.green * paint_color.green,
                    q.color.blue * paint_color.blue,
                    q.color.alpha * paint_color.alpha,
                )
            };
            adjusted_quads.push(AtlasGlyphQuad {
                dest_rect,
                atlas_rect: q.atlas_rect,
                color,
                is_color: q.is_color,
            });
        }

        let prev_len = self.commands.len();
        let inner_blend = if paint.image_filter.is_some() {
            BlendMode::SrcOver
        } else {
            paint.blend_mode
        };
        self.emit_glyph_quads_with_blend(adjusted_quads, transform, inner_blend, use_linear_filter);
        if self.commands.len() > prev_len {
            self.sync_glyph_atlases();
            if let Some(imf) = &paint.image_filter {
                let mut sub_commands: Vec<DrawCommand> = self.commands.drain(prev_len..).collect();
                for cmd in &mut sub_commands {
                    if let DrawCommand::GlyphAtlas(b) = cmd {
                        b.clip_rect = None;
                    }
                }
                if let Some(bounds) = commands_device_bounds(&sub_commands) {
                    self.commands.push(DrawCommand::SaveLayer {
                        bounds,
                        commands: sub_commands,
                        alpha: 1.0,
                        blend_mode: paint.blend_mode,
                        color_filter: None,
                        image_filter: Some(imf.to_layer_image_filter()),
                        clip_rect: self.current_scissor_rect(),
                    });
                }
            }
        }
    }

    pub fn draw_text(&mut self, text: &str, style: TextStyle, origin: Point) {
        let tx = self.get_current_transform();
        if tx.is_translation_scale_only() && tx.e(0, 0) > 0.0 && tx.e(1, 1) > 0.0 {
            let dev_pt = tx.transform_point(origin);
            let dev_origin = Point::new(dev_pt.x.round(), dev_pt.y.round());
            let scale_y = tx.e(1, 1).abs().clamp(0.25, 8.0);
            let scale_x = tx.e(0, 0).abs().clamp(0.25, 8.0);
            let scaled_style = TextStyle {
                font: style.font,
                font_size: (style.font_size * scale_y).round().max(1.0),
                line_height: (style.line_height * scale_y).round().max(1.0),
                color: style.color,
                max_width: style.max_width.map(|w| (w * scale_x).round().max(1.0)),
            };
            let frame = self
                .typographer_mut()
                .shape_text(text, scaled_style, dev_origin);
            self.draw_text_frame_with_transform(
                &frame,
                Point::ZERO,
                &Paint::fill(style.color),
                Matrix::IDENTITY,
                false,
            );
            return;
        }

        let ctm_scale = tx.get_max_basis_length_xy().clamp(0.25, 8.0);
        let frame = if (ctm_scale - 1.0).abs() > 1e-3 {
            let scaled_style = TextStyle {
                font: style.font,
                font_size: style.font_size * ctm_scale,
                line_height: style.line_height * ctm_scale,
                color: style.color,
                max_width: style.max_width.map(|w| w * ctm_scale),
            };
            let mut f = self
                .typographer_mut()
                .shape_text(text, scaled_style, Point::ZERO);
            let inv_scale = 1.0 / ctm_scale;
            for q in &mut f.quads {
                q.dest_rect = Rect::from_xywh(
                    q.dest_rect.get_left() * inv_scale,
                    q.dest_rect.get_top() * inv_scale,
                    q.dest_rect.get_width() * inv_scale,
                    q.dest_rect.get_height() * inv_scale,
                );
            }
            f.bounds = Rect::from_xywh(
                f.bounds.get_left() * inv_scale,
                f.bounds.get_top() * inv_scale,
                f.bounds.get_width() * inv_scale,
                f.bounds.get_height() * inv_scale,
            );
            f.longest_line *= inv_scale;
            f
        } else {
            self.typographer_mut().shape_text(text, style, Point::ZERO)
        };
        self.draw_text_frame(&frame, origin, &Paint::fill(style.color));
    }

    pub fn draw_icon(&mut self, icon: MaterialIcon, size: Scalar, color: Color, origin: Point) {
        let mut buf = [0u8; 4];
        let s = icon.codepoint().encode_utf8(&mut buf);
        let style = TextStyle {
            font: BuiltinFont::MaterialIcons,
            font_size: size,
            line_height: size,
            color,
            max_width: None,
        };
        self.draw_text(s, style, origin);
    }

    pub fn glyph_to_path(
        &mut self,
        ch: char,
        font: BuiltinFont,
        font_size: Scalar,
        origin: Point,
    ) -> Path {
        self.typographer_mut()
            .glyph_to_path(ch, font, font_size, origin)
    }

    pub fn draw_glyph_path(
        &mut self,
        ch: char,
        font: BuiltinFont,
        font_size: Scalar,
        origin: Point,
        paint: &Paint,
    ) {
        let path = self
            .typographer_mut()
            .glyph_to_path(ch, font, font_size, origin);
        self.draw_path(&path, paint);
    }

    fn emit_paragraph_quads(&mut self, paragraph: &Paragraph, offset: Point) {
        if paragraph.quads.is_empty() {
            return;
        }
        let Some(top) = self.stack.last() else {
            return;
        };
        if top.clip_rects.is_empty() {
            self.replay_stats.culled_count += 1;
            return;
        }
        let canvas_rect = Rect::from_xywh(0.0, 0.0, self.width as Scalar, self.height as Scalar);
        let Some(canvas_bounds) = intersect_rects(canvas_rect, self.initial_cull_rect) else {
            self.replay_stats.culled_count += 1;
            return;
        };

        let transform = top.transform;
        let is_unit_translation = (transform.e(0, 0) - 1.0).abs() <= 1e-3
            && (transform.e(1, 1) - 1.0).abs() <= 1e-3
            && transform.e(0, 1).abs() <= 1e-3
            && transform.e(1, 0).abs() <= 1e-3
            && !transform.has_perspective_2d();

        if let Some(qb) = paragraph.quad_bounds {
            if !qb.is_empty() {
                let shifted_bounds = Rect::from_xywh(
                    qb.get_left() + offset.x,
                    qb.get_top() + offset.y,
                    qb.get_width(),
                    qb.get_height(),
                );
                let dev_bounds = transform_rect(&transform, shifted_bounds);
                let shift_all_quads =
                    |quads: &[AtlasGlyphQuad], off: Point| -> Vec<AtlasGlyphQuad> {
                        if off.x == 0.0 && off.y == 0.0 {
                            quads.to_vec()
                        } else {
                            quads
                                .iter()
                                .map(|q| AtlasGlyphQuad {
                                    dest_rect: Rect::from_xywh(
                                        q.dest_rect.get_left() + off.x,
                                        q.dest_rect.get_top() + off.y,
                                        q.dest_rect.get_width(),
                                        q.dest_rect.get_height(),
                                    ),
                                    atlas_rect: q.atlas_rect,
                                    color: q.color,
                                    is_color: q.is_color,
                                })
                                .collect()
                        }
                    };

                if !top.has_explicit_clip {
                    if intersect_rects(dev_bounds, canvas_bounds).is_none() {
                        self.replay_stats.culled_count += 1;
                        return;
                    }
                    if rect_contains_rect(canvas_bounds, dev_bounds) {
                        let surviving = shift_all_quads(&paragraph.quads, offset);
                        let alpha_atlas = if paragraph.uses_alpha {
                            self.default_alpha_atlas.clone()
                        } else {
                            None
                        };
                        let color_atlas = if paragraph.uses_color {
                            self.default_color_atlas.clone()
                        } else {
                            None
                        };
                        self.commands.push(DrawCommand::GlyphAtlas(
                            GlyphAtlasBatch::new(
                                surviving,
                                transform,
                                alpha_atlas,
                                color_atlas,
                                BlendMode::SrcOver,
                                None,
                            )
                            .with_linear_filter(!is_unit_translation),
                        ));
                        if self.typographer.is_some() {
                            self.sync_glyph_atlases();
                        }
                        return;
                    }
                } else if top.clip_rects.len() == 1 {
                    let Some(effective_clip) = intersect_rects(top.clip_rects[0], canvas_bounds)
                    else {
                        self.replay_stats.culled_count += 1;
                        return;
                    };
                    if intersect_rects(dev_bounds, effective_clip).is_none() {
                        self.replay_stats.culled_count += 1;
                        return;
                    }
                    if rect_contains_rect(effective_clip, dev_bounds) {
                        let surviving = shift_all_quads(&paragraph.quads, offset);
                        let alpha_atlas = if paragraph.uses_alpha {
                            self.default_alpha_atlas.clone()
                        } else {
                            None
                        };
                        let color_atlas = if paragraph.uses_color {
                            self.default_color_atlas.clone()
                        } else {
                            None
                        };
                        self.commands.push(DrawCommand::GlyphAtlas(
                            GlyphAtlasBatch::new(
                                surviving,
                                transform,
                                alpha_atlas,
                                color_atlas,
                                BlendMode::SrcOver,
                                None,
                            )
                            .with_linear_filter(!is_unit_translation),
                        ));
                        if self.typographer.is_some() {
                            self.sync_glyph_atlases();
                        }
                        return;
                    }
                }
            }
        }

        let prev_len = self.commands.len();
        self.emit_glyph_quads_slice_with_offset(
            &paragraph.quads,
            offset,
            transform,
            BlendMode::SrcOver,
            !is_unit_translation,
        );
        if self.commands.len() > prev_len {
            if self.typographer.is_some() {
                self.sync_glyph_atlases();
            }
        } else {
            self.replay_stats.culled_count += 1;
        }
    }

    pub fn draw_paragraph(&mut self, paragraph: &Paragraph, offset: Point) {
        let bg_count = paragraph
            .background_rects
            .len()
            .saturating_sub(paragraph.decoration_rects.len());
        for &(rect, color) in paragraph.background_rects.iter().take(bg_count) {
            let shifted = Rect::from_xywh(
                rect.get_left() + offset.x,
                rect.get_top() + offset.y,
                rect.get_width(),
                rect.get_height(),
            );
            self.draw_rect(shifted, &Paint::fill(color));
        }

        self.emit_paragraph_quads(paragraph, offset);

        for &(rect, color) in &paragraph.decoration_rects {
            let shifted = Rect::from_xywh(
                rect.get_left() + offset.x,
                rect.get_top() + offset.y,
                rect.get_width(),
                rect.get_height(),
            );
            self.draw_rect(shifted, &Paint::fill(color));
        }
    }

    pub fn draw_image(
        &mut self,
        image: &ImageTexture,
        offset: Point,
        paint: &Paint,
        linear_filter: bool,
    ) {
        let src = Rect::from_xywh(0.0, 0.0, image.width as Scalar, image.height as Scalar);
        let dst = Rect::from_xywh(
            offset.x,
            offset.y,
            image.width as Scalar,
            image.height as Scalar,
        );
        self.draw_image_rect(image, src, dst, paint, linear_filter);
    }

    pub fn draw_image_rect(
        &mut self,
        image: &ImageTexture,
        src: Rect,
        dst: Rect,
        paint: &Paint,
        linear_filter: bool,
    ) {
        if src.is_empty() || dst.is_empty() || image.width == 0 || image.height == 0 {
            return;
        }
        let transform = self.get_current_transform();
        let clip_targets =
            self.resolve_clip_targets_filtered(Some(dst), &transform, paint.image_filter.as_ref());
        if clip_targets.is_empty() {
            self.replay_stats.culled_count += 1;
            return;
        }
        let alpha = paint.color.alpha.clamp(0.0, 1.0);
        let needs_filter_layer = paint.image_filter.is_some() || paint.color_filter.is_some();
        if needs_filter_layer {
            let dev_bounds = transform_rect(&transform, dst);
            if dev_bounds.is_empty() {
                return;
            }
            let layer_cf = paint
                .color_filter
                .as_ref()
                .map(ColorFilter::to_layer_color_filter);
            let layer_if = paint
                .image_filter
                .as_ref()
                .map(ImageFilter::to_layer_image_filter);
            for clip_rect in clip_targets {
                let inner_cmd = DrawCommand::TextureFill(TextureFillBatch::from_rect(
                    src,
                    dst,
                    transform,
                    image.clone(),
                    1.0,
                    linear_filter,
                    BlendMode::SrcOver,
                    None,
                ));
                self.commands.push(DrawCommand::SaveLayer {
                    bounds: dev_bounds,
                    commands: vec![inner_cmd],
                    alpha,
                    blend_mode: paint.blend_mode,
                    color_filter: layer_cf.clone(),
                    image_filter: layer_if.clone(),
                    clip_rect,
                });
            }
            return;
        }
        for clip_rect in clip_targets {
            self.commands
                .push(DrawCommand::TextureFill(TextureFillBatch::from_rect(
                    src,
                    dst,
                    transform,
                    image.clone(),
                    alpha,
                    linear_filter,
                    paint.blend_mode,
                    clip_rect,
                )));
        }
    }

    pub fn draw_image_nine(
        &mut self,
        image: &ImageTexture,
        center: Rect,
        dst: Rect,
        paint: &Paint,
        linear_filter: bool,
    ) {
        if dst.is_empty() || image.width == 0 || image.height == 0 {
            return;
        }
        let img_w = image.width as Scalar;
        let img_h = image.height as Scalar;
        let c_left = center.get_left().clamp(0.0, img_w);
        let c_top = center.get_top().clamp(0.0, img_h);
        let c_right = center.get_right().clamp(c_left, img_w);
        let c_bottom = center.get_bottom().clamp(c_top, img_h);

        let left_w = c_left;
        let right_w = img_w - c_right;
        let top_h = c_top;
        let bottom_h = img_h - c_bottom;

        let dst_w = dst.get_width();
        let dst_h = dst.get_height();
        let fixed_w = left_w + right_w;
        let (d_left_w, d_right_w) = if fixed_w > dst_w && fixed_w > 0.0 {
            let scale_x = dst_w / fixed_w;
            (left_w * scale_x, right_w * scale_x)
        } else {
            (left_w, right_w)
        };
        let fixed_h = top_h + bottom_h;
        let (d_top_h, d_bottom_h) = if fixed_h > dst_h && fixed_h > 0.0 {
            let scale_y = dst_h / fixed_h;
            (top_h * scale_y, bottom_h * scale_y)
        } else {
            (top_h, bottom_h)
        };

        let sx = [0.0, c_left, c_right, img_w];
        let sy = [0.0, c_top, c_bottom, img_h];
        let dx = [
            dst.get_left(),
            dst.get_left() + d_left_w,
            dst.get_right() - d_right_w,
            dst.get_right(),
        ];
        let dy = [
            dst.get_top(),
            dst.get_top() + d_top_h,
            dst.get_bottom() - d_bottom_h,
            dst.get_bottom(),
        ];

        let needs_filter_layer = paint.image_filter.is_some() || paint.color_filter.is_some();
        if needs_filter_layer {
            let transform = self.get_current_transform();
            let clip_targets = self.resolve_clip_targets_filtered(
                Some(dst),
                &transform,
                paint.image_filter.as_ref(),
            );
            if clip_targets.is_empty() {
                self.replay_stats.culled_count += 1;
                return;
            }
            let dev_bounds = transform_rect(&transform, dst);
            if dev_bounds.is_empty() {
                return;
            }
            let mut patches = Vec::with_capacity(9);
            for row in 0..3 {
                for col in 0..3 {
                    let src_patch = Rect::from_ltrb(sx[col], sy[row], sx[col + 1], sy[row + 1]);
                    let dst_patch = Rect::from_ltrb(dx[col], dy[row], dx[col + 1], dy[row + 1]);
                    if src_patch.get_width() > 0.0
                        && src_patch.get_height() > 0.0
                        && dst_patch.get_width() > 0.0
                        && dst_patch.get_height() > 0.0
                    {
                        patches.push(DrawCommand::TextureFill(TextureFillBatch::from_rect(
                            src_patch,
                            dst_patch,
                            transform,
                            image.clone(),
                            1.0,
                            linear_filter,
                            BlendMode::SrcOver,
                            None,
                        )));
                    }
                }
            }
            if patches.is_empty() {
                return;
            }
            let alpha = paint.color.alpha.clamp(0.0, 1.0);
            let layer_cf = paint
                .color_filter
                .as_ref()
                .map(ColorFilter::to_layer_color_filter);
            let layer_if = paint
                .image_filter
                .as_ref()
                .map(ImageFilter::to_layer_image_filter);
            for clip_rect in clip_targets {
                self.commands.push(DrawCommand::SaveLayer {
                    bounds: dev_bounds,
                    commands: patches.clone(),
                    alpha,
                    blend_mode: paint.blend_mode,
                    color_filter: layer_cf.clone(),
                    image_filter: layer_if.clone(),
                    clip_rect,
                });
            }
            return;
        }

        for row in 0..3 {
            for col in 0..3 {
                let src_patch = Rect::from_ltrb(sx[col], sy[row], sx[col + 1], sy[row + 1]);
                let dst_patch = Rect::from_ltrb(dx[col], dy[row], dx[col + 1], dy[row + 1]);
                if src_patch.get_width() > 0.0
                    && src_patch.get_height() > 0.0
                    && dst_patch.get_width() > 0.0
                    && dst_patch.get_height() > 0.0
                {
                    self.draw_image_rect(image, src_patch, dst_patch, paint, linear_filter);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_atlas(
        &mut self,
        atlas: &ImageTexture,
        rst_transforms: &[[Scalar; 4]],
        tex_rects: &[Rect],
        _colors: Option<&[Color]>,
        blend_mode: BlendMode,
        _cull_rect: Option<Rect>,
        paint: Option<&Paint>,
        linear_filter: bool,
    ) {
        let count = rst_transforms.len().min(tex_rects.len());
        if count == 0 || atlas.width == 0 || atlas.height == 0 {
            return;
        }
        let Some(top) = self.stack.last() else {
            return;
        };
        if top.clip_rects.is_empty() {
            return;
        }
        let transform = top.transform;
        let clip_targets: Vec<Option<Rect>> = if top.has_explicit_clip {
            top.clip_rects.iter().copied().map(Some).collect()
        } else {
            vec![None]
        };

        let inv_w = 1.0 / (atlas.width.max(1) as Scalar);
        let inv_h = 1.0 / (atlas.height.max(1) as Scalar);
        let mut vertices = Vec::with_capacity(count * 4);
        let mut indices = Vec::with_capacity(count * 6);

        for i in 0..count {
            let [scos, ssin, tx, ty] = rst_transforms[i];
            let tr = tex_rects[i];
            let w = tr.get_width();
            let h = tr.get_height();
            if w <= 0.0 || h <= 0.0 {
                continue;
            }
            let u0 = tr.get_left() * inv_w;
            let v0 = tr.get_top() * inv_h;
            let u1 = tr.get_right() * inv_w;
            let v1 = tr.get_bottom() * inv_h;

            let base = vertices.len() as u32;
            vertices.push(UvVertex::new(tx, ty, u0, v0));
            vertices.push(UvVertex::new(tx + scos * w, ty + ssin * w, u1, v0));
            vertices.push(UvVertex::new(
                tx + scos * w - ssin * h,
                ty + ssin * w + scos * h,
                u1,
                v1,
            ));
            vertices.push(UvVertex::new(tx - ssin * h, ty + scos * h, u0, v1));
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        if vertices.is_empty() {
            return;
        }
        let alpha = paint.map_or(1.0, |p| p.color.alpha.clamp(0.0, 1.0));
        let effective_blend = paint.map_or(blend_mode, |p| p.blend_mode);

        if let Some(p) = paint.filter(|p| p.image_filter.is_some() || p.color_filter.is_some()) {
            let Some(dev_bounds) = uv_vertices_device_bounds(&vertices, &transform) else {
                return;
            };
            let layer_cf = p
                .color_filter
                .as_ref()
                .map(ColorFilter::to_layer_color_filter);
            let layer_if = p
                .image_filter
                .as_ref()
                .map(ImageFilter::to_layer_image_filter);
            for clip_rect in clip_targets {
                let inner_cmd = DrawCommand::TextureFill(TextureFillBatch {
                    vertices: vertices.clone(),
                    indices: indices.clone(),
                    transform,
                    texture: atlas.clone(),
                    alpha: 1.0,
                    tile_mode_x: TileMode::Clamp,
                    tile_mode_y: TileMode::Clamp,
                    linear_filter,
                    blend_mode: BlendMode::SrcOver,
                    clip_rect: None,
                });
                self.commands.push(DrawCommand::SaveLayer {
                    bounds: dev_bounds,
                    commands: vec![inner_cmd],
                    alpha,
                    blend_mode: effective_blend,
                    color_filter: layer_cf.clone(),
                    image_filter: layer_if.clone(),
                    clip_rect,
                });
            }
            return;
        }

        for clip_rect in clip_targets {
            self.commands
                .push(DrawCommand::TextureFill(TextureFillBatch {
                    vertices: vertices.clone(),
                    indices: indices.clone(),
                    transform,
                    texture: atlas.clone(),
                    alpha,
                    tile_mode_x: TileMode::Clamp,
                    tile_mode_y: TileMode::Clamp,
                    linear_filter,
                    blend_mode: effective_blend,
                    clip_rect,
                }));
        }
    }

    pub fn draw_vertices(
        &mut self,
        positions: &[Point],
        tex_coords: Option<&[Point]>,
        colors: Option<&[Color]>,
        indices: &[u16],
        blend_mode: BlendMode,
        paint: &Paint,
    ) {
        if positions.len() < 3 {
            return;
        }
        let tri_indices: Vec<u32> = if indices.is_empty() {
            (0..positions.len() as u32).collect()
        } else {
            indices
                .iter()
                .copied()
                .map(u32::from)
                .filter(|&idx| (idx as usize) < positions.len())
                .collect()
        };
        if tri_indices.len() < 3 {
            return;
        }

        let transform = self.get_current_transform();
        if let Some(ColorSource::Image {
            texture,
            tile_mode_x,
            tile_mode_y,
            matrix,
            linear_filter,
        }) = paint.effective_color_source()
        {
            let Some(top) = self.stack.last() else {
                return;
            };
            if top.clip_rects.is_empty() {
                return;
            }
            let clip_targets: Vec<Option<Rect>> = if top.has_explicit_clip {
                top.clip_rects.iter().copied().map(Some).collect()
            } else {
                vec![None]
            };
            let inv_w = 1.0 / (texture.width.max(1) as Scalar);
            let inv_h = 1.0 / (texture.height.max(1) as Scalar);
            let uv_mat = invert_matrix(matrix).unwrap_or(Matrix::IDENTITY);
            let vertices: Vec<UvVertex> = positions
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let uv_pt = tex_coords
                        .and_then(|tcs| tcs.get(i).copied())
                        .unwrap_or_else(|| uv_mat.transform_point(p));
                    UvVertex::new(p.x, p.y, uv_pt.x * inv_w, uv_pt.y * inv_h)
                })
                .collect();
            let alpha = paint.color.alpha.clamp(0.0, 1.0);
            if paint.image_filter.is_some() || paint.color_filter.is_some() {
                let Some(dev_bounds) = uv_vertices_device_bounds(&vertices, &transform) else {
                    return;
                };
                let layer_cf = paint
                    .color_filter
                    .as_ref()
                    .map(ColorFilter::to_layer_color_filter);
                let layer_if = paint
                    .image_filter
                    .as_ref()
                    .map(ImageFilter::to_layer_image_filter);
                for clip_rect in clip_targets {
                    let inner_cmd = DrawCommand::TextureFill(TextureFillBatch {
                        vertices: vertices.clone(),
                        indices: tri_indices.clone(),
                        transform,
                        texture: texture.clone(),
                        alpha: 1.0,
                        tile_mode_x: *tile_mode_x,
                        tile_mode_y: *tile_mode_y,
                        linear_filter: *linear_filter,
                        blend_mode: BlendMode::SrcOver,
                        clip_rect: None,
                    });
                    self.commands.push(DrawCommand::SaveLayer {
                        bounds: dev_bounds,
                        commands: vec![inner_cmd],
                        alpha,
                        blend_mode: paint.blend_mode,
                        color_filter: layer_cf.clone(),
                        image_filter: layer_if.clone(),
                        clip_rect,
                    });
                }
                return;
            }
            for clip_rect in clip_targets {
                self.commands
                    .push(DrawCommand::TextureFill(TextureFillBatch {
                        vertices: vertices.clone(),
                        indices: tri_indices.clone(),
                        transform,
                        texture: texture.clone(),
                        alpha,
                        tile_mode_x: *tile_mode_x,
                        tile_mode_y: *tile_mode_y,
                        linear_filter: *linear_filter,
                        blend_mode: paint.blend_mode,
                        clip_rect,
                    }));
            }
            return;
        }

        if let Some(v_colors) = colors {
            if v_colors.len() >= positions.len() {
                let start_cmd_idx = self.commands.len();
                let has_imf = paint.image_filter.is_some();
                for &[t0, t1, t2] in tri_indices.as_chunks::<3>().0 {
                    let (i0, i1, i2) = (t0 as usize, t1 as usize, t2 as usize);
                    let avg = (v_colors[i0] + v_colors[i1] + v_colors[i2]) * (1.0 / 3.0);
                    let tri_color = if blend_mode == BlendMode::Dst {
                        avg
                    } else if blend_mode == BlendMode::Src {
                        paint.effective_solid_color()
                    } else {
                        avg.blend(paint.effective_solid_color(), blend_mode)
                    };
                    let mut tri_paint = paint.clone();
                    tri_paint.color = tri_color;
                    tri_paint.shader = None;
                    tri_paint.color_source = None;
                    if has_imf {
                        tri_paint.image_filter = None;
                        tri_paint.color_filter = None;
                        tri_paint.blend_mode = BlendMode::SrcOver;
                    }
                    let mesh = TessellatedMesh {
                        vertices: vec![
                            Vertex2D::from_point(positions[i0]),
                            Vertex2D::from_point(positions[i1]),
                            Vertex2D::from_point(positions[i2]),
                        ],
                        indices: vec![0, 1, 2],
                    };
                    self.emit_mesh_with_transform(mesh, transform, &tri_paint);
                }
                if let Some(imf) = &paint.image_filter {
                    let mut sub_commands: Vec<DrawCommand> =
                        self.commands.drain(start_cmd_idx..).collect();
                    for cmd in &mut sub_commands {
                        if let DrawCommand::SolidFill { clip_rect, .. } = cmd {
                            *clip_rect = None;
                        }
                    }
                    if let Some(bounds) = commands_device_bounds(&sub_commands) {
                        self.commands.push(DrawCommand::SaveLayer {
                            bounds,
                            commands: sub_commands,
                            alpha: 1.0,
                            blend_mode: paint.blend_mode,
                            color_filter: None,
                            image_filter: Some(imf.to_layer_image_filter()),
                            clip_rect: self.current_scissor_rect(),
                        });
                    }
                }
                return;
            }
        }

        let mesh = TessellatedMesh {
            vertices: positions
                .iter()
                .copied()
                .map(Vertex2D::from_point)
                .collect(),
            indices: tri_indices,
        };
        self.emit_mesh_with_transform(mesh, transform, paint);
    }

    pub fn draw_shadow(
        &mut self,
        path: &Path,
        color: Color,
        elevation: Scalar,
        transparent_occluder: bool,
        device_pixel_ratio: Scalar,
    ) {
        if elevation <= 0.0 || color.alpha <= 0.0 {
            return;
        }
        let Some(bounds) = path.get_bounding_box() else {
            return;
        };
        if bounds.is_empty() {
            return;
        }
        let dpr = device_pixel_ratio.max(1.0);
        let sigma = (0.5 * elevation * dpr).clamp(0.5, 64.0);
        let dy = (elevation * 0.5).clamp(0.5, 24.0);
        let shadow_rect = Rect::from_xywh(
            bounds.get_left(),
            bounds.get_top() + dy,
            bounds.get_width(),
            bounds.get_height(),
        );
        let corner_radius = (bounds.get_width().min(bounds.get_height()) * 0.1).min(12.0);
        let alpha_scale = if transparent_occluder { 0.5 } else { 0.35 };
        let shadow_color = color.with_alpha((color.alpha * alpha_scale).clamp(0.0, 1.0));
        self.draw_rrect_shadow(shadow_rect, corner_radius, sigma, shadow_color);
    }

    pub fn draw_picture(&mut self, picture: &DisplayList) {
        let base_count = self.get_save_count();
        self.save();
        self.draw_display_list(picture);
        self.restore_to_count(base_count);
    }

    /// Renders one of the 6 `flutter-wasm-compare` `PolymorphicCard` categories (`polymorphic_widgets.dart`)
    /// inside `rect` using CTM `translate`/`scale` and `clip_rect`:
    /// - `0`: `WAVE #<index>` sparkline + `MaterialIcon::ShowChart`
    /// - `1`: `CORE #<index>` `<gauge> ops` + `MaterialIcon::Speed`
    /// - `2`: `SYNC <index>` circular arc + linear progress bar
    /// - `3`: `Node <index>` amber pill chip + `MaterialIcon::Bolt` / `MaterialIcon::AutoAwesome`
    /// - `4`: `TH #<index>` status + `MaterialIcon::CheckCircle` / `MaterialIcon::RadioButtonUnchecked`
    /// - `5`: `WASM #<index>` teal card + `MaterialIcon::Memory`
    pub fn draw_polymorphic_card(&mut self, index: usize, animation_value: Scalar, rect: Rect) {
        if rect.is_empty() {
            return;
        }
        self.save();
        self.clip_rect(rect);
        self.translate(rect.get_left(), rect.get_top());
        self.scale(rect.get_width() / 140.0, rect.get_height() / 80.0);

        // Outer card box: 140x80 design space with 1px margin
        let card_rect = Rect::from_xywh(1.0, 1.0, 138.0, 78.0);
        let card_radii = RoundingRadii::from_single_radius(4.0);
        self.draw_rrect(
            card_rect,
            card_radii,
            &Paint::fill(Color::make_rgba8(0x13, 0x17, 0x22, 0xFF)),
        );
        self.draw_rrect(
            card_rect,
            card_radii,
            &Paint::stroke(Color::new(1.0, 1.0, 1.0, 0.14), 1.0),
        );

        match index % 6 {
            0 => {
                // _buildSparklineCard: WAVE #<index> + ShowChart icon + 8-point sine wave sparkline
                let orange = Color::make_rgba8(0xFF, 0xAB, 0x40, 0xFF);
                self.draw_text(
                    &format!("WAVE #{index}"),
                    TextStyle {
                        font: BuiltinFont::RobotoBold,
                        font_size: 11.0,
                        line_height: 13.0,
                        color: orange,
                        max_width: None,
                    },
                    Point::new(8.0, 8.0),
                );
                self.draw_icon(
                    MaterialIcon::ShowChart,
                    14.0,
                    orange,
                    Point::new(118.0, 7.0),
                );

                let x0 = 8.0;
                let y0 = 26.0;
                let w = 124.0;
                let h = 44.0;
                let mut pb = PathBuilder::new();
                let points_count = 8usize;
                for i in 0..points_count {
                    let x = x0 + (i as Scalar / (points_count - 1) as Scalar) * w;
                    let phase = animation_value * 2.0 * std::f32::consts::PI
                        + (i as Scalar) * 0.7
                        + (index as Scalar);
                    let y = (y0 + h * 0.5) + phase.sin() * (h * 0.35);
                    if i == 0 {
                        pb.move_to(Point::new(x, y));
                    } else {
                        pb.line_to(Point::new(x, y));
                    }
                }
                self.draw_path(
                    &pb.build(FillType::NonZero),
                    &Paint::stroke(orange, 2.0)
                        .with_stroke_cap(Cap::Round)
                        .with_stroke_join(Join::Round),
                );
            }
            1 => {
                // _buildGaugeCard: CORE #<index> + "<gauge> ops" + Speed icon
                let cyan = Color::make_rgba8(0x18, 0xFF, 0xFF, 0xFF);
                let gauge = if animation_value == 0.0 && index == 1 {
                    75
                } else {
                    (((animation_value * 3.0 * std::f32::consts::PI + (index as Scalar) * 1.2)
                        .sin()
                        + 1.0)
                        * 50.0)
                        .round() as i32
                };
                self.draw_text(
                    &format!("CORE #{index}"),
                    TextStyle {
                        font: BuiltinFont::RobotoBold,
                        font_size: 11.0,
                        line_height: 13.0,
                        color: cyan,
                        max_width: None,
                    },
                    Point::new(8.0, 20.0),
                );
                self.draw_text(
                    &format!("{gauge} ops"),
                    TextStyle {
                        font: BuiltinFont::RobotoMonoBold,
                        font_size: 13.0,
                        line_height: 16.0,
                        color: Color::WHITE,
                        max_width: None,
                    },
                    Point::new(8.0, 40.0),
                );
                self.draw_icon(MaterialIcon::Speed, 22.0, cyan, Point::new(106.0, 28.0));
            }
            2 => {
                // _buildProgressCard: CircularProgressIndicator + SYNC <index> + LinearProgressIndicator
                let purple = Color::make_rgba8(0xE0, 0x40, 0xFB, 0xFF);
                let track = Color::new(1.0, 1.0, 1.0, 0.14);
                let progress = (((animation_value * 2.0 * std::f32::consts::PI
                    + (index as Scalar) * 0.5)
                    .cos()
                    + 1.0)
                    * 0.5)
                    .clamp(0.15, 0.95);

                self.draw_circle(Point::new(22.0, 40.0), 12.0, &Paint::stroke(track, 3.2));
                self.draw_arc(
                    Rect::from_xywh(10.0, 28.0, 24.0, 24.0),
                    Radians(-std::f32::consts::FRAC_PI_2),
                    Radians(progress * 2.0 * std::f32::consts::PI),
                    false,
                    &Paint::stroke(purple, 3.2).with_stroke_cap(Cap::Round),
                );

                self.draw_text(
                    &format!("SYNC {index}"),
                    TextStyle {
                        font: BuiltinFont::RobotoBold,
                        font_size: 11.0,
                        line_height: 13.0,
                        color: purple,
                        max_width: None,
                    },
                    Point::new(42.0, 22.0),
                );
                let bar_radii = RoundingRadii::from_single_radius(2.5);
                self.draw_rrect(
                    Rect::from_xywh(42.0, 44.0, 86.0, 5.0),
                    bar_radii,
                    &Paint::fill(track),
                );
                self.draw_rrect(
                    Rect::from_xywh(42.0, 44.0, 86.0 * progress, 5.0),
                    bar_radii,
                    &Paint::fill(purple),
                );
            }
            3 => {
                // _buildChipCard: centered amber pill chip + Bolt/AutoAwesome icon + "Node <index>"
                let amber = Color::make_rgba8(0xFF, 0xD7, 0x40, 0xFF);
                let chip_rect = Rect::from_xywh(20.0, 24.0, 100.0, 32.0);
                let chip_radii = RoundingRadii::from_single_radius(16.0);
                self.draw_rrect(
                    chip_rect,
                    chip_radii,
                    &Paint::fill(Color::new(1.0, 0.76, 0.03, 0.18)),
                );
                self.draw_rrect(
                    chip_rect,
                    chip_radii,
                    &Paint::stroke(Color::new(1.0, 0.84, 0.25, 0.42), 1.0),
                );
                let icon = if index % 2 == 1 {
                    MaterialIcon::Bolt
                } else {
                    MaterialIcon::AutoAwesome
                };
                self.draw_icon(icon, 14.0, amber, Point::new(30.0, 32.0));
                self.draw_text(
                    &format!("Node {index}"),
                    TextStyle {
                        font: BuiltinFont::RobotoBold,
                        font_size: 11.0,
                        line_height: 13.0,
                        color: amber,
                        max_width: None,
                    },
                    Point::new(48.0, 33.0),
                );
            }
            4 => {
                // _buildStatusCard: "TH #<index>" + CheckCircle / RadioButtonUnchecked
                let active = animation_value >= 0.0
                    || (animation_value * 4.0 * std::f32::consts::PI + (index as Scalar)).sin()
                        > 0.0;
                let color = if active {
                    Color::make_rgba8(0x69, 0xF0, 0xAE, 0xFF)
                } else {
                    Color::new(1.0, 1.0, 1.0, 0.38)
                };
                let icon = if active {
                    MaterialIcon::CheckCircle
                } else {
                    MaterialIcon::RadioButtonUnchecked
                };
                self.draw_text(
                    &format!("TH #{index}"),
                    TextStyle {
                        font: BuiltinFont::RobotoBold,
                        font_size: 11.0,
                        line_height: 14.0,
                        color,
                        max_width: None,
                    },
                    Point::new(10.0, 32.0),
                );
                self.draw_icon(icon, 18.0, color, Point::new(108.0, 29.0));
            }
            _ => {
                // _buildWaveCard: centered teal container + Memory icon + "WASM #<index>"
                let teal = Color::make_rgba8(0x64, 0xFF, 0xDA, 0xFF);
                let box_rect = Rect::from_xywh(16.0, 24.0, 108.0, 32.0);
                let box_radii = RoundingRadii::from_single_radius(5.0);
                self.draw_rrect(
                    box_rect,
                    box_radii,
                    &Paint::fill(Color::new(0.39, 1.0, 0.85, 0.16)),
                );
                self.draw_rrect(
                    box_rect,
                    box_radii,
                    &Paint::stroke(Color::new(0.39, 1.0, 0.85, 0.45), 1.2),
                );
                self.draw_icon(MaterialIcon::Memory, 14.0, teal, Point::new(24.0, 32.0));
                self.draw_text(
                    &format!("WASM #{index}"),
                    TextStyle {
                        font: BuiltinFont::RobotoMonoBold,
                        font_size: 11.0,
                        line_height: 13.0,
                        color: teal,
                        max_width: None,
                    },
                    Point::new(42.0, 33.0),
                );
            }
        }

        self.restore();
    }

    pub fn glyph_batches(&self) -> &[PendingGlyphBatch] {
        &[]
    }

    pub fn draw_display_list(&mut self, display_list: &DisplayList) {
        display_list.dispatch(self);
    }

    pub fn commands(&self) -> &[DrawCommand] {
        &self.commands
    }

    pub fn render(&self, renderer: &WgpuRenderer) -> Result<Vec<u8>, String> {
        renderer.render_to_rgba(self.width, self.height, self.clear_color, &self.commands)
    }

    pub fn render_to_pixels(&self, renderer: &WgpuRenderer) -> Result<Vec<u8>, String> {
        self.render(renderer)
    }

    pub fn render_to_png(
        &self,
        renderer: &WgpuRenderer,
        path: impl AsRef<FsPath>,
    ) -> Result<Vec<u8>, String> {
        let rgba = self.render_to_pixels(renderer)?;
        save_rgba_png(path.as_ref(), self.width, self.height, &rgba)?;
        Ok(rgba)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct QuadrantStats {
    pub non_background_pixels: usize,
    pub total_pixels: usize,
    pub mean_r: f64,
    pub mean_g: f64,
    pub mean_b: f64,
    pub distinct_color_buckets: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShowcaseStats {
    pub width: u32,
    pub height: u32,
    pub quadrants: [QuadrantStats; 4],
}

/// Builds the 800x600 4-quadrant Impeller GPU diagnostic showcase scene:
/// - Quadrant 1 (Top-Left): GPU HUD title + Analytical Gaussian `RRectBlur` drop-shadow cards +
///   rounded rectangles with nested per-corner `RoundingRadii` and linear gradient fills (`TileMode::Clamp` & `TileMode::Mirror`).
/// - Quadrant 2 (Top-Right): GPU HUD title + Complex cubic/quadratic Bezier paths (subdivided via Wang's formula) and
///   self-intersecting star/ring polygons demonstrating `FillType::NonZero` vs `FillType::Odd` with radial gradients.
/// - Quadrant 3 (Bottom-Left): GPU HUD title + Stroke gallery showing `Cap::Butt`, `Cap::Round`, `Cap::Square` and
///   `Join::Miter` (with miter limit), `Join::Round`, `Join::Bevel`, and `kurbo::dash` dashed paths.
/// - Quadrant 4 (Bottom-Right): GPU HUD title + 2x3 matrix of the 6 `flutter-wasm-compare` `PolymorphicCard`s
///   (live `Roboto` text, `MaterialIcons` glyphs, sparklines, gauges, progress arcs, and chips) +
///   Porter-Duff additive RGB blend circles + stroked vector glyph outline + `ClipOperation::Difference` hierarchy.
pub fn build_showcase_canvas() -> Canvas {
    let bg = Color::new(0.06, 0.07, 0.10, 1.0);
    let mut canvas = Canvas::new(800, 600, bg);
    draw_showcase_scene(&mut canvas);
    canvas
}

/// Draws the 800x600 4-quadrant Impeller GPU diagnostic showcase scene onto `canvas`.
pub fn draw_showcase_scene(canvas: &mut Canvas) {
    let panel_bg = Paint::fill(Color::new(0.10, 0.12, 0.17, 1.0));
    let panel_radii = RoundingRadii::from_single_radius(14.0);

    // =========================================================================
    // Quadrant 1 (Top-Left: 0..400, 0..300)
    // Analytical Gaussian RRectBlur shadows + per-corner RoundingRadii + LinearGradients + HUD
    // =========================================================================
    canvas.save();
    canvas.clip_rect(Rect::from_xywh(0.0, 0.0, 400.0, 300.0));
    canvas.draw_rrect(
        Rect::from_xywh(12.0, 12.0, 376.0, 276.0),
        panel_radii,
        &panel_bg,
    );

    // Card 1: Analytical Gaussian drop-shadow + Asymmetric per-corner RRect + Clamp LinearGradient
    canvas.draw_rrect_shadow(
        Rect::from_xywh(34.0, 48.0, 152.0, 96.0),
        20.0,
        14.0,
        Color::new(0.0, 0.82, 1.0, 0.85),
    );
    let card1_radii = RoundingRadii::from_radii(
        Size::new(28.0, 28.0),
        Size::new(6.0, 6.0),
        Size::new(6.0, 6.0),
        Size::new(28.0, 28.0),
    );
    let card1_grad = LinearGradientParams {
        start_point: Point::new(30.0, 42.0),
        end_point: Point::new(182.0, 138.0),
        stops: vec![
            GradientStop {
                offset: 0.0,
                color: Color::new(0.05, 0.75, 0.98, 1.0),
            },
            GradientStop {
                offset: 0.5,
                color: Color::new(0.25, 0.38, 0.95, 1.0),
            },
            GradientStop {
                offset: 1.0,
                color: Color::new(0.58, 0.18, 0.92, 1.0),
            },
        ],
        tile_mode: TileMode::Clamp,
        alpha: 1.0,
    };
    canvas.draw_rrect(
        Rect::from_xywh(30.0, 42.0, 152.0, 96.0),
        card1_radii,
        &Paint::fill(Color::WHITE).with_color_source(ColorSource::LinearGradient(card1_grad)),
    );
    canvas.draw_text(
        "CLAMP",
        TextStyle {
            font: BuiltinFont::RobotoMonoBold,
            font_size: 11.0,
            line_height: 14.0,
            color: Color::new(1.0, 1.0, 1.0, 0.92),
            max_width: None,
        },
        Point::new(82.0, 82.0),
    );

    // Card 2: Analytical Gaussian drop-shadow + Per-corner RRect + Mirror LinearGradient
    canvas.draw_rrect_shadow(
        Rect::from_xywh(218.0, 48.0, 152.0, 96.0),
        22.0,
        16.0,
        Color::new(1.0, 0.28, 0.55, 0.85),
    );
    let card2_radii = RoundingRadii::from_radii(
        Size::new(8.0, 8.0),
        Size::new(30.0, 30.0),
        Size::new(30.0, 30.0),
        Size::new(8.0, 8.0),
    );
    let card2_grad = LinearGradientParams {
        start_point: Point::new(214.0, 42.0),
        end_point: Point::new(264.0, 92.0),
        stops: vec![
            GradientStop {
                offset: 0.0,
                color: Color::new(1.0, 0.32, 0.48, 1.0),
            },
            GradientStop {
                offset: 0.5,
                color: Color::new(1.0, 0.76, 0.18, 1.0),
            },
            GradientStop {
                offset: 1.0,
                color: Color::new(0.95, 0.15, 0.65, 1.0),
            },
        ],
        tile_mode: TileMode::Mirror,
        alpha: 1.0,
    };
    canvas.draw_rrect(
        Rect::from_xywh(214.0, 42.0, 152.0, 96.0),
        card2_radii,
        &Paint::fill(Color::WHITE).with_color_source(ColorSource::LinearGradient(card2_grad)),
    );
    canvas.draw_text(
        "MIRROR",
        TextStyle {
            font: BuiltinFont::RobotoMonoBold,
            font_size: 11.0,
            line_height: 14.0,
            color: Color::new(0.12, 0.08, 0.18, 0.95),
            max_width: None,
        },
        Point::new(262.0, 82.0),
    );
    canvas.draw_text(
        "Q1 \u{00B7} RRECT BLUR SHADOWS & GRADIENTS",
        TextStyle {
            font: BuiltinFont::RobotoBold,
            font_size: 12.5,
            line_height: 15.0,
            color: Color::new(0.84, 0.92, 1.0, 0.96),
            max_width: None,
        },
        Point::new(26.0, 18.0),
    );

    // Card 3 (Bottom of Q1): Elevated wide banner with Paint::mask_blur_sigma + nested diff_rrect
    let shadow_paint = Paint::fill(Color::new(0.20, 0.95, 0.65, 0.75)).with_mask_blur_sigma(12.0);
    canvas.draw_rrect(
        Rect::from_xywh(36.0, 172.0, 328.0, 92.0),
        RoundingRadii::from_single_radius(20.0),
        &shadow_paint,
    );
    let banner_grad = LinearGradientParams {
        start_point: Point::new(32.0, 166.0),
        end_point: Point::new(140.0, 258.0),
        stops: vec![
            GradientStop {
                offset: 0.0,
                color: Color::new(0.08, 0.82, 0.58, 1.0),
            },
            GradientStop {
                offset: 1.0,
                color: Color::new(0.05, 0.45, 0.72, 1.0),
            },
        ],
        tile_mode: TileMode::Mirror,
        alpha: 1.0,
    };
    canvas.draw_rrect(
        Rect::from_xywh(32.0, 166.0, 328.0, 92.0),
        RoundingRadii::from_radii(
            Size::new(24.0, 16.0),
            Size::new(12.0, 24.0),
            Size::new(24.0, 12.0),
            Size::new(16.0, 24.0),
        ),
        &Paint::fill(Color::WHITE).with_color_source(ColorSource::LinearGradient(banner_grad)),
    );
    canvas.draw_diff_rrect(
        Rect::from_xywh(52.0, 182.0, 288.0, 60.0),
        RoundingRadii::from_single_radius(14.0),
        Rect::from_xywh(64.0, 192.0, 264.0, 40.0),
        RoundingRadii::from_single_radius(8.0),
        &Paint::fill(Color::new(1.0, 1.0, 1.0, 0.85)),
    );
    canvas.restore();

    // =========================================================================
    // Quadrant 2 (Top-Right: 400..800, 0..300)
    // Cubic/Quadratic Bezier curves (Wang's formula) + NonZero vs Odd Star/Ring polygons + HUD
    // =========================================================================
    canvas.save();
    canvas.clip_rect(Rect::from_xywh(400.0, 0.0, 400.0, 300.0));
    canvas.draw_rrect(
        Rect::from_xywh(412.0, 12.0, 376.0, 276.0),
        panel_radii,
        &panel_bg,
    );
    canvas.draw_text(
        "Q2 \u{00B7} WANG'S BEZIERS & PSLG WINDING",
        TextStyle {
            font: BuiltinFont::RobotoBold,
            font_size: 12.5,
            line_height: 15.0,
            color: Color::new(1.0, 0.90, 0.72, 0.96),
            max_width: None,
        },
        Point::new(426.0, 18.0),
    );

    // Top wave ribbon: Cubic + Quadratic Beziers with RadialGradient
    let mut wave_pb = PathBuilder::new();
    wave_pb
        .move_to(Point::new(430.0, 74.0))
        .cubic_to(
            Point::new(495.0, 22.0),
            Point::new(555.0, 126.0),
            Point::new(610.0, 70.0),
        )
        .cubic_to(
            Point::new(665.0, 18.0),
            Point::new(725.0, 122.0),
            Point::new(770.0, 64.0),
        )
        .line_to(Point::new(770.0, 106.0))
        .quad_to(Point::new(690.0, 138.0), Point::new(610.0, 100.0))
        .quad_to(Point::new(520.0, 54.0), Point::new(430.0, 108.0))
        .close();
    let wave_grad = RadialGradientParams {
        center: Point::new(600.0, 74.0),
        radius: 175.0,
        stops: vec![
            GradientStop {
                offset: 0.0,
                color: Color::new(1.0, 0.92, 0.25, 1.0),
            },
            GradientStop {
                offset: 0.55,
                color: Color::new(1.0, 0.45, 0.15, 1.0),
            },
            GradientStop {
                offset: 1.0,
                color: Color::new(0.75, 0.10, 0.45, 1.0),
            },
        ],
        tile_mode: TileMode::Clamp,
        alpha: 1.0,
    };
    canvas.draw_path(
        &wave_pb.build(FillType::NonZero),
        &Paint::fill(Color::WHITE).with_color_source(ColorSource::RadialGradient(wave_grad)),
    );

    // Helper closure to build a self-intersecting 5-pointed pentagram + outer ring contour
    let build_star_ring = |cx: Scalar, cy: Scalar, outer_r: Scalar, fill_type: FillType| -> Path {
        let mut pb = PathBuilder::new();
        for i in 0..5 {
            let idx = (i * 2) % 5;
            let angle =
                -std::f32::consts::FRAC_PI_2 + (idx as Scalar) * 2.0 * std::f32::consts::PI / 5.0;
            let pt = Point::new(cx + outer_r * angle.cos(), cy + outer_r * angle.sin());
            if i == 0 {
                pb.move_to(pt);
            } else {
                pb.line_to(pt);
            }
        }
        pb.close();
        pb.add_circle(Point::new(cx, cy), outer_r * 0.32);
        pb.build(fill_type)
    };

    // Left Star: FillType::NonZero with warm gold->crimson radial gradient
    let star_nonzero = build_star_ring(508.0, 198.0, 62.0, FillType::NonZero);
    let nonzero_grad = RadialGradientParams {
        center: Point::new(508.0, 198.0),
        radius: 64.0,
        stops: vec![
            GradientStop {
                offset: 0.0,
                color: Color::new(1.0, 0.95, 0.35, 1.0),
            },
            GradientStop {
                offset: 0.6,
                color: Color::new(0.98, 0.38, 0.12, 1.0),
            },
            GradientStop {
                offset: 1.0,
                color: Color::new(0.65, 0.08, 0.32, 1.0),
            },
        ],
        tile_mode: TileMode::Clamp,
        alpha: 1.0,
    };
    canvas.draw_path(
        &star_nonzero,
        &Paint::fill(Color::WHITE).with_color_source(ColorSource::RadialGradient(nonzero_grad)),
    );
    canvas.draw_text(
        "NON-ZERO",
        TextStyle {
            font: BuiltinFont::RobotoMonoBold,
            font_size: 10.5,
            line_height: 13.0,
            color: Color::new(1.0, 0.85, 0.55, 0.92),
            max_width: None,
        },
        Point::new(476.0, 264.0),
    );

    // Right Star: FillType::Odd (Even-Odd) with cool cyan->emerald->indigo radial gradient
    let star_odd = build_star_ring(688.0, 198.0, 62.0, FillType::Odd);
    let odd_grad = RadialGradientParams {
        center: Point::new(688.0, 198.0),
        radius: 40.0,
        stops: vec![
            GradientStop {
                offset: 0.0,
                color: Color::new(0.20, 1.0, 0.78, 1.0),
            },
            GradientStop {
                offset: 0.6,
                color: Color::new(0.10, 0.62, 0.98, 1.0),
            },
            GradientStop {
                offset: 1.0,
                color: Color::new(0.42, 0.20, 0.92, 1.0),
            },
        ],
        tile_mode: TileMode::Mirror,
        alpha: 1.0,
    };
    canvas.draw_path(
        &star_odd,
        &Paint::fill(Color::WHITE).with_color_source(ColorSource::RadialGradient(odd_grad)),
    );
    canvas.draw_text(
        "EVEN-ODD",
        TextStyle {
            font: BuiltinFont::RobotoMonoBold,
            font_size: 10.5,
            line_height: 13.0,
            color: Color::new(0.55, 0.92, 1.0, 0.92),
            max_width: None,
        },
        Point::new(656.0, 264.0),
    );
    canvas.restore();

    // =========================================================================
    // Quadrant 3 (Bottom-Left: 0..400, 300..600)
    // Stroke gallery: Cap::Butt, Cap::Round, Cap::Square + Join::Miter, Join::Round, Join::Bevel + kurbo::dash
    // =========================================================================
    canvas.save();
    canvas.clip_rect(Rect::from_xywh(0.0, 300.0, 400.0, 300.0));
    canvas.draw_rrect(
        Rect::from_xywh(12.0, 312.0, 376.0, 276.0),
        panel_radii,
        &panel_bg,
    );
    canvas.draw_text(
        "Q3 \u{00B7} STROKE CAPS, JOINS & KURBO DASH",
        TextStyle {
            font: BuiltinFont::RobotoBold,
            font_size: 12.5,
            line_height: 15.0,
            color: Color::new(0.74, 0.96, 0.88, 0.96),
            max_width: None,
        },
        Point::new(26.0, 318.0),
    );

    // Cap showcase rows (Butt, Round, Square) at x = 52..168
    let caps = [
        (Cap::Butt, 356.0, Color::new(1.0, 0.38, 0.36, 1.0)),
        (Cap::Round, 390.0, Color::new(0.28, 0.92, 0.56, 1.0)),
        (Cap::Square, 424.0, Color::new(0.32, 0.68, 1.0, 1.0)),
    ];
    for (cap, y, color) in caps {
        let stroke_paint = Paint::stroke(color, 16.0).with_stroke_cap(cap);
        canvas.draw_line(Point::new(52.0, y), Point::new(168.0, y), &stroke_paint);
    }
    let rail_paint = Paint::stroke(Color::new(1.0, 1.0, 1.0, 0.45), 2.0);
    canvas.draw_line(
        Point::new(52.0, 342.0),
        Point::new(52.0, 438.0),
        &rail_paint,
    );
    canvas.draw_line(
        Point::new(168.0, 342.0),
        Point::new(168.0, 438.0),
        &rail_paint,
    );

    // Join showcase on the right side of Q3: Miter (sharp), Round, Bevel
    let joins = [
        (Join::Miter, 4.0, 215.0, Color::new(1.0, 0.78, 0.22, 1.0)),
        (Join::Round, 4.0, 275.0, Color::new(0.75, 0.42, 1.0, 1.0)),
        (Join::Bevel, 4.0, 335.0, Color::new(0.20, 0.90, 0.88, 1.0)),
    ];
    for (join, miter_limit, x_base, color) in joins {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(x_base - 18.0, 430.0))
            .line_to(Point::new(x_base + 8.0, 355.0))
            .line_to(Point::new(x_base + 34.0, 430.0));
        let paint = Paint::stroke(color, 13.0)
            .with_stroke_join(join)
            .with_stroke_miter(miter_limit)
            .with_stroke_cap(Cap::Butt);
        canvas.draw_path(&pb.build(FillType::NonZero), &paint);
    }

    // Multi-angle & multi-width stroke zig-zag (acute miter-limit bevel fallback vs wide miter)
    let mut zigzag_pb = PathBuilder::new();
    zigzag_pb
        .move_to(Point::new(36.0, 522.0))
        .line_to(Point::new(88.0, 466.0))
        .line_to(Point::new(136.0, 528.0))
        .line_to(Point::new(164.0, 464.0))
        .line_to(Point::new(192.0, 530.0))
        .line_to(Point::new(260.0, 470.0))
        .line_to(Point::new(358.0, 520.0));
    canvas.draw_path(
        &zigzag_pb.build(FillType::NonZero),
        &Paint::stroke(Color::new(0.98, 0.52, 0.20, 1.0), 11.0)
            .with_stroke_join(Join::Miter)
            .with_stroke_miter(2.2)
            .with_stroke_cap(Cap::Round),
    );

    // Curved dashed path along bottom of Q3 using kurbo::dash
    let mut dash_curve_pb = PathBuilder::new();
    dash_curve_pb.move_to(Point::new(36.0, 562.0)).cubic_to(
        Point::new(120.0, 544.0),
        Point::new(280.0, 578.0),
        Point::new(364.0, 558.0),
    );
    canvas.draw_dashed_path(
        &dash_curve_pb.build(FillType::NonZero),
        0.0,
        &[16.0, 10.0, 6.0, 10.0],
        &Paint::stroke(Color::new(0.45, 0.88, 1.0, 1.0), 5.5).with_stroke_cap(Cap::Round),
    );
    canvas.restore();

    // =========================================================================
    // Quadrant 4 (Bottom-Right: 400..800, 300..600)
    // 2x3 flutter-wasm-compare PolymorphicCard matrix + Additive RGB blends + Stroked Glyph Path
    // =========================================================================
    canvas.save();
    canvas.clip_rect(Rect::from_xywh(400.0, 300.0, 400.0, 300.0));
    canvas.draw_rrect(
        Rect::from_xywh(412.0, 312.0, 376.0, 276.0),
        panel_radii,
        &panel_bg,
    );
    canvas.draw_text(
        "Q4 \u{00B7} POLYMORPHIC CARDS, BLENDS & GLYPH PATH",
        TextStyle {
            font: BuiltinFont::RobotoBold,
            font_size: 12.0,
            line_height: 15.0,
            color: Color::new(0.92, 0.84, 1.0, 0.96),
            max_width: None,
        },
        Point::new(424.0, 318.0),
    );

    // Left/Center of Q4: 2x3 matrix of the 6 flutter-wasm-compare PolymorphicCards
    let card_w = 118.0;
    let card_h = 72.0;
    let gap_x = 6.0;
    let gap_y = 6.0;
    for idx in 0..6usize {
        let col = (idx % 2) as Scalar;
        let row = (idx / 2) as Scalar;
        let x = 420.0 + col * (card_w + gap_x);
        let y = 342.0 + row * (card_h + gap_y);
        let anim = if idx == 1 { 0.0 } else { 0.18 };
        canvas.draw_polymorphic_card(idx, anim, Rect::from_xywh(x, y, card_w, card_h));
    }

    // Top-Right of Q4: Additive (BlendMode::Plus) RGB overlapping color circles
    canvas.draw_circle(
        Point::new(714.0, 376.0),
        24.0,
        &Paint::fill(Color::new(0.95, 0.15, 0.20, 0.90)).with_blend_mode(BlendMode::SrcOver),
    );
    canvas.draw_circle(
        Point::new(742.0, 376.0),
        24.0,
        &Paint::fill(Color::new(0.15, 0.95, 0.25, 0.90)).with_blend_mode(BlendMode::Plus),
    );
    canvas.draw_circle(
        Point::new(728.0, 400.0),
        24.0,
        &Paint::fill(Color::new(0.20, 0.35, 0.98, 0.90)).with_blend_mode(BlendMode::Plus),
    );

    // Middle-Right of Q4: Stroked & filled vector glyph outline ('R' extracted via glyph_to_path)
    canvas.draw_glyph_path(
        'R',
        BuiltinFont::RobotoBold,
        52.0,
        Point::new(708.0, 488.0),
        &Paint::fill(Color::new(0.15, 0.85, 1.0, 0.28)),
    );
    canvas.draw_glyph_path(
        'R',
        BuiltinFont::RobotoBold,
        52.0,
        Point::new(708.0, 488.0),
        &Paint::stroke(Color::new(1.0, 0.82, 0.25, 1.0), 2.2)
            .with_stroke_join(Join::Round)
            .with_stroke_cap(Cap::Round),
    );

    // Bottom-Right of Q4: Nested save / translate / rotate / scale / ClipOperation::Difference rosette
    canvas.save();
    canvas.clip_rect(Rect::from_xywh(676.0, 496.0, 104.0, 84.0));
    canvas.clip_rect_with_op(
        Rect::from_xywh(720.0, 530.0, 16.0, 16.0),
        ClipOperation::Difference,
    );
    canvas.translate(728.0, 538.0);
    let blade_colors = [
        Color::new(1.0, 0.35, 0.55, 0.88),
        Color::new(1.0, 0.72, 0.20, 0.88),
        Color::new(0.25, 0.95, 0.62, 0.88),
        Color::new(0.22, 0.72, 1.0, 0.88),
        Color::new(0.68, 0.38, 1.0, 0.88),
        Color::new(1.0, 0.48, 0.28, 0.88),
    ];
    for (i, col) in blade_colors.iter().enumerate() {
        canvas.save();
        let angle = Radians((i as Scalar) * std::f32::consts::PI / 3.0);
        canvas.rotate(angle);
        let s = 1.0 - (i as Scalar) * 0.05;
        canvas.scale(s, s);
        canvas.draw_rrect(
            Rect::from_xywh(-7.0, -36.0, 14.0, 32.0),
            RoundingRadii::from_single_radius(6.0),
            &Paint::fill(*col).with_blend_mode(BlendMode::SrcOver),
        );
        canvas.restore();
    }
    canvas.restore();

    canvas.restore();
}

const QUARTER_CIRCLE_5: [(Scalar, Scalar); 5] = [
    (1.0, 0.0),
    (0.9238795, 0.38268343),
    (0.70710677, 0.70710677),
    (0.38268343, 0.9238795),
    (0.0, 1.0),
];

fn append_filled_rect_mesh(mesh: &mut TessellatedMesh, rect: Rect) {
    if rect.is_empty() {
        return;
    }
    let l = rect.get_left();
    let t = rect.get_top();
    let r = rect.get_right();
    let b = rect.get_bottom();
    let base = mesh.vertices.len() as u32;
    mesh.vertices.extend_from_slice(&[
        Vertex2D::new(l, t),
        Vertex2D::new(r, t),
        Vertex2D::new(r, b),
        Vertex2D::new(l, b),
    ]);
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn build_rrect_perimeter(rect: Rect, rx: Scalar, ry: Scalar) -> [Vertex2D; 20] {
    let l = rect.get_left();
    let t = rect.get_top();
    let r = rect.get_right();
    let b = rect.get_bottom();
    let rx = rx.clamp(0.0, rect.get_width() * 0.5);
    let ry = ry.clamp(0.0, rect.get_height() * 0.5);

    let mut pts = [Vertex2D::new(0.0, 0.0); 20];
    // Bottom-right (0..5): center (r - rx, b - ry), +x * cos, +y * sin
    for (i, &(c, s)) in QUARTER_CIRCLE_5.iter().enumerate() {
        pts[i] = Vertex2D::new((r - rx) + rx * c, (b - ry) + ry * s);
    }
    // Bottom-left (5..10): center (l + rx, b - ry), -x * sin, +y * cos
    for (i, &(c, s)) in QUARTER_CIRCLE_5.iter().enumerate() {
        pts[5 + i] = Vertex2D::new((l + rx) - rx * s, (b - ry) + ry * c);
    }
    // Top-left (10..15): center (l + rx, t + ry), -x * cos, -y * sin
    for (i, &(c, s)) in QUARTER_CIRCLE_5.iter().enumerate() {
        pts[10 + i] = Vertex2D::new((l + rx) - rx * c, (t + ry) - ry * s);
    }
    // Top-right (15..20): center (r - rx, t + ry), +x * sin, -y * cos
    for (i, &(c, s)) in QUARTER_CIRCLE_5.iter().enumerate() {
        pts[15 + i] = Vertex2D::new((r - rx) + rx * s, (t + ry) - ry * c);
    }
    pts
}

fn append_filled_rrect_mesh(mesh: &mut TessellatedMesh, rect: Rect, rx: Scalar, ry: Scalar) {
    if rect.is_empty() {
        return;
    }
    if rx <= 0.5 && ry <= 0.5 {
        append_filled_rect_mesh(mesh, rect);
        return;
    }
    let pts = build_rrect_perimeter(rect, rx, ry);
    let base = mesh.vertices.len() as u32;
    mesh.vertices.extend_from_slice(&pts);
    for i in 1..19u32 {
        mesh.indices
            .extend_from_slice(&[base, base + i, base + i + 1]);
    }
}

fn append_stroked_rrect_mesh(
    mesh: &mut TessellatedMesh,
    rect: Rect,
    rx: Scalar,
    ry: Scalar,
    stroke_width: Scalar,
) {
    if rect.is_empty() || stroke_width <= 0.0 {
        return;
    }
    let half = stroke_width * 0.5;
    let outer_rect = Rect::from_ltrb(
        rect.get_left() - half,
        rect.get_top() - half,
        rect.get_right() + half,
        rect.get_bottom() + half,
    );
    let inner_w = (rect.get_width() - stroke_width).max(0.0);
    let inner_h = (rect.get_height() - stroke_width).max(0.0);
    if inner_w <= 0.0 || inner_h <= 0.0 {
        append_filled_rrect_mesh(mesh, outer_rect, rx + half, ry + half);
        return;
    }
    let inner_rect = Rect::from_xywh(
        rect.get_left() + half,
        rect.get_top() + half,
        inner_w,
        inner_h,
    );
    let outer_pts = build_rrect_perimeter(outer_rect, rx + half, ry + half);
    let inner_pts = build_rrect_perimeter(inner_rect, (rx - half).max(0.0), (ry - half).max(0.0));
    let base = mesh.vertices.len() as u32;
    for i in 0..20 {
        mesh.vertices.push(outer_pts[i]);
        mesh.vertices.push(inner_pts[i]);
    }
    for i in 0..20u32 {
        let next = (i + 1) % 20;
        let o0 = base + i * 2;
        let i0 = o0 + 1;
        let o1 = base + next * 2;
        let i1 = o1 + 1;
        mesh.indices.extend_from_slice(&[o0, o1, i1, o0, i1, i0]);
    }
}

fn append_filled_circle_mesh(mesh: &mut TessellatedMesh, center: Point, radius: Scalar) {
    if radius <= 0.0 {
        return;
    }
    let base = mesh.vertices.len() as u32;
    const N: usize = 16;
    for i in 0..N {
        let angle = (i as Scalar) * (2.0 * std::f32::consts::PI / (N as Scalar));
        let (s, c) = angle.sin_cos();
        mesh.vertices
            .push(Vertex2D::new(center.x + radius * c, center.y + radius * s));
    }
    for i in 1..(N as u32 - 1) {
        mesh.indices
            .extend_from_slice(&[base, base + i, base + i + 1]);
    }
}

fn append_stroked_arc_mesh(
    mesh: &mut TessellatedMesh,
    center: Point,
    radii: Size,
    start_angle: Scalar,
    sweep_angle: Scalar,
    stroke_w: Scalar,
    segments: usize,
) {
    let rx = radii.width;
    let ry = radii.height;
    if rx <= 0.0 || ry <= 0.0 || stroke_w <= 0.0 || sweep_angle.abs() <= 1e-4 {
        return;
    }
    let segs = segments.max(4);
    let half = stroke_w * 0.5;
    let rx_out = rx + half;
    let ry_out = ry + half;
    let rx_in = (rx - half).max(0.0);
    let ry_in = (ry - half).max(0.0);
    let base = mesh.vertices.len() as u32;
    for i in 0..=segs {
        let t = (i as Scalar) / (segs as Scalar);
        let a = start_angle + sweep_angle * t;
        let (s, c) = a.sin_cos();
        mesh.vertices
            .push(Vertex2D::new(center.x + rx_out * c, center.y + ry_out * s));
        mesh.vertices
            .push(Vertex2D::new(center.x + rx_in * c, center.y + ry_in * s));
    }
    for i in 0..(segs as u32) {
        let o0 = base + i * 2;
        let i0 = o0 + 1;
        let o1 = base + (i + 1) * 2;
        let i1 = o1 + 1;
        mesh.indices.extend_from_slice(&[o0, o1, i1, o0, i1, i0]);
    }
}

fn append_stroked_polyline_mesh(mesh: &mut TessellatedMesh, points: &[Point], stroke_w: Scalar) {
    if points.len() < 2 || stroke_w <= 0.0 {
        return;
    }
    let half = stroke_w * 0.5;
    for pair in points.windows(2) {
        let p0 = pair[0];
        let p1 = pair[1];
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len <= 1e-4 {
            continue;
        }
        let nx = -dy / len * half;
        let ny = dx / len * half;
        let base = mesh.vertices.len() as u32;
        mesh.vertices.extend_from_slice(&[
            Vertex2D::new(p0.x + nx, p0.y + ny),
            Vertex2D::new(p1.x + nx, p1.y + ny),
            Vertex2D::new(p1.x - nx, p1.y - ny),
            Vertex2D::new(p0.x - nx, p0.y - ny),
        ]);
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ShapedLabelKey {
    text: String,
    font: BuiltinFont,
    font_size_milli: u32,
    line_height_milli: u32,
    max_width_milli: Option<u32>,
}

struct FrameLabelShaper<'a> {
    typographer: &'a mut TypographerContext,
    base_tx: Matrix,
    cache: std::collections::HashMap<ShapedLabelKey, Vec<AtlasGlyphQuad>>,
    quads: Vec<AtlasGlyphQuad>,
}

impl<'a> FrameLabelShaper<'a> {
    fn new(typographer: &'a mut TypographerContext, base_tx: Matrix, capacity: usize) -> Self {
        Self {
            typographer,
            base_tx,
            cache: std::collections::HashMap::with_capacity(32),
            quads: Vec::with_capacity(capacity),
        }
    }

    fn push_text(
        &mut self,
        text: &str,
        style: TextStyle,
        logical_origin: Point,
        clip_bounds: Option<Rect>,
    ) {
        let dev_pt = self.base_tx.transform_point(logical_origin);
        let dev_origin_x = dev_pt.x.round();
        let dev_origin_y = dev_pt.y.round();
        let scale_y = self.base_tx.e(1, 1).abs().clamp(0.25, 8.0);
        let scale_x = self.base_tx.e(0, 0).abs().clamp(0.25, 8.0);

        let scaled_font_size = (style.font_size * scale_y).round().max(1.0);
        let scaled_line_height = (style.line_height * scale_y).round().max(1.0);
        let scaled_max_w = style.max_width.map(|w| (w * scale_x).round().max(1.0));
        let key = ShapedLabelKey {
            text: text.to_string(),
            font: style.font,
            font_size_milli: (scaled_font_size * 1000.0) as u32,
            line_height_milli: (scaled_line_height * 1000.0) as u32,
            max_width_milli: scaled_max_w.map(|w| (w * 1000.0) as u32),
        };

        if !self.cache.contains_key(&key) {
            let frame = self.typographer.shape_text(
                text,
                TextStyle {
                    font: style.font,
                    font_size: scaled_font_size,
                    line_height: scaled_line_height,
                    color: Color::WHITE,
                    max_width: scaled_max_w,
                },
                Point::ZERO,
            );
            self.cache.insert(key.clone(), frame.quads);
        }

        let dev_clip = clip_bounds.map(|r| {
            let p0 = self
                .base_tx
                .transform_point(Point::new(r.get_left(), r.get_top()));
            let p1 = self
                .base_tx
                .transform_point(Point::new(r.get_right(), r.get_bottom()));
            Rect::from_ltrb(
                p0.x.min(p1.x),
                p0.y.min(p1.y),
                p0.x.max(p1.x),
                p0.y.max(p1.y),
            )
        });

        if let Some(template_quads) = self.cache.get(&key) {
            for q in template_quads {
                let dest_rect = Rect::from_xywh(
                    q.dest_rect.get_left() + dev_origin_x,
                    q.dest_rect.get_top() + dev_origin_y,
                    q.dest_rect.get_width(),
                    q.dest_rect.get_height(),
                );
                if let Some(clip) = dev_clip {
                    if dest_rect.get_right() > clip.get_right() + 1.0
                        || dest_rect.get_bottom() > clip.get_bottom() + 1.0
                        || dest_rect.get_left() < clip.get_left() - 1.0
                        || dest_rect.get_top() < clip.get_top() - 1.0
                    {
                        continue;
                    }
                }
                self.quads.push(AtlasGlyphQuad {
                    dest_rect,
                    atlas_rect: q.atlas_rect,
                    color: style.color,
                    is_color: q.is_color,
                });
            }
        }
    }

    fn push_icon(
        &mut self,
        icon_font: FontId,
        icon: MaterialIcon,
        logical_size: Scalar,
        color: Color,
        logical_origin: Point,
        clip_bounds: Option<Rect>,
    ) {
        let mut buf = [0u8; 4];
        let s = icon.codepoint().encode_utf8(&mut buf);
        self.push_text(
            s,
            TextStyle {
                font: icon_font,
                font_size: logical_size,
                line_height: logical_size,
                color,
                max_width: None,
            },
            logical_origin,
            clip_bounds,
        );
    }
}

/// Renders the `flutter-wasm-compare` `MorphingLayoutMatrix` (`morphing_layout_matrix.dart`)
/// workload onto `canvas`:
/// - Computes dynamic grid `columns`, `rows`, `dynamic_aspect = (width / height) * (1.0 + 0.35 * sin(anim_value * 2 * PI))`,
///   and cell dimensions with `min_cell_extent = 8.0`.
/// - Draws `node_count` polymorphic cards (`WAVE`, `CORE`, `SYNC`, `Node`, `TH`, `WASM`) with batched
///   GPU draw commands and adaptive label/glyph caching for 1,000+ and 5,000+ card grids.
#[allow(clippy::too_many_arguments)]
pub fn draw_morphing_grid_workload(
    canvas: &mut Canvas,
    typographer: &mut TypographerContext,
    font_id: FontId,
    mono_font_id: FontId,
    icon_font_id: FontId,
    width: Scalar,
    height: Scalar,
    node_count: usize,
    anim_value: Scalar,
) {
    if node_count == 0 || width <= 0.0 || height <= 0.0 {
        return;
    }

    let t = (anim_value * 2.0 * std::f32::consts::PI).sin();
    let dynamic_aspect = (width / height) * (1.0 + 0.35 * t);
    let columns =
        ((node_count as Scalar * dynamic_aspect).sqrt().round() as usize).clamp(1, node_count);
    let rows = node_count.div_ceil(columns);

    let min_cell_extent: Scalar = 8.0;
    let raw_spacing: Scalar = if node_count > 1000 {
        1.0
    } else if node_count > 200 {
        2.0
    } else {
        4.0
    };
    let max_spacing_x = if columns > 1 {
        ((width - (columns as Scalar) * min_cell_extent) / ((columns - 1) as Scalar))
            .clamp(0.0, raw_spacing)
    } else {
        0.0
    };
    let max_spacing_y = if rows > 1 {
        ((height - (rows as Scalar) * min_cell_extent) / ((rows - 1) as Scalar))
            .clamp(0.0, raw_spacing)
    } else {
        0.0
    };
    let spacing = max_spacing_x.min(max_spacing_y);

    let total_spacing_x = spacing * (columns.saturating_sub(1) as Scalar);
    let total_spacing_y = spacing * (rows.saturating_sub(1) as Scalar);
    let cell_w = ((width - total_spacing_x) / (columns as Scalar)).max(min_cell_extent);
    let cell_h = ((height - total_spacing_y) / (rows as Scalar)).max(min_cell_extent);

    let sx = cell_w / 140.0;
    let sy = cell_h / 80.0;
    let avg_scale = (sx + sy) * 0.5;
    let skip_glyphs = (node_count > 400 && cell_h < 22.0) || cell_h < 12.0 || cell_w < 20.0;
    let use_static_labels = node_count > 120;
    let draw_borders = node_count <= 1200 && cell_w >= 20.0 && cell_h >= 12.0;

    let mut card_bg_mesh = TessellatedMesh::default();
    let mut card_border_mesh = TessellatedMesh::default();
    let mut track_mesh = TessellatedMesh::default();
    let mut amber_fill_mesh = TessellatedMesh::default();
    let mut amber_stroke_mesh = TessellatedMesh::default();
    let mut teal_fill_mesh = TessellatedMesh::default();
    let mut teal_stroke_mesh = TessellatedMesh::default();
    let mut orange_mesh = TessellatedMesh::default();
    let mut cyan_mesh = TessellatedMesh::default();
    let mut purple_mesh = TessellatedMesh::default();
    let mut green_status_mesh = TessellatedMesh::default();
    let mut dim_status_mesh = TessellatedMesh::default();

    let base_tx = canvas.get_current_transform();
    let mut shaper = FrameLabelShaper::new(
        typographer,
        base_tx,
        if skip_glyphs { 0 } else { node_count * 8 },
    );

    let orange = Color::make_rgba8(0xFF, 0xAB, 0x40, 0xFF);
    let cyan = Color::make_rgba8(0x18, 0xFF, 0xFF, 0xFF);
    let purple = Color::make_rgba8(0xE0, 0x40, 0xFB, 0xFF);
    let amber = Color::make_rgba8(0xFF, 0xD7, 0x40, 0xFF);
    let green_active = Color::make_rgba8(0x69, 0xF0, 0xAE, 0xFF);
    let dim_inactive = Color::new(1.0, 1.0, 1.0, 0.38);
    let teal = Color::make_rgba8(0x64, 0xFF, 0xDA, 0xFF);

    for index in 0..node_count {
        let col = index % columns;
        let row = index / columns;
        let x = (col as Scalar) * (cell_w + spacing);
        let y = (row as Scalar) * (cell_h + spacing);
        if x >= width || y >= height {
            continue;
        }

        let card_rect = Rect::from_xywh(
            x + 1.0 * sx,
            y + 1.0 * sy,
            (138.0 * sx).max(1.0),
            (78.0 * sy).max(1.0),
        );
        let card_rx = 4.0 * sx;
        let card_ry = 4.0 * sy;
        append_filled_rrect_mesh(&mut card_bg_mesh, card_rect, card_rx, card_ry);
        if draw_borders {
            append_stroked_rrect_mesh(
                &mut card_border_mesh,
                card_rect,
                card_rx,
                card_ry,
                (1.0 * avg_scale).max(0.75),
            );
        }

        let clip = Some(card_rect);
        match index % 6 {
            0 => {
                // WAVE sparkline card
                let x0 = x + 8.0 * sx;
                let y0 = y + 26.0 * sy;
                let w = 124.0 * sx;
                let h = 44.0 * sy;
                let pts_len = if node_count > 400 { 5usize } else { 8usize };
                let mut pts = [Point::ZERO; 8];
                for (i, pt) in pts.iter_mut().enumerate().take(pts_len) {
                    let px = x0 + (i as Scalar / (pts_len - 1) as Scalar) * w;
                    let phase = anim_value * 2.0 * std::f32::consts::PI
                        + (i as Scalar) * 0.7
                        + (index as Scalar);
                    let py = (y0 + h * 0.5) + phase.sin() * (h * 0.35);
                    *pt = Point::new(px, py);
                }
                append_stroked_polyline_mesh(
                    &mut orange_mesh,
                    &pts[..pts_len],
                    (2.0 * avg_scale).max(1.0),
                );
                if !skip_glyphs {
                    let label = if use_static_labels {
                        "WAVE".to_string()
                    } else {
                        format!("WAVE #{index}")
                    };
                    shaper.push_text(
                        &label,
                        TextStyle {
                            font: font_id,
                            font_size: 11.0 * sy,
                            line_height: 13.0 * sy,
                            color: orange,
                            max_width: None,
                        },
                        Point::new(x + 8.0 * sx, y + 8.0 * sy),
                        clip,
                    );
                    shaper.push_icon(
                        icon_font_id,
                        MaterialIcon::ShowChart,
                        14.0 * sy,
                        orange,
                        Point::new(x + 118.0 * sx, y + 7.0 * sy),
                        clip,
                    );
                }
            }
            1 => {
                // CORE gauge card
                let gauge_frac =
                    ((anim_value * 3.0 * std::f32::consts::PI + (index as Scalar) * 1.2).sin()
                        + 1.0)
                        * 0.5;
                let bar_rect = Rect::from_xywh(
                    x + 8.0 * sx,
                    y + 62.0 * sy,
                    (124.0 * gauge_frac.clamp(0.15, 1.0)) * sx,
                    (4.0 * sy).max(1.0),
                );
                append_filled_rrect_mesh(&mut cyan_mesh, bar_rect, 2.0 * sx, 2.0 * sy);
                if !skip_glyphs {
                    let title = if use_static_labels {
                        "CORE".to_string()
                    } else {
                        format!("CORE #{index}")
                    };
                    let ops = if use_static_labels {
                        "75 ops".to_string()
                    } else {
                        let gauge = (gauge_frac * 100.0).round() as i32;
                        format!("{gauge} ops")
                    };
                    shaper.push_text(
                        &title,
                        TextStyle {
                            font: font_id,
                            font_size: 11.0 * sy,
                            line_height: 13.0 * sy,
                            color: cyan,
                            max_width: None,
                        },
                        Point::new(x + 8.0 * sx, y + 18.0 * sy),
                        clip,
                    );
                    shaper.push_text(
                        &ops,
                        TextStyle {
                            font: mono_font_id,
                            font_size: 13.0 * sy,
                            line_height: 16.0 * sy,
                            color: Color::WHITE,
                            max_width: None,
                        },
                        Point::new(x + 8.0 * sx, y + 38.0 * sy),
                        clip,
                    );
                    shaper.push_icon(
                        icon_font_id,
                        MaterialIcon::Speed,
                        22.0 * sy,
                        cyan,
                        Point::new(x + 106.0 * sx, y + 26.0 * sy),
                        clip,
                    );
                }
            }
            2 => {
                // SYNC circular + linear progress card
                let progress =
                    (((anim_value * 2.0 * std::f32::consts::PI + (index as Scalar) * 0.5).cos()
                        + 1.0)
                        * 0.5)
                        .clamp(0.15, 0.95);
                let center = Point::new(x + 22.0 * sx, y + 40.0 * sy);
                let arc_segs = if node_count > 400 { 8 } else { 14 };
                append_stroked_arc_mesh(
                    &mut track_mesh,
                    center,
                    Size::new(12.0 * sx, 12.0 * sy),
                    0.0,
                    2.0 * std::f32::consts::PI,
                    (3.2 * avg_scale).max(1.0),
                    arc_segs,
                );
                append_stroked_arc_mesh(
                    &mut purple_mesh,
                    center,
                    Size::new(12.0 * sx, 12.0 * sy),
                    -std::f32::consts::FRAC_PI_2,
                    progress * 2.0 * std::f32::consts::PI,
                    (3.2 * avg_scale).max(1.0),
                    arc_segs,
                );
                let bar_track =
                    Rect::from_xywh(x + 42.0 * sx, y + 44.0 * sy, 86.0 * sx, (5.0 * sy).max(1.0));
                let bar_fill = Rect::from_xywh(
                    x + 42.0 * sx,
                    y + 44.0 * sy,
                    86.0 * progress * sx,
                    (5.0 * sy).max(1.0),
                );
                append_filled_rrect_mesh(&mut track_mesh, bar_track, 2.5 * sx, 2.5 * sy);
                append_filled_rrect_mesh(&mut purple_mesh, bar_fill, 2.5 * sx, 2.5 * sy);
                if !skip_glyphs {
                    let label = if use_static_labels {
                        "SYNC".to_string()
                    } else {
                        format!("SYNC {index}")
                    };
                    shaper.push_text(
                        &label,
                        TextStyle {
                            font: font_id,
                            font_size: 11.0 * sy,
                            line_height: 13.0 * sy,
                            color: purple,
                            max_width: None,
                        },
                        Point::new(x + 42.0 * sx, y + 22.0 * sy),
                        clip,
                    );
                }
            }
            3 => {
                // Node amber pill chip card
                let chip_rect =
                    Rect::from_xywh(x + 20.0 * sx, y + 24.0 * sy, 100.0 * sx, 32.0 * sy);
                append_filled_rrect_mesh(&mut amber_fill_mesh, chip_rect, 16.0 * sx, 16.0 * sy);
                if draw_borders {
                    append_stroked_rrect_mesh(
                        &mut amber_stroke_mesh,
                        chip_rect,
                        16.0 * sx,
                        16.0 * sy,
                        (1.0 * avg_scale).max(0.75),
                    );
                }
                if !skip_glyphs {
                    let icon = if index % 2 == 1 {
                        MaterialIcon::Bolt
                    } else {
                        MaterialIcon::AutoAwesome
                    };
                    shaper.push_icon(
                        icon_font_id,
                        icon,
                        14.0 * sy,
                        amber,
                        Point::new(x + 30.0 * sx, y + 32.0 * sy),
                        clip,
                    );
                    let label = if use_static_labels {
                        "Node".to_string()
                    } else {
                        format!("Node {index}")
                    };
                    shaper.push_text(
                        &label,
                        TextStyle {
                            font: font_id,
                            font_size: 11.0 * sy,
                            line_height: 13.0 * sy,
                            color: amber,
                            max_width: None,
                        },
                        Point::new(x + 48.0 * sx, y + 33.0 * sy),
                        clip,
                    );
                }
            }
            4 => {
                // TH status card
                let active =
                    (anim_value * 4.0 * std::f32::consts::PI + (index as Scalar)).sin() > 0.0;
                let color = if active { green_active } else { dim_inactive };
                let target_mesh = if active {
                    &mut green_status_mesh
                } else {
                    &mut dim_status_mesh
                };
                if skip_glyphs {
                    append_filled_circle_mesh(
                        target_mesh,
                        Point::new(x + 114.0 * sx, y + 38.0 * sy),
                        (7.0 * avg_scale).max(1.5),
                    );
                } else {
                    let label = if use_static_labels {
                        "TH".to_string()
                    } else {
                        format!("TH #{index}")
                    };
                    let icon = if active {
                        MaterialIcon::CheckCircle
                    } else {
                        MaterialIcon::RadioButtonUnchecked
                    };
                    shaper.push_text(
                        &label,
                        TextStyle {
                            font: font_id,
                            font_size: 11.0 * sy,
                            line_height: 14.0 * sy,
                            color,
                            max_width: None,
                        },
                        Point::new(x + 10.0 * sx, y + 32.0 * sy),
                        clip,
                    );
                    shaper.push_icon(
                        icon_font_id,
                        icon,
                        18.0 * sy,
                        color,
                        Point::new(x + 108.0 * sx, y + 29.0 * sy),
                        clip,
                    );
                }
            }
            _ => {
                // WASM teal box card
                let box_rect = Rect::from_xywh(x + 16.0 * sx, y + 24.0 * sy, 108.0 * sx, 32.0 * sy);
                append_filled_rrect_mesh(&mut teal_fill_mesh, box_rect, 5.0 * sx, 5.0 * sy);
                if draw_borders {
                    append_stroked_rrect_mesh(
                        &mut teal_stroke_mesh,
                        box_rect,
                        5.0 * sx,
                        5.0 * sy,
                        (1.2 * avg_scale).max(0.75),
                    );
                }
                if !skip_glyphs {
                    shaper.push_icon(
                        icon_font_id,
                        MaterialIcon::Memory,
                        14.0 * sy,
                        teal,
                        Point::new(x + 24.0 * sx, y + 32.0 * sy),
                        clip,
                    );
                    let label = if use_static_labels {
                        "WASM".to_string()
                    } else {
                        format!("WASM #{index}")
                    };
                    shaper.push_text(
                        &label,
                        TextStyle {
                            font: mono_font_id,
                            font_size: 11.0 * sy,
                            line_height: 13.0 * sy,
                            color: teal,
                            max_width: None,
                        },
                        Point::new(x + 42.0 * sx, y + 33.0 * sy),
                        clip,
                    );
                }
            }
        }
    }

    canvas.emit_mesh(
        card_bg_mesh,
        &Paint::fill(Color::make_rgba8(0x13, 0x17, 0x22, 0xFF)),
    );
    canvas.emit_mesh(
        card_border_mesh,
        &Paint::fill(Color::new(1.0, 1.0, 1.0, 0.14)),
    );
    canvas.emit_mesh(track_mesh, &Paint::fill(Color::new(1.0, 1.0, 1.0, 0.14)));
    canvas.emit_mesh(
        amber_fill_mesh,
        &Paint::fill(Color::new(1.0, 0.76, 0.03, 0.18)),
    );
    canvas.emit_mesh(
        amber_stroke_mesh,
        &Paint::fill(Color::new(1.0, 0.84, 0.25, 0.42)),
    );
    canvas.emit_mesh(
        teal_fill_mesh,
        &Paint::fill(Color::new(0.39, 1.0, 0.85, 0.16)),
    );
    canvas.emit_mesh(
        teal_stroke_mesh,
        &Paint::fill(Color::new(0.39, 1.0, 0.85, 0.45)),
    );
    canvas.emit_mesh(orange_mesh, &Paint::fill(orange));
    canvas.emit_mesh(cyan_mesh, &Paint::fill(cyan));
    canvas.emit_mesh(purple_mesh, &Paint::fill(purple));
    canvas.emit_mesh(green_status_mesh, &Paint::fill(green_active));
    canvas.emit_mesh(dim_status_mesh, &Paint::fill(dim_inactive));

    let glyph_quads = shaper.quads;
    canvas.emit_glyph_quads_batch(glyph_quads, Matrix::IDENTITY, false);
    canvas.sync_glyph_atlases_with(typographer);
}

fn collect_bouncy_leaves(
    node_index: usize,
    count: usize,
    depth: usize,
    rect: Rect,
    anim_value: Scalar,
    leaves: &mut Vec<(usize, Rect)>,
) {
    if count <= 1 || rect.get_width() < 6.0 || rect.get_height() < 6.0 {
        leaves.push((node_index, rect));
        return;
    }

    let left_count = count / 2;
    let right_count = count - left_count;

    let hash = (node_index as u64).wrapping_mul(2654435761);
    let is_reversed = ((hash >> 16) & 1) == 1;
    let phase_offset = ((node_index as Scalar) * 0.17) % 1.0;
    let local_t = (anim_value + phase_offset) % 1.0;
    let wave = (local_t * 2.0 * std::f32::consts::PI).sin();
    let delta = (wave * 3200.0).round() as i32;
    let flex1 = (5000 + delta).clamp(1200, 8800) as Scalar;
    let flex2 = (5000 - delta).clamp(1200, 8800) as Scalar;
    let ratio1 = flex1 / (flex1 + flex2);

    let (first_idx, first_cnt, second_idx, second_cnt) = if is_reversed {
        (
            node_index + left_count,
            right_count,
            node_index + 1,
            left_count,
        )
    } else {
        (
            node_index + 1,
            left_count,
            node_index + left_count,
            right_count,
        )
    };

    if depth.is_multiple_of(2) {
        let h1 = rect.get_height() * ratio1;
        let r1 = Rect::from_xywh(rect.get_left(), rect.get_top(), rect.get_width(), h1);
        let r2 = Rect::from_xywh(
            rect.get_left(),
            rect.get_top() + h1,
            rect.get_width(),
            (rect.get_height() - h1).max(0.0),
        );
        collect_bouncy_leaves(first_idx, first_cnt, depth + 1, r1, anim_value, leaves);
        collect_bouncy_leaves(second_idx, second_cnt, depth + 1, r2, anim_value, leaves);
    } else {
        let w1 = rect.get_width() * ratio1;
        let r1 = Rect::from_xywh(rect.get_left(), rect.get_top(), w1, rect.get_height());
        let r2 = Rect::from_xywh(
            rect.get_left() + w1,
            rect.get_top(),
            (rect.get_width() - w1).max(0.0),
            rect.get_height(),
        );
        collect_bouncy_leaves(first_idx, first_cnt, depth + 1, r1, anim_value, leaves);
        collect_bouncy_leaves(second_idx, second_cnt, depth + 1, r2, anim_value, leaves);
    }
}

/// Renders the `flutter-wasm-compare` `BouncyLayoutMatrix` (`bouncy_layout_matrix.dart`)
/// recursive binary space partitioning workload onto `canvas`:
/// - Splits the viewport recursively using deterministic Knuth multiplicative hashing
///   (`(node_index as u64).wrapping_mul(2654435761)`), `is_reversed = ((hash >> 16) & 1) == 1`,
///   `phase_offset = (node_index * 0.17) % 1.0`, `flex: 5000 ± delta` (`1200..8800`), alternating
///   `Column` (`depth % 2 == 0`) and `Row` (`depth % 2 == 1`).
/// - Renders 7 distinct leaf widget types (button, checkbox, reflowing paragraph text, timer picker,
///   circular progress arc, slider, and mini app bar) with batched GPU meshes and glyph runs.
#[allow(clippy::too_many_arguments)]
pub fn draw_bouncy_workload(
    canvas: &mut Canvas,
    typographer: &mut TypographerContext,
    font_id: FontId,
    mono_font_id: FontId,
    icon_font_id: FontId,
    width: Scalar,
    height: Scalar,
    node_count: usize,
    anim_value: Scalar,
) {
    if node_count == 0 || width <= 0.0 || height <= 0.0 {
        return;
    }

    let root_rect = Rect::from_xywh(4.0, 4.0, (width - 8.0).max(1.0), (height - 8.0).max(1.0));
    let mut leaves = Vec::with_capacity(node_count);
    collect_bouncy_leaves(0, node_count, 0, root_rect, anim_value, &mut leaves);

    let mut pane_bg_mesh = TessellatedMesh::default();
    let mut pane_border_mesh = TessellatedMesh::default();
    let mut appbar_bg_mesh = TessellatedMesh::default();
    let mut track_mesh = TessellatedMesh::default();
    let mut blue_btn_mesh = TessellatedMesh::default();
    let mut green_mesh = TessellatedMesh::default();
    let mut amber_bg_mesh = TessellatedMesh::default();
    let mut amber_border_mesh = TessellatedMesh::default();
    let mut purple_mesh = TessellatedMesh::default();
    let mut cyan_mesh = TessellatedMesh::default();
    let mut teal_skeleton_mesh = TessellatedMesh::default();

    let base_tx = canvas.get_current_transform();
    let mut shaper = FrameLabelShaper::new(typographer, base_tx, node_count * 10);
    let use_static_labels = node_count > 100;

    let blue_btn = Color::make_rgba8(0x1E, 0x88, 0xE5, 0xFF);
    let green = Color::make_rgba8(0x00, 0xE6, 0x76, 0xFF);
    let amber = Color::make_rgba8(0xFF, 0xAB, 0x40, 0xFF);
    let purple = Color::make_rgba8(0xE0, 0x40, 0xFB, 0xFF);
    let cyan = Color::make_rgba8(0x18, 0xFF, 0xFF, 0xFF);
    let teal = Color::make_rgba8(0x64, 0xFF, 0xDA, 0xFF);

    for (node_index, cell_rect) in leaves {
        let pane = Rect::from_xywh(
            cell_rect.get_left() + 1.5,
            cell_rect.get_top() + 1.5,
            (cell_rect.get_width() - 3.0).max(1.0),
            (cell_rect.get_height() - 3.0).max(1.0),
        );
        let pw = pane.get_width();
        let ph = pane.get_height();
        let px = pane.get_left();
        let py = pane.get_top();
        let cx = px + pw * 0.5;
        let cy = py + ph * 0.5;

        append_filled_rrect_mesh(&mut pane_bg_mesh, pane, 6.0, 6.0);
        if pw >= 12.0 && ph >= 10.0 {
            append_stroked_rrect_mesh(&mut pane_border_mesh, pane, 6.0, 6.0, 1.0);
        }

        let draw_glyphs =
            pw >= 28.0 && ph >= 16.0 && (node_count <= 300 || (pw >= 38.0 && ph >= 20.0));
        let clip = Some(pane);

        match node_index % 7 {
            0 => {
                // Button leaf
                let bw = (pw - 12.0).clamp(6.0, 132.0);
                let bh = (ph - 10.0).clamp(6.0, 32.0);
                let btn_rect = Rect::from_xywh(cx - bw * 0.5, cy - bh * 0.5, bw, bh);
                append_filled_rrect_mesh(&mut blue_btn_mesh, btn_rect, 6.0, 6.0);
                if draw_glyphs && bw >= 36.0 && bh >= 14.0 {
                    shaper.push_icon(
                        icon_font_id,
                        MaterialIcon::Bolt,
                        12.0,
                        Color::WHITE,
                        Point::new(btn_rect.get_left() + 6.0, cy - 6.0),
                        Some(btn_rect),
                    );
                    let label = if use_static_labels {
                        "ACTION".to_string()
                    } else {
                        format!("ACTION #{node_index}")
                    };
                    shaper.push_text(
                        &label,
                        TextStyle {
                            font: font_id,
                            font_size: 10.5,
                            line_height: 12.5,
                            color: Color::WHITE,
                            max_width: None,
                        },
                        Point::new(btn_rect.get_left() + 20.0, cy - 6.0),
                        Some(btn_rect),
                    );
                }
            }
            1 => {
                // Checkbox leaf
                let cb_size = 16.0_f32.min(pw * 0.35).min(ph * 0.6).max(4.0);
                let cb_rect = Rect::from_xywh(px + 8.0, cy - cb_size * 0.5, cb_size, cb_size);
                append_filled_rrect_mesh(&mut green_mesh, cb_rect, 3.5, 3.5);
                if draw_glyphs && pw >= 44.0 {
                    let label = if use_static_labels {
                        "OPT".to_string()
                    } else {
                        format!("OPT #{node_index}")
                    };
                    shaper.push_text(
                        &label,
                        TextStyle {
                            font: font_id,
                            font_size: 11.0,
                            line_height: 13.0,
                            color: Color::new(1.0, 1.0, 1.0, 0.78),
                            max_width: None,
                        },
                        Point::new(cb_rect.get_right() + 6.0, cy - 6.5),
                        clip,
                    );
                }
            }
            2 => {
                // Reflowing paragraph text leaf
                let bar_w = (pw - 12.0).max(2.0);
                append_filled_rrect_mesh(
                    &mut track_mesh,
                    Rect::from_xywh(px + 6.0, py + 5.0, bar_w, 2.5),
                    1.2,
                    1.2,
                );
                if draw_glyphs && ph >= 22.0 {
                    let wrap_w = if use_static_labels {
                        (((pw - 12.0).max(24.0) / 24.0).round() * 24.0).max(24.0)
                    } else {
                        (pw - 12.0).max(20.0)
                    };
                    let para = if use_static_labels {
                        "Dynamic layout reflow stress test under shifting constraints.".to_string()
                    } else {
                        format!(
                            "Node #{node_index}: Dynamic layout reflow stress test under shifting constraints."
                        )
                    };
                    shaper.push_text(
                        &para,
                        TextStyle {
                            font: font_id,
                            font_size: 9.5,
                            line_height: 11.5,
                            color: Color::new(1.0, 1.0, 1.0, 0.65),
                            max_width: Some(wrap_w),
                        },
                        Point::new(px + 6.0, py + 10.0),
                        clip,
                    );
                }
            }
            3 => {
                // Timer picker leaf
                let tw = (pw - 10.0).clamp(6.0, 118.0);
                let th = (ph - 8.0).clamp(6.0, 32.0);
                let trect = Rect::from_xywh(cx - tw * 0.5, cy - th * 0.5, tw, th);
                append_filled_rrect_mesh(&mut amber_bg_mesh, trect, 6.0, 6.0);
                append_stroked_rrect_mesh(&mut amber_border_mesh, trect, 6.0, 6.0, 1.0);
                if draw_glyphs && tw >= 38.0 && th >= 14.0 {
                    shaper.push_icon(
                        icon_font_id,
                        MaterialIcon::Speed,
                        13.0,
                        amber,
                        Point::new(trect.get_left() + 6.0, cy - 6.5),
                        Some(trect),
                    );
                    let time_str = if use_static_labels {
                        "04:42".to_string()
                    } else {
                        let secs = ((anim_value * 60.0) as usize + node_index) % 60;
                        let mins = (node_index % 12) + 1;
                        format!("{mins:02}:{secs:02}")
                    };
                    shaper.push_text(
                        &time_str,
                        TextStyle {
                            font: mono_font_id,
                            font_size: 11.5,
                            line_height: 13.5,
                            color: amber,
                            max_width: None,
                        },
                        Point::new(trect.get_left() + 22.0, cy - 6.5),
                        Some(trect),
                    );
                }
            }
            4 => {
                // Circular progress indicator leaf
                let rad = (pw.min(ph) * 0.28).clamp(3.0, 13.0);
                let progress = 0.25
                    + 0.65
                        * ((anim_value * 2.0 * std::f32::consts::PI + (node_index as Scalar))
                            .sin()
                            * 0.5
                            + 0.5);
                let arc_center = if pw > 74.0 {
                    Point::new(px + rad + 10.0, cy)
                } else {
                    Point::new(cx, cy)
                };
                append_stroked_arc_mesh(
                    &mut track_mesh,
                    arc_center,
                    Size::new(rad, rad),
                    0.0,
                    2.0 * std::f32::consts::PI,
                    2.8,
                    12,
                );
                append_stroked_arc_mesh(
                    &mut purple_mesh,
                    arc_center,
                    Size::new(rad, rad),
                    -std::f32::consts::FRAC_PI_2,
                    progress * 2.0 * std::f32::consts::PI,
                    2.8,
                    12,
                );
                if draw_glyphs && pw > 74.0 {
                    let pct = if use_static_labels {
                        "72%".to_string()
                    } else {
                        format!("{}%", (progress * 100.0).round() as i32)
                    };
                    shaper.push_text(
                        &pct,
                        TextStyle {
                            font: mono_font_id,
                            font_size: 11.0,
                            line_height: 13.0,
                            color: purple,
                            max_width: None,
                        },
                        Point::new(arc_center.x + rad + 8.0, cy - 6.5),
                        clip,
                    );
                }
            }
            5 => {
                // Slider leaf
                let val = 0.2
                    + 0.6
                        * ((anim_value * 2.0 * std::f32::consts::PI
                            + (node_index as Scalar) * 0.5)
                            .cos()
                            * 0.5
                            + 0.5);
                let sx0 = px + 8.0;
                let sw = (pw - 16.0).max(4.0);
                let sy0 = cy + 2.0;
                append_filled_rrect_mesh(
                    &mut track_mesh,
                    Rect::from_xywh(sx0, sy0, sw, 4.0),
                    2.0,
                    2.0,
                );
                append_filled_rrect_mesh(
                    &mut cyan_mesh,
                    Rect::from_xywh(sx0, sy0, sw * val, 4.0),
                    2.0,
                    2.0,
                );
                let thumb_r = 5.0_f32.min(ph * 0.25).max(2.0);
                append_filled_circle_mesh(
                    &mut cyan_mesh,
                    Point::new(sx0 + sw * val, sy0 + 2.0),
                    thumb_r,
                );
                if draw_glyphs && ph >= 28.0 {
                    let label = if use_static_labels {
                        "SLIDER".to_string()
                    } else {
                        format!("SLIDER #{node_index}")
                    };
                    shaper.push_text(
                        &label,
                        TextStyle {
                            font: mono_font_id,
                            font_size: 9.5,
                            line_height: 11.5,
                            color: cyan,
                            max_width: None,
                        },
                        Point::new(sx0, py + 5.0),
                        clip,
                    );
                }
            }
            _ => {
                // Mini AppBar leaf
                let bar_h = (ph * 0.38).clamp(4.0, 22.0);
                let header_rect = Rect::from_xywh(px + 1.0, py + 1.0, (pw - 2.0).max(2.0), bar_h);
                append_filled_rrect_mesh(&mut appbar_bg_mesh, header_rect, 4.0, 4.0);
                if ph > bar_h + 12.0 && pw > 16.0 {
                    let body_y = py + bar_h + 6.0;
                    append_filled_rrect_mesh(
                        &mut track_mesh,
                        Rect::from_xywh(px + 8.0, body_y, (pw - 16.0) * 0.75, 4.0),
                        2.0,
                        2.0,
                    );
                    if ph > bar_h + 22.0 {
                        append_filled_rrect_mesh(
                            &mut teal_skeleton_mesh,
                            Rect::from_xywh(px + 8.0, body_y + 8.0, (pw - 16.0) * 0.5, 4.0),
                            2.0,
                            2.0,
                        );
                    }
                }
                if draw_glyphs && bar_h >= 14.0 && pw >= 44.0 {
                    shaper.push_icon(
                        icon_font_id,
                        MaterialIcon::Memory,
                        11.0,
                        teal,
                        Point::new(px + 6.0, py + (bar_h - 11.0) * 0.5),
                        Some(header_rect),
                    );
                    let title = if use_static_labels {
                        "PANE".to_string()
                    } else {
                        format!("PANE #{node_index}")
                    };
                    shaper.push_text(
                        &title,
                        TextStyle {
                            font: font_id,
                            font_size: 9.5,
                            line_height: 11.5,
                            color: Color::WHITE,
                            max_width: None,
                        },
                        Point::new(px + 20.0, py + (bar_h - 11.0) * 0.5),
                        Some(header_rect),
                    );
                }
            }
        }
    }

    canvas.emit_mesh(
        pane_bg_mesh,
        &Paint::fill(Color::make_rgba8(0x16, 0x1B, 0x26, 0xFF)),
    );
    canvas.emit_mesh(
        appbar_bg_mesh,
        &Paint::fill(Color::make_rgba8(0x1F, 0x26, 0x33, 0xFF)),
    );
    canvas.emit_mesh(
        pane_border_mesh,
        &Paint::fill(Color::new(1.0, 1.0, 1.0, 0.12)),
    );
    canvas.emit_mesh(track_mesh, &Paint::fill(Color::new(1.0, 1.0, 1.0, 0.14)));
    canvas.emit_mesh(
        amber_bg_mesh,
        &Paint::fill(Color::new(1.0, 0.67, 0.25, 0.12)),
    );
    canvas.emit_mesh(
        amber_border_mesh,
        &Paint::fill(Color::new(1.0, 0.67, 0.25, 0.40)),
    );
    canvas.emit_mesh(
        teal_skeleton_mesh,
        &Paint::fill(Color::new(0.39, 1.0, 0.85, 0.24)),
    );
    canvas.emit_mesh(blue_btn_mesh, &Paint::fill(blue_btn));
    canvas.emit_mesh(green_mesh, &Paint::fill(green));
    canvas.emit_mesh(purple_mesh, &Paint::fill(purple));
    canvas.emit_mesh(cyan_mesh, &Paint::fill(cyan));

    let glyph_quads = shaper.quads;
    canvas.emit_glyph_quads_batch(glyph_quads, Matrix::IDENTITY, false);
    canvas.sync_glyph_atlases_with(typographer);
}

/// Verifies that all 4 quadrants of the rendered 800x600 showcase image have non-empty,
/// distinct rendered pixel distributions.
pub fn verify_showcase_pixels(
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<ShowcaseStats, String> {
    let expected_len = (width as usize) * (height as usize) * 4;
    if rgba.len() != expected_len {
        return Err(format!(
            "Expected RGBA buffer of {} bytes for {}x{}, got {}",
            expected_len,
            width,
            height,
            rgba.len()
        ));
    }

    let bg_r = (0.06f32 * 255.0).round() as i32;
    let bg_g = (0.07f32 * 255.0).round() as i32;
    let bg_b = (0.10f32 * 255.0).round() as i32;

    let half_w = width / 2;
    let half_h = height / 2;
    let regions = [
        (0, 0, half_w, half_h),          // Q1: Top-Left
        (half_w, 0, width, half_h),      // Q2: Top-Right
        (0, half_h, half_w, height),     // Q3: Bottom-Left
        (half_w, half_h, width, height), // Q4: Bottom-Right
    ];

    let mut stats = [QuadrantStats {
        non_background_pixels: 0,
        total_pixels: 0,
        mean_r: 0.0,
        mean_g: 0.0,
        mean_b: 0.0,
        distinct_color_buckets: 0,
    }; 4];

    for (q_idx, &(x0, y0, x1, y1)) in regions.iter().enumerate() {
        let mut non_bg = 0usize;
        let mut total = 0usize;
        let mut sum_r = 0u64;
        let mut sum_g = 0u64;
        let mut sum_b = 0u64;
        let mut buckets = [false; 512]; // 3-bit quantized RGB (8x8x8)

        for y in y0..y1 {
            for x in x0..x1 {
                let idx = ((y * width + x) as usize) * 4;
                let r = rgba[idx];
                let g = rgba[idx + 1];
                let b = rgba[idx + 2];
                total += 1;
                sum_r += r as u64;
                sum_g += g as u64;
                sum_b += b as u64;

                let diff =
                    (r as i32 - bg_r).abs() + (g as i32 - bg_g).abs() + (b as i32 - bg_b).abs();
                if diff > 6 {
                    non_bg += 1;
                }
                let b_idx =
                    (((r >> 5) as usize) << 6) | (((g >> 5) as usize) << 3) | ((b >> 5) as usize);
                buckets[b_idx] = true;
            }
        }

        let distinct = buckets.iter().filter(|&&b| b).count();
        stats[q_idx] = QuadrantStats {
            non_background_pixels: non_bg,
            total_pixels: total,
            mean_r: (sum_r as f64) / (total.max(1) as f64),
            mean_g: (sum_g as f64) / (total.max(1) as f64),
            mean_b: (sum_b as f64) / (total.max(1) as f64),
            distinct_color_buckets: distinct,
        };

        if non_bg < 1000 {
            return Err(format!(
                "Quadrant {} has too few rendered non-background pixels: {} / {}",
                q_idx + 1,
                non_bg,
                total
            ));
        }
        if distinct < 4 {
            return Err(format!(
                "Quadrant {} has too few distinct colors: {}",
                q_idx + 1,
                distinct
            ));
        }
    }

    // Verify all 4 quadrants have mutually distinct color distributions
    for i in 0..4 {
        for j in (i + 1)..4 {
            let qi = &stats[i];
            let qj = &stats[j];
            let mean_dist = (qi.mean_r - qj.mean_r).abs()
                + (qi.mean_g - qj.mean_g).abs()
                + (qi.mean_b - qj.mean_b).abs();
            let count_diff = qi.non_background_pixels.abs_diff(qj.non_background_pixels);
            if mean_dist < 0.5 && count_diff == 0 {
                return Err(format!(
                    "Quadrants {} and {} have identical pixel distributions",
                    i + 1,
                    j + 1
                ));
            }
        }
    }

    Ok(ShowcaseStats {
        width,
        height,
        quadrants: stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn get_pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let idx = ((y * width + x) as usize) * 4;
        [rgba[idx], rgba[idx + 1], rgba[idx + 2], rgba[idx + 3]]
    }

    fn shared_test_renderer() -> std::sync::MutexGuard<'static, WgpuRenderer> {
        static RENDERER: std::sync::OnceLock<std::sync::Mutex<WgpuRenderer>> =
            std::sync::OnceLock::new();
        RENDERER
            .get_or_init(|| {
                std::sync::Mutex::new(WgpuRenderer::new_headless().expect("headless wgpu renderer"))
            })
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// C++ source: canvas_unittests.cc:TransformMultipliesCorrectly
    #[test]
    fn test_transform_multiplies_correctly() {
        let mut canvas = Canvas::new(100, 100, Color::BLACK);
        assert_eq!(canvas.get_current_transform(), Matrix::IDENTITY);

        canvas.translate(100.0, 200.0);
        let m1 = canvas.get_current_transform();
        assert_relative_eq!(m1.m[12], 100.0, epsilon = 1e-4);
        assert_relative_eq!(m1.m[13], 200.0, epsilon = 1e-4);

        canvas.rotate(Radians(std::f32::consts::FRAC_PI_2));
        let m2 = canvas.get_current_transform();
        assert_relative_eq!(m2.m[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(m2.m[1], 1.0, epsilon = 1e-4);
        assert_relative_eq!(m2.m[4], -1.0, epsilon = 1e-4);
        assert_relative_eq!(m2.m[5], 0.0, epsilon = 1e-4);
        assert_relative_eq!(m2.m[12], 100.0, epsilon = 1e-4);
        assert_relative_eq!(m2.m[13], 200.0, epsilon = 1e-4);

        canvas.scale_3d(2.0, 3.0, 0.0);
        let m3 = canvas.get_current_transform();
        assert_relative_eq!(m3.m[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(m3.m[1], 2.0, epsilon = 1e-4);
        assert_relative_eq!(m3.m[4], -3.0, epsilon = 1e-4);
        assert_relative_eq!(m3.m[5], 0.0, epsilon = 1e-4);
        assert_relative_eq!(m3.m[10], 0.0, epsilon = 1e-4);

        canvas.translate(100.0, 200.0);
        let m4 = canvas.get_current_transform();
        assert_relative_eq!(m4.m[12], -500.0, epsilon = 1e-3);
        assert_relative_eq!(m4.m[13], 400.0, epsilon = 1e-3);
    }

    /// C++ source: canvas_unittests.cc:CanvasCanPushPopCTM
    #[test]
    fn test_canvas_can_push_pop_ctm() {
        let mut canvas = Canvas::new(100, 100, Color::BLACK);
        assert_eq!(canvas.get_save_count(), 1);
        assert!(!canvas.restore());

        canvas.translate(100.0, 100.0);
        canvas.save();
        assert_eq!(canvas.get_save_count(), 2);
        assert_eq!(
            canvas.get_current_transform(),
            Matrix::make_translation(100.0, 100.0, 0.0)
        );

        canvas.scale(2.0, 2.0);
        canvas.skew(0.5, 0.25);
        canvas.save();
        assert_eq!(canvas.get_save_count(), 3);
        canvas.reset_transform();
        assert_eq!(canvas.get_current_transform(), Matrix::IDENTITY);

        canvas.restore_to_count(1);
        assert_eq!(canvas.get_save_count(), 1);
        assert_eq!(
            canvas.get_current_transform(),
            Matrix::make_translation(100.0, 100.0, 0.0)
        );
    }

    /// C++ source: paint_unittests.cc:DefaultsMatchDlPaintDefaults
    #[test]
    fn test_paint_defaults_and_optional_stroke() {
        let mut paint = Paint::default();
        assert!(paint.anti_alias);
        assert!(!paint.invert_colors);
        assert_eq!(paint.color, Color::BLACK);
        assert!(paint.effective_color_source().is_none());
        assert_eq!(paint.style, DrawStyle::Fill);
        assert_eq!(paint.blend_mode, BlendMode::SrcOver);
        assert_eq!(paint.stroke_width, 0.0);
        assert_eq!(paint.stroke_miter, 4.0);
        assert_eq!(paint.stroke_cap, Cap::Butt);
        assert_eq!(paint.stroke_join, Join::Miter);
        assert!(paint.can_apply_opacity_peephole());

        paint.stroke_cap = Cap::Round;
        paint.stroke_join = Join::Round;
        paint.stroke_width = 20.0;
        paint.stroke_miter = 100.0;
        assert!(paint.get_stroke().is_none());

        paint.style = DrawStyle::Stroke;
        let stroke = paint
            .get_stroke()
            .expect("stroke parameters when DrawStyle::Stroke");
        assert_eq!(stroke.cap, Cap::Round);
        assert_eq!(stroke.join, Join::Round);
        assert_eq!(stroke.width, 20.0);
        assert_eq!(stroke.miter_limit, 100.0);
    }

    /// C++ source: paint_unittests.cc:GradientStopConversion
    #[test]
    fn test_paint_gradient_stop_conversion() {
        // Missing 0.0 stop
        let missing_zero = Paint::convert_stops(&[
            GradientStop {
                offset: 0.5,
                color: Color::BLUE,
            },
            GradientStop {
                offset: 1.0,
                color: Color::RED,
            },
        ]);
        assert_eq!(missing_zero.len(), 3);
        assert_relative_eq!(missing_zero[0].offset, 0.0);
        assert_eq!(missing_zero[0].color, Color::BLUE);
        assert_relative_eq!(missing_zero[1].offset, 0.5);
        assert_relative_eq!(missing_zero[2].offset, 1.0);

        // Missing 1.0 stop
        let missing_last = Paint::convert_stops(&[
            GradientStop {
                offset: 0.0,
                color: Color::BLUE,
            },
            GradientStop {
                offset: 0.5,
                color: Color::RED,
            },
        ]);
        assert_eq!(missing_last.len(), 3);
        assert_relative_eq!(missing_last[2].offset, 1.0);
        assert_eq!(missing_last[2].color, Color::RED);

        // Non-monotonic & >1.0 clamping
        let non_monotonic = Paint::convert_stops(&[
            GradientStop {
                offset: 0.0,
                color: Color::BLUE,
            },
            GradientStop {
                offset: 0.5,
                color: Color::GREEN,
            },
            GradientStop {
                offset: 0.4,
                color: Color::GREEN,
            },
            GradientStop {
                offset: 100.0,
                color: Color::RED,
            },
        ]);
        assert_relative_eq!(non_monotonic[0].offset, 0.0);
        assert_relative_eq!(non_monotonic[1].offset, 0.5);
        assert_relative_eq!(non_monotonic[2].offset, 0.5);
        assert_relative_eq!(non_monotonic[3].offset, 1.0);
    }

    /// C++ source: aiks_unittests.cc:CanRenderColoredRect
    #[test]
    fn test_can_render_colored_rect() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(128, 128, Color::BLACK);
        canvas.draw_rect(
            Rect::from_xywh(24.0, 24.0, 80.0, 80.0),
            &Paint::fill(Color::BLUE),
        );
        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render colored rect");
        assert_eq!(get_pixel(&rgba, 128, 64, 64), [0, 0, 255, 255]);
        assert_eq!(get_pixel(&rgba, 128, 8, 8), [0, 0, 0, 255]);
    }

    /// C++ source: aiks_unittests.cc:CanRenderColorFilterWithInvertColorsDrawPaint
    #[test]
    fn test_can_render_invert_colors_draw_paint() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(64, 64, Color::BLACK);
        canvas.scale(0.25, 0.25);
        let mut paint = Paint::fill(Color::new(1.0, 0.2, 0.0, 1.0));
        paint.invert_colors = true;
        canvas.draw_paint(&paint);

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render draw_paint");
        let px = get_pixel(&rgba, 64, 48, 48);
        assert_eq!(px[0], 0);
        assert!(px[1] >= 200);
        assert_eq!(px[2], 255);
        assert_eq!(px[3], 255);
    }

    /// C++ source: aiks_unittests.cc:CanRenderLinearGradientClamp
    #[test]
    fn test_can_render_linear_gradient_clamp_and_mirror() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(120, 80, Color::BLACK);
        let grad = LinearGradientParams {
            start_point: Point::new(20.0, 0.0),
            end_point: Point::new(100.0, 0.0),
            stops: vec![
                GradientStop {
                    offset: 0.0,
                    color: Color::RED,
                },
                GradientStop {
                    offset: 1.0,
                    color: Color::BLUE,
                },
            ],
            tile_mode: TileMode::Clamp,
            alpha: 1.0,
        };
        canvas.draw_rect(
            Rect::from_xywh(0.0, 0.0, 120.0, 80.0),
            &Paint::fill(Color::WHITE).with_color_source(ColorSource::LinearGradient(grad)),
        );
        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render linear gradient");
        let left = get_pixel(&rgba, 120, 10, 40);
        let mid = get_pixel(&rgba, 120, 60, 40);
        let right = get_pixel(&rgba, 120, 110, 40);
        assert!(
            left[0] > 230 && left[2] < 25,
            "left clamped to red: {:?}",
            left
        );
        assert!(
            mid[0] > 90 && mid[2] > 90,
            "mid interpolated purple: {:?}",
            mid
        );
        assert!(
            right[2] > 230 && right[0] < 25,
            "right clamped to blue: {:?}",
            right
        );
    }

    /// C++ source: aiks_unittests.cc:CanRenderRadialGradientWithDitheringEnabled
    #[test]
    fn test_can_render_radial_gradient() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(128, 128, Color::BLACK);
        let grad = RadialGradientParams {
            center: Point::new(64.0, 64.0),
            radius: 48.0,
            stops: vec![
                GradientStop {
                    offset: 0.0,
                    color: Color::WHITE,
                },
                GradientStop {
                    offset: 1.0,
                    color: Color::RED,
                },
            ],
            tile_mode: TileMode::Clamp,
            alpha: 1.0,
        };
        canvas.draw_circle(
            Point::new(64.0, 64.0),
            56.0,
            &Paint::fill(Color::WHITE).with_color_source(ColorSource::RadialGradient(grad)),
        );
        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render radial gradient");
        let center = get_pixel(&rgba, 128, 64, 64);
        let edge = get_pixel(&rgba, 128, 64 + 46, 64);
        assert!(center[0] > 230 && center[1] > 230 && center[2] > 230);
        assert!(edge[0] > 220 && edge[1] < 40 && edge[2] < 40);
    }

    /// C++ source: aiks_unittests.cc:CanRenderNestedClips
    #[test]
    fn test_can_render_nested_clips() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(120, 120, Color::BLACK);

        canvas.save();
        canvas.clip_rect(Rect::from_xywh(10.0, 10.0, 40.0, 40.0));
        canvas.restore();

        canvas.clip_rect(Rect::from_xywh(20.0, 20.0, 70.0, 70.0));
        canvas.clip_rect(Rect::from_xywh(50.0, 50.0, 60.0, 60.0));

        // Intersection of [20..90, 20..90] and [50..110, 50..110] is [50..90, 50..90]
        let cull = canvas
            .get_current_local_cull_rect()
            .expect("non-empty clip");
        assert_relative_eq!(cull.get_left(), 50.0, epsilon = 1e-4);
        assert_relative_eq!(cull.get_top(), 50.0, epsilon = 1e-4);
        assert_relative_eq!(cull.get_right(), 90.0, epsilon = 1e-4);
        assert_relative_eq!(cull.get_bottom(), 90.0, epsilon = 1e-4);

        canvas.draw_rect(
            Rect::from_xywh(0.0, 0.0, 120.0, 120.0),
            &Paint::fill(Color::GREEN),
        );
        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render nested clips");
        assert_eq!(get_pixel(&rgba, 120, 70, 70), [0, 255, 0, 255]);
        assert_eq!(get_pixel(&rgba, 120, 35, 35), [0, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 120, 100, 100), [0, 0, 0, 255]);
    }

    /// C++ source: aiks_unittests.cc:CanRenderDifferenceClips
    #[test]
    fn test_can_render_difference_clips() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(120, 120, Color::BLACK);

        canvas.clip_rect(Rect::from_xywh(20.0, 20.0, 80.0, 80.0));
        canvas.save();
        canvas.clip_rect_with_op(
            Rect::from_xywh(45.0, 45.0, 30.0, 30.0),
            ClipOperation::Difference,
        );
        canvas.draw_paint(&Paint::fill(Color::RED));
        canvas.restore();

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render difference clips");
        // Inside outer clip [20..100], outside difference hole [45..75] -> RED
        assert_eq!(get_pixel(&rgba, 120, 30, 30), [255, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 120, 85, 85), [255, 0, 0, 255]);
        // Inside difference hole [45..75] -> untouched BLACK
        assert_eq!(get_pixel(&rgba, 120, 60, 60), [0, 0, 0, 255]);
        // Outside outer clip -> untouched BLACK
        assert_eq!(get_pixel(&rgba, 120, 10, 10), [0, 0, 0, 255]);
    }

    /// C++ source: aiks_unittests.cc:CanRenderWithContiguousClipRestores
    #[test]
    fn test_can_render_with_contiguous_clip_restores() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(100, 100, Color::BLACK);
        canvas.draw_paint(&Paint::fill(Color::RED));

        canvas.save();
        canvas.clip_rect(Rect::from_xywh(10.0, 10.0, 20.0, 20.0));
        canvas.clip_rect(Rect::from_xywh(60.0, 60.0, 20.0, 20.0));
        // Disjoint clips -> empty coverage
        assert!(canvas.get_current_local_cull_rect().is_none());
        canvas.draw_paint(&Paint::fill(Color::BLUE));
        canvas.restore();

        canvas.draw_paint(&Paint::fill(Color::GREEN));
        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render clip restore");
        assert_eq!(get_pixel(&rgba, 100, 50, 50), [0, 255, 0, 255]);
    }

    /// C++ source: aiks_unittests.cc:CanRenderStrokes
    #[test]
    fn test_can_render_strokes_and_caps() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(140, 100, Color::BLACK);

        // Butt cap line from x=40 to x=100 at y=30 (width 20)
        canvas.draw_line(
            Point::new(40.0, 30.0),
            Point::new(100.0, 30.0),
            &Paint::stroke(Color::RED, 20.0).with_stroke_cap(Cap::Butt),
        );
        // Square cap line from x=40 to x=100 at y=70 (width 20 -> extends x=30..110)
        canvas.draw_line(
            Point::new(40.0, 70.0),
            Point::new(100.0, 70.0),
            &Paint::stroke(Color::GREEN, 20.0).with_stroke_cap(Cap::Square),
        );

        let rgba = canvas.render_to_pixels(&renderer).expect("render strokes");
        assert_eq!(get_pixel(&rgba, 140, 70, 30), [255, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 140, 34, 30), [0, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 140, 70, 70), [0, 255, 0, 255]);
    }

    /// C++ source: entity_unittests.cc:StrokeCapAndJoinTest
    #[test]
    fn test_stroke_cap_and_join() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(160, 120, Color::BLACK);
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(30.0, 90.0))
            .line_to(Point::new(80.0, 30.0))
            .line_to(Point::new(130.0, 90.0));
        let path = pb.build(FillType::NonZero);

        let paint = Paint::stroke(Color::RED, 16.0)
            .with_stroke_cap(Cap::Round)
            .with_stroke_join(Join::Miter)
            .with_stroke_miter(4.0);
        canvas.draw_path(&path, &paint);

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render stroke cap and join");
        // Midpoint of first leg (55, 60) should be red
        assert_eq!(get_pixel(&rgba, 160, 55, 60), [255, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 160, 105, 60), [255, 0, 0, 255]);
    }

    /// C++ source: entity_unittests.cc:ThreeStrokesInOnePath
    #[test]
    fn test_three_strokes_in_one_path() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(120, 140, Color::BLACK);
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(60.0, 10.0))
            .line_to(Point::new(60.0, 40.0))
            .move_to(Point::new(60.0, 55.0))
            .line_to(Point::new(60.0, 85.0))
            .move_to(Point::new(60.0, 100.0))
            .line_to(Point::new(60.0, 130.0));

        canvas.draw_path(
            &pb.build(FillType::NonZero),
            &Paint::stroke(Color::RED, 10.0),
        );
        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render three strokes");
        assert_eq!(get_pixel(&rgba, 120, 60, 25), [255, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 120, 60, 47), [0, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 120, 60, 70), [255, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 120, 60, 92), [0, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 120, 60, 115), [255, 0, 0, 255]);
    }

    /// C++ source: aiks_unittests.cc:PaintBlendModeIsRespected
    #[test]
    fn test_paint_blend_mode_is_respected() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(160, 120, Color::BLACK);

        let mut dl = DisplayListBuilder::new();
        dl.draw_circle(
            Point::new(65.0, 60.0),
            35.0,
            Paint::fill(Color::RED).with_blend_mode(BlendMode::Plus),
        )
        .draw_circle(
            Point::new(95.0, 60.0),
            35.0,
            Paint::fill(Color::GREEN).with_blend_mode(BlendMode::Plus),
        );
        canvas.draw_display_list(&dl.build());

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render blend modes");
        // Left only -> RED
        assert_eq!(get_pixel(&rgba, 160, 45, 60), [255, 0, 0, 255]);
        // Overlap at (80, 60) -> RED + GREEN = YELLOW
        assert_eq!(get_pixel(&rgba, 160, 80, 60), [255, 255, 0, 255]);
        // Right only -> GREEN
        assert_eq!(get_pixel(&rgba, 160, 115, 60), [0, 255, 0, 255]);
    }

    /// C++ source: aiks_unittests.cc:CanRenderRRectShadowBlur
    #[test]
    fn test_can_render_rrect_shadow_blur() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(160, 160, Color::BLACK);
        canvas.draw_rrect_shadow(
            Rect::from_xywh(40.0, 40.0, 80.0, 80.0),
            16.0,
            10.0,
            Color::WHITE,
        );
        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render rrect shadow");
        let center = get_pixel(&rgba, 160, 80, 80);
        let border = get_pixel(&rgba, 160, 40, 80);
        let outside = get_pixel(&rgba, 160, 4, 4);
        assert!(
            center[0] > 235,
            "center of shadow should be bright: {:?}",
            center
        );
        assert!(
            border[0] > 60 && border[0] < 200,
            "border should have partial Gaussian falloff: {:?}",
            border
        );
        assert!(
            outside[0] < 10,
            "far outside should be near zero: {:?}",
            outside
        );
    }

    /// C++ source: aiks_unittests.cc:ShowcaseMultiQuadrantRender
    #[test]
    fn test_showcase_multi_quadrant_render() {
        let renderer = shared_test_renderer();
        let canvas = build_showcase_canvas();
        let out_path = std::env::temp_dir().join("impeller_canvas_test_showcase.png");
        let rgba = canvas
            .render_to_png(&renderer, &out_path)
            .expect("render showcase to png");
        let stats = verify_showcase_pixels(&rgba, 800, 600).expect("showcase pixel verification");
        assert_eq!(stats.width, 800);
        assert_eq!(stats.height, 600);
        assert!(out_path.exists());
    }

    /// C++ source: aiks_unittests.cc:CanRenderTextFrame
    #[test]
    fn test_can_render_shaped_text() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(220, 80, Color::BLACK);
        canvas.draw_text(
            "Impeller GPU",
            TextStyle {
                font: BuiltinFont::RobotoBold,
                font_size: 26.0,
                line_height: 30.0,
                color: Color::GREEN,
                max_width: None,
            },
            Point::new(14.0, 20.0),
        );
        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render shaped text");
        let green_pixels = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[1] > 80 && px[0] < 30 && px[2] < 30)
            .count();
        assert!(
            green_pixels > 120,
            "expected >120 green glyph pixels, got {}",
            green_pixels
        );
        assert_eq!(get_pixel(&rgba, 220, 4, 4), [0, 0, 0, 255]);
    }

    /// C++ source: aiks_unittests.cc:CanRenderTransformedAndClippedTextFrame
    #[test]
    fn test_can_render_transformed_and_clipped_text_frame() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(240, 100, Color::BLACK);
        let frame = canvas.shape_text(
            "CLIPPED TEXT",
            TextStyle {
                font: BuiltinFont::RobotoBold,
                font_size: 24.0,
                line_height: 28.0,
                color: Color::WHITE,
                max_width: None,
            },
            Point::ZERO,
        );

        canvas.save();
        // Clip strictly to the left half [0..100, 0..100]
        canvas.clip_rect(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        canvas.translate(12.0, 24.0);
        canvas.scale(1.2, 1.2);
        canvas.draw_text_frame(&frame, Point::ZERO, &Paint::fill(Color::RED));
        canvas.restore();

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render transformed and clipped text frame");

        let mut left_red = 0usize;
        let mut right_red = 0usize;
        for y in 0..100u32 {
            for x in 0..240u32 {
                let px = get_pixel(&rgba, 240, x, y);
                if px[0] > 40 {
                    if x < 100 {
                        left_red += 1;
                    } else {
                        right_red += 1;
                    }
                }
            }
        }
        assert!(
            left_red > 80,
            "expected red glyph pixels inside left clip rect, got {}",
            left_red
        );
        assert_eq!(
            right_red, 0,
            "expected 0 red glyph pixels outside clip rect x >= 100, got {}",
            right_red
        );
    }

    /// C++ source: aiks_unittests.cc:CanRenderMaterialIconGlyphs
    #[test]
    fn test_can_render_material_icon_glyphs() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(160, 80, Color::BLACK);
        let cyan = Color::make_rgba8(0x18, 0xFF, 0xFF, 0xFF);
        let green = Color::make_rgba8(0x00, 0xE6, 0x76, 0xFF);
        canvas.draw_icon(MaterialIcon::Speed, 36.0, cyan, Point::new(20.0, 18.0));
        canvas.draw_icon(
            MaterialIcon::CheckCircle,
            36.0,
            green,
            Point::new(96.0, 18.0),
        );

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render material icons");
        let mut left_icon_px = 0usize;
        let mut right_icon_px = 0usize;
        for y in 0..80u32 {
            for x in 0..160u32 {
                let px = get_pixel(&rgba, 160, x, y);
                if x < 80 && px[1] > 80 && px[2] > 80 {
                    left_icon_px += 1;
                } else if x >= 80 && px[1] > 80 && px[0] < 30 {
                    right_icon_px += 1;
                }
            }
        }
        assert!(
            left_icon_px > 50,
            "expected Speed icon pixels on left half, got {}",
            left_icon_px
        );
        assert!(
            right_icon_px > 50,
            "expected CheckCircle icon pixels on right half, got {}",
            right_icon_px
        );
    }

    /// C++ source: aiks_unittests.cc:CanRenderStrokedGlyphPathOutline
    #[test]
    fn test_can_render_stroked_glyph_path_outline() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(120, 120, Color::BLACK);
        let outline_path =
            canvas.glyph_to_path('O', BuiltinFont::RobotoBold, 72.0, Point::new(22.0, 86.0));
        assert!(
            !outline_path.is_empty(),
            "glyph_to_path('O') should produce non-empty vector contours"
        );

        canvas.draw_glyph_path(
            'O',
            BuiltinFont::RobotoBold,
            72.0,
            Point::new(22.0, 86.0),
            &Paint::stroke(Color::YELLOW, 3.0).with_stroke_join(Join::Round),
        );

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render stroked glyph path");
        let yellow_px = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] > 180 && px[1] > 180 && px[2] < 30)
            .count();
        assert!(
            yellow_px > 60,
            "expected >60 yellow stroked glyph outline pixels, got {}",
            yellow_px
        );
    }

    /// C++ source: aiks_unittests.cc:CanRenderDashedPathWithKurboDash
    #[test]
    fn test_can_render_dashed_path() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(200, 80, Color::BLACK);
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(10.0, 40.0))
            .line_to(Point::new(190.0, 40.0));
        let path = pb.build(FillType::NonZero);

        canvas.draw_dashed_path(
            &path,
            0.0,
            &[20.0, 20.0],
            &Paint::stroke(Color::RED, 12.0).with_stroke_cap(Cap::Butt),
        );

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render dashed path");
        // x=20 is inside first dash [10..30], x=40 is inside first gap [30..50], x=60 is inside second dash [50..70]
        assert_eq!(get_pixel(&rgba, 200, 20, 40), [255, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 200, 40, 40), [0, 0, 0, 255]);
        assert_eq!(get_pixel(&rgba, 200, 60, 40), [255, 0, 0, 255]);
    }

    /// C++ source: aiks_unittests.cc:CanRenderPolymorphicCardSuite
    #[test]
    fn test_can_render_polymorphic_card_suite() {
        let renderer = shared_test_renderer();
        let bg = Color::new(0.05, 0.06, 0.08, 1.0);
        let mut canvas = Canvas::new(320, 270, bg);
        let mut card_rects = Vec::with_capacity(6);

        for idx in 0..6usize {
            let col = (idx % 2) as Scalar;
            let row = (idx / 2) as Scalar;
            let rect = Rect::from_xywh(10.0 + col * 154.0, 10.0 + row * 86.0, 146.0, 80.0);
            card_rects.push(rect);
            canvas.draw_polymorphic_card(idx, 0.15, rect);
        }

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render polymorphic card suite");
        let bg_r = (0.05f32 * 255.0).round() as i32;
        let bg_g = (0.06f32 * 255.0).round() as i32;
        let bg_b = (0.08f32 * 255.0).round() as i32;

        for (idx, rect) in card_rects.iter().enumerate() {
            let x0 = rect.get_left() as u32;
            let y0 = rect.get_top() as u32;
            let x1 = rect.get_right() as u32;
            let y1 = rect.get_bottom() as u32;
            let mut non_bg = 0usize;
            let mut bright_accent = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let px = get_pixel(&rgba, 320, x, y);
                    let diff = (px[0] as i32 - bg_r).abs()
                        + (px[1] as i32 - bg_g).abs()
                        + (px[2] as i32 - bg_b).abs();
                    if diff > 10 {
                        non_bg += 1;
                    }
                    if px[0] > 80 || px[1] > 80 || px[2] > 80 {
                        bright_accent += 1;
                    }
                }
            }
            assert!(
                non_bg > 1000,
                "PolymorphicCard #{} should render card background and content, got {} non-bg pixels",
                idx,
                non_bg
            );
            assert!(
                bright_accent > 40,
                "PolymorphicCard #{} should render bright text/icon/accent pixels, got {}",
                idx,
                bright_accent
            );
        }
    }

    /// C++ source: aiks_unittests.cc:CanRenderMorphingGridWorkload
    #[test]
    fn test_can_render_morphing_grid_workload_64_and_500() {
        let renderer = shared_test_renderer();
        let bg = Color::make_rgba8(0x09, 0x0C, 0x10, 0xFF);
        let mut typographer = TypographerContext::new();

        let mut canvas_64 = Canvas::new(640, 480, bg);
        draw_morphing_grid_workload(
            &mut canvas_64,
            &mut typographer,
            BuiltinFont::RobotoBold,
            BuiltinFont::RobotoMonoBold,
            BuiltinFont::MaterialIcons,
            640.0,
            480.0,
            64,
            0.25,
        );
        assert!(
            !canvas_64.commands().is_empty(),
            "64-node morphing grid should emit draw commands"
        );
        assert!(
            canvas_64.commands().len() <= 20,
            "64-node morphing grid should coalesce draw commands into <= 20 batches, got {}",
            canvas_64.commands().len()
        );
        let rgba_64 = canvas_64
            .render_to_pixels(&renderer)
            .expect("render 64-node morphing grid");
        let non_bg_64 = rgba_64
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| {
                (px[0] as i32 - 0x09).abs()
                    + (px[1] as i32 - 0x0C).abs()
                    + (px[2] as i32 - 0x10).abs()
                    > 10
            })
            .count();
        assert!(
            non_bg_64 > 150_000,
            "expected >150k non-bg pixels for 64-node grid, got {}",
            non_bg_64
        );

        let mut canvas_500 = Canvas::new(640, 480, bg);
        draw_morphing_grid_workload(
            &mut canvas_500,
            &mut typographer,
            BuiltinFont::RobotoBold,
            BuiltinFont::RobotoMonoBold,
            BuiltinFont::MaterialIcons,
            640.0,
            480.0,
            500,
            0.55,
        );
        assert!(
            !canvas_500.commands().is_empty(),
            "500-node morphing grid should emit draw commands"
        );
        assert!(
            canvas_500.commands().len() <= 20,
            "500-node morphing grid should coalesce draw commands into <= 20 batches, got {}",
            canvas_500.commands().len()
        );
        let rgba_500 = canvas_500
            .render_to_pixels(&renderer)
            .expect("render 500-node morphing grid");
        let non_bg_500 = rgba_500
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| {
                (px[0] as i32 - 0x09).abs()
                    + (px[1] as i32 - 0x0C).abs()
                    + (px[2] as i32 - 0x10).abs()
                    > 10
            })
            .count();
        assert!(
            non_bg_500 > 150_000,
            "expected >150k non-bg pixels for 500-node grid, got {}",
            non_bg_500
        );
    }

    /// C++ source: aiks_unittests.cc:CanRenderBouncyLayoutWorkload
    #[test]
    fn test_can_render_bouncy_workload_64_and_100() {
        let renderer = shared_test_renderer();
        let bg = Color::make_rgba8(0x09, 0x0C, 0x10, 0xFF);
        let mut typographer = TypographerContext::new();

        for &node_count in &[64usize, 100usize] {
            let mut canvas = Canvas::new(640, 480, bg);
            draw_bouncy_workload(
                &mut canvas,
                &mut typographer,
                BuiltinFont::RobotoBold,
                BuiltinFont::RobotoMonoBold,
                BuiltinFont::MaterialIcons,
                640.0,
                480.0,
                node_count,
                0.35,
            );
            assert!(
                !canvas.commands().is_empty(),
                "{}-node bouncy workload should emit draw commands",
                node_count
            );
            assert!(
                canvas.commands().len() <= 20,
                "{}-node bouncy workload should coalesce draw commands into <= 20 batches, got {}",
                node_count,
                canvas.commands().len()
            );
            let rgba = canvas
                .render_to_pixels(&renderer)
                .expect("render bouncy workload");
            let non_bg = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|px| {
                    (px[0] as i32 - 0x09).abs()
                        + (px[1] as i32 - 0x0C).abs()
                        + (px[2] as i32 - 0x10).abs()
                        > 10
                })
                .count();
            assert!(
                non_bg > 150_000,
                "expected >150k non-bg pixels for {}-node bouncy layout, got {}",
                node_count,
                non_bg
            );
        }
    }

    /// C++ source: aiks_unittests.cc:CanRenderSaveLayerWithOpacityAndColorFilter
    #[test]
    fn test_can_render_save_layer_with_opacity_and_color_filter() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(100, 100, Color::BLACK);

        let layer_paint = Paint::fill(Color::new(1.0, 1.0, 1.0, 0.5))
            .with_color_filter(ColorFilter::Mode(Color::GREEN, BlendMode::SrcIn));
        canvas.save_layer(
            Some(Rect::from_xywh(10.0, 10.0, 80.0, 80.0)),
            Some(&layer_paint),
        );
        canvas.draw_rect(
            Rect::from_xywh(20.0, 20.0, 60.0, 60.0),
            &Paint::fill(Color::RED),
        );
        canvas.restore();

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render save_layer");
        let center_px = get_pixel(&rgba, 100, 50, 50);
        assert_eq!(center_px[0], 0, "red should be replaced by green filter");
        assert!(
            center_px[1] >= 120 && center_px[1] <= 135,
            "green channel should be ~50% opacity over black, got {:?}",
            center_px
        );
    }

    /// C++ source: aiks_unittests.cc:CanRenderParagraphImageAtlasVerticesAndShadow
    #[test]
    fn test_can_render_paragraph_image_atlas_vertices_and_shadow() {
        let renderer = shared_test_renderer();
        let mut canvas = Canvas::new(240, 180, Color::BLACK);

        // 1. draw_paragraph with multi-span RichTextStyle + underline decoration
        let mut para_style = ParagraphStyle::default();
        para_style.default_text_style.set_font_size(18.0);
        para_style.default_text_style.set_color(Color::WHITE);
        let mut pb = ParagraphBuilder::new(para_style);
        let mut span_style = RichTextStyle::default();
        span_style.set_font_size(18.0);
        span_style.set_color(Color::new(0.0, 1.0, 1.0, 1.0));
        span_style.decoration = 0x1;
        pb.push_style(span_style);
        pb.add_text("Impeller");
        pb.pop();
        let mut para = pb.build();
        para.layout(canvas.typographer_mut(), 200.0);
        canvas.draw_paragraph(&para, Point::new(10.0, 10.0));

        // 2. draw_image_rect & draw_atlas with a 2x2 solid RGBA texture
        let tex = ImageTexture::new(
            77,
            2,
            2,
            vec![
                255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
            ],
        );
        canvas.draw_image_rect(
            &tex,
            Rect::from_xywh(0.0, 0.0, 2.0, 2.0),
            Rect::from_xywh(10.0, 60.0, 40.0, 40.0),
            &Paint::fill(Color::WHITE),
            false,
        );
        canvas.draw_atlas(
            &tex,
            &[[10.0, 0.0, 60.0, 60.0]],
            &[Rect::from_xywh(0.0, 0.0, 2.0, 2.0)],
            None,
            BlendMode::SrcOver,
            None,
            None,
            false,
        );

        // 3. draw_vertices triangle
        canvas.draw_vertices(
            &[
                Point::new(120.0, 60.0),
                Point::new(170.0, 60.0),
                Point::new(145.0, 110.0),
            ],
            None,
            Some(&[Color::GREEN, Color::GREEN, Color::GREEN]),
            &[0, 1, 2],
            BlendMode::Dst,
            &Paint::fill(Color::WHITE),
        );

        // 4. draw_shadow + draw_picture
        let mut shadow_pb = PathBuilder::new();
        shadow_pb.add_rect(Rect::from_xywh(180.0, 60.0, 40.0, 40.0));
        let shadow_path = shadow_pb.build(FillType::NonZero);
        let mut dl_builder = DisplayListBuilder::new();
        dl_builder.draw_shadow(shadow_path, Color::YELLOW, 6.0, true, 1.0);
        let pic = Arc::new(dl_builder.build());
        canvas.draw_picture(&pic);

        let rgba = canvas
            .render_to_pixels(&renderer)
            .expect("render paragraph, image, atlas, vertices, and shadow");
        // Verify draw_image_rect red quad at (30, 80)
        assert_eq!(get_pixel(&rgba, 240, 30, 80), [255, 0, 0, 255]);
        // Verify draw_vertices green triangle at (145, 75)
        let tri_px = get_pixel(&rgba, 240, 145, 75);
        assert!(
            tri_px[1] > 200 && tri_px[0] < 20,
            "expected green triangle pixel, got {:?}",
            tri_px
        );
    }

    /// C++ source: aiks_unittests.cc:PreTessellationClipCullingAndScissorElision
    #[test]
    fn test_pre_tessellation_clip_culling_and_scissor_elision() {
        let mut canvas = Canvas::new(200, 200, Color::BLACK);
        let fill_paint = Paint::fill(Color::RED);
        let stroke_paint = Paint::stroke(Color::GREEN, 4.0);
        let radii = RoundingRadii::from_single_radius(6.0);

        let mut para_style = ParagraphStyle::default();
        para_style.default_text_style.set_font_size(14.0);
        para_style.default_text_style.set_color(Color::WHITE);
        let mut pb = ParagraphBuilder::new(para_style);
        pb.add_text("Clip");
        let mut para = pb.build();
        para.layout(canvas.typographer_mut(), 80.0);

        // 1. Offscreen frustum culling: primitives outside 200x200 emit 0 commands.
        canvas.draw_rect(Rect::from_xywh(250.0, 10.0, 40.0, 40.0), &fill_paint);
        canvas.draw_rrect(Rect::from_xywh(-60.0, 10.0, 40.0, 40.0), radii, &fill_paint);
        canvas.draw_circle(Point::new(100.0, 260.0), 20.0, &fill_paint);
        canvas.draw_paragraph(&para, Point::new(250.0, 250.0));
        assert_eq!(
            canvas.commands().len(),
            0,
            "offscreen primitives must be culled before tessellation"
        );

        // 2. Explicit clip rect [20, 20, 120, 120]:
        let clip_box = Rect::from_xywh(20.0, 20.0, 100.0, 100.0);
        canvas.save();
        canvas.clip_rect(clip_box);

        // 2a. Primitives outside clip_box (still inside canvas) are culled before tessellation.
        canvas.draw_rect(Rect::from_xywh(130.0, 30.0, 40.0, 40.0), &fill_paint);
        canvas.draw_rrect(Rect::from_xywh(130.0, 30.0, 40.0, 40.0), radii, &fill_paint);
        canvas.draw_circle(Point::new(160.0, 60.0), 15.0, &fill_paint);
        canvas.draw_paragraph(&para, Point::new(140.0, 40.0));
        assert_eq!(
            canvas.commands().len(),
            0,
            "clipped-out primitives must emit 0 commands"
        );

        // 2b. Interior primitives strictly inside clip_box elide hardware scissor (clip_rect == None).
        canvas.draw_rect(Rect::from_xywh(30.0, 30.0, 20.0, 20.0), &fill_paint);
        canvas.draw_rect(Rect::from_xywh(60.0, 30.0, 20.0, 20.0), &stroke_paint);
        canvas.draw_rrect(Rect::from_xywh(30.0, 60.0, 30.0, 20.0), radii, &fill_paint);
        canvas.draw_circle(Point::new(95.0, 70.0), 10.0, &fill_paint);
        canvas.draw_paragraph(&para, Point::new(30.0, 90.0));
        assert_eq!(canvas.commands().len(), 5);
        for (idx, cmd) in canvas.commands().iter().enumerate() {
            assert_eq!(
                cmd.clip_rect(),
                None,
                "interior command #{idx} strictly inside clip_box should elide scissor"
            );
        }

        // Verify fast-path draw_rect vertex/index counts (4v/6i fill, 8v/24i miter stroke).
        if let DrawCommand::SolidFill { mesh, .. } = &canvas.commands()[0] {
            assert_eq!(mesh.vertices.len(), 4);
            assert_eq!(mesh.indices.len(), 6);
        } else {
            panic!("expected SolidFill for fast-path filled draw_rect");
        }
        if let DrawCommand::SolidFill { mesh, .. } = &canvas.commands()[1] {
            assert_eq!(mesh.vertices.len(), 8);
            assert_eq!(mesh.indices.len(), 24);
        } else {
            panic!("expected SolidFill for fast-path stroked draw_rect");
        }

        // 2c. Straddling primitives intersecting clip_box boundary keep hardware scissor Some(clip_box).
        canvas.draw_rrect(Rect::from_xywh(10.0, 30.0, 30.0, 20.0), radii, &fill_paint);
        canvas.draw_paragraph(&para, Point::new(18.0, 90.0));
        assert_eq!(canvas.commands().len(), 7);
        assert_eq!(canvas.commands()[5].clip_rect(), Some(clip_box));
        assert_eq!(canvas.commands()[6].clip_rect(), Some(clip_box));

        canvas.restore();
    }

    /// C++ source: aiks_unittests.cc:SinglePassParagraphQuadEmitAndCanvasReset
    #[test]
    fn test_single_pass_paragraph_quad_emit_and_canvas_reset() {
        let mut canvas = Canvas::new(160, 120, Color::BLACK);
        let quads = [
            AtlasGlyphQuad {
                dest_rect: Rect::from_xywh(0.0, 0.0, 12.0, 14.0),
                atlas_rect: Rect::from_xywh(1.0, 1.0, 12.0, 14.0),
                color: Color::WHITE,
                is_color: false,
            },
            AtlasGlyphQuad {
                dest_rect: Rect::from_xywh(200.0, 0.0, 12.0, 14.0),
                atlas_rect: Rect::from_xywh(14.0, 1.0, 12.0, 14.0),
                color: Color::WHITE,
                is_color: false,
            },
        ];

        canvas.save();
        canvas.clip_rect(Rect::from_xywh(10.0, 10.0, 100.0, 80.0));
        canvas.translate(5.0, 5.0);
        canvas.emit_glyph_quads_slice_with_offset(
            &quads,
            Point::new(20.0, 25.0),
            canvas.get_current_transform(),
            BlendMode::SrcOver,
            false,
        );
        assert_eq!(canvas.commands().len(), 1);
        if let DrawCommand::GlyphAtlas(batch) = &canvas.commands()[0] {
            // Only the first quad survives after shifting by (20, 25) + translate(5, 5) -> [25, 30, 37, 44].
            assert_eq!(batch.quads.len(), 1);
            assert_eq!(
                batch.quads[0].dest_rect,
                Rect::from_xywh(20.0, 25.0, 12.0, 14.0)
            );
            // Strictly contained inside clip [10, 10, 110, 90], so scissor is elided.
            assert_eq!(batch.clip_rect, None);
        } else {
            panic!("expected GlyphAtlas command");
        }

        // Push additional commands and stack frames, then verify Canvas::reset retains capacities.
        for i in 0..8 {
            canvas.draw_rect(
                Rect::from_xywh(12.0 + i as Scalar * 4.0, 12.0, 3.0, 3.0),
                &Paint::fill(Color::RED),
            );
        }
        let cmd_cap_before = canvas.commands.capacity();
        let stack_cap_before = canvas.stack.capacity();
        assert!(cmd_cap_before >= 9);
        assert!(stack_cap_before >= 2);

        canvas.reset(320, 240, Color::BLUE);
        assert_eq!(canvas.width(), 320);
        assert_eq!(canvas.height(), 240);
        assert_eq!(canvas.clear_color(), Color::BLUE);
        assert_eq!(canvas.commands().len(), 0);
        assert_eq!(canvas.get_save_count(), 1);
        assert_eq!(canvas.get_current_transform(), Matrix::IDENTITY);
        assert_eq!(canvas.get_current_clip(), None);
        assert!(canvas.commands.capacity() >= cmd_cap_before);
        assert!(canvas.stack.capacity() >= stack_cap_before);
    }

    /// C++ source: aiks_unittests.cc:CachedPathStrokedCircleAnalyticArcAndParagraphAabb
    #[test]
    fn test_phase9_cached_path_stroked_circle_analytic_arc_and_paragraph_aabb() {
        let mut canvas = Canvas::new(240, 200, Color::BLACK);

        // 1. Default glyph atlases + paragraph-level AABB fast-path:
        let mut para_style = ParagraphStyle::default();
        para_style.default_text_style.set_font_size(14.0);
        para_style.default_text_style.set_color(Color::WHITE);
        let mut pb = ParagraphBuilder::new(para_style);
        pb.add_text("Fast");
        let mut para = pb.build();
        let mut typo = TypographerContext::new();
        para.layout(&mut typo, 100.0);
        assert!(para.quad_bounds.is_some());
        assert!(para.uses_alpha);

        let alpha_arc = Arc::new(typo.alpha_atlas.descriptor_without_pixels());
        let color_arc = Arc::new(typo.color_atlas.descriptor_without_pixels());
        canvas
            .set_default_glyph_atlases(Some(Arc::clone(&alpha_arc)), Some(Arc::clone(&color_arc)));

        canvas.save();
        canvas.clip_rect(Rect::from_xywh(10.0, 10.0, 120.0, 80.0));
        canvas.draw_paragraph(&para, Point::new(20.0, 20.0));
        // Offscreen paragraph culled in O(1) via quad_bounds:
        canvas.draw_paragraph(&para, Point::new(150.0, 20.0));
        canvas.restore();

        assert_eq!(canvas.commands().len(), 1);
        if let DrawCommand::GlyphAtlas(batch) = &canvas.commands()[0] {
            assert_eq!(
                batch.clip_rect, None,
                "interior paragraph should elide scissor"
            );
            assert!(
                batch
                    .alpha_atlas
                    .as_ref()
                    .is_some_and(|a| Arc::ptr_eq(a, &alpha_arc)),
                "paragraph fast-path must pre-bind default_alpha_atlas"
            );
        } else {
            panic!("expected GlyphAtlas command");
        }

        // 2. Origin-centered stroked circle + analytic arc + cached path:
        canvas.draw_circle(
            Point::new(60.0, 140.0),
            20.0,
            &Paint::stroke(Color::GREEN, 3.0),
        );
        canvas.draw_arc(
            Rect::from_xywh(100.0, 120.0, 40.0, 40.0),
            Radians(0.0),
            Radians(std::f32::consts::PI),
            true,
            &Paint::fill(Color::RED),
        );
        let mut path_pb = PathBuilder::new();
        path_pb.move_to(Point::new(160.0, 120.0));
        path_pb.line_to(Point::new(200.0, 120.0));
        path_pb.line_to(Point::new(180.0, 160.0));
        path_pb.close();
        let tri_path = path_pb.build(FillType::NonZero);
        assert!(tri_path.cache_id() != 0);
        canvas.draw_path(&tri_path, &Paint::fill(Color::BLUE));
        canvas.draw_path(&tri_path, &Paint::fill(Color::BLUE));
        assert_eq!(canvas.commands().len(), 5);

        // Verify origin-centered circle transform maps Point::ZERO to (60, 140):
        if let DrawCommand::SolidFill { transform, .. } = &canvas.commands()[1] {
            let p = transform.transform_point(Point::ZERO);
            assert!((p.x - 60.0).abs() <= 1e-4);
            assert!((p.y - 140.0).abs() <= 1e-4);
        } else {
            panic!("expected SolidFill for stroked circle");
        }
    }

    /// C++ source: aiks_unittests.cc:DiffRoundRectFastPathAndCanvasReplayStats
    #[test]
    fn test_phase10_diff_rrect_fast_path_and_replay_stats() {
        let mut builder = DisplayListBuilder::new();
        let outer = Rect::from_xywh(20.0, 30.0, 120.0, 80.0);
        let inner = Rect::from_xywh(24.0, 34.0, 112.0, 72.0);
        // 1. Nested rounded border -> analytic DiffRoundRectGeometry fast-path:
        builder.draw_diff_rrect(
            outer,
            RoundingRadii::from_single_radius(12.0),
            inner,
            RoundingRadii::from_single_radius(8.0),
            Paint::fill(Color::RED),
        );
        // 2. Non-nested / poking corner -> PSLG fallback:
        builder.draw_diff_rrect(
            outer,
            RoundingRadii::from_single_radius(36.0),
            Rect::from_xywh(21.0, 31.0, 118.0, 78.0),
            RoundingRadii::from_single_radius(0.0),
            Paint::fill(Color::BLUE),
        );
        // 3. RRect + Path + SaveLayer + Restore:
        builder.save_layer(Some(outer), None);
        builder.draw_rrect(
            outer,
            RoundingRadii::from_single_radius(10.0),
            Paint::fill(Color::GREEN),
        );
        let mut pb = PathBuilder::new();
        pb.add_rect(Rect::from_xywh(40.0, 50.0, 30.0, 20.0));
        builder.draw_path(pb.build(FillType::NonZero), Paint::fill(Color::WHITE));
        builder.restore();

        let dl = builder.build();
        let mut canvas = Canvas::new(240, 200, Color::BLACK);
        canvas.draw_display_list(&dl);

        let stats = canvas.replay_stats();
        assert_eq!(stats.op_count, 6);
        assert_eq!(stats.drrect_count, 2);
        assert_eq!(stats.rrect_count, 1);
        assert_eq!(stats.path_count, 1);
        assert_eq!(stats.save_layer_count, 1);
        assert_eq!(stats.paragraph_count, 0);
        assert_eq!(stats.picture_count, 0);

        // First command is the origin-shifted analytic DiffRoundRectGeometry SolidFill:
        if let DrawCommand::SolidFill { transform, .. } = &canvas.commands()[0] {
            let p = transform.transform_point(Point::ZERO);
            assert!((p.x - 20.0).abs() <= 1e-4);
            assert!((p.y - 30.0).abs() <= 1e-4);
        } else {
            panic!("expected SolidFill for analytic DiffRoundRectGeometry");
        }

        // Reset clears replay_stats:
        canvas.reset(240, 200, Color::BLACK);
        assert_eq!(canvas.replay_stats(), CanvasReplayStats::default());
    }

    /// C++ source: aiks_unittests.cc:OneHitPathMeshFilterAndSubOpReplayStats
    #[test]
    fn test_phase11_one_hit_path_filter_and_sub_op_telemetry() {
        GEOMETRY_MESH_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            cache.paths.clear();
            cache.seen_path_ids.clear();
        });

        // 1. Convex polygon path -> 1-hit seen_path_ids filter on first draw, cached on second:
        let mut pb_convex = PathBuilder::new();
        pb_convex.move_to(Point::new(10.0, 10.0));
        pb_convex.line_to(Point::new(60.0, 10.0));
        pb_convex.line_to(Point::new(35.0, 50.0));
        pb_convex.close();
        let convex_path = pb_convex.build(FillType::NonZero);
        let cid = convex_path.cache_id();
        assert_ne!(cid, 0);

        let mut canvas = Canvas::new(200, 200, Color::BLACK);
        canvas.draw_path(&convex_path, &Paint::fill(Color::RED));

        GEOMETRY_MESH_CACHE.with(|cell| {
            let cache = cell.borrow();
            assert!(
                cache.seen_path_ids.contains(&cid),
                "first draw must record cid in seen_path_ids"
            );
            assert!(
                !cache.paths.keys().any(|k| k.path_cache_id == cid),
                "first draw must NOT insert 1-hit animated path into cache.paths"
            );
        });

        // Second draw of the same Path caches it in cache.paths:
        canvas.draw_path(&convex_path, &Paint::fill(Color::RED));
        GEOMETRY_MESH_CACHE.with(|cell| {
            let cache = cell.borrow();
            assert!(
                cache.paths.keys().any(|k| k.path_cache_id == cid),
                "second draw of static path must insert mesh into cache.paths"
            );
        });

        // 2. Concave / self-intersecting path -> PslgGeneral:
        let mut pb_concave = PathBuilder::new();
        pb_concave.move_to(Point::new(10.0, 70.0));
        pb_concave.line_to(Point::new(70.0, 70.0));
        pb_concave.line_to(Point::new(40.0, 90.0));
        pb_concave.line_to(Point::new(70.0, 110.0));
        pb_concave.line_to(Point::new(10.0, 110.0));
        pb_concave.close();
        let concave_path = pb_concave.build(FillType::NonZero);
        canvas.draw_path(&concave_path, &Paint::fill(Color::GREEN));

        // 3. Stroked path:
        canvas.draw_path(&convex_path, &Paint::stroke(Color::BLUE, 3.0));

        // 4. Culled draws outside 200x200 viewport:
        canvas.draw_rect(
            Rect::from_xywh(500.0, 500.0, 40.0, 40.0),
            &Paint::fill(Color::WHITE),
        );
        let mut pb_offscreen = PathBuilder::new();
        pb_offscreen.add_rect(Rect::from_xywh(-200.0, -200.0, 50.0, 50.0));
        canvas.draw_path(
            &pb_offscreen.build(FillType::NonZero),
            &Paint::fill(Color::WHITE),
        );

        let stats = canvas.replay_stats();
        assert_eq!(stats.path_fill_count, 3);
        assert_eq!(stats.path_convex_fast_count, 2);
        assert_eq!(stats.path_pslg_count, 1);
        assert_eq!(stats.path_stroke_count, 1);
        assert_eq!(stats.culled_count, 2);
    }

    /// C++ source: aiks_unittests.cc:StencilClippingSweepConicalAndNineSliceImage
    #[test]
    fn test_phase12a_stencil_clipping_sweep_conical_and_draw_image_nine() {
        let mut canvas = Canvas::new(200, 200, Color::BLACK);

        // 1. Zero-radius clip_rrect must use scissor fast-path (0 stencil commands):
        canvas.save();
        canvas.clip_rrect(
            Rect::from_xywh(10.0, 10.0, 180.0, 180.0),
            RoundingRadii::ZERO,
            ClipOperation::Intersect,
        );
        assert!(
            canvas.commands().is_empty(),
            "zero-radius clip_rrect must emit 0 stencil commands"
        );
        canvas.restore();
        assert!(canvas.commands().is_empty());

        // 2. Rounded clip_rrect + clip_path must emit PushClip and LIFO PopClip on restore:
        canvas.save();
        canvas.clip_rrect(
            Rect::from_xywh(20.0, 20.0, 140.0, 140.0),
            RoundingRadii::from_single_radius(16.0),
            ClipOperation::Intersect,
        );
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(30.0, 30.0));
        pb.line_to(Point::new(150.0, 30.0));
        pb.line_to(Point::new(90.0, 150.0));
        pb.close();
        canvas.clip_path(&pb.build(FillType::NonZero), ClipOperation::Difference);

        let sweep = SweepGradientParams::two_stop(
            Point::new(90.0, 90.0),
            0.0,
            360.0,
            Color::RED,
            Color::BLUE,
            TileMode::Clamp,
        );
        canvas.draw_rect(
            Rect::from_xywh(20.0, 20.0, 140.0, 140.0),
            &Paint::fill(Color::WHITE).with_color_source(ColorSource::SweepGradient(sweep)),
        );
        let conical = ConicalGradientParams::two_stop(
            Point::new(60.0, 60.0),
            4.0,
            Point::new(110.0, 110.0),
            40.0,
            Color::GREEN,
            Color::WHITE,
            TileMode::Clamp,
        );
        canvas.draw_rect(
            Rect::from_xywh(40.0, 40.0, 80.0, 80.0),
            &Paint::fill(Color::WHITE).with_color_source(ColorSource::ConicalGradient(conical)),
        );
        canvas.restore();

        let cmds = canvas.commands();
        assert_eq!(cmds.len(), 6);
        assert!(matches!(
            cmds[0],
            DrawCommand::PushClip {
                is_difference: false,
                ..
            }
        ));
        assert!(matches!(
            cmds[1],
            DrawCommand::PushClip {
                is_difference: true,
                ..
            }
        ));
        assert!(matches!(cmds[2], DrawCommand::SweepGradientFill { .. }));
        assert!(matches!(cmds[3], DrawCommand::ConicalGradientFill { .. }));
        assert!(matches!(cmds[4], DrawCommand::PopClip { .. }));
        assert!(matches!(cmds[5], DrawCommand::PopClip { .. }));

        // 3. 9-slice draw_image_nine splits 30x30 image with [10,10,20,20] center into 9 quads:
        canvas.reset(200, 200, Color::BLACK);
        let img = ImageTexture::new(1, 30, 30, vec![255u8; 30 * 30 * 4]);
        canvas.draw_image_nine(
            &img,
            Rect::from_ltrb(10.0, 10.0, 20.0, 20.0),
            Rect::from_xywh(10.0, 20.0, 100.0, 80.0),
            &Paint::fill(Color::WHITE),
            true,
        );
        assert_eq!(canvas.commands().len(), 9);
        // Top-left corner (index 0) must remain unscaled 10x10 at (10, 20):
        if let DrawCommand::TextureFill(batch) = &canvas.commands()[0] {
            assert_relative_eq!(batch.vertices[0].position[0], 10.0, epsilon = 1e-4);
            assert_relative_eq!(batch.vertices[0].position[1], 20.0, epsilon = 1e-4);
            assert_relative_eq!(batch.vertices[2].position[0], 20.0, epsilon = 1e-4);
            assert_relative_eq!(batch.vertices[2].position[1], 30.0, epsilon = 1e-4);
        } else {
            panic!("expected TextureFill for 9-slice corner");
        }
        // Center patch (index 4) must stretch to [20..100, 30..90] (80x60):
        if let DrawCommand::TextureFill(batch) = &canvas.commands()[4] {
            assert_relative_eq!(batch.vertices[0].position[0], 20.0, epsilon = 1e-4);
            assert_relative_eq!(batch.vertices[0].position[1], 30.0, epsilon = 1e-4);
            assert_relative_eq!(batch.vertices[2].position[0], 100.0, epsilon = 1e-4);
            assert_relative_eq!(batch.vertices[2].position[1], 90.0, epsilon = 1e-4);
        } else {
            panic!("expected TextureFill for 9-slice center");
        }
    }

    /// C++ source: aiks_unittests.cc:CanCollapseIntoParentPassVsSaveLayerOffscreen
    #[test]
    fn test_phase12b_save_layer_collapse_vs_offscreen_emission() {
        let renderer = shared_test_renderer();

        // 1. Single-child save_layer with 50% opacity and no ImageFilter must collapse into parent pass:
        let mut canvas = Canvas::new(100, 100, Color::BLACK);
        let half_alpha_paint = Paint::fill(Color::new(1.0, 1.0, 1.0, 0.5));
        canvas.save_layer(
            Some(Rect::from_xywh(10.0, 10.0, 80.0, 80.0)),
            Some(&half_alpha_paint),
        );
        canvas.draw_rect(
            Rect::from_xywh(20.0, 20.0, 60.0, 60.0),
            &Paint::fill(Color::RED),
        );
        canvas.restore();
        assert_eq!(canvas.commands().len(), 1);
        assert!(
            matches!(canvas.commands()[0], DrawCommand::SolidFill { .. }),
            "single-child opacity save_layer must collapse into SolidFill without SaveLayer"
        );

        // 2. Multi-child save_layer with 50% opacity must emit offscreen DrawCommand::SaveLayer
        //    so overlapping children do NOT double-blend:
        let mut group_canvas = Canvas::new(120, 100, Color::BLACK);
        group_canvas.save_layer(
            Some(Rect::from_xywh(10.0, 10.0, 100.0, 80.0)),
            Some(&half_alpha_paint),
        );
        group_canvas.draw_rect(
            Rect::from_xywh(20.0, 20.0, 50.0, 50.0),
            &Paint::fill(Color::RED),
        );
        group_canvas.draw_rect(
            Rect::from_xywh(40.0, 20.0, 50.0, 50.0),
            &Paint::fill(Color::RED),
        );
        group_canvas.restore();

        assert_eq!(group_canvas.commands().len(), 1);
        assert!(
            matches!(group_canvas.commands()[0], DrawCommand::SaveLayer { .. }),
            "multi-child opacity save_layer must emit offscreen DrawCommand::SaveLayer"
        );

        let rgba = group_canvas
            .render_to_pixels(&renderer)
            .expect("render multi-child save_layer group opacity");
        let left_px = get_pixel(&rgba, 120, 30, 45);
        let overlap_px = get_pixel(&rgba, 120, 55, 45);
        assert!(
            left_px[0] >= 120 && left_px[0] <= 135,
            "left non-overlap should be ~50% red, got {:?}",
            left_px
        );
        assert!(
            (overlap_px[0] as i32 - left_px[0] as i32).abs() <= 2,
            "overlap region must match group opacity without double-blending: left={:?}, overlap={:?}",
            left_px,
            overlap_px
        );
    }

    /// C++ source: aiks_image_filter_unittests.cc:CanRenderBlurImageFilterAndTextureColorFilter
    #[test]
    fn test_phase12b_blur_image_filter_and_texture_color_filter() {
        let renderer = shared_test_renderer();

        // 1. Standalone draw_rect with ImageFilter::Blur wraps into SaveLayer and renders Gaussian halo:
        let mut blur_canvas = Canvas::new(120, 120, Color::BLACK);
        let blur_paint = Paint::fill(Color::WHITE).with_image_filter(ImageFilter::Blur {
            sigma_x: 6.0,
            sigma_y: 6.0,
            tile_mode: TileMode::Decal,
        });
        blur_canvas.draw_rect(Rect::from_xywh(40.0, 40.0, 40.0, 40.0), &blur_paint);
        assert_eq!(blur_canvas.commands().len(), 1);
        assert!(matches!(
            blur_canvas.commands()[0],
            DrawCommand::SaveLayer {
                image_filter: Some(LayerImageFilter::Blur { .. }),
                ..
            }
        ));

        let blur_rgba = blur_canvas
            .render_to_pixels(&renderer)
            .expect("render blurred rect");
        let center_px = get_pixel(&blur_rgba, 120, 60, 60);
        let halo_px = get_pixel(&blur_rgba, 120, 36, 60);
        let corner_px = get_pixel(&blur_rgba, 120, 4, 4);
        assert!(
            center_px[0] > 210,
            "interior of blurred white rect should be bright, got {:?}",
            center_px
        );
        assert!(
            halo_px[0] > 25 && halo_px[0] < 190,
            "outside border at x=36 should have Gaussian blur halo falloff, got {:?}",
            halo_px
        );
        assert!(
            corner_px[0] < 5,
            "far corner should remain black, got {:?}",
            corner_px
        );

        // 2. draw_image_rect with ColorFilter::Mode wraps into SaveLayer with LayerColorFilter:
        let mut tex_canvas = Canvas::new(100, 100, Color::BLACK);
        let red_tex = ImageTexture::new(
            99,
            2,
            2,
            vec![
                255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
            ],
        );
        let cf_paint = Paint::fill(Color::WHITE)
            .with_color_filter(ColorFilter::Mode(Color::GREEN, BlendMode::SrcIn));
        tex_canvas.draw_image_rect(
            &red_tex,
            Rect::from_xywh(0.0, 0.0, 2.0, 2.0),
            Rect::from_xywh(20.0, 20.0, 60.0, 60.0),
            &cf_paint,
            false,
        );
        assert_eq!(tex_canvas.commands().len(), 1);
        assert!(matches!(
            tex_canvas.commands()[0],
            DrawCommand::SaveLayer {
                color_filter: Some(LayerColorFilter::Mode { .. }),
                ..
            }
        ));

        let tex_rgba = tex_canvas
            .render_to_pixels(&renderer)
            .expect("render color-filtered texture");
        let filtered_px = get_pixel(&tex_rgba, 100, 50, 50);
        assert_eq!(
            filtered_px[0], 0,
            "red texture channel must be filtered out"
        );
        assert!(
            filtered_px[1] > 245,
            "green channel must be full intensity after SrcIn(GREEN) filter, got {:?}",
            filtered_px
        );
    }
}
