#![forbid(unsafe_code)]

pub use impeller_geometry::{
    BlendMode, Color, Matrix, Point, Radians, Rect, Scalar, Size, TileMode, K_EH_CLOSE_ENOUGH,
};
pub use impeller_path::{Cap, FillType, Join, Path, PathBuilder, PathComponent, RoundingRadii};
use impeller_tessellator::{
    compute_stroke_alpha_coverage, CircleGeometry, FillPathGeometry, LineGeometry,
    RoundRectGeometry, StrokeParameters, StrokePathGeometry, TessellatedMesh, Vertex2D,
};
pub use impeller_typographer::{
    AtlasGlyphQuad, BuiltinFont, GlyphAtlas, GlyphAtlasType, MaterialIcon, SubpixelBin, TextFrame,
    TextStyle, TypographerContext,
};
pub use impeller_wgpu::{
    compare_golden_rgba, save_rgba_png, DrawCommand, GlyphAtlasBatch, GoldenDiffMetrics,
    GradientStop, LinearGradientParams, RRectBlurParams, RadialGradientParams, WgpuRenderer,
    DEFAULT_TEXT_CONTRAST,
};
use std::path::Path as FsPath;
use std::sync::Arc;

pub use impeller_wgpu::WgpuRenderer as WgpuContext;

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

#[derive(Clone, Debug, PartialEq)]
pub enum PaintShader {
    Solid(Color),
    LinearGradient(LinearGradientParams),
    RadialGradient(RadialGradientParams),
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
        if self.invert_colors {
            Color::new(
                (1.0 - base.red).clamp(0.0, 1.0),
                (1.0 - base.green).clamp(0.0, 1.0),
                (1.0 - base.blue).clamp(0.0, 1.0),
                base.alpha,
            )
        } else {
            base
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
            && self.mask_blur_sigma <= K_EH_CLOSE_ENOUGH
    }

    pub fn is_shadow_blur_draw_operation(&self) -> bool {
        if self.style != DrawStyle::Fill {
            return false;
        }
        if matches!(
            self.effective_color_source(),
            Some(ColorSource::LinearGradient(_) | ColorSource::RadialGradient(_))
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
    Restore,
    RestoreToCount(usize),
    Translate(Scalar, Scalar),
    Scale(Scalar, Scalar),
    Rotate(Radians),
    Skew(Scalar, Scalar),
    Transform(Matrix),
    ResetTransform,
    ClipRect(Rect, ClipOperation),
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
    DrawTextFrame(TextFrame, Point, Paint),
    DrawText(String, TextStyle, Point),
    DrawIcon(MaterialIcon, Scalar, Color, Point),
    DrawGlyphPath(char, BuiltinFont, Scalar, Point, Paint),
    DrawPolymorphicCard(usize, Scalar, Rect),
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
        for op in &self.ops {
            match op {
                DisplayListOp::Save => canvas.save(),
                DisplayListOp::Restore => {
                    canvas.restore();
                }
                DisplayListOp::RestoreToCount(c) => canvas.restore_to_count(*c),
                DisplayListOp::Translate(tx, ty) => canvas.translate(*tx, *ty),
                DisplayListOp::Scale(sx, sy) => canvas.scale(*sx, *sy),
                DisplayListOp::Rotate(r) => canvas.rotate(*r),
                DisplayListOp::Skew(sx, sy) => canvas.skew(*sx, *sy),
                DisplayListOp::Transform(m) => canvas.transform(*m),
                DisplayListOp::ResetTransform => canvas.reset_transform(),
                DisplayListOp::ClipRect(rect, op) => canvas.clip_rect_with_op(*rect, *op),
                DisplayListOp::ClipPath(path, op) => canvas.clip_path(path, *op),
                DisplayListOp::DrawPaint(paint) => canvas.draw_paint(paint),
                DisplayListOp::DrawColor(color, mode) => canvas.draw_color(*color, *mode),
                DisplayListOp::DrawRect(rect, paint) => canvas.draw_rect(*rect, paint),
                DisplayListOp::DrawRRect(rect, radii, paint) => {
                    canvas.draw_rrect(*rect, *radii, paint)
                }
                DisplayListOp::DrawDiffRRect(or, orad, ir, irad, paint) => {
                    canvas.draw_diff_rrect(*or, *orad, *ir, *irad, paint)
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
                DisplayListOp::DrawPath(path, paint) => canvas.draw_path(path, paint),
                DisplayListOp::DrawRRectShadow(rect, radius, sigma, color) => {
                    canvas.draw_rrect_shadow(*rect, *radius, *sigma, *color)
                }
                DisplayListOp::DrawTextFrame(frame, origin, paint) => {
                    canvas.draw_text_frame(frame, *origin, paint)
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
        self.ops
            .push(DisplayListOp::DrawArc(oval_bounds, start, sweep, use_center, paint));
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
        self.ops
            .push(DisplayListOp::DrawDashedLine(p0, p1, on_length, off_length, paint));
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
        self.ops
            .push(DisplayListOp::DrawRRectShadow(rect, corner_radius, blur_sigma, color));
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

    pub fn draw_text(&mut self, text: impl Into<String>, style: TextStyle, origin: Point) -> &mut Self {
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
        self.ops
            .push(DisplayListOp::DrawGlyphPath(ch, font, font_size, origin, paint));
        self
    }

    pub fn draw_polymorphic_card(
        &mut self,
        index: usize,
        animation_value: Scalar,
        rect: Rect,
    ) -> &mut Self {
        self.ops
            .push(DisplayListOp::DrawPolymorphicCard(index, animation_value, rect));
        self
    }

    pub fn build(&self) -> DisplayList {
        DisplayList {
            ops: self.ops.clone(),
        }
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
    let min_x = corners.iter().map(|p| p.x).fold(Scalar::INFINITY, Scalar::min);
    let min_y = corners.iter().map(|p| p.y).fold(Scalar::INFINITY, Scalar::min);
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
    glyph_batches: Vec<PendingGlyphBatch>,
    cached_alpha_atlas: Option<(u64, Arc<GlyphAtlas>)>,
    cached_color_atlas: Option<(u64, Arc<GlyphAtlas>)>,
}

impl Canvas {
    pub fn new(width: u32, height: u32, clear_color: Color) -> Self {
        let cull_rect = Rect::from_xywh(0.0, 0.0, width as Scalar, height as Scalar);
        Self::with_cull_rect(width, height, clear_color, cull_rect)
    }

    pub fn with_cull_rect(
        width: u32,
        height: u32,
        clear_color: Color,
        cull_rect: Rect,
    ) -> Self {
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
            }],
            commands: Vec::new(),
            typographer: None,
            glyph_batches: Vec::new(),
            cached_alpha_atlas: None,
            cached_color_atlas: None,
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

    fn sync_glyph_atlases(&mut self) {
        let Some(ref typographer) = self.typographer else {
            return;
        };
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

        for cmd in &mut self.commands {
            if let DrawCommand::GlyphAtlas(batch) = cmd {
                if batch.quads.iter().any(|q| !q.is_color) {
                    batch.alpha_atlas = Some(Arc::clone(&alpha_arc));
                }
                if batch.quads.iter().any(|q| q.is_color) {
                    batch.color_atlas = Some(Arc::clone(&color_arc));
                }
            }
        }
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
        if let Some(top) = self.stack.last().cloned() {
            self.stack.push(top);
        }
    }

    pub fn restore(&mut self) -> bool {
        if self.stack.len() > 1 {
            self.stack.pop();
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

    pub fn clip_rect(&mut self, rect: Rect) {
        self.clip_rect_with_op(rect, ClipOperation::Intersect);
    }

    pub fn clip_rect_with_op(&mut self, rect: Rect, op: ClipOperation) {
        let tx = self.get_current_transform();
        let transformed = transform_rect(&tx, rect);
        self.apply_device_clip_rect(transformed, op);
    }

    pub fn clip_path(&mut self, path: &Path, op: ClipOperation) {
        let tx = self.get_current_transform();
        let transformed = path
            .get_transformed_bounding_box(&tx)
            .unwrap_or(Rect::ZERO);
        self.apply_device_clip_rect(transformed, op);
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

    fn emit_mesh_with_transform(&mut self, mesh: TessellatedMesh, transform: Matrix, paint: &Paint) {
        self.emit_mesh_with_transform_and_coverage(mesh, transform, paint, 1.0);
    }

    fn emit_mesh_with_transform_and_coverage(
        &mut self,
        mesh: TessellatedMesh,
        transform: Matrix,
        paint: &Paint,
        coverage: Scalar,
    ) {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return;
        }
        let cov = coverage.clamp(0.0, 1.0);
        if cov <= 0.0 {
            return;
        }
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

        for clip_rect in clip_targets {
            match paint.effective_color_source() {
                Some(ColorSource::LinearGradient(grad)) => {
                    let mut normalized = grad.clone();
                    normalized.stops = Paint::convert_stops(&grad.stops);
                    normalized.alpha = (normalized.alpha * cov).clamp(0.0, 1.0);
                    self.commands.push(DrawCommand::LinearGradientFill {
                        mesh: mesh.clone(),
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
                        mesh: mesh.clone(),
                        transform,
                        gradient: normalized,
                        blend_mode: paint.blend_mode,
                        clip_rect,
                    });
                }
                Some(ColorSource::Solid(_)) | None => {
                    let mut color = paint.effective_solid_color();
                    color.alpha = (color.alpha * cov).clamp(0.0, 1.0);
                    self.commands.push(DrawCommand::SolidFill {
                        mesh: mesh.clone(),
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
        let transform = self.get_current_transform();
        let scale = transform.get_max_basis_length_xy().max(1e-4);
        match paint.style {
            DrawStyle::Fill => {
                let mesh = FillPathGeometry::new(path.clone()).get_mesh(scale.max(1.0));
                self.emit_mesh(mesh, paint);
            }
            DrawStyle::Stroke => {
                let width = paint.effective_stroke_width();
                let cov = compute_stroke_alpha_coverage(&transform, width);
                let mesh = StrokePathGeometry::new(
                    path.clone(),
                    StrokeParameters {
                        width,
                        cap: paint.stroke_cap,
                        join: paint.stroke_join,
                        miter_limit: paint.stroke_miter,
                    },
                )
                .get_mesh(scale);
                self.emit_mesh_with_transform_and_coverage(mesh, transform, paint, cov);
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
        let mut pb = PathBuilder::new();
        pb.add_rect(rect);
        self.draw_path(&pb.build(FillType::NonZero), paint);
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
        let scale = transform.get_max_basis_length_xy().max(1e-4);
        match paint.style {
            DrawStyle::Fill => {
                let mesh = RoundRectGeometry::new(rect, radii).get_mesh(scale.max(1.0));
                self.emit_mesh(mesh, paint);
            }
            DrawStyle::Stroke => {
                let width = paint.effective_stroke_width();
                let cov = compute_stroke_alpha_coverage(&transform, width);
                let stroke = StrokeParameters {
                    width: width.max(1.0 / scale),
                    cap: paint.stroke_cap,
                    join: paint.stroke_join,
                    miter_limit: paint.stroke_miter,
                };
                let mesh = RoundRectGeometry::new(rect, radii).get_stroked_mesh(stroke, scale);
                self.emit_mesh_with_transform_and_coverage(mesh, transform, paint, cov);
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
        let mut pb = PathBuilder::new();
        pb.add_oval(rect);
        self.draw_path(&pb.build(FillType::NonZero), paint);
    }

    pub fn draw_circle(&mut self, center: Point, radius: Scalar, paint: &Paint) {
        if radius <= 0.0 {
            return;
        }
        if paint.is_shadow_blur_draw_operation() {
            let rect = Rect::from_xywh(
                center.x - radius,
                center.y - radius,
                radius * 2.0,
                radius * 2.0,
            );
            self.draw_rrect_shadow_with_blend(
                rect,
                radius,
                paint.mask_blur_sigma,
                paint.effective_solid_color(),
                paint.blend_mode,
            );
            return;
        }
        let scale = self.get_current_transform().get_max_basis_length_xy().max(1.0);
        match paint.style {
            DrawStyle::Fill => {
                let mesh = CircleGeometry::new(center, radius).get_mesh(scale);
                self.emit_mesh(mesh, paint);
            }
            DrawStyle::Stroke => {
                let mut pb = PathBuilder::new();
                pb.add_circle(center, radius);
                self.draw_path(&pb.build(FillType::NonZero), paint);
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
        let mut pb = PathBuilder::new();
        pb.add_arc(oval_bounds, start, sweep, use_center);
        if paint.style == DrawStyle::Fill && !use_center {
            pb.close();
        }
        self.draw_path(&pb.build(FillType::NonZero), paint);
    }

    pub fn draw_line(&mut self, p0: Point, p1: Point, paint: &Paint) {
        let transform = self.get_current_transform();
        let scale = transform.get_max_basis_length_xy().max(1e-4);
        let width = paint.effective_stroke_width();
        let cov = compute_stroke_alpha_coverage(&transform, width);
        let mesh = LineGeometry::new(p0, p1, width, paint.stroke_cap).get_mesh(scale);
        self.emit_mesh_with_transform_and_coverage(mesh, transform, paint, cov);
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
            let seg_mesh = LineGeometry::new(seg_p0, seg_p1, width, paint.stroke_cap).get_mesh(scale);
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
                    let r = Rect::from_xywh(pt.x - radius, pt.y - radius, radius * 2.0, radius * 2.0);
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
        self.draw_rrect_shadow_with_blend(rect, corner_radius, blur_sigma, color, BlendMode::SrcOver);
    }

    pub fn draw_rrect_shadow_with_blend(
        &mut self,
        rect: Rect,
        corner_radius: Scalar,
        blur_sigma: Scalar,
        color: Color,
        blend_mode: BlendMode,
    ) {
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
        for clip_rect in clip_targets {
            self.commands.push(DrawCommand::RRectBlur {
                transform,
                params: RRectBlurParams {
                    rect,
                    corner_radius,
                    blur_sigma,
                    color,
                },
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
        let clip_targets: Vec<Option<Rect>> = if top.has_explicit_clip {
            top.clip_rects.iter().copied().map(Some).collect()
        } else {
            vec![None]
        };

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

        for clip_rect in clip_targets {
            let command_index = self.commands.len();
            self.glyph_batches.push(PendingGlyphBatch {
                command_index,
                transform,
                quads: adjusted_quads.clone(),
                blend_mode: paint.blend_mode,
                clip_rect,
            });
            self.commands.push(DrawCommand::GlyphAtlas(
                GlyphAtlasBatch::new(
                    adjusted_quads.clone(),
                    transform,
                    None,
                    None,
                    paint.blend_mode,
                    clip_rect,
                )
                .with_linear_filter(use_linear_filter),
            ));
        }
        self.sync_glyph_atlases();
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
                self.draw_icon(MaterialIcon::ShowChart, 14.0, orange, Point::new(118.0, 7.0));

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

                self.draw_circle(
                    Point::new(22.0, 40.0),
                    12.0,
                    &Paint::stroke(track, 3.2),
                );
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
        &self.glyph_batches
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
            let angle = -std::f32::consts::FRAC_PI_2
                + (idx as Scalar) * 2.0 * std::f32::consts::PI / 5.0;
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
    canvas.draw_line(Point::new(52.0, 342.0), Point::new(52.0, 438.0), &rail_paint);
    canvas.draw_line(Point::new(168.0, 342.0), Point::new(168.0, 438.0), &rail_paint);

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
    dash_curve_pb
        .move_to(Point::new(36.0, 562.0))
        .cubic_to(
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
    canvas
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

                let diff = (r as i32 - bg_r).abs()
                    + (g as i32 - bg_g).abs()
                    + (b as i32 - bg_b).abs();
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
                std::sync::Mutex::new(
                    WgpuRenderer::new_headless().expect("headless wgpu renderer"),
                )
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
        let stroke = paint.get_stroke().expect("stroke parameters when DrawStyle::Stroke");
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
        let rgba = canvas.render_to_pixels(&renderer).expect("render colored rect");
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

        let rgba = canvas.render_to_pixels(&renderer).expect("render draw_paint");
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
        assert!(left[0] > 230 && left[2] < 25, "left clamped to red: {:?}", left);
        assert!(mid[0] > 90 && mid[2] > 90, "mid interpolated purple: {:?}", mid);
        assert!(right[2] > 230 && right[0] < 25, "right clamped to blue: {:?}", right);
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
        let cull = canvas.get_current_local_cull_rect().expect("non-empty clip");
        assert_relative_eq!(cull.get_left(), 50.0, epsilon = 1e-4);
        assert_relative_eq!(cull.get_top(), 50.0, epsilon = 1e-4);
        assert_relative_eq!(cull.get_right(), 90.0, epsilon = 1e-4);
        assert_relative_eq!(cull.get_bottom(), 90.0, epsilon = 1e-4);

        canvas.draw_rect(
            Rect::from_xywh(0.0, 0.0, 120.0, 120.0),
            &Paint::fill(Color::GREEN),
        );
        let rgba = canvas.render_to_pixels(&renderer).expect("render nested clips");
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
        let rgba = canvas.render_to_pixels(&renderer).expect("render clip restore");
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

        canvas.draw_path(&pb.build(FillType::NonZero), &Paint::stroke(Color::RED, 10.0));
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

        let rgba = canvas.render_to_pixels(&renderer).expect("render blend modes");
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
        let rgba = canvas.render_to_pixels(&renderer).expect("render rrect shadow");
        let center = get_pixel(&rgba, 160, 80, 80);
        let border = get_pixel(&rgba, 160, 40, 80);
        let outside = get_pixel(&rgba, 160, 4, 4);
        assert!(center[0] > 235, "center of shadow should be bright: {:?}", center);
        assert!(
            border[0] > 60 && border[0] < 200,
            "border should have partial Gaussian falloff: {:?}",
            border
        );
        assert!(outside[0] < 10, "far outside should be near zero: {:?}", outside);
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
            .as_chunks::<4>().0.iter()
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
            .as_chunks::<4>().0.iter()
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
}
