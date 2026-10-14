//! Rich-text `ParagraphBuilder` and `Paragraph` layout engine (`impeller_typographer::paragraph`),
//! supporting multi-span `RichTextStyle`, horizontal alignment, line metrics, UTF-16 code unit
//! queries, inline `PlaceholderSpan`s, background/decoration rects, and `cosmic-text` shaping.

use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;

use cosmic_text::{
    Attrs, BidiParagraphs, Buffer, CacheKeyFlags, Family, Metrics, Shaping, Stretch, Style, Weight,
    Wrap,
};
use impeller_geometry::{Color, Point, Rect, Scalar};

use crate::{
    AtlasGlyphQuad, BuiltinFont, FontGlyphPair, ScaledFont, SubpixelPosition, TypographerContext,
};

/// Horizontal paragraph alignment matching Flutter `ui.TextAlign` / Skwasm `paragraphStyle_setTextAlign`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextAlign {
    #[default]
    Left = 0,
    Right = 1,
    Center = 2,
    Justify = 3,
    Start = 4,
    End = 5,
}

impl TextAlign {
    pub const fn from_skwasm(value: i32) -> Self {
        match value {
            1 => Self::Right,
            2 => Self::Center,
            3 => Self::Justify,
            4 => Self::Start,
            5 => Self::End,
            _ => Self::Left,
        }
    }

    pub const fn to_skwasm(self) -> i32 {
        self as i32
    }

    pub const fn resolve(self, direction: TextDirection) -> Self {
        match self {
            Self::Start => match direction {
                TextDirection::Ltr => Self::Left,
                TextDirection::Rtl => Self::Right,
            },
            Self::End => match direction {
                TextDirection::Ltr => Self::Right,
                TextDirection::Rtl => Self::Left,
            },
            other => other,
        }
    }
}

/// Paragraph and text span directionality matching Flutter `ui.TextDirection` / Skwasm (`0 = Rtl, 1 = Ltr`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextDirection {
    Rtl = 0,
    #[default]
    Ltr = 1,
}

impl TextDirection {
    pub const fn from_skwasm(value: i32) -> Self {
        if value == 0 {
            Self::Rtl
        } else {
            Self::Ltr
        }
    }

    pub const fn to_skwasm(self) -> i32 {
        match self {
            Self::Rtl => 0,
            Self::Ltr => 1,
        }
    }
}

/// Drop shadow attached to a [`RichTextStyle`] (`textStyle_addShadow`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextShadow {
    pub color: Color,
    pub offset: Point,
    pub blur_sigma: Scalar,
}

/// OpenType font feature tag and value (`textStyle_addFontFeature`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FontFeature {
    pub tag: String,
    pub value: i32,
}

/// OpenType variable font axis tag and value (`textStyle_setFontVariations`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FontVariation {
    pub tag: u32,
    pub value: Scalar,
}

/// Rich text span styling (`SkwasmTextStyle` / Flutter `ui.TextStyle`).
#[derive(Debug, Clone, PartialEq)]
pub struct RichTextStyle {
    pub color: Color,
    pub has_color: bool,
    pub foreground_paint_color: Option<Color>,
    pub background_paint_color: Option<Color>,
    /// Bitmask: `0x1` = Underline, `0x2` = Overline, `0x4` = LineThrough.
    pub decoration: i32,
    pub decoration_color: Option<Color>,
    pub decoration_style: i32,
    pub decoration_thickness: Scalar,
    /// CSS weight `100..=900` (`400` normal, `700` bold).
    pub font_weight: u16,
    pub has_font_weight: bool,
    /// `0` = Upright, `1` = Italic, `2` = Oblique.
    pub font_slant: i32,
    pub has_font_slant: bool,
    pub text_baseline: i32,
    pub font_families: Vec<String>,
    pub font_size: Scalar,
    pub has_font_size: bool,
    pub letter_spacing: Scalar,
    pub word_spacing: Scalar,
    pub height_multiplier: Option<Scalar>,
    pub half_leading: bool,
    pub locale: Option<String>,
    pub shadows: Vec<TextShadow>,
    pub font_features: Vec<FontFeature>,
    pub font_variations: Vec<FontVariation>,
}

impl Default for RichTextStyle {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            has_color: false,
            foreground_paint_color: None,
            background_paint_color: None,
            decoration: 0,
            decoration_color: None,
            decoration_style: 0,
            decoration_thickness: 1.0,
            font_weight: 400,
            has_font_weight: false,
            font_slant: 0,
            has_font_slant: false,
            text_baseline: 0,
            font_families: Vec::new(),
            font_size: 14.0,
            has_font_size: false,
            letter_spacing: 0.0,
            word_spacing: 0.0,
            height_multiplier: None,
            half_leading: false,
            locale: None,
            shadows: Vec::new(),
            font_features: Vec::new(),
            font_variations: Vec::new(),
        }
    }
}

impl RichTextStyle {
    pub fn set_color(&mut self, color: Color) {
        self.color = color;
        self.has_color = true;
    }

    pub fn set_font_size(&mut self, size: Scalar) {
        self.font_size = size.max(1.0);
        self.has_font_size = true;
    }

    /// Maps Flutter `ui.FontWeight.index` (`0..=8` -> `100..=900`) or direct CSS weight (`100..=900`)
    /// and `ui.FontStyle.index` (`0` = Upright, `1` = Italic).
    pub fn set_flutter_font_style(&mut self, weight_index_or_css: i32, slant: i32) {
        let weight = if (0..=8).contains(&weight_index_or_css) {
            ((weight_index_or_css + 1) * 100) as u16
        } else {
            weight_index_or_css.clamp(100, 900) as u16
        };
        self.font_weight = weight;
        self.has_font_weight = true;
        self.font_slant = slant;
        self.has_font_slant = true;
    }

    /// Merges unset properties from `parent` so nested Flutter `TextSpan`s inherit parent properties.
    pub fn merged_with_parent(&self, parent: &Self) -> Self {
        let mut out = self.clone();
        if !out.has_color
            && out.foreground_paint_color.is_none()
            && out.color == Color::WHITE
            && (parent.has_color
                || parent.foreground_paint_color.is_some()
                || parent.color != Color::WHITE)
        {
            out.color = parent.color;
            out.has_color = parent.has_color;
            out.foreground_paint_color = parent.foreground_paint_color;
        }
        if !out.has_font_size && (out.font_size - 14.0).abs() < 1e-4 {
            out.font_size = parent.font_size;
            out.has_font_size = parent.has_font_size;
        }
        if !out.has_font_weight && out.font_weight == 400 {
            out.font_weight = parent.font_weight;
            out.has_font_weight = parent.has_font_weight;
        }
        if !out.has_font_slant && out.font_slant == 0 {
            out.font_slant = parent.font_slant;
            out.has_font_slant = parent.has_font_slant;
        }
        if out.font_families.is_empty() {
            out.font_families.clone_from(&parent.font_families);
        }
        if out.height_multiplier.is_none() {
            out.height_multiplier = parent.height_multiplier;
        }
        if out.background_paint_color.is_none() {
            out.background_paint_color = parent.background_paint_color;
        }
        if out.decoration == 0 && parent.decoration != 0 {
            out.decoration = parent.decoration;
            out.decoration_color = parent.decoration_color;
            out.decoration_style = parent.decoration_style;
            out.decoration_thickness = parent.decoration_thickness;
        }
        if out.shadows.is_empty() && !parent.shadows.is_empty() {
            out.shadows.clone_from(&parent.shadows);
        }
        out
    }
}

/// Paragraph strut configuration (`SkwasmStrutStyle` / Flutter `ui.StrutStyle`).
#[derive(Debug, Clone, PartialEq)]
pub struct StrutStyle {
    pub font_families: Vec<String>,
    pub font_size: Option<Scalar>,
    pub height_multiplier: Option<Scalar>,
    pub half_leading: bool,
    pub leading: Option<Scalar>,
    pub font_weight: u16,
    pub font_slant: i32,
    pub force_strut_height: bool,
}

impl Default for StrutStyle {
    fn default() -> Self {
        Self {
            font_families: Vec::new(),
            font_size: None,
            height_multiplier: None,
            half_leading: false,
            leading: None,
            font_weight: 400,
            font_slant: 0,
            force_strut_height: false,
        }
    }
}

impl StrutStyle {
    pub fn set_flutter_font_style(&mut self, weight_index_or_css: i32, slant: i32) {
        self.font_weight = if (0..=8).contains(&weight_index_or_css) {
            ((weight_index_or_css + 1) * 100) as u16
        } else {
            weight_index_or_css.clamp(100, 900) as u16
        };
        self.font_slant = slant;
    }
}

/// Paragraph-level layout configuration (`SkwasmParagraphStyle` / Flutter `ui.ParagraphStyle`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParagraphStyle {
    pub text_align: TextAlign,
    pub text_direction: TextDirection,
    pub height_multiplier: Option<Scalar>,
    pub apply_first_ascent: bool,
    pub apply_last_descent: bool,
    pub max_lines: Option<usize>,
    pub ellipsis: Option<String>,
    pub strut_style: Option<StrutStyle>,
    pub default_text_style: RichTextStyle,
    pub apply_rounding_hack: bool,
}

impl Default for ParagraphStyle {
    fn default() -> Self {
        Self {
            text_align: TextAlign::Left,
            text_direction: TextDirection::Ltr,
            height_multiplier: None,
            apply_first_ascent: true,
            apply_last_descent: true,
            max_lines: None,
            ellipsis: None,
            strut_style: None,
            default_text_style: RichTextStyle::default(),
            apply_rounding_hack: true,
        }
    }
}

/// Inline widget placeholder span (`paragraphBuilder_addPlaceholder`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaceholderSpan {
    pub width: Scalar,
    pub height: Scalar,
    /// `0` = Baseline, `1` = AboveBaseline, `2` = BelowBaseline, `3` = Top, `4` = Bottom, `5` = Middle.
    pub alignment: i32,
    pub baseline_offset: Scalar,
    pub baseline: i32,
}

/// Per-line layout metrics (`SkwasmLineMetrics` / Flutter `ui.LineMetrics`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParagraphLineMetrics {
    pub hard_break: bool,
    /// Positive distance from baseline to top of line (`ascent >= 0.0`).
    pub ascent: Scalar,
    /// Positive distance from baseline to bottom of line (`descent >= 0.0`).
    pub descent: Scalar,
    pub unscaled_ascent: Scalar,
    pub height: Scalar,
    pub width: Scalar,
    pub left: Scalar,
    /// Vertical offset from top of paragraph (`y = 0`) to this line's baseline.
    pub baseline: Scalar,
    pub line_number: usize,
    /// Inclusive start UTF-16 code unit index.
    pub start_index: usize,
    /// Exclusive end UTF-16 code unit index (excluding trailing newline).
    pub end_index: usize,
    /// Exclusive end UTF-16 code unit index excluding trailing whitespace.
    pub end_excluding_whitespaces: usize,
    /// Exclusive end UTF-16 code unit index including trailing newline if present.
    pub end_including_newline: usize,
}

/// Bounding box and text direction for a character range or placeholder (`SkwasmTextBoxList`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextBox {
    pub rect: Rect,
    pub direction: TextDirection,
}

/// Grapheme cluster metrics returned by `get_glyph_info_at` and `get_closest_glyph_info_at`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphInfo {
    pub grapheme_cluster_bounds: Rect,
    pub grapheme_cluster_code_unit_range_start: usize,
    pub grapheme_cluster_code_unit_range_end: usize,
    pub is_ltr: bool,
}

/// Recorded span inside a [`ParagraphBuilder`].
#[derive(Debug, Clone, PartialEq)]
pub struct RichTextSpan {
    pub utf8_range: Range<usize>,
    pub utf16_range: Range<usize>,
    pub style: RichTextStyle,
}

/// Recorded inline placeholder inside a [`ParagraphBuilder`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaceholderEntry {
    pub utf8_offset: usize,
    pub utf16_offset: usize,
    pub span: PlaceholderSpan,
    pub text_direction: TextDirection,
}

/// Rich-text paragraph builder (`SkwasmParagraphBuilder` / Flutter `ui.ParagraphBuilder`).
#[derive(Debug, Clone)]
pub struct ParagraphBuilder {
    pub paragraph_style: ParagraphStyle,
    style_stack: Vec<RichTextStyle>,
    text_utf8: String,
    text_utf16: Vec<u16>,
    spans: Vec<RichTextSpan>,
    placeholders: Vec<PlaceholderEntry>,
    grapheme_breaks_utf16: Vec<u32>,
    word_breaks_utf16: Vec<u32>,
    line_breaks_utf16: Vec<i32>,
}

impl ParagraphBuilder {
    pub fn new(paragraph_style: ParagraphStyle) -> Self {
        let root_style = paragraph_style.default_text_style.clone();
        Self {
            paragraph_style,
            style_stack: vec![root_style],
            text_utf8: String::new(),
            text_utf16: Vec::new(),
            spans: Vec::new(),
            placeholders: Vec::new(),
            grapheme_breaks_utf16: Vec::new(),
            word_breaks_utf16: Vec::new(),
            line_breaks_utf16: Vec::new(),
        }
    }

    pub fn current_style(&self) -> &RichTextStyle {
        self.style_stack
            .last()
            .unwrap_or(&self.paragraph_style.default_text_style)
    }

    pub fn push_style(&mut self, style: RichTextStyle) {
        let merged = style.merged_with_parent(self.current_style());
        self.style_stack.push(merged);
    }

    pub fn pop(&mut self) {
        if self.style_stack.len() > 1 {
            self.style_stack.pop();
        }
    }

    pub fn add_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let utf8_start = self.text_utf8.len();
        let utf16_start = self.text_utf16.len();
        self.text_utf8.push_str(text);
        self.text_utf16.extend(text.encode_utf16());
        let utf8_end = self.text_utf8.len();
        let utf16_end = self.text_utf16.len();

        let style = self.current_style().clone();
        self.spans.push(RichTextSpan {
            utf8_range: utf8_start..utf8_end,
            utf16_range: utf16_start..utf16_end,
            style,
        });
    }

    pub fn add_text_utf16(&mut self, utf16_units: &[u16]) {
        if utf16_units.is_empty() {
            return;
        }
        let decoded = String::from_utf16_lossy(utf16_units);
        self.add_text(&decoded);
    }

    pub fn add_placeholder(&mut self, placeholder: PlaceholderSpan) {
        let utf8_offset = self.text_utf8.len();
        let utf16_offset = self.text_utf16.len();
        self.placeholders.push(PlaceholderEntry {
            utf8_offset,
            utf16_offset,
            span: placeholder,
            text_direction: self.paragraph_style.text_direction,
        });
        let mut ph_style = self.current_style().clone();
        ph_style.set_font_size(placeholder.height.max(1.0));
        let utf8_start = self.text_utf8.len();
        let utf16_start = self.text_utf16.len();
        self.text_utf8.push('\u{FFFC}');
        self.text_utf16.push(0xFFFC);
        self.spans.push(RichTextSpan {
            utf8_range: utf8_start..self.text_utf8.len(),
            utf16_range: utf16_start..self.text_utf16.len(),
            style: ph_style,
        });
    }

    pub fn utf8_text(&self) -> &str {
        &self.text_utf8
    }

    pub fn utf16_units(&self) -> &[u16] {
        &self.text_utf16
    }

    pub fn set_grapheme_breaks_utf16(&mut self, breaks: Vec<u32>) {
        self.grapheme_breaks_utf16 = breaks;
    }

    pub fn set_word_breaks_utf16(&mut self, breaks: Vec<u32>) {
        self.word_breaks_utf16 = breaks;
    }

    pub fn set_line_breaks_utf16(&mut self, breaks: Vec<i32>) {
        self.line_breaks_utf16 = breaks;
    }

    pub fn build(self) -> Paragraph {
        let utf8_to_utf16 = build_utf8_to_utf16_map(&self.text_utf8);
        Paragraph {
            paragraph_style: self.paragraph_style,
            text_utf8: self.text_utf8,
            text_utf16: self.text_utf16,
            utf8_to_utf16,
            spans: self.spans,
            placeholders: self.placeholders,
            quads: Vec::new(),
            background_rects: Vec::new(),
            decoration_rects: Vec::new(),
            lines: Vec::new(),
            placeholder_boxes: Vec::new(),
            width: 0.0,
            height: 0.0,
            longest_line: 0.0,
            min_intrinsic_width: 0.0,
            max_intrinsic_width: 0.0,
            alphabetic_baseline: 0.0,
            ideographic_baseline: 0.0,
            did_exceed_max_lines: false,
            quad_bounds: None,
            uses_alpha: false,
            uses_color: false,
            has_laid_out: false,
            laid_out_epoch: 0,
            last_layout_finite: false,
            glyph_clusters: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PositionedGlyphCluster {
    bounds: Rect,
    utf16_start: usize,
    utf16_end: usize,
    line_number: usize,
    is_ltr: bool,
}

const MAX_PARAGRAPH_LAYOUT_CACHE_ENTRIES: usize = 2048;
const MAX_RESOLVED_FONT_CACHE_ENTRIES: usize = 512;

/// Cached layout result for a shaped [`Paragraph`] stored inside [`TypographerContext`].
#[derive(Debug, Clone)]
pub struct CachedParagraphLayout {
    pub quads: Vec<AtlasGlyphQuad>,
    pub background_rects: Vec<(Rect, Color)>,
    pub decoration_rects: Vec<(Rect, Color)>,
    pub lines: Vec<ParagraphLineMetrics>,
    pub placeholder_boxes: Vec<(Rect, TextDirection)>,
    pub(crate) glyph_clusters: Vec<PositionedGlyphCluster>,
    pub width: Scalar,
    pub height: Scalar,
    pub longest_line: Scalar,
    pub min_intrinsic_width: Scalar,
    pub max_intrinsic_width: Scalar,
    pub alphabetic_baseline: Scalar,
    pub ideographic_baseline: Scalar,
    pub did_exceed_max_lines: bool,
    pub quad_bounds: Option<Rect>,
    pub uses_alpha: bool,
    pub uses_color: bool,
}

struct DeterministicHasher(u64);

impl DeterministicHasher {
    const SEED: u64 = 0x517cc1b727220a95;

    #[inline]
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    #[inline]
    fn mix(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(Self::SEED);
    }
}

impl Hasher for DeterministicHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rem) = bytes.as_chunks::<8>();
        for &chunk in chunks {
            self.mix(u64::from_le_bytes(chunk));
        }
        if !rem.is_empty() {
            let mut tail = [0u8; 8];
            tail[..rem.len()].copy_from_slice(rem);
            self.mix(u64::from_le_bytes(tail));
        }
        self.mix(bytes.len() as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.mix(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.mix(u64::from(i));
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.mix(u64::from(i));
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.mix(i as u32 as u64);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.mix(i as u64);
    }
}

fn hash_color(h: &mut DeterministicHasher, c: Color) {
    c.red.to_bits().hash(h);
    c.green.to_bits().hash(h);
    c.blue.to_bits().hash(h);
    c.alpha.to_bits().hash(h);
}

fn hash_opt_color(h: &mut DeterministicHasher, oc: Option<Color>) {
    match oc {
        Some(c) => {
            1u8.hash(h);
            hash_color(h, c);
        }
        None => 0u8.hash(h),
    }
}

fn hash_rich_text_style(h: &mut DeterministicHasher, s: &RichTextStyle) {
    hash_color(h, s.color);
    s.has_color.hash(h);
    hash_opt_color(h, s.foreground_paint_color);
    hash_opt_color(h, s.background_paint_color);
    s.decoration.hash(h);
    hash_opt_color(h, s.decoration_color);
    s.decoration_style.hash(h);
    s.decoration_thickness.to_bits().hash(h);
    s.font_weight.hash(h);
    s.font_slant.hash(h);
    s.font_size.to_bits().hash(h);
    s.height_multiplier.map(f32::to_bits).hash(h);
    s.font_families.hash(h);
    s.letter_spacing.to_bits().hash(h);
    s.word_spacing.to_bits().hash(h);
    s.half_leading.hash(h);
    s.locale.hash(h);
    s.shadows.len().hash(h);
    for sh in &s.shadows {
        hash_color(h, sh.color);
        sh.offset.x.to_bits().hash(h);
        sh.offset.y.to_bits().hash(h);
        sh.blur_sigma.to_bits().hash(h);
    }
}

fn build_utf8_to_utf16_map(s: &str) -> Vec<usize> {
    let mut map = vec![0usize; s.len() + 1];
    let mut utf16_pos = 0usize;
    for (byte_idx, ch) in s.char_indices() {
        let len_utf8 = ch.len_utf8();
        let len_utf16 = ch.len_utf16();
        for offset in 0..len_utf8 {
            map[byte_idx + offset] = utf16_pos;
        }
        utf16_pos += len_utf16;
    }
    map[s.len()] = utf16_pos;
    map
}

fn resolve_span_line_height(style: &RichTextStyle, para_style: &ParagraphStyle) -> Scalar {
    let font_size = style.font_size.max(1.0);
    let mut mult = style.height_multiplier.or(para_style.height_multiplier);
    if let Some(strut) = &para_style.strut_style {
        if strut.force_strut_height || mult.is_none() {
            if let Some(sm) = strut.height_multiplier {
                mult = Some(sm);
            }
        }
    }
    match mult {
        Some(m) => (font_size * m.max(0.5)).max(font_size * 1.15),
        None => font_size * 1.2,
    }
}

fn resolve_family_name(ctx: &TypographerContext, families: &[String], weight: u16) -> String {
    for fam in families {
        let lower = fam.to_ascii_lowercase();
        let short = fam.rsplit('/').next().unwrap_or(fam.as_str());
        let short_lower = short.to_ascii_lowercase();

        if lower.contains("icons") || short_lower.contains("icons") {
            if let Some(found) = ctx.font_system.db().faces().find(|f| {
                f.families.iter().any(|(n, _)| {
                    n.eq_ignore_ascii_case(fam) || n.eq_ignore_ascii_case(short)
                })
            }) {
                if let Some((name, _)) = found.families.first() {
                    return name.clone();
                }
            }
            return ctx.fonts[3].family.clone();
        }

        if lower == "monospace"
            || lower == "robotomono"
            || lower == "roboto mono"
            || lower.contains("dejavu")
            || lower.contains("courier")
            || lower.contains("mono")
        {
            return ctx.fonts[2].family.clone();
        }

        if let Some(found) = ctx.font_system.db().faces().find(|f| {
            f.families
                .iter()
                .any(|(n, _)| n.eq_ignore_ascii_case(fam) || n.eq_ignore_ascii_case(short))
        }) {
            if let Some((name, _)) = found.families.first() {
                return name.clone();
            }
        }
    }

    if weight >= 600 {
        ctx.fonts[1].family.clone()
    } else {
        ctx.fonts[0].family.clone()
    }
}

fn resolve_slant_and_flags(
    ctx: &TypographerContext,
    resolved_family: &str,
    font_slant: i32,
) -> (Style, CacheKeyFlags) {
    let requested_style = match font_slant {
        1 => Style::Italic,
        2 => Style::Oblique,
        _ => return (Style::Normal, CacheKeyFlags::empty()),
    };

    let has_matching_face = ctx.font_system.db().faces().any(|face| {
        face.style == requested_style
            && face
                .families
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case(resolved_family))
    });

    if has_matching_face {
        (requested_style, CacheKeyFlags::empty())
    } else {
        (Style::Normal, CacheKeyFlags::FAKE_ITALIC)
    }
}

fn resolve_family_and_slant_cached(
    ctx: &mut TypographerContext,
    families: &[String],
    weight: u16,
    font_slant: i32,
) -> (String, Style, CacheKeyFlags) {
    let key = (families.to_vec(), weight, font_slant);
    if let Some(cached) = ctx.resolved_font_cache.get(&key) {
        return cached.clone();
    }
    let fam = resolve_family_name(ctx, families, weight);
    let (style, flags) = resolve_slant_and_flags(ctx, &fam, font_slant);
    let val = (fam, style, flags);
    if ctx.resolved_font_cache.len() >= MAX_RESOLVED_FONT_CACHE_ENTRIES {
        ctx.resolved_font_cache.clear();
    }
    ctx.resolved_font_cache.insert(key, val.clone());
    val
}

/// Shaped, laid-out rich-text paragraph (`SkwasmParagraph` / Flutter `ui.Paragraph`).
#[derive(Debug, Clone)]
pub struct Paragraph {
    pub paragraph_style: ParagraphStyle,
    pub text_utf8: String,
    pub text_utf16: Vec<u16>,
    utf8_to_utf16: Vec<usize>,
    pub spans: Vec<RichTextSpan>,
    pub placeholders: Vec<PlaceholderEntry>,
    /// Positioned, tinted glyph quads ready for `GlyphAtlasBatch` rendering.
    pub quads: Vec<AtlasGlyphQuad>,
    /// Background and text decoration rectangles `(rect, color)` in paragraph local space.
    pub background_rects: Vec<(Rect, Color)>,
    /// Subset of `background_rects` specifically representing underline/overline/line-through decorations.
    pub decoration_rects: Vec<(Rect, Color)>,
    pub lines: Vec<ParagraphLineMetrics>,
    pub placeholder_boxes: Vec<(Rect, TextDirection)>,
    pub width: Scalar,
    pub height: Scalar,
    pub longest_line: Scalar,
    pub min_intrinsic_width: Scalar,
    pub max_intrinsic_width: Scalar,
    pub alphabetic_baseline: Scalar,
    pub ideographic_baseline: Scalar,
    pub did_exceed_max_lines: bool,
    pub quad_bounds: Option<Rect>,
    pub uses_alpha: bool,
    pub uses_color: bool,
    pub has_laid_out: bool,
    laid_out_epoch: u64,
    last_layout_finite: bool,
    glyph_clusters: Vec<PositionedGlyphCluster>,
}

impl Paragraph {
    fn compute_paragraph_layout_hash(&self, effective_width: Scalar, finite_width: bool) -> u64 {
        let mut h = DeterministicHasher::new();
        self.text_utf8.hash(&mut h);
        finite_width.hash(&mut h);
        ((effective_width * 16.0).round() as i64).hash(&mut h);

        self.spans.len().hash(&mut h);
        for sp in &self.spans {
            sp.utf8_range.start.hash(&mut h);
            sp.utf8_range.end.hash(&mut h);
            hash_rich_text_style(&mut h, &sp.style);
        }

        self.placeholders.len().hash(&mut h);
        for ph in &self.placeholders {
            ph.utf8_offset.hash(&mut h);
            ph.utf16_offset.hash(&mut h);
            ph.span.width.to_bits().hash(&mut h);
            ph.span.height.to_bits().hash(&mut h);
            ph.span.alignment.hash(&mut h);
            ph.span.baseline_offset.to_bits().hash(&mut h);
            ph.span.baseline.hash(&mut h);
            ph.text_direction.hash(&mut h);
        }

        self.paragraph_style.text_align.hash(&mut h);
        self.paragraph_style.text_direction.hash(&mut h);
        self.paragraph_style
            .height_multiplier
            .map(f32::to_bits)
            .hash(&mut h);
        self.paragraph_style.apply_first_ascent.hash(&mut h);
        self.paragraph_style.apply_last_descent.hash(&mut h);
        self.paragraph_style.max_lines.hash(&mut h);
        self.paragraph_style.ellipsis.hash(&mut h);
        self.paragraph_style.apply_rounding_hack.hash(&mut h);

        match &self.paragraph_style.strut_style {
            Some(st) => {
                1u8.hash(&mut h);
                st.font_families.hash(&mut h);
                st.font_size.map(f32::to_bits).hash(&mut h);
                st.height_multiplier.map(f32::to_bits).hash(&mut h);
                st.half_leading.hash(&mut h);
                st.leading.map(f32::to_bits).hash(&mut h);
                st.font_weight.hash(&mut h);
                st.font_slant.hash(&mut h);
                st.force_strut_height.hash(&mut h);
            }
            None => 0u8.hash(&mut h),
        }

        hash_rich_text_style(&mut h, &self.paragraph_style.default_text_style);
        h.finish()
    }

    fn apply_cached_layout(&mut self, cached: &CachedParagraphLayout) {
        self.quads.clone_from(&cached.quads);
        self.background_rects.clone_from(&cached.background_rects);
        self.decoration_rects.clone_from(&cached.decoration_rects);
        self.lines.clone_from(&cached.lines);
        self.placeholder_boxes.clone_from(&cached.placeholder_boxes);
        self.glyph_clusters.clone_from(&cached.glyph_clusters);
        self.width = cached.width;
        self.height = cached.height;
        self.longest_line = cached.longest_line;
        self.min_intrinsic_width = cached.min_intrinsic_width;
        self.max_intrinsic_width = cached.max_intrinsic_width;
        self.alphabetic_baseline = cached.alphabetic_baseline;
        self.ideographic_baseline = cached.ideographic_baseline;
        self.did_exceed_max_lines = cached.did_exceed_max_lines;
        self.quad_bounds = cached.quad_bounds;
        self.uses_alpha = cached.uses_alpha;
        self.uses_color = cached.uses_color;
    }

    /// Shapes and lays out the paragraph at `max_width` using `ctx`, rasterizing all visible
    /// glyphs into `ctx.alpha_atlas` / `ctx.color_atlas` and computing all line, box, and UTF-16 metrics.
    pub fn layout(&mut self, ctx: &mut TypographerContext, max_width: Scalar) {
        let finite_width = max_width.is_finite() && max_width >= 0.0;
        let effective_width = if finite_width { max_width } else { 0.0 };

        if self.has_laid_out
            && self.laid_out_epoch == ctx.cache_epoch
            && self.last_layout_finite == finite_width
            && (!finite_width || (self.width - effective_width).abs() < 1e-3)
        {
            return;
        }

        let layout_hash = self.compute_paragraph_layout_hash(effective_width, finite_width);
        if let Some(cached) = ctx.paragraph_layout_cache.get(&layout_hash).cloned() {
            self.apply_cached_layout(&cached);
            self.has_laid_out = true;
            self.laid_out_epoch = ctx.cache_epoch;
            self.last_layout_finite = finite_width;
            return;
        }

        self.quads.clear();
        self.background_rects.clear();
        self.decoration_rects.clear();
        self.lines.clear();
        self.placeholder_boxes.clear();
        self.glyph_clusters.clear();
        self.longest_line = 0.0;
        self.min_intrinsic_width = 0.0;
        self.max_intrinsic_width = 0.0;
        self.did_exceed_max_lines = false;
        self.quad_bounds = None;
        self.uses_alpha = false;
        self.uses_color = false;
        self.width = effective_width;

        let default_style = &self.paragraph_style.default_text_style;
        let default_font_size = default_style.font_size.max(1.0);
        let default_line_h = resolve_span_line_height(default_style, &self.paragraph_style);
        let default_ascent = (default_font_size * 0.93).min(default_line_h);

        if self.text_utf8.is_empty() || self.spans.is_empty() {
            self.height = default_line_h.round();
            self.alphabetic_baseline = default_ascent.round();
            self.ideographic_baseline = self.height;
            self.has_laid_out = true;
            self.laid_out_epoch = ctx.cache_epoch;
            self.last_layout_finite = finite_width;
            return;
        }

        let mut buffer = Buffer::new(
            &mut ctx.font_system,
            Metrics::new(default_font_size, default_line_h),
        );
        if finite_width && max_width > 0.0 {
            buffer.set_wrap(&mut ctx.font_system, Wrap::WordOrGlyph);
            buffer.set_size(&mut ctx.font_system, Some(max_width.max(1.0)), None);
        } else {
            buffer.set_wrap(&mut ctx.font_system, Wrap::None);
            buffer.set_size(&mut ctx.font_system, None, None);
        }

        let mut resolved_families_and_slants = Vec::with_capacity(self.spans.len());
        for span in &self.spans {
            resolved_families_and_slants.push(resolve_family_and_slant_cached(
                ctx,
                &span.style.font_families,
                span.style.font_weight,
                span.style.font_slant,
            ));
        }

        let (default_family, default_font_style, default_flags) = resolve_family_and_slant_cached(
            ctx,
            &default_style.font_families,
            default_style.font_weight,
            default_style.font_slant,
        );
        let default_attrs = Attrs::new()
            .family(Family::Name(&default_family))
            .weight(Weight(default_style.font_weight.clamp(100, 900)))
            .style(default_font_style)
            .cache_key_flags(default_flags)
            .stretch(Stretch::Normal);

        {
            let rich_iter = self.spans.iter().enumerate().map(|(idx, span)| {
                let slice = &self.text_utf8[span.utf8_range.clone()];
                let font_size = span.style.font_size.max(1.0);
                let line_h = resolve_span_line_height(&span.style, &self.paragraph_style);
                let (ref fam, font_style, flags) = resolved_families_and_slants[idx];
                let attrs = Attrs::new()
                    .family(Family::Name(fam))
                    .weight(Weight(span.style.font_weight.clamp(100, 900)))
                    .style(font_style)
                    .cache_key_flags(flags)
                    .stretch(Stretch::Normal)
                    .metrics(Metrics::new(font_size, line_h))
                    .metadata(idx);
                (slice, attrs)
            });
            buffer.set_rich_text(&mut ctx.font_system, rich_iter, default_attrs, Shaping::Advanced);
        }
        buffer.shape_until_scroll(&mut ctx.font_system, false);

        // Precompute UTF-8 byte range of each BidiParagraph line inside self.text_utf8.
        let base_ptr = self.text_utf8.as_ptr() as usize;
        let para_ranges: Vec<Range<usize>> = BidiParagraphs::new(&self.text_utf8)
            .map(|line| {
                let start = (line.as_ptr() as usize) - base_ptr;
                start..(start + line.len())
            })
            .collect();

        let effective_align = self
            .paragraph_style
            .text_align
            .resolve(self.paragraph_style.text_direction);

        let max_font_ceil = self
            .spans
            .iter()
            .map(|s| s.style.font_size.max(1.0).ceil())
            .fold(default_font_size.ceil(), Scalar::max);
        let mut first_line_y_pad: Option<Scalar> = None;
        let mut current_para_line_i: Option<usize> = None;
        let mut current_para_unwrapped_w: Scalar = 0.0;
        let mut current_word_w: Scalar = 0.0;
        let mut last_line_bottom: Scalar = 0.0;

        for (line_idx, run) in buffer.layout_runs().enumerate() {
            if let Some(max_l) = self.paragraph_style.max_lines {
                if line_idx >= max_l {
                    self.did_exceed_max_lines = true;
                    break;
                }
            }

            let para_range = para_ranges
                .get(run.line_i)
                .cloned()
                .unwrap_or(0..self.text_utf8.len());

            // Track unwrapped paragraph width for max_intrinsic_width.
            if current_para_line_i == Some(run.line_i) {
                current_para_unwrapped_w += run.line_w;
            } else {
                current_para_line_i = Some(run.line_i);
                current_para_unwrapped_w = run.line_w;
            }
            self.max_intrinsic_width = self.max_intrinsic_width.max(current_para_unwrapped_w);

            let mut line_w = run.line_w;
            // Account for inline placeholder width overrides if any placeholder sits on this line.
            for glyph in run.glyphs {
                let g_utf8_start = (para_range.start + glyph.start).min(self.text_utf8.len());
                let g_utf16_start = self.utf8_to_utf16[g_utf8_start];
                if let Some(ph) = self
                    .placeholders
                    .iter()
                    .find(|p| p.utf16_offset == g_utf16_start)
                {
                    if ph.span.width > glyph.w {
                        line_w += ph.span.width - glyph.w;
                    }
                }
            }

            self.longest_line = self.longest_line.max(line_w);

            let align_dx = if finite_width && max_width > line_w {
                match effective_align {
                    TextAlign::Center => ((max_width - line_w) * 0.5).max(0.0),
                    TextAlign::Right => (max_width - line_w).max(0.0),
                    TextAlign::Left | TextAlign::Justify | TextAlign::Start | TextAlign::End => 0.0,
                }
            } else {
                0.0
            };

            let raw_baseline_y = run.line_y.ceil();
            let y_pad = *first_line_y_pad
                .get_or_insert_with(|| (max_font_ceil - raw_baseline_y).max(0.0));
            let baseline_y = raw_baseline_y + y_pad;
            let line_top = (run.line_top + y_pad).floor().max(0.0);
            let line_height = run.line_height.max(default_line_h).ceil();
            let line_bottom = (line_top + line_height).max(baseline_y + 2.0);
            last_line_bottom = last_line_bottom.max(line_bottom);

            let mut line_start_utf16 = usize::MAX;
            let mut line_end_utf16 = 0usize;
            let mut line_end_no_ws_utf16 = 0usize;

            // Track per-span horizontal extents on this line for background & decoration rects.
            let mut span_line_bounds: Vec<(usize, Scalar, Scalar)> = Vec::new();
            let mut extra_placeholder_dx: Scalar = 0.0;

            for glyph in run.glyphs {
                let span_idx = glyph.metadata.min(self.spans.len() - 1);
                let span = &self.spans[span_idx];

                let g_utf8_start = (para_range.start + glyph.start).min(self.text_utf8.len());
                let g_utf8_end = (para_range.start + glyph.end).min(self.text_utf8.len());
                let g_utf16_start = self.utf8_to_utf16[g_utf8_start];
                let g_utf16_end = self.utf8_to_utf16[g_utf8_end].max(g_utf16_start);

                line_start_utf16 = line_start_utf16.min(g_utf16_start);
                line_end_utf16 = line_end_utf16.max(g_utf16_end);

                let cluster_str = self
                    .text_utf8
                    .get(g_utf8_start..g_utf8_end)
                    .unwrap_or_default();
                let is_whitespace = !cluster_str.is_empty() && cluster_str.chars().all(char::is_whitespace);
                if !is_whitespace {
                    line_end_no_ws_utf16 = line_end_no_ws_utf16.max(g_utf16_end);
                    current_word_w += glyph.w;
                    self.min_intrinsic_width = self.min_intrinsic_width.max(current_word_w);
                } else {
                    current_word_w = 0.0;
                }

                let glyph_x_local = align_dx + glyph.x + extra_placeholder_dx;

                // Handle inline widget placeholder character ('\u{FFFC}').
                if cluster_str == "\u{FFFC}" {
                    if let Some(ph) = self
                        .placeholders
                        .iter()
                        .find(|p| p.utf16_offset == g_utf16_start)
                    {
                        let ph_top = match ph.span.alignment {
                            // AboveBaseline
                            1 => baseline_y - ph.span.height,
                            // BelowBaseline
                            2 => baseline_y,
                            // Top
                            3 => line_top,
                            // Bottom
                            4 => line_bottom - ph.span.height,
                            // Middle
                            5 => line_top + (line_height - ph.span.height) * 0.5,
                            // Baseline (0) or default
                            _ => baseline_y - ph.span.baseline_offset.max(ph.span.height),
                        };
                        let ph_rect =
                            Rect::from_xywh(glyph_x_local, ph_top, ph.span.width, ph.span.height);
                        self.placeholder_boxes.push((ph_rect, ph.text_direction));
                        self.glyph_clusters.push(PositionedGlyphCluster {
                            bounds: ph_rect,
                            utf16_start: g_utf16_start,
                            utf16_end: g_utf16_end,
                            line_number: line_idx,
                            is_ltr: glyph.level.is_ltr(),
                        });
                        if ph.span.width > glyph.w {
                            extra_placeholder_dx += ph.span.width - glyph.w;
                        }
                        continue;
                    }
                }

                let cluster_w = glyph.w.max(1.0);
                let cluster_rect =
                    Rect::from_xywh(glyph_x_local, line_top, cluster_w, line_bottom - line_top);
                self.glyph_clusters.push(PositionedGlyphCluster {
                    bounds: cluster_rect,
                    utf16_start: g_utf16_start,
                    utf16_end: g_utf16_end,
                    line_number: line_idx,
                    is_ltr: glyph.level.is_ltr(),
                });

                if let Some(last_span) = span_line_bounds.last_mut() {
                    if last_span.0 == span_idx {
                        last_span.1 = last_span.1.min(glyph_x_local);
                        last_span.2 = last_span.2.max(glyph_x_local + glyph.w);
                    } else {
                        span_line_bounds.push((
                            span_idx,
                            glyph_x_local,
                            glyph_x_local + glyph.w,
                        ));
                    }
                } else {
                    span_line_bounds.push((span_idx, glyph_x_local, glyph_x_local + glyph.w));
                }

                let gx = glyph_x_local + glyph.font_size * glyph.x_offset;
                let gy = baseline_y + (glyph.y - glyph.font_size * glyph.y_offset).trunc();
                let (int_pos, subpixel_pos) = SubpixelPosition::quantize(Point::new(gx, gy));

                let resolved_builtin = ctx.builtin_for_id(glyph.font_id, BuiltinFont::RobotoRegular);
                let scaled_font = ScaledFont::new(resolved_builtin, glyph.font_size);
                let pair = FontGlyphPair::new(scaled_font, glyph.glyph_id, subpixel_pos);

                let (atlas_rect, is_color, bearing, size) =
                    ctx.resolve_or_rasterize_glyph(pair, glyph.font_id, glyph.cache_key_flags);

                if size.width <= 0 || size.height <= 0 || atlas_rect.is_empty() {
                    continue;
                }

                let dest_rect = Rect::from_xywh(
                    (int_pos.x + bearing.x) as Scalar,
                    (int_pos.y + bearing.y) as Scalar,
                    size.width as Scalar,
                    size.height as Scalar,
                );
                let glyph_color = span
                    .style
                    .foreground_paint_color
                    .unwrap_or(span.style.color);

                self.uses_alpha |= !is_color;
                self.uses_color |= is_color;

                for shadow in &span.style.shadows {
                    let shadow_dest = Rect::from_xywh(
                        dest_rect.origin.x + shadow.offset.x,
                        dest_rect.origin.y + shadow.offset.y,
                        dest_rect.size.width,
                        dest_rect.size.height,
                    );
                    self.quad_bounds = Some(match self.quad_bounds {
                        Some(prev) => prev.union(shadow_dest),
                        None => shadow_dest,
                    });
                    self.quads.push(AtlasGlyphQuad {
                        dest_rect: shadow_dest,
                        atlas_rect,
                        color: shadow.color,
                        is_color,
                    });
                }

                self.quad_bounds = Some(match self.quad_bounds {
                    Some(prev) => prev.union(dest_rect),
                    None => dest_rect,
                });
                self.quads.push(AtlasGlyphQuad {
                    dest_rect,
                    atlas_rect,
                    color: glyph_color,
                    is_color,
                });
            }
            current_word_w = 0.0;

            for (span_idx, sx0, sx1) in span_line_bounds {
                let sw = (sx1 - sx0).max(0.0);
                if sw <= 0.0 {
                    continue;
                }
                let span_style = &self.spans[span_idx].style;
                if let Some(bg_color) = span_style.background_paint_color {
                    if bg_color.alpha > 0.0 {
                        self.background_rects.push((
                            Rect::from_xywh(sx0, line_top, sw, line_bottom - line_top),
                            bg_color,
                        ));
                    }
                }
                if span_style.decoration != 0 {
                    let fg = span_style
                        .foreground_paint_color
                        .unwrap_or(span_style.color);
                    let dec_color = span_style.decoration_color.unwrap_or(fg);
                    let thickness = (span_style.font_size / 14.0
                        * span_style.decoration_thickness.max(0.5))
                    .max(1.0);
                    // 0x1 = Underline
                    if (span_style.decoration & 0x1) != 0 {
                        let r = Rect::from_xywh(sx0, baseline_y + 1.0, sw, thickness);
                        self.background_rects.push((r, dec_color));
                        self.decoration_rects.push((r, dec_color));
                    }
                    // 0x2 = Overline
                    if (span_style.decoration & 0x2) != 0 {
                        let r = Rect::from_xywh(sx0, line_top, sw, thickness);
                        self.background_rects.push((r, dec_color));
                        self.decoration_rects.push((r, dec_color));
                    }
                    // 0x4 = LineThrough
                    if (span_style.decoration & 0x4) != 0 {
                        let r = Rect::from_xywh(
                            sx0,
                            baseline_y - span_style.font_size * 0.3,
                            sw,
                            thickness,
                        );
                        self.background_rects.push((r, dec_color));
                        self.decoration_rects.push((r, dec_color));
                    }
                }
            }

            if line_start_utf16 == usize::MAX {
                let para_start_utf16 = self.utf8_to_utf16[para_range.start];
                line_start_utf16 = para_start_utf16;
                line_end_utf16 = para_start_utf16;
                line_end_no_ws_utf16 = para_start_utf16;
            }
            if line_end_no_ws_utf16 < line_start_utf16 {
                line_end_no_ws_utf16 = line_start_utf16;
            }

            let para_end_utf16 = self.utf8_to_utf16[para_range.end];
            let has_hard_break = line_end_utf16 == para_end_utf16
                && self
                    .text_utf8
                    .get(para_range.end..)
                    .is_some_and(|rest| rest.starts_with('\n') || rest.starts_with("\r\n"));
            let end_inc_newline = if has_hard_break {
                let nl_bytes = if self
                    .text_utf8
                    .get(para_range.end..)
                    .is_some_and(|r| r.starts_with("\r\n"))
                {
                    2
                } else {
                    1
                };
                self.utf8_to_utf16[(para_range.end + nl_bytes).min(self.text_utf8.len())]
            } else {
                line_end_utf16
            };

            let ascent = (baseline_y - line_top).max(1.0);
            let descent = (line_bottom - baseline_y).max(1.0);

            self.lines.push(ParagraphLineMetrics {
                hard_break: has_hard_break,
                ascent,
                descent,
                unscaled_ascent: ascent,
                height: (line_bottom - line_top).max(1.0),
                width: line_w,
                left: align_dx,
                baseline: baseline_y,
                line_number: line_idx,
                start_index: line_start_utf16,
                end_index: line_end_utf16,
                end_excluding_whitespaces: line_end_no_ws_utf16,
                end_including_newline: end_inc_newline,
            });
        }

        if self.lines.is_empty() {
            self.height = default_line_h.round();
            self.alphabetic_baseline = default_ascent.round();
            self.ideographic_baseline = self.height;
        } else {
            self.height = last_line_bottom.ceil();
            self.alphabetic_baseline = self.lines[0].baseline;
            self.ideographic_baseline = self.lines[0].baseline + self.lines[0].descent;
        }

        if !finite_width {
            self.width = self.longest_line.ceil();
        }
        self.max_intrinsic_width = self.max_intrinsic_width.max(self.longest_line);
        self.min_intrinsic_width = self.min_intrinsic_width.min(self.max_intrinsic_width);

        if ctx.paragraph_layout_cache.len() >= MAX_PARAGRAPH_LAYOUT_CACHE_ENTRIES {
            ctx.paragraph_layout_cache.clear();
        }
        ctx.paragraph_layout_cache.insert(
            layout_hash,
            Arc::new(CachedParagraphLayout {
                quads: self.quads.clone(),
                background_rects: self.background_rects.clone(),
                decoration_rects: self.decoration_rects.clone(),
                lines: self.lines.clone(),
                placeholder_boxes: self.placeholder_boxes.clone(),
                glyph_clusters: self.glyph_clusters.clone(),
                width: self.width,
                height: self.height,
                longest_line: self.longest_line,
                min_intrinsic_width: self.min_intrinsic_width,
                max_intrinsic_width: self.max_intrinsic_width,
                alphabetic_baseline: self.alphabetic_baseline,
                ideographic_baseline: self.ideographic_baseline,
                did_exceed_max_lines: self.did_exceed_max_lines,
                quad_bounds: self.quad_bounds,
                uses_alpha: self.uses_alpha,
                uses_color: self.uses_color,
            }),
        );
        self.has_laid_out = true;
        self.laid_out_epoch = ctx.cache_epoch;
        self.last_layout_finite = finite_width;
    }

    /// Returns `(utf16_offset, affinity)` where `affinity`: `0` = Upstream, `1` = Downstream
    /// (matching Flutter `ui.TextAffinity.index`: `upstream = 0`, `downstream = 1`).
    pub fn get_position_for_offset(&self, dx: Scalar, dy: Scalar) -> (usize, i32) {
        if self.lines.is_empty() {
            return (0, 1);
        }
        let target_line = self
            .lines
            .iter()
            .find(|l| {
                let top = l.baseline - l.ascent;
                let bottom = l.baseline + l.descent;
                dy >= top && dy <= bottom
            })
            .unwrap_or_else(|| {
                if dy < self.lines[0].baseline - self.lines[0].ascent {
                    &self.lines[0]
                } else {
                    self.lines.last().unwrap()
                }
            });

        let line_no = target_line.line_number;
        let mut line_clusters: Vec<&PositionedGlyphCluster> = self
            .glyph_clusters
            .iter()
            .filter(|c| c.line_number == line_no)
            .collect();
        if line_clusters.is_empty() {
            return (target_line.start_index, 1);
        }
        line_clusters.sort_by(|a, b| a.bounds.get_left().total_cmp(&b.bounds.get_left()));

        if dx <= line_clusters[0].bounds.get_left() {
            return (line_clusters[0].utf16_start, 1);
        }
        for c in &line_clusters {
            let left = c.bounds.get_left();
            let right = c.bounds.get_right();
            if dx >= left && dx <= right {
                let mid = (left + right) * 0.5;
                if dx < mid {
                    return (c.utf16_start, 1);
                } else {
                    return (c.utf16_end, 0);
                }
            }
        }
        let last = line_clusters.last().unwrap();
        (last.utf16_end, 0)
    }

    pub fn get_glyph_info_at(&self, code_unit_offset: usize) -> Option<GlyphInfo> {
        let cluster = self.glyph_clusters.iter().find(|c| {
            code_unit_offset >= c.utf16_start && code_unit_offset < c.utf16_end
        })?;
        Some(GlyphInfo {
            grapheme_cluster_bounds: cluster.bounds,
            grapheme_cluster_code_unit_range_start: cluster.utf16_start,
            grapheme_cluster_code_unit_range_end: cluster.utf16_end,
            is_ltr: cluster.is_ltr,
        })
    }

    pub fn get_closest_glyph_info_at(&self, dx: Scalar, dy: Scalar) -> Option<GlyphInfo> {
        if self.glyph_clusters.is_empty() {
            return None;
        }
        let target_line_no = self
            .lines
            .iter()
            .find(|l| dy >= (l.baseline - l.ascent) && dy <= (l.baseline + l.descent))
            .map(|l| l.line_number)
            .unwrap_or_else(|| {
                if dy < 0.0 {
                    0
                } else {
                    self.lines.len().saturating_sub(1)
                }
            });

        let best = self
            .glyph_clusters
            .iter()
            .filter(|c| c.line_number == target_line_no)
            .min_by(|a, b| {
                let da = ((a.bounds.get_left() + a.bounds.get_right()) * 0.5 - dx).abs();
                let db = ((b.bounds.get_left() + b.bounds.get_right()) * 0.5 - dx).abs();
                da.total_cmp(&db)
            })
            .or_else(|| self.glyph_clusters.first())?;

        Some(GlyphInfo {
            grapheme_cluster_bounds: best.bounds,
            grapheme_cluster_code_unit_range_start: best.utf16_start,
            grapheme_cluster_code_unit_range_end: best.utf16_end,
            is_ltr: best.is_ltr,
        })
    }

    /// Returns `(start_utf16, end_utf16)` word boundaries around `char_pos` (UTF-16 code unit offset).
    pub fn get_word_boundary(&self, char_pos: usize) -> (usize, usize) {
        if self.text_utf8.is_empty() {
            return (0, 0);
        }
        let total_utf16 = self.text_utf16.len();
        let clamped_pos = char_pos.min(total_utf16.saturating_sub(1));

        let chars: Vec<(usize, usize, char)> = self
            .text_utf8
            .char_indices()
            .map(|(byte_i, ch)| {
                let u16_start = self.utf8_to_utf16[byte_i];
                let u16_end = u16_start + ch.len_utf16();
                (u16_start, u16_end, ch)
            })
            .collect();

        let Some(target_idx) = chars
            .iter()
            .position(|&(s, e, _)| clamped_pos >= s && clamped_pos < e)
        else {
            return (char_pos.min(total_utf16), char_pos.min(total_utf16));
        };

        let is_word_char = |c: char| c.is_alphanumeric() || c == '_' || c == '\'';
        let target_kind = is_word_char(chars[target_idx].2);
        let target_ws = chars[target_idx].2.is_whitespace();

        let mut start_i = target_idx;
        while start_i > 0 {
            let prev_c = chars[start_i - 1].2;
            if is_word_char(prev_c) == target_kind && prev_c.is_whitespace() == target_ws {
                start_i -= 1;
            } else {
                break;
            }
        }

        let mut end_i = target_idx;
        while end_i + 1 < chars.len() {
            let next_c = chars[end_i + 1].2;
            if is_word_char(next_c) == target_kind && next_c.is_whitespace() == target_ws {
                end_i += 1;
            } else {
                break;
            }
        }

        (chars[start_i].0, chars[end_i].1)
    }

    pub fn get_line_number_at(&self, char_offset: usize) -> Option<usize> {
        self.lines
            .iter()
            .find(|l| char_offset >= l.start_index && char_offset <= l.end_including_newline)
            .map(|l| l.line_number)
    }

    pub fn get_boxes_for_range(&self, start_utf16: usize, end_utf16: usize) -> Vec<TextBox> {
        if start_utf16 >= end_utf16 || self.glyph_clusters.is_empty() {
            return Vec::new();
        }
        let mut boxes: Vec<TextBox> = Vec::new();

        for line in &self.lines {
            let mut matching: Vec<&PositionedGlyphCluster> = self
                .glyph_clusters
                .iter()
                .filter(|c| {
                    c.line_number == line.line_number
                        && c.utf16_end > start_utf16
                        && c.utf16_start < end_utf16
                })
                .collect();
            if matching.is_empty() {
                continue;
            }
            matching.sort_by(|a, b| a.bounds.get_left().total_cmp(&b.bounds.get_left()));

            let mut cur_rect = matching[0].bounds;
            let mut cur_dir = if matching[0].is_ltr {
                TextDirection::Ltr
            } else {
                TextDirection::Rtl
            };

            for c in matching.iter().skip(1) {
                let dir = if c.is_ltr {
                    TextDirection::Ltr
                } else {
                    TextDirection::Rtl
                };
                if dir == cur_dir && (c.bounds.get_left() - cur_rect.get_right()).abs() <= 2.0 {
                    cur_rect = cur_rect.union(c.bounds);
                } else {
                    boxes.push(TextBox {
                        rect: cur_rect,
                        direction: cur_dir,
                    });
                    cur_rect = c.bounds;
                    cur_dir = dir;
                }
            }
            boxes.push(TextBox {
                rect: cur_rect,
                direction: cur_dir,
            });
        }

        boxes
    }

    pub fn get_boxes_for_placeholders(&self) -> Vec<TextBox> {
        self.placeholder_boxes
            .iter()
            .map(|&(rect, direction)| TextBox { rect, direction })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MaterialIcon, MATERIAL_ICONS_OTF, ROBOTO_BOLD_TTF};
    use approx::assert_abs_diff_eq;

    /// C++ source: typographer_unittests.cc:CanRegisterDynamicFontBytesAndFilterCodepoints
    #[test]
    fn test_register_font_bytes_and_codepoint_coverage() {
        let mut ctx = TypographerContext::new();
        let ids = ctx.register_font_bytes(MATERIAL_ICONS_OTF, Some("packages/custom/MaterialIcons"));
        assert!(!ids.is_empty());

        let chart_cp = MaterialIcon::ShowChart.codepoint() as u32;
        assert!(ctx.is_codepoint_covered(chart_cp));
        assert!(TypographerContext::is_codepoint_covered_by_font_bytes(
            MATERIAL_ICONS_OTF,
            chart_cp
        ));
        assert!(!TypographerContext::is_codepoint_covered_by_font_bytes(
            ROBOTO_BOLD_TTF,
            chart_cp
        ));
        assert!(TypographerContext::is_codepoint_covered_by_font_bytes(
            ROBOTO_BOLD_TTF,
            u32::from(b'A')
        ));
    }

    /// C++ source: typographer_unittests.cc:ParagraphBuilderShapesMultiSpanRichTextAndAlignments
    #[test]
    fn test_paragraph_builder_multi_span_and_center_alignment() {
        let mut ctx = TypographerContext::new();
        let mut para_style = ParagraphStyle {
            text_align: TextAlign::Center,
            ..Default::default()
        };
        para_style.default_text_style.set_font_size(16.0);
        para_style.default_text_style.set_color(Color::WHITE);

        let mut builder = ParagraphBuilder::new(para_style);
        builder.add_text("Hello ");

        let mut bold_red = RichTextStyle::default();
        bold_red.set_flutter_font_style(6, 0); // w700 Bold
        bold_red.set_color(Color::RED);
        bold_red.decoration = 0x1; // Underline
        bold_red.background_paint_color = Some(Color::make_rgba8(20, 40, 60, 255));
        builder.push_style(bold_red);
        builder.add_text("Flutter");
        builder.pop();

        let mut para = builder.build();
        para.layout(&mut ctx, 300.0);

        assert_abs_diff_eq!(para.width, 300.0);
        assert!(para.height >= 16.0);
        assert!(para.longest_line > 40.0 && para.longest_line < 300.0);
        assert_eq!(para.lines.len(), 1);
        // Centered line must have positive horizontal left margin.
        assert!(para.lines[0].left > 20.0);
        assert_eq!(para.quads.len(), 12); // "Hello " (5 glyphs) + "Flutter" (7 glyphs)
        assert!(para.quads.iter().any(|q| q.color == Color::RED));
        // Background + underline rects generated for "Flutter".
        assert!(para.background_rects.len() >= 2);
        assert_eq!(para.decoration_rects.len(), 1);
    }

    /// C++ source: typographer_unittests.cc:ParagraphSupportsUtf16QueriesAndMaxLines
    #[test]
    fn test_paragraph_utf16_queries_placeholders_and_max_lines() {
        let mut ctx = TypographerContext::new();
        let mut para_style = ParagraphStyle {
            max_lines: Some(2),
            ..Default::default()
        };
        para_style.default_text_style.set_font_size(16.0);

        let mut builder = ParagraphBuilder::new(para_style);
        let utf16_hello: Vec<u16> = "Alpha Beta ".encode_utf16().collect();
        builder.add_text_utf16(&utf16_hello);
        builder.add_placeholder(PlaceholderSpan {
            width: 24.0,
            height: 18.0,
            alignment: 1, // AboveBaseline
            baseline_offset: 0.0,
            baseline: 0,
        });
        builder.add_text("\nSecond line\nThird line hidden");

        let mut para = builder.build();
        para.layout(&mut ctx, 400.0);

        assert_eq!(para.lines.len(), 2);
        assert!(para.did_exceed_max_lines);
        assert_eq!(para.get_boxes_for_placeholders().len(), 1);
        assert_abs_diff_eq!(para.get_boxes_for_placeholders()[0].rect.get_width(), 24.0);

        let (wb_start, wb_end) = para.get_word_boundary(1);
        assert_eq!((wb_start, wb_end), (0, 5)); // "Alpha"

        let boxes = para.get_boxes_for_range(0, 5);
        assert!(!boxes.is_empty());
        assert!(boxes[0].rect.get_width() > 20.0);

        let ginfo = para.get_glyph_info_at(0).expect("glyph info at offset 0");
        assert_eq!(ginfo.grapheme_cluster_code_unit_range_start, 0);
        assert!(ginfo.is_ltr);

        let line_no = para.get_line_number_at(0);
        assert_eq!(line_no, Some(0));
    }

    /// C++ source: typographer_unittests.cc:ParagraphSynthesizesFakeItalicWhenOnlyUprightFacesExist
    #[test]
    fn test_paragraph_italic_fallback_synthesizes_fake_italic_without_panic() {
        let mut ctx = TypographerContext::new();

        // Lay out upright "Not run yet" first.
        let mut upright_builder = ParagraphBuilder::new(ParagraphStyle::default());
        let mut upright_style = RichTextStyle::default();
        upright_style.set_font_size(14.0);
        upright_style.set_flutter_font_style(3, 0); // w400, normal (0)
        upright_builder.push_style(upright_style);
        upright_builder.add_text("Not run yet");
        upright_builder.pop();
        let mut upright_para = upright_builder.build();
        upright_para.layout(&mut ctx, 200.0);
        assert!(!upright_para.quads.is_empty());

        // Lay out Italic (font_slant = 1) and Oblique (font_slant = 2) when only upright faces exist.
        let mut italic_builder = ParagraphBuilder::new(ParagraphStyle::default());
        let mut italic_style = RichTextStyle::default();
        italic_style.set_font_size(14.0);
        italic_style.set_flutter_font_style(3, 1); // w400, italic (1)
        italic_builder.push_style(italic_style);
        italic_builder.add_text("Not run yet");
        italic_builder.pop();
        let mut italic_para = italic_builder.build();
        italic_para.layout(&mut ctx, 200.0);

        assert_eq!(italic_para.quads.len(), upright_para.quads.len());
        // Synthetic italic must occupy distinct atlas cache entries from upright glyphs.
        assert_ne!(
            italic_para.quads[0].atlas_rect,
            upright_para.quads[0].atlas_rect
        );

        let mut oblique_builder = ParagraphBuilder::new(ParagraphStyle::default());
        let mut oblique_style = RichTextStyle::default();
        oblique_style.set_font_size(14.0);
        oblique_style.set_flutter_font_style(3, 2); // w400, oblique (2)
        oblique_builder.push_style(oblique_style);
        oblique_builder.add_text("Not run yet");
        oblique_builder.pop();
        let mut oblique_para = oblique_builder.build();
        oblique_para.layout(&mut ctx, 200.0);
        assert_eq!(oblique_para.quads.len(), upright_para.quads.len());
    }

    /// C++ source: typographer_unittests.cc:ParagraphLayoutCacheReusesShapingAndInvalidatesOnFontChange
    #[test]
    fn test_paragraph_layout_cache_hit_and_invalidation() {
        let mut ctx = TypographerContext::new();
        assert_eq!(ctx.cached_paragraph_layout_count(), 0);

        let build_sample = || {
            let mut style = ParagraphStyle::default();
            style.default_text_style.set_font_size(15.0);
            let mut b = ParagraphBuilder::new(style);
            b.add_text("WAVE #6 — Cached Paragraph Layout");
            b.build()
        };

        let mut p1 = build_sample();
        p1.layout(&mut ctx, 260.0);
        assert!(p1.has_laid_out);
        assert_eq!(ctx.cached_paragraph_layout_count(), 1);
        let p1_quads = p1.quads.clone();
        let p1_longest = p1.longest_line;

        // Re-laying out the same Paragraph instance at the same width takes the fast-path.
        p1.layout(&mut ctx, 260.0);
        assert_eq!(ctx.cached_paragraph_layout_count(), 1);

        // Constructing a fresh Paragraph with identical text + style + width hits paragraph_layout_cache.
        let mut p2 = build_sample();
        assert!(!p2.has_laid_out);
        p2.layout(&mut ctx, 260.0);
        assert!(p2.has_laid_out);
        assert_eq!(ctx.cached_paragraph_layout_count(), 1);
        assert_eq!(p2.quads, p1_quads);
        assert_abs_diff_eq!(p2.longest_line, p1_longest);

        // Different width adds a second cache entry.
        p2.layout(&mut ctx, 120.0);
        assert_eq!(ctx.cached_paragraph_layout_count(), 2);

        // register_font_bytes invalidates paragraph_layout_cache and resolved_font_cache.
        ctx.register_font_bytes(ROBOTO_BOLD_TTF, Some("CustomRobotoBold"));
        assert_eq!(ctx.cached_paragraph_layout_count(), 0);

        // Re-laying out repopulates the cache, and clear_caches clears it again.
        p1.layout(&mut ctx, 260.0);
        assert_eq!(ctx.cached_paragraph_layout_count(), 1);
        ctx.clear_caches();
        assert_eq!(ctx.cached_paragraph_layout_count(), 0);
    }

    /// C++ source: typographer_unittests.cc:ParagraphPrecomputesQuadBoundsAndAtlasUsageFlags
    #[test]
    fn test_paragraph_precomputed_quad_bounds_and_atlas_usage_flags() {
        let mut ctx = TypographerContext::new();

        let build_sample = || {
            let mut style = ParagraphStyle::default();
            style.default_text_style.set_font_size(18.0);
            style.default_text_style.shadows.push(TextShadow {
                color: Color::BLACK,
                offset: Point::new(2.0, 3.0),
                blur_sigma: 0.0,
            });
            let mut b = ParagraphBuilder::new(style);
            b.add_text("Precomputed Bounds");
            b.build()
        };

        let mut p1 = build_sample();
        assert!(p1.quad_bounds.is_none());
        assert!(!p1.uses_alpha);
        assert!(!p1.uses_color);

        p1.layout(&mut ctx, 320.0);
        let bounds = p1.quad_bounds.expect("laid-out paragraph must have quad_bounds");
        assert!(bounds.get_width() > 50.0);
        assert!(bounds.get_height() > 10.0);
        assert!(p1.uses_alpha);
        assert!(!p1.uses_color);

        // Verify quad_bounds equals the exact union of all emitted quads (foreground + shadows).
        let expected_union = p1
            .quads
            .iter()
            .fold(None::<Rect>, |acc, q| match acc {
                Some(prev) => Some(prev.union(q.dest_rect)),
                None => Some(q.dest_rect),
            })
            .unwrap();
        assert_eq!(bounds, expected_union);

        // Verify CachedParagraphLayout preserves quad_bounds, uses_alpha, and uses_color on cache hit.
        let mut p2 = build_sample();
        p2.layout(&mut ctx, 320.0);
        assert_eq!(p2.quad_bounds, Some(expected_union));
        assert!(p2.uses_alpha);
        assert!(!p2.uses_color);
    }
}
