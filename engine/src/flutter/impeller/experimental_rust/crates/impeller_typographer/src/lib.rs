#![forbid(unsafe_code)]

//! Impeller Typographer (`impeller/typographer/`):
//! - Shelf/skyline rectangle packing (`etagere` + `RectanglePacker` API parity with `rectangle_packer.cc`)
//! - Subpixel position quantization (`SubpixelPosition`, `[0.0, 0.25, 0.5, 0.75]`)
//! - Shaped `TextFrame` / `TextRun` generation via `cosmic-text` (`rustybuzz`, `unicode-bidi`, `unicode-linebreak`, `swash`)
//! - `GlyphAtlas` & `GlyphAtlasContext` (A8 alpha mask & RGBA8 color glyph atlases)
//! - Embedded `Roboto-Regular`, `Roboto-Bold`, `DejaVuSansMono-Bold`, and `MaterialIcons-Regular` fonts + vector glyph outline extraction to `impeller_path::Path`.

pub mod paragraph;
pub mod rectangle_packer;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cosmic_text::{
    Attrs, Buffer, CacheKey, CacheKeyFlags, Command, Family, FontSystem, Metrics, Shaping, Stretch,
    Style, SwashCache, SwashContent, Weight, Wrap,
};
use impeller_geometry::{Color, IPoint, ISize, Point, Rect, Scalar};
use impeller_path::{FillType, Path, PathBuilder};

pub use paragraph::*;
pub use rectangle_packer::{AllocId, RectanglePacker, SkylineRectanglePacker};

#[cfg(any(all(feature = "embedded-fonts", not(target_arch = "wasm32")), test))]
pub const ROBOTO_REGULAR_TTF: &[u8] = include_bytes!("../../../assets/fonts/Roboto-Regular.ttf");
#[cfg(any(all(feature = "embedded-fonts", not(target_arch = "wasm32")), test))]
pub const ROBOTO_BOLD_TTF: &[u8] = include_bytes!("../../../assets/fonts/Roboto-Bold.ttf");
#[cfg(any(all(feature = "embedded-fonts", not(target_arch = "wasm32")), test))]
pub const ROBOTO_MONO_BOLD_TTF: &[u8] =
    include_bytes!("../../../assets/fonts/DejaVuSansMono-Bold.ttf");
#[cfg(any(all(feature = "embedded-fonts", not(target_arch = "wasm32")), test))]
pub const MATERIAL_ICONS_OTF: &[u8] =
    include_bytes!("../../../assets/fonts/MaterialIcons-Regular.otf");

/// Material Icon codepoints used across `flutter-wasm-compare` (`polymorphic_widgets.dart`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialIcon {
    ShowChart,
    Speed,
    Bolt,
    AutoAwesome,
    CheckCircle,
    RadioButtonUnchecked,
    Memory,
}

impl MaterialIcon {
    pub const fn codepoint(self) -> char {
        match self {
            Self::ShowChart => '\u{e59f}',
            Self::Speed => '\u{e5e0}',
            Self::Bolt => '\u{e0ee}',
            Self::AutoAwesome => '\u{e0b7}',
            Self::CheckCircle => '\u{e159}',
            Self::RadioButtonUnchecked => '\u{e504}',
            Self::Memory => '\u{e3db}',
        }
    }
}

/// Quantized subpixel position along X/Y matching Impeller's `SubpixelPosition` (`typographer/font_glyph_pair.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SubpixelBin {
    #[default]
    Zero = 0,
    Quarter = 1,
    Half = 2,
    ThreeQuarters = 3,
}

impl SubpixelBin {
    pub fn from_fract(fract: Scalar) -> Self {
        let f = fract.rem_euclid(1.0);
        if !(0.125..0.875).contains(&f) {
            Self::Zero
        } else if f < 0.375 {
            Self::Quarter
        } else if f < 0.625 {
            Self::Half
        } else {
            Self::ThreeQuarters
        }
    }

    pub const fn offset(self) -> Scalar {
        match self {
            Self::Zero => 0.0,
            Self::Quarter => 0.25,
            Self::Half => 0.5,
            Self::ThreeQuarters => 0.75,
        }
    }

    const fn to_cosmic(self) -> cosmic_text::SubpixelBin {
        match self {
            Self::Zero => cosmic_text::SubpixelBin::Zero,
            Self::Quarter => cosmic_text::SubpixelBin::One,
            Self::Half => cosmic_text::SubpixelBin::Two,
            Self::ThreeQuarters => cosmic_text::SubpixelBin::Three,
        }
    }
}

/// Quantized 2D subpixel position matching Impeller's `SubpixelPosition` (`impeller/typographer/font_glyph_pair.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SubpixelPosition {
    pub x: SubpixelBin,
    pub y: SubpixelBin,
}

impl SubpixelPosition {
    pub const ZERO: Self = Self {
        x: SubpixelBin::Zero,
        y: SubpixelBin::Zero,
    };

    pub const fn new(x: SubpixelBin, y: SubpixelBin) -> Self {
        Self { x, y }
    }

    pub fn from_point(p: Point) -> Self {
        Self {
            x: SubpixelBin::from_fract(p.x),
            y: SubpixelBin::from_fract(p.y),
        }
    }

    /// Splits a floating-point coordinate into an integer pixel anchor and quantized [`SubpixelPosition`]
    /// such that `int_point + subpixel_offset` rounds to the nearest `0.25` pixel bin.
    pub fn quantize(p: Point) -> (IPoint, Self) {
        let int_x = (p.x + 0.125).floor() as i64;
        let int_y = (p.y + 0.125).floor() as i64;
        (IPoint::new(int_x, int_y), Self::from_point(p))
    }
}

/// Font family selector for the embedded font collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BuiltinFont {
    #[default]
    RobotoRegular,
    RobotoBold,
    RobotoMonoBold,
    MaterialIcons,
}

/// A font and quantized font size in milli-pixels (`impeller/typographer/font_glyph_pair.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScaledFont {
    pub font: BuiltinFont,
    pub font_size_milli: u32,
}

impl ScaledFont {
    pub fn new(font: BuiltinFont, font_size: Scalar) -> Self {
        let font_size_milli = (font_size.max(0.0) * 1000.0).round() as u32;
        Self {
            font,
            font_size_milli,
        }
    }

    pub fn font_size(self) -> Scalar {
        (self.font_size_milli as Scalar) / 1000.0
    }
}

/// Re-exported `fontdb::ID` handle for dynamically registered fonts.
pub type FontDbId = cosmic_text::fontdb::ID;

/// Cache key tuple `(BuiltinFont, glyph_id, font_size_milli, subpixel_x, subpixel_y)`.
pub type GlyphCacheKey = (BuiltinFont, u16, u32, SubpixelBin, SubpixelBin);

/// Internal atlas cache key `(fontdb::ID, glyph_id, font_size_milli, subpixel_x, subpixel_y, cache_key_flags_bits)`
/// so dynamically registered fonts and synthetic styles (e.g. `FAKE_ITALIC`) cache without collision.
pub(crate) type InternalGlyphCacheKey = (
    cosmic_text::fontdb::ID,
    u16,
    u32,
    SubpixelBin,
    SubpixelBin,
    u32,
);

/// Cached atlas glyph metadata `(atlas_rect, is_color, bearing, size)`.
pub type GlyphCacheValue = (Rect, bool, IPoint, ISize);

/// A scaled font, glyph ID, and subpixel offset matching Impeller's `FontGlyphPair` (`impeller/typographer/font_glyph_pair.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontGlyphPair {
    pub scaled_font: ScaledFont,
    pub glyph_id: u16,
    pub subpixel_position: SubpixelPosition,
}

impl FontGlyphPair {
    pub const fn new(
        scaled_font: ScaledFont,
        glyph_id: u16,
        subpixel_position: SubpixelPosition,
    ) -> Self {
        Self {
            scaled_font,
            glyph_id,
            subpixel_position,
        }
    }

    pub const fn cache_key(self) -> GlyphCacheKey {
        (
            self.scaled_font.font,
            self.glyph_id,
            self.scaled_font.font_size_milli,
            self.subpixel_position.x,
            self.subpixel_position.y,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlyphAtlasType {
    /// Single-channel 8-bit coverage mask (`R8Unorm`) modulated by vertex tint color.
    AlphaBitmap,
    /// Four-channel 8-bit premultiplied RGBA (`Rgba8Unorm`) for color glyphs.
    ColorBitmap,
}

/// A positioned, rasterized glyph quad ready to be uploaded & drawn by `impeller_wgpu`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasGlyphQuad {
    /// Destination screen/local rectangle `[left, top, width, height]` in the `TextFrame`'s local coordinate space.
    pub dest_rect: Rect,
    /// Pixel rectangle `[x, y, width, height]` inside the `GlyphAtlas` texture.
    pub atlas_rect: Rect,
    /// Per-glyph tint color (premultiplied or unpremultiplied RGBA; modulated in `glyph_atlas.wgsl`).
    pub color: Color,
    /// Whether this glyph samples `ColorBitmap` (`true`) or `AlphaBitmap` (`false`).
    pub is_color: bool,
}

static NEXT_ATLAS_ID: AtomicU64 = AtomicU64::new(1);

/// A GPU-ready glyph atlas backing buffer (`AlphaBitmap` `R8` or `ColorBitmap` `RGBA8`).
#[derive(Debug, Clone)]
pub struct GlyphAtlas {
    pub atlas_id: u64,
    pub atlas_type: GlyphAtlasType,
    pub size: ISize,
    /// Raw pixel bytes (`width * height` for `AlphaBitmap`, `width * height * 4` for `ColorBitmap`).
    pub pixels: Vec<u8>,
    pub generation: u64,
    pub resize_epoch: u64,
    pub row_generations: Vec<u64>,
}

impl GlyphAtlas {
    pub fn new(atlas_type: GlyphAtlasType, width: u32, height: u32) -> Self {
        let bpp = match atlas_type {
            GlyphAtlasType::AlphaBitmap => 1usize,
            GlyphAtlasType::ColorBitmap => 4usize,
        };
        let h_usize = height as usize;
        Self {
            atlas_id: NEXT_ATLAS_ID.fetch_add(1, Ordering::Relaxed),
            atlas_type,
            size: ISize::new(i64::from(width), i64::from(height)),
            pixels: vec![0u8; (width as usize) * h_usize * bpp],
            generation: 1,
            resize_epoch: 1,
            row_generations: vec![1u64; h_usize],
        }
    }

    /// Marks rows in `[y_start, y_start + padded_h)` as modified at `self.generation`.
    pub(crate) fn mark_dirty_rows(&mut self, y_start: i64, padded_h: i64) {
        let y0 = y_start.max(0) as usize;
        let y1 = (y_start + padded_h).clamp(0, self.size.height) as usize;
        let end = y1.min(self.row_generations.len());
        if y0 < end {
            self.row_generations[y0..end].fill(self.generation);
        }
    }

    /// Returns the half-open row span `(y_min, y_max)` modified since `uploaded_generation`,
    /// or `None` if `uploaded_generation >= self.generation`.
    pub fn dirty_row_range_since(&self, uploaded_generation: u64) -> Option<(u32, u32)> {
        if uploaded_generation >= self.generation {
            return None;
        }
        if uploaded_generation == 0 {
            return Some((0, self.size.height.max(1) as u32));
        }
        let first_y = self
            .row_generations
            .iter()
            .position(|&row_gen| row_gen > uploaded_generation)?;
        let last_y = self
            .row_generations
            .iter()
            .rposition(|&row_gen| row_gen > uploaded_generation)
            .unwrap_or(first_y);
        Some((first_y as u32, (last_y + 1) as u32))
    }

    /// Returns a zero-allocation metadata clone of this `GlyphAtlas` with empty `pixels`
    /// and `row_generations` vectors (`0` heap allocations).
    pub fn descriptor_without_pixels(&self) -> Self {
        Self {
            atlas_id: self.atlas_id,
            atlas_type: self.atlas_type,
            size: self.size,
            pixels: Vec::new(),
            generation: self.generation,
            resize_epoch: self.resize_epoch,
            row_generations: Vec::new(),
        }
    }

    /// Grows the backing pixel buffer to `(new_width, new_height)`, copying all existing
    /// pixel rows into the top-left corner so existing atlas coordinates remain valid,
    /// and bumps `generation` and `resize_epoch`.
    pub fn grow(&mut self, new_width: u32, new_height: u32) {
        let old_w = self.size.width.max(0) as usize;
        let old_h = self.size.height.max(0) as usize;
        let new_w = (new_width as usize).max(old_w);
        let new_h = (new_height as usize).max(old_h);
        if new_w == old_w && new_h == old_h {
            return;
        }
        let bpp = match self.atlas_type {
            GlyphAtlasType::AlphaBitmap => 1usize,
            GlyphAtlasType::ColorBitmap => 4usize,
        };
        let mut new_pixels = vec![0u8; new_w * new_h * bpp];
        let old_row_bytes = old_w * bpp;
        let new_row_bytes = new_w * bpp;
        for row in 0..old_h {
            let src_start = row * old_row_bytes;
            let dst_start = row * new_row_bytes;
            new_pixels[dst_start..dst_start + old_row_bytes]
                .copy_from_slice(&self.pixels[src_start..src_start + old_row_bytes]);
        }
        self.pixels = new_pixels;
        self.size = ISize::new(new_w as i64, new_h as i64);
        self.generation = self.generation.wrapping_add(1);
        self.resize_epoch = self.resize_epoch.wrapping_add(1);
        self.row_generations = vec![self.generation; new_h];
    }
}

/// Style parameters for shaping a `TextFrame`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextStyle {
    pub font: BuiltinFont,
    pub font_size: Scalar,
    pub line_height: Scalar,
    pub color: Color,
    pub max_width: Option<Scalar>,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font: BuiltinFont::RobotoRegular,
            font_size: 14.0,
            line_height: 18.0,
            color: Color::WHITE,
            max_width: None,
        }
    }
}

/// Shaped text frame containing positioned glyph runs and tight bounding box (`impeller/typographer/text_frame.h`).
#[derive(Debug, Clone)]
pub struct TextFrame {
    pub bounds: Rect,
    pub quads: Vec<AtlasGlyphQuad>,
    pub line_count: usize,
    pub longest_line: Scalar,
}

#[derive(Debug, Clone)]
pub(crate) struct RegisteredFont {
    pub(crate) id: cosmic_text::fontdb::ID,
    pub(crate) family: String,
    pub(crate) weight: Weight,
    pub(crate) style: Style,
    pub(crate) stretch: Stretch,
}

pub(crate) type ResolvedFontCacheKey = (Vec<String>, u16, i32);
pub(crate) type ResolvedFontCacheValue = (String, Style, CacheKeyFlags);

/// Persistent typographer context holding `cosmic_text::FontSystem`, `SwashCache`, and `GlyphAtlas`.
pub struct TypographerContext {
    pub alpha_atlas: GlyphAtlas,
    pub color_atlas: GlyphAtlas,
    alpha_packer: RectanglePacker,
    color_packer: RectanglePacker,
    glyph_cache: HashMap<InternalGlyphCacheKey, GlyphCacheValue>,
    pub(crate) paragraph_layout_cache: HashMap<u64, Arc<CachedParagraphLayout>>,
    pub(crate) resolved_font_cache: HashMap<ResolvedFontCacheKey, ResolvedFontCacheValue>,
    pub(crate) cache_epoch: u64,
    pub(crate) font_system: FontSystem,
    swash_cache: SwashCache,
    pub(crate) fonts: [RegisteredFont; 4],
}

impl std::fmt::Debug for TypographerContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypographerContext")
            .field("alpha_atlas_size", &self.alpha_atlas.size)
            .field("color_atlas_size", &self.color_atlas.size)
            .field("cached_glyphs", &self.glyph_cache.len())
            .field(
                "cached_paragraph_layouts",
                &self.paragraph_layout_cache.len(),
            )
            .finish()
    }
}

impl Default for TypographerContext {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_ATLAS_DIM: i64 = 8192;

#[cfg(any(all(feature = "embedded-fonts", not(target_arch = "wasm32")), test))]
fn load_embedded_face(
    db: &mut cosmic_text::fontdb::Database,
    bytes: &'static [u8],
    force_proportional: bool,
) -> RegisteredFont {
    let ids = db.load_font_source(cosmic_text::fontdb::Source::Binary(Arc::new(bytes)));
    let mut id = ids[0];
    let mut face = db.face(id).expect("embedded font slice must parse").clone();

    if force_proportional && face.monospaced {
        let ps_name = face.post_script_name.clone();
        db.remove_face(id);
        face.id = cosmic_text::fontdb::ID::dummy();
        face.monospaced = false;
        db.push_face_info(face);
        face = db
            .faces()
            .find(|f| f.post_script_name == ps_name)
            .expect("re-inserted proportional face must exist")
            .clone();
        id = face.id;
    }

    let family = face
        .families
        .first()
        .map(|(name, _)| name.clone())
        .unwrap_or_default();
    RegisteredFont {
        id,
        family,
        weight: face.weight,
        style: face.style,
        stretch: face.stretch,
    }
}

const fn font_index(font: BuiltinFont) -> usize {
    match font {
        BuiltinFont::RobotoRegular => 0,
        BuiltinFont::RobotoBold => 1,
        BuiltinFont::RobotoMonoBold => 2,
        BuiltinFont::MaterialIcons => 3,
    }
}

impl TypographerContext {
    pub fn new() -> Self {
        Self::with_atlas_sizes(1024, 512)
    }

    /// Creates a [`TypographerContext`] without loading embedded font assets (`0` faces initially),
    /// initializing placeholder font slots that are populated dynamically by [`Self::register_font_bytes`].
    pub fn with_atlas_sizes_empty(alpha_dim: u32, color_dim: u32) -> Self {
        let mut db = cosmic_text::fontdb::Database::new();
        db.set_sans_serif_family("Roboto".to_string());
        db.set_serif_family("Roboto".to_string());
        db.set_monospace_family("Roboto".to_string());

        let font_system = FontSystem::new_with_locale_and_db("en-US".to_string(), db);
        let alpha_w = alpha_dim.max(16);
        let color_w = color_dim.max(16);
        let dummy_id = cosmic_text::fontdb::ID::dummy();

        let make_placeholder = |family: &str, weight: Weight| RegisteredFont {
            id: dummy_id,
            family: family.to_string(),
            weight,
            style: Style::Normal,
            stretch: Stretch::Normal,
        };

        Self {
            alpha_atlas: GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, alpha_w, alpha_w),
            color_atlas: GlyphAtlas::new(GlyphAtlasType::ColorBitmap, color_w, color_w),
            alpha_packer: RectanglePacker::new(i64::from(alpha_w), i64::from(alpha_w)),
            color_packer: RectanglePacker::new(i64::from(color_w), i64::from(color_w)),
            glyph_cache: HashMap::new(),
            paragraph_layout_cache: HashMap::new(),
            resolved_font_cache: HashMap::new(),
            cache_epoch: 1,
            font_system,
            swash_cache: SwashCache::new(),
            fonts: [
                make_placeholder("Roboto", Weight::NORMAL),
                make_placeholder("Roboto", Weight::BOLD),
                make_placeholder("Roboto", Weight::BOLD),
                make_placeholder("MaterialIcons", Weight::NORMAL),
            ],
        }
    }

    /// Creates a [`TypographerContext`] with custom initial square dimensions for the
    /// alpha and color glyph atlases (primarily useful for testing dynamic atlas growth).
    pub fn with_atlas_sizes(alpha_dim: u32, color_dim: u32) -> Self {
        #[cfg(any(all(feature = "embedded-fonts", not(target_arch = "wasm32")), test))]
        {
            let mut db = cosmic_text::fontdb::Database::new();
            let roboto_reg = load_embedded_face(&mut db, ROBOTO_REGULAR_TTF, false);
            let roboto_bold = load_embedded_face(&mut db, ROBOTO_BOLD_TTF, false);
            let roboto_mono_bold = load_embedded_face(&mut db, ROBOTO_MONO_BOLD_TTF, false);
            let material_icons = load_embedded_face(&mut db, MATERIAL_ICONS_OTF, true);

            db.set_sans_serif_family(roboto_reg.family.clone());
            db.set_serif_family(roboto_reg.family.clone());
            db.set_monospace_family(roboto_mono_bold.family.clone());

            let font_system = FontSystem::new_with_locale_and_db("en-US".to_string(), db);
            let alpha_w = alpha_dim.max(16);
            let color_w = color_dim.max(16);

            Self {
                alpha_atlas: GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, alpha_w, alpha_w),
                color_atlas: GlyphAtlas::new(GlyphAtlasType::ColorBitmap, color_w, color_w),
                alpha_packer: RectanglePacker::new(i64::from(alpha_w), i64::from(alpha_w)),
                color_packer: RectanglePacker::new(i64::from(color_w), i64::from(color_w)),
                glyph_cache: HashMap::new(),
                paragraph_layout_cache: HashMap::new(),
                resolved_font_cache: HashMap::new(),
                cache_epoch: 1,
                font_system,
                swash_cache: SwashCache::new(),
                fonts: [roboto_reg, roboto_bold, roboto_mono_bold, material_icons],
            }
        }
        #[cfg(not(any(all(feature = "embedded-fonts", not(target_arch = "wasm32")), test)))]
        {
            Self::with_atlas_sizes_empty(alpha_dim, color_dim)
        }
    }

    /// Returns the number of distinct `(BuiltinFont, glyph_id, font_size_milli, SubpixelPosition)`
    /// entries currently cached in the glyph atlases.
    pub fn cached_glyph_count(&self) -> usize {
        self.glyph_cache.len()
    }

    /// Returns the number of distinct `Paragraph` layouts currently cached in `paragraph_layout_cache`.
    pub fn cached_paragraph_layout_count(&self) -> usize {
        self.paragraph_layout_cache.len()
    }

    /// Looks up a cached glyph entry by its [`GlyphCacheKey`].
    pub fn lookup_cached_glyph(&self, key: &GlyphCacheKey) -> Option<GlyphCacheValue> {
        let font_id = self.registered_font(key.0).id;
        self.glyph_cache
            .get(&(font_id, key.1, key.2, key.3, key.4, 0))
            .copied()
    }

    /// Dynamically registers raw TrueType/OpenType font bytes into the `FontSystem` database
    /// (supporting `fontCollection_registerTypeface` in `impeller_skwasm_ffi`).
    ///
    /// If `family_alias` is provided (e.g. `"MaterialIcons"`, `"Roboto"`,
    /// `"packages/cupertino_icons/CupertinoIcons"`), adds the alias to the face's family list
    /// and disables `monospaced` on icon fonts so `cosmic-text` never substitutes a monospace fallback.
    pub fn register_font_bytes(
        &mut self,
        bytes: &[u8],
        family_alias: Option<&str>,
    ) -> Vec<cosmic_text::fontdb::ID> {
        if bytes.is_empty() {
            return Vec::new();
        }
        let shared_bytes: Arc<dyn AsRef<[u8]> + Sync + Send> = Arc::new(bytes.to_vec());
        let raw_ids = self
            .font_system
            .db_mut()
            .load_font_source(cosmic_text::fontdb::Source::Binary(shared_bytes));

        let alias_trimmed = family_alias.map(str::trim).filter(|s| !s.is_empty());
        let mut final_ids = Vec::with_capacity(raw_ids.len());

        for raw_id in raw_ids {
            let Some(mut face) = self.font_system.db().face(raw_id).cloned() else {
                continue;
            };
            let mut modified = false;

            if let Some(alias) = alias_trimmed {
                if !face.families.iter().any(|(name, _)| name == alias) {
                    face.families.push((
                        alias.to_string(),
                        cosmic_text::fontdb::Language::English_UnitedStates,
                    ));
                    modified = true;
                }
                if let Some(short) = alias.rsplit('/').next() {
                    if short != alias && !face.families.iter().any(|(name, _)| name == short) {
                        face.families.push((
                            short.to_string(),
                            cosmic_text::fontdb::Language::English_UnitedStates,
                        ));
                        modified = true;
                    }
                }
            }

            let is_icon_font = alias_trimmed
                .is_some_and(|a| a.to_ascii_lowercase().contains("icons"))
                || face
                    .families
                    .iter()
                    .any(|(name, _)| name.to_ascii_lowercase().contains("icons"));
            if is_icon_font && face.monospaced {
                face.monospaced = false;
                modified = true;
            }

            let resolved_id = if modified {
                let db = self.font_system.db_mut();
                db.remove_face(raw_id);
                face.id = cosmic_text::fontdb::ID::dummy();
                db.push_face_info(face.clone());
                db.faces().last().map_or(raw_id, |inserted| inserted.id)
            } else {
                raw_id
            };

            let matches_material_icons = alias_trimmed.is_some_and(|a| {
                a.eq_ignore_ascii_case("MaterialIcons") || a.eq_ignore_ascii_case("Material Icons")
            }) || face.families.iter().any(|(name, _)| {
                name.eq_ignore_ascii_case("MaterialIcons")
                    || name.eq_ignore_ascii_case("Material Icons")
            });
            if matches_material_icons {
                let fam = alias_trimmed
                    .map(str::to_string)
                    .or_else(|| face.families.first().map(|(n, _)| n.clone()))
                    .unwrap_or_else(|| "MaterialIcons".to_string());
                self.fonts[3] = RegisteredFont {
                    id: resolved_id,
                    family: fam,
                    weight: face.weight,
                    style: face.style,
                    stretch: face.stretch,
                };
            } else if !is_icon_font {
                let primary_fam = alias_trimmed
                    .map(str::to_string)
                    .or_else(|| face.families.first().map(|(n, _)| n.clone()))
                    .unwrap_or_else(|| "Roboto".to_string());
                let reg_entry = RegisteredFont {
                    id: resolved_id,
                    family: primary_fam.clone(),
                    weight: face.weight,
                    style: face.style,
                    stretch: face.stretch,
                };

                let matches_mono = alias_trimmed.is_some_and(|a| {
                    let lower = a.to_ascii_lowercase();
                    lower == "monospace"
                        || lower == "robotomono"
                        || lower == "roboto mono"
                        || lower.contains("dejavu")
                        || lower.contains("courier")
                }) || (alias_trimmed.is_none() && face.monospaced);

                let matches_roboto = alias_trimmed
                    .is_some_and(|a| a.eq_ignore_ascii_case("Roboto"))
                    || (alias_trimmed.is_none()
                        && face
                            .families
                            .iter()
                            .any(|(n, _)| n.eq_ignore_ascii_case("Roboto")));

                if matches_mono {
                    self.fonts[2] = reg_entry.clone();
                    self.font_system
                        .db_mut()
                        .set_monospace_family(primary_fam.clone());
                }

                if matches_roboto {
                    if face.weight.0 >= 600 {
                        self.fonts[1] = reg_entry.clone();
                        if self.fonts[0].id == cosmic_text::fontdb::ID::dummy() {
                            self.fonts[0] = reg_entry.clone();
                            self.font_system
                                .db_mut()
                                .set_sans_serif_family(primary_fam.clone());
                            self.font_system
                                .db_mut()
                                .set_serif_family(primary_fam.clone());
                        }
                    } else {
                        self.fonts[0] = reg_entry.clone();
                        self.font_system
                            .db_mut()
                            .set_sans_serif_family(primary_fam.clone());
                        self.font_system
                            .db_mut()
                            .set_serif_family(primary_fam.clone());
                        if self.fonts[1].id == cosmic_text::fontdb::ID::dummy()
                            || self.fonts[1].weight.0 < 600
                        {
                            self.fonts[1] = reg_entry.clone();
                        }
                    }
                    if self.fonts[2].id == cosmic_text::fontdb::ID::dummy()
                        || self.fonts[2].family.eq_ignore_ascii_case("Roboto")
                    {
                        self.fonts[2] = reg_entry;
                        self.font_system.db_mut().set_monospace_family(primary_fam);
                    }
                } else if self.fonts[0].id == cosmic_text::fontdb::ID::dummy() {
                    self.fonts[0] = reg_entry.clone();
                    self.font_system
                        .db_mut()
                        .set_sans_serif_family(primary_fam.clone());
                    self.font_system
                        .db_mut()
                        .set_serif_family(primary_fam.clone());
                    if self.fonts[1].id == cosmic_text::fontdb::ID::dummy() {
                        self.fonts[1] = reg_entry.clone();
                    }
                    if self.fonts[2].id == cosmic_text::fontdb::ID::dummy() {
                        self.fonts[2] = reg_entry;
                        self.font_system.db_mut().set_monospace_family(primary_fam);
                    }
                }
            }

            final_ids.push(resolved_id);
        }

        if !final_ids.is_empty() {
            self.paragraph_layout_cache.clear();
            self.resolved_font_cache.clear();
            self.cache_epoch = self.cache_epoch.wrapping_add(1).max(1);
        }

        final_ids
    }

    /// Clears cached glyph entries and paragraph layout caches (for `fontCollection_clearCaches`).
    pub fn clear_caches(&mut self) {
        self.glyph_cache.clear();
        self.paragraph_layout_cache.clear();
        self.resolved_font_cache.clear();
        self.cache_epoch = self.cache_epoch.wrapping_add(1).max(1);
        self.alpha_packer.reset();
        self.color_packer.reset();
        self.alpha_atlas.pixels.fill(0);
        self.color_atlas.pixels.fill(0);
        self.alpha_atlas.generation = self.alpha_atlas.generation.wrapping_add(1);
        self.color_atlas.generation = self.color_atlas.generation.wrapping_add(1);
        self.alpha_atlas.resize_epoch = self.alpha_atlas.resize_epoch.wrapping_add(1);
        self.color_atlas.resize_epoch = self.color_atlas.resize_epoch.wrapping_add(1);
        self.alpha_atlas
            .row_generations
            .fill(self.alpha_atlas.generation);
        self.color_atlas
            .row_generations
            .fill(self.color_atlas.generation);
    }

    /// Checks whether any registered typeface in this [`TypographerContext`] covers `codepoint`.
    pub fn is_codepoint_covered(&mut self, codepoint: u32) -> bool {
        let Some(ch) = char::from_u32(codepoint) else {
            return false;
        };
        let face_ids: Vec<cosmic_text::fontdb::ID> =
            self.font_system.db().faces().map(|f| f.id).collect();
        for id in face_ids {
            if let Some(font) = self.font_system.get_font(id) {
                if font
                    .rustybuzz()
                    .glyph_index(ch)
                    .is_some_and(|gid| gid.0 != 0)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Checks whether a specific raw TrueType/OpenType font byte slice covers `codepoint`
    /// (for `typefaces_filterCoveredCodePoints`).
    pub fn is_codepoint_covered_by_font_bytes(bytes: &[u8], codepoint: u32) -> bool {
        let Some(ch) = char::from_u32(codepoint) else {
            return false;
        };
        let Ok(face) = cosmic_text::ttf_parser::Face::parse(bytes, 0) else {
            return false;
        };
        face.glyph_index(ch).is_some_and(|gid| gid.0 != 0)
    }

    pub(crate) fn registered_font(&self, font: BuiltinFont) -> &RegisteredFont {
        &self.fonts[font_index(font)]
    }

    pub(crate) fn builtin_for_id(
        &self,
        id: cosmic_text::fontdb::ID,
        fallback: BuiltinFont,
    ) -> BuiltinFont {
        if id == self.fonts[0].id {
            BuiltinFont::RobotoRegular
        } else if id == self.fonts[1].id {
            BuiltinFont::RobotoBold
        } else if id == self.fonts[2].id {
            BuiltinFont::RobotoMonoBold
        } else if id == self.fonts[3].id {
            BuiltinFont::MaterialIcons
        } else {
            fallback
        }
    }

    fn shape_buffer(&mut self, text: &str, style: TextStyle) -> Buffer {
        let font_size = style.font_size.max(1.0);
        // Ensure line_height >= font_size * 1.2 so cosmic-text centering_offset is never negative.
        let line_height = style.line_height.max(font_size * 1.2);
        let metrics = Metrics::new(font_size, line_height);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        if let Some(max_w) = style.max_width {
            buffer.set_wrap(&mut self.font_system, Wrap::WordOrGlyph);
            buffer.set_size(&mut self.font_system, Some(max_w.max(1.0)), None);
        } else {
            buffer.set_wrap(&mut self.font_system, Wrap::None);
            buffer.set_size(&mut self.font_system, None, None);
        }

        let reg = self.registered_font(style.font).clone();
        let attrs = Attrs::new()
            .family(Family::Name(&reg.family))
            .weight(reg.weight)
            .style(reg.style)
            .stretch(reg.stretch);

        buffer.set_text(&mut self.font_system, text, attrs, Shaping::Advanced);
        buffer.shape_until_scroll(&mut self.font_system, false);
        buffer
    }

    fn pack_with_growth(
        packer: &mut RectanglePacker,
        atlas: &mut GlyphAtlas,
        padded_w: i64,
        padded_h: i64,
    ) -> Option<IPoint> {
        if let Some(pt) = packer.add_rect(padded_w, padded_h) {
            return Some(pt);
        }

        while packer.width() < MAX_ATLAS_DIM || packer.height() < MAX_ATLAS_DIM {
            let min_w = (padded_w.max(1) as u64).next_power_of_two() as i64;
            let min_h = (padded_h.max(1) as u64).next_power_of_two() as i64;
            let next_w = (packer.width() * 2).max(min_w).min(MAX_ATLAS_DIM);
            let next_h = (packer.height() * 2).max(min_h).min(MAX_ATLAS_DIM);
            if next_w == packer.width() && next_h == packer.height() {
                break;
            }
            if !packer.grow(ISize::new(next_w, next_h)) {
                break;
            }
            atlas.grow(next_w as u32, next_h as u32);
            if let Some(pt) = packer.add_rect(padded_w, padded_h) {
                return Some(pt);
            }
        }
        None
    }

    fn pack_alpha_mask(
        &mut self,
        key: InternalGlyphCacheKey,
        bearing: IPoint,
        w: i64,
        h: i64,
        data: &[u8],
    ) -> GlyphCacheValue {
        let size = ISize::new(w, h);
        if w <= 0 || h <= 0 {
            let empty = (Rect::ZERO, false, bearing, ISize::ZERO);
            self.glyph_cache.insert(key, empty);
            return empty;
        }
        // Add 1-pixel zeroed border on all 4 sides to prevent bilinear sampling bleed.
        let padded_w = w + 2;
        let padded_h = h + 2;
        let Some(packed) = Self::pack_with_growth(
            &mut self.alpha_packer,
            &mut self.alpha_atlas,
            padded_w,
            padded_h,
        ) else {
            return (Rect::ZERO, false, bearing, ISize::ZERO);
        };
        let dst_x = (packed.x + 1) as usize;
        let dst_y = (packed.y + 1) as usize;
        let row_len = w as usize;
        let atlas_stride = self.alpha_atlas.size.width as usize;

        for row in 0..(h as usize) {
            let src_offset = row * row_len;
            let dst_offset = (dst_y + row) * atlas_stride + dst_x;
            self.alpha_atlas.pixels[dst_offset..dst_offset + row_len]
                .copy_from_slice(&data[src_offset..src_offset + row_len]);
        }
        self.alpha_atlas.generation = self.alpha_atlas.generation.wrapping_add(1);
        self.alpha_atlas.mark_dirty_rows(packed.y, padded_h);

        let atlas_rect =
            Rect::from_xywh(dst_x as Scalar, dst_y as Scalar, w as Scalar, h as Scalar);
        let val = (atlas_rect, false, bearing, size);
        self.glyph_cache.insert(key, val);
        val
    }

    /// Rasterizes a glyph outline via `ttf_parser` + 8x sub-scanline analytic non-zero winding AA.
    /// Used as an automatic fallback for OpenType CFF (`OTTO`) fonts such as `MaterialIcons-Regular.otf`
    /// where `swash` / `skrifa` returns `None`.
    fn rasterize_ttf_outline_mask(
        &mut self,
        key: InternalGlyphCacheKey,
        pair: FontGlyphPair,
        font_id: cosmic_text::fontdb::ID,
        font_size: Scalar,
    ) -> GlyphCacheValue {
        let sub_origin = Point::new(
            pair.subpixel_position.x.offset(),
            pair.subpixel_position.y.offset(),
        );
        let mut builder = PathBuilder::new();
        if !Self::append_glyph_outline_from_ttf(
            &mut self.font_system,
            font_id,
            pair.glyph_id,
            font_size,
            sub_origin,
            &mut builder,
        ) {
            let empty = (Rect::ZERO, false, IPoint::ZERO, ISize::ZERO);
            self.glyph_cache.insert(key, empty);
            return empty;
        }

        let path = builder.take_path(FillType::NonZero);
        let Some(bbox) = path.get_bounding_box() else {
            let empty = (Rect::ZERO, false, IPoint::ZERO, ISize::ZERO);
            self.glyph_cache.insert(key, empty);
            return empty;
        };

        let min_x = bbox.get_left().floor() as i64;
        let min_y = bbox.get_top().floor() as i64;
        let max_x = bbox.get_right().ceil() as i64;
        let max_y = bbox.get_bottom().ceil() as i64;
        let w = max_x - min_x;
        let h = max_y - min_y;
        let bearing = IPoint::new(min_x, min_y);

        if w <= 0 || h <= 0 {
            let empty = (Rect::ZERO, false, bearing, ISize::ZERO);
            self.glyph_cache.insert(key, empty);
            return empty;
        }

        let polyline = path.create_polyline(4.0);
        let mut edges = Vec::new();
        for contour in &polyline.contours {
            if contour.end_index <= contour.start_index + 1 {
                continue;
            }
            let pts = &polyline.points[contour.start_index..contour.end_index];
            for i in 0..pts.len() {
                let p0 = pts[i];
                let p1 = if i + 1 < pts.len() {
                    pts[i + 1]
                } else {
                    pts[0]
                };
                if (p1.y - p0.y).abs() > 1e-6 {
                    edges.push((p0, p1));
                }
            }
        }

        let w_usize = w as usize;
        let h_usize = h as usize;
        let mut bitmap = vec![0u8; w_usize * h_usize];
        let mut accum = vec![0.0f32; w_usize];
        let mut crossings: Vec<(Scalar, i32)> = Vec::new();

        for row in 0..h_usize {
            accum.fill(0.0);
            for sub in 0..8 {
                let scan_y = (min_y as Scalar) + (row as Scalar) + ((sub as Scalar) + 0.5) * 0.125;
                crossings.clear();
                for &(p0, p1) in &edges {
                    let (y0, y1, x0, x1, dir) = if p0.y < p1.y {
                        (p0.y, p1.y, p0.x, p1.x, 1i32)
                    } else {
                        (p1.y, p0.y, p1.x, p0.x, -1i32)
                    };
                    if scan_y >= y0 && scan_y < y1 {
                        let t = (scan_y - y0) / (y1 - y0);
                        crossings.push((x0 + t * (x1 - x0), dir));
                    }
                }
                if crossings.len() < 2 {
                    continue;
                }
                crossings.sort_by(|a, b| a.0.total_cmp(&b.0));

                let mut winding = 0i32;
                let mut span_start: Option<Scalar> = None;
                for &(x, dir) in &crossings {
                    let prev_winding = winding;
                    winding += dir;
                    if prev_winding == 0 && winding != 0 {
                        span_start = Some(x);
                    } else if prev_winding != 0 && winding == 0 {
                        if let Some(sx) = span_start.take() {
                            let lx0 = (sx - min_x as Scalar).clamp(0.0, w as Scalar);
                            let lx1 = (x - min_x as Scalar).clamp(0.0, w as Scalar);
                            if lx1 > lx0 {
                                let col_start = (lx0.floor() as usize).min(w_usize - 1);
                                let col_end = (lx1.ceil() as usize).min(w_usize).max(col_start + 1);
                                for (col, acc) in
                                    accum.iter_mut().enumerate().take(col_end).skip(col_start)
                                {
                                    let px0 = col as Scalar;
                                    let px1 = (col + 1) as Scalar;
                                    let overlap = (lx1.min(px1) - lx0.max(px0)).max(0.0);
                                    *acc += overlap * 0.125;
                                }
                            }
                        }
                    }
                }
            }
            let row_offset = row * w_usize;
            for (col, &cov) in accum.iter().enumerate() {
                bitmap[row_offset + col] = (cov.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }

        self.pack_alpha_mask(key, bearing, w, h, &bitmap)
    }

    pub(crate) fn resolve_or_rasterize_glyph(
        &mut self,
        pair: FontGlyphPair,
        font_id: cosmic_text::fontdb::ID,
        flags: CacheKeyFlags,
    ) -> GlyphCacheValue {
        let key: InternalGlyphCacheKey = (
            font_id,
            pair.glyph_id,
            pair.scaled_font.font_size_milli,
            pair.subpixel_position.x,
            pair.subpixel_position.y,
            flags.bits(),
        );
        if let Some(&cached) = self.glyph_cache.get(&key) {
            return cached;
        }

        let font_size = pair.scaled_font.font_size().max(1.0);
        let cosmic_key = CacheKey {
            font_id,
            glyph_id: pair.glyph_id,
            font_size_bits: font_size.to_bits(),
            x_bin: pair.subpixel_position.x.to_cosmic(),
            y_bin: pair.subpixel_position.y.to_cosmic(),
            flags,
        };

        let Some(image) = self
            .swash_cache
            .get_image_uncached(&mut self.font_system, cosmic_key)
        else {
            return self.rasterize_ttf_outline_mask(key, pair, font_id, font_size);
        };

        let w = i64::from(image.placement.width);
        let h = i64::from(image.placement.height);
        let bearing = IPoint::new(
            i64::from(image.placement.left),
            -i64::from(image.placement.top),
        );
        let size = ISize::new(w, h);

        if w <= 0 || h <= 0 {
            let empty = (Rect::ZERO, false, bearing, ISize::ZERO);
            self.glyph_cache.insert(key, empty);
            return empty;
        }

        // Add 1-pixel zeroed border on all 4 sides to prevent bilinear sampling bleed.
        let padded_w = w + 2;
        let padded_h = h + 2;

        match image.content {
            SwashContent::Mask => self.pack_alpha_mask(key, bearing, w, h, &image.data),
            SwashContent::Color | SwashContent::SubpixelMask => {
                let Some(packed) = Self::pack_with_growth(
                    &mut self.color_packer,
                    &mut self.color_atlas,
                    padded_w,
                    padded_h,
                ) else {
                    return (Rect::ZERO, true, bearing, ISize::ZERO);
                };
                let dst_x = (packed.x + 1) as usize;
                let dst_y = (packed.y + 1) as usize;
                let row_pixels = w as usize;
                let atlas_stride_bytes = (self.color_atlas.size.width as usize) * 4;

                for row in 0..(h as usize) {
                    for col in 0..row_pixels {
                        let dst_offset = (dst_y + row) * atlas_stride_bytes + (dst_x + col) * 4;
                        match image.content {
                            SwashContent::Color => {
                                let src_offset = (row * row_pixels + col) * 4;
                                let r = image.data[src_offset];
                                let g = image.data[src_offset + 1];
                                let b = image.data[src_offset + 2];
                                let a = image.data[src_offset + 3];
                                // Premultiply RGBA8 for ColorBitmap sampling in glyph_atlas.wgsl
                                let af = u16::from(a);
                                self.color_atlas.pixels[dst_offset] =
                                    ((u16::from(r) * af + 127) / 255) as u8;
                                self.color_atlas.pixels[dst_offset + 1] =
                                    ((u16::from(g) * af + 127) / 255) as u8;
                                self.color_atlas.pixels[dst_offset + 2] =
                                    ((u16::from(b) * af + 127) / 255) as u8;
                                self.color_atlas.pixels[dst_offset + 3] = a;
                            }
                            SwashContent::SubpixelMask => {
                                let src_offset = (row * row_pixels + col) * 3;
                                if src_offset + 2 < image.data.len() {
                                    let r = image.data[src_offset];
                                    let g = image.data[src_offset + 1];
                                    let b = image.data[src_offset + 2];
                                    let a = r.max(g).max(b);
                                    self.color_atlas.pixels[dst_offset] = r;
                                    self.color_atlas.pixels[dst_offset + 1] = g;
                                    self.color_atlas.pixels[dst_offset + 2] = b;
                                    self.color_atlas.pixels[dst_offset + 3] = a;
                                }
                            }
                            SwashContent::Mask => unreachable!(),
                        }
                    }
                }
                self.color_atlas.generation = self.color_atlas.generation.wrapping_add(1);
                self.color_atlas.mark_dirty_rows(packed.y, padded_h);

                let atlas_rect =
                    Rect::from_xywh(dst_x as Scalar, dst_y as Scalar, w as Scalar, h as Scalar);
                let val = (atlas_rect, true, bearing, size);
                self.glyph_cache.insert(key, val);
                val
            }
        }
    }

    pub fn shape_text(&mut self, text: &str, style: TextStyle, origin: Point) -> TextFrame {
        if text.is_empty() {
            return TextFrame {
                bounds: Rect::from_xywh(origin.x, origin.y, 0.0, 0.0),
                quads: Vec::new(),
                line_count: 0,
                longest_line: 0.0,
            };
        }

        let buffer = self.shape_buffer(text, style);
        let mut quads = Vec::new();
        let mut line_count = 0usize;
        let mut longest_line: Scalar = 0.0;
        let mut tight_bounds: Option<Rect> = None;

        let font_ceil = style.font_size.max(1.0).ceil();
        let mut first_line_y_pad: Option<Scalar> = None;

        for run in buffer.layout_runs() {
            line_count += 1;
            longest_line = longest_line.max(run.line_w);

            let raw_baseline_y = (origin.y + run.line_y).ceil();
            let y_pad = *first_line_y_pad
                .get_or_insert_with(|| (origin.y.ceil() + font_ceil - raw_baseline_y).max(0.0));
            let baseline_y = raw_baseline_y + y_pad;

            for glyph in run.glyphs {
                let gx = origin.x + glyph.x + glyph.font_size * glyph.x_offset;
                let gy = baseline_y + (glyph.y - glyph.font_size * glyph.y_offset).trunc();
                let (int_pos, subpixel_pos) = SubpixelPosition::quantize(Point::new(gx, gy));

                let resolved_builtin = self.builtin_for_id(glyph.font_id, style.font);
                let scaled_font = ScaledFont::new(resolved_builtin, glyph.font_size);
                let pair = FontGlyphPair::new(scaled_font, glyph.glyph_id, subpixel_pos);

                let (atlas_rect, is_color, bearing, size) =
                    self.resolve_or_rasterize_glyph(pair, glyph.font_id, glyph.cache_key_flags);

                if size.width <= 0 || size.height <= 0 || atlas_rect.is_empty() {
                    continue;
                }

                let dest_rect = Rect::from_xywh(
                    (int_pos.x + bearing.x) as Scalar,
                    (int_pos.y + bearing.y) as Scalar,
                    size.width as Scalar,
                    size.height as Scalar,
                );

                tight_bounds = Some(match tight_bounds {
                    Some(prev) => prev.union(dest_rect),
                    None => dest_rect,
                });

                quads.push(AtlasGlyphQuad {
                    dest_rect,
                    atlas_rect,
                    color: style.color,
                    is_color,
                });
            }
        }

        let bounds = tight_bounds.unwrap_or_else(|| {
            if line_count > 0 {
                Rect::from_xywh(
                    origin.x,
                    origin.y,
                    longest_line,
                    (line_count as Scalar) * style.line_height.max(1.0),
                )
            } else {
                Rect::from_xywh(origin.x, origin.y, 0.0, 0.0)
            }
        });

        TextFrame {
            bounds,
            quads,
            line_count,
            longest_line,
        }
    }

    /// Shapes and rasterizes a single [`MaterialIcon`] glyph into a [`TextFrame`].
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
        self.shape_text(s, style, origin)
    }

    fn append_outline_commands(
        builder: &mut PathBuilder,
        commands: &[Command],
        base_origin: Point,
    ) {
        let map_pt = |x: f32, y: f32| -> Point { Point::new(base_origin.x + x, base_origin.y - y) };
        for &cmd in commands {
            match cmd {
                Command::MoveTo(p) => {
                    builder.move_to(map_pt(p.x, p.y));
                }
                Command::LineTo(p) => {
                    builder.line_to(map_pt(p.x, p.y));
                }
                Command::QuadTo(cp, p) => {
                    builder.quad_to(map_pt(cp.x, cp.y), map_pt(p.x, p.y));
                }
                Command::CurveTo(cp1, cp2, p) => {
                    builder.cubic_to(map_pt(cp1.x, cp1.y), map_pt(cp2.x, cp2.y), map_pt(p.x, p.y));
                }
                Command::Close => {
                    builder.close();
                }
            }
        }
    }

    fn append_glyph_outline_from_ttf(
        font_system: &mut FontSystem,
        font_id: cosmic_text::fontdb::ID,
        glyph_id: u16,
        font_size: Scalar,
        base_origin: Point,
        builder: &mut PathBuilder,
    ) -> bool {
        struct TtfPathOutlineAdapter<'a> {
            builder: &'a mut PathBuilder,
            scale: Scalar,
            base_origin: Point,
        }

        impl TtfPathOutlineAdapter<'_> {
            #[inline]
            fn map_pt(&self, x: f32, y: f32) -> Point {
                Point::new(
                    self.base_origin.x + x * self.scale,
                    self.base_origin.y - y * self.scale,
                )
            }
        }

        impl cosmic_text::ttf_parser::OutlineBuilder for TtfPathOutlineAdapter<'_> {
            fn move_to(&mut self, x: f32, y: f32) {
                self.builder.move_to(self.map_pt(x, y));
            }

            fn line_to(&mut self, x: f32, y: f32) {
                self.builder.line_to(self.map_pt(x, y));
            }

            fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
                self.builder.quad_to(self.map_pt(x1, y1), self.map_pt(x, y));
            }

            fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
                self.builder
                    .cubic_to(self.map_pt(x1, y1), self.map_pt(x2, y2), self.map_pt(x, y));
            }

            fn close(&mut self) {
                self.builder.close();
            }
        }

        let Some(font) = font_system.get_font(font_id) else {
            return false;
        };
        let face = font.rustybuzz();
        let upem = face.units_per_em().max(1) as Scalar;
        let scale = font_size / upem;
        let mut adapter = TtfPathOutlineAdapter {
            builder,
            scale,
            base_origin,
        };
        face.outline_glyph(cosmic_text::ttf_parser::GlyphId(glyph_id), &mut adapter)
            .is_some()
    }

    /// Extracts the vector outline of a single character `ch` in `font` at `font_size`
    /// placed with its baseline origin at `origin` (`impeller_path::Path`).
    pub fn glyph_to_path(
        &mut self,
        ch: char,
        font: BuiltinFont,
        font_size: Scalar,
        origin: Point,
    ) -> Path {
        let clamped_size = font_size.max(1.0);
        let reg_id = self.registered_font(font).id;

        let direct_glyph = self
            .font_system
            .get_font(reg_id)
            .and_then(|f| f.rustybuzz().glyph_index(ch))
            .map(|gid| (reg_id, gid.0));

        let (font_id, glyph_id) = match direct_glyph {
            Some((fid, gid)) if gid != 0 => (fid, gid),
            _ => {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                let shaped = self.shape_buffer(
                    s,
                    TextStyle {
                        font,
                        font_size: clamped_size,
                        line_height: clamped_size,
                        color: Color::WHITE,
                        max_width: None,
                    },
                );
                let first = shaped
                    .layout_runs()
                    .flat_map(|r| r.glyphs.iter())
                    .next()
                    .map(|g| (g.font_id, g.glyph_id));
                let Some((fid, gid)) = first else {
                    return Path::default();
                };
                if gid == 0 {
                    return Path::default();
                }
                (fid, gid)
            }
        };

        let cache_key = CacheKey {
            font_id,
            glyph_id,
            font_size_bits: clamped_size.to_bits(),
            x_bin: cosmic_text::SubpixelBin::Zero,
            y_bin: cosmic_text::SubpixelBin::Zero,
            flags: CacheKeyFlags::empty(),
        };

        let mut builder = PathBuilder::new();
        if let Some(commands) = self
            .swash_cache
            .get_outline_commands(&mut self.font_system, cache_key)
        {
            Self::append_outline_commands(&mut builder, commands, origin);
        } else {
            Self::append_glyph_outline_from_ttf(
                &mut self.font_system,
                font_id,
                glyph_id,
                clamped_size,
                origin,
                &mut builder,
            );
        }
        builder.take_path(FillType::NonZero)
    }

    /// Shapes `text` with `style` and extracts the combined vector outlines of all shaped glyphs
    /// into a single [`Path`] relative to `origin`.
    pub fn text_to_path(&mut self, text: &str, style: TextStyle, origin: Point) -> Path {
        if text.is_empty() {
            return Path::default();
        }
        let buffer = self.shape_buffer(text, style);
        let mut builder = PathBuilder::new();

        for run in buffer.layout_runs() {
            for glyph in run.glyphs {
                if glyph.glyph_id == 0 {
                    continue;
                }
                let gx = origin.x + glyph.x + glyph.font_size * glyph.x_offset;
                let gy = origin.y + run.line_y + glyph.y - glyph.font_size * glyph.y_offset;
                let clamped_size = glyph.font_size.max(1.0);
                let cache_key = CacheKey {
                    font_id: glyph.font_id,
                    glyph_id: glyph.glyph_id,
                    font_size_bits: clamped_size.to_bits(),
                    x_bin: cosmic_text::SubpixelBin::Zero,
                    y_bin: cosmic_text::SubpixelBin::Zero,
                    flags: glyph.cache_key_flags,
                };
                if let Some(commands) = self
                    .swash_cache
                    .get_outline_commands(&mut self.font_system, cache_key)
                {
                    Self::append_outline_commands(&mut builder, commands, Point::new(gx, gy));
                } else {
                    Self::append_glyph_outline_from_ttf(
                        &mut self.font_system,
                        glyph.font_id,
                        glyph.glyph_id,
                        clamped_size,
                        Point::new(gx, gy),
                        &mut builder,
                    );
                }
            }
        }
        builder.take_path(FillType::NonZero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// C++ source: rectangle_packer_unittests.cc:RectanglePackerCanPackSingleRect
    #[test]
    fn test_rectangle_packer_single_rect() {
        let mut packer = RectanglePacker::factory(128, 128);
        assert!(packer.is_empty());
        assert_eq!(packer.size(), ISize::new(128, 128));
        assert_abs_diff_eq!(packer.percent_full(), 0.0);

        let pt = packer.add_rect(32, 16).expect("should pack 32x16 rect");
        assert_eq!(pt, IPoint::new(0, 0));
        assert!(!packer.is_empty());
        assert_abs_diff_eq!(
            packer.percent_full(),
            (32.0 * 16.0) / (128.0 * 128.0),
            epsilon = 1e-6
        );
    }

    /// C++ source: rectangle_packer_unittests.cc:RectanglePackerFillsRows
    #[test]
    fn test_rectangle_packer_fills_rows() {
        let mut packer = SkylineRectanglePacker::new(257, 256);
        let mut last_pt = IPoint::ZERO;
        // 16 rects of width 16 fit along the first shelf (16 * 16 = 256 <= 257).
        for _ in 0..16 {
            last_pt = packer.add_rect(16, 16).expect("first row should fit");
        }
        assert_eq!(last_pt, IPoint::new(15 * 16, 0));

        // Next 16 rects should occupy the second shelf at y = 16.
        for _ in 0..16 {
            last_pt = packer.add_rect(16, 16).expect("second row should fit");
        }
        assert_eq!(last_pt, IPoint::new(15 * 16, 16));
    }

    /// C++ source: rectangle_packer_unittests.cc:RectanglePackerAddsNonoverlapingRectangles
    #[test]
    fn test_rectangle_packer_nonoverlapping_rectangles() {
        let mut packer = RectanglePacker::new(128, 128);
        let specs = [(24, 20), (40, 20), (50, 20), (60, 32), (48, 32), (32, 40)];
        let mut placed: Vec<Rect> = Vec::new();

        for (w, h) in specs {
            let pt = packer
                .add_rect(w, h)
                .expect("rectangle should fit in 128x128");
            let rect = Rect::from_xywh(pt.x as Scalar, pt.y as Scalar, w as Scalar, h as Scalar);
            assert!(rect.get_right() <= 128.0 && rect.get_bottom() <= 128.0);
            for prev in &placed {
                assert!(
                    !rect.intersects_with_rect(*prev),
                    "packed rect {:?} overlaps {:?}",
                    rect,
                    prev
                );
            }
            placed.push(rect);
        }
    }

    /// C++ source: rectangle_packer_unittests.cc:RectanglePackerRejectsInvalidAndOversizedRects
    #[test]
    fn test_rectangle_packer_rejects_invalid_and_oversized() {
        let mut packer = RectanglePacker::new(64, 64);
        assert_eq!(packer.add_rect(0, 10), None);
        assert_eq!(packer.add_rect(10, 0), None);
        assert_eq!(packer.add_rect(-4, 10), None);
        assert_eq!(packer.add_rect(65, 16), None);
        assert_eq!(packer.add_rect(16, 65), None);
        assert!(packer.is_empty());
    }

    /// C++ source: rectangle_packer_unittests.cc:RectanglePackerPercentFullAndReset
    #[test]
    fn test_rectangle_packer_percent_full_and_reset() {
        let mut packer = RectanglePacker::new(100, 100);
        assert!(packer.add_rect(50, 50).is_some());
        assert_abs_diff_eq!(packer.percent_full(), 0.25, epsilon = 1e-6);

        packer.reset();
        assert!(packer.is_empty());
        assert_abs_diff_eq!(packer.percent_full(), 0.0, epsilon = 1e-6);
        assert_eq!(packer.add_rect(50, 50), Some(IPoint::ZERO));
    }

    /// C++ source: rectangle_packer_unittests.cc:RectanglePackerCanDeallocateAndReuseSpace
    #[test]
    fn test_rectangle_packer_deallocate_and_reuse() {
        let mut packer = RectanglePacker::new(64, 64);
        let (id_a, pt_a) = packer.allocate(32, 32).expect("alloc A");
        let (id_b, pt_b) = packer.allocate(32, 32).expect("alloc B");
        assert_ne!(pt_a, pt_b);
        assert_abs_diff_eq!(packer.percent_full(), 0.5, epsilon = 1e-6);

        assert!(packer.deallocate(id_a));
        assert!(!packer.deallocate(id_a));
        assert_abs_diff_eq!(packer.percent_full(), 0.25, epsilon = 1e-6);

        // Re-allocating 32x32 should succeed in the reclaimed space.
        let (_id_c, pt_c) = packer.allocate(32, 32).expect("alloc C in reclaimed shelf");
        assert_eq!(pt_c, pt_a);
        assert!(packer.deallocate(id_b));
    }

    /// C++ source: rectangle_packer_unittests.cc:RectanglePackerCanGrowWhenFull
    #[test]
    fn test_rectangle_packer_grow_when_full() {
        let mut packer = RectanglePacker::new(32, 32);
        let (id0, pt0) = packer.allocate(32, 32).expect("fill 32x32");
        assert_eq!(pt0, IPoint::ZERO);
        assert_eq!(packer.add_rect(16, 16), None);

        // Shrinking must fail.
        assert!(!packer.grow(ISize::new(16, 16)));

        // Growing to 64x64 preserves existing allocation at (0, 0) and opens new shelf regions.
        assert!(packer.grow(ISize::new(64, 64)));
        assert_eq!(packer.size(), ISize::new(64, 64));

        let pt1 = packer
            .add_rect(24, 24)
            .expect("should pack into grown region");
        let r0 = Rect::from_xywh(pt0.x as Scalar, pt0.y as Scalar, 32.0, 32.0);
        let r1 = Rect::from_xywh(pt1.x as Scalar, pt1.y as Scalar, 24.0, 24.0);
        assert!(!r0.intersects_with_rect(r1));
        assert!(packer.deallocate(id0));
    }

    /// C++ source: typographer_unittests.cc:SubpixelBinQuantizationBoundaries
    #[test]
    fn test_subpixel_bin_quantization_boundaries() {
        assert_eq!(SubpixelBin::from_fract(0.0), SubpixelBin::Zero);
        assert_eq!(SubpixelBin::from_fract(0.124), SubpixelBin::Zero);
        assert_eq!(SubpixelBin::from_fract(0.125), SubpixelBin::Quarter);
        assert_eq!(SubpixelBin::from_fract(0.374), SubpixelBin::Quarter);
        assert_eq!(SubpixelBin::from_fract(0.375), SubpixelBin::Half);
        assert_eq!(SubpixelBin::from_fract(0.624), SubpixelBin::Half);
        assert_eq!(SubpixelBin::from_fract(0.625), SubpixelBin::ThreeQuarters);
        assert_eq!(SubpixelBin::from_fract(0.874), SubpixelBin::ThreeQuarters);
        assert_eq!(SubpixelBin::from_fract(0.875), SubpixelBin::Zero);

        assert_abs_diff_eq!(SubpixelBin::Zero.offset(), 0.0);
        assert_abs_diff_eq!(SubpixelBin::Quarter.offset(), 0.25);
        assert_abs_diff_eq!(SubpixelBin::Half.offset(), 0.5);
        assert_abs_diff_eq!(SubpixelBin::ThreeQuarters.offset(), 0.75);
    }

    /// C++ source: typographer_unittests.cc:SubpixelPositionAndFontGlyphPairEquality
    #[test]
    fn test_subpixel_position_and_font_glyph_pair_cache_key() {
        let (int_a, sub_a) = SubpixelPosition::quantize(Point::new(10.9, -0.25));
        assert_eq!(int_a, IPoint::new(11, -1));
        assert_eq!(
            sub_a,
            SubpixelPosition::new(SubpixelBin::Zero, SubpixelBin::ThreeQuarters)
        );

        let sf1 = ScaledFont::new(BuiltinFont::RobotoBold, 16.5);
        assert_eq!(sf1.font_size_milli, 16_500);
        assert_abs_diff_eq!(sf1.font_size(), 16.5, epsilon = 1e-4);

        let pair1 = FontGlyphPair::new(sf1, 42, sub_a);
        let pair2 = FontGlyphPair::new(sf1, 42, sub_a);
        let pair3 = FontGlyphPair::new(sf1, 42, SubpixelPosition::ZERO);
        assert_eq!(pair1.cache_key(), pair2.cache_key());
        assert_ne!(pair1.cache_key(), pair3.cache_key());
    }

    /// C++ source: typographer_unittests.cc:CanCreateTypographerContextWithEmbeddedFonts
    #[test]
    fn test_typographer_context_initializes_embedded_fonts_without_fs() {
        let ctx = TypographerContext::new();
        assert_eq!(ctx.alpha_atlas.size, ISize::new(1024, 1024));
        assert_eq!(ctx.color_atlas.size, ISize::new(512, 512));
        assert_eq!(ctx.cached_glyph_count(), 0);
        // Only the 4 embedded font slices are loaded into the font database.
        assert_eq!(ctx.font_system.db().faces().count(), 4);
    }

    /// C++ source: typographer_unittests.cc:CanShapeEmptyAndWhitespaceTextFrames
    #[test]
    fn test_shape_empty_and_whitespace_text() {
        let mut ctx = TypographerContext::new();
        let empty_frame = ctx.shape_text("", TextStyle::default(), Point::new(5.0, 10.0));
        assert_eq!(empty_frame.line_count, 0);
        assert!(empty_frame.quads.is_empty());
        assert_eq!(empty_frame.bounds, Rect::from_xywh(5.0, 10.0, 0.0, 0.0));

        let ws_frame = ctx.shape_text("    ", TextStyle::default(), Point::new(5.0, 10.0));
        assert_eq!(ws_frame.line_count, 1);
        assert!(ws_frame.quads.is_empty());
        assert!(ws_frame.longest_line > 0.0);
    }

    /// C++ source: typographer_unittests.cc:CanShapeSingleLineTextFrame
    #[test]
    fn test_shape_single_line_text_frame() {
        let mut ctx = TypographerContext::new();
        let style = TextStyle {
            font: BuiltinFont::RobotoRegular,
            font_size: 20.0,
            line_height: 24.0,
            color: Color::RED,
            max_width: None,
        };
        let frame = ctx.shape_text("Impeller", style, Point::new(10.0, 12.0));
        assert_eq!(frame.line_count, 1);
        assert_eq!(frame.quads.len(), 8);
        assert!(frame.longest_line > 50.0);
        assert!(!frame.bounds.is_empty());

        for w in frame.quads.windows(2) {
            assert!(
                w[1].dest_rect.get_left() >= w[0].dest_rect.get_left(),
                "LTR glyphs should advance monotonically along X"
            );
            assert_eq!(w[0].color, Color::RED);
            assert!(!w[0].is_color);
        }
    }

    /// C++ source: typographer_unittests.cc:CanWrapMultiLineTextFrame
    #[test]
    fn test_shape_wrapped_multi_line_text_frame() {
        let mut ctx = TypographerContext::new();
        let text = "Impeller shelf rectangle packing and multi line word wrapping";
        let unwrapped = ctx.shape_text(
            text,
            TextStyle {
                font: BuiltinFont::RobotoRegular,
                font_size: 16.0,
                line_height: 20.0,
                color: Color::WHITE,
                max_width: None,
            },
            Point::ZERO,
        );
        let wrapped = ctx.shape_text(
            text,
            TextStyle {
                font: BuiltinFont::RobotoRegular,
                font_size: 16.0,
                line_height: 20.0,
                color: Color::WHITE,
                max_width: Some(120.0),
            },
            Point::ZERO,
        );

        assert_eq!(unwrapped.line_count, 1);
        assert!(
            wrapped.line_count >= 3,
            "expected >= 3 wrapped lines, got {}",
            wrapped.line_count
        );
        assert!(wrapped.longest_line <= 125.0);
        assert!(wrapped.bounds.get_height() > unwrapped.bounds.get_height() * 2.0);
    }

    /// C++ source: typographer_unittests.cc:CanShapeExplicitNewlines
    #[test]
    fn test_shape_explicit_newlines() {
        let mut ctx = TypographerContext::new();
        let frame = ctx.shape_text(
            "Line1\nLine2\nLine3",
            TextStyle {
                font: BuiltinFont::RobotoBold,
                font_size: 14.0,
                line_height: 18.0,
                color: Color::WHITE,
                max_width: None,
            },
            Point::ZERO,
        );
        assert_eq!(frame.line_count, 3);
        assert_eq!(frame.quads.len(), 15);
    }

    /// C++ source: typographer_unittests.cc:CanShapeBidiMixedLtrAndRtlOrdering
    #[test]
    fn test_shape_bidi_ltr_and_rtl_ordering() {
        let mut ctx = TypographerContext::new();
        let style = TextStyle {
            font: BuiltinFont::RobotoRegular,
            font_size: 20.0,
            line_height: 24.0,
            color: Color::WHITE,
            max_width: None,
        };
        // Shapes mixed LTR prefix ("i ") + two tokens ("L W") in normal LTR vs RTL override
        // ("i \u{202E}L W\u{202C}"). In RobotoRegular, width('i') < width('L') < width('W').
        // In pure LTR ("i L W"), left-to-right visual order is ['i', 'L', 'W'].
        // In mixed LTR+RTL ("i \u{202E}L W\u{202C}"), the LTR prefix 'i' stays leftmost while
        // the RTL span reorders its tokens visually to ['W', 'L'], giving ['i', 'W', 'L'].
        let ltr = ctx.shape_text("i L W", style, Point::ZERO);
        let mixed_bidi = ctx.shape_text("i \u{202E}L W\u{202C}", style, Point::ZERO);
        assert_eq!(ltr.quads.len(), 3);
        assert_eq!(mixed_bidi.quads.len(), 3);

        let mut ltr_sorted = ltr.quads.clone();
        ltr_sorted.sort_by(|a, b| a.dest_rect.get_left().total_cmp(&b.dest_rect.get_left()));
        let mut bidi_sorted = mixed_bidi.quads.clone();
        bidi_sorted.sort_by(|a, b| a.dest_rect.get_left().total_cmp(&b.dest_rect.get_left()));

        // Leftmost glyph in both is LTR 'i' (narrowest).
        assert_eq!(
            ltr_sorted[0].dest_rect.get_width(),
            bidi_sorted[0].dest_rect.get_width()
        );
        // In pure LTR, middle glyph 'L' is narrower than rightmost glyph 'W'.
        assert!(ltr_sorted[1].dest_rect.get_width() < ltr_sorted[2].dest_rect.get_width());
        // In mixed LTR+RTL, middle glyph 'W' is wider than rightmost glyph 'L'.
        assert!(bidi_sorted[1].dest_rect.get_width() > bidi_sorted[2].dest_rect.get_width());
    }

    /// C++ source: typographer_unittests.cc:CanShapeMaterialIconsCodepoints
    #[test]
    fn test_shape_material_icons_all_variants() {
        assert_eq!(MaterialIcon::ShowChart.codepoint(), '\u{e59f}');
        assert_eq!(MaterialIcon::Speed.codepoint(), '\u{e5e0}');
        assert_eq!(MaterialIcon::Bolt.codepoint(), '\u{e0ee}');
        assert_eq!(MaterialIcon::AutoAwesome.codepoint(), '\u{e0b7}');
        assert_eq!(MaterialIcon::CheckCircle.codepoint(), '\u{e159}');
        assert_eq!(MaterialIcon::RadioButtonUnchecked.codepoint(), '\u{e504}');
        assert_eq!(MaterialIcon::Memory.codepoint(), '\u{e3db}');

        let mut ctx = TypographerContext::new();
        let icons = [
            MaterialIcon::ShowChart,
            MaterialIcon::Speed,
            MaterialIcon::Bolt,
            MaterialIcon::AutoAwesome,
            MaterialIcon::CheckCircle,
            MaterialIcon::RadioButtonUnchecked,
            MaterialIcon::Memory,
        ];

        for icon in icons {
            let frame = ctx.shape_icon(icon, 28.0, Color::WHITE, Point::new(4.0, 4.0));
            assert_eq!(frame.line_count, 1);
            assert_eq!(
                frame.quads.len(),
                1,
                "expected 1 rasterized glyph quad for {:?}",
                icon
            );
            let q = frame.quads[0];
            assert!(q.dest_rect.get_width() >= 10.0 && q.dest_rect.get_height() >= 10.0);
            assert!(q.atlas_rect.get_width() >= 10.0 && q.atlas_rect.get_height() >= 10.0);
        }
    }

    /// C++ source: typographer_unittests.cc:GlyphAtlasReusesCachedGlyphsAcrossFrames
    #[test]
    fn test_glyph_atlas_caches_repeated_glyphs_across_frames() {
        let mut ctx = TypographerContext::new();
        let style = TextStyle {
            font: BuiltinFont::RobotoBold,
            font_size: 18.0,
            line_height: 22.0,
            color: Color::WHITE,
            max_width: None,
        };

        let frame1 = ctx.shape_text("AAAA", style, Point::new(10.0, 10.0));
        assert_eq!(frame1.quads.len(), 4);
        let gen_after_first = ctx.alpha_atlas.generation;
        let count_after_first = ctx.cached_glyph_count();
        assert!(count_after_first >= 1);

        // Shaping the exact same text at the same subpixel alignment must reuse the cached
        // atlas rects without bumping generation or allocating new cache entries.
        let frame2 = ctx.shape_text("AAAA", style, Point::new(20.0, 30.0));
        assert_eq!(ctx.alpha_atlas.generation, gen_after_first);
        assert_eq!(ctx.cached_glyph_count(), count_after_first);
        for (q1, q2) in frame1.quads.iter().zip(frame2.quads.iter()) {
            assert_eq!(q1.atlas_rect, q2.atlas_rect);
        }

        // Shifting by 0.5px introduces a new SubpixelBin::Half cache entry.
        let _frame3 = ctx.shape_text("A", style, Point::new(20.5, 30.0));
        assert!(ctx.alpha_atlas.generation > gen_after_first);
        assert!(ctx.cached_glyph_count() > count_after_first);
    }

    /// C++ source: typographer_unittests.cc:GlyphAtlasHasOnePixelBorderPadding
    #[test]
    fn test_glyph_atlas_one_pixel_border_padding() {
        let mut ctx = TypographerContext::new();
        let frame = ctx.shape_icon(MaterialIcon::CheckCircle, 32.0, Color::WHITE, Point::ZERO);
        assert_eq!(frame.quads.len(), 1);
        let ar = frame.quads[0].atlas_rect;
        let x0 = ar.get_left() as usize;
        let y0 = ar.get_top() as usize;
        let w = ar.get_width() as usize;
        let h = ar.get_height() as usize;
        let stride = ctx.alpha_atlas.size.width as usize;

        assert!(x0 >= 1 && y0 >= 1);
        // Verify top and bottom 1px border rows are strictly 0.
        for x in (x0 - 1)..=(x0 + w) {
            assert_eq!(ctx.alpha_atlas.pixels[(y0 - 1) * stride + x], 0);
            assert_eq!(ctx.alpha_atlas.pixels[(y0 + h) * stride + x], 0);
        }
        // Verify left and right 1px border columns are strictly 0.
        for y in (y0 - 1)..=(y0 + h) {
            assert_eq!(ctx.alpha_atlas.pixels[y * stride + (x0 - 1)], 0);
            assert_eq!(ctx.alpha_atlas.pixels[y * stride + (x0 + w)], 0);
        }
    }

    /// C++ source: typographer_unittests.cc:GlyphAtlasAutomaticallyGrowsWhenFull
    #[test]
    fn test_glyph_atlas_auto_grows_when_packer_fills() {
        let mut ctx = TypographerContext::with_atlas_sizes(64, 64);
        assert_eq!(ctx.alpha_atlas.size, ISize::new(64, 64));

        // Rasterize several large glyphs that exceed a 64x64 atlas, triggering automatic growth.
        let frame = ctx.shape_text(
            "WMQ@#%&8",
            TextStyle {
                font: BuiltinFont::RobotoBold,
                font_size: 42.0,
                line_height: 48.0,
                color: Color::WHITE,
                max_width: None,
            },
            Point::ZERO,
        );
        assert_eq!(frame.quads.len(), 8);
        assert!(
            ctx.alpha_atlas.size.width >= 128 && ctx.alpha_atlas.size.height >= 128,
            "expected alpha_atlas to grow beyond 64x64, got {:?}",
            ctx.alpha_atlas.size
        );
        // Verify every packed glyph quad in the grown atlas has non-zero coverage pixels.
        let stride = ctx.alpha_atlas.size.width as usize;
        for q in &frame.quads {
            let x0 = q.atlas_rect.get_left() as usize;
            let y0 = q.atlas_rect.get_top() as usize;
            let w = q.atlas_rect.get_width() as usize;
            let h = q.atlas_rect.get_height() as usize;
            let mut sum = 0u64;
            for r in 0..h {
                for c in 0..w {
                    sum += u64::from(ctx.alpha_atlas.pixels[(y0 + r) * stride + (x0 + c)]);
                }
            }
            assert!(
                sum > 0,
                "glyph quad {:?} has empty atlas pixels after grow",
                q
            );
        }
    }

    /// C++ source: typographer_unittests.cc:CanExtractGlyphOutlineToPath
    #[test]
    fn test_glyph_to_path_generates_non_empty_contours() {
        let mut ctx = TypographerContext::new();
        let path_o = ctx.glyph_to_path('O', BuiltinFont::RobotoBold, 64.0, Point::new(20.0, 80.0));
        assert!(!path_o.is_empty());
        assert!(
            !path_o.is_single_contour(),
            "'O' should contain both outer and inner contours"
        );
        let bounds_o = path_o.get_bounding_box().expect("bounding box for 'O'");
        assert!(bounds_o.get_width() > 20.0 && bounds_o.get_height() > 30.0);
        // Baseline is at y = 80.0, so top of 'O' must be above 80.0.
        assert!(bounds_o.get_top() < 80.0 && bounds_o.get_bottom() <= 84.0);

        let icon_path = ctx.glyph_to_path(
            MaterialIcon::Bolt.codepoint(),
            BuiltinFont::MaterialIcons,
            48.0,
            Point::new(10.0, 60.0),
        );
        assert!(!icon_path.is_empty());
    }

    /// C++ source: typographer_unittests.cc:CanExtractShapedTextOutlineToPath
    #[test]
    fn test_text_to_path_generates_positioned_word_contours() {
        let mut ctx = TypographerContext::new();
        let style = TextStyle {
            font: BuiltinFont::RobotoRegular,
            font_size: 32.0,
            line_height: 38.0,
            color: Color::WHITE,
            max_width: None,
        };
        let single = ctx.text_to_path("R", style, Point::ZERO);
        let word = ctx.text_to_path("Rust", style, Point::ZERO);
        assert!(!single.is_empty());
        assert!(!word.is_empty());

        let single_bb = single.get_bounding_box().expect("single bounds");
        let word_bb = word.get_bounding_box().expect("word bounds");
        assert!(word_bb.get_width() > single_bb.get_width() * 2.5);
        assert!(word.get_component_count() > single.get_component_count() * 2);
    }

    /// C++ source: typographer_unittests.cc:DifferentBuiltinFontsProduceDistinctMetrics
    #[test]
    fn test_builtin_fonts_distinct_metrics_and_weights() {
        let mut ctx = TypographerContext::new();
        let mono_i = ctx.shape_text(
            "iiii",
            TextStyle {
                font: BuiltinFont::RobotoMonoBold,
                font_size: 20.0,
                line_height: 24.0,
                color: Color::WHITE,
                max_width: None,
            },
            Point::ZERO,
        );
        let mono_w = ctx.shape_text(
            "WWWW",
            TextStyle {
                font: BuiltinFont::RobotoMonoBold,
                font_size: 20.0,
                line_height: 24.0,
                color: Color::WHITE,
                max_width: None,
            },
            Point::ZERO,
        );
        let prop_i = ctx.shape_text(
            "iiii",
            TextStyle {
                font: BuiltinFont::RobotoRegular,
                font_size: 20.0,
                line_height: 24.0,
                color: Color::WHITE,
                max_width: None,
            },
            Point::ZERO,
        );
        let prop_w = ctx.shape_text(
            "WWWW",
            TextStyle {
                font: BuiltinFont::RobotoRegular,
                font_size: 20.0,
                line_height: 24.0,
                color: Color::WHITE,
                max_width: None,
            },
            Point::ZERO,
        );

        // Monospace font advances 'iiii' and 'WWWW' identically, while proportional Roboto differs significantly.
        assert_abs_diff_eq!(mono_i.longest_line, mono_w.longest_line, epsilon = 0.5);
        assert!(prop_w.longest_line > prop_i.longest_line * 1.8);
    }

    /// C++ source: typographer_unittests.cc:GlyphAtlasUppercaseAscenderScanlineFidelity
    #[test]
    fn test_uppercase_top_scanline_not_cropped_across_sizes() {
        let mut ctx = TypographerContext::new();
        let sizes: [Scalar; 5] = [10.0, 11.0, 12.5, 14.0, 18.0];
        let origins = [
            Point::new(0.0, 0.0),
            Point::new(3.25, 4.35),
            Point::new(7.5, 10.75),
            Point::new(12.8, 19.9),
        ];
        let phrases = ["CLAMP", "MIRROR", "Q1 · RRECT BLUR", "WAVE #0"];

        for &font_size in &sizes {
            for &origin in &origins {
                for &phrase in &phrases {
                    let style = TextStyle {
                        font: BuiltinFont::RobotoBold,
                        font_size,
                        line_height: font_size,
                        color: Color::WHITE,
                        max_width: None,
                    };
                    let frame = ctx.shape_text(phrase, style, origin);
                    assert!(!frame.quads.is_empty());

                    for (idx, q) in frame.quads.iter().enumerate() {
                        assert!(
                            q.dest_rect.get_top() >= origin.y,
                            "phrase={phrase:?} size={font_size} origin={origin:?} quad[{idx}] top={} < origin.y={}",
                            q.dest_rect.get_top(),
                            origin.y
                        );
                    }

                    if phrase == "CLAMP" || phrase == "MIRROR" {
                        let chars: Vec<char> = phrase.chars().collect();
                        assert_eq!(frame.quads.len(), chars.len());

                        // Flat-top / flat-bottom reference glyph ('L' in CLAMP, 'M' in MIRROR)
                        let ref_idx = if phrase == "CLAMP" { 1 } else { 0 };
                        let ref_top = frame.quads[ref_idx].dest_rect.get_top();
                        let ref_bottom = frame.quads[ref_idx].dest_rect.get_bottom();

                        for (ch, q) in chars.iter().copied().zip(frame.quads.iter()) {
                            let top_diff = (q.dest_rect.get_top() - ref_top).abs();
                            if matches!(ch, 'C' | 'O' | 'Q') {
                                assert!(
                                    top_diff <= 1.0,
                                    "phrase={phrase:?} size={font_size} char={ch:?} top={} vs ref_top={ref_top}",
                                    q.dest_rect.get_top()
                                );
                                assert!(
                                    (q.dest_rect.get_bottom() - ref_bottom).abs() <= 1.0,
                                    "phrase={phrase:?} size={font_size} char={ch:?} bottom={} vs ref_bottom={ref_bottom}",
                                    q.dest_rect.get_bottom()
                                );
                            } else {
                                assert_eq!(
                                    q.dest_rect.get_top(),
                                    ref_top,
                                    "flat-top char={ch:?} in {phrase:?} at size={font_size} must match ref_top"
                                );
                                assert_eq!(
                                    q.dest_rect.get_bottom(),
                                    ref_bottom,
                                    "flat-bottom char={ch:?} in {phrase:?} at size={font_size} must match ref_bottom"
                                );
                            }
                        }
                    }
                }

                // Verify top row (row = 0) of flat-top uppercase glyphs (L, M, P, I, R, E, W)
                // in alpha_atlas.pixels has crisp uncropped peak coverage >= 180.
                let flat_top_chars = "LMPIREW";
                let flat_frame = ctx.shape_text(
                    flat_top_chars,
                    TextStyle {
                        font: BuiltinFont::RobotoBold,
                        font_size,
                        line_height: font_size * 1.2,
                        color: Color::WHITE,
                        max_width: None,
                    },
                    origin,
                );
                assert_eq!(flat_frame.quads.len(), flat_top_chars.len());
                let stride = ctx.alpha_atlas.size.width as usize;
                for (ch, q) in flat_top_chars.chars().zip(flat_frame.quads.iter()) {
                    let x0 = q.atlas_rect.get_left() as usize;
                    let y0 = q.atlas_rect.get_top() as usize;
                    let w = q.atlas_rect.get_width() as usize;
                    let top_row_max = (0..w)
                        .map(|c| ctx.alpha_atlas.pixels[y0 * stride + x0 + c])
                        .max()
                        .unwrap_or(0);
                    assert!(
                        top_row_max >= 180,
                        "char={ch:?} size={font_size} origin={origin:?} top row max coverage={top_row_max} < 180"
                    );
                }
            }
        }
    }

    /// C++ source: typographer_unittests.cc:GlyphAtlasDirtyRowRangeAndResizeEpochTracking
    #[test]
    fn test_glyph_atlas_dirty_row_range_resize_epoch_and_descriptor() {
        let mut ctx = TypographerContext::with_atlas_sizes(64, 64);
        assert_eq!(ctx.alpha_atlas.generation, 1);
        assert_eq!(ctx.alpha_atlas.resize_epoch, 1);
        assert_eq!(ctx.alpha_atlas.row_generations.len(), 64);
        assert_eq!(ctx.alpha_atlas.dirty_row_range_since(1), None);
        assert_eq!(ctx.alpha_atlas.dirty_row_range_since(0), Some((0, 64)));

        let initial_id = ctx.alpha_atlas.atlas_id;
        assert!(initial_id >= 1);

        // Shape a single glyph; generation increases, resize_epoch stays 1, dirty rows cover only the glyph's padded span.
        let frame = ctx.shape_text(
            "A",
            TextStyle {
                font: BuiltinFont::RobotoRegular,
                font_size: 16.0,
                line_height: 20.0,
                color: Color::WHITE,
                max_width: None,
            },
            Point::new(0.0, 0.0),
        );
        assert_eq!(frame.quads.len(), 1);
        let gen_after_a = ctx.alpha_atlas.generation;
        assert!(gen_after_a > 1);
        assert_eq!(ctx.alpha_atlas.resize_epoch, 1);

        let dirty_a = ctx
            .alpha_atlas
            .dirty_row_range_since(1)
            .expect("dirty row range after packing 'A'");
        let glyph_h = frame.quads[0].atlas_rect.get_height() as u32;
        assert_eq!(dirty_a.0, 0);
        // Padded height is glyph_h + 2 (1px top + 1px bottom border).
        assert_eq!(dirty_a.1, glyph_h + 2);
        assert!(dirty_a.1 < 64);
        assert_eq!(ctx.alpha_atlas.dirty_row_range_since(gen_after_a), None);

        // Zero-allocation descriptor preserves metadata without cloning pixels/row_generations.
        let desc = ctx.alpha_atlas.descriptor_without_pixels();
        assert_eq!(desc.atlas_id, initial_id);
        assert_eq!(desc.atlas_type, GlyphAtlasType::AlphaBitmap);
        assert_eq!(desc.size, ctx.alpha_atlas.size);
        assert_eq!(desc.generation, gen_after_a);
        assert_eq!(desc.resize_epoch, 1);
        assert!(desc.pixels.is_empty());
        assert_eq!(desc.pixels.capacity(), 0);
        assert!(desc.row_generations.is_empty());
        assert_eq!(desc.row_generations.capacity(), 0);

        // Force atlas growth; resize_epoch increments and row_generations resizes.
        ctx.alpha_atlas.grow(128, 128);
        assert_eq!(ctx.alpha_atlas.atlas_id, initial_id);
        assert_eq!(ctx.alpha_atlas.resize_epoch, 2);
        assert_eq!(ctx.alpha_atlas.row_generations.len(), 128);
        assert_eq!(
            ctx.alpha_atlas.dirty_row_range_since(gen_after_a),
            Some((0, 128))
        );

        // clear_caches increments resize_epoch again and resets all row_generations.
        let gen_after_grow = ctx.alpha_atlas.generation;
        ctx.clear_caches();
        assert_eq!(ctx.alpha_atlas.resize_epoch, 3);
        assert!(ctx.alpha_atlas.generation > gen_after_grow);
        assert_eq!(
            ctx.alpha_atlas.dirty_row_range_since(gen_after_grow),
            Some((0, 128))
        );
    }

    /// C++ source: typographer_unittests.cc:EmptyTypographerContextPopulatesFallbackSlotsOnRegisterFont
    #[test]
    fn test_typographer_context_empty_init_and_dynamic_fallback_slots() {
        let mut ctx = TypographerContext::with_atlas_sizes_empty(512, 256);
        assert_eq!(ctx.font_system.db().faces().count(), 0);

        // Register only Roboto-Regular ("Roboto") and MaterialIcons ("MaterialIcons"), matching Flutter Web FontManifest.json.
        let roboto_ids = ctx.register_font_bytes(ROBOTO_REGULAR_TTF, Some("Roboto"));
        let icon_ids = ctx.register_font_bytes(MATERIAL_ICONS_OTF, Some("MaterialIcons"));
        assert!(!roboto_ids.is_empty());
        assert!(!icon_ids.is_empty());

        // Fallback slots [0], [1], and [2] must all resolve to "Roboto" when Bold/Mono are not registered separately.
        assert_eq!(ctx.fonts[0].family, "Roboto");
        assert_eq!(ctx.fonts[1].family, "Roboto");
        assert_eq!(ctx.fonts[2].family, "Roboto");
        assert_eq!(ctx.fonts[3].family, "MaterialIcons");

        // Regular, Bold, Monospace, and MaterialIcon shaping must all emit non-empty glyph quads.
        let reg_frame = ctx.shape_text("Regular", TextStyle::default(), Point::ZERO);
        let bold_frame = ctx.shape_text(
            "Bold",
            TextStyle {
                font: BuiltinFont::RobotoBold,
                ..TextStyle::default()
            },
            Point::ZERO,
        );
        let mono_frame = ctx.shape_text(
            "Mono",
            TextStyle {
                font: BuiltinFont::RobotoMonoBold,
                ..TextStyle::default()
            },
            Point::ZERO,
        );
        let icon_frame = ctx.shape_icon(MaterialIcon::Bolt, 24.0, Color::WHITE, Point::ZERO);

        assert_eq!(reg_frame.quads.len(), 7);
        assert_eq!(bold_frame.quads.len(), 4);
        assert_eq!(mono_frame.quads.len(), 4);
        assert_eq!(icon_frame.quads.len(), 1);
    }
}
