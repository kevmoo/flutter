//! C-ABI types, opaque handle structs, and enum constants matching upstream `impeller.h`.

use std::ffi::{c_char, c_void};

use impeller_canvas::{ClipOperation, DrawStyle};
use impeller_geometry::{BlendMode, Color, ISize, Matrix, Point, Rect, Size, TileMode};
use impeller_path::{Cap, FillType, Join, RoundingRadii};
use impeller_typographer::{TextAlign, TextDirection};

/// `IMPELLER_MAKE_VERSION(1, 1, 4, 0)` = `(1 << 29) | (1 << 22) | (4 << 12) | 0`.
pub const IMPELLER_VERSION: u32 = (1 << 29) | (1 << 22) | (4 << 12);

macro_rules! define_handle {
    ($struct_name:ident, $ptr_name:ident) => {
        #[repr(C)]
        pub struct $struct_name {
            _private: [u8; 0],
        }
        pub type $ptr_name = *mut $struct_name;
    };
}

define_handle!(ImpellerContext_, ImpellerContext);
define_handle!(ImpellerDisplayList_, ImpellerDisplayList);
define_handle!(ImpellerDisplayListBuilder_, ImpellerDisplayListBuilder);
define_handle!(ImpellerPaint_, ImpellerPaint);
define_handle!(ImpellerColorFilter_, ImpellerColorFilter);
define_handle!(ImpellerColorSource_, ImpellerColorSource);
define_handle!(ImpellerImageFilter_, ImpellerImageFilter);
define_handle!(ImpellerMaskFilter_, ImpellerMaskFilter);
define_handle!(ImpellerTypographyContext_, ImpellerTypographyContext);
define_handle!(ImpellerParagraph_, ImpellerParagraph);
define_handle!(ImpellerParagraphBuilder_, ImpellerParagraphBuilder);
define_handle!(ImpellerParagraphStyle_, ImpellerParagraphStyle);
define_handle!(ImpellerLineMetrics_, ImpellerLineMetrics);
define_handle!(ImpellerGlyphInfo_, ImpellerGlyphInfo);
define_handle!(ImpellerPath_, ImpellerPath);
define_handle!(ImpellerPathBuilder_, ImpellerPathBuilder);
define_handle!(ImpellerSurface_, ImpellerSurface);
define_handle!(ImpellerTexture_, ImpellerTexture);
define_handle!(ImpellerVulkanSwapchain_, ImpellerVulkanSwapchain);
define_handle!(ImpellerFragmentProgram_, ImpellerFragmentProgram);

pub type ImpellerCallback = Option<unsafe extern "C" fn(user_data: *mut c_void)>;
pub type ImpellerProcAddressCallback =
    Option<unsafe extern "C" fn(proc_name: *const c_char, user_data: *mut c_void) -> *mut c_void>;
pub type ImpellerVulkanProcAddressCallback = Option<
    unsafe extern "C" fn(
        vulkan_instance: *mut c_void,
        vulkan_proc_name: *const c_char,
        user_data: *mut c_void,
    ) -> *mut c_void,
>;

pub type ImpellerFillType = u32;
pub const K_IMPELLER_FILL_TYPE_NON_ZERO: ImpellerFillType = 0;
pub const K_IMPELLER_FILL_TYPE_ODD: ImpellerFillType = 1;

pub type ImpellerClipOperation = u32;
pub const K_IMPELLER_CLIP_OPERATION_DIFFERENCE: ImpellerClipOperation = 0;
pub const K_IMPELLER_CLIP_OPERATION_INTERSECT: ImpellerClipOperation = 1;

pub type ImpellerBlendMode = u32;
pub const K_IMPELLER_BLEND_MODE_CLEAR: ImpellerBlendMode = 0;
pub const K_IMPELLER_BLEND_MODE_SOURCE: ImpellerBlendMode = 1;
pub const K_IMPELLER_BLEND_MODE_DESTINATION: ImpellerBlendMode = 2;
pub const K_IMPELLER_BLEND_MODE_SOURCE_OVER: ImpellerBlendMode = 3;
pub const K_IMPELLER_BLEND_MODE_DESTINATION_OVER: ImpellerBlendMode = 4;
pub const K_IMPELLER_BLEND_MODE_SOURCE_IN: ImpellerBlendMode = 5;
pub const K_IMPELLER_BLEND_MODE_DESTINATION_IN: ImpellerBlendMode = 6;
pub const K_IMPELLER_BLEND_MODE_SOURCE_OUT: ImpellerBlendMode = 7;
pub const K_IMPELLER_BLEND_MODE_DESTINATION_OUT: ImpellerBlendMode = 8;
pub const K_IMPELLER_BLEND_MODE_SOURCE_A_TOP: ImpellerBlendMode = 9;
pub const K_IMPELLER_BLEND_MODE_DESTINATION_A_TOP: ImpellerBlendMode = 10;
pub const K_IMPELLER_BLEND_MODE_XOR: ImpellerBlendMode = 11;
pub const K_IMPELLER_BLEND_MODE_PLUS: ImpellerBlendMode = 12;
pub const K_IMPELLER_BLEND_MODE_MODULATE: ImpellerBlendMode = 13;
pub const K_IMPELLER_BLEND_MODE_SCREEN: ImpellerBlendMode = 14;
pub const K_IMPELLER_BLEND_MODE_OVERLAY: ImpellerBlendMode = 15;
pub const K_IMPELLER_BLEND_MODE_DARKEN: ImpellerBlendMode = 16;
pub const K_IMPELLER_BLEND_MODE_LIGHTEN: ImpellerBlendMode = 17;
pub const K_IMPELLER_BLEND_MODE_COLOR_DODGE: ImpellerBlendMode = 18;
pub const K_IMPELLER_BLEND_MODE_COLOR_BURN: ImpellerBlendMode = 19;
pub const K_IMPELLER_BLEND_MODE_HARD_LIGHT: ImpellerBlendMode = 20;
pub const K_IMPELLER_BLEND_MODE_SOFT_LIGHT: ImpellerBlendMode = 21;
pub const K_IMPELLER_BLEND_MODE_DIFFERENCE: ImpellerBlendMode = 22;
pub const K_IMPELLER_BLEND_MODE_EXCLUSION: ImpellerBlendMode = 23;
pub const K_IMPELLER_BLEND_MODE_MULTIPLY: ImpellerBlendMode = 24;
pub const K_IMPELLER_BLEND_MODE_HUE: ImpellerBlendMode = 25;
pub const K_IMPELLER_BLEND_MODE_SATURATION: ImpellerBlendMode = 26;
pub const K_IMPELLER_BLEND_MODE_COLOR: ImpellerBlendMode = 27;
pub const K_IMPELLER_BLEND_MODE_LUMINOSITY: ImpellerBlendMode = 28;

pub type ImpellerDrawStyle = u32;
pub const K_IMPELLER_DRAW_STYLE_FILL: ImpellerDrawStyle = 0;
pub const K_IMPELLER_DRAW_STYLE_STROKE: ImpellerDrawStyle = 1;
pub const K_IMPELLER_DRAW_STYLE_STROKE_AND_FILL: ImpellerDrawStyle = 2;

pub type ImpellerStrokeCap = u32;
pub const K_IMPELLER_STROKE_CAP_BUTT: ImpellerStrokeCap = 0;
pub const K_IMPELLER_STROKE_CAP_ROUND: ImpellerStrokeCap = 1;
pub const K_IMPELLER_STROKE_CAP_SQUARE: ImpellerStrokeCap = 2;

pub type ImpellerStrokeJoin = u32;
pub const K_IMPELLER_STROKE_JOIN_MITER: ImpellerStrokeJoin = 0;
pub const K_IMPELLER_STROKE_JOIN_ROUND: ImpellerStrokeJoin = 1;
pub const K_IMPELLER_STROKE_JOIN_BEVEL: ImpellerStrokeJoin = 2;

pub type ImpellerPixelFormat = u32;
pub const K_IMPELLER_PIXEL_FORMAT_RGBA8888: ImpellerPixelFormat = 0;

pub type ImpellerTextureSampling = u32;
pub const K_IMPELLER_TEXTURE_SAMPLING_NEAREST_NEIGHBOR: ImpellerTextureSampling = 0;
pub const K_IMPELLER_TEXTURE_SAMPLING_LINEAR: ImpellerTextureSampling = 1;

pub type ImpellerTileMode = u32;
pub const K_IMPELLER_TILE_MODE_CLAMP: ImpellerTileMode = 0;
pub const K_IMPELLER_TILE_MODE_REPEAT: ImpellerTileMode = 1;
pub const K_IMPELLER_TILE_MODE_MIRROR: ImpellerTileMode = 2;
pub const K_IMPELLER_TILE_MODE_DECAL: ImpellerTileMode = 3;

pub type ImpellerBlurStyle = u32;
pub const K_IMPELLER_BLUR_STYLE_NORMAL: ImpellerBlurStyle = 0;
pub const K_IMPELLER_BLUR_STYLE_SOLID: ImpellerBlurStyle = 1;
pub const K_IMPELLER_BLUR_STYLE_OUTER: ImpellerBlurStyle = 2;
pub const K_IMPELLER_BLUR_STYLE_INNER: ImpellerBlurStyle = 3;

pub type ImpellerColorSpace = u32;
pub const K_IMPELLER_COLOR_SPACE_SRGB: ImpellerColorSpace = 0;
pub const K_IMPELLER_COLOR_SPACE_EXTENDED_SRGB: ImpellerColorSpace = 1;
pub const K_IMPELLER_COLOR_SPACE_DISPLAY_P3: ImpellerColorSpace = 2;

pub type ImpellerFontWeight = u32;
pub const K_IMPELLER_FONT_WEIGHT_100: ImpellerFontWeight = 0;
pub const K_IMPELLER_FONT_WEIGHT_200: ImpellerFontWeight = 1;
pub const K_IMPELLER_FONT_WEIGHT_300: ImpellerFontWeight = 2;
pub const K_IMPELLER_FONT_WEIGHT_400: ImpellerFontWeight = 3;
pub const K_IMPELLER_FONT_WEIGHT_500: ImpellerFontWeight = 4;
pub const K_IMPELLER_FONT_WEIGHT_600: ImpellerFontWeight = 5;
pub const K_IMPELLER_FONT_WEIGHT_700: ImpellerFontWeight = 6;
pub const K_IMPELLER_FONT_WEIGHT_800: ImpellerFontWeight = 7;
pub const K_IMPELLER_FONT_WEIGHT_900: ImpellerFontWeight = 8;

pub type ImpellerFontStyle = u32;
pub const K_IMPELLER_FONT_STYLE_NORMAL: ImpellerFontStyle = 0;
pub const K_IMPELLER_FONT_STYLE_ITALIC: ImpellerFontStyle = 1;

pub type ImpellerTextAlignment = u32;
pub const K_IMPELLER_TEXT_ALIGNMENT_LEFT: ImpellerTextAlignment = 0;
pub const K_IMPELLER_TEXT_ALIGNMENT_RIGHT: ImpellerTextAlignment = 1;
pub const K_IMPELLER_TEXT_ALIGNMENT_CENTER: ImpellerTextAlignment = 2;
pub const K_IMPELLER_TEXT_ALIGNMENT_JUSTIFY: ImpellerTextAlignment = 3;
pub const K_IMPELLER_TEXT_ALIGNMENT_START: ImpellerTextAlignment = 4;
pub const K_IMPELLER_TEXT_ALIGNMENT_END: ImpellerTextAlignment = 5;

pub type ImpellerTextDirection = u32;
pub const K_IMPELLER_TEXT_DIRECTION_RTL: ImpellerTextDirection = 0;
pub const K_IMPELLER_TEXT_DIRECTION_LTR: ImpellerTextDirection = 1;

pub type ImpellerTextDecorationType = i32;
pub const K_IMPELLER_TEXT_DECORATION_TYPE_NONE: ImpellerTextDecorationType = 0;
pub const K_IMPELLER_TEXT_DECORATION_TYPE_UNDERLINE: ImpellerTextDecorationType = 1 << 0;
pub const K_IMPELLER_TEXT_DECORATION_TYPE_OVERLINE: ImpellerTextDecorationType = 1 << 1;
pub const K_IMPELLER_TEXT_DECORATION_TYPE_LINE_THROUGH: ImpellerTextDecorationType = 1 << 2;

pub type ImpellerTextDecorationStyle = u32;
pub const K_IMPELLER_TEXT_DECORATION_STYLE_SOLID: ImpellerTextDecorationStyle = 0;
pub const K_IMPELLER_TEXT_DECORATION_STYLE_DOUBLE: ImpellerTextDecorationStyle = 1;
pub const K_IMPELLER_TEXT_DECORATION_STYLE_DOTTED: ImpellerTextDecorationStyle = 2;
pub const K_IMPELLER_TEXT_DECORATION_STYLE_DASHED: ImpellerTextDecorationStyle = 3;
pub const K_IMPELLER_TEXT_DECORATION_STYLE_WAVY: ImpellerTextDecorationStyle = 4;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ImpellerRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ImpellerPoint {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ImpellerSize {
    pub width: f32,
    pub height: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ImpellerISize {
    pub width: i64,
    pub height: i64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ImpellerRange {
    pub start: u64,
    pub end: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ImpellerMatrix {
    pub m: [f32; 16],
}

impl Default for ImpellerMatrix {
    fn default() -> Self {
        Self {
            m: Matrix::IDENTITY.m,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ImpellerColorMatrix {
    pub m: [f32; 20],
}

impl Default for ImpellerColorMatrix {
    fn default() -> Self {
        Self { m: [0.0; 20] }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ImpellerRoundingRadii {
    pub top_left: ImpellerSize,
    pub bottom_left: ImpellerSize,
    pub top_right: ImpellerSize,
    pub bottom_right: ImpellerSize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ImpellerColor {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub alpha: f32,
    pub color_space: ImpellerColorSpace,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ImpellerTextureDescriptor {
    pub pixel_format: ImpellerPixelFormat,
    pub size: ImpellerISize,
    pub mip_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ImpellerMapping {
    pub data: *const u8,
    pub length: u64,
    pub on_release: ImpellerCallback,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ImpellerContextVulkanSettings {
    pub user_data: *mut c_void,
    pub proc_address_callback: ImpellerVulkanProcAddressCallback,
    pub enable_validation: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ImpellerContextVulkanInfo {
    pub vk_instance: *mut c_void,
    pub vk_physical_device: *mut c_void,
    pub vk_logical_device: *mut c_void,
    pub graphics_queue_family_index: u32,
    pub graphics_queue_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ImpellerTextDecoration {
    pub types: i32,
    pub color: ImpellerColor,
    pub style: ImpellerTextDecorationStyle,
    pub thickness_multiplier: f32,
}

impl From<ImpellerPoint> for Point {
    fn from(p: ImpellerPoint) -> Self {
        Self::new(p.x, p.y)
    }
}

impl From<Point> for ImpellerPoint {
    fn from(p: Point) -> Self {
        Self { x: p.x, y: p.y }
    }
}

impl From<ImpellerSize> for Size {
    fn from(s: ImpellerSize) -> Self {
        Self::new(s.width, s.height)
    }
}

impl From<Size> for ImpellerSize {
    fn from(s: Size) -> Self {
        Self {
            width: s.width,
            height: s.height,
        }
    }
}

impl From<ImpellerISize> for ISize {
    fn from(s: ImpellerISize) -> Self {
        Self::new(s.width, s.height)
    }
}

impl From<ImpellerRect> for Rect {
    fn from(r: ImpellerRect) -> Self {
        Self::from_xywh(r.x, r.y, r.width, r.height)
    }
}

impl From<Rect> for ImpellerRect {
    fn from(r: Rect) -> Self {
        Self {
            x: r.origin.x,
            y: r.origin.y,
            width: r.size.width,
            height: r.size.height,
        }
    }
}

impl From<ImpellerMatrix> for Matrix {
    fn from(m: ImpellerMatrix) -> Self {
        Self { m: m.m }
    }
}

impl From<Matrix> for ImpellerMatrix {
    fn from(m: Matrix) -> Self {
        Self { m: m.m }
    }
}

impl From<ImpellerColor> for Color {
    fn from(c: ImpellerColor) -> Self {
        Self::new(c.red, c.green, c.blue, c.alpha)
    }
}

impl From<ImpellerRoundingRadii> for RoundingRadii {
    fn from(r: ImpellerRoundingRadii) -> Self {
        Self {
            top_left: r.top_left.into(),
            bottom_left: r.bottom_left.into(),
            top_right: r.top_right.into(),
            bottom_right: r.bottom_right.into(),
        }
    }
}

pub const fn to_fill_type(fill: ImpellerFillType) -> FillType {
    match fill {
        K_IMPELLER_FILL_TYPE_ODD => FillType::Odd,
        _ => FillType::NonZero,
    }
}

pub const fn to_clip_op(op: ImpellerClipOperation) -> ClipOperation {
    match op {
        K_IMPELLER_CLIP_OPERATION_DIFFERENCE => ClipOperation::Difference,
        _ => ClipOperation::Intersect,
    }
}

pub const fn to_blend_mode(mode: ImpellerBlendMode) -> BlendMode {
    match mode {
        K_IMPELLER_BLEND_MODE_CLEAR => BlendMode::Clear,
        K_IMPELLER_BLEND_MODE_SOURCE => BlendMode::Src,
        K_IMPELLER_BLEND_MODE_DESTINATION => BlendMode::Dst,
        K_IMPELLER_BLEND_MODE_SOURCE_OVER => BlendMode::SrcOver,
        K_IMPELLER_BLEND_MODE_DESTINATION_OVER => BlendMode::DstOver,
        K_IMPELLER_BLEND_MODE_SOURCE_IN => BlendMode::SrcIn,
        K_IMPELLER_BLEND_MODE_DESTINATION_IN => BlendMode::DstIn,
        K_IMPELLER_BLEND_MODE_SOURCE_OUT => BlendMode::SrcOut,
        K_IMPELLER_BLEND_MODE_DESTINATION_OUT => BlendMode::DstOut,
        K_IMPELLER_BLEND_MODE_SOURCE_A_TOP => BlendMode::SrcATop,
        K_IMPELLER_BLEND_MODE_DESTINATION_A_TOP => BlendMode::DstATop,
        K_IMPELLER_BLEND_MODE_XOR => BlendMode::Xor,
        K_IMPELLER_BLEND_MODE_PLUS => BlendMode::Plus,
        K_IMPELLER_BLEND_MODE_MODULATE => BlendMode::Modulate,
        K_IMPELLER_BLEND_MODE_SCREEN => BlendMode::Screen,
        K_IMPELLER_BLEND_MODE_OVERLAY => BlendMode::Overlay,
        K_IMPELLER_BLEND_MODE_DARKEN => BlendMode::Darken,
        K_IMPELLER_BLEND_MODE_LIGHTEN => BlendMode::Lighten,
        K_IMPELLER_BLEND_MODE_COLOR_DODGE => BlendMode::ColorDodge,
        K_IMPELLER_BLEND_MODE_COLOR_BURN => BlendMode::ColorBurn,
        K_IMPELLER_BLEND_MODE_HARD_LIGHT => BlendMode::HardLight,
        K_IMPELLER_BLEND_MODE_SOFT_LIGHT => BlendMode::SoftLight,
        K_IMPELLER_BLEND_MODE_DIFFERENCE => BlendMode::Difference,
        K_IMPELLER_BLEND_MODE_EXCLUSION => BlendMode::Exclusion,
        K_IMPELLER_BLEND_MODE_MULTIPLY => BlendMode::Multiply,
        K_IMPELLER_BLEND_MODE_HUE => BlendMode::Hue,
        K_IMPELLER_BLEND_MODE_SATURATION => BlendMode::Saturation,
        K_IMPELLER_BLEND_MODE_COLOR => BlendMode::Color,
        K_IMPELLER_BLEND_MODE_LUMINOSITY => BlendMode::Luminosity,
        _ => BlendMode::SrcOver,
    }
}

pub const fn to_draw_style(style: ImpellerDrawStyle) -> (DrawStyle, bool) {
    match style {
        K_IMPELLER_DRAW_STYLE_STROKE => (DrawStyle::Stroke, false),
        K_IMPELLER_DRAW_STYLE_STROKE_AND_FILL => (DrawStyle::Fill, true),
        _ => (DrawStyle::Fill, false),
    }
}

pub const fn to_stroke_cap(cap: ImpellerStrokeCap) -> Cap {
    match cap {
        K_IMPELLER_STROKE_CAP_ROUND => Cap::Round,
        K_IMPELLER_STROKE_CAP_SQUARE => Cap::Square,
        _ => Cap::Butt,
    }
}

pub const fn to_stroke_join(join: ImpellerStrokeJoin) -> Join {
    match join {
        K_IMPELLER_STROKE_JOIN_ROUND => Join::Round,
        K_IMPELLER_STROKE_JOIN_BEVEL => Join::Bevel,
        _ => Join::Miter,
    }
}

pub const fn to_tile_mode(mode: ImpellerTileMode) -> TileMode {
    match mode {
        K_IMPELLER_TILE_MODE_REPEAT => TileMode::Repeat,
        K_IMPELLER_TILE_MODE_MIRROR => TileMode::Mirror,
        K_IMPELLER_TILE_MODE_DECAL => TileMode::Decal,
        _ => TileMode::Clamp,
    }
}

pub const fn to_text_align(align: ImpellerTextAlignment) -> TextAlign {
    match align {
        K_IMPELLER_TEXT_ALIGNMENT_RIGHT => TextAlign::Right,
        K_IMPELLER_TEXT_ALIGNMENT_CENTER => TextAlign::Center,
        K_IMPELLER_TEXT_ALIGNMENT_JUSTIFY => TextAlign::Justify,
        K_IMPELLER_TEXT_ALIGNMENT_START => TextAlign::Start,
        K_IMPELLER_TEXT_ALIGNMENT_END => TextAlign::End,
        _ => TextAlign::Left,
    }
}

pub const fn to_text_direction(dir: ImpellerTextDirection) -> TextDirection {
    match dir {
        K_IMPELLER_TEXT_DIRECTION_RTL => TextDirection::Rtl,
        _ => TextDirection::Ltr,
    }
}

pub const fn from_text_direction(dir: TextDirection) -> ImpellerTextDirection {
    match dir {
        TextDirection::Rtl => K_IMPELLER_TEXT_DIRECTION_RTL,
        TextDirection::Ltr => K_IMPELLER_TEXT_DIRECTION_LTR,
    }
}

pub const fn to_css_font_weight(weight: ImpellerFontWeight) -> u16 {
    match weight {
        K_IMPELLER_FONT_WEIGHT_100 => 100,
        K_IMPELLER_FONT_WEIGHT_200 => 200,
        K_IMPELLER_FONT_WEIGHT_300 => 300,
        K_IMPELLER_FONT_WEIGHT_400 => 400,
        K_IMPELLER_FONT_WEIGHT_500 => 500,
        K_IMPELLER_FONT_WEIGHT_600 => 600,
        K_IMPELLER_FONT_WEIGHT_700 => 700,
        K_IMPELLER_FONT_WEIGHT_800 => 800,
        K_IMPELLER_FONT_WEIGHT_900 => 900,
        _ => 400,
    }
}
