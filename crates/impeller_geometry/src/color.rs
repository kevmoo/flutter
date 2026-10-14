use crate::scalar::{scalar_nearly_equal, Scalar, K_EH_CLOSE_ENOUGH};
use crate::vector::{Vector3, Vector4};
use bytemuck::{Pod, Zeroable};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum BlendMode {
    Clear = 0,
    Src,
    Dst,
    #[default]
    SrcOver,
    DstOver,
    SrcIn,
    DstIn,
    SrcOut,
    DstOut,
    SrcATop,
    DstATop,
    Xor,
    Plus,
    Modulate,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Multiply,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl BlendMode {
    pub const ALL: [Self; 29] = [
        Self::Clear,
        Self::Src,
        Self::Dst,
        Self::SrcOver,
        Self::DstOver,
        Self::SrcIn,
        Self::DstIn,
        Self::SrcOut,
        Self::DstOut,
        Self::SrcATop,
        Self::DstATop,
        Self::Xor,
        Self::Plus,
        Self::Modulate,
        Self::Screen,
        Self::Overlay,
        Self::Darken,
        Self::Lighten,
        Self::ColorDodge,
        Self::ColorBurn,
        Self::HardLight,
        Self::SoftLight,
        Self::Difference,
        Self::Exclusion,
        Self::Multiply,
        Self::Hue,
        Self::Saturation,
        Self::Color,
        Self::Luminosity,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Clear => "Clear",
            Self::Src => "Src",
            Self::Dst => "Dst",
            Self::SrcOver => "SrcOver",
            Self::DstOver => "DstOver",
            Self::SrcIn => "SrcIn",
            Self::DstIn => "DstIn",
            Self::SrcOut => "SrcOut",
            Self::DstOut => "DstOut",
            Self::SrcATop => "SrcATop",
            Self::DstATop => "DstATop",
            Self::Xor => "Xor",
            Self::Plus => "Plus",
            Self::Modulate => "Modulate",
            Self::Screen => "Screen",
            Self::Overlay => "Overlay",
            Self::Darken => "Darken",
            Self::Lighten => "Lighten",
            Self::ColorDodge => "ColorDodge",
            Self::ColorBurn => "ColorBurn",
            Self::HardLight => "HardLight",
            Self::SoftLight => "SoftLight",
            Self::Difference => "Difference",
            Self::Exclusion => "Exclusion",
            Self::Multiply => "Multiply",
            Self::Hue => "Hue",
            Self::Saturation => "Saturation",
            Self::Color => "Color",
            Self::Luminosity => "Luminosity",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum TileMode {
    #[default]
    Clamp = 0,
    Repeat = 1,
    Mirror = 2,
    Decal = 3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct ColorMatrix {
    pub array: [Scalar; 20],
}

impl Default for ColorMatrix {
    #[inline]
    fn default() -> Self {
        Self { array: [0.0; 20] }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Color {
    pub red: Scalar,
    pub green: Scalar,
    pub blue: Scalar,
    pub alpha: Scalar,
}

impl Color {
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    pub const BLACK_TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    pub const WHITE_TRANSPARENT: Self = Self::new(1.0, 1.0, 1.0, 0.0);
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);
    pub const RED: Self = Self::new(1.0, 0.0, 0.0, 1.0);
    pub const GREEN: Self = Self::new(0.0, 1.0, 0.0, 1.0);
    pub const BLUE: Self = Self::new(0.0, 0.0, 1.0, 1.0);
    pub const CORNFLOWER_BLUE: Self =
        Self::new(100.0 / 255.0, 149.0 / 255.0, 237.0 / 255.0, 1.0);
    pub const LIME_GREEN: Self = Self::new(50.0 / 255.0, 205.0 / 255.0, 50.0 / 255.0, 1.0);
    pub const YELLOW: Self = Self::new(1.0, 1.0, 0.0, 1.0);

    #[inline]
    pub const fn new(red: Scalar, green: Scalar, blue: Scalar, alpha: Scalar) -> Self {
        Self {
            red,
            green,
            blue,
            alpha,
        }
    }

    #[inline]
    pub fn make_rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::new(
            (r as Scalar) / 255.0,
            (g as Scalar) / 255.0,
            (b as Scalar) / 255.0,
            (a as Scalar) / 255.0,
        )
    }

    #[inline]
    pub const fn with_alpha(self, new_alpha: Scalar) -> Self {
        Self::new(self.red, self.green, self.blue, new_alpha)
    }

    #[inline]
    pub fn premultiply(&self) -> Self {
        Self::new(
            self.red * self.alpha,
            self.green * self.alpha,
            self.blue * self.alpha,
            self.alpha,
        )
    }

    #[inline]
    pub fn unpremultiply(&self) -> Self {
        if scalar_nearly_equal(self.alpha, 0.0, K_EH_CLOSE_ENOUGH) {
            return Self::BLACK_TRANSPARENT;
        }
        Self::new(
            self.red / self.alpha,
            self.green / self.alpha,
            self.blue / self.alpha,
            self.alpha,
        )
    }

    #[inline]
    pub fn to_array_premultiplied(&self) -> [Scalar; 4] {
        let p = self.premultiply();
        [p.red, p.green, p.blue, p.alpha]
    }

    #[inline]
    pub fn to_r8g8b8a8(&self) -> [u8; 4] {
        [
            (self.red * 255.0).round() as u8,
            (self.green * 255.0).round() as u8,
            (self.blue * 255.0).round() as u8,
            (self.alpha * 255.0).round() as u8,
        ]
    }

    #[inline]
    pub fn to_icolor(color: Self) -> u32 {
        let a = ((color.alpha * 255.0).round() as i64 & 0xff) as u32;
        let r = ((color.red * 255.0).round() as i64 & 0xff) as u32;
        let g = ((color.green * 255.0).round() as i64 & 0xff) as u32;
        let b = ((color.blue * 255.0).round() as i64 & 0xff) as u32;
        (a << 24) | (r << 16) | (g << 8) | b
    }

    #[inline]
    pub fn to_argb(&self) -> u32 {
        let rgba = self.to_r8g8b8a8();
        ((rgba[3] as u32) << 24)
            | ((rgba[0] as u32) << 16)
            | ((rgba[1] as u32) << 8)
            | (rgba[2] as u32)
    }

    #[inline]
    pub fn lerp(a: Self, b: Self, t: Scalar) -> Self {
        a + (b - a) * t
    }

    #[inline]
    pub fn clamp01(&self) -> Self {
        Self::new(
            self.red.clamp(0.0, 1.0),
            self.green.clamp(0.0, 1.0),
            self.blue.clamp(0.0, 1.0),
            self.alpha.clamp(0.0, 1.0),
        )
    }

    #[inline]
    pub fn is_wide_gamut(&self) -> bool {
        self.red < -K_EH_CLOSE_ENOUGH
            || self.red > 1.0 + K_EH_CLOSE_ENOUGH
            || self.green < -K_EH_CLOSE_ENOUGH
            || self.green > 1.0 + K_EH_CLOSE_ENOUGH
            || self.blue < -K_EH_CLOSE_ENOUGH
            || self.blue > 1.0 + K_EH_CLOSE_ENOUGH
    }

    #[inline]
    pub fn is_transparent(&self) -> bool {
        self.alpha == 0.0
    }

    #[inline]
    pub fn is_opaque(&self) -> bool {
        self.alpha == 1.0
    }

    #[inline]
    pub fn nearly_equal(&self, other: Self, tolerance: Scalar) -> bool {
        scalar_nearly_equal(self.red, other.red, tolerance)
            && scalar_nearly_equal(self.green, other.green, tolerance)
            && scalar_nearly_equal(self.blue, other.blue, tolerance)
            && scalar_nearly_equal(self.alpha, other.alpha, tolerance)
    }

    pub fn apply_color_matrix(&self, color_matrix: &ColorMatrix) -> Self {
        let c = &color_matrix.array;
        Self::new(
            c[0] * self.red + c[1] * self.green + c[2] * self.blue + c[3] * self.alpha + c[4],
            c[5] * self.red + c[6] * self.green + c[7] * self.blue + c[8] * self.alpha + c[9],
            c[10] * self.red + c[11] * self.green + c[12] * self.blue + c[13] * self.alpha + c[14],
            c[15] * self.red + c[16] * self.green + c[17] * self.blue + c[18] * self.alpha + c[19],
        )
        .clamp01()
    }

    pub fn linear_to_srgb(&self) -> Self {
        let conv = |c: Scalar| -> Scalar {
            if c <= 0.0031308 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            }
        };
        Self::new(
            conv(self.red),
            conv(self.green),
            conv(self.blue),
            self.alpha,
        )
    }

    pub fn srgb_to_linear(&self) -> Self {
        let conv = |c: Scalar| -> Scalar {
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        Self::new(
            conv(self.red),
            conv(self.green),
            conv(self.blue),
            self.alpha,
        )
    }

    /// Blends an unpremultiplied destination color (`self`) with an unpremultiplied
    /// `src` color using `blend_mode`, returning an unpremultiplied result.
    /// Can be called as `dst.blend(src, mode)` or `Color::blend(dst, src, mode)`.
    pub fn blend(self, src: Self, blend_mode: BlendMode) -> Self {
        let dst = self;
        match blend_mode {
            BlendMode::Clear => Self::BLACK_TRANSPARENT,
            BlendMode::Src => src,
            BlendMode::Dst => dst,
            BlendMode::SrcOver => {
                (src.premultiply() + dst.premultiply() * (1.0 - src.alpha)).unpremultiply()
            }
            BlendMode::DstOver => {
                (dst.premultiply() + src.premultiply() * (1.0 - dst.alpha)).unpremultiply()
            }
            BlendMode::SrcIn => (src.premultiply() * dst.alpha).unpremultiply(),
            BlendMode::DstIn => (dst.premultiply() * src.alpha).unpremultiply(),
            BlendMode::SrcOut => (src.premultiply() * (1.0 - dst.alpha)).unpremultiply(),
            BlendMode::DstOut => (dst.premultiply() * (1.0 - src.alpha)).unpremultiply(),
            BlendMode::SrcATop => {
                (src.premultiply() * dst.alpha + dst.premultiply() * (1.0 - src.alpha))
                    .unpremultiply()
            }
            BlendMode::DstATop => {
                (dst.premultiply() * src.alpha + src.premultiply() * (1.0 - dst.alpha))
                    .unpremultiply()
            }
            BlendMode::Xor => {
                (src.premultiply() * (1.0 - dst.alpha) + dst.premultiply() * (1.0 - src.alpha))
                    .unpremultiply()
            }
            BlendMode::Plus => {
                min_color(src.premultiply() + dst.premultiply(), 1.0).unpremultiply()
            }
            BlendMode::Modulate => (src.premultiply() * dst.premultiply()).unpremultiply(),
            BlendMode::Screen => do_color_blend(dst, src, |d, s| s + d - s * d),
            BlendMode::Overlay => do_color_blend(dst, src, |d, s| {
                let screen_src = d * 2.0 - 1.0;
                let screen = screen_src + s - screen_src * s;
                component_choose(s * (d * 2.0), screen, d, 0.5)
            }),
            BlendMode::Darken => do_color_blend(dst, src, |d, s| d.min(s)),
            BlendMode::Lighten => do_color_blend(dst, src, |d, s| d.max(s)),
            BlendMode::ColorDodge => do_color_blend_components(dst, src, |d, s| {
                if d < K_EH_CLOSE_ENOUGH {
                    return 0.0;
                }
                if 1.0 - s < K_EH_CLOSE_ENOUGH {
                    return 1.0;
                }
                1.0_f32.min(d / (1.0 - s))
            }),
            BlendMode::ColorBurn => do_color_blend_components(dst, src, |d, s| {
                if 1.0 - d < K_EH_CLOSE_ENOUGH {
                    return 1.0;
                }
                if s < K_EH_CLOSE_ENOUGH {
                    return 0.0;
                }
                1.0 - 1.0_f32.min((1.0 - d) / s)
            }),
            BlendMode::HardLight => do_color_blend(dst, src, |d, s| {
                let screen_src = s * 2.0 - 1.0;
                let screen = screen_src + d - screen_src * d;
                component_choose(d * (s * 2.0), screen, s, 0.5)
            }),
            BlendMode::SoftLight => do_color_blend(dst, src, |d, s| {
                let big_d = component_choose(
                    ((d * 16.0 - 12.0) * d + 4.0) * d,
                    Vector3::new(d.x.sqrt(), d.y.sqrt(), d.z.sqrt()),
                    d,
                    0.25,
                );
                component_choose(
                    d - (Vector3::splat(1.0) - s * 2.0) * d * (Vector3::splat(1.0) - d),
                    d + (s * 2.0 - 1.0) * (big_d - d),
                    s,
                    0.5,
                )
            }),
            BlendMode::Difference => do_color_blend(dst, src, |d, s| (d - s).abs()),
            BlendMode::Exclusion => do_color_blend(dst, src, |d, s| d + s - d * s * 2.0),
            BlendMode::Multiply => do_color_blend(dst, src, |d, s| d * s),
            BlendMode::Hue => do_color_blend(dst, src, |d, s| {
                set_luminosity(set_saturation(s, saturation(d)), luminosity(d))
            }),
            BlendMode::Saturation => do_color_blend(dst, src, |d, s| {
                set_luminosity(set_saturation(d, saturation(s)), luminosity(d))
            }),
            BlendMode::Color => do_color_blend(dst, src, |d, s| set_luminosity(s, luminosity(d))),
            BlendMode::Luminosity => {
                do_color_blend(dst, src, |d, s| set_luminosity(d, luminosity(s)))
            }
        }
    }
}

#[inline]
fn min_color(c: Color, threshold: Scalar) -> Color {
    Color::new(
        c.red.min(threshold),
        c.green.min(threshold),
        c.blue.min(threshold),
        c.alpha.min(threshold),
    )
}

#[inline]
fn luminosity(color: Vector3) -> Scalar {
    color.x * 0.3 + color.y * 0.59 + color.z * 0.11
}

#[inline]
fn clip_color(mut color: Vector3) -> Vector3 {
    let lum = luminosity(color);
    let mn = color.x.min(color.y).min(color.z);
    let mx = color.x.max(color.y).max(color.z);
    if mn < 0.0 {
        color = Vector3::splat(lum) + (((color - lum) * lum) / (lum - mn + K_EH_CLOSE_ENOUGH));
    }
    if mx > 1.0 {
        color = Vector3::splat(lum)
            + (((color - lum) * (1.0 - lum)) / (mx - lum + K_EH_CLOSE_ENOUGH));
    }
    color
}

#[inline]
fn set_luminosity(color: Vector3, lum: Scalar) -> Vector3 {
    let relative_lum = lum - luminosity(color);
    clip_color(color + relative_lum)
}

#[inline]
fn saturation(color: Vector3) -> Scalar {
    color.x.max(color.y).max(color.z) - color.x.min(color.y).min(color.z)
}

#[inline]
fn set_saturation(color: Vector3, sat: Scalar) -> Vector3 {
    let mn = color.x.min(color.y).min(color.z);
    let mx = color.x.max(color.y).max(color.z);
    if mn < mx {
        ((color - mn) * sat) / (mx - mn)
    } else {
        Vector3::ZERO
    }
}

#[inline]
fn component_choose(a: Vector3, b: Vector3, value: Vector3, cutoff: Scalar) -> Vector3 {
    Vector3::new(
        if value.x > cutoff { b.x } else { a.x },
        if value.y > cutoff { b.y } else { a.y },
        if value.z > cutoff { b.z } else { a.z },
    )
}

#[inline]
const fn to_rgb(color: Color) -> Vector3 {
    Vector3::new(color.red, color.green, color.blue)
}

#[inline]
const fn from_rgb(color: Vector3, alpha: Scalar) -> Color {
    Color::new(color.x, color.y, color.z, alpha)
}

#[inline]
fn apply_blended_color(dst: Color, src: Color, blend_result: Vector3) -> Color {
    let dst_pm = dst.premultiply();
    let src_pm = from_rgb(blend_result, src.alpha * dst.alpha).premultiply()
        + src.premultiply() * (1.0 - dst.alpha);
    src_pm + dst_pm * (1.0 - src_pm.alpha)
}

#[inline]
fn do_color_blend<F>(dst: Color, src: Color, blend_rgb_func: F) -> Color
where
    F: FnOnce(Vector3, Vector3) -> Vector3,
{
    let blend_result = blend_rgb_func(to_rgb(dst), to_rgb(src));
    apply_blended_color(dst, src, blend_result).unpremultiply()
}

#[inline]
fn do_color_blend_components<F>(dst: Color, src: Color, mut blend_func: F) -> Color
where
    F: FnMut(Scalar, Scalar) -> Scalar,
{
    let blend_result = Vector3::new(
        blend_func(dst.red, src.red),
        blend_func(dst.green, src.green),
        blend_func(dst.blue, src.blue),
    );
    apply_blended_color(dst, src, blend_result).unpremultiply()
}

impl From<Vector4> for Color {
    #[inline]
    fn from(v: Vector4) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }
}

impl Add for Color {
    type Output = Self;

    #[inline]
    fn add(self, c: Self) -> Self {
        Self::new(
            self.red + c.red,
            self.green + c.green,
            self.blue + c.blue,
            self.alpha + c.alpha,
        )
    }
}

impl Add<Scalar> for Color {
    type Output = Self;

    #[inline]
    fn add(self, v: Scalar) -> Self {
        Self::new(
            self.red + v,
            self.green + v,
            self.blue + v,
            self.alpha + v,
        )
    }
}

impl Sub for Color {
    type Output = Self;

    #[inline]
    fn sub(self, c: Self) -> Self {
        Self::new(
            self.red - c.red,
            self.green - c.green,
            self.blue - c.blue,
            self.alpha - c.alpha,
        )
    }
}

impl Sub<Scalar> for Color {
    type Output = Self;

    #[inline]
    fn sub(self, v: Scalar) -> Self {
        Self::new(
            self.red - v,
            self.green - v,
            self.blue - v,
            self.alpha - v,
        )
    }
}

impl Mul for Color {
    type Output = Self;

    #[inline]
    fn mul(self, c: Self) -> Self {
        Self::new(
            self.red * c.red,
            self.green * c.green,
            self.blue * c.blue,
            self.alpha * c.alpha,
        )
    }
}

impl Mul<Scalar> for Color {
    type Output = Self;

    #[inline]
    fn mul(self, v: Scalar) -> Self {
        Self::new(
            self.red * v,
            self.green * v,
            self.blue * v,
            self.alpha * v,
        )
    }
}

impl Mul<Color> for Scalar {
    type Output = Color;

    #[inline]
    fn mul(self, c: Color) -> Color {
        c * self
    }
}

impl Div<Scalar> for Color {
    type Output = Self;

    #[inline]
    fn div(self, v: Scalar) -> Self {
        Self::new(
            self.red / v,
            self.green / v,
            self.blue / v,
            self.alpha / v,
        )
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ColorHSLA {
    pub hue: Scalar,
    pub saturation: Scalar,
    pub lightness: Scalar,
    pub alpha: Scalar,
}

impl ColorHSLA {
    #[inline]
    pub const fn new(hue: Scalar, saturation: Scalar, lightness: Scalar, alpha: Scalar) -> Self {
        Self {
            hue,
            saturation,
            lightness,
            alpha,
        }
    }

    pub fn from_color(color: Color) -> Self {
        let r = color.red;
        let g = color.green;
        let b = color.blue;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let l = (max + min) * 0.5;
        let d = max - min;

        if d == 0.0 {
            return Self::new(0.0, 0.0, l, color.alpha);
        }

        let s = if l > 0.5 {
            d / (2.0 - max - min)
        } else {
            d / (max + min)
        };

        let mut h = if max == r {
            (g - b) / d + if g < b { 6.0 } else { 0.0 }
        } else if max == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
        h /= 6.0;

        Self::new(h, s, l, color.alpha)
    }

    pub fn to_color(&self) -> Color {
        if self.saturation == 0.0 {
            return Color::new(self.lightness, self.lightness, self.lightness, self.alpha);
        }

        let hue_to_rgb = |p: Scalar, q: Scalar, mut t: Scalar| -> Scalar {
            if t < 0.0 {
                t += 1.0;
            }
            if t > 1.0 {
                t -= 1.0;
            }
            if t < 1.0 / 6.0 {
                return p + (q - p) * 6.0 * t;
            }
            if t < 1.0 / 2.0 {
                return q;
            }
            if t < 2.0 / 3.0 {
                return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
            }
            p
        };

        let q = if self.lightness < 0.5 {
            self.lightness * (1.0 + self.saturation)
        } else {
            self.lightness + self.saturation - self.lightness * self.saturation
        };
        let p = 2.0 * self.lightness - q;

        let r = hue_to_rgb(p, q, self.hue + 1.0 / 3.0);
        let g = hue_to_rgb(p, q, self.hue);
        let b = hue_to_rgb(p, q, self.hue - 1.0 / 3.0);

        Color::new(r, g, b, self.alpha)
    }
}

impl From<Color> for ColorHSLA {
    #[inline]
    fn from(c: Color) -> Self {
        Self::from_color(c)
    }
}

impl From<ColorHSLA> for Color {
    #[inline]
    fn from(hsla: ColorHSLA) -> Self {
        hsla.to_color()
    }
}
