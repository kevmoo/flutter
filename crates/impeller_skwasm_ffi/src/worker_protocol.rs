//! OffscreenCanvas Web Worker multi-threading protocol (`st=0`) for `impeller_skwasm_ffi`.
//!
//! When `crossOriginIsolated` is true and single-threaded mode is not forced, the main UI thread
//! keeps `TypographerContext` (`cosmic-text` + `swash`) and `DisplayList` recording/replay, while
//! transferring `OffscreenCanvas` ownership to a dedicated Web Worker (`impeller_worker.js`).
//! Each frame, `WorkerFramePacket` serializes the frame's `DrawCommand` stream alongside dirty
//! `GlyphAtlas` row slices (`dirty_row_range_since`) and newly referenced `ImageTexture` bytes
//! into a compact transferable binary `ArrayBuffer`, eliminating cross-thread mutable state and
//! FreeType/SkMutex data races (`flutter/flutter#193695`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use impeller_geometry::{BlendMode, Color, ISize, Matrix, Point, Rect, TileMode};
use impeller_tessellator::{TessellatedMesh, Vertex2D};
use impeller_typographer::{AtlasGlyphQuad, GlyphAtlas, GlyphAtlasType};
use impeller_wgpu::{
    ConicalGradientParams, DrawCommand, GlyphAtlasBatch, GradientStop, ImageTexture,
    LayerColorFilter, LayerImageFilter, LinearGradientParams, RRectBlurParams,
    RadialGradientParams, SweepGradientParams, TextureFillBatch, UvVertex,
};

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const PACKET_MAGIC: [u8; 4] = *b"IRWF";
const PACKET_VERSION: u32 = 1;

/// Incremental or full-reset row slice delta for a [`GlyphAtlas`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtlasRowDelta {
    pub atlas_id: u64,
    pub atlas_type: GlyphAtlasType,
    pub width: u32,
    pub height: u32,
    pub generation: u64,
    pub resize_epoch: u64,
    pub full_reset: bool,
    pub y_min: u32,
    pub y_max: u32,
    pub row_pixels: Vec<u8>,
}

/// Newly referenced [`ImageTexture`] payload sent to the worker once per `image.id`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageDelta {
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub rgba_premultiplied: Vec<u8>,
}

/// Self-contained binary frame packet transferred from the main thread to `impeller_worker.js`.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerFramePacket {
    pub width: u32,
    pub height: u32,
    pub alpha_used: bool,
    pub color_used: bool,
    pub alpha_delta: Option<AtlasRowDelta>,
    pub color_delta: Option<AtlasRowDelta>,
    pub new_images: Vec<ImageDelta>,
    pub disposed_image_ids: Vec<u64>,
    pub picture_commands: Vec<Vec<DrawCommand>>,
}

/// Scans `commands` recursively to determine whether alpha and/or color glyph atlases are sampled.
pub fn commands_atlas_usage(commands: &[DrawCommand]) -> (bool, bool) {
    let mut uses_alpha = false;
    let mut uses_color = false;
    for cmd in commands {
        match cmd {
            DrawCommand::GlyphAtlas(batch) => {
                for q in &batch.quads {
                    if !q.dest_rect.is_empty() && !q.atlas_rect.is_empty() {
                        if q.is_color {
                            uses_color = true;
                        } else {
                            uses_alpha = true;
                        }
                    }
                }
            }
            DrawCommand::SaveLayer { commands: sub, .. } => {
                let (sa, sc) = commands_atlas_usage(sub);
                uses_alpha |= sa;
                uses_color |= sc;
            }
            _ => {}
        }
    }
    (uses_alpha, uses_color)
}

/// Collects any [`ImageTexture`] referenced by `commands` whose `id` is not yet in `uploaded_ids`.
pub fn collect_missing_images(
    commands: &[DrawCommand],
    uploaded_ids: &mut HashSet<u64>,
    out_deltas: &mut Vec<ImageDelta>,
) {
    for cmd in commands {
        match cmd {
            DrawCommand::TextureFill(batch) => {
                if batch.texture.width > 0
                    && batch.texture.height > 0
                    && !batch.texture.rgba_premultiplied.is_empty()
                    && uploaded_ids.insert(batch.texture.id)
                {
                    out_deltas.push(ImageDelta {
                        id: batch.texture.id,
                        width: batch.texture.width,
                        height: batch.texture.height,
                        rgba_premultiplied: batch.texture.rgba_premultiplied.as_slice().to_vec(),
                    });
                }
            }
            DrawCommand::SaveLayer { commands: sub, .. } => {
                collect_missing_images(sub, uploaded_ids, out_deltas);
            }
            _ => {}
        }
    }
}

/// Builds an incremental (or initial/resized full-reset) [`AtlasRowDelta`] if `atlas` has changed
/// relative to `prev_state` `(atlas_id, generation, resize_epoch)`.
pub fn build_atlas_delta(
    atlas: &GlyphAtlas,
    prev_state: &mut (u64, u64, u64),
) -> Option<AtlasRowDelta> {
    let (prev_id, prev_gen, prev_epoch) = *prev_state;
    let w = atlas.size.width.max(1) as u32;
    let h = atlas.size.height.max(1) as u32;
    let bpp = match atlas.atlas_type {
        GlyphAtlasType::AlphaBitmap => 1usize,
        GlyphAtlasType::ColorBitmap => 4usize,
    };
    let row_bytes = (w as usize) * bpp;

    if prev_id != atlas.atlas_id || prev_epoch != atlas.resize_epoch {
        let y_max = if atlas.resize_epoch == 1 {
            if atlas.generation <= 1 {
                0
            } else {
                atlas
                    .row_generations
                    .iter()
                    .rposition(|&g| g > 1)
                    .map_or(0, |r| ((r + 1) as u32).min(h))
            }
        } else {
            h
        };
        let byte_end = ((y_max as usize) * row_bytes).min(atlas.pixels.len());
        let row_pixels = atlas.pixels[..byte_end].to_vec();
        *prev_state = (atlas.atlas_id, atlas.generation, atlas.resize_epoch);
        return Some(AtlasRowDelta {
            atlas_id: atlas.atlas_id,
            atlas_type: atlas.atlas_type,
            width: w,
            height: h,
            generation: atlas.generation,
            resize_epoch: atlas.resize_epoch,
            full_reset: true,
            y_min: 0,
            y_max,
            row_pixels,
        });
    }

    if atlas.generation > prev_gen {
        let (raw_y_min, raw_y_max) = atlas.dirty_row_range_since(prev_gen).unwrap_or((0, h));
        let y_min = raw_y_min.min(h);
        let y_max = raw_y_max.min(h);
        let byte_start = ((y_min as usize) * row_bytes).min(atlas.pixels.len());
        let byte_end = ((y_max as usize) * row_bytes).min(atlas.pixels.len());
        let row_pixels = if byte_start < byte_end {
            atlas.pixels[byte_start..byte_end].to_vec()
        } else {
            Vec::new()
        };
        *prev_state = (atlas.atlas_id, atlas.generation, atlas.resize_epoch);
        return Some(AtlasRowDelta {
            atlas_id: atlas.atlas_id,
            atlas_type: atlas.atlas_type,
            width: w,
            height: h,
            generation: atlas.generation,
            resize_epoch: atlas.resize_epoch,
            full_reset: false,
            y_min,
            y_max,
            row_pixels,
        });
    }

    None
}

/// Applies an [`AtlasRowDelta`] onto the worker's CPU [`GlyphAtlas`] mirror so subsequent
/// `WgpuRenderer::sync_glyph_atlas` calls upload only the dirty row span.
pub fn apply_atlas_delta(atlas: &mut GlyphAtlas, delta: &AtlasRowDelta) {
    let bpp = match delta.atlas_type {
        GlyphAtlasType::AlphaBitmap => 1usize,
        GlyphAtlasType::ColorBitmap => 4usize,
    };
    let w = delta.width.max(1) as usize;
    let h = delta.height.max(1) as usize;
    let row_bytes = w * bpp;
    let total_bytes = w * h * bpp;

    if delta.full_reset
        || atlas.atlas_id != delta.atlas_id
        || atlas.resize_epoch != delta.resize_epoch
        || atlas.pixels.len() != total_bytes
    {
        atlas.atlas_id = delta.atlas_id;
        atlas.atlas_type = delta.atlas_type;
        atlas.size = ISize::new(w as i64, h as i64);
        atlas.pixels = vec![0u8; total_bytes];
        atlas.row_generations = vec![1u64; h];
    }

    let y0 = (delta.y_min as usize).min(h);
    let y1 = (delta.y_max as usize).min(h);
    if y1 > y0 {
        let dst_start = y0 * row_bytes;
        let max_copy = (y1 - y0) * row_bytes;
        let copy_len = max_copy
            .min(delta.row_pixels.len())
            .min(total_bytes.saturating_sub(dst_start));
        if copy_len > 0 {
            atlas.pixels[dst_start..dst_start + copy_len]
                .copy_from_slice(&delta.row_pixels[..copy_len]);
        }
        atlas.row_generations[y0..y1].fill(delta.generation);
    }
    atlas.generation = delta.generation;
    atlas.resize_epoch = delta.resize_epoch;
}

struct PacketWriter {
    buf: Vec<u8>,
}

impl PacketWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }

    #[inline]
    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    #[inline]
    fn write_bool(&mut self, v: bool) {
        self.buf.push(u8::from(v));
    }

    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn write_f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.write_u32(bytes.len() as u32);
        self.buf.extend_from_slice(bytes);
    }

    #[inline]
    fn write_point(&mut self, pt: Point) {
        self.write_f32(pt.x);
        self.write_f32(pt.y);
    }

    #[inline]
    fn write_rect(&mut self, r: Rect) {
        self.write_f32(r.origin.x);
        self.write_f32(r.origin.y);
        self.write_f32(r.size.width);
        self.write_f32(r.size.height);
    }

    #[inline]
    fn write_opt_rect(&mut self, opt: Option<Rect>) {
        match opt {
            Some(r) => {
                self.write_u8(1);
                self.write_rect(r);
            }
            None => self.write_u8(0),
        }
    }

    #[inline]
    fn write_color(&mut self, c: Color) {
        self.write_f32(c.red);
        self.write_f32(c.green);
        self.write_f32(c.blue);
        self.write_f32(c.alpha);
    }

    #[inline]
    fn write_matrix(&mut self, m: &Matrix) {
        for &val in &m.m {
            self.write_f32(val);
        }
    }

    #[inline]
    fn write_blend_mode(&mut self, bm: BlendMode) {
        self.write_u8(bm as u8);
    }

    #[inline]
    fn write_tile_mode(&mut self, tm: TileMode) {
        let code = match tm {
            TileMode::Clamp => 0u8,
            TileMode::Repeat => 1u8,
            TileMode::Mirror => 2u8,
            TileMode::Decal => 3u8,
        };
        self.write_u8(code);
    }

    fn write_mesh(&mut self, mesh: &TessellatedMesh) {
        self.write_u32(mesh.vertices.len() as u32);
        for v in &mesh.vertices {
            self.write_f32(v.position[0]);
            self.write_f32(v.position[1]);
        }
        self.write_u32(mesh.indices.len() as u32);
        for &idx in &mesh.indices {
            self.write_u32(idx);
        }
    }

    fn write_stops(&mut self, stops: &[GradientStop]) {
        self.write_u32(stops.len() as u32);
        for s in stops {
            self.write_f32(s.offset);
            self.write_color(s.color);
        }
    }

    fn write_atlas_type(&mut self, t: GlyphAtlasType) {
        let code = match t {
            GlyphAtlasType::AlphaBitmap => 0u8,
            GlyphAtlasType::ColorBitmap => 1u8,
        };
        self.write_u8(code);
    }

    fn write_atlas_descriptor(&mut self, opt: &Option<Arc<GlyphAtlas>>) {
        match opt {
            Some(a) => {
                self.write_u8(1);
                self.write_u64(a.atlas_id);
                self.write_atlas_type(a.atlas_type);
                self.write_u32(a.size.width.max(1) as u32);
                self.write_u32(a.size.height.max(1) as u32);
                self.write_u64(a.generation);
                self.write_u64(a.resize_epoch);
            }
            None => self.write_u8(0),
        }
    }

    fn write_atlas_delta(&mut self, opt: &Option<AtlasRowDelta>) {
        match opt {
            Some(d) => {
                self.write_u8(1);
                self.write_u64(d.atlas_id);
                self.write_atlas_type(d.atlas_type);
                self.write_u32(d.width);
                self.write_u32(d.height);
                self.write_u64(d.generation);
                self.write_u64(d.resize_epoch);
                self.write_bool(d.full_reset);
                self.write_u32(d.y_min);
                self.write_u32(d.y_max);
                self.write_bytes(&d.row_pixels);
            }
            None => self.write_u8(0),
        }
    }

    fn write_color_filter(&mut self, cf: &LayerColorFilter) {
        match cf {
            LayerColorFilter::Mode(c, bm) => {
                self.write_u8(0);
                self.write_color(*c);
                self.write_blend_mode(*bm);
            }
            LayerColorFilter::Matrix(m) => {
                self.write_u8(1);
                for &v in m {
                    self.write_f32(v);
                }
            }
            LayerColorFilter::SrgbToLinear => self.write_u8(2),
            LayerColorFilter::LinearToSrgb => self.write_u8(3),
            LayerColorFilter::Compose(outer, inner) => {
                self.write_u8(4);
                self.write_color_filter(outer);
                self.write_color_filter(inner);
            }
        }
    }

    fn write_image_filter(&mut self, imf: &LayerImageFilter) {
        match imf {
            LayerImageFilter::Blur {
                sigma_x,
                sigma_y,
                tile_mode,
            } => {
                self.write_u8(0);
                self.write_f32(*sigma_x);
                self.write_f32(*sigma_y);
                self.write_tile_mode(*tile_mode);
            }
            LayerImageFilter::Dilate { radius_x, radius_y } => {
                self.write_u8(1);
                self.write_f32(*radius_x);
                self.write_f32(*radius_y);
            }
            LayerImageFilter::Erode { radius_x, radius_y } => {
                self.write_u8(2);
                self.write_f32(*radius_x);
                self.write_f32(*radius_y);
            }
            LayerImageFilter::Matrix(m) => {
                self.write_u8(3);
                self.write_matrix(m);
            }
            LayerImageFilter::ColorFilter(cf) => {
                self.write_u8(4);
                self.write_color_filter(cf);
            }
            LayerImageFilter::Compose(outer, inner) => {
                self.write_u8(5);
                self.write_image_filter(outer);
                self.write_image_filter(inner);
            }
        }
    }

    fn write_commands(&mut self, commands: &[DrawCommand]) {
        self.write_u32(commands.len() as u32);
        for cmd in commands {
            self.write_command(cmd);
        }
    }

    fn write_command(&mut self, cmd: &DrawCommand) {
        match cmd {
            DrawCommand::SolidFill {
                mesh,
                transform,
                color,
                blend_mode,
                clip_rect,
            } => {
                self.write_u8(0);
                self.write_mesh(mesh);
                self.write_matrix(transform);
                self.write_color(*color);
                self.write_blend_mode(*blend_mode);
                self.write_opt_rect(*clip_rect);
            }
            DrawCommand::LinearGradientFill {
                mesh,
                transform,
                gradient,
                blend_mode,
                clip_rect,
            } => {
                self.write_u8(1);
                self.write_mesh(mesh);
                self.write_matrix(transform);
                self.write_point(gradient.start_point);
                self.write_point(gradient.end_point);
                self.write_stops(&gradient.stops);
                self.write_tile_mode(gradient.tile_mode);
                self.write_f32(gradient.alpha);
                self.write_blend_mode(*blend_mode);
                self.write_opt_rect(*clip_rect);
            }
            DrawCommand::RadialGradientFill {
                mesh,
                transform,
                gradient,
                blend_mode,
                clip_rect,
            } => {
                self.write_u8(2);
                self.write_mesh(mesh);
                self.write_matrix(transform);
                self.write_point(gradient.center);
                self.write_f32(gradient.radius);
                self.write_stops(&gradient.stops);
                self.write_tile_mode(gradient.tile_mode);
                self.write_f32(gradient.alpha);
                self.write_blend_mode(*blend_mode);
                self.write_opt_rect(*clip_rect);
            }
            DrawCommand::SweepGradientFill {
                mesh,
                transform,
                gradient,
                blend_mode,
                clip_rect,
            } => {
                self.write_u8(3);
                self.write_mesh(mesh);
                self.write_matrix(transform);
                self.write_point(gradient.center);
                self.write_f32(gradient.start_angle_deg);
                self.write_f32(gradient.end_angle_deg);
                self.write_stops(&gradient.stops);
                self.write_tile_mode(gradient.tile_mode);
                self.write_f32(gradient.alpha);
                self.write_blend_mode(*blend_mode);
                self.write_opt_rect(*clip_rect);
            }
            DrawCommand::ConicalGradientFill {
                mesh,
                transform,
                gradient,
                blend_mode,
                clip_rect,
            } => {
                self.write_u8(4);
                self.write_mesh(mesh);
                self.write_matrix(transform);
                self.write_point(gradient.start_center);
                self.write_f32(gradient.start_radius);
                self.write_point(gradient.end_center);
                self.write_f32(gradient.end_radius);
                self.write_stops(&gradient.stops);
                self.write_tile_mode(gradient.tile_mode);
                self.write_f32(gradient.alpha);
                self.write_blend_mode(*blend_mode);
                self.write_opt_rect(*clip_rect);
            }
            DrawCommand::RRectBlur {
                transform,
                params,
                blend_mode,
                clip_rect,
            } => {
                self.write_u8(5);
                self.write_matrix(transform);
                self.write_rect(params.rect);
                self.write_f32(params.corner_radius);
                self.write_f32(params.blur_sigma);
                self.write_color(params.color);
                self.write_blend_mode(*blend_mode);
                self.write_opt_rect(*clip_rect);
            }
            DrawCommand::GlyphAtlas(batch) => {
                self.write_u8(6);
                self.write_u32(batch.quads.len() as u32);
                for q in &batch.quads {
                    self.write_rect(q.dest_rect);
                    self.write_rect(q.atlas_rect);
                    self.write_color(q.color);
                    self.write_bool(q.is_color);
                }
                self.write_matrix(&batch.transform);
                self.write_atlas_descriptor(&batch.alpha_atlas);
                self.write_atlas_descriptor(&batch.color_atlas);
                self.write_blend_mode(batch.blend_mode);
                self.write_opt_rect(batch.clip_rect);
                match batch.linear_filter {
                    Some(v) => {
                        self.write_u8(1);
                        self.write_bool(v);
                    }
                    None => self.write_u8(0),
                }
                self.write_bool(batch.use_text_color);
                self.write_f32(batch.text_contrast);
                self.write_f32(batch.alpha);
            }
            DrawCommand::TextureFill(batch) => {
                self.write_u8(7);
                self.write_u32(batch.vertices.len() as u32);
                for v in &batch.vertices {
                    self.write_f32(v.position[0]);
                    self.write_f32(v.position[1]);
                    self.write_f32(v.uv[0]);
                    self.write_f32(v.uv[1]);
                }
                self.write_u32(batch.indices.len() as u32);
                for &idx in &batch.indices {
                    self.write_u32(idx);
                }
                self.write_matrix(&batch.transform);
                self.write_u64(batch.texture.id);
                self.write_u32(batch.texture.width);
                self.write_u32(batch.texture.height);
                self.write_f32(batch.alpha);
                self.write_tile_mode(batch.tile_mode_x);
                self.write_tile_mode(batch.tile_mode_y);
                self.write_bool(batch.linear_filter);
                self.write_blend_mode(batch.blend_mode);
                self.write_opt_rect(batch.clip_rect);
            }
            DrawCommand::PushClip {
                mesh,
                transform,
                is_difference,
                clip_rect,
            } => {
                self.write_u8(8);
                self.write_mesh(mesh);
                self.write_matrix(transform);
                self.write_bool(*is_difference);
                self.write_opt_rect(*clip_rect);
            }
            DrawCommand::PopClip { clip_rect } => {
                self.write_u8(9);
                self.write_opt_rect(*clip_rect);
            }
            DrawCommand::SaveLayer {
                bounds,
                commands,
                alpha,
                blend_mode,
                color_filter,
                image_filter,
                backdrop_filter,
                clip_rect,
            } => {
                self.write_u8(10);
                self.write_rect(*bounds);
                self.write_commands(commands);
                self.write_f32(*alpha);
                self.write_blend_mode(*blend_mode);
                match color_filter {
                    Some(cf) => {
                        self.write_u8(1);
                        self.write_color_filter(cf);
                    }
                    None => self.write_u8(0),
                }
                match image_filter {
                    Some(imf) => {
                        self.write_u8(1);
                        self.write_image_filter(imf);
                    }
                    None => self.write_u8(0),
                }
                match backdrop_filter {
                    Some(bf) => {
                        self.write_u8(1);
                        self.write_image_filter(bf);
                    }
                    None => self.write_u8(0),
                }
                self.write_opt_rect(*clip_rect);
            }
        }
    }
}

struct PacketReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    cached_alpha_desc: Option<Arc<GlyphAtlas>>,
    cached_color_desc: Option<Arc<GlyphAtlas>>,
}

impl<'a> PacketReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            cached_alpha_desc: None,
            cached_color_desc: None,
        }
    }

    #[inline]
    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N], &'static str> {
        if self.pos + N > self.bytes.len() {
            return Err("unexpected EOF in WorkerFramePacket");
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8, &'static str> {
        Ok(self.read_exact::<1>()?[0])
    }

    #[inline]
    fn read_bool(&mut self) -> Result<bool, &'static str> {
        Ok(self.read_u8()? != 0)
    }

    #[inline]
    fn read_u32(&mut self) -> Result<u32, &'static str> {
        Ok(u32::from_le_bytes(self.read_exact::<4>()?))
    }

    #[inline]
    fn read_u64(&mut self) -> Result<u64, &'static str> {
        Ok(u64::from_le_bytes(self.read_exact::<8>()?))
    }

    #[inline]
    fn read_f32(&mut self) -> Result<f32, &'static str> {
        Ok(f32::from_le_bytes(self.read_exact::<4>()?))
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>, &'static str> {
        let len = self.read_u32()? as usize;
        if self.pos + len > self.bytes.len() {
            return Err("byte slice exceeds packet length");
        }
        let slice = self.bytes[self.pos..self.pos + len].to_vec();
        self.pos += len;
        Ok(slice)
    }

    #[inline]
    fn read_point(&mut self) -> Result<Point, &'static str> {
        let x = self.read_f32()?;
        let y = self.read_f32()?;
        Ok(Point::new(x, y))
    }

    #[inline]
    fn read_rect(&mut self) -> Result<Rect, &'static str> {
        let x = self.read_f32()?;
        let y = self.read_f32()?;
        let w = self.read_f32()?;
        let h = self.read_f32()?;
        Ok(Rect::from_xywh(x, y, w, h))
    }

    #[inline]
    fn read_opt_rect(&mut self) -> Result<Option<Rect>, &'static str> {
        if self.read_u8()? != 0 {
            Ok(Some(self.read_rect()?))
        } else {
            Ok(None)
        }
    }

    #[inline]
    fn read_color(&mut self) -> Result<Color, &'static str> {
        let r = self.read_f32()?;
        let g = self.read_f32()?;
        let b = self.read_f32()?;
        let a = self.read_f32()?;
        Ok(Color::new(r, g, b, a))
    }

    #[inline]
    fn read_matrix(&mut self) -> Result<Matrix, &'static str> {
        let mut m = [0.0f32; 16];
        for slot in &mut m {
            *slot = self.read_f32()?;
        }
        Ok(Matrix { m })
    }

    #[inline]
    fn read_blend_mode(&mut self) -> Result<BlendMode, &'static str> {
        let idx = self.read_u8()? as usize;
        Ok(BlendMode::ALL
            .get(idx)
            .copied()
            .unwrap_or(BlendMode::SrcOver))
    }

    #[inline]
    fn read_tile_mode(&mut self) -> Result<TileMode, &'static str> {
        Ok(match self.read_u8()? {
            1 => TileMode::Repeat,
            2 => TileMode::Mirror,
            3 => TileMode::Decal,
            _ => TileMode::Clamp,
        })
    }

    fn read_mesh(&mut self) -> Result<TessellatedMesh, &'static str> {
        let v_len = self.read_u32()? as usize;
        let mut vertices = Vec::with_capacity(v_len);
        for _ in 0..v_len {
            let x = self.read_f32()?;
            let y = self.read_f32()?;
            vertices.push(Vertex2D { position: [x, y] });
        }
        let i_len = self.read_u32()? as usize;
        let mut indices = Vec::with_capacity(i_len);
        for _ in 0..i_len {
            indices.push(self.read_u32()?);
        }
        Ok(TessellatedMesh { vertices, indices })
    }

    fn read_stops(&mut self) -> Result<Vec<GradientStop>, &'static str> {
        let len = self.read_u32()? as usize;
        let mut stops = Vec::with_capacity(len);
        for _ in 0..len {
            let offset = self.read_f32()?;
            let color = self.read_color()?;
            stops.push(GradientStop::new(offset, color));
        }
        Ok(stops)
    }

    fn read_atlas_type(&mut self) -> Result<GlyphAtlasType, &'static str> {
        Ok(match self.read_u8()? {
            1 => GlyphAtlasType::ColorBitmap,
            _ => GlyphAtlasType::AlphaBitmap,
        })
    }

    fn read_atlas_descriptor(&mut self) -> Result<Option<Arc<GlyphAtlas>>, &'static str> {
        if self.read_u8()? == 0 {
            return Ok(None);
        }
        let atlas_id = self.read_u64()?;
        let atlas_type = self.read_atlas_type()?;
        let width = self.read_u32()?;
        let height = self.read_u32()?;
        let generation = self.read_u64()?;
        let resize_epoch = self.read_u64()?;

        let cache_slot = match atlas_type {
            GlyphAtlasType::AlphaBitmap => &mut self.cached_alpha_desc,
            GlyphAtlasType::ColorBitmap => &mut self.cached_color_desc,
        };
        if let Some(existing) = cache_slot {
            if existing.atlas_id == atlas_id
                && existing.size.width == i64::from(width)
                && existing.size.height == i64::from(height)
                && existing.generation == generation
                && existing.resize_epoch == resize_epoch
            {
                return Ok(Some(Arc::clone(existing)));
            }
        }

        let arc = Arc::new(GlyphAtlas {
            atlas_id,
            atlas_type,
            size: ISize::new(i64::from(width), i64::from(height)),
            pixels: Vec::new(),
            generation,
            resize_epoch,
            row_generations: Vec::new(),
        });
        *cache_slot = Some(Arc::clone(&arc));
        Ok(Some(arc))
    }

    fn read_atlas_delta(&mut self) -> Result<Option<AtlasRowDelta>, &'static str> {
        if self.read_u8()? == 0 {
            return Ok(None);
        }
        let atlas_id = self.read_u64()?;
        let atlas_type = self.read_atlas_type()?;
        let width = self.read_u32()?;
        let height = self.read_u32()?;
        let generation = self.read_u64()?;
        let resize_epoch = self.read_u64()?;
        let full_reset = self.read_bool()?;
        let y_min = self.read_u32()?;
        let y_max = self.read_u32()?;
        let row_pixels = self.read_bytes()?;
        Ok(Some(AtlasRowDelta {
            atlas_id,
            atlas_type,
            width,
            height,
            generation,
            resize_epoch,
            full_reset,
            y_min,
            y_max,
            row_pixels,
        }))
    }

    fn read_color_filter(&mut self) -> Result<LayerColorFilter, &'static str> {
        match self.read_u8()? {
            0 => {
                let c = self.read_color()?;
                let bm = self.read_blend_mode()?;
                Ok(LayerColorFilter::Mode(c, bm))
            }
            1 => {
                let mut m = [0.0f32; 20];
                for v in &mut m {
                    *v = self.read_f32()?;
                }
                Ok(LayerColorFilter::Matrix(m))
            }
            2 => Ok(LayerColorFilter::SrgbToLinear),
            3 => Ok(LayerColorFilter::LinearToSrgb),
            4 => {
                let outer = self.read_color_filter()?;
                let inner = self.read_color_filter()?;
                Ok(LayerColorFilter::Compose(Box::new(outer), Box::new(inner)))
            }
            _ => Err("invalid LayerColorFilter tag"),
        }
    }

    fn read_image_filter(&mut self) -> Result<LayerImageFilter, &'static str> {
        match self.read_u8()? {
            0 => {
                let sigma_x = self.read_f32()?;
                let sigma_y = self.read_f32()?;
                let tile_mode = self.read_tile_mode()?;
                Ok(LayerImageFilter::Blur {
                    sigma_x,
                    sigma_y,
                    tile_mode,
                })
            }
            1 => {
                let radius_x = self.read_f32()?;
                let radius_y = self.read_f32()?;
                Ok(LayerImageFilter::Dilate { radius_x, radius_y })
            }
            2 => {
                let radius_x = self.read_f32()?;
                let radius_y = self.read_f32()?;
                Ok(LayerImageFilter::Erode { radius_x, radius_y })
            }
            3 => Ok(LayerImageFilter::Matrix(self.read_matrix()?)),
            4 => Ok(LayerImageFilter::ColorFilter(self.read_color_filter()?)),
            5 => {
                let outer = self.read_image_filter()?;
                let inner = self.read_image_filter()?;
                Ok(LayerImageFilter::Compose(Box::new(outer), Box::new(inner)))
            }
            _ => Err("invalid LayerImageFilter tag"),
        }
    }

    fn read_commands(
        &mut self,
        images: &HashMap<u64, ImageTexture>,
    ) -> Result<Vec<DrawCommand>, &'static str> {
        let len = self.read_u32()? as usize;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_command(images)?);
        }
        Ok(out)
    }

    fn read_command(
        &mut self,
        images: &HashMap<u64, ImageTexture>,
    ) -> Result<DrawCommand, &'static str> {
        match self.read_u8()? {
            0 => {
                let mesh = self.read_mesh()?;
                let transform = self.read_matrix()?;
                let color = self.read_color()?;
                let blend_mode = self.read_blend_mode()?;
                let clip_rect = self.read_opt_rect()?;
                Ok(DrawCommand::SolidFill {
                    mesh,
                    transform,
                    color,
                    blend_mode,
                    clip_rect,
                })
            }
            1 => {
                let mesh = self.read_mesh()?;
                let transform = self.read_matrix()?;
                let start_point = self.read_point()?;
                let end_point = self.read_point()?;
                let stops = self.read_stops()?;
                let tile_mode = self.read_tile_mode()?;
                let alpha = self.read_f32()?;
                let blend_mode = self.read_blend_mode()?;
                let clip_rect = self.read_opt_rect()?;
                Ok(DrawCommand::LinearGradientFill {
                    mesh,
                    transform,
                    gradient: LinearGradientParams::new(
                        start_point,
                        end_point,
                        stops,
                        tile_mode,
                        alpha,
                    ),
                    blend_mode,
                    clip_rect,
                })
            }
            2 => {
                let mesh = self.read_mesh()?;
                let transform = self.read_matrix()?;
                let center = self.read_point()?;
                let radius = self.read_f32()?;
                let stops = self.read_stops()?;
                let tile_mode = self.read_tile_mode()?;
                let alpha = self.read_f32()?;
                let blend_mode = self.read_blend_mode()?;
                let clip_rect = self.read_opt_rect()?;
                Ok(DrawCommand::RadialGradientFill {
                    mesh,
                    transform,
                    gradient: RadialGradientParams::new(center, radius, stops, tile_mode, alpha),
                    blend_mode,
                    clip_rect,
                })
            }
            3 => {
                let mesh = self.read_mesh()?;
                let transform = self.read_matrix()?;
                let center = self.read_point()?;
                let start_angle_deg = self.read_f32()?;
                let end_angle_deg = self.read_f32()?;
                let stops = self.read_stops()?;
                let tile_mode = self.read_tile_mode()?;
                let alpha = self.read_f32()?;
                let blend_mode = self.read_blend_mode()?;
                let clip_rect = self.read_opt_rect()?;
                Ok(DrawCommand::SweepGradientFill {
                    mesh,
                    transform,
                    gradient: SweepGradientParams::new(
                        center,
                        start_angle_deg,
                        end_angle_deg,
                        stops,
                        tile_mode,
                        alpha,
                    ),
                    blend_mode,
                    clip_rect,
                })
            }
            4 => {
                let mesh = self.read_mesh()?;
                let transform = self.read_matrix()?;
                let start_center = self.read_point()?;
                let start_radius = self.read_f32()?;
                let end_center = self.read_point()?;
                let end_radius = self.read_f32()?;
                let stops = self.read_stops()?;
                let tile_mode = self.read_tile_mode()?;
                let alpha = self.read_f32()?;
                let blend_mode = self.read_blend_mode()?;
                let clip_rect = self.read_opt_rect()?;
                Ok(DrawCommand::ConicalGradientFill {
                    mesh,
                    transform,
                    gradient: ConicalGradientParams::new(
                        start_center,
                        start_radius,
                        end_center,
                        end_radius,
                        stops,
                        tile_mode,
                        alpha,
                    ),
                    blend_mode,
                    clip_rect,
                })
            }
            5 => {
                let transform = self.read_matrix()?;
                let rect = self.read_rect()?;
                let corner_radius = self.read_f32()?;
                let blur_sigma = self.read_f32()?;
                let color = self.read_color()?;
                let blend_mode = self.read_blend_mode()?;
                let clip_rect = self.read_opt_rect()?;
                Ok(DrawCommand::RRectBlur {
                    transform,
                    params: RRectBlurParams::new(rect, corner_radius, blur_sigma, color),
                    blend_mode,
                    clip_rect,
                })
            }
            6 => {
                let q_len = self.read_u32()? as usize;
                let mut quads = Vec::with_capacity(q_len);
                for _ in 0..q_len {
                    let dest_rect = self.read_rect()?;
                    let atlas_rect = self.read_rect()?;
                    let color = self.read_color()?;
                    let is_color = self.read_bool()?;
                    quads.push(AtlasGlyphQuad {
                        dest_rect,
                        atlas_rect,
                        color,
                        is_color,
                    });
                }
                let transform = self.read_matrix()?;
                let alpha_atlas = self.read_atlas_descriptor()?;
                let color_atlas = self.read_atlas_descriptor()?;
                let blend_mode = self.read_blend_mode()?;
                let clip_rect = self.read_opt_rect()?;
                let linear_filter = if self.read_u8()? != 0 {
                    Some(self.read_bool()?)
                } else {
                    None
                };
                let use_text_color = self.read_bool()?;
                let text_contrast = self.read_f32()?;
                let alpha = self.read_f32()?;
                Ok(DrawCommand::GlyphAtlas(GlyphAtlasBatch {
                    quads,
                    transform,
                    alpha_atlas,
                    color_atlas,
                    blend_mode,
                    clip_rect,
                    linear_filter,
                    use_text_color,
                    text_contrast,
                    alpha,
                }))
            }
            7 => {
                let v_len = self.read_u32()? as usize;
                let mut vertices = Vec::with_capacity(v_len);
                for _ in 0..v_len {
                    let x = self.read_f32()?;
                    let y = self.read_f32()?;
                    let u = self.read_f32()?;
                    let v = self.read_f32()?;
                    vertices.push(UvVertex::new(x, y, u, v));
                }
                let i_len = self.read_u32()? as usize;
                let mut indices = Vec::with_capacity(i_len);
                for _ in 0..i_len {
                    indices.push(self.read_u32()?);
                }
                let transform = self.read_matrix()?;
                let tex_id = self.read_u64()?;
                let tex_w = self.read_u32()?;
                let tex_h = self.read_u32()?;
                let texture = images
                    .get(&tex_id)
                    .cloned()
                    .unwrap_or_else(|| ImageTexture::new(tex_id, tex_w, tex_h, Vec::new()));
                let alpha = self.read_f32()?;
                let tile_mode_x = self.read_tile_mode()?;
                let tile_mode_y = self.read_tile_mode()?;
                let linear_filter = self.read_bool()?;
                let blend_mode = self.read_blend_mode()?;
                let clip_rect = self.read_opt_rect()?;
                Ok(DrawCommand::TextureFill(TextureFillBatch {
                    vertices,
                    indices,
                    transform,
                    texture,
                    alpha,
                    tile_mode_x,
                    tile_mode_y,
                    linear_filter,
                    blend_mode,
                    clip_rect,
                }))
            }
            8 => {
                let mesh = self.read_mesh()?;
                let transform = self.read_matrix()?;
                let is_difference = self.read_bool()?;
                let clip_rect = self.read_opt_rect()?;
                Ok(DrawCommand::PushClip {
                    mesh,
                    transform,
                    is_difference,
                    clip_rect,
                })
            }
            9 => {
                let clip_rect = self.read_opt_rect()?;
                Ok(DrawCommand::PopClip { clip_rect })
            }
            10 => {
                let bounds = self.read_rect()?;
                let commands = self.read_commands(images)?;
                let alpha = self.read_f32()?;
                let blend_mode = self.read_blend_mode()?;
                let color_filter = if self.read_u8()? != 0 {
                    Some(self.read_color_filter()?)
                } else {
                    None
                };
                let image_filter = if self.read_u8()? != 0 {
                    Some(self.read_image_filter()?)
                } else {
                    None
                };
                let backdrop_filter = if self.read_u8()? != 0 {
                    Some(self.read_image_filter()?)
                } else {
                    None
                };
                let clip_rect = self.read_opt_rect()?;
                Ok(DrawCommand::SaveLayer {
                    bounds,
                    commands,
                    alpha,
                    blend_mode,
                    color_filter,
                    image_filter,
                    backdrop_filter,
                    clip_rect,
                })
            }
            _ => Err("invalid DrawCommand tag"),
        }
    }
}

/// Encodes a [`WorkerFramePacket`] into a compact binary `Vec<u8>`.
pub fn encode_frame_packet(packet: &WorkerFramePacket) -> Vec<u8> {
    let mut writer = PacketWriter::with_capacity(4096);
    writer.buf.extend_from_slice(&PACKET_MAGIC);
    writer.write_u32(PACKET_VERSION);
    writer.write_u32(packet.width);
    writer.write_u32(packet.height);
    writer.write_bool(packet.alpha_used);
    writer.write_bool(packet.color_used);
    writer.write_atlas_delta(&packet.alpha_delta);
    writer.write_atlas_delta(&packet.color_delta);

    writer.write_u32(packet.new_images.len() as u32);
    for img in &packet.new_images {
        writer.write_u64(img.id);
        writer.write_u32(img.width);
        writer.write_u32(img.height);
        writer.write_bytes(&img.rgba_premultiplied);
    }

    writer.write_u32(packet.disposed_image_ids.len() as u32);
    for &id in &packet.disposed_image_ids {
        writer.write_u64(id);
    }

    writer.write_u32(packet.picture_commands.len() as u32);
    for cmds in &packet.picture_commands {
        writer.write_commands(cmds);
    }
    writer.buf
}

/// Decodes a [`WorkerFramePacket`] from bytes, updating `worker_images` with any disposed or
/// newly uploaded [`ImageTexture`]s before resolving [`TextureFillBatch`] references.
pub fn decode_frame_packet(
    bytes: &[u8],
    worker_images: &mut HashMap<u64, ImageTexture>,
) -> Result<WorkerFramePacket, &'static str> {
    let mut reader = PacketReader::new(bytes);
    let magic = reader.read_exact::<4>()?;
    if magic != PACKET_MAGIC {
        return Err("invalid WorkerFramePacket magic");
    }
    let version = reader.read_u32()?;
    if version != PACKET_VERSION {
        return Err("unsupported WorkerFramePacket version");
    }

    let width = reader.read_u32()?;
    let height = reader.read_u32()?;
    let alpha_used = reader.read_bool()?;
    let color_used = reader.read_bool()?;
    let alpha_delta = reader.read_atlas_delta()?;
    let color_delta = reader.read_atlas_delta()?;

    let new_img_len = reader.read_u32()? as usize;
    let mut new_images = Vec::with_capacity(new_img_len);
    for _ in 0..new_img_len {
        let id = reader.read_u64()?;
        let w = reader.read_u32()?;
        let h = reader.read_u32()?;
        let rgba = reader.read_bytes()?;
        worker_images.insert(id, ImageTexture::new(id, w, h, rgba.clone()));
        new_images.push(ImageDelta {
            id,
            width: w,
            height: h,
            rgba_premultiplied: rgba,
        });
    }

    let disp_len = reader.read_u32()? as usize;
    let mut disposed_image_ids = Vec::with_capacity(disp_len);
    for _ in 0..disp_len {
        let id = reader.read_u64()?;
        worker_images.remove(&id);
        disposed_image_ids.push(id);
    }

    let pic_len = reader.read_u32()? as usize;
    let mut picture_commands = Vec::with_capacity(pic_len);
    for _ in 0..pic_len {
        picture_commands.push(reader.read_commands(worker_images)?);
    }

    Ok(WorkerFramePacket {
        width,
        height,
        alpha_used,
        color_used,
        alpha_delta,
        color_delta,
        new_images,
        disposed_image_ids,
        picture_commands,
    })
}

// =============================================================================
// Dedicated Web Worker Wasm Entry Points (`impeller_worker.js`)
// =============================================================================

#[cfg(target_arch = "wasm32")]
struct WorkerGpuSurface {
    renderer: impeller_wgpu::WgpuRenderer,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    offscreen_canvas: web_sys::OffscreenCanvas,
    width: u32,
    height: u32,
}

#[cfg(target_arch = "wasm32")]
struct WorkerState {
    alpha_atlas: GlyphAtlas,
    color_atlas: GlyphAtlas,
    images: HashMap<u64, ImageTexture>,
    surfaces: HashMap<u32, WorkerGpuSurface>,
}

#[cfg(target_arch = "wasm32")]
impl WorkerState {
    fn new() -> Self {
        Self {
            alpha_atlas: GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 1024, 1024),
            color_atlas: GlyphAtlas::new(GlyphAtlasType::ColorBitmap, 512, 512),
            images: HashMap::new(),
            surfaces: HashMap::new(),
        }
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WORKER_STATE: RefCell<WorkerState> = RefCell::new(WorkerState::new());
}

#[cfg(target_arch = "wasm32")]
fn with_worker_state<R>(f: impl FnOnce(&mut WorkerState) -> R) -> R {
    WORKER_STATE.with(|cell| f(&mut cell.borrow_mut()))
}

#[cfg(target_arch = "wasm32")]
fn worker_now_ms() -> f64 {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("performance"))
        .ok()
        .and_then(|perf| perf.dyn_into::<web_sys::Performance>().ok())
        .map_or(0.0, |perf| perf.now())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn skwasm_worker_init_surface_js(
    surface_handle: u32,
    canvas: JsValue,
    width: u32,
    height: u32,
    prefer_webgpu: bool,
) -> bool {
    let Ok(offscreen) = canvas.dyn_into::<web_sys::OffscreenCanvas>() else {
        return false;
    };
    let target_w = width.max(offscreen.width()).max(1);
    let target_h = height.max(offscreen.height()).max(1);
    offscreen.set_width(target_w);
    offscreen.set_height(target_h);

    let Ok((renderer, surface, mut config)) =
        impeller_wgpu::WgpuRenderer::new_for_offscreen_canvas_with_backend(
            offscreen.clone(),
            prefer_webgpu,
        )
        .await
    else {
        return false;
    };

    if config.width != target_w || config.height != target_h {
        config.width = target_w;
        config.height = target_h;
        surface.configure(renderer.device(), &config);
    }

    with_worker_state(|ws| {
        ws.surfaces.insert(
            surface_handle,
            WorkerGpuSurface {
                renderer,
                surface,
                config,
                offscreen_canvas: offscreen,
                width: target_w,
                height: target_h,
            },
        );
    });
    true
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn skwasm_worker_set_size_js(surface_handle: u32, width: u32, height: u32) {
    let w = width.max(1);
    let h = height.max(1);
    with_worker_state(|ws| {
        if let Some(surf) = ws.surfaces.get_mut(&surface_handle) {
            surf.width = w;
            surf.height = h;
            if surf.config.width != w || surf.config.height != h {
                surf.offscreen_canvas.set_width(w);
                surf.offscreen_canvas.set_height(h);
                surf.config.width = w;
                surf.config.height = h;
                surf.surface.configure(surf.renderer.device(), &surf.config);
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn skwasm_worker_destroy_surface_js(surface_handle: u32) {
    with_worker_state(|ws| {
        ws.surfaces.remove(&surface_handle);
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn skwasm_worker_render_frame_js(surface_handle: u32, packet_bytes: &[u8]) -> JsValue {
    let t_start = worker_now_ms();
    let bitmaps = js_sys::Array::new();
    let mut atlas_sync_ms = 0.0;
    let mut wgpu_render_ms = 0.0;
    let mut bitmap_transfer_ms = 0.0;

    with_worker_state(|ws| {
        let Ok(packet) = decode_frame_packet(packet_bytes, &mut ws.images) else {
            return;
        };

        if let Some(alpha_delta) = &packet.alpha_delta {
            apply_atlas_delta(&mut ws.alpha_atlas, alpha_delta);
        }
        if let Some(color_delta) = &packet.color_delta {
            apply_atlas_delta(&mut ws.color_atlas, color_delta);
        }

        let Some(surf) = ws.surfaces.get_mut(&surface_handle) else {
            return;
        };

        let w = packet.width.max(1);
        let h = packet.height.max(1);
        surf.width = w;
        surf.height = h;
        if surf.config.width != w || surf.config.height != h {
            surf.offscreen_canvas.set_width(w);
            surf.offscreen_canvas.set_height(h);
            surf.config.width = w;
            surf.config.height = h;
            surf.surface.configure(surf.renderer.device(), &surf.config);
        }

        let t_atlas = worker_now_ms();
        if packet.alpha_used || packet.alpha_delta.is_some() {
            let _ = surf.renderer.sync_glyph_atlas(&ws.alpha_atlas);
        }
        if packet.color_used || packet.color_delta.is_some() {
            let _ = surf.renderer.sync_glyph_atlas(&ws.color_atlas);
        }
        atlas_sync_ms = worker_now_ms() - t_atlas;

        for cmds in &packet.picture_commands {
            let t_wgpu = worker_now_ms();
            let frame_res = match surf.surface.get_current_texture() {
                Ok(frame) => Ok(frame),
                Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                    surf.surface.configure(surf.renderer.device(), &surf.config);
                    surf.surface.get_current_texture()
                }
                Err(e) => Err(e),
            };

            if let Ok(frame) = frame_res {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let _ = surf.renderer.render_to_surface_view(
                    &view,
                    w,
                    h,
                    surf.config.format,
                    Color::TRANSPARENT,
                    cmds,
                );
                frame.present();
                wgpu_render_ms += worker_now_ms() - t_wgpu;

                let t_bmp = worker_now_ms();
                if let Ok(bmp) = surf.offscreen_canvas.transfer_to_image_bitmap() {
                    bitmaps.push(&bmp);
                }
                bitmap_transfer_ms += worker_now_ms() - t_bmp;
            } else {
                wgpu_render_ms += worker_now_ms() - t_wgpu;
            }
        }
    });

    let worker_elapsed_ms = (worker_now_ms() - t_start).max(0.0);
    let out = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&out, &JsValue::from_str("imageBitmaps"), &bitmaps);
    let _ = js_sys::Reflect::set(
        &out,
        &JsValue::from_str("workerElapsedMs"),
        &JsValue::from_f64(worker_elapsed_ms),
    );
    let _ = js_sys::Reflect::set(
        &out,
        &JsValue::from_str("atlasSyncMs"),
        &JsValue::from_f64(atlas_sync_ms),
    );
    let _ = js_sys::Reflect::set(
        &out,
        &JsValue::from_str("wgpuRenderMs"),
        &JsValue::from_f64(wgpu_render_ms),
    );
    let _ = js_sys::Reflect::set(
        &out,
        &JsValue::from_str("bitmapTransferMs"),
        &JsValue::from_f64(bitmap_transfer_ms),
    );
    out.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C++ source: skwasm/surface_mt.cpp:WorkerFramePacketRoundTripAllDrawCommands
    #[test]
    fn test_worker_frame_packet_round_trip_all_draw_commands() {
        let mut main_alpha = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 64, 64);
        main_alpha.pixels[128..192].fill(0x7F);
        main_alpha.generation = 2;
        main_alpha.row_generations[2] = 2;

        let mut alpha_state = (0u64, 0u64, 0u64);
        let alpha_delta = build_atlas_delta(&main_alpha, &mut alpha_state);
        assert!(alpha_delta
            .as_ref()
            .is_some_and(|d| d.full_reset && d.y_max == 3));

        let alpha_desc = Arc::new(main_alpha.descriptor_without_pixels());
        let tex = ImageTexture::new(42, 2, 2, vec![255u8; 16]);
        let mesh = TessellatedMesh {
            vertices: vec![
                Vertex2D {
                    position: [0.0, 0.0],
                },
                Vertex2D {
                    position: [10.0, 0.0],
                },
                Vertex2D {
                    position: [10.0, 10.0],
                },
            ],
            indices: vec![0, 1, 2],
        };
        let clip = Some(Rect::from_xywh(1.0, 2.0, 50.0, 60.0));

        let cmds = vec![
            DrawCommand::SolidFill {
                mesh: mesh.clone(),
                transform: Matrix::IDENTITY,
                color: Color::RED,
                blend_mode: BlendMode::SrcOver,
                clip_rect: clip,
            },
            DrawCommand::LinearGradientFill {
                mesh: mesh.clone(),
                transform: Matrix::make_translation(4.0, 5.0, 0.0),
                gradient: LinearGradientParams::two_stop(
                    Point::new(0.0, 0.0),
                    Point::new(10.0, 10.0),
                    Color::RED,
                    Color::BLUE,
                    TileMode::Mirror,
                ),
                blend_mode: BlendMode::Screen,
                clip_rect: None,
            },
            DrawCommand::RadialGradientFill {
                mesh: mesh.clone(),
                transform: Matrix::IDENTITY,
                gradient: RadialGradientParams::two_stop(
                    Point::new(5.0, 5.0),
                    12.0,
                    Color::GREEN,
                    Color::WHITE,
                    TileMode::Clamp,
                ),
                blend_mode: BlendMode::SrcOver,
                clip_rect: clip,
            },
            DrawCommand::SweepGradientFill {
                mesh: mesh.clone(),
                transform: Matrix::IDENTITY,
                gradient: SweepGradientParams::two_stop(
                    Point::new(8.0, 8.0),
                    0.0,
                    270.0,
                    Color::RED,
                    Color::GREEN,
                    TileMode::Repeat,
                ),
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::ConicalGradientFill {
                mesh: mesh.clone(),
                transform: Matrix::IDENTITY,
                gradient: ConicalGradientParams::two_stop(
                    Point::new(2.0, 2.0),
                    1.0,
                    Point::new(8.0, 8.0),
                    10.0,
                    Color::WHITE,
                    Color::BLACK,
                    TileMode::Decal,
                ),
                blend_mode: BlendMode::SrcOver,
                clip_rect: None,
            },
            DrawCommand::RRectBlur {
                transform: Matrix::IDENTITY,
                params: RRectBlurParams::new(
                    Rect::from_xywh(4.0, 4.0, 30.0, 20.0),
                    6.0,
                    3.5,
                    Color::BLACK,
                ),
                blend_mode: BlendMode::SrcOver,
                clip_rect: clip,
            },
            DrawCommand::GlyphAtlas(
                GlyphAtlasBatch::new(
                    vec![AtlasGlyphQuad {
                        dest_rect: Rect::from_xywh(10.0, 12.0, 8.0, 14.0),
                        atlas_rect: Rect::from_xywh(1.0, 2.0, 8.0, 14.0),
                        color: Color::WHITE,
                        is_color: false,
                    }],
                    Matrix::IDENTITY,
                    Some(Arc::clone(&alpha_desc)),
                    None,
                    BlendMode::SrcOver,
                    clip,
                )
                .with_linear_filter(true)
                .with_use_text_color(true)
                .with_text_contrast(1.4)
                .with_alpha(0.85),
            ),
            DrawCommand::TextureFill(TextureFillBatch::from_rect(
                Rect::from_xywh(0.0, 0.0, 2.0, 2.0),
                Rect::from_xywh(20.0, 20.0, 40.0, 40.0),
                Matrix::IDENTITY,
                tex.clone(),
                0.9,
                true,
                BlendMode::SrcOver,
                clip,
            )),
            DrawCommand::PushClip {
                mesh: mesh.clone(),
                transform: Matrix::IDENTITY,
                is_difference: true,
                clip_rect: clip,
            },
            DrawCommand::PopClip { clip_rect: clip },
            DrawCommand::SaveLayer {
                bounds: Rect::from_xywh(0.0, 0.0, 100.0, 100.0),
                commands: vec![DrawCommand::SolidFill {
                    mesh,
                    transform: Matrix::IDENTITY,
                    color: Color::BLUE,
                    blend_mode: BlendMode::Multiply,
                    clip_rect: None,
                }],
                alpha: 0.75,
                blend_mode: BlendMode::SrcOver,
                color_filter: Some(LayerColorFilter::Compose(
                    Box::new(LayerColorFilter::SrgbToLinear),
                    Box::new(LayerColorFilter::Mode(Color::GREEN, BlendMode::SrcIn)),
                )),
                image_filter: Some(LayerImageFilter::Compose(
                    Box::new(LayerImageFilter::Blur {
                        sigma_x: 3.0,
                        sigma_y: 4.0,
                        tile_mode: TileMode::Decal,
                    }),
                    Box::new(LayerImageFilter::Dilate {
                        radius_x: 1.0,
                        radius_y: 2.0,
                    }),
                )),
                backdrop_filter: Some(LayerImageFilter::Erode {
                    radius_x: 1.0,
                    radius_y: 1.0,
                }),
                clip_rect: clip,
            },
        ];

        let mut uploaded_imgs = HashSet::new();
        let mut new_images = Vec::new();
        collect_missing_images(&cmds, &mut uploaded_imgs, &mut new_images);
        assert_eq!(new_images.len(), 1);
        assert_eq!(new_images[0].id, 42);

        let (alpha_used, color_used) = commands_atlas_usage(&cmds);
        assert!(alpha_used);
        assert!(!color_used);

        let packet = WorkerFramePacket {
            width: 800,
            height: 600,
            alpha_used,
            color_used,
            alpha_delta,
            color_delta: None,
            new_images,
            disposed_image_ids: vec![99],
            picture_commands: vec![cmds.clone()],
        };

        let bytes = encode_frame_packet(&packet);
        let mut worker_images = HashMap::new();
        worker_images.insert(99, ImageTexture::new(99, 1, 1, vec![0; 4]));

        let decoded = decode_frame_packet(&bytes, &mut worker_images).expect("decode packet");
        assert_eq!(decoded.width, 800);
        assert_eq!(decoded.height, 600);
        assert_eq!(decoded.picture_commands, vec![cmds]);
        assert!(!worker_images.contains_key(&99));
        assert!(worker_images.contains_key(&42));

        let mut worker_alpha = GlyphAtlas::new(GlyphAtlasType::AlphaBitmap, 16, 16);
        apply_atlas_delta(
            &mut worker_alpha,
            decoded.alpha_delta.as_ref().expect("alpha delta"),
        );
        assert_eq!(worker_alpha.size, ISize::new(64, 64));
        assert_eq!(worker_alpha.generation, 2);
        assert_eq!(&worker_alpha.pixels[128..192], &[0x7Fu8; 64]);

        // Incremental row update on frame 2 transfers only the dirty row span.
        main_alpha.pixels[320..384].fill(0xAA);
        main_alpha.generation = 3;
        main_alpha.row_generations[5] = 3;
        let inc_delta = build_atlas_delta(&main_alpha, &mut alpha_state).expect("inc delta");
        assert!(!inc_delta.full_reset);
        assert_eq!((inc_delta.y_min, inc_delta.y_max), (5, 6));
        assert_eq!(inc_delta.row_pixels.len(), 64);

        apply_atlas_delta(&mut worker_alpha, &inc_delta);
        assert_eq!(worker_alpha.generation, 3);
        assert_eq!(&worker_alpha.pixels[128..192], &[0x7Fu8; 64]);
        assert_eq!(&worker_alpha.pixels[320..384], &[0xAAu8; 64]);
        assert_eq!(worker_alpha.dirty_row_range_since(2), Some((5, 6)));
    }
}
