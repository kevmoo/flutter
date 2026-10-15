#![allow(unsafe_code)]
#![allow(
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::not_unsafe_ptr_arg_deref
)]

//! Impeller Skwasm C-ABI FFI bridge (`impeller_skwasm_ffi`):
//! Drop-in Rust + WebGPU/WebGL2 implementation of Flutter Web's `skwasm` C-ABI
//! (`engine/src/flutter/lib/web_ui/skwasm/`), enabling `main.dart.wasm` (`dart2wasm`)
//! to record display lists, shape paragraphs, and render 2x SSAA frames on `impeller_wgpu`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use impeller_canvas::{
    BlendMode, Canvas, Cap, ClipOperation, ColorFilter, ColorSource, ConicalGradientParams,
    DisplayList, DisplayListBuilder, DrawStyle, GradientStop, ImageFilter, ImageTexture, Join,
    LinearGradientParams, Paint, RadialGradientParams, RoundingRadii, SweepGradientParams,
    TileMode, VertexMode,
};
use impeller_geometry::{Color, Matrix, Point, Radians, Rect, Scalar, Size};
use impeller_path::{FillType, Path, PathBuilder};
use impeller_typographer::{
    FontFeature, FontVariation, Paragraph, ParagraphBuilder, ParagraphLineMetrics, ParagraphStyle,
    PlaceholderSpan, RichTextStyle, StrutStyle, TextAlign, TextBox, TextDirection, TextShadow,
    TypographerContext,
};
use wasm_bindgen::prelude::*;

const SCRATCH_STACK_BYTES: usize = 2 * 1024 * 1024;
const RECORDING_CANVAS_DIM: u32 = 16_384;

static NEXT_IMAGE_ID: AtomicU64 = AtomicU64::new(1);

fn decode_argb(argb: u32) -> Color {
    let a = ((argb >> 24) & 0xFF) as u8;
    let r = ((argb >> 16) & 0xFF) as u8;
    let g = ((argb >> 8) & 0xFF) as u8;
    let b = (argb & 0xFF) as u8;
    Color::make_rgba8(r, g, b, a)
}

fn decode_blend_mode(mode: i32) -> BlendMode {
    BlendMode::ALL
        .get(mode.max(0) as usize)
        .copied()
        .unwrap_or(BlendMode::SrcOver)
}

fn decode_tile_mode(mode: i32) -> TileMode {
    match mode {
        1 => TileMode::Repeat,
        2 => TileMode::Mirror,
        3 => TileMode::Decal,
        _ => TileMode::Clamp,
    }
}

fn decode_cap(cap: i32) -> Cap {
    match cap {
        1 => Cap::Round,
        2 => Cap::Square,
        _ => Cap::Butt,
    }
}

fn decode_join(join: i32) -> Join {
    match join {
        1 => Join::Round,
        2 => Join::Bevel,
        _ => Join::Miter,
    }
}

fn decode_fill_type(fill_type: i32) -> FillType {
    match fill_type {
        1 | 3 => FillType::Odd,
        _ => FillType::NonZero,
    }
}

unsafe fn read_rect_ptr(ptr: *const f32) -> Option<Rect> {
    if ptr.is_null() {
        return None;
    }
    let slice = std::slice::from_raw_parts(ptr, 4);
    Some(Rect::from_ltrb(slice[0], slice[1], slice[2], slice[3]))
}

unsafe fn read_rrect_ptr(ptr: *const f32) -> Option<(Rect, RoundingRadii)> {
    if ptr.is_null() {
        return None;
    }
    let s = std::slice::from_raw_parts(ptr, 12);
    let rect = Rect::from_ltrb(s[0], s[1], s[2], s[3]);
    let radii = RoundingRadii::from_radii(
        Size::new(s[4].max(0.0), s[5].max(0.0)),
        Size::new(s[6].max(0.0), s[7].max(0.0)),
        Size::new(s[10].max(0.0), s[11].max(0.0)),
        Size::new(s[8].max(0.0), s[9].max(0.0)),
    );
    Some((rect, radii))
}

unsafe fn read_matrix33_ptr(ptr: *const f32) -> Matrix {
    if ptr.is_null() {
        return Matrix::IDENTITY;
    }
    let s = std::slice::from_raw_parts(ptr, 9);
    // SkMatrix: [scaleX, skewX, transX, skewY, scaleY, transY, persp0, persp1, persp2]
    Matrix {
        m: [
            s[0], s[3], 0.0, s[6], // col 0
            s[1], s[4], 0.0, s[7], // col 1
            0.0, 0.0, 1.0, 0.0, // col 2
            s[2], s[5], 0.0, s[8], // col 3
        ],
    }
}

unsafe fn read_matrix44_ptr(ptr: *const f32) -> Matrix {
    if ptr.is_null() {
        return Matrix::IDENTITY;
    }
    let s = std::slice::from_raw_parts(ptr, 16);
    let mut m = [0.0f32; 16];
    m.copy_from_slice(s);
    Matrix { m }
}

fn path_contains_point(path: &Path, pt: Point) -> bool {
    let Some(bbox) = path.get_bounding_box() else {
        return false;
    };
    if !bbox.contains(pt) {
        return false;
    }
    let polyline = path.create_polyline(1.0);
    let mut winding = 0i32;
    for contour in &polyline.contours {
        if contour.end_index <= contour.start_index + 1 {
            continue;
        }
        let pts = &polyline.points[contour.start_index..contour.end_index];
        for i in 0..pts.len() {
            let p0 = pts[i];
            let p1 = pts[(i + 1) % pts.len()];
            if p0.y <= pt.y {
                if p1.y > pt.y {
                    let cross = (p1.x - p0.x) * (pt.y - p0.y) - (pt.x - p0.x) * (p1.y - p0.y);
                    if cross > 0.0 {
                        winding += 1;
                    }
                }
            } else if p1.y <= pt.y {
                let cross = (p1.x - p0.x) * (pt.y - p0.y) - (pt.x - p0.x) * (p1.y - p0.y);
                if cross < 0.0 {
                    winding -= 1;
                }
            }
        }
    }
    match path.get_fill_type() {
        FillType::NonZero => winding != 0,
        FillType::Odd => (winding & 1) != 0,
    }
}

struct SkDataEntry {
    buf: Vec<u8>,
    len: usize,
}

struct SkStringEntry {
    buf: Vec<u8>,
    len: usize,
}

struct SkString16Entry {
    buf: Vec<u16>,
    len: usize,
}

#[derive(Clone)]
struct PathEntry {
    builder: PathBuilder,
    fill_type: FillType,
    current_point: Point,
    cached_path: RefCell<Option<Path>>,
}

impl PathEntry {
    fn new() -> Self {
        Self {
            builder: PathBuilder::new(),
            fill_type: FillType::NonZero,
            current_point: Point::ZERO,
            cached_path: RefCell::new(None),
        }
    }

    fn mark_dirty(&mut self) {
        *self.cached_path.borrow_mut() = None;
    }

    fn build_path(&self) -> Path {
        if let Some(cached) = self.cached_path.borrow().as_ref() {
            return cached.clone();
        }
        let built = self.builder.clone().build(self.fill_type);
        *self.cached_path.borrow_mut() = Some(built.clone());
        built
    }
}

#[derive(Clone)]
struct VerticesEntry {
    mode: VertexMode,
    positions: Vec<Point>,
    tex_coords: Option<Vec<Point>>,
    colors: Option<Vec<Color>>,
    indices: Vec<u16>,
}

struct RecordingCanvasEntry {
    canvas: Canvas,
    dl_builder: DisplayListBuilder,
    cull_rect: Rect,
}

struct PictureRecorderEntry {
    active_canvas_handle: Option<u32>,
    cull_rect: Rect,
}

#[derive(Clone)]
struct PictureEntry {
    display_list: Arc<DisplayList>,
    cull_rect: Rect,
    approx_bytes: usize,
}

#[cfg(target_arch = "wasm32")]
enum PendingSurfaceTask {
    SetSize {
        width: u32,
        height: u32,
        callback_id: u32,
    },
    RenderPictures {
        pic_handles: Vec<u32>,
        callback_id: u32,
    },
}

#[cfg(target_arch = "wasm32")]
struct GpuSurfaceState {
    renderer: impeller_wgpu::WgpuRenderer,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    offscreen_canvas: web_sys::OffscreenCanvas,
}

struct SurfaceEntry {
    width: u32,
    height: u32,
    callback_pointer: u32,
    resource_cache_limit: usize,
    canvas: Canvas,
    #[cfg(target_arch = "wasm32")]
    initializing: bool,
    #[cfg(target_arch = "wasm32")]
    pending_tasks: Vec<PendingSurfaceTask>,
    #[cfg(target_arch = "wasm32")]
    gpu: Option<GpuSurfaceState>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Copy, Clone, Default)]
struct FrameTimingSample {
    total_ms: f64,
    atlas_sync_ms: f64,
    dl_replay_ms: f64,
    wgpu_render_ms: f64,
    bitmap_transfer_ms: f64,
    command_count: u32,
    replay_stats: impeller_canvas::CanvasReplayStats,
}

#[cfg(target_arch = "wasm32")]
struct RasterTelemetryRing {
    samples: [FrameTimingSample; 60],
    cursor: usize,
    len: usize,
    total_frames: u64,
}

#[cfg(target_arch = "wasm32")]
impl RasterTelemetryRing {
    fn new() -> Self {
        Self {
            samples: [FrameTimingSample::default(); 60],
            cursor: 0,
            len: 0,
            total_frames: 0,
        }
    }

    fn record_and_publish(&mut self, sample: FrameTimingSample) {
        self.samples[self.cursor] = sample;
        self.cursor = (self.cursor + 1) % 60;
        if self.len < 60 {
            self.len += 1;
        }
        self.total_frames += 1;

        let inv_n = 1.0 / (self.len.max(1) as f64);
        let mut sum_total = 0.0;
        let mut sum_dl = 0.0;
        let mut sum_wgpu = 0.0;
        let mut sum_bmp = 0.0;
        let mut sum_atlas = 0.0;
        for s in &self.samples[..self.len] {
            sum_total += s.total_ms;
            sum_dl += s.dl_replay_ms;
            sum_wgpu += s.wgpu_render_ms;
            sum_bmp += s.bitmap_transfer_ms;
            sum_atlas += s.atlas_sync_ms;
        }

        let telemetry_obj = js_sys::Object::new();
        let set_f64 = |key: &str, val: f64| {
            let _ = js_sys::Reflect::set(
                &telemetry_obj,
                &JsValue::from_str(key),
                &JsValue::from_f64(val),
            );
        };
        let set_u32 = |key: &str, val: u32| {
            let _ =
                js_sys::Reflect::set(&telemetry_obj, &JsValue::from_str(key), &JsValue::from(val));
        };
        set_f64("avgTotalMs", sum_total * inv_n);
        set_f64("avgDlReplayMs", sum_dl * inv_n);
        set_f64("avgWgpuRenderMs", sum_wgpu * inv_n);
        set_f64("avgBitmapTransferMs", sum_bmp * inv_n);
        set_f64("avgAtlasSyncMs", sum_atlas * inv_n);
        set_f64("lastTotalMs", sample.total_ms);
        set_f64("lastDlReplayMs", sample.dl_replay_ms);
        set_f64("lastWgpuRenderMs", sample.wgpu_render_ms);
        set_f64("lastBitmapTransferMs", sample.bitmap_transfer_ms);
        set_f64("lastAtlasSyncMs", sample.atlas_sync_ms);
        set_u32("lastCommandCount", sample.command_count);
        set_u32("lastOpCount", sample.replay_stats.op_count);
        set_u32("lastRRectCount", sample.replay_stats.rrect_count);
        set_u32("lastDRRectCount", sample.replay_stats.drrect_count);
        set_u32("lastPathCount", sample.replay_stats.path_count);
        set_u32("lastParagraphCount", sample.replay_stats.paragraph_count);
        set_u32("lastPictureCount", sample.replay_stats.picture_count);
        set_u32("lastSaveLayerCount", sample.replay_stats.save_layer_count);
        set_u32("pathFill", sample.replay_stats.path_fill_count);
        set_u32("pathStroke", sample.replay_stats.path_stroke_count);
        set_u32("pathConvexFast", sample.replay_stats.path_convex_fast_count);
        set_u32("pathPslg", sample.replay_stats.path_pslg_count);
        set_u32("culled", sample.replay_stats.culled_count);

        let last_stats_obj = js_sys::Object::new();
        let set_stat_u32 = |key: &str, val: u32| {
            let _ = js_sys::Reflect::set(
                &last_stats_obj,
                &JsValue::from_str(key),
                &JsValue::from(val),
            );
        };
        set_stat_u32("opCount", sample.replay_stats.op_count);
        set_stat_u32("rrectCount", sample.replay_stats.rrect_count);
        set_stat_u32("drrectCount", sample.replay_stats.drrect_count);
        set_stat_u32("pathCount", sample.replay_stats.path_count);
        set_stat_u32("paragraphCount", sample.replay_stats.paragraph_count);
        set_stat_u32("pictureCount", sample.replay_stats.picture_count);
        set_stat_u32("saveLayerCount", sample.replay_stats.save_layer_count);
        set_stat_u32("pathFill", sample.replay_stats.path_fill_count);
        set_stat_u32("pathStroke", sample.replay_stats.path_stroke_count);
        set_stat_u32("pathConvexFast", sample.replay_stats.path_convex_fast_count);
        set_stat_u32("pathPslg", sample.replay_stats.path_pslg_count);
        set_stat_u32("culled", sample.replay_stats.culled_count);
        let _ = js_sys::Reflect::set(
            &telemetry_obj,
            &JsValue::from_str("lastStats"),
            &last_stats_obj,
        );

        let _ = js_sys::Reflect::set(
            &telemetry_obj,
            &JsValue::from_str("frames"),
            &JsValue::from_f64(self.total_frames as f64),
        );
        let _ = js_sys::Reflect::set(
            &js_sys::global(),
            &JsValue::from_str("__impeller_rs_telemetry"),
            &telemetry_obj,
        );
    }
}

struct SkwasmState {
    next_handle: u32,
    next_callback_id: u32,
    scratch_stack: Box<[u8]>,
    scratch_stack_offset: usize,
    typographer: TypographerContext,
    cached_alpha_atlas: Option<(u64, u64, Arc<impeller_typographer::GlyphAtlas>)>,
    cached_color_atlas: Option<(u64, u64, Arc<impeller_typographer::GlyphAtlas>)>,
    sk_data: HashMap<u32, SkDataEntry>,
    sk_strings: HashMap<u32, SkStringEntry>,
    sk_strings_16: HashMap<u32, SkString16Entry>,
    unicode_pos_buffers: HashMap<u32, Vec<u32>>,
    line_break_buffers: HashMap<u32, Vec<i32>>,
    paths: HashMap<u32, PathEntry>,
    paints: HashMap<u32, Paint>,
    shaders: HashMap<u32, ColorSource>,
    mask_filters: HashMap<u32, f32>,
    color_filters: HashMap<u32, ColorFilter>,
    image_filters: HashMap<u32, ImageFilter>,
    images: HashMap<u32, ImageTexture>,
    vertices: HashMap<u32, VerticesEntry>,
    runtime_effects: HashMap<u32, String>,
    uniform_data: HashMap<u32, Vec<u8>>,
    font_collections: HashMap<u32, ()>,
    typefaces: HashMap<u32, Vec<u8>>,
    text_styles: HashMap<u32, RichTextStyle>,
    paragraph_styles: HashMap<u32, ParagraphStyle>,
    strut_styles: HashMap<u32, StrutStyle>,
    paragraph_builders: HashMap<u32, (ParagraphBuilder, Vec<u8>)>,
    paragraphs: HashMap<u32, Arc<Paragraph>>,
    text_box_lists: HashMap<u32, Vec<TextBox>>,
    line_metrics: HashMap<u32, ParagraphLineMetrics>,
    recorders: HashMap<u32, PictureRecorderEntry>,
    canvases: HashMap<u32, RecordingCanvasEntry>,
    pictures: HashMap<u32, PictureEntry>,
    surfaces: HashMap<u32, SurfaceEntry>,
    #[cfg(target_arch = "wasm32")]
    js_callbacks: HashMap<u32, js_sys::Function>,
    #[cfg(target_arch = "wasm32")]
    telemetry: RasterTelemetryRing,
}

impl SkwasmState {
    fn new() -> Self {
        Self {
            next_handle: 1,
            next_callback_id: 1,
            scratch_stack: vec![0u8; SCRATCH_STACK_BYTES].into_boxed_slice(),
            scratch_stack_offset: 0,
            typographer: TypographerContext::new(),
            cached_alpha_atlas: None,
            cached_color_atlas: None,
            sk_data: HashMap::new(),
            sk_strings: HashMap::new(),
            sk_strings_16: HashMap::new(),
            unicode_pos_buffers: HashMap::new(),
            line_break_buffers: HashMap::new(),
            paths: HashMap::new(),
            paints: HashMap::new(),
            shaders: HashMap::new(),
            mask_filters: HashMap::new(),
            color_filters: HashMap::new(),
            image_filters: HashMap::new(),
            images: HashMap::new(),
            vertices: HashMap::new(),
            runtime_effects: HashMap::new(),
            uniform_data: HashMap::new(),
            font_collections: HashMap::new(),
            typefaces: HashMap::new(),
            text_styles: HashMap::new(),
            paragraph_styles: HashMap::new(),
            strut_styles: HashMap::new(),
            paragraph_builders: HashMap::new(),
            paragraphs: HashMap::new(),
            text_box_lists: HashMap::new(),
            line_metrics: HashMap::new(),
            recorders: HashMap::new(),
            canvases: HashMap::new(),
            pictures: HashMap::new(),
            surfaces: HashMap::new(),
            #[cfg(target_arch = "wasm32")]
            js_callbacks: HashMap::new(),
            #[cfg(target_arch = "wasm32")]
            telemetry: RasterTelemetryRing::new(),
        }
    }

    fn alloc_handle(&mut self) -> u32 {
        let id = self.next_handle;
        self.next_handle = self.next_handle.wrapping_add(1).max(1);
        id
    }

    fn alloc_callback_id(&mut self) -> u32 {
        let id = self.next_callback_id;
        self.next_callback_id = self.next_callback_id.wrapping_add(1).max(1);
        id
    }

    fn read_sk_string(&self, handle: u32) -> Option<String> {
        if let Some(entry) = self.sk_strings.get(&handle) {
            return Some(String::from_utf8_lossy(&entry.buf[..entry.len]).into_owned());
        }
        if let Some(entry16) = self.sk_strings_16.get(&handle) {
            return Some(String::from_utf16_lossy(&entry16.buf[..entry16.len]));
        }
        None
    }

    fn shared_glyph_atlases(
        &mut self,
    ) -> (
        Arc<impeller_typographer::GlyphAtlas>,
        Arc<impeller_typographer::GlyphAtlas>,
    ) {
        let alpha_gen = self.typographer.alpha_atlas.generation;
        let alpha_epoch = self.typographer.alpha_atlas.resize_epoch;
        let alpha_arc = match &self.cached_alpha_atlas {
            Some((g, e, arc)) if *g == alpha_gen && *e == alpha_epoch => Arc::clone(arc),
            _ => {
                let arc = Arc::new(self.typographer.alpha_atlas.descriptor_without_pixels());
                self.cached_alpha_atlas = Some((alpha_gen, alpha_epoch, Arc::clone(&arc)));
                arc
            }
        };

        let color_gen = self.typographer.color_atlas.generation;
        let color_epoch = self.typographer.color_atlas.resize_epoch;
        let color_arc = match &self.cached_color_atlas {
            Some((g, e, arc)) if *g == color_gen && *e == color_epoch => Arc::clone(arc),
            _ => {
                let arc = Arc::new(self.typographer.color_atlas.descriptor_without_pixels());
                self.cached_color_atlas = Some((color_gen, color_epoch, Arc::clone(&arc)));
                arc
            }
        };

        (alpha_arc, color_arc)
    }
}

thread_local! {
    static STATE: RefCell<SkwasmState> = RefCell::new(SkwasmState::new());
}

fn with_state<R>(f: impl FnOnce(&mut SkwasmState) -> R) -> R {
    STATE.with(|cell| f(&mut cell.borrow_mut()))
}

#[cfg(target_arch = "wasm32")]
fn now_milliseconds() -> f64 {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("performance"))
        .ok()
        .and_then(|perf| perf.dyn_into::<web_sys::Performance>().ok())
        .map_or(0.0, |perf| perf.now())
}

#[cfg(target_arch = "wasm32")]
fn trace_performance() -> Option<web_sys::Performance> {
    let global = js_sys::global();
    let enabled = js_sys::Reflect::get(&global, &JsValue::from_str("__impeller_rs_trace"))
        .ok()
        .is_some_and(|v| v == JsValue::TRUE);
    if !enabled {
        return None;
    }
    js_sys::Reflect::get(&global, &JsValue::from_str("performance"))
        .ok()
        .and_then(|perf| perf.dyn_into::<web_sys::Performance>().ok())
}

#[cfg(target_arch = "wasm32")]
fn schedule_surface_callback(
    surface_handle: u32,
    callback_id: u32,
    context_val: u32,
    js_context: JsValue,
) {
    wasm_bindgen_futures::spawn_local(async move {
        let _ =
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL)).await;
        let cb_fn = with_state(|st| {
            let cb_ptr = st.surfaces.get(&surface_handle)?.callback_pointer;
            st.js_callbacks.get(&cb_ptr).cloned()
        });
        if let Some(cb) = cb_fn {
            let _ = cb.call3(
                &JsValue::NULL,
                &JsValue::from(callback_id),
                &JsValue::from(context_val),
                &js_context,
            );
        }
    });
}

// =============================================================================
// JS Interop Shims (`impeller_skwasm_loader.js` -> `impeller_skwasm_ffi`)
// =============================================================================

#[wasm_bindgen]
pub fn skwasm_register_callback_js(callback: JsValue) -> u32 {
    #[cfg(target_arch = "wasm32")]
    {
        with_state(|st| {
            let id = st.alloc_handle();
            if let Ok(func) = callback.dyn_into::<js_sys::Function>() {
                st.js_callbacks.insert(id, func);
            }
            id
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = callback;
        with_state(|st| st.alloc_handle())
    }
}

#[cfg(target_arch = "wasm32")]
fn invoke_surface_callback_now(
    surface_handle: u32,
    callback_id: u32,
    context_val: u32,
    js_context: &JsValue,
) {
    let cb_fn = with_state(|st| {
        let cb_ptr = st.surfaces.get(&surface_handle)?.callback_pointer;
        st.js_callbacks.get(&cb_ptr).cloned()
    });
    if let Some(cb) = cb_fn {
        let _ = cb.call3(
            &JsValue::NULL,
            &JsValue::from(callback_id),
            &JsValue::from(context_val),
            js_context,
        );
    }
}

#[cfg(target_arch = "wasm32")]
fn commands_need_atlas_sync(commands: &[impeller_wgpu::DrawCommand]) -> bool {
    commands.iter().any(|cmd| match cmd {
        impeller_wgpu::DrawCommand::GlyphAtlas(b) => {
            b.alpha_atlas.is_none() && b.color_atlas.is_none() && !b.quads.is_empty()
        }
        impeller_wgpu::DrawCommand::SaveLayer { commands: sub, .. } => {
            commands_need_atlas_sync(sub)
        }
        _ => false,
    })
}

#[cfg(target_arch = "wasm32")]
fn render_pictures_now(surface_handle: u32, pic_handles: &[u32]) -> JsValue {
    let start_ms = now_milliseconds();
    let trace_perf = trace_performance();
    let bitmaps = js_sys::Array::new();

    with_state(|st| {
        let mut pictures_to_render = Vec::with_capacity(pic_handles.len());
        for pid in pic_handles {
            if let Some(pic) = st.pictures.get(pid) {
                pictures_to_render.push(Arc::clone(&pic.display_list));
            }
        }
        let (alpha_arc, color_arc) = st.shared_glyph_atlases();
        let typographer = &st.typographer;

        let mut atlas_sync_ms = 0.0;
        let mut dl_replay_ms = 0.0;
        let mut wgpu_render_ms = 0.0;
        let mut bitmap_transfer_ms = 0.0;
        let mut last_cmd_count = 0u32;
        let mut last_replay_stats = impeller_canvas::CanvasReplayStats::default();

        if let Some(surf) = st.surfaces.get_mut(&surface_handle) {
            let w = surf.width.max(1);
            let h = surf.height.max(1);
            if let Some(gpu) = &mut surf.gpu {
                if gpu.config.width != w || gpu.config.height != h {
                    gpu.offscreen_canvas.set_width(w);
                    gpu.offscreen_canvas.set_height(h);
                    gpu.config.width = w;
                    gpu.config.height = h;
                    gpu.surface.configure(gpu.renderer.device(), &gpu.config);
                }
                let t_atlas_start = now_milliseconds();
                let _ = gpu.renderer.sync_glyph_atlas(&typographer.alpha_atlas);
                let _ = gpu.renderer.sync_glyph_atlas(&typographer.color_atlas);
                atlas_sync_ms = now_milliseconds() - t_atlas_start;

                surf.canvas.set_default_glyph_atlases(
                    Some(Arc::clone(&alpha_arc)),
                    Some(Arc::clone(&color_arc)),
                );

                for dl in &pictures_to_render {
                    if let Some(perf) = &trace_perf {
                        let _ = perf.mark("impeller:dl_replay:start");
                    }
                    let t_replay_start = now_milliseconds();
                    surf.canvas.reset(w, h, Color::TRANSPARENT);
                    surf.canvas.draw_display_list(dl);
                    if commands_need_atlas_sync(surf.canvas.commands()) {
                        surf.canvas
                            .sync_glyph_atlases_with_arcs(&alpha_arc, &color_arc);
                    }
                    dl_replay_ms += now_milliseconds() - t_replay_start;
                    if let Some(perf) = &trace_perf {
                        let _ = perf.mark("impeller:dl_replay:end");
                        let _ = perf.measure_with_start_mark_and_end_mark(
                            "impeller:dl_replay",
                            "impeller:dl_replay:start",
                            "impeller:dl_replay:end",
                        );
                    }
                    last_cmd_count = surf.canvas.commands().len() as u32;
                    last_replay_stats = surf.canvas.replay_stats();

                    if let Some(perf) = &trace_perf {
                        let _ = perf.mark("impeller:wgpu_render:start");
                    }
                    let t_wgpu_start = now_milliseconds();
                    let frame_res = match gpu.surface.get_current_texture() {
                        Ok(frame) => Ok(frame),
                        Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                            gpu.surface.configure(gpu.renderer.device(), &gpu.config);
                            gpu.surface.get_current_texture()
                        }
                        Err(e) => Err(e),
                    };

                    if let Ok(frame) = frame_res {
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let _ = gpu.renderer.render_to_surface_view(
                            &view,
                            w,
                            h,
                            gpu.config.format,
                            Color::TRANSPARENT,
                            surf.canvas.commands(),
                        );
                        frame.present();
                        wgpu_render_ms += now_milliseconds() - t_wgpu_start;
                        if let Some(perf) = &trace_perf {
                            let _ = perf.mark("impeller:wgpu_render:end");
                            let _ = perf.measure_with_start_mark_and_end_mark(
                                "impeller:wgpu_render",
                                "impeller:wgpu_render:start",
                                "impeller:wgpu_render:end",
                            );
                            let _ = perf.mark("impeller:bitmap_transfer:start");
                        }

                        let t_bmp_start = now_milliseconds();
                        if let Ok(bmp) = gpu.offscreen_canvas.transfer_to_image_bitmap() {
                            bitmaps.push(&bmp);
                        }
                        bitmap_transfer_ms += now_milliseconds() - t_bmp_start;
                        if let Some(perf) = &trace_perf {
                            let _ = perf.mark("impeller:bitmap_transfer:end");
                            let _ = perf.measure_with_start_mark_and_end_mark(
                                "impeller:bitmap_transfer",
                                "impeller:bitmap_transfer:start",
                                "impeller:bitmap_transfer:end",
                            );
                        }
                    } else {
                        wgpu_render_ms += now_milliseconds() - t_wgpu_start;
                        if let Some(perf) = &trace_perf {
                            let _ = perf.mark("impeller:wgpu_render:end");
                            let _ = perf.measure_with_start_mark_and_end_mark(
                                "impeller:wgpu_render",
                                "impeller:wgpu_render:start",
                                "impeller:wgpu_render:end",
                            );
                        }
                    }
                }
            }
        }

        let total_ms = now_milliseconds() - start_ms;
        st.telemetry.record_and_publish(FrameTimingSample {
            total_ms,
            atlas_sync_ms,
            dl_replay_ms,
            wgpu_render_ms,
            bitmap_transfer_ms,
            command_count: last_cmd_count,
            replay_stats: last_replay_stats,
        });
    });

    let end_ms = now_milliseconds();
    let raster_result = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&raster_result, &JsValue::from_str("imageBitmaps"), &bitmaps);
    let _ = js_sys::Reflect::set(
        &raster_result,
        &JsValue::from_str("rasterStartMilliseconds"),
        &JsValue::from_f64(start_ms),
    );
    let _ = js_sys::Reflect::set(
        &raster_result,
        &JsValue::from_str("rasterEndMilliseconds"),
        &JsValue::from_f64(end_ms),
    );
    raster_result.into()
}

#[wasm_bindgen]
pub fn skwasm_surface_set_canvas_js(surface_handle: u32, canvas: JsValue) -> u32 {
    let (callback_id, context_lost_cb_id) = with_state(|st| {
        #[cfg(target_arch = "wasm32")]
        if let Some(surf) = st.surfaces.get_mut(&surface_handle) {
            surf.initializing = true;
        }
        (st.alloc_callback_id(), st.alloc_callback_id())
    });

    #[cfg(target_arch = "wasm32")]
    {
        let prefer_webgpu = js_sys::Reflect::get(
            &js_sys::global(),
            &JsValue::from_str("__impellerPreferWebGpu"),
        )
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

        wasm_bindgen_futures::spawn_local(async move {
            let _ = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
                .await;
            if let Ok(offscreen) = canvas.dyn_into::<web_sys::OffscreenCanvas>() {
                if let Ok((renderer, surface, mut config)) =
                    impeller_wgpu::WgpuRenderer::new_for_offscreen_canvas_with_backend(
                        offscreen.clone(),
                        prefer_webgpu,
                    )
                    .await
                {
                    with_state(|st| {
                        if let Some(surf_entry) = st.surfaces.get_mut(&surface_handle) {
                            let target_w = if surf_entry.width > 1 {
                                surf_entry.width
                            } else {
                                config.width.max(1)
                            };
                            let target_h = if surf_entry.height > 1 {
                                surf_entry.height
                            } else {
                                config.height.max(1)
                            };
                            surf_entry.width = target_w;
                            surf_entry.height = target_h;
                            if config.width != target_w || config.height != target_h {
                                offscreen.set_width(target_w);
                                offscreen.set_height(target_h);
                                config.width = target_w;
                                config.height = target_h;
                                surface.configure(renderer.device(), &config);
                            }
                            surf_entry.gpu = Some(GpuSurfaceState {
                                renderer,
                                surface,
                                config,
                                offscreen_canvas: offscreen,
                            });
                        }
                    });
                }
            }
            let pending = with_state(|st| {
                if let Some(surf_entry) = st.surfaces.get_mut(&surface_handle) {
                    surf_entry.initializing = false;
                    std::mem::take(&mut surf_entry.pending_tasks)
                } else {
                    Vec::new()
                }
            });

            invoke_surface_callback_now(
                surface_handle,
                callback_id,
                context_lost_cb_id,
                &JsValue::NULL,
            );

            for task in pending {
                match task {
                    PendingSurfaceTask::SetSize {
                        width,
                        height,
                        callback_id: sz_cb,
                    } => {
                        with_state(|st| {
                            if let Some(surf) = st.surfaces.get_mut(&surface_handle) {
                                surf.width = width;
                                surf.height = height;
                                if let Some(gpu) = &mut surf.gpu {
                                    gpu.offscreen_canvas.set_width(width);
                                    gpu.offscreen_canvas.set_height(height);
                                    gpu.config.width = width;
                                    gpu.config.height = height;
                                    gpu.surface.configure(gpu.renderer.device(), &gpu.config);
                                }
                            }
                        });
                        invoke_surface_callback_now(surface_handle, sz_cb, 0, &JsValue::NULL);
                    }
                    PendingSurfaceTask::RenderPictures {
                        pic_handles,
                        callback_id: rp_cb,
                    } => {
                        let raster_res = render_pictures_now(surface_handle, &pic_handles);
                        invoke_surface_callback_now(surface_handle, rp_cb, 0, &raster_res);
                    }
                }
            }
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (surface_handle, canvas, context_lost_cb_id);
    }

    callback_id
}

// =============================================================================
// 1. Emscripten Stack & Heap Shim + Skwasm Module Info
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn emscripten_stack_get_current() -> usize {
    with_state(|st| st.scratch_stack.as_ptr() as usize + st.scratch_stack_offset)
}

#[no_mangle]
pub unsafe extern "C" fn _emscripten_stack_restore(ptr: usize) {
    with_state(|st| {
        let base = st.scratch_stack.as_ptr() as usize;
        if ptr >= base && ptr <= base + st.scratch_stack.len() {
            st.scratch_stack_offset = ptr - base;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn _emscripten_stack_alloc(size: usize) -> usize {
    with_state(|st| {
        let aligned = (size.max(1) + 15) & !15;
        if st.scratch_stack_offset + aligned > st.scratch_stack.len() {
            st.scratch_stack_offset = 0;
        }
        let ptr = st.scratch_stack.as_mut_ptr() as usize + st.scratch_stack_offset;
        st.scratch_stack_offset += aligned;
        ptr
    })
}

#[no_mangle]
pub unsafe extern "C" fn emscripten_stack_get_free() -> usize {
    with_state(|st| {
        st.scratch_stack
            .len()
            .saturating_sub(st.scratch_stack_offset)
    })
}

#[no_mangle]
pub unsafe extern "C" fn emscripten_builtin_malloc(size: usize) -> *mut u8 {
    let payload = size.max(1);
    let total = (payload + 16 + 15) & !15;
    let Ok(layout) = std::alloc::Layout::from_size_align(total, 16) else {
        return std::ptr::null_mut();
    };
    let raw = std::alloc::alloc_zeroed(layout);
    if raw.is_null() {
        return std::ptr::null_mut();
    }
    *(raw as *mut usize) = total;
    raw.add(16)
}

#[no_mangle]
pub unsafe extern "C" fn emscripten_builtin_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let raw = ptr.sub(16);
    let total = *(raw as *const usize);
    if let Ok(layout) = std::alloc::Layout::from_size_align(total, 16) {
        std::alloc::dealloc(raw, layout);
    }
}

#[no_mangle]
pub unsafe extern "C" fn skwasm_isHeavy() -> i32 {
    1
}

#[no_mangle]
pub unsafe extern "C" fn skwasm_isMultiThreaded() -> i32 {
    0
}

#[no_mangle]
pub unsafe extern "C" fn skwasm_isWimp() -> i32 {
    1
}

#[no_mangle]
pub unsafe extern "C" fn skwasm_getLiveObjectCounts(out_counts: *mut u32) {
    if out_counts.is_null() {
        return;
    }
    with_state(|st| {
        let slice = std::slice::from_raw_parts_mut(out_counts, 16);
        slice.fill(0);
        slice[0] = st.paths.len() as u32;
        slice[1] = st.paints.len() as u32;
        slice[2] = st.pictures.len() as u32;
        slice[3] = st.images.len() as u32;
        slice[4] = st.paragraphs.len() as u32;
    });
}

// =============================================================================
// 2. SkData, SkString, SkString16, UnicodePositionBuffer, LineBreakBuffer
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn skData_create(size: usize) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.sk_data.insert(
            id,
            SkDataEntry {
                buf: vec![0u8; size.max(1)],
                len: size,
            },
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn skData_getPointer(handle: u32) -> *mut u8 {
    with_state(|st| {
        st.sk_data
            .get_mut(&handle)
            .map_or(std::ptr::null_mut(), |e| e.buf.as_mut_ptr())
    })
}

#[no_mangle]
pub unsafe extern "C" fn skData_getConstPointer(handle: u32) -> *const u8 {
    with_state(|st| {
        st.sk_data
            .get(&handle)
            .map_or(std::ptr::null(), |e| e.buf.as_ptr())
    })
}

#[no_mangle]
pub unsafe extern "C" fn skData_getSize(handle: u32) -> usize {
    with_state(|st| st.sk_data.get(&handle).map_or(0, |e| e.len))
}

#[no_mangle]
pub unsafe extern "C" fn skData_dispose(handle: u32) {
    with_state(|st| {
        st.sk_data.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn skString_allocate(size: usize) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.sk_strings.insert(
            id,
            SkStringEntry {
                buf: vec![0u8; size.max(1)],
                len: size,
            },
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn skString_getData(handle: u32) -> *mut u8 {
    with_state(|st| {
        st.sk_strings
            .get_mut(&handle)
            .map_or(std::ptr::null_mut(), |e| e.buf.as_mut_ptr())
    })
}

#[no_mangle]
pub unsafe extern "C" fn skString_getLength(handle: u32) -> i32 {
    with_state(|st| st.sk_strings.get(&handle).map_or(0, |e| e.len as i32))
}

#[no_mangle]
pub unsafe extern "C" fn skString_free(handle: u32) {
    with_state(|st| {
        st.sk_strings.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn skString16_allocate(size: usize) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.sk_strings_16.insert(
            id,
            SkString16Entry {
                buf: vec![0u16; size.max(1)],
                len: size,
            },
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn skString16_getData(handle: u32) -> *mut u16 {
    with_state(|st| {
        st.sk_strings_16
            .get_mut(&handle)
            .map_or(std::ptr::null_mut(), |e| e.buf.as_mut_ptr())
    })
}

#[no_mangle]
pub unsafe extern "C" fn skString16_free(handle: u32) {
    with_state(|st| {
        st.sk_strings_16.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn unicodePositionBuffer_create(size: usize) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.unicode_pos_buffers.insert(id, vec![0u32; size.max(1)]);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn unicodePositionBuffer_getDataPointer(handle: u32) -> *mut u32 {
    with_state(|st| {
        st.unicode_pos_buffers
            .get_mut(&handle)
            .map_or(std::ptr::null_mut(), |b| b.as_mut_ptr())
    })
}

#[no_mangle]
pub unsafe extern "C" fn unicodePositionBuffer_free(handle: u32) {
    with_state(|st| {
        st.unicode_pos_buffers.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn lineBreakBuffer_create(size: usize) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.line_break_buffers
            .insert(id, vec![0i32; size.max(1) * 2]);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineBreakBuffer_getDataPointer(handle: u32) -> *mut i32 {
    with_state(|st| {
        st.line_break_buffers
            .get_mut(&handle)
            .map_or(std::ptr::null_mut(), |b| b.as_mut_ptr())
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineBreakBuffer_free(handle: u32) {
    with_state(|st| {
        st.line_break_buffers.remove(&handle);
    });
}

// =============================================================================
// 3. Surface Lifecycle & Rendering (`SkwasmSurface`)
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn surface_create() -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.surfaces.insert(
            id,
            SurfaceEntry {
                width: 1,
                height: 1,
                callback_pointer: 0,
                resource_cache_limit: 0,
                canvas: Canvas::new(1, 1, Color::TRANSPARENT),
                #[cfg(target_arch = "wasm32")]
                initializing: false,
                #[cfg(target_arch = "wasm32")]
                pending_tasks: Vec::new(),
                #[cfg(target_arch = "wasm32")]
                gpu: None,
            },
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn surface_destroy(handle: u32) {
    with_state(|st| {
        st.surfaces.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn surface_setCallbackHandler(handle: u32, callback_pointer: u32) {
    with_state(|st| {
        if let Some(surf) = st.surfaces.get_mut(&handle) {
            surf.callback_pointer = callback_pointer;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn surface_setResourceCacheLimitBytes(handle: u32, bytes: i32) {
    with_state(|st| {
        if let Some(surf) = st.surfaces.get_mut(&handle) {
            surf.resource_cache_limit = bytes.max(0) as usize;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn surface_setSize(handle: u32, width: i32, height: i32) -> u32 {
    let w = width.max(1) as u32;
    let h = height.max(1) as u32;
    let callback_id = with_state(|st| {
        let cb_id = st.alloc_callback_id();
        if let Some(surf) = st.surfaces.get_mut(&handle) {
            surf.width = w;
            surf.height = h;
            surf.canvas.reset(w, h, Color::TRANSPARENT);
        }
        cb_id
    });

    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async move {
            let _ = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
                .await;
            let queued = with_state(|st| {
                let Some(surf) = st.surfaces.get_mut(&handle) else {
                    return false;
                };
                if surf.initializing && surf.gpu.is_none() {
                    surf.pending_tasks.push(PendingSurfaceTask::SetSize {
                        width: w,
                        height: h,
                        callback_id,
                    });
                    return true;
                }
                if let Some(gpu) = &mut surf.gpu {
                    gpu.offscreen_canvas.set_width(w);
                    gpu.offscreen_canvas.set_height(h);
                    gpu.config.width = w;
                    gpu.config.height = h;
                    gpu.surface.configure(gpu.renderer.device(), &gpu.config);
                }
                false
            });
            if !queued {
                invoke_surface_callback_now(handle, callback_id, 0, &JsValue::NULL);
            }
        });
    }

    callback_id
}

#[no_mangle]
pub unsafe extern "C" fn surface_renderPictures(
    handle: u32,
    pictures_ptr: *const u32,
    count: i32,
) -> u32 {
    let pic_handles: Vec<u32> = if !pictures_ptr.is_null() && count > 0 {
        std::slice::from_raw_parts(pictures_ptr, count as usize).to_vec()
    } else {
        Vec::new()
    };

    let callback_id = with_state(|st| st.alloc_callback_id());

    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async move {
            let _ = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
                .await;
            let queued = with_state(|st| {
                let Some(surf) = st.surfaces.get_mut(&handle) else {
                    return false;
                };
                if surf.initializing && surf.gpu.is_none() {
                    surf.pending_tasks.push(PendingSurfaceTask::RenderPictures {
                        pic_handles: pic_handles.clone(),
                        callback_id,
                    });
                    return true;
                }
                false
            });
            if !queued {
                let raster_result = render_pictures_now(handle, &pic_handles);
                invoke_surface_callback_now(handle, callback_id, 0, &raster_result);
            }
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (handle, pic_handles);
    }

    callback_id
}

#[no_mangle]
pub unsafe extern "C" fn surface_renderPicture(handle: u32, picture: u32) -> u32 {
    surface_renderPictures(handle, &picture as *const u32, 1)
}

#[no_mangle]
pub unsafe extern "C" fn surface_rasterizeImage(
    handle: u32,
    image_handle: u32,
    _format: i32,
) -> u32 {
    let (callback_id, data_handle) = with_state(|st| {
        let cb_id = st.alloc_callback_id();
        let bytes = st
            .images
            .get(&image_handle)
            .map_or_else(|| vec![0u8; 4], |img| img.rgba_premultiplied.to_vec());
        let len = bytes.len();
        let data_id = st.alloc_handle();
        st.sk_data.insert(data_id, SkDataEntry { buf: bytes, len });
        (cb_id, data_id)
    });

    #[cfg(target_arch = "wasm32")]
    schedule_surface_callback(handle, callback_id, data_handle, JsValue::NULL);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (handle, data_handle);

    callback_id
}

#[no_mangle]
pub unsafe extern "C" fn surface_getGlContext(_handle: u32) -> u32 {
    1
}

#[no_mangle]
pub unsafe extern "C" fn surface_triggerContextLoss(handle: u32) -> u32 {
    let callback_id = with_state(|st| st.alloc_callback_id());
    #[cfg(target_arch = "wasm32")]
    schedule_surface_callback(handle, callback_id, 0, JsValue::NULL);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = handle;
    callback_id
}

// =============================================================================
// 4. Path Construction & Queries (`SkwasmPath`)
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn path_create() -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.paths.insert(id, PathEntry::new());
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn path_dispose(handle: u32) {
    with_state(|st| {
        st.paths.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_copy(handle: u32) -> u32 {
    with_state(|st| {
        let cloned = st
            .paths
            .get(&handle)
            .cloned()
            .unwrap_or_else(PathEntry::new);
        let id = st.alloc_handle();
        st.paths.insert(id, cloned);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn path_setFillType(handle: u32, fill_type: i32) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            p.fill_type = decode_fill_type(fill_type);
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_getFillType(handle: u32) -> i32 {
    with_state(|st| {
        st.paths.get(&handle).map_or(0, |p| match p.fill_type {
            FillType::NonZero => 0,
            FillType::Odd => 1,
        })
    })
}

#[no_mangle]
pub unsafe extern "C" fn path_moveTo(handle: u32, x: f32, y: f32) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            let pt = Point::new(x, y);
            p.builder.move_to(pt);
            p.current_point = pt;
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_relativeMoveTo(handle: u32, dx: f32, dy: f32) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            let pt = Point::new(p.current_point.x + dx, p.current_point.y + dy);
            p.builder.move_to(pt);
            p.current_point = pt;
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_lineTo(handle: u32, x: f32, y: f32) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            let pt = Point::new(x, y);
            p.builder.line_to(pt);
            p.current_point = pt;
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_relativeLineTo(handle: u32, dx: f32, dy: f32) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            let pt = Point::new(p.current_point.x + dx, p.current_point.y + dy);
            p.builder.line_to(pt);
            p.current_point = pt;
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_quadraticBezierTo(handle: u32, x1: f32, y1: f32, x2: f32, y2: f32) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            let cp = Point::new(x1, y1);
            let end = Point::new(x2, y2);
            p.builder.quadratic_curve_to(cp, end);
            p.current_point = end;
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_relativeQuadraticBezierTo(
    handle: u32,
    dx1: f32,
    dy1: f32,
    dx2: f32,
    dy2: f32,
) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            let cur = p.current_point;
            let cp = Point::new(cur.x + dx1, cur.y + dy1);
            let end = Point::new(cur.x + dx2, cur.y + dy2);
            p.builder.quadratic_curve_to(cp, end);
            p.current_point = end;
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_cubicTo(
    handle: u32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            let cp1 = Point::new(x1, y1);
            let cp2 = Point::new(x2, y2);
            let end = Point::new(x3, y3);
            p.builder.cubic_curve_to(cp1, cp2, end);
            p.current_point = end;
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_relativeCubicTo(
    handle: u32,
    dx1: f32,
    dy1: f32,
    dx2: f32,
    dy2: f32,
    dx3: f32,
    dy3: f32,
) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            let cur = p.current_point;
            let cp1 = Point::new(cur.x + dx1, cur.y + dy1);
            let cp2 = Point::new(cur.x + dx2, cur.y + dy2);
            let end = Point::new(cur.x + dx3, cur.y + dy3);
            p.builder.cubic_curve_to(cp1, cp2, end);
            p.current_point = end;
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_conicTo(
    handle: u32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    weight: f32,
) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            let p0 = p.current_point;
            let cp = Point::new(x1, y1);
            let end = Point::new(x2, y2);
            let w = weight.clamp(0.01, 100.0);
            let k = (4.0 * w) / (3.0 * (1.0 + w));
            let c1 = Point::new(p0.x + k * (cp.x - p0.x), p0.y + k * (cp.y - p0.y));
            let c2 = Point::new(end.x + k * (cp.x - end.x), end.y + k * (cp.y - end.y));
            p.builder.cubic_curve_to(c1, c2, end);
            p.current_point = end;
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_relativeConicTo(
    handle: u32,
    dx1: f32,
    dy1: f32,
    dx2: f32,
    dy2: f32,
    weight: f32,
) {
    let cur = with_state(|st| {
        st.paths
            .get(&handle)
            .map_or(Point::ZERO, |p| p.current_point)
    });
    path_conicTo(
        handle,
        cur.x + dx1,
        cur.y + dy1,
        cur.x + dx2,
        cur.y + dy2,
        weight,
    );
}

#[no_mangle]
pub unsafe extern "C" fn path_arcToOval(
    handle: u32,
    rect_ptr: *const f32,
    start_angle_deg: f32,
    sweep_angle_deg: f32,
    force_move_to: i32,
) {
    let Some(rect) = read_rect_ptr(rect_ptr) else {
        return;
    };
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            p.builder.arc_to_oval(
                rect,
                Radians(start_angle_deg.to_radians()),
                Radians(sweep_angle_deg.to_radians()),
                force_move_to != 0,
            );
            p.current_point = p.builder.current_point();
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_arcToRotated(
    handle: u32,
    rx: f32,
    ry: f32,
    x_axis_rotate: f32,
    arc_size: i32,
    path_direction: i32,
    x: f32,
    y: f32,
) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            p.builder.svg_arc_to(
                Size::new(rx, ry),
                x_axis_rotate,
                arc_size != 0,
                path_direction == 0,
                Point::new(x, y),
            );
            p.current_point = p.builder.current_point();
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_relativeArcToRotated(
    handle: u32,
    rx: f32,
    ry: f32,
    x_axis_rotate: f32,
    arc_size: i32,
    path_direction: i32,
    dx: f32,
    dy: f32,
) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            let cur = p.builder.current_point();
            p.builder.svg_arc_to(
                Size::new(rx, ry),
                x_axis_rotate,
                arc_size != 0,
                path_direction == 0,
                Point::new(cur.x + dx, cur.y + dy),
            );
            p.current_point = p.builder.current_point();
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_addRect(handle: u32, rect_ptr: *const f32) {
    let Some(rect) = read_rect_ptr(rect_ptr) else {
        return;
    };
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            p.builder.add_rect(rect);
            p.current_point = p.builder.current_point();
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_addOval(handle: u32, rect_ptr: *const f32) {
    let Some(rect) = read_rect_ptr(rect_ptr) else {
        return;
    };
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            p.builder.add_oval(rect);
            p.current_point = p.builder.current_point();
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_addArc(
    handle: u32,
    rect_ptr: *const f32,
    start_angle_deg: f32,
    sweep_angle_deg: f32,
) {
    let Some(rect) = read_rect_ptr(rect_ptr) else {
        return;
    };
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            p.builder.add_arc(
                rect,
                Radians(start_angle_deg.to_radians()),
                Radians(sweep_angle_deg.to_radians()),
                false,
            );
            p.current_point = p.builder.current_point();
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_addPolygon(
    handle: u32,
    points_ptr: *const f32,
    point_count: i32,
    close: i32,
) {
    if points_ptr.is_null() || point_count <= 0 {
        return;
    }
    let floats = std::slice::from_raw_parts(points_ptr, (point_count as usize) * 2);
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            for i in 0..(point_count as usize) {
                let pt = Point::new(floats[i * 2], floats[i * 2 + 1]);
                if i == 0 {
                    p.builder.move_to(pt);
                } else {
                    p.builder.line_to(pt);
                }
                p.current_point = pt;
            }
            if close != 0 {
                p.builder.close();
                p.current_point = p.builder.current_point();
            }
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_addRRect(handle: u32, rrect_ptr: *const f32) {
    let Some((rect, radii)) = read_rrect_ptr(rrect_ptr) else {
        return;
    };
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            p.builder.add_rounded_rect(rect, radii);
            p.current_point = p.builder.current_point();
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_addPath(
    handle: u32,
    other_handle: u32,
    matrix33_ptr: *const f32,
    _extend_path: i32,
) {
    let has_matrix = !matrix33_ptr.is_null();
    let mat = read_matrix33_ptr(matrix33_ptr);
    with_state(|st| {
        let Some(other_entry) = st.paths.get(&other_handle).cloned() else {
            return;
        };
        let other_path = other_entry.build_path();
        if let Some(p) = st.paths.get_mut(&handle) {
            if !has_matrix {
                p.builder.add_path(&other_path);
                p.current_point = other_entry.current_point;
            } else {
                let mut tmp = PathBuilder::new();
                tmp.add_path(&other_path);
                tmp.transform(&mat);
                p.builder.add_path(&tmp.build(other_path.get_fill_type()));
                p.current_point = mat.transform_point(other_entry.current_point);
            }
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_close(handle: u32) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            p.builder.close();
            p.current_point = p.builder.current_point();
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_reset(handle: u32) {
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            p.builder = PathBuilder::new();
            p.current_point = Point::ZERO;
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_contains(handle: u32, x: f32, y: f32) -> i32 {
    with_state(|st| {
        let Some(p) = st.paths.get(&handle) else {
            return 0;
        };
        let path = p.build_path();
        i32::from(path_contains_point(&path, Point::new(x, y)))
    })
}

#[no_mangle]
pub unsafe extern "C" fn path_transform(handle: u32, matrix33_ptr: *const f32) {
    let mat = read_matrix33_ptr(matrix33_ptr);
    with_state(|st| {
        if let Some(p) = st.paths.get_mut(&handle) {
            p.builder.transform(&mat);
            p.current_point = mat.transform_point(p.current_point);
            p.mark_dirty();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn path_getBounds(handle: u32, out_rect_ptr: *mut f32) {
    if out_rect_ptr.is_null() {
        return;
    }
    let bounds = with_state(|st| {
        st.paths
            .get(&handle)
            .and_then(|p| p.build_path().get_bounding_box())
            .unwrap_or(Rect::ZERO)
    });
    let out = std::slice::from_raw_parts_mut(out_rect_ptr, 4);
    out[0] = bounds.get_left();
    out[1] = bounds.get_top();
    out[2] = bounds.get_right();
    out[3] = bounds.get_bottom();
}

#[no_mangle]
pub unsafe extern "C" fn path_combine(op: i32, path1: u32, path2: u32) -> u32 {
    with_state(|st| {
        let Some(p1) = st.paths.get(&path1) else {
            return 0;
        };
        let Some(p2) = st.paths.get(&path2) else {
            return 0;
        };
        let path_a = p1.build_path();
        let path_b = p2.build_path();
        let path_op = impeller_tessellator::PathOp::from_i32(op)
            .unwrap_or(impeller_tessellator::PathOp::Union);
        let combined = impeller_tessellator::combine_paths(&path_a, &path_b, path_op, 1.0);
        let mut builder = PathBuilder::default();
        builder.add_path(&combined);
        let cur_pt = builder.current_point();
        let fill_type = combined.get_fill_type();
        let entry = PathEntry {
            builder,
            fill_type,
            current_point: cur_pt,
            cached_path: RefCell::new(Some(combined)),
        };
        let id = st.alloc_handle();
        st.paths.insert(id, entry);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn path_getSvgString(_handle: u32) -> u32 {
    skString_allocate(0)
}

// =============================================================================
// 5. Paint, Shaders, MaskFilter, ColorFilter, ImageFilter
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn paint_create(
    _is_anti_alias: i32,
    blend_mode: i32,
    color: i32,
    style: i32,
    stroke_width: f32,
    stroke_cap: i32,
    stroke_join: i32,
    stroke_miter_limit: f32,
    invert_colors: i32,
) -> u32 {
    let mut paint = Paint::fill(decode_argb(color as u32));
    paint.blend_mode = decode_blend_mode(blend_mode);
    paint.style = if style == 1 {
        DrawStyle::Stroke
    } else {
        DrawStyle::Fill
    };
    paint.stroke_width = stroke_width.max(0.0);
    paint.stroke_cap = decode_cap(stroke_cap);
    paint.stroke_join = decode_join(stroke_join);
    paint.stroke_miter = stroke_miter_limit.max(0.0);
    paint.invert_colors = invert_colors != 0;

    with_state(|st| {
        let id = st.alloc_handle();
        st.paints.insert(id, paint);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn paint_dispose(handle: u32) {
    with_state(|st| {
        st.paints.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn paint_setShader(paint_handle: u32, shader_handle: u32) {
    with_state(|st| {
        let src = st.shaders.get(&shader_handle).cloned();
        if let Some(p) = st.paints.get_mut(&paint_handle) {
            p.color_source = src;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paint_setMaskFilter(paint_handle: u32, mask_filter_handle: u32) {
    with_state(|st| {
        let sigma = st
            .mask_filters
            .get(&mask_filter_handle)
            .copied()
            .unwrap_or(0.0);
        if let Some(p) = st.paints.get_mut(&paint_handle) {
            p.mask_blur_sigma = sigma;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paint_setColorFilter(paint_handle: u32, color_filter_handle: u32) {
    with_state(|st| {
        let cf = st.color_filters.get(&color_filter_handle).cloned();
        if let Some(p) = st.paints.get_mut(&paint_handle) {
            p.color_filter = cf;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paint_setImageFilter(paint_handle: u32, image_filter_handle: u32) {
    with_state(|st| {
        let imf = st.image_filters.get(&image_filter_handle).cloned();
        if let Some(p) = st.paints.get_mut(&paint_handle) {
            p.image_filter = imf;
        }
    });
}

unsafe fn parse_gradient_stops(
    colors_ptr: *const u32,
    stops_ptr: *const f32,
    count: i32,
) -> Vec<GradientStop> {
    if colors_ptr.is_null() || count <= 0 {
        return Vec::new();
    }
    let n = count as usize;
    let raw_colors = std::slice::from_raw_parts(colors_ptr, n);
    let raw_stops = if stops_ptr.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(stops_ptr, n))
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let offset = match raw_stops {
            Some(s) => s[i].clamp(0.0, 1.0),
            None => {
                if n <= 1 {
                    0.0
                } else {
                    (i as Scalar) / ((n - 1) as Scalar)
                }
            }
        };
        out.push(GradientStop {
            offset,
            color: decode_argb(raw_colors[i]),
        });
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn shader_createLinearGradient(
    end_points_ptr: *const f32,
    colors_ptr: *const u32,
    stops_ptr: *const f32,
    count: i32,
    tile_mode: i32,
    matrix33_ptr: *const f32,
) -> u32 {
    if end_points_ptr.is_null() {
        return 0;
    }
    let pts = std::slice::from_raw_parts(end_points_ptr, 4);
    let local_mat = read_matrix33_ptr(matrix33_ptr);
    let p0 = local_mat.transform_point(Point::new(pts[0], pts[1]));
    let p1 = local_mat.transform_point(Point::new(pts[2], pts[3]));
    let stops = parse_gradient_stops(colors_ptr, stops_ptr, count);
    let src = ColorSource::LinearGradient(LinearGradientParams {
        start_point: p0,
        end_point: p1,
        stops,
        tile_mode: decode_tile_mode(tile_mode),
        alpha: 1.0,
    });
    with_state(|st| {
        let id = st.alloc_handle();
        st.shaders.insert(id, src);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn shader_createRadialGradient(
    center_x: f32,
    center_y: f32,
    radius: f32,
    colors_ptr: *const u32,
    stops_ptr: *const f32,
    count: i32,
    tile_mode: i32,
    matrix33_ptr: *const f32,
) -> u32 {
    let local_mat = read_matrix33_ptr(matrix33_ptr);
    let center = local_mat.transform_point(Point::new(center_x, center_y));
    let scale = local_mat.get_max_basis_length_xy().max(1e-4);
    let stops = parse_gradient_stops(colors_ptr, stops_ptr, count);
    let src = ColorSource::RadialGradient(RadialGradientParams {
        center,
        radius: (radius * scale).max(1e-3),
        stops,
        tile_mode: decode_tile_mode(tile_mode),
        alpha: 1.0,
    });
    with_state(|st| {
        let id = st.alloc_handle();
        st.shaders.insert(id, src);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn shader_createConicalGradient(
    end_points_ptr: *const f32,
    start_radius: f32,
    end_radius: f32,
    colors_ptr: *const u32,
    stops_ptr: *const f32,
    count: i32,
    tile_mode: i32,
    matrix33_ptr: *const f32,
) -> u32 {
    let (s_pt, e_pt) = if end_points_ptr.is_null() {
        (Point::ZERO, Point::ZERO)
    } else {
        let pts = std::slice::from_raw_parts(end_points_ptr, 4);
        (Point::new(pts[0], pts[1]), Point::new(pts[2], pts[3]))
    };
    let local_mat = read_matrix33_ptr(matrix33_ptr);
    let start_center = local_mat.transform_point(s_pt);
    let end_center = local_mat.transform_point(e_pt);
    let scale = local_mat.get_max_basis_length_xy().max(1e-4);
    let stops = parse_gradient_stops(colors_ptr, stops_ptr, count);
    let src = ColorSource::ConicalGradient(ConicalGradientParams::new(
        start_center,
        (start_radius * scale).max(0.0),
        end_center,
        (end_radius * scale).max(0.0),
        stops,
        decode_tile_mode(tile_mode),
        1.0,
    ));
    with_state(|st| {
        let id = st.alloc_handle();
        st.shaders.insert(id, src);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn shader_createSweepGradient(
    center_x: f32,
    center_y: f32,
    colors_ptr: *const u32,
    stops_ptr: *const f32,
    count: i32,
    tile_mode: i32,
    start_angle: f32,
    end_angle: f32,
    matrix33_ptr: *const f32,
) -> u32 {
    let local_mat = read_matrix33_ptr(matrix33_ptr);
    let center = local_mat.transform_point(Point::new(center_x, center_y));
    let stops = parse_gradient_stops(colors_ptr, stops_ptr, count);
    let src = ColorSource::SweepGradient(SweepGradientParams::new(
        center,
        start_angle,
        end_angle,
        stops,
        decode_tile_mode(tile_mode),
        1.0,
    ));
    with_state(|st| {
        let id = st.alloc_handle();
        st.shaders.insert(id, src);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn shader_createFromImage(
    image_handle: u32,
    tile_mode_x: i32,
    tile_mode_y: i32,
    quality: i32,
    matrix33_ptr: *const f32,
) -> u32 {
    let mat = read_matrix33_ptr(matrix33_ptr);
    with_state(|st| {
        let Some(tex) = st.images.get(&image_handle).cloned() else {
            return 0;
        };
        let src = ColorSource::Image {
            texture: tex,
            tile_mode_x: decode_tile_mode(tile_mode_x),
            tile_mode_y: decode_tile_mode(tile_mode_y),
            matrix: mat,
            linear_filter: quality != 0,
        };
        let id = st.alloc_handle();
        st.shaders.insert(id, src);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn shader_dispose(handle: u32) {
    with_state(|st| {
        st.shaders.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn runtimeEffect_create(source_handle: u32) -> u32 {
    with_state(|st| {
        let src = st.read_sk_string(source_handle).unwrap_or_default();
        let id = st.alloc_handle();
        st.runtime_effects.insert(id, src);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn runtimeEffect_dispose(handle: u32) {
    with_state(|st| {
        st.runtime_effects.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn runtimeEffect_getUniformSize(_handle: u32) -> usize {
    64
}

#[no_mangle]
pub unsafe extern "C" fn uniformData_create(size: i32) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.uniform_data.insert(id, vec![0u8; size.max(1) as usize]);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn uniformData_getPointer(handle: u32) -> *mut u8 {
    with_state(|st| {
        st.uniform_data
            .get_mut(&handle)
            .map_or(std::ptr::null_mut(), |u| u.as_mut_ptr())
    })
}

#[no_mangle]
pub unsafe extern "C" fn uniformData_dispose(handle: u32) {
    with_state(|st| {
        st.uniform_data.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn shader_createRuntimeEffectShader(
    _runtime_effect: u32,
    uniforms: u32,
    _child_shaders: *const u32,
    _child_count: usize,
) -> u32 {
    with_state(|st| {
        let color = st
            .uniform_data
            .get(&uniforms)
            .and_then(|bytes| {
                if bytes.len() >= 16 {
                    let r = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    let g = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
                    let b = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
                    let a = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
                    Some(Color::new(
                        r.clamp(0.0, 1.0),
                        g.clamp(0.0, 1.0),
                        b.clamp(0.0, 1.0),
                        a.clamp(0.0, 1.0),
                    ))
                } else {
                    None
                }
            })
            .unwrap_or(Color::WHITE);
        let id = st.alloc_handle();
        st.shaders.insert(id, ColorSource::Solid(color));
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn maskFilter_createBlur(_blur_style: i32, sigma: f32) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.mask_filters.insert(id, sigma.max(0.0));
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn maskFilter_dispose(handle: u32) {
    with_state(|st| {
        st.mask_filters.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn colorFilter_createMode(color: i32, mode: i32) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.color_filters.insert(
            id,
            ColorFilter::Mode(decode_argb(color as u32), decode_blend_mode(mode)),
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn colorFilter_createMatrix(matrix_ptr: *const f32) -> u32 {
    let mut mat = [0.0f32; 20];
    if !matrix_ptr.is_null() {
        let s = std::slice::from_raw_parts(matrix_ptr, 20);
        mat.copy_from_slice(s);
    } else {
        mat[0] = 1.0;
        mat[6] = 1.0;
        mat[12] = 1.0;
        mat[18] = 1.0;
    }
    with_state(|st| {
        let id = st.alloc_handle();
        st.color_filters.insert(id, ColorFilter::Matrix(mat));
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn colorFilter_createSRGBToLinearGamma() -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.color_filters.insert(id, ColorFilter::SrgbToLinear);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn colorFilter_createLinearToSRGBGamma() -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.color_filters.insert(id, ColorFilter::LinearToSrgb);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn colorFilter_createCompose(outer_handle: u32, inner_handle: u32) -> u32 {
    with_state(|st| {
        let outer = st
            .color_filters
            .get(&outer_handle)
            .cloned()
            .unwrap_or(ColorFilter::Mode(Color::TRANSPARENT, BlendMode::Dst));
        let inner = st
            .color_filters
            .get(&inner_handle)
            .cloned()
            .unwrap_or(ColorFilter::Mode(Color::TRANSPARENT, BlendMode::Dst));
        let id = st.alloc_handle();
        st.color_filters
            .insert(id, ColorFilter::Compose(Box::new(outer), Box::new(inner)));
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn colorFilter_dispose(handle: u32) {
    with_state(|st| {
        st.color_filters.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn imageFilter_createBlur(sigma_x: f32, sigma_y: f32, tile_mode: i32) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.image_filters.insert(
            id,
            ImageFilter::Blur {
                sigma_x: sigma_x.max(0.0),
                sigma_y: sigma_y.max(0.0),
                tile_mode: decode_tile_mode(tile_mode),
            },
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn imageFilter_createDilate(radius_x: f32, radius_y: f32) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.image_filters.insert(
            id,
            ImageFilter::Dilate {
                radius_x: radius_x.max(0.0),
                radius_y: radius_y.max(0.0),
            },
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn imageFilter_createErode(radius_x: f32, radius_y: f32) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.image_filters.insert(
            id,
            ImageFilter::Erode {
                radius_x: radius_x.max(0.0),
                radius_y: radius_y.max(0.0),
            },
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn imageFilter_createMatrix(matrix33_ptr: *const f32, _quality: i32) -> u32 {
    let mat = read_matrix33_ptr(matrix33_ptr);
    with_state(|st| {
        let id = st.alloc_handle();
        st.image_filters.insert(id, ImageFilter::Matrix(mat));
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn imageFilter_createFromColorFilter(color_filter_handle: u32) -> u32 {
    with_state(|st| {
        let cf = st
            .color_filters
            .get(&color_filter_handle)
            .cloned()
            .unwrap_or(ColorFilter::Mode(Color::TRANSPARENT, BlendMode::Dst));
        let id = st.alloc_handle();
        st.image_filters.insert(id, ImageFilter::ColorFilter(cf));
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn imageFilter_compose(outer_handle: u32, inner_handle: u32) -> u32 {
    with_state(|st| {
        let outer = st
            .image_filters
            .get(&outer_handle)
            .cloned()
            .unwrap_or(ImageFilter::Matrix(Matrix::IDENTITY));
        let inner = st
            .image_filters
            .get(&inner_handle)
            .cloned()
            .unwrap_or(ImageFilter::Matrix(Matrix::IDENTITY));
        let id = st.alloc_handle();
        st.image_filters
            .insert(id, ImageFilter::Compose(Box::new(outer), Box::new(inner)));
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn imageFilter_dispose(handle: u32) {
    with_state(|st| {
        st.image_filters.remove(&handle);
    });
}

fn expand_rect_for_filter(filter: &ImageFilter, rect: Rect) -> Rect {
    match filter {
        ImageFilter::Blur {
            sigma_x, sigma_y, ..
        } => {
            let dx = sigma_x * 3.0;
            let dy = sigma_y * 3.0;
            Rect::from_ltrb(
                rect.get_left() - dx,
                rect.get_top() - dy,
                rect.get_right() + dx,
                rect.get_bottom() + dy,
            )
        }
        ImageFilter::Dilate { radius_x, radius_y } => Rect::from_ltrb(
            rect.get_left() - radius_x,
            rect.get_top() - radius_y,
            rect.get_right() + radius_x,
            rect.get_bottom() + radius_y,
        ),
        ImageFilter::Erode { .. } | ImageFilter::ColorFilter(_) => rect,
        ImageFilter::Matrix(m) => {
            let p0 = m.transform_point(Point::new(rect.get_left(), rect.get_top()));
            let p1 = m.transform_point(Point::new(rect.get_right(), rect.get_top()));
            let p2 = m.transform_point(Point::new(rect.get_right(), rect.get_bottom()));
            let p3 = m.transform_point(Point::new(rect.get_left(), rect.get_bottom()));
            let min_x = p0.x.min(p1.x).min(p2.x).min(p3.x);
            let min_y = p0.y.min(p1.y).min(p2.y).min(p3.y);
            let max_x = p0.x.max(p1.x).max(p2.x).max(p3.x);
            let max_y = p0.y.max(p1.y).max(p2.y).max(p3.y);
            Rect::from_ltrb(min_x, min_y, max_x, max_y)
        }
        ImageFilter::Compose(outer, inner) => {
            let r_inner = expand_rect_for_filter(inner, rect);
            expand_rect_for_filter(outer, r_inner)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn imageFilter_getFilterBounds(handle: u32, in_out_irect: *mut i32) {
    if in_out_irect.is_null() {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(in_out_irect, 4);
    let in_rect = Rect::from_ltrb(
        slice[0] as f32,
        slice[1] as f32,
        slice[2] as f32,
        slice[3] as f32,
    );
    let out_rect = with_state(|st| {
        st.image_filters
            .get(&handle)
            .map_or(in_rect, |f| expand_rect_for_filter(f, in_rect))
    });
    slice[0] = out_rect.get_left().floor() as i32;
    slice[1] = out_rect.get_top().floor() as i32;
    slice[2] = out_rect.get_right().ceil() as i32;
    slice[3] = out_rect.get_bottom().ceil() as i32;
}

// =============================================================================
// 6. Image & Vertices (`SkwasmImage` & `SkwasmVertices`)
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn image_createFromPixels(
    data_handle: u32,
    width: i32,
    height: i32,
    pixel_format: i32,
    row_bytes: usize,
) -> u32 {
    let w = width.max(1) as u32;
    let h = height.max(1) as u32;
    with_state(|st| {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        if let Some(entry) = st.sk_data.get(&data_handle) {
            let stride = row_bytes.max((w * 4) as usize);
            for y in 0..(h as usize) {
                let src_row = y * stride;
                let dst_row = y * (w as usize) * 4;
                for x in 0..(w as usize) {
                    let si = src_row + x * 4;
                    let di = dst_row + x * 4;
                    if si + 3 < entry.buf.len() {
                        let (r, g, b, a) = if pixel_format == 1 {
                            // BGRA8888 -> RGBA8888
                            (
                                entry.buf[si + 2],
                                entry.buf[si + 1],
                                entry.buf[si],
                                entry.buf[si + 3],
                            )
                        } else {
                            (
                                entry.buf[si],
                                entry.buf[si + 1],
                                entry.buf[si + 2],
                                entry.buf[si + 3],
                            )
                        };
                        // Premultiply straight RGBA
                        let af = u16::from(a);
                        rgba[di] = ((u16::from(r) * af + 127) / 255) as u8;
                        rgba[di + 1] = ((u16::from(g) * af + 127) / 255) as u8;
                        rgba[di + 2] = ((u16::from(b) * af + 127) / 255) as u8;
                        rgba[di + 3] = a;
                    }
                }
            }
        }
        let img_id = NEXT_IMAGE_ID.fetch_add(1, Ordering::Relaxed);
        let tex = ImageTexture::new(img_id, w, h, rgba);
        let id = st.alloc_handle();
        st.images.insert(id, tex);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn image_createFromPicture(
    picture_handle: u32,
    width: i32,
    height: i32,
) -> u32 {
    let w = width.max(1) as u32;
    let h = height.max(1) as u32;
    with_state(|st| {
        #[cfg(not(target_arch = "wasm32"))]
        let rgba = {
            let mut buf = vec![0u8; (w * h * 4) as usize];
            if let Some(dl) = st
                .pictures
                .get(&picture_handle)
                .map(|p| Arc::clone(&p.display_list))
            {
                let (alpha_arc, color_arc) = st.shared_glyph_atlases();
                let mut canvas = Canvas::new(w, h, Color::TRANSPARENT);
                canvas.draw_display_list(&dl);
                canvas.sync_glyph_atlases_with_arcs(&alpha_arc, &color_arc);
                if let Ok(renderer) = impeller_wgpu::WgpuRenderer::new_headless() {
                    let _ = renderer.sync_glyph_atlas(&st.typographer.alpha_atlas);
                    let _ = renderer.sync_glyph_atlas(&st.typographer.color_atlas);
                    if let Ok(pixels) = canvas.render_to_pixels(&renderer) {
                        buf = pixels;
                    }
                }
            }
            buf
        };
        #[cfg(target_arch = "wasm32")]
        let rgba = {
            let _ = picture_handle;
            vec![0u8; (w * h * 4) as usize]
        };

        let img_id = NEXT_IMAGE_ID.fetch_add(1, Ordering::Relaxed);
        let tex = ImageTexture::new(img_id, w, h, rgba);
        let id = st.alloc_handle();
        st.images.insert(id, tex);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn image_dispose(handle: u32) {
    with_state(|st| {
        st.images.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn image_getWidth(handle: u32) -> i32 {
    with_state(|st| st.images.get(&handle).map_or(0, |img| img.width as i32))
}

#[no_mangle]
pub unsafe extern "C" fn image_getHeight(handle: u32) -> i32 {
    with_state(|st| st.images.get(&handle).map_or(0, |img| img.height as i32))
}

#[no_mangle]
pub unsafe extern "C" fn vertices_create(
    vertex_mode: i32,
    vertex_count: i32,
    positions_ptr: *const f32,
    tex_coords_ptr: *const f32,
    colors_ptr: *const u32,
    index_count: i32,
    indices_ptr: *const u16,
) -> u32 {
    if positions_ptr.is_null() || vertex_count <= 0 {
        return 0;
    }
    let vcount = vertex_count as usize;
    let pos_slice = std::slice::from_raw_parts(positions_ptr, vcount * 2);
    let mut positions = Vec::with_capacity(vcount);
    for i in 0..vcount {
        positions.push(Point::new(pos_slice[i * 2], pos_slice[i * 2 + 1]));
    }
    let tex_coords = if tex_coords_ptr.is_null() {
        None
    } else {
        let ts = std::slice::from_raw_parts(tex_coords_ptr, vcount * 2);
        let mut tvec = Vec::with_capacity(vcount);
        for i in 0..vcount {
            tvec.push(Point::new(ts[i * 2], ts[i * 2 + 1]));
        }
        Some(tvec)
    };
    let colors = if colors_ptr.is_null() {
        None
    } else {
        let cs = std::slice::from_raw_parts(colors_ptr, vcount);
        Some(cs.iter().map(|&c| decode_argb(c)).collect())
    };
    let mode = match vertex_mode {
        1 => VertexMode::TriangleStrip,
        2 => VertexMode::TriangleFan,
        _ => VertexMode::Triangles,
    };

    let raw_indices: Vec<u16> = if !indices_ptr.is_null() && index_count > 0 {
        std::slice::from_raw_parts(indices_ptr, index_count as usize).to_vec()
    } else {
        (0..(vcount as u16)).collect()
    };

    let mut tri_indices = Vec::new();
    match mode {
        VertexMode::Triangles => {
            tri_indices = raw_indices;
        }
        VertexMode::TriangleStrip => {
            if raw_indices.len() >= 3 {
                for i in 0..(raw_indices.len() - 2) {
                    if i % 2 == 0 {
                        tri_indices.extend_from_slice(&[
                            raw_indices[i],
                            raw_indices[i + 1],
                            raw_indices[i + 2],
                        ]);
                    } else {
                        tri_indices.extend_from_slice(&[
                            raw_indices[i + 1],
                            raw_indices[i],
                            raw_indices[i + 2],
                        ]);
                    }
                }
            }
        }
        VertexMode::TriangleFan => {
            if raw_indices.len() >= 3 {
                let root = raw_indices[0];
                for i in 1..(raw_indices.len() - 1) {
                    tri_indices.extend_from_slice(&[root, raw_indices[i], raw_indices[i + 1]]);
                }
            }
        }
    }

    with_state(|st| {
        let id = st.alloc_handle();
        st.vertices.insert(
            id,
            VerticesEntry {
                mode,
                positions,
                tex_coords,
                colors,
                indices: tri_indices,
            },
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn vertices_dispose(handle: u32) {
    with_state(|st| {
        st.vertices.remove(&handle);
    });
}

// =============================================================================
// 7. PictureRecorder, Picture, & Canvas (`SkwasmCanvas` / `SkwasmPicture`)
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn pictureRecorder_create() -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.recorders.insert(
            id,
            PictureRecorderEntry {
                active_canvas_handle: None,
                cull_rect: Rect::from_ltrb(-1e9, -1e9, 1e9, 1e9),
            },
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn pictureRecorder_dispose(handle: u32) {
    with_state(|st| {
        if let Some(rec) = st.recorders.remove(&handle) {
            if let Some(cid) = rec.active_canvas_handle {
                st.canvases.remove(&cid);
            }
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn pictureRecorder_beginRecording(
    handle: u32,
    cull_rect_ptr: *const f32,
) -> u32 {
    let cull_rect =
        read_rect_ptr(cull_rect_ptr).unwrap_or_else(|| Rect::from_ltrb(-1e9, -1e9, 1e9, 1e9));
    with_state(|st| {
        let cid = st.alloc_handle();
        st.canvases.insert(
            cid,
            RecordingCanvasEntry {
                canvas: Canvas::new(
                    RECORDING_CANVAS_DIM,
                    RECORDING_CANVAS_DIM,
                    Color::TRANSPARENT,
                ),
                dl_builder: DisplayListBuilder::new(),
                cull_rect,
            },
        );
        if let Some(rec) = st.recorders.get_mut(&handle) {
            rec.active_canvas_handle = Some(cid);
            rec.cull_rect = cull_rect;
        }
        cid
    })
}

#[no_mangle]
pub unsafe extern "C" fn pictureRecorder_endRecording(handle: u32) -> u32 {
    with_state(|st| {
        let (dl, cull_rect, approx_bytes) = st
            .recorders
            .get_mut(&handle)
            .and_then(|rec| {
                let cid = rec.active_canvas_handle.take()?;
                let entry = st.canvases.remove(&cid)?;
                let dl = entry.dl_builder.into_display_list();
                let bytes = dl.len() * 64;
                Some((Arc::new(dl), entry.cull_rect, bytes))
            })
            .unwrap_or_else(|| {
                (
                    Arc::new(DisplayList::default()),
                    Rect::from_ltrb(0.0, 0.0, 0.0, 0.0),
                    0,
                )
            });

        let pid = st.alloc_handle();
        st.pictures.insert(
            pid,
            PictureEntry {
                display_list: dl,
                cull_rect,
                approx_bytes,
            },
        );
        pid
    })
}

#[no_mangle]
pub unsafe extern "C" fn picture_dispose(handle: u32) {
    with_state(|st| {
        st.pictures.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn picture_approximateBytesUsed(handle: u32) -> i32 {
    with_state(|st| {
        st.pictures
            .get(&handle)
            .map_or(0, |p| p.approx_bytes as i32)
    })
}

#[no_mangle]
pub unsafe extern "C" fn picture_getCullRect(handle: u32, out_rect_ptr: *mut f32) {
    if out_rect_ptr.is_null() {
        return;
    }
    let rect = with_state(|st| st.pictures.get(&handle).map_or(Rect::ZERO, |p| p.cull_rect));
    let out = std::slice::from_raw_parts_mut(out_rect_ptr, 4);
    out[0] = rect.get_left();
    out[1] = rect.get_top();
    out[2] = rect.get_right();
    out[3] = rect.get_bottom();
}

#[no_mangle]
pub unsafe extern "C" fn canvas_save(canvas_handle: u32) {
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas.save();
            c.dl_builder.save();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_saveLayer(
    canvas_handle: u32,
    rect_ptr: *const f32,
    paint_handle: u32,
    backdrop_filter_handle: u32,
) {
    let bounds = read_rect_ptr(rect_ptr);
    with_state(|st| {
        let paint = st.paints.get(&paint_handle).cloned();
        let backdrop_filter = if backdrop_filter_handle != 0 {
            st.image_filters.get(&backdrop_filter_handle).cloned()
        } else {
            None
        };
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas
                .save_layer_with_backdrop(bounds, paint.as_ref(), backdrop_filter.as_ref());
            c.dl_builder
                .save_layer_with_backdrop(bounds, paint, backdrop_filter);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_restore(canvas_handle: u32) {
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas.restore();
            c.dl_builder.restore();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_restoreToCount(canvas_handle: u32, count: i32) {
    let target = count.max(1) as usize;
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas.restore_to_count(target);
            c.dl_builder.restore_to_count(target);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_getSaveCount(canvas_handle: u32) -> i32 {
    with_state(|st| {
        st.canvases
            .get(&canvas_handle)
            .map_or(1, |c| c.canvas.get_save_count() as i32)
    })
}

#[no_mangle]
pub unsafe extern "C" fn canvas_translate(canvas_handle: u32, dx: f32, dy: f32) {
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas.translate(dx, dy);
            c.dl_builder.translate(dx, dy);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_scale(canvas_handle: u32, sx: f32, sy: f32) {
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas.scale(sx, sy);
            c.dl_builder.scale(sx, sy);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_rotate(canvas_handle: u32, degrees: f32) {
    let rad = Radians(degrees.to_radians());
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas.rotate(rad);
            c.dl_builder.rotate(rad);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_skew(canvas_handle: u32, sx: f32, sy: f32) {
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas.skew(sx, sy);
            c.dl_builder.skew(sx, sy);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_transform(canvas_handle: u32, matrix44_ptr: *const f32) {
    let mat = read_matrix44_ptr(matrix44_ptr);
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas.transform(mat);
            c.dl_builder.transform(mat);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_clipRect(
    canvas_handle: u32,
    rect_ptr: *const f32,
    op: i32,
    _antialias: i32,
) {
    let Some(rect) = read_rect_ptr(rect_ptr) else {
        return;
    };
    let clip_op = if op == 0 {
        ClipOperation::Difference
    } else {
        ClipOperation::Intersect
    };
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas.clip_rect_with_op(rect, clip_op);
            c.dl_builder.clip_rect_with_op(rect, clip_op);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_clipRRect(
    canvas_handle: u32,
    rrect_ptr: *const f32,
    _antialias: i32,
) {
    let Some((rect, radii)) = read_rrect_ptr(rrect_ptr) else {
        return;
    };
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas.clip_rect_with_op(rect, ClipOperation::Intersect);
            c.dl_builder
                .clip_rrect(rect, radii, ClipOperation::Intersect);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_clipPath(canvas_handle: u32, path_handle: u32, _antialias: i32) {
    with_state(|st| {
        let Some(p) = st.paths.get(&path_handle) else {
            return;
        };
        let path = p.build_path();
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.canvas
                .clip_path_bounds_only(&path, ClipOperation::Intersect);
            c.dl_builder.clip_path(path, ClipOperation::Intersect);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawColor(canvas_handle: u32, color: i32, blend_mode: i32) {
    let col = decode_argb(color as u32);
    let bm = decode_blend_mode(blend_mode);
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_color(col, bm);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_clear(canvas_handle: u32, color: u32) {
    let col = decode_argb(color);
    with_state(|st| {
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_color(col, BlendMode::Src);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawPaint(canvas_handle: u32, paint_handle: u32) {
    with_state(|st| {
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_paint(paint);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawLine(
    canvas_handle: u32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    paint_handle: u32,
) {
    with_state(|st| {
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        let p0 = Point::new(x1, y1);
        let p1 = Point::new(x2, y2);
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_line(p0, p1, paint);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawRect(
    canvas_handle: u32,
    rect_ptr: *const f32,
    paint_handle: u32,
) {
    let Some(rect) = read_rect_ptr(rect_ptr) else {
        return;
    };
    with_state(|st| {
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_rect(rect, paint);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawOval(
    canvas_handle: u32,
    rect_ptr: *const f32,
    paint_handle: u32,
) {
    let Some(rect) = read_rect_ptr(rect_ptr) else {
        return;
    };
    with_state(|st| {
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_oval(rect, paint);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawRRect(
    canvas_handle: u32,
    rrect_ptr: *const f32,
    paint_handle: u32,
) {
    let Some((rect, radii)) = read_rrect_ptr(rrect_ptr) else {
        return;
    };
    with_state(|st| {
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_rrect(rect, radii, paint);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawDRRect(
    canvas_handle: u32,
    outer_rrect_ptr: *const f32,
    inner_rrect_ptr: *const f32,
    paint_handle: u32,
) {
    let (Some((outer_r, outer_rad)), Some((inner_r, inner_rad))) = (
        read_rrect_ptr(outer_rrect_ptr),
        read_rrect_ptr(inner_rrect_ptr),
    ) else {
        return;
    };
    with_state(|st| {
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder
                .draw_diff_rrect(outer_r, outer_rad, inner_r, inner_rad, paint);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawCircle(
    canvas_handle: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    paint_handle: u32,
) {
    with_state(|st| {
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        let center = Point::new(cx, cy);
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_circle(center, radius, paint);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawArc(
    canvas_handle: u32,
    rect_ptr: *const f32,
    start_angle_deg: f32,
    sweep_angle_deg: f32,
    use_center: i32,
    paint_handle: u32,
) {
    let Some(rect) = read_rect_ptr(rect_ptr) else {
        return;
    };
    let start_rad = Radians(start_angle_deg.to_radians());
    let sweep_rad = Radians(sweep_angle_deg.to_radians());
    let uc = use_center != 0;
    with_state(|st| {
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_arc(rect, start_rad, sweep_rad, uc, paint);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawPath(canvas_handle: u32, path_handle: u32, paint_handle: u32) {
    with_state(|st| {
        let Some(p_entry) = st.paths.get(&path_handle) else {
            return;
        };
        let path = p_entry.build_path();
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_path(path, paint);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawShadow(
    canvas_handle: u32,
    path_handle: u32,
    elevation: f32,
    device_pixel_ratio: f32,
    color: i32,
    transparent_occluder: i32,
) {
    with_state(|st| {
        let Some(p_entry) = st.paths.get(&path_handle) else {
            return;
        };
        let path = p_entry.build_path();
        let col = decode_argb(color as u32);
        let to = transparent_occluder != 0;
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder
                .draw_shadow(path, col, elevation, to, device_pixel_ratio);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawParagraph(
    canvas_handle: u32,
    paragraph_handle: u32,
    x: f32,
    y: f32,
) {
    with_state(|st| {
        let Some(para) = st.paragraphs.get(&paragraph_handle).map(Arc::clone) else {
            return;
        };
        let offset = Point::new(x, y);
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_paragraph(para, offset);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawPicture(canvas_handle: u32, picture_handle: u32) {
    with_state(|st| {
        let Some(pic) = st.pictures.get(&picture_handle) else {
            return;
        };
        let dl = Arc::clone(&pic.display_list);
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_picture(dl);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawImage(
    canvas_handle: u32,
    image_handle: u32,
    x: f32,
    y: f32,
    paint_handle: u32,
    filter_quality: i32,
) {
    with_state(|st| {
        let Some(img) = st.images.get(&image_handle).cloned() else {
            return;
        };
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        let offset = Point::new(x, y);
        let linear = filter_quality != 0;
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_image(img, offset, paint, linear);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawImageRect(
    canvas_handle: u32,
    image_handle: u32,
    src_rect_ptr: *const f32,
    dst_rect_ptr: *const f32,
    paint_handle: u32,
    filter_quality: i32,
) {
    let (Some(src), Some(dst)) = (read_rect_ptr(src_rect_ptr), read_rect_ptr(dst_rect_ptr)) else {
        return;
    };
    with_state(|st| {
        let Some(img) = st.images.get(&image_handle).cloned() else {
            return;
        };
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        let linear = filter_quality != 0;
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_image_rect(img, src, dst, paint, linear);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawImageNine(
    canvas_handle: u32,
    image_handle: u32,
    center_irect_ptr: *const i32,
    dst_rect_ptr: *const f32,
    paint_handle: u32,
    filter_quality: i32,
) {
    let Some(dst) = read_rect_ptr(dst_rect_ptr) else {
        return;
    };
    if center_irect_ptr.is_null() {
        return;
    }
    let ir = std::slice::from_raw_parts(center_irect_ptr, 4);
    let center = Rect::from_ltrb(ir[0] as f32, ir[1] as f32, ir[2] as f32, ir[3] as f32);
    with_state(|st| {
        let Some(img) = st.images.get(&image_handle).cloned() else {
            return;
        };
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        let linear = filter_quality != 0;
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder
                .draw_image_nine(img, center, dst, paint, linear);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawPoints(
    canvas_handle: u32,
    point_mode: i32,
    points_ptr: *const f32,
    count: i32,
    paint_handle: u32,
) {
    if points_ptr.is_null() || count <= 0 {
        return;
    }
    let n = count as usize;
    let slice = std::slice::from_raw_parts(points_ptr, n * 2);
    with_state(|st| {
        let mut paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        let w = paint.stroke_width.max(1.0);
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            match point_mode {
                0 => {
                    // Points
                    let mut fill_paint = paint.clone();
                    fill_paint.style = DrawStyle::Fill;
                    let half = w * 0.5;
                    for i in 0..n {
                        let px = slice[i * 2];
                        let py = slice[i * 2 + 1];
                        let r = Rect::from_xywh(px - half, py - half, w, w);
                        c.dl_builder.draw_rect(r, fill_paint.clone());
                    }
                }
                1 => {
                    // Lines (pairs)
                    paint.style = DrawStyle::Stroke;
                    let mut i = 0;
                    while i + 1 < n {
                        let p0 = Point::new(slice[i * 2], slice[i * 2 + 1]);
                        let p1 = Point::new(slice[(i + 1) * 2], slice[(i + 1) * 2 + 1]);
                        c.dl_builder.draw_line(p0, p1, paint.clone());
                        i += 2;
                    }
                }
                _ => {
                    // Polygon (polyline)
                    paint.style = DrawStyle::Stroke;
                    let mut pb = PathBuilder::new();
                    for i in 0..n {
                        let pt = Point::new(slice[i * 2], slice[i * 2 + 1]);
                        if i == 0 {
                            pb.move_to(pt);
                        } else {
                            pb.line_to(pt);
                        }
                    }
                    let path = pb.build(FillType::NonZero);
                    c.dl_builder.draw_path(path, paint);
                }
            }
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawVertices(
    canvas_handle: u32,
    vertices_handle: u32,
    blend_mode: i32,
    paint_handle: u32,
) {
    with_state(|st| {
        let Some(v) = st.vertices.get(&vertices_handle).cloned() else {
            return;
        };
        let paint = st.paints.get(&paint_handle).cloned().unwrap_or_default();
        let bm = decode_blend_mode(blend_mode);
        let _ = v.mode;
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder
                .draw_vertices(v.positions, v.tex_coords, v.colors, v.indices, bm, paint);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_drawAtlas(
    canvas_handle: u32,
    atlas_handle: u32,
    rst_transforms_ptr: *const f32,
    tex_rects_ptr: *const f32,
    colors_ptr: *const u32,
    sprite_count: i32,
    blend_mode: i32,
    cull_rect_ptr: *const f32,
    paint_handle: u32,
    filter_quality: i32,
) {
    if rst_transforms_ptr.is_null() || tex_rects_ptr.is_null() || sprite_count <= 0 {
        return;
    }
    let n = sprite_count as usize;
    let rst_slice = std::slice::from_raw_parts(rst_transforms_ptr, n * 4);
    let rect_slice = std::slice::from_raw_parts(tex_rects_ptr, n * 4);
    let mut rst_vec = Vec::with_capacity(n);
    let mut rect_vec = Vec::with_capacity(n);
    for i in 0..n {
        rst_vec.push([
            rst_slice[i * 4],
            rst_slice[i * 4 + 1],
            rst_slice[i * 4 + 2],
            rst_slice[i * 4 + 3],
        ]);
        rect_vec.push(Rect::from_ltrb(
            rect_slice[i * 4],
            rect_slice[i * 4 + 1],
            rect_slice[i * 4 + 2],
            rect_slice[i * 4 + 3],
        ));
    }
    let colors_vec: Option<Vec<Color>> = if colors_ptr.is_null() {
        None
    } else {
        let cs = std::slice::from_raw_parts(colors_ptr, n);
        Some(cs.iter().map(|&c| decode_argb(c)).collect())
    };
    let cull_rect = read_rect_ptr(cull_rect_ptr);
    let bm = decode_blend_mode(blend_mode);
    let linear = filter_quality != 0;

    with_state(|st| {
        let Some(atlas) = st.images.get(&atlas_handle).cloned() else {
            return;
        };
        let paint = st.paints.get(&paint_handle).cloned();
        if let Some(c) = st.canvases.get_mut(&canvas_handle) {
            c.dl_builder.draw_atlas(
                atlas, rst_vec, rect_vec, colors_vec, bm, cull_rect, paint, linear,
            );
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn canvas_getTransform(canvas_handle: u32, out_matrix44_ptr: *mut f32) {
    if out_matrix44_ptr.is_null() {
        return;
    }
    let mat = with_state(|st| {
        st.canvases
            .get(&canvas_handle)
            .map_or(Matrix::IDENTITY, |c| c.canvas.get_current_transform())
    });
    let out = std::slice::from_raw_parts_mut(out_matrix44_ptr, 16);
    out.copy_from_slice(&mat.m);
}

#[no_mangle]
pub unsafe extern "C" fn canvas_getDeviceClipBounds(canvas_handle: u32, out_irect_ptr: *mut i32) {
    if out_irect_ptr.is_null() {
        return;
    }
    let bounds = with_state(|st| {
        st.canvases
            .get(&canvas_handle)
            .and_then(|c| c.canvas.get_local_coverage_limit())
            .unwrap_or_else(|| {
                Rect::from_xywh(
                    0.0,
                    0.0,
                    RECORDING_CANVAS_DIM as f32,
                    RECORDING_CANVAS_DIM as f32,
                )
            })
    });
    let out = std::slice::from_raw_parts_mut(out_irect_ptr, 4);
    out[0] = bounds.get_left().floor() as i32;
    out[1] = bounds.get_top().floor() as i32;
    out[2] = bounds.get_right().ceil() as i32;
    out[3] = bounds.get_bottom().ceil() as i32;
}

#[no_mangle]
pub unsafe extern "C" fn canvas_quickReject(canvas_handle: u32, rect_ptr: *const f32) -> i32 {
    let Some(rect) = read_rect_ptr(rect_ptr) else {
        return 1;
    };
    if rect.is_empty() {
        return 1;
    }
    with_state(|st| {
        let Some(c) = st.canvases.get(&canvas_handle) else {
            return 0;
        };
        let Some(local_clip) = c.canvas.get_current_local_cull_rect() else {
            return 1;
        };
        i32::from(!local_clip.intersects_with_rect(rect))
    })
}

// =============================================================================
// 8. FontCollection, Typeface, TextStyle, ParagraphStyle, StrutStyle
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn fontCollection_create() -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.font_collections.insert(id, ());
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn fontCollection_dispose(handle: u32) {
    with_state(|st| {
        st.font_collections.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn fontCollection_clearCaches(_handle: u32) {
    with_state(|st| {
        st.typographer.clear_caches();
        st.cached_alpha_atlas = None;
        st.cached_color_atlas = None;
    });
}

#[no_mangle]
pub unsafe extern "C" fn typeface_create(sk_data_handle: u32) -> u32 {
    with_state(|st| {
        let bytes = st
            .sk_data
            .get(&sk_data_handle)
            .map_or_else(Vec::new, |e| e.buf[..e.len].to_vec());
        st.typographer.register_font_bytes(&bytes, None);
        let id = st.alloc_handle();
        st.typefaces.insert(id, bytes);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn typeface_dispose(handle: u32) {
    with_state(|st| {
        st.typefaces.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn fontCollection_registerTypeface(
    _collection_handle: u32,
    typeface_handle: u32,
    font_name_handle: u32,
) {
    with_state(|st| {
        let alias = st.read_sk_string(font_name_handle);
        if let Some(bytes) = st.typefaces.get(&typeface_handle).cloned() {
            st.typographer.register_font_bytes(&bytes, alias.as_deref());
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn typefaces_filterCoveredCodePoints(
    typefaces_ptr: *const u32,
    typeface_count: i32,
    codepoints_ptr: *mut u32,
    codepoint_count: i32,
) -> i32 {
    if typefaces_ptr.is_null()
        || codepoints_ptr.is_null()
        || typeface_count <= 0
        || codepoint_count <= 0
    {
        return 0;
    }
    let tf_ids = std::slice::from_raw_parts(typefaces_ptr, typeface_count as usize);
    let cps = std::slice::from_raw_parts_mut(codepoints_ptr, codepoint_count as usize);
    with_state(|st| {
        let mut out_idx = 0usize;
        for i in 0..cps.len() {
            let cp = cps[i];
            let covered = tf_ids.iter().any(|tf_id| {
                st.typefaces
                    .get(tf_id)
                    .is_some_and(|b| TypographerContext::is_codepoint_covered_by_font_bytes(b, cp))
            });
            if covered {
                cps[out_idx] = cp;
                out_idx += 1;
            }
        }
        out_idx as i32
    })
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_create() -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.text_styles.insert(id, RichTextStyle::default());
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_copy(handle: u32) -> u32 {
    with_state(|st| {
        let style = st.text_styles.get(&handle).cloned().unwrap_or_default();
        let id = st.alloc_handle();
        st.text_styles.insert(id, style);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_dispose(handle: u32) {
    with_state(|st| {
        st.text_styles.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setColor(handle: u32, color: i32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.set_color(decode_argb(color as u32));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setDecoration(handle: u32, decoration: i32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.decoration = decoration;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setDecorationColor(handle: u32, color: i32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.decoration_color = Some(decode_argb(color as u32));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setDecorationStyle(handle: u32, style: i32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.decoration_style = style;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setDecorationThickness(handle: u32, thickness: f32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.decoration_thickness = thickness.max(0.0);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setFontStyle(handle: u32, weight: i32, slant: i32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.set_flutter_font_style(weight, slant);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setTextBaseline(handle: u32, baseline: i32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.text_baseline = baseline;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_clearFontFamilies(handle: u32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.font_families.clear();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_addFontFamilies(
    handle: u32,
    families_ptr: *const u32,
    count: i32,
) {
    if families_ptr.is_null() || count <= 0 {
        return;
    }
    let str_handles = std::slice::from_raw_parts(families_ptr, count as usize);
    with_state(|st| {
        let SkwasmState {
            sk_strings,
            sk_strings_16,
            text_styles,
            ..
        } = st;
        let Some(ts) = text_styles.get_mut(&handle) else {
            return;
        };
        ts.font_families.reserve(str_handles.len());
        for &sh in str_handles {
            let s = if let Some(entry) = sk_strings.get(&sh) {
                Some(String::from_utf8_lossy(&entry.buf[..entry.len]).into_owned())
            } else {
                sk_strings_16
                    .get(&sh)
                    .map(|entry16| String::from_utf16_lossy(&entry16.buf[..entry16.len]))
            };
            if let Some(s) = s {
                if !s.is_empty() {
                    ts.font_families.push(s);
                }
            }
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setFontSize(handle: u32, size: f32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.set_font_size(size);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setLetterSpacing(handle: u32, spacing: f32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.letter_spacing = spacing;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setWordSpacing(handle: u32, spacing: f32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.word_spacing = spacing;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setHeight(handle: u32, height: f32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.height_multiplier = if height > 0.0 { Some(height) } else { None };
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setHalfLeading(handle: u32, half_leading: i32) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.half_leading = half_leading != 0;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setLocale(handle: u32, locale_handle: u32) {
    with_state(|st| {
        let loc = st.read_sk_string(locale_handle);
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.locale = loc;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setBackground(handle: u32, paint_handle: u32) {
    with_state(|st| {
        let color = st
            .paints
            .get(&paint_handle)
            .map(Paint::effective_solid_color);
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.background_paint_color = color;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setForeground(handle: u32, paint_handle: u32) {
    with_state(|st| {
        let color = st
            .paints
            .get(&paint_handle)
            .map(Paint::effective_solid_color);
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.foreground_paint_color = color;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_addShadow(
    handle: u32,
    color: i32,
    offset_x: f32,
    offset_y: f32,
    blur_sigma: f32,
) {
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.shadows.push(TextShadow {
                color: decode_argb(color as u32),
                offset: Point::new(offset_x, offset_y),
                blur_sigma: blur_sigma.max(0.0),
            });
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_addFontFeature(
    handle: u32,
    feature_name_handle: u32,
    value: i32,
) {
    with_state(|st| {
        let Some(tag) = st.read_sk_string(feature_name_handle) else {
            return;
        };
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.font_features.push(FontFeature { tag, value });
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn textStyle_setFontVariations(
    handle: u32,
    axes_ptr: *const u32,
    values_ptr: *const f32,
    count: i32,
) {
    if axes_ptr.is_null() || values_ptr.is_null() || count <= 0 {
        return;
    }
    let axes = std::slice::from_raw_parts(axes_ptr, count as usize);
    let vals = std::slice::from_raw_parts(values_ptr, count as usize);
    with_state(|st| {
        if let Some(ts) = st.text_styles.get_mut(&handle) {
            ts.font_variations.clear();
            for i in 0..(count as usize) {
                ts.font_variations.push(FontVariation {
                    tag: axes[i],
                    value: vals[i],
                });
            }
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_create() -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.paragraph_styles.insert(id, ParagraphStyle::default());
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_dispose(handle: u32) {
    with_state(|st| {
        st.paragraph_styles.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_setTextAlign(handle: u32, text_align: i32) {
    with_state(|st| {
        if let Some(ps) = st.paragraph_styles.get_mut(&handle) {
            ps.text_align = TextAlign::from_skwasm(text_align);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_setTextDirection(handle: u32, text_direction: i32) {
    with_state(|st| {
        if let Some(ps) = st.paragraph_styles.get_mut(&handle) {
            ps.text_direction = TextDirection::from_skwasm(text_direction);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_setMaxLines(handle: u32, max_lines: usize) {
    with_state(|st| {
        if let Some(ps) = st.paragraph_styles.get_mut(&handle) {
            ps.max_lines = if max_lines > 0 && max_lines < usize::MAX {
                Some(max_lines)
            } else {
                None
            };
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_setHeight(handle: u32, height: f32) {
    with_state(|st| {
        if let Some(ps) = st.paragraph_styles.get_mut(&handle) {
            ps.height_multiplier = if height > 0.0 { Some(height) } else { None };
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_setTextHeightBehavior(
    handle: u32,
    apply_first_ascent: i32,
    apply_last_descent: i32,
) {
    with_state(|st| {
        if let Some(ps) = st.paragraph_styles.get_mut(&handle) {
            ps.apply_first_ascent = apply_first_ascent != 0;
            ps.apply_last_descent = apply_last_descent != 0;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_setEllipsis(handle: u32, ellipsis_handle: u32) {
    with_state(|st| {
        let ell = st.read_sk_string(ellipsis_handle);
        if let Some(ps) = st.paragraph_styles.get_mut(&handle) {
            ps.ellipsis = ell;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_setStrutStyle(handle: u32, strut_handle: u32) {
    with_state(|st| {
        let strut = st.strut_styles.get(&strut_handle).cloned();
        if let Some(ps) = st.paragraph_styles.get_mut(&handle) {
            ps.strut_style = strut;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_setTextStyle(handle: u32, text_style_handle: u32) {
    with_state(|st| {
        let Some(ts) = st.text_styles.get(&text_style_handle).cloned() else {
            return;
        };
        if let Some(ps) = st.paragraph_styles.get_mut(&handle) {
            ps.default_text_style = ts;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphStyle_setApplyRoundingHack(handle: u32, apply: i32) {
    with_state(|st| {
        if let Some(ps) = st.paragraph_styles.get_mut(&handle) {
            ps.apply_rounding_hack = apply != 0;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn strutStyle_create() -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.strut_styles.insert(id, StrutStyle::default());
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn strutStyle_dispose(handle: u32) {
    with_state(|st| {
        st.strut_styles.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn strutStyle_setFontFamilies(
    handle: u32,
    families_ptr: *const u32,
    count: i32,
) {
    if families_ptr.is_null() || count <= 0 {
        return;
    }
    let str_handles = std::slice::from_raw_parts(families_ptr, count as usize);
    with_state(|st| {
        let SkwasmState {
            sk_strings,
            sk_strings_16,
            strut_styles,
            ..
        } = st;
        let Some(ss) = strut_styles.get_mut(&handle) else {
            return;
        };
        ss.font_families.clear();
        ss.font_families.reserve(str_handles.len());
        for &sh in str_handles {
            let s = if let Some(entry) = sk_strings.get(&sh) {
                Some(String::from_utf8_lossy(&entry.buf[..entry.len]).into_owned())
            } else {
                sk_strings_16
                    .get(&sh)
                    .map(|entry16| String::from_utf16_lossy(&entry16.buf[..entry16.len]))
            };
            if let Some(s) = s {
                if !s.is_empty() {
                    ss.font_families.push(s);
                }
            }
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn strutStyle_setFontSize(handle: u32, size: f32) {
    with_state(|st| {
        if let Some(ss) = st.strut_styles.get_mut(&handle) {
            ss.font_size = Some(size.max(1.0));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn strutStyle_setHeight(handle: u32, height: f32) {
    with_state(|st| {
        if let Some(ss) = st.strut_styles.get_mut(&handle) {
            ss.height_multiplier = if height > 0.0 { Some(height) } else { None };
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn strutStyle_setHalfLeading(handle: u32, half_leading: i32) {
    with_state(|st| {
        if let Some(ss) = st.strut_styles.get_mut(&handle) {
            ss.half_leading = half_leading != 0;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn strutStyle_setLeading(handle: u32, leading: f32) {
    with_state(|st| {
        if let Some(ss) = st.strut_styles.get_mut(&handle) {
            ss.leading = Some(leading);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn strutStyle_setFontStyle(handle: u32, weight: i32, slant: i32) {
    with_state(|st| {
        if let Some(ss) = st.strut_styles.get_mut(&handle) {
            ss.set_flutter_font_style(weight, slant);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn strutStyle_setForceStrutHeight(handle: u32, force: i32) {
    with_state(|st| {
        if let Some(ss) = st.strut_styles.get_mut(&handle) {
            ss.force_strut_height = force != 0;
        }
    });
}

// =============================================================================
// 9. ParagraphBuilder, Paragraph, TextBoxList, LineMetrics
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_create(
    paragraph_style_handle: u32,
    _font_collection_handle: u32,
) -> u32 {
    with_state(|st| {
        let pstyle = st
            .paragraph_styles
            .get(&paragraph_style_handle)
            .cloned()
            .unwrap_or_default();
        let id = st.alloc_handle();
        st.paragraph_builders
            .insert(id, (ParagraphBuilder::new(pstyle), Vec::new()));
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_dispose(handle: u32) {
    with_state(|st| {
        st.paragraph_builders.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_pushStyle(handle: u32, text_style_handle: u32) {
    with_state(|st| {
        let Some(ts) = st.text_styles.get(&text_style_handle).cloned() else {
            return;
        };
        if let Some((pb, _)) = st.paragraph_builders.get_mut(&handle) {
            pb.push_style(ts);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_pop(handle: u32) {
    with_state(|st| {
        if let Some((pb, _)) = st.paragraph_builders.get_mut(&handle) {
            pb.pop();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_addText(handle: u32, sk_string16_handle: u32) {
    with_state(|st| {
        let SkwasmState {
            sk_strings_16,
            paragraph_builders,
            ..
        } = st;
        let Some(entry16) = sk_strings_16.get(&sk_string16_handle) else {
            return;
        };
        let units = &entry16.buf[..entry16.len];
        if let Some((pb, utf8_cache)) = paragraph_builders.get_mut(&handle) {
            pb.add_text_utf16(units);
            utf8_cache.clear();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_addPlaceholder(
    handle: u32,
    width: f32,
    height: f32,
    alignment: i32,
    baseline_offset: f32,
    baseline: i32,
) {
    with_state(|st| {
        if let Some((pb, utf8_cache)) = st.paragraph_builders.get_mut(&handle) {
            pb.add_placeholder(PlaceholderSpan {
                width,
                height,
                alignment,
                baseline_offset,
                baseline,
            });
            utf8_cache.clear();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_getUtf8Text(
    handle: u32,
    out_size: *mut u32,
) -> *const u8 {
    with_state(|st| {
        let Some((pb, utf8_cache)) = st.paragraph_builders.get_mut(&handle) else {
            if !out_size.is_null() {
                *out_size = 0;
            }
            return std::ptr::null();
        };
        if utf8_cache.is_empty() && !pb.utf8_text().is_empty() {
            utf8_cache.extend_from_slice(pb.utf8_text().as_bytes());
        }
        if !out_size.is_null() {
            *out_size = utf8_cache.len() as u32;
        }
        if utf8_cache.is_empty() {
            std::ptr::null()
        } else {
            utf8_cache.as_ptr()
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_setGraphemeBreaksUtf16(
    handle: u32,
    _pos_buffer_handle: u32,
) {
    with_state(|st| {
        if let Some((pb, _)) = st.paragraph_builders.get_mut(&handle) {
            pb.set_grapheme_breaks_utf16(Vec::new());
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_setWordBreaksUtf16(handle: u32, _pos_buffer_handle: u32) {
    with_state(|st| {
        if let Some((pb, _)) = st.paragraph_builders.get_mut(&handle) {
            pb.set_word_breaks_utf16(Vec::new());
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_setLineBreaksUtf16(
    handle: u32,
    _line_break_buffer_handle: u32,
) {
    with_state(|st| {
        if let Some((pb, _)) = st.paragraph_builders.get_mut(&handle) {
            pb.set_line_breaks_utf16(Vec::new());
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraphBuilder_build(handle: u32) -> u32 {
    with_state(|st| {
        let para = st.paragraph_builders.remove(&handle).map_or_else(
            || ParagraphBuilder::new(ParagraphStyle::default()).build(),
            |(pb, _)| pb.build(),
        );
        let id = st.alloc_handle();
        st.paragraphs.insert(id, Arc::new(para));
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_dispose(handle: u32) {
    with_state(|st| {
        st.paragraphs.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_layout(handle: u32, width: f32) {
    with_state(|st| {
        let SkwasmState {
            paragraphs,
            typographer,
            ..
        } = st;
        let Some(para) = paragraphs.get_mut(&handle) else {
            return;
        };
        if let Some(p) = Arc::get_mut(para) {
            p.layout(typographer, width);
        } else {
            Arc::make_mut(para).layout(typographer, width);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getWidth(handle: u32) -> f32 {
    with_state(|st| st.paragraphs.get(&handle).map_or(0.0, |p| p.width))
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getHeight(handle: u32) -> f32 {
    with_state(|st| st.paragraphs.get(&handle).map_or(0.0, |p| p.height))
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getLongestLine(handle: u32) -> f32 {
    with_state(|st| st.paragraphs.get(&handle).map_or(0.0, |p| p.longest_line))
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getMinIntrinsicWidth(handle: u32) -> f32 {
    with_state(|st| {
        st.paragraphs
            .get(&handle)
            .map_or(0.0, |p| p.min_intrinsic_width)
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getMaxIntrinsicWidth(handle: u32) -> f32 {
    with_state(|st| {
        st.paragraphs
            .get(&handle)
            .map_or(0.0, |p| p.max_intrinsic_width)
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getAlphabeticBaseline(handle: u32) -> f32 {
    with_state(|st| {
        st.paragraphs
            .get(&handle)
            .map_or(0.0, |p| p.alphabetic_baseline)
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getIdeographicBaseline(handle: u32) -> f32 {
    with_state(|st| {
        st.paragraphs
            .get(&handle)
            .map_or(0.0, |p| p.ideographic_baseline)
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getDidExceedMaxLines(handle: u32) -> i32 {
    with_state(|st| {
        st.paragraphs
            .get(&handle)
            .map_or(0, |p| i32::from(p.did_exceed_max_lines))
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getPositionForOffset(
    handle: u32,
    dx: f32,
    dy: f32,
    out_affinity: *mut i32,
) -> i32 {
    let (pos, affinity) = with_state(|st| {
        st.paragraphs
            .get(&handle)
            .map_or((0, 1), |p| p.get_position_for_offset(dx, dy))
    });
    if !out_affinity.is_null() {
        *out_affinity = affinity;
    }
    pos as i32
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getClosestGlyphInfoAtCoordinate(
    handle: u32,
    dx: f32,
    dy: f32,
    out_bounds_ptr: *mut f32,
    out_range_ptr: *mut u32,
    out_bool_flags_ptr: *mut u8,
) -> i32 {
    let info = with_state(|st| {
        st.paragraphs
            .get(&handle)
            .and_then(|p| p.get_closest_glyph_info_at(dx, dy))
    });
    let Some(gi) = info else {
        return 0;
    };
    if !out_bounds_ptr.is_null() {
        let b = std::slice::from_raw_parts_mut(out_bounds_ptr, 4);
        b[0] = gi.grapheme_cluster_bounds.get_left();
        b[1] = gi.grapheme_cluster_bounds.get_top();
        b[2] = gi.grapheme_cluster_bounds.get_right();
        b[3] = gi.grapheme_cluster_bounds.get_bottom();
    }
    if !out_range_ptr.is_null() {
        let r = std::slice::from_raw_parts_mut(out_range_ptr, 2);
        r[0] = gi.grapheme_cluster_code_unit_range_start as u32;
        r[1] = gi.grapheme_cluster_code_unit_range_end as u32;
    }
    if !out_bool_flags_ptr.is_null() {
        *out_bool_flags_ptr = u8::from(gi.is_ltr);
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getGlyphInfoAt(
    handle: u32,
    code_unit_offset: u32,
    out_bounds_ptr: *mut f32,
    out_range_ptr: *mut u32,
    out_bool_flags_ptr: *mut u8,
) -> i32 {
    let info = with_state(|st| {
        st.paragraphs
            .get(&handle)
            .and_then(|p| p.get_glyph_info_at(code_unit_offset as usize))
    });
    let Some(gi) = info else {
        return 0;
    };
    if !out_bounds_ptr.is_null() {
        let b = std::slice::from_raw_parts_mut(out_bounds_ptr, 4);
        b[0] = gi.grapheme_cluster_bounds.get_left();
        b[1] = gi.grapheme_cluster_bounds.get_top();
        b[2] = gi.grapheme_cluster_bounds.get_right();
        b[3] = gi.grapheme_cluster_bounds.get_bottom();
    }
    if !out_range_ptr.is_null() {
        let r = std::slice::from_raw_parts_mut(out_range_ptr, 2);
        r[0] = gi.grapheme_cluster_code_unit_range_start as u32;
        r[1] = gi.grapheme_cluster_code_unit_range_end as u32;
    }
    if !out_bool_flags_ptr.is_null() {
        *out_bool_flags_ptr = u8::from(gi.is_ltr);
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getWordBoundary(
    handle: u32,
    position: u32,
    out_range_ptr: *mut i32,
) {
    if out_range_ptr.is_null() {
        return;
    }
    let (start, end) = with_state(|st| {
        st.paragraphs
            .get(&handle)
            .map_or((0, 0), |p| p.get_word_boundary(position as usize))
    });
    let out = std::slice::from_raw_parts_mut(out_range_ptr, 2);
    out[0] = start as i32;
    out[1] = end as i32;
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getLineCount(handle: u32) -> usize {
    with_state(|st| st.paragraphs.get(&handle).map_or(0, |p| p.lines.len()))
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getLineNumberAt(handle: u32, char_index: usize) -> i32 {
    with_state(|st| {
        st.paragraphs
            .get(&handle)
            .and_then(|p| p.get_line_number_at(char_index))
            .map_or(-1, |ln| ln as i32)
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getLineMetricsAtIndex(handle: u32, index: usize) -> u32 {
    with_state(|st| {
        let Some(lm) = st
            .paragraphs
            .get(&handle)
            .and_then(|p| p.lines.get(index))
            .cloned()
        else {
            return 0;
        };
        let id = st.alloc_handle();
        st.line_metrics.insert(id, lm);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getBoxesForRange(
    handle: u32,
    start: i32,
    end: i32,
    _height_style: i32,
    _width_style: i32,
) -> u32 {
    with_state(|st| {
        let boxes = st.paragraphs.get(&handle).map_or_else(Vec::new, |p| {
            p.get_boxes_for_range(start.max(0) as usize, end.max(0) as usize)
        });
        let id = st.alloc_handle();
        st.text_box_lists.insert(id, boxes);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getBoxesForPlaceholders(handle: u32) -> u32 {
    with_state(|st| {
        let boxes = st
            .paragraphs
            .get(&handle)
            .map_or_else(Vec::new, |p| p.get_boxes_for_placeholders());
        let id = st.alloc_handle();
        st.text_box_lists.insert(id, boxes);
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn paragraph_getUnresolvedCodePoints(
    _handle: u32,
    _out_codepoints_ptr: *mut u32,
    _out_length: i32,
) -> i32 {
    0
}

#[no_mangle]
pub unsafe extern "C" fn textBoxList_dispose(handle: u32) {
    with_state(|st| {
        st.text_box_lists.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn textBoxList_getLength(handle: u32) -> usize {
    with_state(|st| st.text_box_lists.get(&handle).map_or(0, Vec::len))
}

#[no_mangle]
pub unsafe extern "C" fn textBoxList_getBoxAtIndex(
    handle: u32,
    index: usize,
    out_rect_ptr: *mut f32,
) -> i32 {
    with_state(|st| {
        let Some(tb) = st.text_box_lists.get(&handle).and_then(|l| l.get(index)) else {
            return TextDirection::Ltr.to_skwasm();
        };
        if !out_rect_ptr.is_null() {
            let out = std::slice::from_raw_parts_mut(out_rect_ptr, 4);
            out[0] = tb.rect.get_left();
            out[1] = tb.rect.get_top();
            out[2] = tb.rect.get_right();
            out[3] = tb.rect.get_bottom();
        }
        tb.direction.to_skwasm()
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_create(
    hard_break: i32,
    ascent: f64,
    descent: f64,
    unscaled_ascent: f64,
    height: f64,
    width: f64,
    left: f64,
    baseline: f64,
    line_number: usize,
) -> u32 {
    with_state(|st| {
        let id = st.alloc_handle();
        st.line_metrics.insert(
            id,
            ParagraphLineMetrics {
                hard_break: hard_break != 0,
                ascent: ascent as f32,
                descent: descent as f32,
                unscaled_ascent: unscaled_ascent as f32,
                height: height as f32,
                width: width as f32,
                left: left as f32,
                baseline: baseline as f32,
                line_number,
                start_index: 0,
                end_index: 0,
                end_excluding_whitespaces: 0,
                end_including_newline: 0,
            },
        );
        id
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_dispose(handle: u32) {
    with_state(|st| {
        st.line_metrics.remove(&handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getHardBreak(handle: u32) -> i32 {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0, |m| i32::from(m.hard_break))
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getAscent(handle: u32) -> f64 {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0.0, |m| f64::from(m.ascent))
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getDescent(handle: u32) -> f64 {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0.0, |m| f64::from(m.descent))
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getUnscaledAscent(handle: u32) -> f64 {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0.0, |m| f64::from(m.unscaled_ascent))
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getHeight(handle: u32) -> f64 {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0.0, |m| f64::from(m.height))
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getWidth(handle: u32) -> f64 {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0.0, |m| f64::from(m.width))
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getLeft(handle: u32) -> f64 {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0.0, |m| f64::from(m.left))
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getBaseline(handle: u32) -> f64 {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0.0, |m| f64::from(m.baseline))
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getLineNumber(handle: u32) -> i32 {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0, |m| m.line_number as i32)
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getStartIndex(handle: u32) -> usize {
    with_state(|st| st.line_metrics.get(&handle).map_or(0, |m| m.start_index))
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getEndIndex(handle: u32) -> usize {
    with_state(|st| st.line_metrics.get(&handle).map_or(0, |m| m.end_index))
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getEndExcludingWhitespaces(handle: u32) -> usize {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0, |m| m.end_excluding_whitespaces)
    })
}

#[no_mangle]
pub unsafe extern "C" fn lineMetrics_getEndIncludingNewline(handle: u32) -> usize {
    with_state(|st| {
        st.line_metrics
            .get(&handle)
            .map_or(0, |m| m.end_including_newline)
    })
}

// =============================================================================
// Unit Tests (`>= 10` tests with `/// C++ source:` provenance)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use impeller_typographer::ROBOTO_BOLD_TTF;

    /// C++ source: skwasm/memory.cpp:EmscriptenStackAndHeapAllocation
    #[test]
    fn test_emscripten_stack_and_heap_allocations() {
        unsafe {
            let saved = emscripten_stack_get_current();
            let free_before = emscripten_stack_get_free();
            assert!(free_before >= 1024 * 1024);

            let p1 = _emscripten_stack_alloc(24);
            let p2 = _emscripten_stack_alloc(48);
            assert_eq!(p1 % 16, 0);
            assert_eq!(p2 % 16, 0);
            assert_eq!(p2 - p1, 32);
            assert!(emscripten_stack_get_free() < free_before);

            _emscripten_stack_restore(saved);
            assert_eq!(emscripten_stack_get_current(), saved);
            assert_eq!(emscripten_stack_get_free(), free_before);

            let hp = emscripten_builtin_malloc(128);
            assert!(!hp.is_null());
            assert_eq!((hp as usize) % 16, 0);
            std::slice::from_raw_parts_mut(hp, 128).fill(0xAB);
            emscripten_builtin_free(hp);
        }
    }

    /// C++ source: skwasm/data.cpp:SkDataAndSkStringRoundtrip
    #[test]
    fn test_sk_data_and_sk_string_roundtrip() {
        unsafe {
            let d = skData_create(5);
            let ptr = skData_getPointer(d);
            assert!(!ptr.is_null());
            std::slice::from_raw_parts_mut(ptr, 5).copy_from_slice(b"hello");
            assert_eq!(skData_getSize(d), 5);
            let cptr = skData_getConstPointer(d);
            assert_eq!(std::slice::from_raw_parts(cptr, 5), b"hello");
            skData_dispose(d);

            let s8 = skString_allocate(6);
            std::slice::from_raw_parts_mut(skString_getData(s8), 6).copy_from_slice(b"Roboto");
            assert_eq!(skString_getLength(s8), 6);
            assert_eq!(
                with_state(|st| st.read_sk_string(s8)),
                Some("Roboto".to_string())
            );
            skString_free(s8);

            let s16 = skString16_allocate(4);
            let utf16_vals: [u16; 4] = [b'W' as u16, b'A' as u16, b'S' as u16, b'M' as u16];
            std::slice::from_raw_parts_mut(skString16_getData(s16), 4).copy_from_slice(&utf16_vals);
            assert_eq!(
                with_state(|st| st.read_sk_string(s16)),
                Some("WASM".to_string())
            );
            skString16_free(s16);
        }
    }

    /// C++ source: skwasm/path.cpp:PathConstructionContainsAndBounds
    #[test]
    fn test_path_construction_contains_and_bounds() {
        unsafe {
            let path = path_create();
            path_setFillType(path, 0);
            assert_eq!(path_getFillType(path), 0);

            let rect = [10.0f32, 20.0, 110.0, 80.0];
            path_addRect(path, rect.as_ptr());
            assert_eq!(path_contains(path, 50.0, 50.0), 1);
            assert_eq!(path_contains(path, 5.0, 5.0), 0);

            let mut bounds = [0.0f32; 4];
            path_getBounds(path, bounds.as_mut_ptr());
            assert_eq!(bounds, [10.0, 20.0, 110.0, 80.0]);

            let copy = path_copy(path);
            let m33 = [1.0f32, 0.0, 15.0, 0.0, 1.0, 25.0, 0.0, 0.0, 1.0];
            path_transform(copy, m33.as_ptr());
            let mut copy_bounds = [0.0f32; 4];
            path_getBounds(copy, copy_bounds.as_mut_ptr());
            assert!((copy_bounds[0] - 25.0).abs() < 1e-3);
            assert!((copy_bounds[1] - 45.0).abs() < 1e-3);

            path_dispose(copy);
            path_dispose(path);
        }
    }

    /// C++ source: skwasm/paint.cpp:PaintShadersAndFiltersLifecycle
    #[test]
    fn test_paint_shaders_and_filters_lifecycle() {
        unsafe {
            let paint = paint_create(1, 3, 0xFF336699u32 as i32, 0, 2.0, 1, 1, 4.0, 0);
            let pts = [0.0f32, 0.0, 100.0, 100.0];
            let colors = [0xFFFF0000u32, 0xFF0000FFu32];
            let stops = [0.0f32, 1.0];
            let shader = shader_createLinearGradient(
                pts.as_ptr(),
                colors.as_ptr(),
                stops.as_ptr(),
                2,
                0,
                std::ptr::null(),
            );
            paint_setShader(paint, shader);

            let mf = maskFilter_createBlur(0, 6.0);
            paint_setMaskFilter(paint, mf);

            let cf = colorFilter_createMode(0x8000FF00u32 as i32, 3);
            paint_setColorFilter(paint, cf);

            let imf = imageFilter_createBlur(4.0, 5.0, 0);
            paint_setImageFilter(paint, imf);

            let mut filter_bounds = [10i32, 20, 110, 120];
            imageFilter_getFilterBounds(imf, filter_bounds.as_mut_ptr());
            assert_eq!(filter_bounds, [-2, 5, 122, 135]);

            imageFilter_dispose(imf);
            colorFilter_dispose(cf);
            maskFilter_dispose(mf);
            shader_dispose(shader);
            paint_dispose(paint);
        }
    }

    /// C++ source: skwasm/fonts.cpp:FontCollectionAndTypefaceRegistration
    #[test]
    fn test_font_collection_and_typeface_registration() {
        unsafe {
            let fc = fontCollection_create();
            let data = skData_create(ROBOTO_BOLD_TTF.len());
            std::slice::from_raw_parts_mut(skData_getPointer(data), ROBOTO_BOLD_TTF.len())
                .copy_from_slice(ROBOTO_BOLD_TTF);

            let tf = typeface_create(data);
            let name = skString_allocate(10);
            std::slice::from_raw_parts_mut(skString_getData(name), 10)
                .copy_from_slice(b"CustomBold");
            fontCollection_registerTypeface(fc, tf, name);

            let mut cps = [u32::from(b'A'), 0x10FFFF];
            let covered = typefaces_filterCoveredCodePoints(tf_as_ptr(&tf), 1, cps.as_mut_ptr(), 2);
            assert_eq!(covered, 1);
            assert_eq!(cps[0], u32::from(b'A'));

            skString_free(name);
            typeface_dispose(tf);
            skData_dispose(data);
            fontCollection_dispose(fc);
        }
    }

    fn tf_as_ptr(tf: &u32) -> *const u32 {
        tf as *const u32
    }

    /// C++ source: skwasm/text/paragraph.cpp:ParagraphBuilderLayoutAndMetrics
    #[test]
    fn test_paragraph_builder_layout_and_metrics() {
        unsafe {
            let fc = fontCollection_create();
            let ps = paragraphStyle_create();
            paragraphStyle_setTextAlign(ps, 0);
            paragraphStyle_setMaxLines(ps, 3);

            let ts = textStyle_create();
            textStyle_setFontSize(ts, 18.0);
            textStyle_setColor(ts, 0xFFFFFFFFu32 as i32);
            textStyle_setFontStyle(ts, 6, 0); // Bold
            paragraphStyle_setTextStyle(ps, ts);

            let pb = paragraphBuilder_create(ps, fc);
            paragraphBuilder_pushStyle(pb, ts);

            let text = "Impeller Skwasm FFI";
            let utf16: Vec<u16> = text.encode_utf16().collect();
            let s16 = skString16_allocate(utf16.len());
            std::slice::from_raw_parts_mut(skString16_getData(s16), utf16.len())
                .copy_from_slice(&utf16);
            paragraphBuilder_addText(pb, s16);

            let mut utf8_len = 0u32;
            let utf8_ptr = paragraphBuilder_getUtf8Text(pb, &mut utf8_len as *mut u32);
            assert!(!utf8_ptr.is_null());
            assert_eq!(utf8_len as usize, text.len());

            paragraphBuilder_pop(pb);
            let para = paragraphBuilder_build(pb);
            paragraph_layout(para, 300.0);

            assert!(paragraph_getWidth(para) > 0.0);
            assert!(paragraph_getHeight(para) > 10.0);
            assert!(paragraph_getLongestLine(para) > 40.0);
            assert!(paragraph_getMaxIntrinsicWidth(para) >= paragraph_getMinIntrinsicWidth(para));
            assert!(paragraph_getAlphabeticBaseline(para) > 0.0);
            assert_eq!(paragraph_getDidExceedMaxLines(para), 0);
            assert!(paragraph_getLineCount(para) >= 1);

            let lm = paragraph_getLineMetricsAtIndex(para, 0);
            assert_ne!(lm, 0);
            assert!(lineMetrics_getWidth(lm) > 40.0);
            assert!(lineMetrics_getHeight(lm) > 10.0);
            lineMetrics_dispose(lm);

            let boxes = paragraph_getBoxesForRange(para, 0, 8, 0, 0);
            assert!(textBoxList_getLength(boxes) >= 1);
            let mut box_rect = [0.0f32; 4];
            let dir = textBoxList_getBoxAtIndex(boxes, 0, box_rect.as_mut_ptr());
            assert_eq!(dir, 1); // LTR
            assert!(box_rect[2] > box_rect[0]);
            textBoxList_dispose(boxes);

            skString16_free(s16);
            paragraph_dispose(para);
            textStyle_dispose(ts);
            paragraphStyle_dispose(ps);
            fontCollection_dispose(fc);
        }
    }

    /// C++ source: skwasm/text/paragraph.cpp:ParagraphGlyphInfoAndWordBoundary
    #[test]
    fn test_paragraph_glyph_info_and_word_boundary() {
        unsafe {
            let fc = fontCollection_create();
            let ps = paragraphStyle_create();
            let ts = textStyle_create();
            textStyle_setFontSize(ts, 16.0);
            paragraphStyle_setTextStyle(ps, ts);

            let pb = paragraphBuilder_create(ps, fc);
            let utf16: Vec<u16> = "Hello World".encode_utf16().collect();
            let s16 = skString16_allocate(utf16.len());
            std::slice::from_raw_parts_mut(skString16_getData(s16), utf16.len())
                .copy_from_slice(&utf16);
            paragraphBuilder_addText(pb, s16);
            let para = paragraphBuilder_build(pb);
            paragraph_layout(para, 200.0);

            let mut bounds = [0.0f32; 4];
            let mut range = [0u32; 2];
            let mut is_ltr = 0u8;
            let ok = paragraph_getGlyphInfoAt(
                para,
                1,
                bounds.as_mut_ptr(),
                range.as_mut_ptr(),
                &mut is_ltr as *mut u8,
            );
            assert_eq!(ok, 1);
            assert_eq!(range, [1, 2]);
            assert_eq!(is_ltr, 1);

            let ok_closest = paragraph_getClosestGlyphInfoAtCoordinate(
                para,
                bounds[0] + 1.0,
                bounds[1] + 2.0,
                bounds.as_mut_ptr(),
                range.as_mut_ptr(),
                &mut is_ltr as *mut u8,
            );
            assert_eq!(ok_closest, 1);

            let mut word = [0i32; 2];
            paragraph_getWordBoundary(para, 1, word.as_mut_ptr());
            assert_eq!(word, [0, 5]);

            skString16_free(s16);
            paragraph_dispose(para);
            textStyle_dispose(ts);
            paragraphStyle_dispose(ps);
            fontCollection_dispose(fc);
        }
    }

    /// C++ source: skwasm/canvas.cpp:PictureRecorderAndCanvasTransformClipStack
    #[test]
    fn test_picture_recorder_and_canvas_transform_clip_stack() {
        unsafe {
            let rec = pictureRecorder_create();
            let cull = [0.0f32, 0.0, 400.0, 300.0];
            let canvas = pictureRecorder_beginRecording(rec, cull.as_ptr());

            assert_eq!(canvas_getSaveCount(canvas), 1);
            canvas_save(canvas);
            assert_eq!(canvas_getSaveCount(canvas), 2);
            canvas_translate(canvas, 25.0, 50.0);
            canvas_scale(canvas, 2.0, 2.0);

            let mut m44 = [0.0f32; 16];
            canvas_getTransform(canvas, m44.as_mut_ptr());
            assert!((m44[0] - 2.0).abs() < 1e-4);
            assert!((m44[12] - 25.0).abs() < 1e-4);
            assert!((m44[13] - 50.0).abs() < 1e-4);

            let clip = [0.0f32, 0.0, 100.0, 100.0];
            canvas_clipRect(canvas, clip.as_ptr(), 1, 1);
            let outside = [200.0f32, 200.0, 250.0, 250.0];
            assert_eq!(canvas_quickReject(canvas, outside.as_ptr()), 1);

            let inside = [10.0f32, 10.0, 40.0, 40.0];
            assert_eq!(canvas_quickReject(canvas, inside.as_ptr()), 0);
            let paint = paint_create(1, 3, 0xFFFF0000u32 as i32, 0, 1.0, 0, 0, 4.0, 0);
            canvas_drawRect(canvas, inside.as_ptr(), paint);

            canvas_restore(canvas);
            assert_eq!(canvas_getSaveCount(canvas), 1);

            let pic = pictureRecorder_endRecording(rec);
            let mut pic_cull = [0.0f32; 4];
            picture_getCullRect(pic, pic_cull.as_mut_ptr());
            assert_eq!(pic_cull, cull);
            assert!(picture_approximateBytesUsed(pic) > 0);

            paint_dispose(paint);
            picture_dispose(pic);
            pictureRecorder_dispose(rec);
        }
    }

    /// C++ source: skwasm/canvas.cpp:CanvasNestedPictureAndSaveLayerReplay
    #[test]
    fn test_canvas_nested_picture_and_save_layer_replay() {
        unsafe {
            let child_rec = pictureRecorder_create();
            let cull = [0.0f32, 0.0, 200.0, 200.0];
            let child_canvas = pictureRecorder_beginRecording(child_rec, cull.as_ptr());
            let paint = paint_create(1, 3, 0xFF00FF00u32 as i32, 0, 1.0, 0, 0, 4.0, 0);
            let rect = [0.0f32, 0.0, 40.0, 40.0];
            canvas_drawRect(child_canvas, rect.as_ptr(), paint);
            let child_pic = pictureRecorder_endRecording(child_rec);

            let root_rec = pictureRecorder_create();
            let root_canvas = pictureRecorder_beginRecording(root_rec, cull.as_ptr());
            let layer_paint = paint_create(1, 3, 0x80FFFFFFu32 as i32, 0, 1.0, 0, 0, 4.0, 0);
            canvas_saveLayer(root_canvas, cull.as_ptr(), layer_paint, 0);
            canvas_translate(root_canvas, 30.0, 40.0);
            canvas_drawPicture(root_canvas, child_pic);
            canvas_restore(root_canvas);
            let root_pic = pictureRecorder_endRecording(root_rec);

            let commands_len = with_state(|st| {
                let dl = &st.pictures.get(&root_pic).unwrap().display_list;
                let mut target = Canvas::new(200, 200, Color::TRANSPARENT);
                target.draw_display_list(dl);
                target.commands().len()
            });
            assert_eq!(commands_len, 1);

            paint_dispose(layer_paint);
            paint_dispose(paint);
            picture_dispose(root_pic);
            picture_dispose(child_pic);
            pictureRecorder_dispose(root_rec);
            pictureRecorder_dispose(child_rec);
        }
    }

    /// C++ source: skwasm/image.cpp:ImageFromPixelsVerticesAndAtlas
    #[test]
    fn test_image_from_pixels_vertices_and_atlas() {
        unsafe {
            let data = skData_create(16);
            std::slice::from_raw_parts_mut(skData_getPointer(data), 16).fill(255);
            let img = image_createFromPixels(data, 2, 2, 0, 8);
            assert_eq!(image_getWidth(img), 2);
            assert_eq!(image_getHeight(img), 2);

            let pos = [0.0f32, 0.0, 20.0, 0.0, 10.0, 20.0];
            let indices = [0u16, 1, 2];
            let verts = vertices_create(
                0,
                3,
                pos.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                3,
                indices.as_ptr(),
            );
            assert_ne!(verts, 0);

            let rec = pictureRecorder_create();
            let cull = [0.0f32, 0.0, 100.0, 100.0];
            let canvas = pictureRecorder_beginRecording(rec, cull.as_ptr());
            let paint = paint_create(1, 3, 0xFFFFFFFFu32 as i32, 0, 1.0, 0, 0, 4.0, 0);

            let src = [0.0f32, 0.0, 2.0, 2.0];
            let dst = [10.0f32, 10.0, 50.0, 50.0];
            canvas_drawImageRect(canvas, img, src.as_ptr(), dst.as_ptr(), paint, 1);
            canvas_drawVertices(canvas, verts, 3, paint);

            let rst = [1.0f32, 0.0, 5.0, 5.0];
            let tex_r = [0.0f32, 0.0, 2.0, 2.0];
            canvas_drawAtlas(
                canvas,
                img,
                rst.as_ptr(),
                tex_r.as_ptr(),
                std::ptr::null(),
                1,
                3,
                std::ptr::null(),
                paint,
                1,
            );

            let pic = pictureRecorder_endRecording(rec);
            assert!(picture_approximateBytesUsed(pic) > 0);

            paint_dispose(paint);
            picture_dispose(pic);
            pictureRecorder_dispose(rec);
            vertices_dispose(verts);
            image_dispose(img);
            skData_dispose(data);
        }
    }

    /// C++ source: skwasm/surface.cpp:SurfaceCreateSetSizeRenderAndRasterize
    #[test]
    fn test_surface_create_set_size_render_and_rasterize() {
        unsafe {
            assert_eq!(skwasm_isHeavy(), 1);
            let surf = surface_create();
            surface_setCallbackHandler(surf, 1);
            surface_setResourceCacheLimitBytes(surf, 8 * 1024 * 1024);

            let cb_size = surface_setSize(surf, 800, 600);
            assert!(cb_size >= 1);

            let rec = pictureRecorder_create();
            let cull = [0.0f32, 0.0, 800.0, 600.0];
            let canvas = pictureRecorder_beginRecording(rec, cull.as_ptr());
            canvas_clear(canvas, 0xFF112233);
            let pic = pictureRecorder_endRecording(rec);

            let cb_render = surface_renderPictures(surf, &pic as *const u32, 1);
            assert!(cb_render > cb_size);

            let data = skData_create(4);
            std::slice::from_raw_parts_mut(skData_getPointer(data), 4)
                .copy_from_slice(&[10, 20, 30, 255]);
            let img = image_createFromPixels(data, 1, 1, 0, 4);
            let cb_rast = surface_rasterizeImage(surf, img, 0);
            assert!(cb_rast > cb_render);

            let mut counts = [0u32; 16];
            skwasm_getLiveObjectCounts(counts.as_mut_ptr());
            assert!(counts[2] >= 1);
            assert!(counts[3] >= 1);

            image_dispose(img);
            skData_dispose(data);
            picture_dispose(pic);
            pictureRecorder_dispose(rec);
            surface_destroy(surf);
        }
    }

    /// C++ source: skwasm/shaders.cpp:RuntimeEffectAndUniformDataShaders
    #[test]
    fn test_runtime_effect_and_uniform_data_shaders() {
        unsafe {
            let src = skString_allocate(12);
            std::slice::from_raw_parts_mut(skString_getData(src), 12)
                .copy_from_slice(b"half4 main()");
            let effect = runtimeEffect_create(src);
            assert_eq!(runtimeEffect_getUniformSize(effect), 64);

            let udata = uniformData_create(16);
            let uptr = uniformData_getPointer(udata) as *mut f32;
            std::slice::from_raw_parts_mut(uptr, 4).copy_from_slice(&[0.2, 0.4, 0.8, 1.0]);

            let shader = shader_createRuntimeEffectShader(effect, udata, std::ptr::null(), 0);
            assert_ne!(shader, 0);

            shader_dispose(shader);
            uniformData_dispose(udata);
            runtimeEffect_dispose(effect);
            skString_free(src);
        }
    }

    /// C++ source: skwasm/canvas.cpp:DeferredRecordingArcParagraphAndSharedAtlasCache
    #[test]
    fn test_deferred_recording_arc_paragraph_and_shared_atlas_cache() {
        unsafe {
            let fc = fontCollection_create();
            let ps = paragraphStyle_create();
            let ts = textStyle_create();
            textStyle_setFontSize(ts, 16.0);
            paragraphStyle_setTextStyle(ps, ts);

            let pb = paragraphBuilder_create(ps, fc);
            let utf16: Vec<u16> = "Zero-Clone Paragraph".encode_utf16().collect();
            let s16 = skString16_allocate(utf16.len());
            std::slice::from_raw_parts_mut(skString16_getData(s16), utf16.len())
                .copy_from_slice(&utf16);
            paragraphBuilder_addText(pb, s16);
            let para = paragraphBuilder_build(pb);
            paragraph_layout(para, 240.0);

            let rec = pictureRecorder_create();
            let cull = [0.0f32, 0.0, 300.0, 200.0];
            let canvas = pictureRecorder_beginRecording(rec, cull.as_ptr());

            let paint = paint_create(1, 3, 0xFFFF0000u32 as i32, 0, 2.0, 0, 0, 4.0, 0);
            let rect = [10.0f32, 10.0, 120.0, 80.0];
            canvas_saveLayer(canvas, cull.as_ptr(), paint, 0);
            canvas_drawRect(canvas, rect.as_ptr(), paint);
            canvas_drawParagraph(canvas, para, 12.0, 24.0);
            canvas_restore(canvas);

            // Fix 3A: Recording canvas emits zero immediate DrawCommands during FFI recording.
            let recording_cmds_len =
                with_state(|st| st.canvases.get(&canvas).unwrap().canvas.commands().len());
            assert_eq!(
                recording_cmds_len, 0,
                "RecordingCanvasEntry.canvas should hold 0 immediate DrawCommands during recording"
            );

            // Fix 3B: Paragraph is shared via Arc between SkwasmState and DisplayListBuilder.
            let para_refcount =
                with_state(|st| Arc::strong_count(st.paragraphs.get(&para).unwrap()));
            assert_eq!(
                para_refcount, 2,
                "Arc<Paragraph> should be shared without deep-cloning Paragraph"
            );

            let pic = pictureRecorder_endRecording(rec);

            // Fix 3C: Consecutive steady-state shared_glyph_atlases calls share the exact same Arc,
            // and Phase 8 descriptor_without_pixels() clones zero pixel bytes.
            let (same_alpha, same_color, zero_pixel_bytes, surf_canvas_ok) = with_state(|st| {
                let (a1, c1) = st.shared_glyph_atlases();
                let (a2, c2) = st.shared_glyph_atlases();
                let zero_pixels = a1.pixels.is_empty() && c1.pixels.is_empty();
                let surf_id = st.alloc_handle();
                st.surfaces.insert(
                    surf_id,
                    SurfaceEntry {
                        width: 64,
                        height: 64,
                        callback_pointer: 0,
                        resource_cache_limit: 0,
                        canvas: Canvas::new(1, 1, Color::TRANSPARENT),
                        #[cfg(target_arch = "wasm32")]
                        initializing: false,
                        #[cfg(target_arch = "wasm32")]
                        pending_tasks: Vec::new(),
                        #[cfg(target_arch = "wasm32")]
                        gpu: None,
                    },
                );
                let surf = st.surfaces.get_mut(&surf_id).unwrap();
                surf.canvas.reset(64, 64, Color::TRANSPARENT);
                let ok = surf.canvas.width() == 64 && surf.canvas.height() == 64;
                st.surfaces.remove(&surf_id);
                (
                    Arc::ptr_eq(&a1, &a2),
                    Arc::ptr_eq(&c1, &c2),
                    zero_pixels,
                    ok,
                )
            });
            assert!(
                same_alpha,
                "steady-state alpha GlyphAtlas Arc must be shared"
            );
            assert!(
                same_color,
                "steady-state color GlyphAtlas Arc must be shared"
            );
            assert!(
                zero_pixel_bytes,
                "shared_glyph_atlases must return zero-copy descriptor_without_pixels() with empty pixel Vec"
            );
            assert!(
                surf_canvas_ok,
                "SurfaceEntry reusable canvas must reset cleanly"
            );

            skString16_free(s16);
            paint_dispose(paint);
            picture_dispose(pic);
            pictureRecorder_dispose(rec);
            paragraph_dispose(para);
            textStyle_dispose(ts);
            paragraphStyle_dispose(ps);
            fontCollection_dispose(fc);
        }
    }

    /// C++ source: impeller/renderer/wasm/skwasm_ffi_unittests.cc:PathCacheIdAndClipRRectFastPath
    #[test]
    fn test_phase9_path_entry_cache_id_and_clip_rrect_fast_path() {
        unsafe {
            let path = path_create();
            path_moveTo(path, 10.0, 10.0);
            path_lineTo(path, 50.0, 10.0);
            path_lineTo(path, 50.0, 50.0);
            path_close(path);

            // First build allocates a cache_id; repeated build_path() reuses it without rebuilding.
            let (id1, id2) = with_state(|st| {
                let entry = st.paths.get(&path).unwrap();
                let p1 = entry.build_path();
                let p2 = entry.build_path();
                (p1.cache_id(), p2.cache_id())
            });
            assert_ne!(id1, 0, "built Path must carry a non-zero cache_id");
            assert_eq!(
                id1, id2,
                "repeated build_path() without mutations must preserve cache_id"
            );

            // Mutating the path via FFI invalidates cached_path and yields a fresh cache_id.
            path_lineTo(path, 60.0, 60.0);
            let id3 = with_state(|st| st.paths.get(&path).unwrap().build_path().cache_id());
            assert_ne!(
                id1, id3,
                "mutating path_* FFI must invalidate cached_path and assign a fresh cache_id"
            );

            // canvas_clipRRect routes through clip_rect_with_op without PathBuilder allocations.
            let rec = pictureRecorder_create();
            let cull = [0.0f32, 0.0, 200.0, 200.0];
            let canvas = pictureRecorder_beginRecording(rec, cull.as_ptr());
            let rrect = [
                20.0f32, 30.0, 120.0, 130.0, // rect (LTRB)
                8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, // 4 corner radii
            ];
            canvas_clipRRect(canvas, rrect.as_ptr(), 1);
            let paint = paint_create(1, 3, 0xFF00FF00u32 as i32, 0, 1.0, 0, 0, 4.0, 0);
            canvas_drawPath(canvas, path, paint);
            let pic = pictureRecorder_endRecording(rec);

            let cmd_count = with_state(|st| {
                let pic_entry = st.pictures.get(&pic).unwrap();
                let mut replay_canvas = Canvas::new(200, 200, Color::TRANSPARENT);
                replay_canvas.draw_display_list(&pic_entry.display_list);
                replay_canvas.commands().len()
            });
            assert!(
                cmd_count >= 1,
                "replaying DisplayList with clipRRect + drawPath must emit clipped draw commands"
            );

            paint_dispose(paint);
            picture_dispose(pic);
            pictureRecorder_dispose(rec);
            path_dispose(path);
        }
    }

    /// C++ source: impeller/renderer/wasm/skwasm_ffi_unittests.cc:BezierPreservingPathTransformAndAddPath
    #[test]
    fn test_phase10_bezier_preserving_path_transform_and_add_path() {
        unsafe {
            let p1 = path_create();
            let rrect = [
                10.0f32, 20.0, 110.0, 70.0, // LTRB
                12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
            ];
            path_addRRect(p1, rrect.as_ptr());

            // Affine translation matrix33 (row-major Skia 3x3: [1, 0, 5, 0, 1, 10, 0, 0, 1])
            let m33 = [1.0f32, 0.0, 5.0, 0.0, 1.0, 10.0, 0.0, 0.0, 1.0];
            path_transform(p1, m33.as_ptr());

            let (is_convex, has_cubics) = with_state(|st| {
                let built = st.paths.get(&p1).unwrap().build_path();
                let cubics = built
                    .components
                    .iter()
                    .any(|c| matches!(c, impeller_path::PathComponent::Cubic(_)));
                (built.is_convex(), cubics)
            });
            assert!(
                is_convex,
                "affine path_transform on single addRRect must preserve Convexity::Convex"
            );
            assert!(
                has_cubics,
                "path_transform must preserve cubic Bezier components instead of polyline-flattening"
            );

            let p2 = path_create();
            path_addPath(p2, p1, m33.as_ptr(), 0);
            let p2_has_cubics = with_state(|st| {
                let built = st.paths.get(&p2).unwrap().build_path();
                built
                    .components
                    .iter()
                    .any(|c| matches!(c, impeller_path::PathComponent::Cubic(_)))
            });
            assert!(
                p2_has_cubics,
                "path_addPath with matrix must preserve cubic Bezier components"
            );

            path_dispose(p2);
            path_dispose(p1);
        }
    }

    /// C++ source: impeller/renderer/wasm/skwasm_ffi_unittests.cc:ArcToOvalSingleContourAndSvgArcToRotated
    #[test]
    fn test_phase10b_arc_to_oval_single_contour_and_svg_arc_to_rotated() {
        unsafe {
            // 1. Pill/slider track constructed via moveTo + arcToOval(..., force_move_to=0) + close
            // must produce a single closed contour (not split into multiple open contours).
            let pill = path_create();
            let left_cap = [10.0f32, 20.0, 30.0, 40.0]; // LTRB (radius 10, center (20, 30))
            let right_cap = [90.0f32, 20.0, 110.0, 40.0]; // LTRB (radius 10, center (100, 30))
            path_moveTo(pill, 20.0, 20.0);
            path_lineTo(pill, 100.0, 20.0);
            path_arcToOval(pill, right_cap.as_ptr(), -90.0, 180.0, 0);
            path_lineTo(pill, 20.0, 40.0);
            path_arcToOval(pill, left_cap.as_ptr(), 90.0, 180.0, 0);
            path_close(pill);

            let (contour_count, is_closed, cur_after_close) = with_state(|st| {
                let entry = st.paths.get(&pill).unwrap();
                let polyline = entry.build_path().create_polyline(1.0);
                (
                    polyline.contours.len(),
                    polyline.contours.first().is_some_and(|c| c.is_closed),
                    entry.current_point,
                )
            });
            assert_eq!(
                contour_count, 1,
                "path_arcToOval(..., force_move_to=0) + path_close must keep a single contour"
            );
            assert!(is_closed, "pill contour must be closed");
            assert!(
                (cur_after_close.x - 20.0).abs() <= 1e-4
                    && (cur_after_close.y - 20.0).abs() <= 1e-4,
                "path_close must sync current_point back to contour start"
            );

            // 2. path_arcToRotated must emit cubic Bezier segments rather than a straight chord.
            let svg_path = path_create();
            path_moveTo(svg_path, 0.0, 0.0);
            path_arcToRotated(svg_path, 30.0, 20.0, 15.0, 0, 0, 40.0, 20.0);
            let (has_cubics, end_pt) = with_state(|st| {
                let entry = st.paths.get(&svg_path).unwrap();
                let built = entry.build_path();
                let cubics = built
                    .components
                    .iter()
                    .any(|c| matches!(c, impeller_path::PathComponent::Cubic(_)));
                (cubics, entry.current_point)
            });
            assert!(
                has_cubics,
                "path_arcToRotated must emit cubic Bezier segments via svg_arc_to"
            );
            assert!(
                (end_pt.x - 40.0).abs() <= 1e-4 && (end_pt.y - 20.0).abs() <= 1e-4,
                "path_arcToRotated must update current_point to the arc endpoint"
            );

            path_dispose(svg_path);
            path_dispose(pill);
        }
    }

    /// C++ source: impeller/renderer/wasm/skwasm_ffi_unittests.cc:PathCombineStencilClipSweepConicalAndImageNine
    #[test]
    fn test_phase12a_path_combine_stencil_clip_sweep_conical_and_image_nine() {
        unsafe {
            // 1. Boolean path_combine (op=0 Difference vs op=1 Intersect):
            let p_a = path_create();
            let rect_a = [0.0f32, 0.0, 100.0, 100.0];
            path_addRect(p_a, rect_a.as_ptr());

            let p_b = path_create();
            let rect_b = [40.0f32, 40.0, 140.0, 140.0];
            path_addRect(p_b, rect_b.as_ptr());

            let p_diff = path_combine(0, p_a, p_b);
            let p_inter = path_combine(1, p_a, p_b);

            let mut diff_bounds = [0.0f32; 4];
            let mut inter_bounds = [0.0f32; 4];
            path_getBounds(p_diff, diff_bounds.as_mut_ptr());
            path_getBounds(p_inter, inter_bounds.as_mut_ptr());

            // Difference of [0..100] \ [40..140] keeps [0..100] outer L-shape;
            // Intersect yields [40..100, 40..100].
            assert!(
                (diff_bounds[0] - 0.0).abs() <= 1e-3 && (diff_bounds[2] - 100.0).abs() <= 1e-3,
                "Difference bounds must span [0..100]: {:?}",
                diff_bounds
            );
            assert!(
                (inter_bounds[0] - 40.0).abs() <= 1e-3
                    && (inter_bounds[1] - 40.0).abs() <= 1e-3
                    && (inter_bounds[2] - 100.0).abs() <= 1e-3
                    && (inter_bounds[3] - 100.0).abs() <= 1e-3,
                "Intersect bounds must be [40, 40, 100, 100]: {:?}",
                inter_bounds
            );
            assert_eq!(path_contains(p_diff, 20.0, 20.0), 1);
            assert_eq!(path_contains(p_diff, 60.0, 60.0), 0);
            assert_eq!(path_contains(p_inter, 60.0, 60.0), 1);
            assert_eq!(path_contains(p_inter, 20.0, 20.0), 0);

            // 2. Stencil clipRRect + SweepGradient + ConicalGradient + drawImageNine:
            let rec = pictureRecorder_create();
            let cull = [0.0f32, 0.0, 200.0, 200.0];
            let canvas = pictureRecorder_beginRecording(rec, cull.as_ptr());

            canvas_save(canvas);
            let rounded_rrect = [
                10.0f32, 10.0, 190.0, 190.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
            ];
            canvas_clipRRect(canvas, rounded_rrect.as_ptr(), 1);

            let colors = [0xFFFF0000u32, 0xFF0000FFu32];
            let stops = [0.0f32, 1.0f32];
            let sweep_sh = shader_createSweepGradient(
                100.0,
                100.0,
                colors.as_ptr(),
                stops.as_ptr(),
                2,
                0,
                0.0,
                360.0,
                std::ptr::null(),
            );
            let sweep_paint = paint_create(1, 3, 0xFFFFFFFFu32 as i32, 0, 1.0, 0, 0, 4.0, 0);
            paint_setShader(sweep_paint, sweep_sh);
            let draw_r = [20.0f32, 20.0, 80.0, 80.0];
            canvas_drawRect(canvas, draw_r.as_ptr(), sweep_paint);

            let conical_pts = [80.0f32, 80.0, 120.0, 120.0];
            let conical_sh = shader_createConicalGradient(
                conical_pts.as_ptr(),
                4.0,
                40.0,
                colors.as_ptr(),
                stops.as_ptr(),
                2,
                0,
                std::ptr::null(),
            );
            let conical_paint = paint_create(1, 3, 0xFFFFFFFFu32 as i32, 0, 1.0, 0, 0, 4.0, 0);
            paint_setShader(conical_paint, conical_sh);
            canvas_drawRect(canvas, draw_r.as_ptr(), conical_paint);

            let img_bytes = 30 * 30 * 4;
            let data = skData_create(img_bytes);
            std::slice::from_raw_parts_mut(skData_getPointer(data), img_bytes).fill(255);
            let img = image_createFromPixels(data, 30, 30, 0, 30 * 4);
            let center_ir = [10i32, 10, 20, 20];
            let dst_nine = [20.0f32, 90.0, 120.0, 170.0];
            let img_paint = paint_create(1, 3, 0xFFFFFFFFu32 as i32, 0, 1.0, 0, 0, 4.0, 0);
            canvas_drawImageNine(
                canvas,
                img,
                center_ir.as_ptr(),
                dst_nine.as_ptr(),
                img_paint,
                1,
            );
            canvas_restore(canvas);

            let pic = pictureRecorder_endRecording(rec);
            let (push_clips, pop_clips, sweeps, conicals, tex_fills) = with_state(|st| {
                let dl = &st.pictures.get(&pic).unwrap().display_list;
                let mut target = Canvas::new(200, 200, Color::TRANSPARENT);
                target.draw_display_list(dl);
                let mut push_c = 0;
                let mut pop_c = 0;
                let mut sw = 0;
                let mut cn = 0;
                let mut tf = 0;
                for cmd in target.commands() {
                    match cmd {
                        impeller_wgpu::DrawCommand::PushClip { .. } => push_c += 1,
                        impeller_wgpu::DrawCommand::PopClip { .. } => pop_c += 1,
                        impeller_wgpu::DrawCommand::SweepGradientFill { .. } => sw += 1,
                        impeller_wgpu::DrawCommand::ConicalGradientFill { .. } => cn += 1,
                        impeller_wgpu::DrawCommand::TextureFill(_) => tf += 1,
                        _ => {}
                    }
                }
                (push_c, pop_c, sw, cn, tf)
            });
            assert_eq!(push_clips, 1);
            assert_eq!(pop_clips, 1);
            assert_eq!(sweeps, 1);
            assert_eq!(conicals, 1);
            assert_eq!(tex_fills, 9);

            image_dispose(img);
            skData_dispose(data);
            paint_dispose(img_paint);
            paint_dispose(conical_paint);
            paint_dispose(sweep_paint);
            shader_dispose(conical_sh);
            shader_dispose(sweep_sh);
            picture_dispose(pic);
            pictureRecorder_dispose(rec);
            path_dispose(p_inter);
            path_dispose(p_diff);
            path_dispose(p_b);
            path_dispose(p_a);
        }
    }

    /// C++ source: dl_op_spy_unittests.cc:CanvasSaveLayerAndImageColorFilterFFI
    #[test]
    fn test_phase12b_ffi_save_layer_and_image_color_filters() {
        unsafe {
            let rec = pictureRecorder_create();
            let cull_rect = [0.0f32, 0.0, 200.0, 200.0];
            let canvas = pictureRecorder_beginRecording(rec, cull_rect.as_ptr());

            // 1. Multi-child saveLayer with blur image filter + color filter:
            let blur_if = imageFilter_createBlur(4.0, 4.0, 3);
            let mode_cf = colorFilter_createMode(0xFF00FF00u32 as i32, 5); // SrcIn green
            let layer_paint = paint_create(1, 3, 0x80FFFFFFu32 as i32, 0, 1.0, 0, 0, 4.0, 0);
            paint_setImageFilter(layer_paint, blur_if);
            paint_setColorFilter(layer_paint, mode_cf);

            let layer_bounds = [20.0f32, 20.0, 120.0, 120.0];
            canvas_saveLayer(canvas, layer_bounds.as_ptr(), layer_paint, 0);

            let child_paint = paint_create(1, 3, 0xFFFF0000u32 as i32, 0, 1.0, 0, 0, 4.0, 0);
            let r1 = [30.0f32, 30.0, 80.0, 80.0];
            let r2 = [60.0f32, 50.0, 110.0, 100.0];
            canvas_drawRect(canvas, r1.as_ptr(), child_paint);
            canvas_drawRect(canvas, r2.as_ptr(), child_paint);
            canvas_restore(canvas);

            // 2. Standalone drawRect with paint imageFilter:
            let standalone_paint = paint_create(1, 3, 0xFFFFFFFFu32 as i32, 0, 1.0, 0, 0, 4.0, 0);
            paint_setImageFilter(standalone_paint, blur_if);
            let r3 = [130.0f32, 30.0, 180.0, 80.0];
            canvas_drawRect(canvas, r3.as_ptr(), standalone_paint);

            let pic = pictureRecorder_endRecording(rec);
            let (save_layer_cmds, blur_layers, cf_layers) = with_state(|st| {
                let dl = &st.pictures.get(&pic).unwrap().display_list;
                let mut target = Canvas::new(200, 200, Color::TRANSPARENT);
                target.draw_display_list(dl);
                let mut sl = 0;
                let mut bl = 0;
                let mut cf = 0;
                for cmd in target.commands() {
                    if let impeller_wgpu::DrawCommand::SaveLayer {
                        image_filter,
                        color_filter,
                        ..
                    } = cmd
                    {
                        sl += 1;
                        if matches!(
                            image_filter,
                            Some(impeller_wgpu::LayerImageFilter::Blur { .. })
                        ) {
                            bl += 1;
                        }
                        if matches!(
                            color_filter,
                            Some(impeller_wgpu::LayerColorFilter::Mode { .. })
                        ) {
                            cf += 1;
                        }
                    }
                }
                (sl, bl, cf)
            });

            assert_eq!(save_layer_cmds, 2);
            assert_eq!(blur_layers, 2);
            assert_eq!(cf_layers, 1);

            paint_dispose(standalone_paint);
            paint_dispose(child_paint);
            paint_dispose(layer_paint);
            colorFilter_dispose(mode_cf);
            imageFilter_dispose(blur_if);
            picture_dispose(pic);
            pictureRecorder_dispose(rec);
        }
    }

    /// C++ source: skwasm/canvas.cpp:CanvasSaveLayerBackdropFilterAndEmptyLayer
    #[test]
    fn test_chunk2_ffi_save_layer_backdrop_preserves_paint_image_filter_and_empty_child() {
        unsafe {
            let rec = pictureRecorder_create();
            let cull_rect = [0.0f32, 0.0, 200.0, 200.0];
            let canvas = pictureRecorder_beginRecording(rec, cull_rect.as_ptr());

            let backdrop_blur = imageFilter_createBlur(8.0, 8.0, 0); // Clamp
            let paint_blur = imageFilter_createBlur(2.0, 2.0, 3); // Decal

            // 1. Empty-child saveLayer with backdrop_filter_handle != 0 and paint_handle == 0:
            let empty_bounds = [10.0f32, 10.0, 90.0, 90.0];
            canvas_saveLayer(canvas, empty_bounds.as_ptr(), 0, backdrop_blur);
            canvas_restore(canvas);

            // 2. Populated saveLayer where paint has its own image_filter AND backdrop_filter_handle != 0:
            //    Both filters must be preserved independently without overwriting paint.image_filter!
            let dual_paint = paint_create(1, 15, 0xCCFFFFFFu32 as i32, 0, 1.0, 0, 0, 4.0, 0); // BlendMode::Overlay (15)
            paint_setImageFilter(dual_paint, paint_blur);
            let dual_bounds = [100.0f32, 10.0, 190.0, 90.0];
            canvas_saveLayer(canvas, dual_bounds.as_ptr(), dual_paint, backdrop_blur);

            let dodge_paint = paint_create(1, 18, 0xFF8040C0u32 as i32, 0, 1.0, 0, 0, 4.0, 0); // BlendMode::ColorDodge (18)
            let lum_paint = paint_create(1, 28, 0xFF40C080u32 as i32, 0, 1.0, 0, 0, 4.0, 0); // BlendMode::Luminosity (28)
            let r1 = [110.0f32, 20.0, 150.0, 60.0];
            let r2 = [130.0f32, 40.0, 180.0, 80.0];
            canvas_drawRect(canvas, r1.as_ptr(), dodge_paint);
            canvas_drawRect(canvas, r2.as_ptr(), lum_paint);
            canvas_restore(canvas);

            let pic = pictureRecorder_endRecording(rec);
            let (empty_bf_found, dual_bf_found, child_blends) = with_state(|st| {
                let dl = &st.pictures.get(&pic).unwrap().display_list;
                let mut target = Canvas::new(200, 200, Color::TRANSPARENT);
                target.draw_display_list(dl);
                let mut empty_ok = false;
                let mut dual_ok = false;
                let mut blends = Vec::new();
                for cmd in target.commands() {
                    if let impeller_wgpu::DrawCommand::SaveLayer {
                        commands,
                        blend_mode,
                        image_filter,
                        backdrop_filter,
                        ..
                    } = cmd
                    {
                        if commands.is_empty()
                            && image_filter.is_none()
                            && matches!(
                                backdrop_filter,
                                Some(impeller_wgpu::LayerImageFilter::Blur { sigma_x, .. })
                                    if (*sigma_x - 8.0).abs() < 1e-4
                            )
                        {
                            empty_ok = true;
                        }
                        if *blend_mode == BlendMode::Overlay
                            && matches!(
                                image_filter,
                                Some(impeller_wgpu::LayerImageFilter::Blur { sigma_x, .. })
                                    if (*sigma_x - 2.0).abs() < 1e-4
                            )
                            && matches!(
                                backdrop_filter,
                                Some(impeller_wgpu::LayerImageFilter::Blur { sigma_x, .. })
                                    if (*sigma_x - 8.0).abs() < 1e-4
                            )
                        {
                            dual_ok = true;
                            for sub in commands {
                                if let impeller_wgpu::DrawCommand::SolidFill {
                                    blend_mode: bm,
                                    ..
                                } = sub
                                {
                                    blends.push(*bm);
                                }
                            }
                        }
                    }
                }
                (empty_ok, dual_ok, blends)
            });

            assert!(
                empty_bf_found,
                "empty-child FFI canvas_saveLayer with backdrop_filter must emit SaveLayer"
            );
            assert!(
                dual_bf_found,
                "FFI canvas_saveLayer must preserve both paint.image_filter (sigma=2) and backdrop_filter (sigma=8)"
            );
            assert_eq!(
                child_blends,
                vec![BlendMode::ColorDodge, BlendMode::Luminosity]
            );

            paint_dispose(lum_paint);
            paint_dispose(dodge_paint);
            paint_dispose(dual_paint);
            imageFilter_dispose(paint_blur);
            imageFilter_dispose(backdrop_blur);
            picture_dispose(pic);
            pictureRecorder_dispose(rec);
        }
    }
}
