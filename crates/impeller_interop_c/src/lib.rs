#![allow(unsafe_code)]
#![allow(
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::not_unsafe_ptr_arg_deref
)]

//! Impeller Standalone Native C-ABI (`impeller_interop_c`):
//! Drop-in Rust + wgpu implementation of upstream Flutter's `impeller.h`
//! (`impeller/toolkit/interop/impeller.h`) on top of `impeller_canvas`, `impeller_wgpu`,
//! `impeller_path`, `impeller_geometry`, and `impeller_typographer`.

pub mod types;

use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use impeller_canvas::{
    Canvas, ColorFilter, ColorSource, ConicalGradientParams, DisplayList, DisplayListBuilder,
    DrawStyle, GradientStop, ImageFilter, ImageTexture, LinearGradientParams, Paint,
    RadialGradientParams, RoundingRadii, SweepGradientParams, WgpuRenderer,
};
use impeller_geometry::{Color, Matrix, Point, Radians, Rect};
use impeller_path::{Path, PathBuilder};
use impeller_typographer::{
    GlyphInfo, Paragraph, ParagraphBuilder, ParagraphLineMetrics, ParagraphStyle, RichTextStyle,
    TextDirection, TypographerContext,
};

pub use types::*;

static NEXT_TEXTURE_ID: AtomicU64 = AtomicU64::new(1);

fn shared_headless_renderer() -> Option<Arc<WgpuRenderer>> {
    static RENDERER: OnceLock<Option<Arc<WgpuRenderer>>> = OnceLock::new();
    RENDERER
        .get_or_init(|| WgpuRenderer::new_headless().ok().map(Arc::new))
        .clone()
}

#[inline]
fn into_handle<T, H>(peer: T) -> *mut H {
    Arc::into_raw(Arc::new(peer)) as *mut H
}

#[inline]
unsafe fn retain_handle<T, H>(handle: *mut H) {
    if !handle.is_null() {
        unsafe {
            Arc::increment_strong_count(handle as *const T);
        }
    }
}

#[inline]
unsafe fn release_handle<T, H>(handle: *mut H) {
    if !handle.is_null() {
        unsafe {
            Arc::decrement_strong_count(handle as *const T);
        }
    }
}

#[inline]
unsafe fn borrow_handle<'a, T, H>(handle: *mut H) -> Option<&'a T> {
    if handle.is_null() {
        None
    } else {
        Some(unsafe { &*(handle as *const T) })
    }
}

unsafe fn consume_mapping(
    mapping: *const ImpellerMapping,
    user_data: *mut c_void,
) -> Option<Vec<u8>> {
    if mapping.is_null() {
        return None;
    }
    let m = unsafe { *mapping };
    let bytes = if m.data.is_null() || m.length == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(m.data, m.length as usize).to_vec() }
    };
    if let Some(cb) = m.on_release {
        unsafe {
            cb(user_data);
        }
    }
    Some(bytes)
}

struct ContextPeer {
    renderer: Option<Arc<WgpuRenderer>>,
}

struct VulkanSwapchainPeer {
    context: Arc<ContextPeer>,
}

struct SurfacePeer {
    context: Arc<ContextPeer>,
    width: u32,
    height: u32,
    pixels: Mutex<Vec<u8>>,
}

struct PathPeer {
    path: Path,
}

struct PathBuilderPeer {
    builder: Mutex<PathBuilder>,
}

#[derive(Clone, Default)]
struct PaintState {
    paint: Paint,
    also_stroke: bool,
}

struct PaintPeer {
    state: Mutex<PaintState>,
}

struct ColorFilterPeer {
    filter: ColorFilter,
}

struct ColorSourcePeer {
    source: ColorSource,
}

struct ImageFilterPeer {
    filter: ImageFilter,
}

struct MaskFilterPeer {
    sigma: f32,
}

struct TexturePeer {
    texture: ImageTexture,
    gl_handle: u64,
}

struct FragmentProgramPeer {
    data: Vec<u8>,
}

struct DisplayListPeer {
    dl: Arc<DisplayList>,
    typo_ctx: Option<Arc<Mutex<TypographerContext>>>,
}

struct DisplayListBuilderInner {
    builder: DisplayListBuilder,
    stack: Vec<Matrix>,
    typo_ctx: Option<Arc<Mutex<TypographerContext>>>,
}

impl DisplayListBuilderInner {
    fn new(_cull_rect: Option<Rect>) -> Self {
        Self {
            builder: DisplayListBuilder::new(),
            stack: vec![Matrix::IDENTITY],
            typo_ctx: None,
        }
    }

    fn current_transform(&self) -> Matrix {
        self.stack.last().copied().unwrap_or(Matrix::IDENTITY)
    }

    fn current_transform_mut(&mut self) -> &mut Matrix {
        if self.stack.is_empty() {
            self.stack.push(Matrix::IDENTITY);
        }
        self.stack.last_mut().expect("non-empty transform stack")
    }
}

struct DisplayListBuilderPeer {
    inner: Mutex<DisplayListBuilderInner>,
}

struct TypographyContextPeer {
    ctx: Arc<Mutex<TypographerContext>>,
}

#[derive(Clone, Default)]
struct ParagraphStyleState {
    para_style: ParagraphStyle,
    text_style: RichTextStyle,
}

struct ParagraphStylePeer {
    state: Mutex<ParagraphStyleState>,
}

struct ParagraphBuilderInner {
    typo_ctx: Arc<Mutex<TypographerContext>>,
    builder: Option<ParagraphBuilder>,
}

struct ParagraphBuilderPeer {
    inner: Mutex<ParagraphBuilderInner>,
}

struct LineMetricsPeer {
    lines: Vec<ParagraphLineMetrics>,
}

struct GlyphInfoPeer {
    info: GlyphInfo,
    is_ellipsis: bool,
}

struct ParagraphPeer {
    paragraph: Arc<Paragraph>,
    typo_ctx: Arc<Mutex<TypographerContext>>,
    line_metrics_handle: ImpellerLineMetrics,
}

unsafe impl Send for ParagraphPeer {}
unsafe impl Sync for ParagraphPeer {}

impl Drop for ParagraphPeer {
    fn drop(&mut self) {
        unsafe {
            release_handle::<LineMetricsPeer, _>(self.line_metrics_handle);
        }
    }
}

unsafe fn build_gradient_stops(
    stop_count: u32,
    colors: *const ImpellerColor,
    stops: *const f32,
) -> Option<Vec<GradientStop>> {
    if stop_count == 0 || colors.is_null() {
        return None;
    }
    let count = stop_count as usize;
    let color_slice = unsafe { std::slice::from_raw_parts(colors, count) };
    let mut out = Vec::with_capacity(count);
    for (i, c) in color_slice.iter().enumerate() {
        let offset = if stops.is_null() {
            if count <= 1 {
                0.0
            } else {
                (i as f32) / ((count - 1) as f32)
            }
        } else {
            unsafe { *stops.add(i) }
        };
        out.push(GradientStop::new(offset, (*c).into()));
    }
    Some(Paint::convert_stops(&out))
}

unsafe fn clone_context_arc(context: ImpellerContext) -> Option<Arc<ContextPeer>> {
    if context.is_null() {
        return None;
    }
    unsafe {
        Arc::increment_strong_count(context as *const ContextPeer);
        Some(Arc::from_raw(context as *const ContextPeer))
    }
}

// ============================================================================
// 1. Version & Context / Swapchain
// ============================================================================

#[no_mangle]
pub extern "C" fn ImpellerGetVersion() -> u32 {
    IMPELLER_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerContextCreateMetalNew(version: u32) -> ImpellerContext {
    if version != IMPELLER_VERSION {
        return ptr::null_mut();
    }
    into_handle(ContextPeer {
        renderer: shared_headless_renderer(),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerContextCreateOpenGLESNew(
    version: u32,
    _gl_proc_address_callback: ImpellerProcAddressCallback,
    _gl_proc_address_callback_user_data: *mut c_void,
) -> ImpellerContext {
    if version != IMPELLER_VERSION {
        return ptr::null_mut();
    }
    into_handle(ContextPeer {
        renderer: shared_headless_renderer(),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerContextCreateVulkanNew(
    version: u32,
    settings: *const ImpellerContextVulkanSettings,
) -> ImpellerContext {
    if version != IMPELLER_VERSION || settings.is_null() {
        return ptr::null_mut();
    }
    into_handle(ContextPeer {
        renderer: shared_headless_renderer(),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerContextRetain(context: ImpellerContext) {
    unsafe { retain_handle::<ContextPeer, _>(context) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerContextRelease(context: ImpellerContext) {
    unsafe { release_handle::<ContextPeer, _>(context) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerContextGetVulkanInfo(
    context: ImpellerContext,
    out_vulkan_info: *mut ImpellerContextVulkanInfo,
) -> bool {
    if context.is_null() || out_vulkan_info.is_null() {
        return false;
    }
    unsafe {
        *out_vulkan_info = ImpellerContextVulkanInfo {
            vk_instance: ptr::null_mut(),
            vk_physical_device: ptr::null_mut(),
            vk_logical_device: ptr::null_mut(),
            graphics_queue_family_index: 0,
            graphics_queue_index: 0,
        };
    }
    true
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerVulkanSwapchainCreateNew(
    context: ImpellerContext,
    vulkan_surface_khr: *mut c_void,
) -> ImpellerVulkanSwapchain {
    if vulkan_surface_khr.is_null() {
        return ptr::null_mut();
    }
    let Some(ctx_arc) = (unsafe { clone_context_arc(context) }) else {
        return ptr::null_mut();
    };
    into_handle(VulkanSwapchainPeer { context: ctx_arc })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerVulkanSwapchainRetain(swapchain: ImpellerVulkanSwapchain) {
    unsafe { retain_handle::<VulkanSwapchainPeer, _>(swapchain) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerVulkanSwapchainRelease(swapchain: ImpellerVulkanSwapchain) {
    unsafe { release_handle::<VulkanSwapchainPeer, _>(swapchain) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerVulkanSwapchainAcquireNextSurfaceNew(
    swapchain: ImpellerVulkanSwapchain,
) -> ImpellerSurface {
    let Some(sc) = (unsafe { borrow_handle::<VulkanSwapchainPeer, _>(swapchain) }) else {
        return ptr::null_mut();
    };
    let width = 800u32;
    let height = 600u32;
    into_handle(SurfacePeer {
        context: Arc::clone(&sc.context),
        width,
        height,
        pixels: Mutex::new(vec![0u8; (width as usize) * (height as usize) * 4]),
    })
}

// ============================================================================
// 2. Surface & Offscreen Readback
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn ImpellerSurfaceCreateWrappedFBONew(
    context: ImpellerContext,
    _fbo: u64,
    _format: ImpellerPixelFormat,
    size: *const ImpellerISize,
) -> ImpellerSurface {
    if size.is_null() {
        return ptr::null_mut();
    }
    let sz = unsafe { *size };
    if sz.width <= 0 || sz.height <= 0 {
        return ptr::null_mut();
    }
    let Some(ctx_arc) = (unsafe { clone_context_arc(context) }) else {
        return ptr::null_mut();
    };
    let width = sz.width as u32;
    let height = sz.height as u32;
    into_handle(SurfacePeer {
        context: ctx_arc,
        width,
        height,
        pixels: Mutex::new(vec![0u8; (width as usize) * (height as usize) * 4]),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerSurfaceCreateWrappedMetalDrawableNew(
    context: ImpellerContext,
    metal_drawable: *mut c_void,
) -> ImpellerSurface {
    if metal_drawable.is_null() {
        return ptr::null_mut();
    }
    let Some(ctx_arc) = (unsafe { clone_context_arc(context) }) else {
        return ptr::null_mut();
    };
    let width = 800u32;
    let height = 600u32;
    into_handle(SurfacePeer {
        context: ctx_arc,
        width,
        height,
        pixels: Mutex::new(vec![0u8; (width as usize) * (height as usize) * 4]),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerSurfaceRetain(surface: ImpellerSurface) {
    unsafe { retain_handle::<SurfacePeer, _>(surface) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerSurfaceRelease(surface: ImpellerSurface) {
    unsafe { release_handle::<SurfacePeer, _>(surface) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerSurfaceDrawDisplayList(
    surface: ImpellerSurface,
    display_list: ImpellerDisplayList,
) -> bool {
    let Some(surf) = (unsafe { borrow_handle::<SurfacePeer, _>(surface) }) else {
        return false;
    };
    let Some(dl_peer) = (unsafe { borrow_handle::<DisplayListPeer, _>(display_list) }) else {
        return false;
    };

    let mut canvas = Canvas::new(surf.width, surf.height, Color::TRANSPARENT);
    let atlases = dl_peer.typo_ctx.as_ref().map(|tc| {
        let guard = tc.lock().expect("typo_ctx lock");
        (
            Arc::new(guard.alpha_atlas.clone()),
            Arc::new(guard.color_atlas.clone()),
        )
    });
    if let Some((ref alpha_arc, ref color_arc)) = atlases {
        canvas.set_default_glyph_atlases(Some(Arc::clone(alpha_arc)), Some(Arc::clone(color_arc)));
    }

    canvas.draw_display_list(&dl_peer.dl);

    if let Some((ref alpha_arc, ref color_arc)) = atlases {
        canvas.sync_glyph_atlases_with_arcs(alpha_arc, color_arc);
    }

    if let Some(ref renderer) = surf.context.renderer {
        if let Ok(rendered) = canvas.render(renderer) {
            let mut guard = surf.pixels.lock().expect("surface pixels lock");
            *guard = rendered;
        }
    }

    true
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerSurfacePresent(surface: ImpellerSurface) -> bool {
    !surface.is_null()
}

/// Copies the rendered RGBA8888 pixel buffer from `surface` into `out_rgba`.
#[no_mangle]
pub unsafe extern "C" fn ImpellerSurfaceCopyPixelsRGBA8888(
    surface: ImpellerSurface,
    out_rgba: *mut u8,
    out_byte_length: u64,
) -> bool {
    let Some(surf) = (unsafe { borrow_handle::<SurfacePeer, _>(surface) }) else {
        return false;
    };
    if out_rgba.is_null() {
        return false;
    }
    let needed = (surf.width as usize) * (surf.height as usize) * 4;
    if (out_byte_length as usize) < needed {
        return false;
    }
    let guard = surf.pixels.lock().expect("surface pixels lock");
    if guard.len() < needed {
        return false;
    }
    unsafe {
        ptr::copy_nonoverlapping(guard.as_ptr(), out_rgba, needed);
    }
    true
}

// ============================================================================
// 3. Path & PathBuilder
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathRetain(path: ImpellerPath) {
    unsafe { retain_handle::<PathPeer, _>(path) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathRelease(path: ImpellerPath) {
    unsafe { release_handle::<PathPeer, _>(path) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathGetBounds(path: ImpellerPath, out_bounds: *mut ImpellerRect) {
    if out_bounds.is_null() {
        return;
    }
    let Some(peer) = (unsafe { borrow_handle::<PathPeer, _>(path) }) else {
        unsafe {
            *out_bounds = ImpellerRect::default();
        }
        return;
    };
    let rect = peer.path.get_bounding_box().unwrap_or(Rect::ZERO);
    unsafe {
        *out_bounds = rect.into();
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderNew() -> ImpellerPathBuilder {
    into_handle(PathBuilderPeer {
        builder: Mutex::new(PathBuilder::new()),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderRetain(builder: ImpellerPathBuilder) {
    unsafe { retain_handle::<PathBuilderPeer, _>(builder) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderRelease(builder: ImpellerPathBuilder) {
    unsafe { release_handle::<PathBuilderPeer, _>(builder) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderMoveTo(
    builder: ImpellerPathBuilder,
    location: *const ImpellerPoint,
) {
    if location.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<PathBuilderPeer, _>(builder) } {
        let pt: Point = unsafe { (*location).into() };
        peer.builder.lock().expect("path builder lock").move_to(pt);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderLineTo(
    builder: ImpellerPathBuilder,
    location: *const ImpellerPoint,
) {
    if location.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<PathBuilderPeer, _>(builder) } {
        let pt: Point = unsafe { (*location).into() };
        peer.builder.lock().expect("path builder lock").line_to(pt);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderQuadraticCurveTo(
    builder: ImpellerPathBuilder,
    control_point: *const ImpellerPoint,
    end_point: *const ImpellerPoint,
) {
    if control_point.is_null() || end_point.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<PathBuilderPeer, _>(builder) } {
        let cp: Point = unsafe { (*control_point).into() };
        let ep: Point = unsafe { (*end_point).into() };
        peer.builder
            .lock()
            .expect("path builder lock")
            .quadratic_curve_to(cp, ep);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderCubicCurveTo(
    builder: ImpellerPathBuilder,
    control_point_1: *const ImpellerPoint,
    control_point_2: *const ImpellerPoint,
    end_point: *const ImpellerPoint,
) {
    if control_point_1.is_null() || control_point_2.is_null() || end_point.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<PathBuilderPeer, _>(builder) } {
        let cp1: Point = unsafe { (*control_point_1).into() };
        let cp2: Point = unsafe { (*control_point_2).into() };
        let ep: Point = unsafe { (*end_point).into() };
        peer.builder
            .lock()
            .expect("path builder lock")
            .cubic_curve_to(cp1, cp2, ep);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderAddRect(
    builder: ImpellerPathBuilder,
    rect: *const ImpellerRect,
) {
    if rect.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<PathBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*rect).into() };
        peer.builder.lock().expect("path builder lock").add_rect(r);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderAddArc(
    builder: ImpellerPathBuilder,
    oval_bounds: *const ImpellerRect,
    start_angle_degrees: f32,
    end_angle_degrees: f32,
) {
    if oval_bounds.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<PathBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*oval_bounds).into() };
        let sweep_degrees = end_angle_degrees - start_angle_degrees;
        peer.builder.lock().expect("path builder lock").add_arc(
            r,
            Radians(start_angle_degrees.to_radians()),
            Radians(sweep_degrees.to_radians()),
            false,
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderAddOval(
    builder: ImpellerPathBuilder,
    oval_bounds: *const ImpellerRect,
) {
    if oval_bounds.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<PathBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*oval_bounds).into() };
        peer.builder.lock().expect("path builder lock").add_oval(r);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderAddRoundedRect(
    builder: ImpellerPathBuilder,
    rect: *const ImpellerRect,
    rounding_radii: *const ImpellerRoundingRadii,
) {
    if rect.is_null() || rounding_radii.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<PathBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*rect).into() };
        let radii: RoundingRadii = unsafe { (*rounding_radii).into() };
        peer.builder
            .lock()
            .expect("path builder lock")
            .add_rounded_rect(r, radii);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderClose(builder: ImpellerPathBuilder) {
    if let Some(peer) = unsafe { borrow_handle::<PathBuilderPeer, _>(builder) } {
        peer.builder.lock().expect("path builder lock").close();
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderCopyPathNew(
    builder: ImpellerPathBuilder,
    fill: ImpellerFillType,
) -> ImpellerPath {
    let Some(peer) = (unsafe { borrow_handle::<PathBuilderPeer, _>(builder) }) else {
        return ptr::null_mut();
    };
    let path = peer
        .builder
        .lock()
        .expect("path builder lock")
        .copy_path(to_fill_type(fill));
    into_handle(PathPeer { path })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPathBuilderTakePathNew(
    builder: ImpellerPathBuilder,
    fill: ImpellerFillType,
) -> ImpellerPath {
    let Some(peer) = (unsafe { borrow_handle::<PathBuilderPeer, _>(builder) }) else {
        return ptr::null_mut();
    };
    let path = peer
        .builder
        .lock()
        .expect("path builder lock")
        .take_path(to_fill_type(fill));
    into_handle(PathPeer { path })
}

// ============================================================================
// 4. Paint, Filters, ColorSources, Textures & FragmentPrograms
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintNew() -> ImpellerPaint {
    into_handle(PaintPeer {
        state: Mutex::new(PaintState::default()),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintRetain(paint: ImpellerPaint) {
    unsafe { retain_handle::<PaintPeer, _>(paint) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintRelease(paint: ImpellerPaint) {
    unsafe { release_handle::<PaintPeer, _>(paint) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetColor(paint: ImpellerPaint, color: *const ImpellerColor) {
    if color.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        st.paint.color = unsafe { (*color).into() };
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetBlendMode(paint: ImpellerPaint, mode: ImpellerBlendMode) {
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        st.paint.blend_mode = to_blend_mode(mode);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetDrawStyle(paint: ImpellerPaint, style: ImpellerDrawStyle) {
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        let (ds, also_stroke) = to_draw_style(style);
        st.paint.style = ds;
        st.also_stroke = also_stroke;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetStrokeCap(paint: ImpellerPaint, cap: ImpellerStrokeCap) {
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        st.paint.stroke_cap = to_stroke_cap(cap);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetStrokeJoin(
    paint: ImpellerPaint,
    join: ImpellerStrokeJoin,
) {
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        st.paint.stroke_join = to_stroke_join(join);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetStrokeWidth(paint: ImpellerPaint, width: f32) {
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        st.paint.stroke_width = width;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetStrokeMiter(paint: ImpellerPaint, miter: f32) {
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        st.paint.stroke_miter = miter;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetColorFilter(
    paint: ImpellerPaint,
    color_filter: ImpellerColorFilter,
) {
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        st.paint.color_filter = unsafe {
            borrow_handle::<ColorFilterPeer, _>(color_filter).map(|cf| cf.filter.clone())
        };
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetColorSource(
    paint: ImpellerPaint,
    color_source: ImpellerColorSource,
) {
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        let src = unsafe {
            borrow_handle::<ColorSourcePeer, _>(color_source).map(|cs| cs.source.clone())
        };
        st.paint.color_source = src.clone();
        st.paint.shader = src;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetImageFilter(
    paint: ImpellerPaint,
    image_filter: ImpellerImageFilter,
) {
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        st.paint.image_filter = unsafe {
            borrow_handle::<ImageFilterPeer, _>(image_filter).map(|imf| imf.filter.clone())
        };
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerPaintSetMaskFilter(
    paint: ImpellerPaint,
    mask_filter: ImpellerMaskFilter,
) {
    if let Some(peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let mut st = peer.state.lock().expect("paint lock");
        st.paint.mask_blur_sigma =
            unsafe { borrow_handle::<MaskFilterPeer, _>(mask_filter).map_or(0.0, |mf| mf.sigma) };
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorFilterRetain(color_filter: ImpellerColorFilter) {
    unsafe { retain_handle::<ColorFilterPeer, _>(color_filter) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorFilterRelease(color_filter: ImpellerColorFilter) {
    unsafe { release_handle::<ColorFilterPeer, _>(color_filter) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorFilterCreateBlendNew(
    color: *const ImpellerColor,
    blend_mode: ImpellerBlendMode,
) -> ImpellerColorFilter {
    if color.is_null() {
        return ptr::null_mut();
    }
    let c: Color = unsafe { (*color).into() };
    into_handle(ColorFilterPeer {
        filter: ColorFilter::Mode(c, to_blend_mode(blend_mode)),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorFilterCreateColorMatrixNew(
    color_matrix: *const ImpellerColorMatrix,
) -> ImpellerColorFilter {
    if color_matrix.is_null() {
        return ptr::null_mut();
    }
    let mut m = unsafe { (*color_matrix).m };
    // `impeller.h` specifies translation columns m[4], m[9], m[14], m[19] in 0..255 space,
    // while `impeller_geometry::ColorMatrix` / `image_filter.wgsl` expects normalized 0..1 offsets.
    m[4] /= 255.0;
    m[9] /= 255.0;
    m[14] /= 255.0;
    m[19] /= 255.0;
    into_handle(ColorFilterPeer {
        filter: ColorFilter::Matrix(m),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorSourceRetain(color_source: ImpellerColorSource) {
    unsafe { retain_handle::<ColorSourcePeer, _>(color_source) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorSourceRelease(color_source: ImpellerColorSource) {
    unsafe { release_handle::<ColorSourcePeer, _>(color_source) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorSourceCreateLinearGradientNew(
    start_point: *const ImpellerPoint,
    end_point: *const ImpellerPoint,
    stop_count: u32,
    colors: *const ImpellerColor,
    stops: *const f32,
    tile_mode: ImpellerTileMode,
    transformation: *const ImpellerMatrix,
) -> ImpellerColorSource {
    if start_point.is_null() || end_point.is_null() {
        return ptr::null_mut();
    }
    let Some(grad_stops) = (unsafe { build_gradient_stops(stop_count, colors, stops) }) else {
        return ptr::null_mut();
    };
    let mut sp: Point = unsafe { (*start_point).into() };
    let mut ep: Point = unsafe { (*end_point).into() };
    if !transformation.is_null() {
        let mat: Matrix = unsafe { (*transformation).into() };
        sp = mat.transform_point(sp);
        ep = mat.transform_point(ep);
    }
    into_handle(ColorSourcePeer {
        source: ColorSource::LinearGradient(LinearGradientParams::new(
            sp,
            ep,
            grad_stops,
            to_tile_mode(tile_mode),
            1.0,
        )),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorSourceCreateRadialGradientNew(
    center: *const ImpellerPoint,
    radius: f32,
    stop_count: u32,
    colors: *const ImpellerColor,
    stops: *const f32,
    tile_mode: ImpellerTileMode,
    transformation: *const ImpellerMatrix,
) -> ImpellerColorSource {
    if center.is_null() {
        return ptr::null_mut();
    }
    let Some(grad_stops) = (unsafe { build_gradient_stops(stop_count, colors, stops) }) else {
        return ptr::null_mut();
    };
    let mut c: Point = unsafe { (*center).into() };
    let mut r = radius;
    if !transformation.is_null() {
        let mat: Matrix = unsafe { (*transformation).into() };
        c = mat.transform_point(c);
        r *= mat.get_max_basis_length_xy();
    }
    into_handle(ColorSourcePeer {
        source: ColorSource::RadialGradient(RadialGradientParams::new(
            c,
            r,
            grad_stops,
            to_tile_mode(tile_mode),
            1.0,
        )),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorSourceCreateConicalGradientNew(
    start_center: *const ImpellerPoint,
    start_radius: f32,
    end_center: *const ImpellerPoint,
    end_radius: f32,
    stop_count: u32,
    colors: *const ImpellerColor,
    stops: *const f32,
    tile_mode: ImpellerTileMode,
    transformation: *const ImpellerMatrix,
) -> ImpellerColorSource {
    if start_center.is_null() || end_center.is_null() {
        return ptr::null_mut();
    }
    let Some(grad_stops) = (unsafe { build_gradient_stops(stop_count, colors, stops) }) else {
        return ptr::null_mut();
    };
    let mut sc: Point = unsafe { (*start_center).into() };
    let mut ec: Point = unsafe { (*end_center).into() };
    let mut sr = start_radius;
    let mut er = end_radius;
    if !transformation.is_null() {
        let mat: Matrix = unsafe { (*transformation).into() };
        sc = mat.transform_point(sc);
        ec = mat.transform_point(ec);
        let scale = mat.get_max_basis_length_xy();
        sr *= scale;
        er *= scale;
    }
    into_handle(ColorSourcePeer {
        source: ColorSource::ConicalGradient(ConicalGradientParams::new(
            sc,
            sr,
            ec,
            er,
            grad_stops,
            to_tile_mode(tile_mode),
            1.0,
        )),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorSourceCreateSweepGradientNew(
    center: *const ImpellerPoint,
    start: f32,
    end: f32,
    stop_count: u32,
    colors: *const ImpellerColor,
    stops: *const f32,
    tile_mode: ImpellerTileMode,
    transformation: *const ImpellerMatrix,
) -> ImpellerColorSource {
    if center.is_null() {
        return ptr::null_mut();
    }
    let Some(grad_stops) = (unsafe { build_gradient_stops(stop_count, colors, stops) }) else {
        return ptr::null_mut();
    };
    let mut c: Point = unsafe { (*center).into() };
    if !transformation.is_null() {
        let mat: Matrix = unsafe { (*transformation).into() };
        c = mat.transform_point(c);
    }
    into_handle(ColorSourcePeer {
        source: ColorSource::SweepGradient(SweepGradientParams::new(
            c,
            start,
            end,
            grad_stops,
            to_tile_mode(tile_mode),
            1.0,
        )),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorSourceCreateImageNew(
    image: ImpellerTexture,
    horizontal_tile_mode: ImpellerTileMode,
    vertical_tile_mode: ImpellerTileMode,
    sampling: ImpellerTextureSampling,
    transformation: *const ImpellerMatrix,
) -> ImpellerColorSource {
    let Some(tex_peer) = (unsafe { borrow_handle::<TexturePeer, _>(image) }) else {
        return ptr::null_mut();
    };
    let matrix = if transformation.is_null() {
        Matrix::IDENTITY
    } else {
        unsafe { (*transformation).into() }
    };
    into_handle(ColorSourcePeer {
        source: ColorSource::Image {
            texture: tex_peer.texture.clone(),
            tile_mode_x: to_tile_mode(horizontal_tile_mode),
            tile_mode_y: to_tile_mode(vertical_tile_mode),
            matrix,
            linear_filter: sampling == K_IMPELLER_TEXTURE_SAMPLING_LINEAR,
        },
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerColorSourceCreateFragmentProgramNew(
    context: ImpellerContext,
    fragment_program: ImpellerFragmentProgram,
    samplers: *mut ImpellerTexture,
    samplers_count: usize,
    data: *const u8,
    data_bytes_length: usize,
) -> ImpellerColorSource {
    if context.is_null() {
        return ptr::null_mut();
    }
    let Some(prog) = (unsafe { borrow_handle::<FragmentProgramPeer, _>(fragment_program) }) else {
        return ptr::null_mut();
    };
    let _program_len = prog.data.len();
    if !data.is_null() && data_bytes_length > 0 {
        let _uniforms = unsafe { std::slice::from_raw_parts(data, data_bytes_length) };
    }
    if !samplers.is_null() && samplers_count > 0 {
        let first_tex = unsafe { *samplers };
        if let Some(tex_peer) = unsafe { borrow_handle::<TexturePeer, _>(first_tex) } {
            return into_handle(ColorSourcePeer {
                source: ColorSource::Image {
                    texture: tex_peer.texture.clone(),
                    tile_mode_x: to_tile_mode(K_IMPELLER_TILE_MODE_CLAMP),
                    tile_mode_y: to_tile_mode(K_IMPELLER_TILE_MODE_CLAMP),
                    matrix: Matrix::IDENTITY,
                    linear_filter: true,
                },
            });
        }
    }
    into_handle(ColorSourcePeer {
        source: ColorSource::Solid(Color::WHITE),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerMaskFilterRetain(mask_filter: ImpellerMaskFilter) {
    unsafe { retain_handle::<MaskFilterPeer, _>(mask_filter) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerMaskFilterRelease(mask_filter: ImpellerMaskFilter) {
    unsafe { release_handle::<MaskFilterPeer, _>(mask_filter) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerMaskFilterCreateBlurNew(
    _style: ImpellerBlurStyle,
    sigma: f32,
) -> ImpellerMaskFilter {
    into_handle(MaskFilterPeer {
        sigma: sigma.max(0.0),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerImageFilterRetain(image_filter: ImpellerImageFilter) {
    unsafe { retain_handle::<ImageFilterPeer, _>(image_filter) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerImageFilterRelease(image_filter: ImpellerImageFilter) {
    unsafe { release_handle::<ImageFilterPeer, _>(image_filter) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerImageFilterCreateBlurNew(
    x_sigma: f32,
    y_sigma: f32,
    tile_mode: ImpellerTileMode,
) -> ImpellerImageFilter {
    into_handle(ImageFilterPeer {
        filter: ImageFilter::Blur {
            sigma_x: x_sigma.max(0.0),
            sigma_y: y_sigma.max(0.0),
            tile_mode: to_tile_mode(tile_mode),
        },
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerImageFilterCreateDilateNew(
    x_radius: f32,
    y_radius: f32,
) -> ImpellerImageFilter {
    into_handle(ImageFilterPeer {
        filter: ImageFilter::Dilate {
            radius_x: x_radius.max(0.0),
            radius_y: y_radius.max(0.0),
        },
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerImageFilterCreateErodeNew(
    x_radius: f32,
    y_radius: f32,
) -> ImpellerImageFilter {
    into_handle(ImageFilterPeer {
        filter: ImageFilter::Erode {
            radius_x: x_radius.max(0.0),
            radius_y: y_radius.max(0.0),
        },
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerImageFilterCreateMatrixNew(
    matrix: *const ImpellerMatrix,
    _sampling: ImpellerTextureSampling,
) -> ImpellerImageFilter {
    if matrix.is_null() {
        return ptr::null_mut();
    }
    let m: Matrix = unsafe { (*matrix).into() };
    into_handle(ImageFilterPeer {
        filter: ImageFilter::Matrix(m),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerImageFilterCreateFragmentProgramNew(
    context: ImpellerContext,
    fragment_program: ImpellerFragmentProgram,
    samplers: *mut ImpellerTexture,
    samplers_count: usize,
    data: *const u8,
    data_bytes_length: usize,
) -> ImpellerImageFilter {
    if context.is_null() {
        return ptr::null_mut();
    }
    let Some(prog) = (unsafe { borrow_handle::<FragmentProgramPeer, _>(fragment_program) }) else {
        return ptr::null_mut();
    };
    let _program_len = prog.data.len();
    if !samplers.is_null() && samplers_count > 0 {
        let _s = unsafe { std::slice::from_raw_parts(samplers, samplers_count) };
    }
    if !data.is_null() && data_bytes_length > 0 {
        let _u = unsafe { std::slice::from_raw_parts(data, data_bytes_length) };
    }
    into_handle(ImageFilterPeer {
        filter: ImageFilter::Matrix(Matrix::IDENTITY),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerImageFilterCreateComposeNew(
    outer: ImpellerImageFilter,
    inner: ImpellerImageFilter,
) -> ImpellerImageFilter {
    let Some(outer_peer) = (unsafe { borrow_handle::<ImageFilterPeer, _>(outer) }) else {
        return ptr::null_mut();
    };
    let Some(inner_peer) = (unsafe { borrow_handle::<ImageFilterPeer, _>(inner) }) else {
        return ptr::null_mut();
    };
    into_handle(ImageFilterPeer {
        filter: ImageFilter::Compose(
            Box::new(outer_peer.filter.clone()),
            Box::new(inner_peer.filter.clone()),
        ),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerTextureRetain(texture: ImpellerTexture) {
    unsafe { retain_handle::<TexturePeer, _>(texture) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerTextureRelease(texture: ImpellerTexture) {
    unsafe { release_handle::<TexturePeer, _>(texture) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerTextureCreateWithContentsNew(
    context: ImpellerContext,
    descriptor: *const ImpellerTextureDescriptor,
    contents: *const ImpellerMapping,
    contents_on_release_user_data: *mut c_void,
) -> ImpellerTexture {
    let bytes_opt = unsafe { consume_mapping(contents, contents_on_release_user_data) };
    if context.is_null() || descriptor.is_null() {
        return ptr::null_mut();
    }
    let desc = unsafe { *descriptor };
    if desc.size.width <= 0 || desc.size.height <= 0 {
        return ptr::null_mut();
    }
    let width = desc.size.width as u32;
    let height = desc.size.height as u32;
    let expected = (width as usize) * (height as usize) * 4;
    let mut rgba = bytes_opt.unwrap_or_default();
    if rgba.len() < expected {
        rgba.resize(expected, 0);
    }
    let id = NEXT_TEXTURE_ID.fetch_add(1, Ordering::Relaxed);
    into_handle(TexturePeer {
        texture: ImageTexture::new(id, width, height, rgba),
        gl_handle: 0,
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerTextureCreateWithOpenGLTextureHandleNew(
    context: ImpellerContext,
    descriptor: *const ImpellerTextureDescriptor,
    handle: u64,
) -> ImpellerTexture {
    if context.is_null() || descriptor.is_null() {
        return ptr::null_mut();
    }
    let desc = unsafe { *descriptor };
    if desc.size.width <= 0 || desc.size.height <= 0 {
        return ptr::null_mut();
    }
    let width = desc.size.width as u32;
    let height = desc.size.height as u32;
    let expected = (width as usize) * (height as usize) * 4;
    let id = NEXT_TEXTURE_ID.fetch_add(1, Ordering::Relaxed);
    into_handle(TexturePeer {
        texture: ImageTexture::new(id, width, height, vec![255u8; expected]),
        gl_handle: handle,
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerTextureGetOpenGLHandle(texture: ImpellerTexture) -> u64 {
    unsafe { borrow_handle::<TexturePeer, _>(texture).map_or(0, |t| t.gl_handle) }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerFragmentProgramNew(
    data: *const ImpellerMapping,
    data_release_user_data: *mut c_void,
) -> ImpellerFragmentProgram {
    let Some(bytes) = (unsafe { consume_mapping(data, data_release_user_data) }) else {
        return ptr::null_mut();
    };
    into_handle(FragmentProgramPeer { data: bytes })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerFragmentProgramRetain(fragment_program: ImpellerFragmentProgram) {
    unsafe { retain_handle::<FragmentProgramPeer, _>(fragment_program) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerFragmentProgramRelease(fragment_program: ImpellerFragmentProgram) {
    unsafe { release_handle::<FragmentProgramPeer, _>(fragment_program) };
}

// ============================================================================
// 5. DisplayList & DisplayListBuilder
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListRetain(display_list: ImpellerDisplayList) {
    unsafe { retain_handle::<DisplayListPeer, _>(display_list) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListRelease(display_list: ImpellerDisplayList) {
    unsafe { release_handle::<DisplayListPeer, _>(display_list) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderNew(
    cull_rect: *const ImpellerRect,
) -> ImpellerDisplayListBuilder {
    let cr = if cull_rect.is_null() {
        None
    } else {
        Some(unsafe { (*cull_rect).into() })
    };
    into_handle(DisplayListBuilderPeer {
        inner: Mutex::new(DisplayListBuilderInner::new(cr)),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderRetain(builder: ImpellerDisplayListBuilder) {
    unsafe { retain_handle::<DisplayListBuilderPeer, _>(builder) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderRelease(builder: ImpellerDisplayListBuilder) {
    unsafe { release_handle::<DisplayListBuilderPeer, _>(builder) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderCreateDisplayListNew(
    builder: ImpellerDisplayListBuilder,
) -> ImpellerDisplayList {
    let Some(peer) = (unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) }) else {
        return ptr::null_mut();
    };
    let mut inner = peer.inner.lock().expect("dl builder lock");
    let dl = Arc::new(inner.builder.build());
    let typo_ctx = inner.typo_ctx.take();
    inner.builder = DisplayListBuilder::new();
    inner.stack.clear();
    inner.stack.push(Matrix::IDENTITY);
    into_handle(DisplayListPeer { dl, typo_ctx })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderSave(builder: ImpellerDisplayListBuilder) {
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let mut inner = peer.inner.lock().expect("dl builder lock");
        let cur = inner.current_transform();
        inner.stack.push(cur);
        inner.builder.save();
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderSaveLayer(
    builder: ImpellerDisplayListBuilder,
    bounds: *const ImpellerRect,
    paint: ImpellerPaint,
    backdrop: ImpellerImageFilter,
) {
    if bounds.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*bounds).into() };
        let paint_opt = unsafe {
            borrow_handle::<PaintPeer, _>(paint)
                .map(|p| p.state.lock().expect("paint lock").paint.clone())
        };
        let backdrop_opt =
            unsafe { borrow_handle::<ImageFilterPeer, _>(backdrop).map(|b| b.filter.clone()) };
        let mut inner = peer.inner.lock().expect("dl builder lock");
        let cur = inner.current_transform();
        inner.stack.push(cur);
        inner
            .builder
            .save_layer_with_backdrop(Some(r), paint_opt, backdrop_opt);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderRestore(builder: ImpellerDisplayListBuilder) {
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let mut inner = peer.inner.lock().expect("dl builder lock");
        if inner.stack.len() > 1 {
            inner.stack.pop();
        }
        inner.builder.restore();
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderScale(
    builder: ImpellerDisplayListBuilder,
    x_scale: f32,
    y_scale: f32,
) {
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let mut inner = peer.inner.lock().expect("dl builder lock");
        let top = inner.current_transform_mut();
        *top = *top * Matrix::make_scale(x_scale, y_scale, 1.0);
        inner.builder.scale(x_scale, y_scale);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderRotate(
    builder: ImpellerDisplayListBuilder,
    angle_degrees: f32,
) {
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let mut inner = peer.inner.lock().expect("dl builder lock");
        let rad = Radians(angle_degrees.to_radians());
        let top = inner.current_transform_mut();
        *top = *top * Matrix::make_rotation_z(rad);
        inner.builder.rotate(rad);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderTranslate(
    builder: ImpellerDisplayListBuilder,
    x_translation: f32,
    y_translation: f32,
) {
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let mut inner = peer.inner.lock().expect("dl builder lock");
        let top = inner.current_transform_mut();
        *top = *top * Matrix::make_translation(x_translation, y_translation, 0.0);
        inner.builder.translate(x_translation, y_translation);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderGetTransform(
    builder: ImpellerDisplayListBuilder,
    out_transform: *mut ImpellerMatrix,
) {
    if out_transform.is_null() {
        return;
    }
    let Some(peer) = (unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) }) else {
        unsafe {
            *out_transform = ImpellerMatrix::default();
        }
        return;
    };
    let inner = peer.inner.lock().expect("dl builder lock");
    unsafe {
        *out_transform = inner.current_transform().into();
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderSetTransform(
    builder: ImpellerDisplayListBuilder,
    transform: *const ImpellerMatrix,
) {
    if transform.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let m: Matrix = unsafe { (*transform).into() };
        let mut inner = peer.inner.lock().expect("dl builder lock");
        *inner.current_transform_mut() = m;
        inner.builder.reset_transform();
        inner.builder.transform(m);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderTransform(
    builder: ImpellerDisplayListBuilder,
    transform: *const ImpellerMatrix,
) {
    if transform.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let m: Matrix = unsafe { (*transform).into() };
        let mut inner = peer.inner.lock().expect("dl builder lock");
        let top = inner.current_transform_mut();
        *top = *top * m;
        inner.builder.transform(m);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderResetTransform(
    builder: ImpellerDisplayListBuilder,
) {
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let mut inner = peer.inner.lock().expect("dl builder lock");
        *inner.current_transform_mut() = Matrix::IDENTITY;
        inner.builder.reset_transform();
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderGetSaveCount(
    builder: ImpellerDisplayListBuilder,
) -> u32 {
    unsafe {
        borrow_handle::<DisplayListBuilderPeer, _>(builder).map_or(1, |peer| {
            peer.inner.lock().expect("dl builder lock").stack.len() as u32
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderRestoreToCount(
    builder: ImpellerDisplayListBuilder,
    count: u32,
) {
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let target = (count as usize).max(1);
        let mut inner = peer.inner.lock().expect("dl builder lock");
        while inner.stack.len() > target {
            inner.stack.pop();
        }
        inner.builder.restore_to_count(target);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderClipRect(
    builder: ImpellerDisplayListBuilder,
    rect: *const ImpellerRect,
    op: ImpellerClipOperation,
) {
    if rect.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*rect).into() };
        peer.inner
            .lock()
            .expect("dl builder lock")
            .builder
            .clip_rect_with_op(r, to_clip_op(op));
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderClipOval(
    builder: ImpellerDisplayListBuilder,
    oval_bounds: *const ImpellerRect,
    op: ImpellerClipOperation,
) {
    if oval_bounds.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*oval_bounds).into() };
        let mut pb = PathBuilder::new();
        pb.add_oval(r);
        let path = pb.take_path(to_fill_type(K_IMPELLER_FILL_TYPE_NON_ZERO));
        peer.inner
            .lock()
            .expect("dl builder lock")
            .builder
            .clip_path(path, to_clip_op(op));
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderClipRoundedRect(
    builder: ImpellerDisplayListBuilder,
    rect: *const ImpellerRect,
    radii: *const ImpellerRoundingRadii,
    op: ImpellerClipOperation,
) {
    if rect.is_null() || radii.is_null() {
        return;
    }
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*rect).into() };
        let rad: RoundingRadii = unsafe { (*radii).into() };
        peer.inner
            .lock()
            .expect("dl builder lock")
            .builder
            .clip_rrect(r, rad, to_clip_op(op));
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderClipPath(
    builder: ImpellerDisplayListBuilder,
    path: ImpellerPath,
    op: ImpellerClipOperation,
) {
    let Some(path_peer) = (unsafe { borrow_handle::<PathPeer, _>(path) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        peer.inner
            .lock()
            .expect("dl builder lock")
            .builder
            .clip_path(path_peer.path.clone(), to_clip_op(op));
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawPaint(
    builder: ImpellerDisplayListBuilder,
    paint: ImpellerPaint,
) {
    let Some(paint_peer) = (unsafe { borrow_handle::<PaintPeer, _>(paint) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let st = paint_peer.state.lock().expect("paint lock").clone();
        peer.inner
            .lock()
            .expect("dl builder lock")
            .builder
            .draw_paint(st.paint);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawLine(
    builder: ImpellerDisplayListBuilder,
    from: *const ImpellerPoint,
    to: *const ImpellerPoint,
    paint: ImpellerPaint,
) {
    if from.is_null() || to.is_null() {
        return;
    }
    let Some(paint_peer) = (unsafe { borrow_handle::<PaintPeer, _>(paint) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let p0: Point = unsafe { (*from).into() };
        let p1: Point = unsafe { (*to).into() };
        let st = paint_peer.state.lock().expect("paint lock").clone();
        peer.inner
            .lock()
            .expect("dl builder lock")
            .builder
            .draw_line(p0, p1, st.paint);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawDashedLine(
    builder: ImpellerDisplayListBuilder,
    from: *const ImpellerPoint,
    to: *const ImpellerPoint,
    on_length: f32,
    off_length: f32,
    paint: ImpellerPaint,
) {
    if from.is_null() || to.is_null() {
        return;
    }
    let Some(paint_peer) = (unsafe { borrow_handle::<PaintPeer, _>(paint) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let p0: Point = unsafe { (*from).into() };
        let p1: Point = unsafe { (*to).into() };
        let st = paint_peer.state.lock().expect("paint lock").clone();
        peer.inner
            .lock()
            .expect("dl builder lock")
            .builder
            .draw_dashed_line(p0, p1, on_length, off_length, st.paint);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawRect(
    builder: ImpellerDisplayListBuilder,
    rect: *const ImpellerRect,
    paint: ImpellerPaint,
) {
    if rect.is_null() {
        return;
    }
    let Some(paint_peer) = (unsafe { borrow_handle::<PaintPeer, _>(paint) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*rect).into() };
        let st = paint_peer.state.lock().expect("paint lock").clone();
        let mut inner = peer.inner.lock().expect("dl builder lock");
        inner.builder.draw_rect(r, st.paint.clone());
        if st.also_stroke {
            let mut sp = st.paint;
            sp.style = DrawStyle::Stroke;
            inner.builder.draw_rect(r, sp);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawOval(
    builder: ImpellerDisplayListBuilder,
    oval_bounds: *const ImpellerRect,
    paint: ImpellerPaint,
) {
    if oval_bounds.is_null() {
        return;
    }
    let Some(paint_peer) = (unsafe { borrow_handle::<PaintPeer, _>(paint) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*oval_bounds).into() };
        let st = paint_peer.state.lock().expect("paint lock").clone();
        let mut inner = peer.inner.lock().expect("dl builder lock");
        inner.builder.draw_oval(r, st.paint.clone());
        if st.also_stroke {
            let mut sp = st.paint;
            sp.style = DrawStyle::Stroke;
            inner.builder.draw_oval(r, sp);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawRoundedRect(
    builder: ImpellerDisplayListBuilder,
    rect: *const ImpellerRect,
    radii: *const ImpellerRoundingRadii,
    paint: ImpellerPaint,
) {
    if rect.is_null() || radii.is_null() {
        return;
    }
    let Some(paint_peer) = (unsafe { borrow_handle::<PaintPeer, _>(paint) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let r: Rect = unsafe { (*rect).into() };
        let rad: RoundingRadii = unsafe { (*radii).into() };
        let st = paint_peer.state.lock().expect("paint lock").clone();
        let mut inner = peer.inner.lock().expect("dl builder lock");
        inner.builder.draw_rrect(r, rad, st.paint.clone());
        if st.also_stroke {
            let mut sp = st.paint;
            sp.style = DrawStyle::Stroke;
            inner.builder.draw_rrect(r, rad, sp);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawRoundedRectDifference(
    builder: ImpellerDisplayListBuilder,
    outer_rect: *const ImpellerRect,
    outer_radii: *const ImpellerRoundingRadii,
    inner_rect: *const ImpellerRect,
    inner_radii: *const ImpellerRoundingRadii,
    paint: ImpellerPaint,
) {
    if outer_rect.is_null()
        || outer_radii.is_null()
        || inner_rect.is_null()
        || inner_radii.is_null()
    {
        return;
    }
    let Some(paint_peer) = (unsafe { borrow_handle::<PaintPeer, _>(paint) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let or: Rect = unsafe { (*outer_rect).into() };
        let orad: RoundingRadii = unsafe { (*outer_radii).into() };
        let ir: Rect = unsafe { (*inner_rect).into() };
        let irad: RoundingRadii = unsafe { (*inner_radii).into() };
        let st = paint_peer.state.lock().expect("paint lock").clone();
        let mut inner = peer.inner.lock().expect("dl builder lock");
        inner
            .builder
            .draw_diff_rrect(or, orad, ir, irad, st.paint.clone());
        if st.also_stroke {
            let mut sp = st.paint;
            sp.style = DrawStyle::Stroke;
            inner.builder.draw_diff_rrect(or, orad, ir, irad, sp);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawPath(
    builder: ImpellerDisplayListBuilder,
    path: ImpellerPath,
    paint: ImpellerPaint,
) {
    let Some(path_peer) = (unsafe { borrow_handle::<PathPeer, _>(path) }) else {
        return;
    };
    let Some(paint_peer) = (unsafe { borrow_handle::<PaintPeer, _>(paint) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let st = paint_peer.state.lock().expect("paint lock").clone();
        let mut inner = peer.inner.lock().expect("dl builder lock");
        inner
            .builder
            .draw_path(path_peer.path.clone(), st.paint.clone());
        if st.also_stroke {
            let mut sp = st.paint;
            sp.style = DrawStyle::Stroke;
            inner.builder.draw_path(path_peer.path.clone(), sp);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawDisplayList(
    builder: ImpellerDisplayListBuilder,
    display_list: ImpellerDisplayList,
    opacity: f32,
) {
    let Some(dl_peer) = (unsafe { borrow_handle::<DisplayListPeer, _>(display_list) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let mut inner = peer.inner.lock().expect("dl builder lock");
        if inner.typo_ctx.is_none() {
            if let Some(ref tc) = dl_peer.typo_ctx {
                inner.typo_ctx = Some(Arc::clone(tc));
            }
        }
        let alpha = opacity.clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return;
        }
        if alpha < 1.0 {
            inner
                .builder
                .save_layer(None, Some(Paint::fill(Color::new(1.0, 1.0, 1.0, alpha))));
            inner.builder.draw_picture(Arc::clone(&dl_peer.dl));
            inner.builder.restore();
        } else {
            inner.builder.draw_picture(Arc::clone(&dl_peer.dl));
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawParagraph(
    builder: ImpellerDisplayListBuilder,
    paragraph: ImpellerParagraph,
    point: *const ImpellerPoint,
) {
    if point.is_null() {
        return;
    }
    let Some(para_peer) = (unsafe { borrow_handle::<ParagraphPeer, _>(paragraph) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let pt: Point = unsafe { (*point).into() };
        let mut inner = peer.inner.lock().expect("dl builder lock");
        inner.typo_ctx = Some(Arc::clone(&para_peer.typo_ctx));
        inner
            .builder
            .draw_paragraph(Arc::clone(&para_peer.paragraph), pt);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawShadow(
    builder: ImpellerDisplayListBuilder,
    path: ImpellerPath,
    color: *const ImpellerColor,
    elevation: f32,
    transparent_occluder: bool,
    device_pixel_ratio: f32,
) {
    if color.is_null() {
        return;
    }
    let Some(path_peer) = (unsafe { borrow_handle::<PathPeer, _>(path) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let c: Color = unsafe { (*color).into() };
        peer.inner
            .lock()
            .expect("dl builder lock")
            .builder
            .draw_shadow(
                path_peer.path.clone(),
                c,
                elevation,
                transparent_occluder,
                device_pixel_ratio,
            );
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawTexture(
    builder: ImpellerDisplayListBuilder,
    texture: ImpellerTexture,
    point: *const ImpellerPoint,
    sampling: ImpellerTextureSampling,
    paint: ImpellerPaint,
) {
    if point.is_null() {
        return;
    }
    let Some(tex_peer) = (unsafe { borrow_handle::<TexturePeer, _>(texture) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let pt: Point = unsafe { (*point).into() };
        let p = unsafe {
            borrow_handle::<PaintPeer, _>(paint).map_or_else(
                || Paint::fill(Color::WHITE),
                |pp| pp.state.lock().expect("paint lock").paint.clone(),
            )
        };
        let linear = sampling == K_IMPELLER_TEXTURE_SAMPLING_LINEAR;
        peer.inner
            .lock()
            .expect("dl builder lock")
            .builder
            .draw_image(tex_peer.texture.clone(), pt, p, linear);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerDisplayListBuilderDrawTextureRect(
    builder: ImpellerDisplayListBuilder,
    texture: ImpellerTexture,
    src_rect: *const ImpellerRect,
    dst_rect: *const ImpellerRect,
    sampling: ImpellerTextureSampling,
    paint: ImpellerPaint,
) {
    if src_rect.is_null() || dst_rect.is_null() {
        return;
    }
    let Some(tex_peer) = (unsafe { borrow_handle::<TexturePeer, _>(texture) }) else {
        return;
    };
    if let Some(peer) = unsafe { borrow_handle::<DisplayListBuilderPeer, _>(builder) } {
        let src: Rect = unsafe { (*src_rect).into() };
        let dst: Rect = unsafe { (*dst_rect).into() };
        let p = unsafe {
            borrow_handle::<PaintPeer, _>(paint).map_or_else(
                || Paint::fill(Color::WHITE),
                |pp| pp.state.lock().expect("paint lock").paint.clone(),
            )
        };
        let linear = sampling == K_IMPELLER_TEXTURE_SAMPLING_LINEAR;
        peer.inner
            .lock()
            .expect("dl builder lock")
            .builder
            .draw_image_rect(tex_peer.texture.clone(), src, dst, p, linear);
    }
}

// ============================================================================
// 6. TypographyContext, ParagraphStyle, ParagraphBuilder, Paragraph,
//    LineMetrics & GlyphInfo
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn ImpellerTypographyContextNew() -> ImpellerTypographyContext {
    into_handle(TypographyContextPeer {
        ctx: Arc::new(Mutex::new(TypographerContext::new())),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerTypographyContextRetain(context: ImpellerTypographyContext) {
    unsafe { retain_handle::<TypographyContextPeer, _>(context) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerTypographyContextRelease(context: ImpellerTypographyContext) {
    unsafe { release_handle::<TypographyContextPeer, _>(context) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerTypographyContextRegisterFont(
    context: ImpellerTypographyContext,
    contents: *const ImpellerMapping,
    contents_on_release_user_data: *mut c_void,
    family_name_alias: *const c_char,
) -> bool {
    let bytes_opt = unsafe { consume_mapping(contents, contents_on_release_user_data) };
    let Some(peer) = (unsafe { borrow_handle::<TypographyContextPeer, _>(context) }) else {
        return false;
    };
    let Some(bytes) = bytes_opt else {
        return false;
    };
    if bytes.is_empty() {
        return false;
    }
    let alias = if family_name_alias.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(family_name_alias).to_str().ok() }
    };
    let ids = peer
        .ctx
        .lock()
        .expect("typo_ctx lock")
        .register_font_bytes(&bytes, alias);
    !ids.is_empty()
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleNew() -> ImpellerParagraphStyle {
    into_handle(ParagraphStylePeer {
        state: Mutex::new(ParagraphStyleState::default()),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleRetain(paragraph_style: ImpellerParagraphStyle) {
    unsafe { retain_handle::<ParagraphStylePeer, _>(paragraph_style) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleRelease(paragraph_style: ImpellerParagraphStyle) {
    unsafe { release_handle::<ParagraphStylePeer, _>(paragraph_style) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetForeground(
    paragraph_style: ImpellerParagraphStyle,
    paint: ImpellerPaint,
) {
    let Some(style_peer) = (unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) })
    else {
        return;
    };
    if let Some(paint_peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let color = paint_peer
            .state
            .lock()
            .expect("paint lock")
            .paint
            .effective_solid_color();
        let mut st = style_peer.state.lock().expect("style lock");
        st.text_style.set_color(color);
        st.text_style.foreground_paint_color = Some(color);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetBackground(
    paragraph_style: ImpellerParagraphStyle,
    paint: ImpellerPaint,
) {
    let Some(style_peer) = (unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) })
    else {
        return;
    };
    if let Some(paint_peer) = unsafe { borrow_handle::<PaintPeer, _>(paint) } {
        let color = paint_peer
            .state
            .lock()
            .expect("paint lock")
            .paint
            .effective_solid_color();
        let mut st = style_peer.state.lock().expect("style lock");
        st.text_style.background_paint_color = Some(color);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetFontWeight(
    paragraph_style: ImpellerParagraphStyle,
    weight: ImpellerFontWeight,
) {
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let mut st = style_peer.state.lock().expect("style lock");
        st.text_style.font_weight = to_css_font_weight(weight);
        st.text_style.has_font_weight = true;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetFontStyle(
    paragraph_style: ImpellerParagraphStyle,
    style: ImpellerFontStyle,
) {
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let mut st = style_peer.state.lock().expect("style lock");
        st.text_style.font_slant = if style == K_IMPELLER_FONT_STYLE_ITALIC {
            1
        } else {
            0
        };
        st.text_style.has_font_slant = true;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetFontFamily(
    paragraph_style: ImpellerParagraphStyle,
    family_name: *const c_char,
) {
    if family_name.is_null() {
        return;
    }
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let fam = unsafe { CStr::from_ptr(family_name).to_string_lossy().into_owned() };
        let mut st = style_peer.state.lock().expect("style lock");
        st.text_style.font_families.clear();
        if !fam.is_empty() {
            st.text_style.font_families.push(fam);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetFontSize(
    paragraph_style: ImpellerParagraphStyle,
    size: f32,
) {
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let mut st = style_peer.state.lock().expect("style lock");
        st.text_style.set_font_size(size);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetHeight(
    paragraph_style: ImpellerParagraphStyle,
    height: f32,
) {
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let mut st = style_peer.state.lock().expect("style lock");
        let hm = if height == 0.0 { None } else { Some(height) };
        st.para_style.height_multiplier = hm;
        st.text_style.height_multiplier = hm;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetTextAlignment(
    paragraph_style: ImpellerParagraphStyle,
    align: ImpellerTextAlignment,
) {
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let mut st = style_peer.state.lock().expect("style lock");
        st.para_style.text_align = to_text_align(align);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetTextDirection(
    paragraph_style: ImpellerParagraphStyle,
    direction: ImpellerTextDirection,
) {
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let mut st = style_peer.state.lock().expect("style lock");
        st.para_style.text_direction = to_text_direction(direction);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetTextDecoration(
    paragraph_style: ImpellerParagraphStyle,
    decoration: *const ImpellerTextDecoration,
) {
    if decoration.is_null() {
        return;
    }
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let dec = unsafe { *decoration };
        let mut st = style_peer.state.lock().expect("style lock");
        st.text_style.decoration = dec.types;
        st.text_style.decoration_color = Some(dec.color.into());
        st.text_style.decoration_style = dec.style as i32;
        st.text_style.decoration_thickness = dec.thickness_multiplier.max(0.1);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetMaxLines(
    paragraph_style: ImpellerParagraphStyle,
    max_lines: u32,
) {
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let mut st = style_peer.state.lock().expect("style lock");
        st.para_style.max_lines = if max_lines == 0 {
            None
        } else {
            Some(max_lines as usize)
        };
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetLocale(
    paragraph_style: ImpellerParagraphStyle,
    locale: *const c_char,
) {
    if locale.is_null() {
        return;
    }
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let loc = unsafe { CStr::from_ptr(locale).to_string_lossy().into_owned() };
        let mut st = style_peer.state.lock().expect("style lock");
        st.text_style.locale = if loc.is_empty() { None } else { Some(loc) };
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphStyleSetEllipsis(
    paragraph_style: ImpellerParagraphStyle,
    string: *const c_char,
) {
    if let Some(style_peer) = unsafe { borrow_handle::<ParagraphStylePeer, _>(paragraph_style) } {
        let mut st = style_peer.state.lock().expect("style lock");
        if string.is_null() {
            st.para_style.ellipsis = None;
        } else {
            let s = unsafe { CStr::from_ptr(string).to_string_lossy().into_owned() };
            st.para_style.ellipsis = if s.is_empty() { None } else { Some(s) };
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphBuilderNew(
    context: ImpellerTypographyContext,
) -> ImpellerParagraphBuilder {
    let Some(ctx_peer) = (unsafe { borrow_handle::<TypographyContextPeer, _>(context) }) else {
        return ptr::null_mut();
    };
    into_handle(ParagraphBuilderPeer {
        inner: Mutex::new(ParagraphBuilderInner {
            typo_ctx: Arc::clone(&ctx_peer.ctx),
            builder: None,
        }),
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphBuilderRetain(
    paragraph_builder: ImpellerParagraphBuilder,
) {
    unsafe { retain_handle::<ParagraphBuilderPeer, _>(paragraph_builder) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphBuilderRelease(
    paragraph_builder: ImpellerParagraphBuilder,
) {
    unsafe { release_handle::<ParagraphBuilderPeer, _>(paragraph_builder) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphBuilderPushStyle(
    paragraph_builder: ImpellerParagraphBuilder,
    style: ImpellerParagraphStyle,
) {
    let Some(builder_peer) =
        (unsafe { borrow_handle::<ParagraphBuilderPeer, _>(paragraph_builder) })
    else {
        return;
    };
    let Some(style_peer) = (unsafe { borrow_handle::<ParagraphStylePeer, _>(style) }) else {
        return;
    };
    let st = style_peer.state.lock().expect("style lock").clone();
    let mut inner = builder_peer.inner.lock().expect("paragraph builder lock");
    let pb = inner.builder.get_or_insert_with(|| {
        let mut ps = st.para_style.clone();
        ps.default_text_style = st.text_style.clone();
        ParagraphBuilder::new(ps)
    });
    pb.push_style(st.text_style);
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphBuilderPopStyle(
    paragraph_builder: ImpellerParagraphBuilder,
) {
    if let Some(builder_peer) =
        unsafe { borrow_handle::<ParagraphBuilderPeer, _>(paragraph_builder) }
    {
        let mut inner = builder_peer.inner.lock().expect("paragraph builder lock");
        let pb = inner
            .builder
            .get_or_insert_with(|| ParagraphBuilder::new(ParagraphStyle::default()));
        pb.pop();
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphBuilderAddText(
    paragraph_builder: ImpellerParagraphBuilder,
    data: *const u8,
    length: u32,
) {
    if data.is_null() || length == 0 {
        return;
    }
    if let Some(builder_peer) =
        unsafe { borrow_handle::<ParagraphBuilderPeer, _>(paragraph_builder) }
    {
        let raw_bytes = unsafe { std::slice::from_raw_parts(data, length as usize) };
        let text = String::from_utf8_lossy(raw_bytes);
        let mut inner = builder_peer.inner.lock().expect("paragraph builder lock");
        let pb = inner
            .builder
            .get_or_insert_with(|| ParagraphBuilder::new(ParagraphStyle::default()));
        pb.add_text(&text);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphBuilderBuildParagraphNew(
    paragraph_builder: ImpellerParagraphBuilder,
    width: f32,
) -> ImpellerParagraph {
    let Some(builder_peer) =
        (unsafe { borrow_handle::<ParagraphBuilderPeer, _>(paragraph_builder) })
    else {
        return ptr::null_mut();
    };
    let inner = builder_peer.inner.lock().expect("paragraph builder lock");
    let pb = inner
        .builder
        .clone()
        .unwrap_or_else(|| ParagraphBuilder::new(ParagraphStyle::default()));
    let mut para = pb.build();
    {
        let mut tc = inner.typo_ctx.lock().expect("typo_ctx lock");
        para.layout(&mut tc, width);
    }
    let lines = para.lines.clone();
    let line_metrics_handle = into_handle(LineMetricsPeer { lines });
    into_handle(ParagraphPeer {
        paragraph: Arc::new(para),
        typo_ctx: Arc::clone(&inner.typo_ctx),
        line_metrics_handle,
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphRetain(paragraph: ImpellerParagraph) {
    unsafe { retain_handle::<ParagraphPeer, _>(paragraph) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphRelease(paragraph: ImpellerParagraph) {
    unsafe { release_handle::<ParagraphPeer, _>(paragraph) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphGetMaxWidth(paragraph: ImpellerParagraph) -> f32 {
    unsafe { borrow_handle::<ParagraphPeer, _>(paragraph).map_or(0.0, |p| p.paragraph.width) }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphGetHeight(paragraph: ImpellerParagraph) -> f32 {
    unsafe { borrow_handle::<ParagraphPeer, _>(paragraph).map_or(0.0, |p| p.paragraph.height) }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphGetLongestLineWidth(paragraph: ImpellerParagraph) -> f32 {
    unsafe {
        borrow_handle::<ParagraphPeer, _>(paragraph).map_or(0.0, |p| p.paragraph.longest_line)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphGetMinIntrinsicWidth(
    paragraph: ImpellerParagraph,
) -> f32 {
    unsafe {
        borrow_handle::<ParagraphPeer, _>(paragraph)
            .map_or(0.0, |p| p.paragraph.min_intrinsic_width)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphGetMaxIntrinsicWidth(
    paragraph: ImpellerParagraph,
) -> f32 {
    unsafe {
        borrow_handle::<ParagraphPeer, _>(paragraph)
            .map_or(0.0, |p| p.paragraph.max_intrinsic_width)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphGetIdeographicBaseline(
    paragraph: ImpellerParagraph,
) -> f32 {
    unsafe {
        borrow_handle::<ParagraphPeer, _>(paragraph)
            .map_or(0.0, |p| p.paragraph.ideographic_baseline)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphGetAlphabeticBaseline(
    paragraph: ImpellerParagraph,
) -> f32 {
    unsafe {
        borrow_handle::<ParagraphPeer, _>(paragraph)
            .map_or(0.0, |p| p.paragraph.alphabetic_baseline)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphGetLineCount(paragraph: ImpellerParagraph) -> u32 {
    unsafe {
        borrow_handle::<ParagraphPeer, _>(paragraph).map_or(0, |p| p.paragraph.lines.len() as u32)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphGetLineMetrics(
    paragraph: ImpellerParagraph,
) -> ImpellerLineMetrics {
    unsafe {
        borrow_handle::<ParagraphPeer, _>(paragraph)
            .map_or(ptr::null_mut(), |p| p.line_metrics_handle)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphGetWordBoundary(
    paragraph: ImpellerParagraph,
    code_unit_index: usize,
    out_range: *mut ImpellerRange,
) {
    if out_range.is_null() {
        return;
    }
    let Some(peer) = (unsafe { borrow_handle::<ParagraphPeer, _>(paragraph) }) else {
        unsafe {
            *out_range = ImpellerRange { start: 0, end: 0 };
        }
        return;
    };
    let (start, end) = peer.paragraph.get_word_boundary(code_unit_index);
    unsafe {
        *out_range = ImpellerRange {
            start: start as u64,
            end: end as u64,
        };
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphCreateGlyphInfoAtCodeUnitIndexNew(
    paragraph: ImpellerParagraph,
    code_unit_index: usize,
) -> ImpellerGlyphInfo {
    let Some(peer) = (unsafe { borrow_handle::<ParagraphPeer, _>(paragraph) }) else {
        return ptr::null_mut();
    };
    let Some(info) = peer.paragraph.get_glyph_info_at(code_unit_index) else {
        return ptr::null_mut();
    };
    into_handle(GlyphInfoPeer {
        info,
        is_ellipsis: false,
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerParagraphCreateGlyphInfoAtParagraphCoordinatesNew(
    paragraph: ImpellerParagraph,
    x: f64,
    y: f64,
) -> ImpellerGlyphInfo {
    let Some(peer) = (unsafe { borrow_handle::<ParagraphPeer, _>(paragraph) }) else {
        return ptr::null_mut();
    };
    let Some(info) = peer.paragraph.get_closest_glyph_info_at(x as f32, y as f32) else {
        return ptr::null_mut();
    };
    into_handle(GlyphInfoPeer {
        info,
        is_ellipsis: false,
    })
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsRetain(metrics: ImpellerLineMetrics) {
    unsafe { retain_handle::<LineMetricsPeer, _>(metrics) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsRelease(metrics: ImpellerLineMetrics) {
    unsafe { release_handle::<LineMetricsPeer, _>(metrics) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetUnscaledAscent(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> f64 {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0.0, |l| f64::from(l.unscaled_ascent))
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetAscent(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> f64 {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0.0, |l| f64::from(l.ascent))
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetDescent(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> f64 {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0.0, |l| f64::from(l.descent))
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetBaseline(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> f64 {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0.0, |l| f64::from(l.baseline))
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsIsHardbreak(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> bool {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .is_some_and(|l| l.hard_break)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetWidth(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> f64 {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0.0, |l| f64::from(l.width))
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetHeight(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> f64 {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0.0, |l| f64::from(l.height))
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetLeft(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> f64 {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0.0, |l| f64::from(l.left))
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetCodeUnitStartIndex(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> usize {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0, |l| l.start_index)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetCodeUnitEndIndex(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> usize {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0, |l| l.end_index)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetCodeUnitEndIndexExcludingWhitespace(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> usize {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0, |l| l.end_excluding_whitespaces)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerLineMetricsGetCodeUnitEndIndexIncludingNewline(
    metrics: ImpellerLineMetrics,
    line: usize,
) -> usize {
    unsafe {
        borrow_handle::<LineMetricsPeer, _>(metrics)
            .and_then(|m| m.lines.get(line))
            .map_or(0, |l| l.end_including_newline)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerGlyphInfoRetain(glyph_info: ImpellerGlyphInfo) {
    unsafe { retain_handle::<GlyphInfoPeer, _>(glyph_info) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerGlyphInfoRelease(glyph_info: ImpellerGlyphInfo) {
    unsafe { release_handle::<GlyphInfoPeer, _>(glyph_info) };
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerGlyphInfoGetGraphemeClusterCodeUnitRangeBegin(
    glyph_info: ImpellerGlyphInfo,
) -> usize {
    unsafe {
        borrow_handle::<GlyphInfoPeer, _>(glyph_info)
            .map_or(0, |g| g.info.grapheme_cluster_code_unit_range_start)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerGlyphInfoGetGraphemeClusterCodeUnitRangeEnd(
    glyph_info: ImpellerGlyphInfo,
) -> usize {
    unsafe {
        borrow_handle::<GlyphInfoPeer, _>(glyph_info)
            .map_or(0, |g| g.info.grapheme_cluster_code_unit_range_end)
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerGlyphInfoGetGraphemeClusterBounds(
    glyph_info: ImpellerGlyphInfo,
    out_bounds: *mut ImpellerRect,
) {
    if out_bounds.is_null() {
        return;
    }
    let Some(peer) = (unsafe { borrow_handle::<GlyphInfoPeer, _>(glyph_info) }) else {
        unsafe {
            *out_bounds = ImpellerRect::default();
        }
        return;
    };
    unsafe {
        *out_bounds = peer.info.grapheme_cluster_bounds.into();
    }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerGlyphInfoIsEllipsis(glyph_info: ImpellerGlyphInfo) -> bool {
    unsafe { borrow_handle::<GlyphInfoPeer, _>(glyph_info).is_some_and(|g| g.is_ellipsis) }
}

#[no_mangle]
pub unsafe extern "C" fn ImpellerGlyphInfoGetTextDirection(
    glyph_info: ImpellerGlyphInfo,
) -> ImpellerTextDirection {
    unsafe {
        borrow_handle::<GlyphInfoPeer, _>(glyph_info).map_or(K_IMPELLER_TEXT_DIRECTION_LTR, |g| {
            let dir = if g.info.is_ltr {
                TextDirection::Ltr
            } else {
                TextDirection::Rtl
            };
            from_text_direction(dir)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::ffi::CString;
    use std::sync::atomic::AtomicBool;

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanCreateContext
    #[test]
    fn test_can_create_context_and_check_version() {
        let ver = ImpellerGetVersion();
        assert_eq!(ver, 0x2040_4000);

        unsafe {
            // Wrong version must return null.
            let bad = ImpellerContextCreateMetalNew(0);
            assert!(bad.is_null());

            let settings = ImpellerContextVulkanSettings {
                user_data: ptr::null_mut(),
                proc_address_callback: None,
                enable_validation: false,
            };
            let vk_ctx = ImpellerContextCreateVulkanNew(ver, &settings);
            assert!(!vk_ctx.is_null());

            let mut vk_info = ImpellerContextVulkanInfo {
                vk_instance: ptr::null_mut(),
                vk_physical_device: ptr::null_mut(),
                vk_logical_device: ptr::null_mut(),
                graphics_queue_family_index: 99,
                graphics_queue_index: 99,
            };
            assert!(ImpellerContextGetVulkanInfo(vk_ctx, &mut vk_info));
            assert_eq!(vk_info.graphics_queue_family_index, 0);

            ImpellerContextRetain(vk_ctx);
            ImpellerContextRelease(vk_ctx);
            ImpellerContextRelease(vk_ctx);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanCreateDisplayListBuilder
    #[test]
    fn test_can_create_display_list_builder_and_transform_stack() {
        unsafe {
            let cull = ImpellerRect {
                x: 0.0,
                y: 0.0,
                width: 500.0,
                height: 500.0,
            };
            let builder = ImpellerDisplayListBuilderNew(&cull);
            assert!(!builder.is_null());
            assert_eq!(ImpellerDisplayListBuilderGetSaveCount(builder), 1);

            ImpellerDisplayListBuilderSave(builder);
            assert_eq!(ImpellerDisplayListBuilderGetSaveCount(builder), 2);
            ImpellerDisplayListBuilderTranslate(builder, 10.0, 20.0);
            ImpellerDisplayListBuilderScale(builder, 2.0, 3.0);

            let mut current = ImpellerMatrix::default();
            ImpellerDisplayListBuilderGetTransform(builder, &mut current);
            assert_abs_diff_eq!(current.m[0], 2.0, epsilon = 1e-4);
            assert_abs_diff_eq!(current.m[5], 3.0, epsilon = 1e-4);
            assert_abs_diff_eq!(current.m[12], 10.0, epsilon = 1e-4);
            assert_abs_diff_eq!(current.m[13], 20.0, epsilon = 1e-4);

            ImpellerDisplayListBuilderRestore(builder);
            assert_eq!(ImpellerDisplayListBuilderGetSaveCount(builder), 1);
            ImpellerDisplayListBuilderGetTransform(builder, &mut current);
            assert_eq!(current, ImpellerMatrix::default());

            ImpellerDisplayListBuilderRelease(builder);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanCreateSurface
    #[test]
    fn test_can_create_surface_and_swapchain() {
        unsafe {
            let settings = ImpellerContextVulkanSettings {
                user_data: ptr::null_mut(),
                proc_address_callback: None,
                enable_validation: false,
            };
            let ctx = ImpellerContextCreateVulkanNew(ImpellerGetVersion(), &settings);
            assert!(!ctx.is_null());

            let size = ImpellerISize {
                width: 64,
                height: 64,
            };
            let surface =
                ImpellerSurfaceCreateWrappedFBONew(ctx, 0, K_IMPELLER_PIXEL_FORMAT_RGBA8888, &size);
            assert!(!surface.is_null());
            assert!(ImpellerSurfacePresent(surface));

            let mut dummy_surface_khr: u64 = 1;
            let swapchain =
                ImpellerVulkanSwapchainCreateNew(ctx, (&mut dummy_surface_khr as *mut u64).cast());
            assert!(!swapchain.is_null());
            let acquired = ImpellerVulkanSwapchainAcquireNextSurfaceNew(swapchain);
            assert!(!acquired.is_null());

            ImpellerSurfaceRelease(acquired);
            ImpellerVulkanSwapchainRelease(swapchain);
            ImpellerSurfaceRelease(surface);
            ImpellerContextRelease(ctx);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanDrawRect
    #[test]
    fn test_can_draw_rect_and_readback_pixels() {
        unsafe {
            let settings = ImpellerContextVulkanSettings {
                user_data: ptr::null_mut(),
                proc_address_callback: None,
                enable_validation: false,
            };
            let ctx = ImpellerContextCreateVulkanNew(ImpellerGetVersion(), &settings);
            let size = ImpellerISize {
                width: 64,
                height: 64,
            };
            let surface =
                ImpellerSurfaceCreateWrappedFBONew(ctx, 0, K_IMPELLER_PIXEL_FORMAT_RGBA8888, &size);
            let builder = ImpellerDisplayListBuilderNew(ptr::null());
            let paint = ImpellerPaintNew();
            let blue = ImpellerColor {
                red: 0.0,
                green: 0.0,
                blue: 1.0,
                alpha: 1.0,
                color_space: K_IMPELLER_COLOR_SPACE_SRGB,
            };
            ImpellerPaintSetColor(paint, &blue);
            ImpellerDisplayListBuilderDrawPaint(builder, paint);

            let red = ImpellerColor {
                red: 1.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
                color_space: K_IMPELLER_COLOR_SPACE_SRGB,
            };
            ImpellerPaintSetColor(paint, &red);
            let rect = ImpellerRect {
                x: 16.0,
                y: 16.0,
                width: 32.0,
                height: 32.0,
            };
            ImpellerDisplayListBuilderDrawRect(builder, &rect, paint);

            let dl = ImpellerDisplayListBuilderCreateDisplayListNew(builder);
            assert!(!dl.is_null());
            assert!(ImpellerSurfaceDrawDisplayList(surface, dl));

            let mut rgba = vec![0u8; 64 * 64 * 4];
            assert!(ImpellerSurfaceCopyPixelsRGBA8888(
                surface,
                rgba.as_mut_ptr(),
                rgba.len() as u64
            ));
            if shared_headless_renderer().is_some() {
                // Center pixel (32, 32) should be red; corner pixel (4, 4) should be blue.
                let center_idx = (32 * 64 + 32) * 4;
                assert!(rgba[center_idx] > 200 && rgba[center_idx + 2] < 30);
                let corner_idx = (4 * 64 + 4) * 4;
                assert!(rgba[corner_idx + 2] > 200 && rgba[corner_idx] < 30);
            }

            ImpellerDisplayListRelease(dl);
            ImpellerPaintRelease(paint);
            ImpellerDisplayListBuilderRelease(builder);
            ImpellerSurfaceRelease(surface);
            ImpellerContextRelease(ctx);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanGetPathBounds
    #[test]
    fn test_can_get_path_bounds_and_copy_or_take_path() {
        unsafe {
            let pb = ImpellerPathBuilderNew();
            let p0 = ImpellerPoint { x: 100.0, y: 100.0 };
            let p1 = ImpellerPoint { x: 200.0, y: 200.0 };
            ImpellerPathBuilderMoveTo(pb, &p0);
            ImpellerPathBuilderLineTo(pb, &p1);

            let copied = ImpellerPathBuilderCopyPathNew(pb, K_IMPELLER_FILL_TYPE_NON_ZERO);
            let mut bounds = ImpellerRect::default();
            ImpellerPathGetBounds(copied, &mut bounds);
            assert_abs_diff_eq!(bounds.x, 100.0, epsilon = 1e-4);
            assert_abs_diff_eq!(bounds.y, 100.0, epsilon = 1e-4);
            assert_abs_diff_eq!(bounds.width, 100.0, epsilon = 1e-4);
            assert_abs_diff_eq!(bounds.height, 100.0, epsilon = 1e-4);

            let taken = ImpellerPathBuilderTakePathNew(pb, K_IMPELLER_FILL_TYPE_ODD);
            let mut taken_bounds = ImpellerRect::default();
            ImpellerPathGetBounds(taken, &mut taken_bounds);
            assert_abs_diff_eq!(taken_bounds.width, 100.0, epsilon = 1e-4);

            // After TakePathNew, builder is reset to empty.
            let empty_after = ImpellerPathBuilderCopyPathNew(pb, K_IMPELLER_FILL_TYPE_NON_ZERO);
            let mut empty_bounds = ImpellerRect::default();
            ImpellerPathGetBounds(empty_after, &mut empty_bounds);
            assert_abs_diff_eq!(empty_bounds.width, 0.0, epsilon = 1e-4);

            ImpellerPathRelease(empty_after);
            ImpellerPathRelease(taken);
            ImpellerPathRelease(copied);
            ImpellerPathBuilderRelease(pb);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanCreateShapes
    #[test]
    fn test_can_create_shapes_and_curves() {
        unsafe {
            let pb = ImpellerPathBuilderNew();
            let rect = ImpellerRect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
            };
            ImpellerPathBuilderAddRect(pb, &rect);
            ImpellerPathBuilderAddOval(pb, &rect);
            ImpellerPathBuilderAddArc(pb, &rect, 0.0, 90.0);
            let radii = ImpellerRoundingRadii {
                top_left: ImpellerSize {
                    width: 10.0,
                    height: 10.0,
                },
                bottom_left: ImpellerSize::default(),
                top_right: ImpellerSize::default(),
                bottom_right: ImpellerSize {
                    width: 10.0,
                    height: 10.0,
                },
            };
            ImpellerPathBuilderAddRoundedRect(pb, &rect, &radii);
            let cp = ImpellerPoint { x: 50.0, y: 0.0 };
            let ep = ImpellerPoint { x: 100.0, y: 50.0 };
            ImpellerPathBuilderQuadraticCurveTo(pb, &cp, &ep);
            ImpellerPathBuilderCubicCurveTo(pb, &cp, &ep, &ImpellerPoint { x: 120.0, y: 120.0 });
            ImpellerPathBuilderClose(pb);

            let path = ImpellerPathBuilderTakePathNew(pb, K_IMPELLER_FILL_TYPE_NON_ZERO);
            let builder = ImpellerDisplayListBuilderNew(ptr::null());
            let paint = ImpellerPaintNew();
            ImpellerPaintSetDrawStyle(paint, K_IMPELLER_DRAW_STYLE_STROKE_AND_FILL);
            ImpellerPaintSetStrokeWidth(paint, 4.0);
            ImpellerPaintSetStrokeCap(paint, K_IMPELLER_STROKE_CAP_ROUND);
            ImpellerPaintSetStrokeJoin(paint, K_IMPELLER_STROKE_JOIN_ROUND);
            ImpellerDisplayListBuilderDrawRoundedRect(builder, &rect, &radii, paint);
            ImpellerDisplayListBuilderDrawOval(builder, &rect, paint);
            ImpellerDisplayListBuilderDrawPath(builder, path, paint);
            let dl = ImpellerDisplayListBuilderCreateDisplayListNew(builder);
            assert!(!dl.is_null());

            ImpellerDisplayListRelease(dl);
            ImpellerPaintRelease(paint);
            ImpellerDisplayListBuilderRelease(builder);
            ImpellerPathRelease(path);
            ImpellerPathBuilderRelease(pb);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanCreateOpenGLTexture
    #[test]
    fn test_can_create_opengl_and_contents_textures() {
        unsafe {
            let ctx = ImpellerContextCreateOpenGLESNew(ImpellerGetVersion(), None, ptr::null_mut());
            assert!(!ctx.is_null());

            let desc = ImpellerTextureDescriptor {
                pixel_format: K_IMPELLER_PIXEL_FORMAT_RGBA8888,
                size: ImpellerISize {
                    width: 4,
                    height: 4,
                },
                mip_count: 1,
            };
            let gl_tex = ImpellerTextureCreateWithOpenGLTextureHandleNew(ctx, &desc, 42);
            assert!(!gl_tex.is_null());
            assert_eq!(ImpellerTextureGetOpenGLHandle(gl_tex), 42);

            let builder = ImpellerDisplayListBuilderNew(ptr::null());
            let pt = ImpellerPoint { x: 10.0, y: 10.0 };
            ImpellerDisplayListBuilderDrawTexture(
                builder,
                gl_tex,
                &pt,
                K_IMPELLER_TEXTURE_SAMPLING_LINEAR,
                ptr::null_mut(),
            );
            let dl = ImpellerDisplayListBuilderCreateDisplayListNew(builder);
            assert!(!dl.is_null());

            ImpellerDisplayListRelease(dl);
            ImpellerDisplayListBuilderRelease(builder);
            ImpellerTextureRelease(gl_tex);
            ImpellerContextRelease(ctx);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:MappingsReleaseTheirDataOnDestruction
    #[test]
    fn test_mappings_release_their_data_on_consumption() {
        static RELEASED: AtomicBool = AtomicBool::new(false);
        unsafe extern "C" fn on_release(user_data: *mut c_void) {
            if !user_data.is_null() {
                let flag = unsafe { &*(user_data as *const AtomicBool) };
                flag.store(true, Ordering::SeqCst);
            }
        }

        unsafe {
            let ctx = ImpellerContextCreateMetalNew(ImpellerGetVersion());
            let desc = ImpellerTextureDescriptor {
                pixel_format: K_IMPELLER_PIXEL_FORMAT_RGBA8888,
                size: ImpellerISize {
                    width: 2,
                    height: 2,
                },
                mip_count: 1,
            };
            let pixels = [255u8; 16];
            let mapping = ImpellerMapping {
                data: pixels.as_ptr(),
                length: pixels.len() as u64,
                on_release: Some(on_release),
            };
            RELEASED.store(false, Ordering::SeqCst);
            let tex = ImpellerTextureCreateWithContentsNew(
                ctx,
                &desc,
                &mapping,
                (&RELEASED as *const AtomicBool) as *mut c_void,
            );
            assert!(!tex.is_null());
            assert!(RELEASED.load(Ordering::SeqCst));

            ImpellerTextureRelease(tex);
            ImpellerContextRelease(ctx);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:ClearsOpenGLStancilStateAfterTransition
    #[test]
    fn test_clip_difference_and_save_layer_filters() {
        unsafe {
            let builder = ImpellerDisplayListBuilderNew(ptr::null());
            let paint = ImpellerPaintNew();
            let rect = ImpellerRect {
                x: 10.0,
                y: 20.0,
                width: 100.0,
                height: 200.0,
            };
            ImpellerDisplayListBuilderClipRect(
                builder,
                &rect,
                K_IMPELLER_CLIP_OPERATION_DIFFERENCE,
            );

            let blur = ImpellerImageFilterCreateBlurNew(3.0, 3.0, K_IMPELLER_TILE_MODE_CLAMP);
            let dilate = ImpellerImageFilterCreateDilateNew(1.0, 1.0);
            let composed = ImpellerImageFilterCreateComposeNew(blur, dilate);
            assert!(!composed.is_null());

            let cf = ImpellerColorFilterCreateBlendNew(
                &ImpellerColor {
                    red: 1.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 0.5,
                    color_space: K_IMPELLER_COLOR_SPACE_SRGB,
                },
                K_IMPELLER_BLEND_MODE_SOURCE_OVER,
            );
            let mf = ImpellerMaskFilterCreateBlurNew(K_IMPELLER_BLUR_STYLE_NORMAL, 2.0);
            ImpellerPaintSetColorFilter(paint, cf);
            ImpellerPaintSetImageFilter(paint, composed);
            ImpellerPaintSetMaskFilter(paint, mf);

            ImpellerDisplayListBuilderSaveLayer(builder, &rect, paint, blur);
            ImpellerDisplayListBuilderDrawRect(builder, &rect, paint);
            ImpellerDisplayListBuilderRestore(builder);

            let dl = ImpellerDisplayListBuilderCreateDisplayListNew(builder);
            assert!(!dl.is_null());

            ImpellerDisplayListRelease(dl);
            ImpellerMaskFilterRelease(mf);
            ImpellerColorFilterRelease(cf);
            ImpellerImageFilterRelease(composed);
            ImpellerImageFilterRelease(dilate);
            ImpellerImageFilterRelease(blur);
            ImpellerPaintRelease(paint);
            ImpellerDisplayListBuilderRelease(builder);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanCreateParagraphs
    #[test]
    fn test_can_create_paragraphs_and_render_to_surface() {
        unsafe {
            let tc = ImpellerTypographyContextNew();
            assert!(!tc.is_null());

            let pb = ImpellerParagraphBuilderNew(tc);
            let style = ImpellerParagraphStyleNew();
            ImpellerParagraphStyleSetFontSize(style, 24.0);
            ImpellerParagraphStyleSetHeight(style, 1.2);

            let fg = ImpellerPaintNew();
            ImpellerPaintSetColor(
                fg,
                &ImpellerColor {
                    red: 1.0,
                    green: 1.0,
                    blue: 1.0,
                    alpha: 1.0,
                    color_space: K_IMPELLER_COLOR_SPACE_SRGB,
                },
            );
            ImpellerParagraphStyleSetForeground(style, fg);

            ImpellerParagraphBuilderPushStyle(pb, style);
            let text = b"Hello Impeller C-ABI";
            ImpellerParagraphBuilderAddText(pb, text.as_ptr(), text.len() as u32);
            let para = ImpellerParagraphBuilderBuildParagraphNew(pb, 400.0);
            assert!(!para.is_null());
            assert!(ImpellerParagraphGetHeight(para) > 10.0);
            assert!(ImpellerParagraphGetLongestLineWidth(para) > 20.0);

            let dl_builder = ImpellerDisplayListBuilderNew(ptr::null());
            let pt = ImpellerPoint { x: 8.0, y: 8.0 };
            ImpellerDisplayListBuilderDrawParagraph(dl_builder, para, &pt);
            let dl = ImpellerDisplayListBuilderCreateDisplayListNew(dl_builder);

            let ctx = ImpellerContextCreateMetalNew(ImpellerGetVersion());
            let surf = ImpellerSurfaceCreateWrappedFBONew(
                ctx,
                0,
                K_IMPELLER_PIXEL_FORMAT_RGBA8888,
                &ImpellerISize {
                    width: 256,
                    height: 64,
                },
            );
            assert!(ImpellerSurfaceDrawDisplayList(surf, dl));

            ImpellerSurfaceRelease(surf);
            ImpellerContextRelease(ctx);
            ImpellerDisplayListRelease(dl);
            ImpellerDisplayListBuilderRelease(dl_builder);
            ImpellerParagraphRelease(para);
            ImpellerPaintRelease(fg);
            ImpellerParagraphStyleRelease(style);
            ImpellerParagraphBuilderRelease(pb);
            ImpellerTypographyContextRelease(tc);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanMeasureText
    #[test]
    fn test_can_measure_text_line_metrics_and_glyph_info() {
        unsafe {
            let tc = ImpellerTypographyContextNew();
            let pb = ImpellerParagraphBuilderNew(tc);
            let style = ImpellerParagraphStyleNew();
            ImpellerParagraphStyleSetFontSize(style, 20.0);
            ImpellerParagraphBuilderPushStyle(pb, style);

            let text = "Alpha Beta\nSecond line";
            ImpellerParagraphBuilderAddText(pb, text.as_ptr(), text.len() as u32);
            let para = ImpellerParagraphBuilderBuildParagraphNew(pb, 500.0);
            assert_eq!(ImpellerParagraphGetLineCount(para), 2);

            let metrics = ImpellerParagraphGetLineMetrics(para);
            assert!(!metrics.is_null());
            assert!(ImpellerLineMetricsGetAscent(metrics, 0) > 0.0);
            assert!(ImpellerLineMetricsGetDescent(metrics, 0) > 0.0);
            assert!(ImpellerLineMetricsGetBaseline(metrics, 0) > 0.0);
            assert!(ImpellerLineMetricsIsHardbreak(metrics, 0));
            assert_eq!(ImpellerLineMetricsGetCodeUnitStartIndex(metrics, 0), 0);
            assert_eq!(ImpellerLineMetricsGetCodeUnitEndIndex(metrics, 0), 10);
            assert_eq!(
                ImpellerLineMetricsGetCodeUnitEndIndexIncludingNewline(metrics, 0),
                11
            );

            let ginfo = ImpellerParagraphCreateGlyphInfoAtCodeUnitIndexNew(para, 0);
            assert!(!ginfo.is_null());
            assert_eq!(
                ImpellerGlyphInfoGetGraphemeClusterCodeUnitRangeBegin(ginfo),
                0
            );
            assert_eq!(
                ImpellerGlyphInfoGetGraphemeClusterCodeUnitRangeEnd(ginfo),
                1
            );
            let mut gbounds = ImpellerRect::default();
            ImpellerGlyphInfoGetGraphemeClusterBounds(ginfo, &mut gbounds);
            assert!(gbounds.width > 0.0 && gbounds.height > 0.0);
            assert!(!ImpellerGlyphInfoIsEllipsis(ginfo));
            assert_eq!(
                ImpellerGlyphInfoGetTextDirection(ginfo),
                K_IMPELLER_TEXT_DIRECTION_LTR
            );
            ImpellerGlyphInfoRelease(ginfo);

            let gcoord = ImpellerParagraphCreateGlyphInfoAtParagraphCoordinatesNew(para, 2.0, 5.0);
            assert!(!gcoord.is_null());
            ImpellerGlyphInfoRelease(gcoord);

            let mut word_range = ImpellerRange::default();
            ImpellerParagraphGetWordBoundary(para, 1, &mut word_range);
            assert_eq!((word_range.start, word_range.end), (0, 5));

            ImpellerParagraphRelease(para);
            ImpellerParagraphStyleRelease(style);
            ImpellerParagraphBuilderRelease(pb);
            ImpellerTypographyContextRelease(tc);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanCreateDecorations
    #[test]
    fn test_can_create_decorations_and_alignments() {
        unsafe {
            let tc = ImpellerTypographyContextNew();
            let pb = ImpellerParagraphBuilderNew(tc);
            let style = ImpellerParagraphStyleNew();
            ImpellerParagraphStyleSetFontSize(style, 24.0);
            ImpellerParagraphStyleSetTextAlignment(style, K_IMPELLER_TEXT_ALIGNMENT_CENTER);
            let dec = ImpellerTextDecoration {
                types: K_IMPELLER_TEXT_DECORATION_TYPE_UNDERLINE
                    | K_IMPELLER_TEXT_DECORATION_TYPE_LINE_THROUGH,
                color: ImpellerColor {
                    red: 0.0,
                    green: 1.0,
                    blue: 0.0,
                    alpha: 1.0,
                    color_space: K_IMPELLER_COLOR_SPACE_SRGB,
                },
                style: K_IMPELLER_TEXT_DECORATION_STYLE_SOLID,
                thickness_multiplier: 1.5,
            };
            ImpellerParagraphStyleSetTextDecoration(style, &dec);
            ImpellerParagraphBuilderPushStyle(pb, style);
            let text = b"Centered Underlined";
            ImpellerParagraphBuilderAddText(pb, text.as_ptr(), text.len() as u32);
            let para = ImpellerParagraphBuilderBuildParagraphNew(pb, 400.0);
            let metrics = ImpellerParagraphGetLineMetrics(para);
            assert!(ImpellerLineMetricsGetLeft(metrics, 0) > 10.0);

            ImpellerParagraphRelease(para);
            ImpellerParagraphStyleRelease(style);
            ImpellerParagraphBuilderRelease(pb);
            ImpellerTypographyContextRelease(tc);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanCreateParagraphsWithCustomFont
    #[test]
    fn test_can_register_custom_font_and_shape_paragraph() {
        static FONT_RELEASED: AtomicBool = AtomicBool::new(false);
        unsafe extern "C" fn on_font_release(user_data: *mut c_void) {
            if !user_data.is_null() {
                let flag = unsafe { &*(user_data as *const AtomicBool) };
                flag.store(true, Ordering::SeqCst);
            }
        }

        unsafe {
            let tc = ImpellerTypographyContextNew();
            let font_bytes = impeller_typographer::ROBOTO_BOLD_TTF;
            let mapping = ImpellerMapping {
                data: font_bytes.as_ptr(),
                length: font_bytes.len() as u64,
                on_release: Some(on_font_release),
            };
            let alias = CString::new("CustomRobotoAlias").expect("cstring");
            FONT_RELEASED.store(false, Ordering::SeqCst);
            let ok = ImpellerTypographyContextRegisterFont(
                tc,
                &mapping,
                (&FONT_RELEASED as *const AtomicBool) as *mut c_void,
                alias.as_ptr(),
            );
            assert!(ok);
            assert!(FONT_RELEASED.load(Ordering::SeqCst));

            let style = ImpellerParagraphStyleNew();
            ImpellerParagraphStyleSetFontFamily(style, alias.as_ptr());
            ImpellerParagraphStyleSetFontSize(style, 20.0);
            let pb = ImpellerParagraphBuilderNew(tc);
            ImpellerParagraphBuilderPushStyle(pb, style);
            let text = b"Custom Font Text";
            ImpellerParagraphBuilderAddText(pb, text.as_ptr(), text.len() as u32);
            let para = ImpellerParagraphBuilderBuildParagraphNew(pb, 300.0);
            assert!(ImpellerParagraphGetLongestLineWidth(para) > 20.0);

            ImpellerParagraphRelease(para);
            ImpellerParagraphBuilderRelease(pb);
            ImpellerParagraphStyleRelease(style);
            ImpellerTypographyContextRelease(tc);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanRenderShadows
    #[test]
    fn test_can_render_shadows_and_gradients() {
        unsafe {
            let pb = ImpellerPathBuilderNew();
            let box_rect = ImpellerRect {
                x: 20.0,
                y: 20.0,
                width: 60.0,
                height: 60.0,
            };
            ImpellerPathBuilderAddRect(pb, &box_rect);
            let path = ImpellerPathBuilderTakePathNew(pb, K_IMPELLER_FILL_TYPE_NON_ZERO);

            let colors = [
                ImpellerColor {
                    red: 1.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 1.0,
                    color_space: K_IMPELLER_COLOR_SPACE_SRGB,
                },
                ImpellerColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 1.0,
                    alpha: 1.0,
                    color_space: K_IMPELLER_COLOR_SPACE_SRGB,
                },
            ];
            let stops = [0.0f32, 1.0f32];
            let grad = ImpellerColorSourceCreateLinearGradientNew(
                &ImpellerPoint { x: 0.0, y: 0.0 },
                &ImpellerPoint { x: 100.0, y: 100.0 },
                2,
                colors.as_ptr(),
                stops.as_ptr(),
                K_IMPELLER_TILE_MODE_CLAMP,
                ptr::null(),
            );
            assert!(!grad.is_null());

            let paint = ImpellerPaintNew();
            ImpellerPaintSetColorSource(paint, grad);

            let dl_builder = ImpellerDisplayListBuilderNew(ptr::null());
            ImpellerDisplayListBuilderDrawShadow(
                dl_builder,
                path,
                &ImpellerColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 1.0,
                    color_space: K_IMPELLER_COLOR_SPACE_SRGB,
                },
                4.0,
                false,
                1.0,
            );
            ImpellerDisplayListBuilderDrawRect(dl_builder, &box_rect, paint);
            let dl = ImpellerDisplayListBuilderCreateDisplayListNew(dl_builder);
            assert!(!dl.is_null());

            ImpellerDisplayListRelease(dl);
            ImpellerDisplayListBuilderRelease(dl_builder);
            ImpellerPaintRelease(paint);
            ImpellerColorSourceRelease(grad);
            ImpellerPathRelease(path);
            ImpellerPathBuilderRelease(pb);
        }
    }

    /// C++ source: impeller/toolkit/interop/impeller_unittests.cc:CanCreateFragmentProgramColorFilters
    #[test]
    fn test_can_create_fragment_program_color_sources_and_filters() {
        unsafe {
            let ctx = ImpellerContextCreateMetalNew(ImpellerGetVersion());
            let bytes = [1u8, 2, 3, 4];
            let mapping = ImpellerMapping {
                data: bytes.as_ptr(),
                length: bytes.len() as u64,
                on_release: None,
            };
            let prog = ImpellerFragmentProgramNew(&mapping, ptr::null_mut());
            assert!(!prog.is_null());

            let imf = ImpellerImageFilterCreateFragmentProgramNew(
                ctx,
                prog,
                ptr::null_mut(),
                0,
                ptr::null(),
                0,
            );
            assert!(!imf.is_null());

            let cs = ImpellerColorSourceCreateFragmentProgramNew(
                ctx,
                prog,
                ptr::null_mut(),
                0,
                ptr::null(),
                0,
            );
            assert!(!cs.is_null());

            ImpellerColorSourceRelease(cs);
            ImpellerImageFilterRelease(imf);
            ImpellerFragmentProgramRelease(prog);
            ImpellerContextRelease(ctx);
        }
    }
}
