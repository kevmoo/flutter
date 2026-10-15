#![forbid(unsafe_code)]

use impeller_canvas::{
    save_rgba_png, Canvas, Cap, ClipOperation, Color, ColorFilter, ColorSource,
    ConicalGradientParams, DisplayList, DisplayListBuilder, FillType, GradientStop, Join,
    LinearGradientParams, Paint, PathBuilder, Point, RadialGradientParams, Radians, Rect,
    RoundSuperellipse, RoundingRadii, Scalar, SweepGradientParams, TileMode, WgpuContext,
};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

const SCRATCH_ROOT: &str =
    "/usr/local/google/home/kevmoo/.gemini/jetski/brain/aaf8f361-7fe2-4fe7-8caa-40ddbfdb8335/scratch";

fn shared_renderer() -> MutexGuard<'static, WgpuContext> {
    static RENDERER: OnceLock<Mutex<WgpuContext>> = OnceLock::new();
    RENDERER
        .get_or_init(|| Mutex::new(WgpuContext::new_headless().expect("headless wgpu renderer")))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn golden_output_dir() -> PathBuf {
    let scratch = PathBuf::from(SCRATCH_ROOT);
    let base = if scratch.exists() {
        scratch
    } else {
        std::env::temp_dir()
    };
    let dir = base.join("aiks_dl_goldens");
    std::fs::create_dir_all(&dir).expect("create aiks_dl_goldens dir");
    dir
}

fn render_and_save_dl(
    scene_name: &str,
    width: u32,
    height: u32,
    clear_color: Color,
    dl: &DisplayList,
) -> Vec<u8> {
    let mut canvas = Canvas::new(width, height, clear_color);
    canvas.draw_display_list(dl);
    let rgba = {
        let renderer = shared_renderer();
        canvas
            .render_to_pixels(&renderer)
            .unwrap_or_else(|e| panic!("failed to render {scene_name}: {e}"))
    };
    let png_path = golden_output_dir().join(format!("{scene_name}.png"));
    save_rgba_png(&png_path, width, height, &rgba)
        .unwrap_or_else(|e| panic!("failed to save {scene_name}.png: {e}"));
    rgba
}

fn render_and_save_dl_with_cull_rect(
    scene_name: &str,
    width: u32,
    height: u32,
    clear_color: Color,
    cull_rect: Rect,
    dl: &DisplayList,
) -> Vec<u8> {
    let mut canvas = Canvas::with_cull_rect(width, height, clear_color, cull_rect);
    canvas.draw_display_list(dl);
    let rgba = {
        let renderer = shared_renderer();
        canvas
            .render_to_pixels(&renderer)
            .unwrap_or_else(|e| panic!("failed to render {scene_name}: {e}"))
    };
    let png_path = golden_output_dir().join(format!("{scene_name}.png"));
    save_rgba_png(&png_path, width, height, &rgba)
        .unwrap_or_else(|e| panic!("failed to save {scene_name}.png: {e}"));
    rgba
}

fn get_pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let idx = ((y * width + x) as usize) * 4;
    [rgba[idx], rgba[idx + 1], rgba[idx + 2], rgba[idx + 3]]
}

fn max_channel_delta(px: [u8; 4], bg: [u8; 4]) -> u8 {
    px[0]
        .abs_diff(bg[0])
        .max(px[1].abs_diff(bg[1]))
        .max(px[2].abs_diff(bg[2]))
}

fn assert_pixel_near(
    rgba: &[u8],
    width: u32,
    x: u32,
    y: u32,
    expected: [u8; 4],
    tol: u8,
    label: &str,
) {
    let actual = get_pixel(rgba, width, x, y);
    for c in 0..4 {
        let diff = actual[c].abs_diff(expected[c]);
        assert!(
            diff <= tol,
            "{label} at ({x}, {y}) channel {c}: actual={actual:?}, expected={expected:?} (tol={tol})"
        );
    }
}

// =============================================================================
// Group A: 5 Basic & Transform scenes (aiks_dl_basic_unittests.cc)
// =============================================================================

/// C++ source: impeller/display_list/aiks_dl_basic_unittests.cc:CanRenderColoredRectPrimitive
#[test]
fn aiks_dl_01_colored_rects_and_cull() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(15, 18, 26, 255);
    let bg_u8 = [15, 18, 26, 255];
    let cull_rect = Rect::from_xywh(16.0, 16.0, 224.0, 224.0);

    let mut builder = DisplayListBuilder::new();

    // 1. Yellow rect completely outside cull_rect (top-left margin 2..12) -> must be culled.
    builder.draw_rect(
        Rect::from_xywh(2.0, 2.0, 10.0, 10.0),
        Paint::fill(Color::make_rgba8(255, 230, 0, 255)),
    );

    // 2. Untransformed red rect in top-left quadrant inside cull_rect.
    builder.draw_rect(
        Rect::from_xywh(28.0, 28.0, 76.0, 76.0),
        Paint::fill(Color::make_rgba8(235, 52, 64, 255)),
    );

    // 3. Translated + scaled green rect in top-right quadrant via save/restore.
    builder.save();
    builder.translate(140.0, 28.0);
    builder.scale(1.5, 1.0);
    builder.draw_rect(
        Rect::from_xywh(0.0, 0.0, 56.0, 76.0),
        Paint::fill(Color::make_rgba8(42, 210, 110, 255)),
    );
    builder.restore();

    // 4. Translated + rotated blue diamond rect in bottom center via save/restore.
    builder.save();
    builder.translate(128.0, 176.0);
    builder.rotate(Radians(FRAC_PI_4));
    builder.draw_rect(
        Rect::from_xywh(-40.0, -40.0, 80.0, 80.0),
        Paint::fill(Color::make_rgba8(56, 132, 255, 255)),
    );
    builder.restore();

    let dl = builder.build();
    let rgba = render_and_save_dl_with_cull_rect(
        "aiks_dl_01_colored_rects_and_cull",
        width,
        height,
        bg,
        cull_rect,
        &dl,
    );

    // Verify culled yellow rect did not draw outside cull_rect.
    assert_pixel_near(&rgba, width, 6, 6, bg_u8, 2, "culled exterior yellow rect");
    // Verify red rect interior.
    assert_pixel_near(
        &rgba,
        width,
        64,
        64,
        [235, 52, 64, 255],
        3,
        "red rect interior",
    );
    // Verify green rect interior (140..224, 28..104).
    assert_pixel_near(
        &rgba,
        width,
        180,
        64,
        [42, 210, 110, 255],
        3,
        "scaled green rect interior",
    );
    // Verify rotated blue diamond interior and outside its rotated edge.
    assert_pixel_near(
        &rgba,
        width,
        128,
        176,
        [56, 132, 255, 255],
        3,
        "rotated blue rect center",
    );
    assert_pixel_near(
        &rgba,
        width,
        76,
        124,
        bg_u8,
        3,
        "outside rotated blue diamond",
    );
}

/// C++ source: impeller/display_list/aiks_dl_basic_unittests.cc:CanRenderDifferentShapesWithSameColorSource
#[test]
fn aiks_dl_02_nested_display_lists_and_transforms() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(14, 16, 24, 255);
    let bg_u8 = [14, 16, 24, 255];

    // Inner reusable badge DisplayList: cyan rounded plate (-28..-28, 56x56) + coral core circle (r=16).
    let mut inner = DisplayListBuilder::new();
    inner.draw_rrect(
        Rect::from_xywh(-28.0, -28.0, 56.0, 56.0),
        RoundingRadii::from_single_radius(10.0),
        Paint::fill(Color::make_rgba8(32, 198, 232, 255)),
    );
    inner.draw_circle(
        Point::new(0.0, 0.0),
        16.0,
        Paint::fill(Color::make_rgba8(255, 92, 108, 255)),
    );
    let badge_dl = Arc::new(inner.build());

    // Middle DisplayList combining two transformed instances of badge_dl.
    let mut mid = DisplayListBuilder::new();
    mid.save();
    mid.translate(64.0, 68.0);
    mid.draw_picture(Arc::clone(&badge_dl));
    mid.restore();

    mid.save();
    mid.translate(188.0, 68.0);
    mid.scale(1.25, 1.25);
    mid.draw_picture(Arc::clone(&badge_dl));
    mid.restore();
    let mid_dl = Arc::new(mid.build());

    // Outer DisplayList replays mid_dl and adds a third rotated+scaled nested badge_dl.
    let mut outer = DisplayListBuilder::new();
    outer.draw_picture(mid_dl);

    outer.save();
    outer.translate(128.0, 184.0);
    outer.rotate(Radians(FRAC_PI_4));
    outer.scale(1.35, 1.35);
    outer.draw_picture(badge_dl);
    outer.restore();

    let rgba = render_and_save_dl(
        "aiks_dl_02_nested_display_lists_and_transforms",
        width,
        height,
        bg,
        &outer.build(),
    );

    // Instance 1 (64, 68): coral center + cyan border.
    assert_pixel_near(&rgba, width, 64, 68, [255, 92, 108, 255], 3, "badge 1 core");
    assert_pixel_near(
        &rgba,
        width,
        44,
        68,
        [32, 198, 232, 255],
        3,
        "badge 1 cyan plate",
    );

    // Instance 2 (188, 68) scaled 1.25x: core radius=20, plate half-width=35.
    assert_pixel_near(
        &rgba,
        width,
        188,
        68,
        [255, 92, 108, 255],
        3,
        "badge 2 core",
    );
    assert_pixel_near(
        &rgba,
        width,
        160,
        68,
        [32, 198, 232, 255],
        3,
        "badge 2 scaled cyan plate",
    );

    // Instance 3 (128, 184) rotated 45 deg & scaled 1.35x: core at (128, 184), rotated corner at (128, 146).
    assert_pixel_near(
        &rgba,
        width,
        128,
        184,
        [255, 92, 108, 255],
        3,
        "badge 3 core",
    );
    assert_pixel_near(
        &rgba,
        width,
        128,
        148,
        [32, 198, 232, 255],
        3,
        "badge 3 rotated top tip",
    );
    assert_pixel_near(&rgba, width, 128, 124, bg_u8, 3, "gap between badges");
}

/// C++ source: impeller/display_list/aiks_dl_basic_unittests.cc:FilledCirclesRenderCorrectly
#[test]
fn aiks_dl_03_ovals_circles_and_arcs() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(16, 18, 26, 255);
    let bg_u8 = [16, 18, 26, 255];

    let mut builder = DisplayListBuilder::new();

    // Top-left: filled gold circle + stroked white ring.
    builder.draw_circle(
        Point::new(64.0, 64.0),
        38.0,
        Paint::fill(Color::make_rgba8(255, 196, 36, 255)),
    );
    builder.draw_circle(
        Point::new(64.0, 64.0),
        48.0,
        Paint::stroke(Color::make_rgba8(240, 245, 255, 255), 6.0),
    );

    // Top-right: filled magenta oval + stroked cyan ring.
    builder.draw_oval(
        Rect::from_xywh(136.0, 32.0, 100.0, 64.0),
        Paint::fill(Color::make_rgba8(224, 64, 188, 255)),
    );
    builder.draw_oval(
        Rect::from_xywh(130.0, 26.0, 112.0, 76.0),
        Paint::stroke(Color::make_rgba8(56, 232, 248, 255), 4.0),
    );

    // Bottom-left: filled emerald pie wedge (use_center = true, 270 deg sweep from 0 to 3*PI/2).
    builder.draw_arc(
        Rect::from_xywh(16.0, 136.0, 96.0, 96.0),
        Radians(0.0),
        Radians(FRAC_PI_2 * 3.0),
        true,
        Paint::fill(Color::make_rgba8(46, 214, 136, 255)),
    );

    // Bottom-right: open stroked orange arc (use_center = false, 180 deg sweep across bottom half).
    builder.draw_arc(
        Rect::from_xywh(140.0, 136.0, 96.0, 96.0),
        Radians(0.0),
        Radians(PI),
        false,
        Paint::stroke(Color::make_rgba8(255, 136, 44, 255), 10.0).with_stroke_cap(Cap::Round),
    );

    let rgba = render_and_save_dl(
        "aiks_dl_03_ovals_circles_and_arcs",
        width,
        height,
        bg,
        &builder.build(),
    );

    // Gold circle interior & white stroke ring.
    assert_pixel_near(
        &rgba,
        width,
        64,
        64,
        [255, 196, 36, 255],
        3,
        "filled gold circle",
    );
    assert_pixel_near(
        &rgba,
        width,
        112,
        64,
        [240, 245, 255, 255],
        10,
        "stroked white circle ring",
    );

    // Magenta oval interior.
    assert_pixel_near(
        &rgba,
        width,
        186,
        64,
        [224, 64, 188, 255],
        3,
        "filled magenta oval",
    );

    // Emerald 270-degree pie wedge (center at 64, 184): bottom-left quadrant is filled, top-right sector is empty.
    assert_pixel_near(
        &rgba,
        width,
        48,
        200,
        [46, 214, 136, 255],
        3,
        "pie wedge filled sector",
    );
    assert_pixel_near(
        &rgba,
        width,
        84,
        164,
        bg_u8,
        3,
        "pie wedge missing 90-deg sector",
    );

    // Stroked arc (center at 188, 184, radius 48): bottom of arc (188, 232) is orange, center (188, 184) is bg.
    assert_pixel_near(
        &rgba,
        width,
        188,
        232,
        [255, 136, 44, 255],
        6,
        "stroked arc bottom apex",
    );
    assert_pixel_near(
        &rgba,
        width,
        188,
        184,
        bg_u8,
        3,
        "open stroked arc hollow center",
    );
}

/// C++ source: impeller/display_list/aiks_dl_basic_unittests.cc:CompareDiffRoundRectAndRoundRect
#[test]
fn aiks_dl_04_rrect_and_drrect_rings() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(14, 16, 22, 255);
    let bg_u8 = [14, 16, 22, 255];

    let mut builder = DisplayListBuilder::new();

    // Outer concentric rounded rect ring via draw_diff_rrect.
    builder.draw_diff_rrect(
        Rect::from_xywh(20.0, 20.0, 216.0, 216.0),
        RoundingRadii::from_single_radius(40.0),
        Rect::from_xywh(48.0, 48.0, 160.0, 160.0),
        RoundingRadii::from_single_radius(24.0),
        Paint::fill(Color::make_rgba8(48, 172, 255, 255)),
    );

    // Middle concentric rounded rect ring via draw_diff_rrect.
    builder.draw_diff_rrect(
        Rect::from_xywh(64.0, 64.0, 128.0, 128.0),
        RoundingRadii::from_single_radius(20.0),
        Rect::from_xywh(88.0, 88.0, 80.0, 80.0),
        RoundingRadii::from_single_radius(12.0),
        Paint::fill(Color::make_rgba8(255, 180, 42, 255)),
    );

    // Innermost solid rounded rect via draw_rrect.
    builder.draw_rrect(
        Rect::from_xywh(104.0, 104.0, 48.0, 48.0),
        RoundingRadii::from_single_radius(12.0),
        Paint::fill(Color::make_rgba8(178, 88, 255, 255)),
    );

    let rgba = render_and_save_dl(
        "aiks_dl_04_rrect_and_drrect_rings",
        width,
        height,
        bg,
        &builder.build(),
    );

    // Outside rounded corner of outer ring -> background.
    assert_pixel_near(&rgba, width, 24, 24, bg_u8, 3, "outside outer rrect corner");
    // Outer ring band (x=34, y=128) -> sky blue.
    assert_pixel_near(
        &rgba,
        width,
        34,
        128,
        [48, 172, 255, 255],
        3,
        "outer drrect ring",
    );
    // Gap between outer and middle ring (x=56, y=128) -> background.
    assert_pixel_near(&rgba, width, 56, 128, bg_u8, 3, "gap between rings 1 and 2");
    // Middle ring band (x=76, y=128) -> amber.
    assert_pixel_near(
        &rgba,
        width,
        76,
        128,
        [255, 180, 42, 255],
        3,
        "middle drrect ring",
    );
    // Gap between middle ring and inner rrect (x=96, y=128) -> background.
    assert_pixel_near(
        &rgba,
        width,
        96,
        128,
        bg_u8,
        3,
        "gap between ring 2 and core",
    );
    // Center core (x=128, y=128) -> violet.
    assert_pixel_near(
        &rgba,
        width,
        128,
        128,
        [178, 88, 255, 255],
        3,
        "inner rrect core",
    );
}

/// C++ source: impeller/display_list/aiks_dl_basic_unittests.cc:CanRenderLinesWithCapsAnglesAndAlphas
#[test]
fn aiks_dl_05_lines_and_caps() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(16, 18, 24, 255);
    let bg_u8 = [16, 18, 24, 255];

    let mut builder = DisplayListBuilder::new();
    let stroke_w: Scalar = 24.0; // half-width = 12.0

    // 1. Butt cap horizontal line at y=52 from x=64..192.
    builder.draw_line(
        Point::new(64.0, 52.0),
        Point::new(192.0, 52.0),
        Paint::stroke(Color::make_rgba8(245, 70, 85, 255), stroke_w).with_stroke_cap(Cap::Butt),
    );

    // 2. Round cap horizontal line at y=112 from x=64..192 (extends cap radius 12 to x=52..204).
    builder.draw_line(
        Point::new(64.0, 112.0),
        Point::new(192.0, 112.0),
        Paint::stroke(Color::make_rgba8(50, 215, 120, 255), stroke_w).with_stroke_cap(Cap::Round),
    );

    // 3. Square cap horizontal line at y=172 from x=64..192 (extends square cap 12 to x=52..204).
    builder.draw_line(
        Point::new(64.0, 172.0),
        Point::new(192.0, 172.0),
        Paint::stroke(Color::make_rgba8(64, 148, 255, 255), stroke_w).with_stroke_cap(Cap::Square),
    );

    // 4. Diagonal round-cap accent line across bottom band.
    builder.draw_line(
        Point::new(48.0, 216.0),
        Point::new(208.0, 236.0),
        Paint::stroke(Color::make_rgba8(255, 204, 56, 255), 12.0)
            .with_stroke_cap(Cap::Round)
            .with_stroke_join(Join::Round),
    );

    let rgba = render_and_save_dl(
        "aiks_dl_05_lines_and_caps",
        width,
        height,
        bg,
        &builder.build(),
    );

    // Butt cap: x=56 (8px before start x=64) is background; x=72 is red.
    assert_pixel_near(&rgba, width, 56, 52, bg_u8, 3, "butt cap does not protrude");
    assert_pixel_near(
        &rgba,
        width,
        72,
        52,
        [245, 70, 85, 255],
        3,
        "butt cap interior",
    );

    // Round cap: centerline at x=56 (8px before start, inside radius 12) is green,
    // while sharp corner (53, 101) (dist = sqrt(11^2 + 11^2) = 15.5 > 12) is background.
    assert_pixel_near(
        &rgba,
        width,
        56,
        112,
        [50, 215, 120, 255],
        3,
        "round cap centerline protrusion",
    );
    assert_pixel_near(
        &rgba,
        width,
        53,
        101,
        bg_u8,
        3,
        "outside round cap curvature",
    );

    // Square cap: both centerline (56, 172) and corner (55, 163) are inside the square cap extension!
    assert_pixel_near(
        &rgba,
        width,
        56,
        172,
        [64, 148, 255, 255],
        3,
        "square cap centerline protrusion",
    );
    assert_pixel_near(
        &rgba,
        width,
        55,
        163,
        [64, 148, 255, 255],
        3,
        "square cap corner protrusion",
    );

    // Diagonal line midpoint (128, 226) is gold.
    assert_pixel_near(
        &rgba,
        width,
        128,
        226,
        [255, 204, 56, 255],
        6,
        "diagonal line midpoint",
    );
}

// =============================================================================
// Group B: 5 Clip & SaveLayer scenes (aiks_dl_clip_unittests.cc & aiks_dl_opacity_unittests.cc)
// =============================================================================

/// C++ source: impeller/display_list/aiks_dl_clip_unittests.cc:CanRenderDifferenceClips
#[test]
fn aiks_dl_06_clip_rect_intersect_and_difference() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(14, 16, 24, 255);
    let bg_u8 = [14, 16, 24, 255];

    let mut builder = DisplayListBuilder::new();
    builder.save();
    builder.clip_rect_with_op(
        Rect::from_xywh(32.0, 32.0, 192.0, 192.0),
        ClipOperation::Intersect,
    );
    builder.clip_rect_with_op(
        Rect::from_xywh(84.0, 84.0, 88.0, 88.0),
        ClipOperation::Difference,
    );
    builder.draw_rect(
        Rect::from_xywh(0.0, 0.0, 256.0, 256.0),
        Paint::fill(Color::make_rgba8(32, 208, 188, 255)),
    );
    builder.restore();

    let rgba = render_and_save_dl(
        "aiks_dl_06_clip_rect_intersect_and_difference",
        width,
        height,
        bg,
        &builder.build(),
    );

    // Outside outer Intersect clip -> background.
    assert_pixel_near(&rgba, width, 16, 128, bg_u8, 2, "outside intersect clip");
    // Inside outer Intersect clip, outside inner Difference clip -> teal.
    assert_pixel_near(
        &rgba,
        width,
        56,
        56,
        [32, 208, 188, 255],
        3,
        "clipped frame top-left",
    );
    assert_pixel_near(
        &rgba,
        width,
        200,
        200,
        [32, 208, 188, 255],
        3,
        "clipped frame bottom-right",
    );
    // Inside inner Difference hole -> background.
    assert_pixel_near(
        &rgba,
        width,
        128,
        128,
        bg_u8,
        2,
        "inside difference rect hole",
    );
}

/// C++ source: impeller/display_list/aiks_dl_clip_unittests.cc:CanRenderNestedClips
#[test]
fn aiks_dl_07_clip_rrect_and_path() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(15, 17, 25, 255);
    let bg_u8 = [15, 17, 25, 255];

    let mut diamond_pb = PathBuilder::new();
    diamond_pb
        .move_to(Point::new(128.0, 12.0))
        .line_to(Point::new(244.0, 128.0))
        .line_to(Point::new(128.0, 244.0))
        .line_to(Point::new(12.0, 128.0))
        .close();
    let diamond_path = diamond_pb.build(FillType::NonZero);

    let mut hole_pb = PathBuilder::new();
    hole_pb
        .move_to(Point::new(100.0, 100.0))
        .line_to(Point::new(156.0, 100.0))
        .line_to(Point::new(156.0, 156.0))
        .line_to(Point::new(100.0, 156.0))
        .close();
    let hole_path = hole_pb.build(FillType::NonZero);

    let mut builder = DisplayListBuilder::new();
    builder.save();
    builder.clip_rrect(
        Rect::from_xywh(28.0, 28.0, 200.0, 200.0),
        RoundingRadii::from_single_radius(44.0),
        ClipOperation::Intersect,
    );
    builder.clip_path(diamond_path, ClipOperation::Intersect);
    builder.clip_path(hole_path, ClipOperation::Difference);

    // Fill two vibrant horizontal bands across the whole canvas.
    builder.draw_rect(
        Rect::from_xywh(0.0, 0.0, 256.0, 128.0),
        Paint::fill(Color::make_rgba8(248, 108, 72, 255)),
    );
    builder.draw_rect(
        Rect::from_xywh(0.0, 128.0, 256.0, 128.0),
        Paint::fill(Color::make_rgba8(92, 136, 255, 255)),
    );
    builder.restore();

    let rgba = render_and_save_dl(
        "aiks_dl_07_clip_rrect_and_path",
        width,
        height,
        bg,
        &builder.build(),
    );

    // Center (128, 128) is inside hole_path (Difference clip) -> background.
    assert_pixel_near(&rgba, width, 128, 128, bg_u8, 2, "center difference hole");
    // Top-center (128, 72) is inside both rrect and diamond, outside hole -> coral.
    assert_pixel_near(
        &rgba,
        width,
        128,
        72,
        [248, 108, 72, 255],
        3,
        "upper clipped region",
    );
    // Bottom-center (128, 184) is inside both rrect and diamond, outside hole -> periwinkle blue.
    assert_pixel_near(
        &rgba,
        width,
        128,
        184,
        [92, 136, 255, 255],
        3,
        "lower clipped region",
    );
    // Point (48, 48) is inside rrect but outside diamond (|48-128| + |48-128| = 160 > 116) -> background.
    assert_pixel_near(
        &rgba,
        width,
        48,
        48,
        bg_u8,
        3,
        "inside rrect but outside diamond path clip",
    );
    // Point (128, 20) is inside diamond tip but outside rrect top (y < 28) -> background.
    assert_pixel_near(
        &rgba,
        width,
        128,
        20,
        bg_u8,
        3,
        "inside diamond tip but outside rrect clip",
    );
}

/// C++ source: impeller/display_list/aiks_dl_basic_unittests.cc:CanRenderFilledRoundSuperellipses
#[test]
fn aiks_dl_08_clip_round_superellipse() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(14, 16, 24, 255);
    let bg_u8 = [14, 16, 24, 255];

    let squircle =
        RoundSuperellipse::from_rect_radius(Rect::from_xywh(24.0, 24.0, 208.0, 208.0), 68.0);

    let mut builder = DisplayListBuilder::new();
    builder.save();
    builder.clip_round_superellipse(squircle, ClipOperation::Intersect);

    // Vibrant 4-quadrant pattern clipped by the round superellipse.
    builder.draw_rect(
        Rect::from_xywh(0.0, 0.0, 128.0, 128.0),
        Paint::fill(Color::make_rgba8(255, 86, 112, 255)),
    );
    builder.draw_rect(
        Rect::from_xywh(128.0, 0.0, 128.0, 128.0),
        Paint::fill(Color::make_rgba8(44, 212, 164, 255)),
    );
    builder.draw_rect(
        Rect::from_xywh(0.0, 128.0, 128.0, 128.0),
        Paint::fill(Color::make_rgba8(255, 196, 48, 255)),
    );
    builder.draw_rect(
        Rect::from_xywh(128.0, 128.0, 128.0, 128.0),
        Paint::fill(Color::make_rgba8(96, 124, 255, 255)),
    );
    builder.restore();

    let rgba = render_and_save_dl(
        "aiks_dl_08_clip_round_superellipse",
        width,
        height,
        bg,
        &builder.build(),
    );

    // All 4 interior quadrants must show their vibrant colors.
    assert_pixel_near(
        &rgba,
        width,
        80,
        80,
        [255, 86, 112, 255],
        3,
        "squircle Q1 interior",
    );
    assert_pixel_near(
        &rgba,
        width,
        176,
        80,
        [44, 212, 164, 255],
        3,
        "squircle Q2 interior",
    );
    assert_pixel_near(
        &rgba,
        width,
        80,
        176,
        [255, 196, 48, 255],
        3,
        "squircle Q3 interior",
    );
    assert_pixel_near(
        &rgba,
        width,
        176,
        176,
        [96, 124, 255, 255],
        3,
        "squircle Q4 interior",
    );

    // Sharp bounding-box corner (28, 28) is clipped out by the superellipse corner curvature.
    assert_pixel_near(
        &rgba,
        width,
        28,
        28,
        bg_u8,
        3,
        "outside squircle corner curvature",
    );
    // Exterior of bounding box -> background.
    assert_pixel_near(&rgba, width, 12, 128, bg_u8, 2, "outside squircle bounds");
}

/// C++ source: impeller/display_list/aiks_dl_opacity_unittests.cc:CanRenderGroupOpacity
#[test]
fn aiks_dl_09_save_layer_group_opacity() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::BLACK;

    let mut builder = DisplayListBuilder::new();

    // Left half: SaveLayer with 0.5 alpha over two overlapping opaque white rects.
    // Because they composite in an offscreen layer first, the overlap region has uniform 0.5 alpha!
    builder.save_layer(
        Some(Rect::from_xywh(16.0, 24.0, 104.0, 208.0)),
        Some(Paint::fill(Color::new(1.0, 1.0, 1.0, 0.5))),
    );
    builder.draw_rect(
        Rect::from_xywh(20.0, 36.0, 64.0, 104.0),
        Paint::fill(Color::WHITE),
    );
    builder.draw_rect(
        Rect::from_xywh(48.0, 100.0, 64.0, 104.0),
        Paint::fill(Color::WHITE),
    );
    builder.restore();

    // Right half: No SaveLayer, drawing two individual 0.5-alpha white rects.
    // Their overlap region double-blends to 0.75 alpha (~191).
    let half_white = Paint::fill(Color::new(1.0, 1.0, 1.0, 0.5));
    builder.draw_rect(
        Rect::from_xywh(144.0, 36.0, 64.0, 104.0),
        half_white.clone(),
    );
    builder.draw_rect(Rect::from_xywh(172.0, 100.0, 64.0, 104.0), half_white);

    let rgba = render_and_save_dl(
        "aiks_dl_09_save_layer_group_opacity",
        width,
        height,
        bg,
        &builder.build(),
    );

    let left_single = get_pixel(&rgba, width, 36, 60);
    let left_overlap = get_pixel(&rgba, width, 64, 120);
    assert!(
        (120..=135).contains(&left_single[0]),
        "SaveLayer single rect should be ~50% white (got {left_single:?})"
    );
    assert!(
        left_single[0].abs_diff(left_overlap[0]) <= 2,
        "SaveLayer group opacity must not double-blend overlap: single={left_single:?}, overlap={left_overlap:?}"
    );

    let right_single = get_pixel(&rgba, width, 160, 60);
    let right_overlap = get_pixel(&rgba, width, 188, 120);
    assert!(
        right_overlap[0] >= right_single[0] + 50,
        "Ungrouped overlap should double-blend brighter than single rect: single={right_single:?}, overlap={right_overlap:?}"
    );
}

/// C++ source: impeller/display_list/aiks_dl_blend_unittests.cc:CanRenderAdvancedBlendColorFilterWithSaveLayer
#[test]
fn aiks_dl_10_save_layer_color_filter_matrix() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(14, 16, 22, 255);

    // 4x5 row-major color matrix rotating RGB channels: R' = B, G' = R, B' = G, A' = A.
    let rgb_rotate_matrix: [f32; 20] = [
        0.0, 0.0, 1.0, 0.0, 0.0, // R' = B
        1.0, 0.0, 0.0, 0.0, 0.0, // G' = R
        0.0, 1.0, 0.0, 0.0, 0.0, // B' = G
        0.0, 0.0, 0.0, 1.0, 0.0, // A' = A
    ];
    let layer_paint =
        Paint::fill(Color::WHITE).with_color_filter(ColorFilter::Matrix(rgb_rotate_matrix));

    let mut builder = DisplayListBuilder::new();
    builder.save_layer(
        Some(Rect::from_xywh(16.0, 16.0, 224.0, 224.0)),
        Some(layer_paint),
    );
    // Child 1: pure red rect -> becomes pure green after SaveLayer ColorFilter::Matrix.
    builder.draw_rect(
        Rect::from_xywh(28.0, 28.0, 84.0, 84.0),
        Paint::fill(Color::new(1.0, 0.0, 0.0, 1.0)),
    );
    // Child 2: pure green circle -> becomes pure blue.
    builder.draw_circle(
        Point::new(186.0, 70.0),
        42.0,
        Paint::fill(Color::new(0.0, 1.0, 0.0, 1.0)),
    );
    // Child 3: pure blue rounded rect -> becomes pure red.
    builder.draw_rrect(
        Rect::from_xywh(56.0, 140.0, 144.0, 76.0),
        RoundingRadii::from_single_radius(18.0),
        Paint::fill(Color::new(0.0, 0.0, 1.0, 1.0)),
    );
    builder.restore();

    let rgba = render_and_save_dl(
        "aiks_dl_10_save_layer_color_filter_matrix",
        width,
        height,
        bg,
        &builder.build(),
    );

    // Red input (70, 70) -> Green output.
    assert_pixel_near(
        &rgba,
        width,
        70,
        70,
        [0, 255, 0, 255],
        4,
        "red child transformed to green",
    );
    // Green input (186, 70) -> Blue output.
    assert_pixel_near(
        &rgba,
        width,
        186,
        70,
        [0, 0, 255, 255],
        4,
        "green child transformed to blue",
    );
    // Blue input (128, 178) -> Red output.
    assert_pixel_near(
        &rgba,
        width,
        128,
        178,
        [255, 0, 0, 255],
        4,
        "blue child transformed to red",
    );
}

// =============================================================================
// Group C: 5 Gradient scenes (aiks_dl_gradient_unittests.cc)
// =============================================================================

/// C++ source: impeller/display_list/aiks_dl_gradient_unittests.cc:CanRenderLinearGradientClamp
#[test]
fn aiks_dl_11_linear_gradient_tile_modes() {
    let width = 320u32;
    let height = 240u32;
    let bg = Color::make_rgba8(16, 18, 26, 255);
    let bg_u8 = [16, 18, 26, 255];

    let mut builder = DisplayListBuilder::new();
    let modes = [
        (16.0, TileMode::Clamp),
        (92.0, TileMode::Repeat),
        (168.0, TileMode::Mirror),
        (244.0, TileMode::Decal),
    ];

    let red = Color::make_rgba8(240, 40, 60, 255);
    let blue = Color::make_rgba8(40, 120, 250, 255);

    for (x0, mode) in modes {
        // Gradient spans only the middle 20px of the 60px strip: [x0 + 20, x0 + 40].
        let grad = LinearGradientParams::two_stop(
            Point::new(x0 + 20.0, 120.0),
            Point::new(x0 + 40.0, 120.0),
            red,
            blue,
            mode,
        );
        builder.draw_rrect(
            Rect::from_xywh(x0, 24.0, 60.0, 192.0),
            RoundingRadii::from_single_radius(8.0),
            Paint::fill(Color::WHITE).with_color_source(ColorSource::LinearGradient(grad)),
        );
    }

    let rgba = render_and_save_dl(
        "aiks_dl_11_linear_gradient_tile_modes",
        width,
        height,
        bg,
        &builder.build(),
    );

    // 1. Clamp strip (x0 = 16): left wing (x=22, t < 0) clamps to red; right wing (x=70, t > 1) clamps to blue.
    assert_pixel_near(
        &rgba,
        width,
        22,
        120,
        [240, 40, 60, 255],
        5,
        "Clamp left wing",
    );
    assert_pixel_near(
        &rgba,
        width,
        70,
        120,
        [40, 120, 250, 255],
        5,
        "Clamp right wing",
    );

    // 2. Repeat strip (x0 = 92, grad 112..132): at x=136 (t = 1.2 -> wraps to 0.2), red dominates over blue.
    let repeat_wrap = get_pixel(&rgba, width, 136, 120);
    assert!(
        repeat_wrap[0] > repeat_wrap[2] + 60,
        "Repeat tile mode at t=1.2 should wrap back to red-dominant, got {repeat_wrap:?}"
    );

    // 3. Mirror strip (x0 = 168, grad 188..208): at x=212 (t = 1.2 -> mirrors to 0.8), blue dominates;
    // at x=226 (t = 1.9 -> mirrors to 0.1), red dominates.
    let mirror_near = get_pixel(&rgba, width, 212, 120);
    let mirror_far = get_pixel(&rgba, width, 226, 120);
    assert!(
        mirror_near[2] > mirror_near[0] + 60,
        "Mirror tile mode at t=1.2 should mirror to 0.8 (blue-dominant), got {mirror_near:?}"
    );
    assert!(
        mirror_far[0] > mirror_far[2] + 60,
        "Mirror tile mode at t=1.9 should mirror to 0.1 (red-dominant), got {mirror_far:?}"
    );

    // 4. Decal strip (x0 = 244, grad 264..284): outside [0, 1] is transparent (canvas background),
    // while midpoint x=274 (t = 0.5) is blended purple!
    assert_pixel_near(
        &rgba,
        width,
        250,
        120,
        bg_u8,
        3,
        "Decal left wing transparent to bg",
    );
    assert_pixel_near(
        &rgba,
        width,
        298,
        120,
        bg_u8,
        3,
        "Decal right wing transparent to bg",
    );
    let decal_mid = get_pixel(&rgba, width, 274, 120);
    assert!(
        max_channel_delta(decal_mid, bg_u8) > 100,
        "Decal center (t=0.5) must be visible gradient color, got {decal_mid:?}"
    );
}

/// C++ source: impeller/display_list/aiks_dl_gradient_unittests.cc:CanRenderRadialGradientManyColors
#[test]
fn aiks_dl_12_radial_gradient_multistop() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(14, 16, 22, 255);

    let center = Point::new(128.5, 128.5);
    let radius = 100.0;
    let stops = vec![
        GradientStop::new(0.0, Color::make_rgba8(255, 255, 255, 255)),
        GradientStop::new(0.18, Color::make_rgba8(255, 220, 20, 255)),
        GradientStop::new(0.45, Color::make_rgba8(245, 45, 70, 255)),
        GradientStop::new(0.78, Color::make_rgba8(20, 215, 240, 255)),
        GradientStop::new(1.0, Color::make_rgba8(30, 40, 170, 255)),
    ];

    let grad = RadialGradientParams::new(center, radius, stops, TileMode::Clamp, 1.0);
    let mut builder = DisplayListBuilder::new();
    builder.draw_circle(
        center,
        108.0,
        Paint::fill(Color::WHITE).with_color_source(ColorSource::RadialGradient(grad)),
    );

    let rgba = render_and_save_dl(
        "aiks_dl_12_radial_gradient_multistop",
        width,
        height,
        bg,
        &builder.build(),
    );

    // Center (r = 0): white.
    assert_pixel_near(
        &rgba,
        width,
        128,
        128,
        [255, 255, 255, 255],
        6,
        "radial stop 0.0 white center",
    );
    // r = 18 -> (146, 128): yellow (high R & G, low B).
    let px_yellow = get_pixel(&rgba, width, 146, 128);
    assert!(
        px_yellow[0] > 220 && px_yellow[1] > 190 && px_yellow[2] < 60,
        "radial stop 0.18 should be yellow, got {px_yellow:?}"
    );
    // r = 45 -> (173, 128): red (high R, low G & B).
    let px_red = get_pixel(&rgba, width, 173, 128);
    assert!(
        px_red[0] > 210 && px_red[1] < 80 && px_red[2] < 100,
        "radial stop 0.45 should be red, got {px_red:?}"
    );
    // r = 78 -> (206, 128): cyan (low R, high G & B).
    let px_cyan = get_pixel(&rgba, width, 206, 128);
    assert!(
        px_cyan[0] < 60 && px_cyan[1] > 180 && px_cyan[2] > 200,
        "radial stop 0.78 should be cyan, got {px_cyan:?}"
    );
    // r = 102 -> (230, 128): clamped outer deep blue.
    let px_blue = get_pixel(&rgba, width, 230, 128);
    assert!(
        px_blue[2] > 140 && px_blue[0] < 60 && px_blue[1] < 70,
        "radial stop 1.0 should be deep blue, got {px_blue:?}"
    );
}

/// C++ source: impeller/display_list/aiks_dl_gradient_unittests.cc:CanRenderSweepGradientClamp
#[test]
fn aiks_dl_13_sweep_gradient_angles() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(14, 16, 24, 255);

    let center = Point::new(128.0, 128.0);
    let stops = vec![
        GradientStop::new(0.0, Color::make_rgba8(255, 50, 70, 255)),
        GradientStop::new(0.33, Color::make_rgba8(50, 230, 120, 255)),
        GradientStop::new(0.66, Color::make_rgba8(50, 140, 255, 255)),
        GradientStop::new(1.0, Color::make_rgba8(255, 200, 40, 255)),
    ];

    // Sweep from 0 deg to 270 deg with Clamp tile mode.
    let sweep = SweepGradientParams::new(center, 0.0, 270.0, stops, TileMode::Clamp, 1.0);

    let mut builder = DisplayListBuilder::new();
    builder.draw_diff_rrect(
        Rect::from_xywh(24.0, 24.0, 208.0, 208.0),
        RoundingRadii::from_single_radius(104.0),
        Rect::from_xywh(68.0, 68.0, 120.0, 120.0),
        RoundingRadii::from_single_radius(60.0),
        Paint::fill(Color::WHITE).with_color_source(ColorSource::SweepGradient(sweep)),
    );

    let rgba = render_and_save_dl(
        "aiks_dl_13_sweep_gradient_angles",
        width,
        height,
        bg,
        &builder.build(),
    );

    // Sample 4 cardinal points around the ring (radius = 82 from center 128,128):
    // +X (210, 128), +Y (128, 210), -X (46, 128), -Y (128, 46).
    let px_right = get_pixel(&rgba, width, 210, 128);
    let px_bottom = get_pixel(&rgba, width, 128, 210);
    let px_left = get_pixel(&rgba, width, 46, 128);
    let px_top = get_pixel(&rgba, width, 128, 46);

    // Verify all 4 cardinal ring points rendered non-background and have distinct angular colors!
    let bg_u8 = [14, 16, 24, 255];
    for (name, px) in [
        ("right", px_right),
        ("bottom", px_bottom),
        ("left", px_left),
        ("top", px_top),
    ] {
        assert!(
            max_channel_delta(px, bg_u8) > 120,
            "sweep ring cardinal point {name} should be vibrant, got {px:?}"
        );
    }
    assert!(
        max_channel_delta(px_right, px_bottom) > 80
            && max_channel_delta(px_bottom, px_left) > 80
            && max_channel_delta(px_left, px_top) > 80,
        "sweep gradient must progress across angles: right={px_right:?}, bottom={px_bottom:?}, left={px_left:?}, top={px_top:?}"
    );
    // Hollow center of diff_rrect ring must be background.
    assert_pixel_near(&rgba, width, 128, 128, bg_u8, 2, "sweep ring hollow center");
}

/// C++ source: impeller/display_list/aiks_dl_gradient_unittests.cc:CanRenderConicalGradient
#[test]
fn aiks_dl_14_two_point_conical_gradient() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(14, 16, 24, 255);

    // Spotlight conical gradient from off-center small focal circle to larger outer circle.
    let start_center = Point::new(88.0, 92.0);
    let start_radius = 12.0;
    let end_center = Point::new(140.0, 140.0);
    let end_radius = 96.0;
    let stops = vec![
        GradientStop::new(0.0, Color::make_rgba8(255, 250, 210, 255)),
        GradientStop::new(0.45, Color::make_rgba8(255, 110, 50, 255)),
        GradientStop::new(1.0, Color::make_rgba8(36, 28, 120, 255)),
    ];

    let conical = ConicalGradientParams::new(
        start_center,
        start_radius,
        end_center,
        end_radius,
        stops,
        TileMode::Clamp,
        1.0,
    );

    let mut builder = DisplayListBuilder::new();
    builder.draw_rrect(
        Rect::from_xywh(20.0, 20.0, 216.0, 216.0),
        RoundingRadii::from_single_radius(28.0),
        Paint::fill(Color::WHITE).with_color_source(ColorSource::ConicalGradient(conical)),
    );

    let rgba = render_and_save_dl(
        "aiks_dl_14_two_point_conical_gradient",
        width,
        height,
        bg,
        &builder.build(),
    );

    // Focal highlight near start_center (88, 92) should be bright warm ivory/yellow.
    let focal_px = get_pixel(&rgba, width, 88, 92);
    assert!(
        focal_px[0] > 235 && focal_px[1] > 215 && focal_px[2] > 160,
        "conical focal highlight at (88, 92) should be bright warm ivory, got {focal_px:?}"
    );

    // Mid-cone sample (140, 140) should be warm orange (R > G > B).
    let mid_px = get_pixel(&rgba, width, 140, 140);
    assert!(
        mid_px[0] > 180 && mid_px[0] > mid_px[2] + 60,
        "conical mid-cone at (140, 140) should be warm orange, got {mid_px:?}"
    );

    // Outer edge sample (216, 216) should clamp toward deep indigo (B > R, B > G).
    let edge_px = get_pixel(&rgba, width, 216, 216);
    assert!(
        edge_px[2] > edge_px[0] && edge_px[2] > edge_px[1],
        "conical outer edge at (216, 216) should be indigo/blue-dominant, got {edge_px:?}"
    );
}

/// C++ source: impeller/display_list/aiks_dl_gradient_unittests.cc:GradientStrokesRenderCorrectly
#[test]
fn aiks_dl_15_gradient_with_local_transform() {
    let width = 256u32;
    let height = 256u32;
    let bg = Color::make_rgba8(15, 17, 24, 255);
    let bg_u8 = [15, 17, 24, 255];

    // Local linear gradient defined along local X from (-60, 0) [Red] to (+60, 0) [Cyan].
    let local_grad = LinearGradientParams::two_stop(
        Point::new(-60.0, 0.0),
        Point::new(60.0, 0.0),
        Color::make_rgba8(250, 50, 80, 255),
        Color::make_rgba8(30, 225, 245, 255),
        TileMode::Clamp,
    );

    let mut builder = DisplayListBuilder::new();

    // Rotate the canvas by +90 degrees around (128, 128) and scale by (1.2, 1.0):
    // Local (-60, 0) maps to device (128, 128 - 72) = (128, 56) [Red],
    // and local (+60, 0) maps to device (128, 128 + 72) = (128, 200) [Cyan]!
    builder.save();
    builder.translate(128.0, 128.0);
    builder.rotate(Radians(FRAC_PI_2));
    builder.scale(1.2, 1.0);

    builder.draw_rrect(
        Rect::from_xywh(-60.0, -44.0, 120.0, 88.0),
        RoundingRadii::from_single_radius(16.0),
        Paint::fill(Color::WHITE)
            .with_color_source(ColorSource::LinearGradient(local_grad.clone())),
    );

    // Also draw a gradient-stroked outer frame in the same rotated/scaled local space.
    builder.draw_rrect(
        Rect::from_xywh(-74.0, -58.0, 148.0, 116.0),
        RoundingRadii::from_single_radius(22.0),
        Paint::stroke(Color::WHITE, 8.0).with_color_source(ColorSource::LinearGradient(local_grad)),
    );
    builder.restore();

    let rgba = render_and_save_dl(
        "aiks_dl_15_gradient_with_local_transform",
        width,
        height,
        bg,
        &builder.build(),
    );

    // Because the local X-axis was rotated +90 deg onto device +Y, the gradient now runs vertically!
    // Device top of rect (128, 68) corresponds to negative local X -> Red-dominant.
    let top_px = get_pixel(&rgba, width, 128, 68);
    assert!(
        top_px[0] > 200 && top_px[2] < 120,
        "rotated local gradient top (128, 68) should be red-dominant, got {top_px:?}"
    );

    // Device bottom of rect (128, 188) corresponds to positive local X -> Cyan-dominant.
    let bottom_px = get_pixel(&rgba, width, 128, 188);
    assert!(
        bottom_px[1] > 180 && bottom_px[2] > 200 && bottom_px[0] < 80,
        "rotated local gradient bottom (128, 188) should be cyan-dominant, got {bottom_px:?}"
    );

    // Device left & right at the same Y (e.g. y=128, x=96 vs x=160) have the same local X=0 -> equal color!
    let mid_left = get_pixel(&rgba, width, 96, 128);
    let mid_right = get_pixel(&rgba, width, 160, 128);
    assert!(
        max_channel_delta(mid_left, mid_right) <= 4,
        "rotated local gradient should be constant along device X at y=128: left={mid_left:?}, right={mid_right:?}"
    );

    // Gap between inner filled rrect and outer gradient-stroked frame (128, 46) is background,
    // while outer gradient-stroked frame at (128, 40) is red-dominant!
    assert_pixel_near(
        &rgba,
        width,
        128,
        48,
        bg_u8,
        4,
        "gap between fill and stroke frame",
    );
    let top_stroke = get_pixel(&rgba, width, 128, 39);
    assert!(
        top_stroke[0] > 200,
        "gradient-stroked outer frame top at (128, 39) should be red, got {top_stroke:?}"
    );
}
