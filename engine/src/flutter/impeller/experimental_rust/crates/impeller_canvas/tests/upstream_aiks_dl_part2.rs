#![forbid(unsafe_code)]

use impeller_canvas::{
    save_rgba_png, BlendMode, BuiltinFont, Canvas, Cap, Color, ColorFilter, ColorSource,
    DisplayList, DisplayListBuilder, FillType, ImageFilter, ImageTexture, Join, Matrix, Paint,
    Path, PathBuilder, Point, Rect, RoundSuperellipse, RoundingRadii, Scalar, Size, TextStyle,
    TileMode, WgpuContext,
};
use std::path::Path as FsPath;
use std::sync::{Mutex, MutexGuard, OnceLock};

const SCRATCH_ROOT: &str =
    "/usr/local/google/home/kevmoo/.gemini/jetski/brain/aaf8f361-7fe2-4fe7-8caa-40ddbfdb8335/scratch";

fn shared_renderer() -> MutexGuard<'static, WgpuContext> {
    static RENDERER: OnceLock<Mutex<WgpuContext>> = OnceLock::new();
    RENDERER
        .get_or_init(|| Mutex::new(WgpuContext::new_headless().expect("headless wgpu renderer")))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn render_dl_and_save_golden(
    scene_name: &str,
    width: u32,
    height: u32,
    clear_color: Color,
    dl: &DisplayList,
) -> Vec<u8> {
    let rgba = {
        let renderer = shared_renderer();
        let mut canvas = Canvas::new(width, height, clear_color);
        canvas.draw_display_list(dl);
        canvas
            .render_to_pixels(&renderer)
            .expect("render display list to pixels")
    };
    assert_eq!(rgba.len(), (width * height * 4) as usize);

    let scratch_dir = FsPath::new(SCRATCH_ROOT);
    if scratch_dir.exists() {
        let goldens_dir = scratch_dir.join("aiks_dl_goldens");
        std::fs::create_dir_all(&goldens_dir).expect("create aiks_dl_goldens directory");
        let png_path = goldens_dir.join(format!("{scene_name}.png"));
        save_rgba_png(&png_path, width, height, &rgba).expect("save golden PNG");
    }
    rgba
}

fn get_pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let idx = ((y * width + x) as usize) * 4;
    [rgba[idx], rgba[idx + 1], rgba[idx + 2], rgba[idx + 3]]
}

fn color_to_rgba8(c: Color) -> [u8; 4] {
    let pm = c.premultiply();
    [
        (pm.red.clamp(0.0, 1.0) * 255.0).round() as u8,
        (pm.green.clamp(0.0, 1.0) * 255.0).round() as u8,
        (pm.blue.clamp(0.0, 1.0) * 255.0).round() as u8,
        (pm.alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

fn max_channel_delta(a: [u8; 4], b: [u8; 4]) -> u8 {
    a[0].abs_diff(b[0])
        .max(a[1].abs_diff(b[1]))
        .max(a[2].abs_diff(b[2]))
        .max(a[3].abs_diff(b[3]))
}

fn assert_pixel_near(actual: [u8; 4], expected: [u8; 4], tol: u8, label: &str) {
    let d = max_channel_delta(actual, expected);
    assert!(
        d <= tol,
        "{label}: pixel {actual:?} differed from expected {expected:?} by {d} (tol={tol})"
    );
}

fn count_non_bg_in_rect(
    rgba: &[u8],
    width: u32,
    bounds: [u32; 4],
    bg: [u8; 4],
    threshold: u8,
) -> usize {
    let [x0, y0, x1, y1] = bounds;
    let mut count = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            if max_channel_delta(get_pixel(rgba, width, x, y), bg) > threshold {
                count += 1;
            }
        }
    }
    count
}

// =============================================================================
// Group D: 5 Blur, MaskFilter & BackdropFilter scenes (aiks_dl_blur_unittests.cc)
// =============================================================================

/// C++ source: impeller/display_list/aiks_dl_blur_unittests.cc:MaskBlurVariantTestNormalTranslucent
#[test]
fn aiks_dl_16_rrect_mask_blur_shadow() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.08, 0.09, 0.13, 1.0);
    let bg_u8 = color_to_rgba8(bg);

    let mut builder = DisplayListBuilder::new();

    // Left card: analytic RRect Gaussian drop shadow + crisp foreground RRect.
    builder.draw_rrect_shadow(
        Rect::from_xywh(36.0, 58.0, 108.0, 128.0),
        20.0,
        12.0,
        Color::new(0.0, 0.85, 1.0, 0.88),
    );
    let fg_left = Color::new(0.16, 0.22, 0.34, 1.0);
    builder.draw_rrect(
        Rect::from_xywh(28.0, 48.0, 108.0, 128.0),
        RoundingRadii::from_single_radius(20.0),
        Paint::fill(fg_left),
    );

    // Right card: Paint mask_blur_sigma translucent glow + crisp foreground RRect.
    builder.draw_rrect(
        Rect::from_xywh(182.0, 58.0, 108.0, 128.0),
        RoundingRadii::from_single_radius(20.0),
        Paint::fill(Color::new(1.0, 0.32, 0.68, 0.88)).with_mask_blur_sigma(10.0),
    );
    let fg_right = Color::new(0.24, 0.16, 0.28, 1.0);
    builder.draw_rrect(
        Rect::from_xywh(174.0, 48.0, 108.0, 128.0),
        RoundingRadii::from_single_radius(20.0),
        Paint::fill(fg_right),
    );

    let dl = builder.build();
    let rgba =
        render_dl_and_save_golden("aiks_dl_16_rrect_mask_blur_shadow", width, height, bg, &dl);

    // Untouched corner matches background.
    assert_pixel_near(get_pixel(&rgba, width, 6, 6), bg_u8, 2, "top-left bg");

    // Foreground RRect interiors are crisp and unblurred.
    assert_pixel_near(
        get_pixel(&rgba, width, 82, 112),
        color_to_rgba8(fg_left),
        3,
        "left foreground RRect center",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 228, 112),
        color_to_rgba8(fg_right),
        3,
        "right foreground RRect center",
    );

    // Cyan shadow halo to the right of the left card (x=142, y=120) glows and decays outward.
    let halo_near_left = get_pixel(&rgba, width, 142, 120);
    let halo_far_left = get_pixel(&rgba, width, 156, 120);
    assert!(
        halo_near_left[2] > bg_u8[2] + 35,
        "expected cyan shadow glow at (142, 120), got {halo_near_left:?}"
    );
    assert!(
        halo_near_left[2] > halo_far_left[2],
        "expected Gaussian shadow falloff from x=142 ({halo_near_left:?}) to x=156 ({halo_far_left:?})"
    );

    // Magenta mask blur halo to the right of the right card (x=288, y=120) glows and decays outward.
    let halo_near_right = get_pixel(&rgba, width, 288, 120);
    let halo_far_right = get_pixel(&rgba, width, 302, 120);
    assert!(
        halo_near_right[0] > bg_u8[0] + 30,
        "expected magenta mask-blur glow at (288, 120), got {halo_near_right:?}"
    );
    assert!(
        halo_near_right[0] > halo_far_right[0],
        "expected mask-blur falloff from x=288 ({halo_near_right:?}) to x=302 ({halo_far_right:?})"
    );
}

/// C++ source: impeller/display_list/aiks_dl_blur_unittests.cc:CanRenderClippedBlur
#[test]
fn aiks_dl_17_gaussian_blur_image_filter_save_layer() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.07, 0.08, 0.12, 1.0);
    let bg_u8 = color_to_rgba8(bg);

    let mut builder = DisplayListBuilder::new();
    let blur_paint = Paint::fill(Color::WHITE).with_image_filter(ImageFilter::Blur {
        sigma_x: 8.0,
        sigma_y: 8.0,
        tile_mode: TileMode::Decal,
    });
    builder.save_layer(
        Some(Rect::from_xywh(24.0, 24.0, 272.0, 192.0)),
        Some(blur_paint),
    );
    builder.draw_rect(
        Rect::from_xywh(56.0, 60.0, 88.0, 120.0),
        Paint::fill(Color::new(0.10, 0.95, 0.42, 1.0)),
    );
    builder.draw_circle(
        Point::new(220.0, 120.0),
        48.0,
        Paint::fill(Color::new(1.0, 0.45, 0.12, 1.0)),
    );
    builder.restore();

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_17_gaussian_blur_image_filter_save_layer",
        width,
        height,
        bg,
        &dl,
    );

    // Deep interior of green rect retains strong green luminance.
    let green_center = get_pixel(&rgba, width, 100, 120);
    assert!(
        green_center[1] > 200 && green_center[0] < 60,
        "expected blurred green core at (100, 120), got {green_center:?}"
    );

    // Deep interior of orange circle retains strong orange luminance.
    let orange_center = get_pixel(&rgba, width, 220, 120);
    assert!(
        orange_center[0] > 200 && orange_center[1] > 90,
        "expected blurred orange core at (220, 120), got {orange_center:?}"
    );

    // Nominal left edge of green rect (x=56, y=120) is smoothly ~50% blurred with background.
    let green_edge = get_pixel(&rgba, width, 56, 120);
    assert!(
        green_edge[1] > bg_u8[1] + 50 && green_edge[1] < green_center[1] - 30,
        "expected intermediate Gaussian-blurred edge at (56, 120), got {green_edge:?}"
    );

    // Halo outside nominal rect (x=48, y=120) receives blurred green bleed.
    let green_halo = get_pixel(&rgba, width, 48, 120);
    assert!(
        green_halo[1] > bg_u8[1] + 15 && green_halo[1] < green_edge[1],
        "expected outward Gaussian halo bleed at (48, 120), got {green_halo:?}"
    );
}

/// C++ source: impeller/display_list/aiks_dl_blur_unittests.cc:CanRenderBackdropBlur
#[test]
fn aiks_dl_18_backdrop_filter_frosted_glass() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.05, 0.06, 0.09, 1.0);

    let mut builder = DisplayListBuilder::new();
    let crimson = Color::new(0.92, 0.18, 0.28, 1.0);
    let cyan = Color::new(0.10, 0.82, 0.95, 1.0);
    for i in 0..8 {
        let stripe_color = if i % 2 == 0 { crimson } else { cyan };
        builder.draw_rect(
            Rect::from_xywh((i as Scalar) * 40.0, 0.0, 40.0, 240.0),
            Paint::fill(stripe_color),
        );
    }

    let glass_rect = Rect::from_xywh(60.0, 56.0, 200.0, 128.0);
    builder.save();
    builder.clip_rect(glass_rect);
    builder.save_layer_with_backdrop(
        Some(glass_rect),
        None,
        Some(ImageFilter::Blur {
            sigma_x: 10.0,
            sigma_y: 10.0,
            tile_mode: TileMode::Clamp,
        }),
    );
    builder.draw_rect(glass_rect, Paint::fill(Color::new(1.0, 1.0, 1.0, 0.16)));
    builder.restore();
    builder.restore();

    builder.draw_rrect(
        glass_rect,
        RoundingRadii::from_single_radius(6.0),
        Paint::stroke(Color::new(1.0, 1.0, 1.0, 0.70), 2.0),
    );

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_18_backdrop_filter_frosted_glass",
        width,
        height,
        bg,
        &dl,
    );

    // Outside frosted glass (y=24) across stripe boundary x=120: sharp step between cyan (x=114) and crimson (x=126).
    let out_left = get_pixel(&rgba, width, 114, 24);
    let out_right = get_pixel(&rgba, width, 126, 24);
    let out_step = max_channel_delta(out_left, out_right);
    assert!(
        out_step > 170,
        "expected sharp high-contrast stripe edge outside glass (y=24), got {out_left:?} vs {out_right:?} (delta={out_step})"
    );

    // Inside frosted glass (y=120) across the same stripe boundary x=120: backdrop blur blends neighbors.
    let in_left = get_pixel(&rgba, width, 114, 120);
    let in_right = get_pixel(&rgba, width, 126, 120);
    let in_step = max_channel_delta(in_left, in_right);
    assert!(
        in_step < out_step / 2,
        "expected frosted backdrop blur to soften stripe step at x=120 (in_step={in_step} vs out_step={out_step})"
    );
    // Both sides near the boundary inside the glass contain mixed red and blue energy.
    assert!(
        in_left[0] > 70 && in_left[2] > 90 && in_right[0] > 90 && in_right[2] > 70,
        "expected blurred color mixing inside glass: left={in_left:?}, right={in_right:?}"
    );
}

/// C++ source: impeller/display_list/aiks_dl_blur_unittests.cc:MaskBlurVariantTestSolidTranslucentWithFilters
#[test]
fn aiks_dl_19_dilate_and_erode_morphology_filters() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.06, 0.07, 0.10, 1.0);
    let bg_u8 = color_to_rgba8(bg);
    let shape_color = Color::new(0.15, 0.95, 0.72, 1.0);

    let mut builder = DisplayListBuilder::new();

    // Column 1 (x=36..76): Unfiltered 40x80 rect.
    builder.draw_rect(
        Rect::from_xywh(36.0, 80.0, 40.0, 80.0),
        Paint::fill(shape_color),
    );

    // Column 2 (x=140..180): Dilated by 10px in X and Y.
    builder.save_layer(
        Some(Rect::from_xywh(116.0, 56.0, 88.0, 128.0)),
        Some(
            Paint::fill(Color::WHITE).with_image_filter(ImageFilter::Dilate {
                radius_x: 10.0,
                radius_y: 10.0,
            }),
        ),
    );
    builder.draw_rect(
        Rect::from_xywh(140.0, 80.0, 40.0, 80.0),
        Paint::fill(shape_color),
    );
    builder.restore();

    // Column 3 (x=244..284): Eroded by 10px in X and Y.
    builder.save_layer(
        Some(Rect::from_xywh(224.0, 60.0, 80.0, 120.0)),
        Some(
            Paint::fill(Color::WHITE).with_image_filter(ImageFilter::Erode {
                radius_x: 10.0,
                radius_y: 10.0,
            }),
        ),
    );
    builder.draw_rect(
        Rect::from_xywh(244.0, 80.0, 40.0, 80.0),
        Paint::fill(shape_color),
    );
    builder.restore();

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_19_dilate_and_erode_morphology_filters",
        width,
        height,
        bg,
        &dl,
    );

    // 6px outside nominal left edge: unfiltered (30, 120) is bg, dilated (134, 120) is bright!
    assert_pixel_near(
        get_pixel(&rgba, width, 30, 120),
        bg_u8,
        3,
        "unfiltered outside",
    );
    let dilated_out = get_pixel(&rgba, width, 134, 120);
    assert!(
        dilated_out[1] > 210,
        "expected dilated pixel 6px outside nominal rect at (134, 120) to be bright, got {dilated_out:?}"
    );

    // 6px inside nominal left edge: unfiltered (42, 120) is bright, eroded (250, 120) is eroded to bg!
    let unfiltered_in = get_pixel(&rgba, width, 42, 120);
    assert!(
        unfiltered_in[1] > 210,
        "expected unfiltered interior at (42, 120) to be bright, got {unfiltered_in:?}"
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 250, 120),
        bg_u8,
        4,
        "eroded 6px inside nominal left edge",
    );

    // Eroded core (264, 120) remains bright.
    let eroded_core = get_pixel(&rgba, width, 264, 120);
    assert!(
        eroded_core[1] > 210,
        "expected eroded center (264, 120) to remain bright, got {eroded_core:?}"
    );

    let count_unfiltered = count_non_bg_in_rect(&rgba, width, [16, 56, 96, 184], bg_u8, 40);
    let count_dilated = count_non_bg_in_rect(&rgba, width, [116, 56, 204, 184], bg_u8, 40);
    let count_eroded = count_non_bg_in_rect(&rgba, width, [224, 56, 304, 184], bg_u8, 40);
    assert!(
        count_dilated > count_unfiltered && count_unfiltered > count_eroded && count_eroded > 400,
        "expected morphology pixel areas dilated ({count_dilated}) > unfiltered ({count_unfiltered}) > eroded ({count_eroded})"
    );
}

/// C++ source: impeller/display_list/aiks_dl_blur_unittests.cc:ComposePaintBlurOuter
#[test]
fn aiks_dl_20_compose_and_matrix_image_filters() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.07, 0.08, 0.11, 1.0);
    let bg_u8 = color_to_rgba8(bg);

    let mut builder = DisplayListBuilder::new();

    // Left: ImageFilter::Compose(ColorFilter::Matrix (swap R <-> B), Blur(6.0)) on a bright red-orange RRect.
    let swap_rb_matrix: [f32; 20] = [
        0.0, 0.0, 1.0, 0.0, 0.0, // R' = B
        0.0, 1.0, 0.0, 0.0, 0.0, // G' = G
        1.0, 0.0, 0.0, 0.0, 0.0, // B' = R
        0.0, 0.0, 0.0, 1.0, 0.0, // A' = A
    ];
    let composed_cf_blur = ImageFilter::Compose(
        Box::new(ImageFilter::ColorFilter(ColorFilter::Matrix(
            swap_rb_matrix,
        ))),
        Box::new(ImageFilter::Blur {
            sigma_x: 6.0,
            sigma_y: 6.0,
            tile_mode: TileMode::Decal,
        }),
    );
    builder.save_layer(
        Some(Rect::from_xywh(20.0, 40.0, 132.0, 160.0)),
        Some(Paint::fill(Color::WHITE).with_image_filter(composed_cf_blur)),
    );
    builder.draw_rrect(
        Rect::from_xywh(40.0, 60.0, 92.0, 120.0),
        RoundingRadii::from_single_radius(16.0),
        Paint::fill(Color::new(0.95, 0.25, 0.08, 1.0)),
    );
    builder.restore();

    // Right: ImageFilter::Compose(Blur(4.0), Matrix(translate(20, 16))) on a lime RRect,
    // plus a thin white reference outline at the unshifted position.
    let unshifted_rect = Rect::from_xywh(180.0, 56.0, 84.0, 96.0);
    let composed_blur_matrix = ImageFilter::Compose(
        Box::new(ImageFilter::Blur {
            sigma_x: 4.0,
            sigma_y: 4.0,
            tile_mode: TileMode::Decal,
        }),
        Box::new(ImageFilter::Matrix(Matrix::make_translation(
            20.0, 16.0, 0.0,
        ))),
    );
    builder.save_layer(
        Some(Rect::from_xywh(168.0, 44.0, 136.0, 152.0)),
        Some(Paint::fill(Color::WHITE).with_image_filter(composed_blur_matrix)),
    );
    builder.draw_rrect(
        unshifted_rect,
        RoundingRadii::from_single_radius(14.0),
        Paint::fill(Color::new(0.45, 0.96, 0.22, 1.0)),
    );
    builder.restore();
    builder.draw_rrect(
        unshifted_rect,
        RoundingRadii::from_single_radius(14.0),
        Paint::stroke(Color::new(1.0, 1.0, 1.0, 0.45), 1.5),
    );

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_20_compose_and_matrix_image_filters",
        width,
        height,
        bg,
        &dl,
    );

    // Left RRect was drawn with high R (0.95) and low B (0.08); after swap_rb + blur, center (86, 120) has high B and low R!
    let left_center = get_pixel(&rgba, width, 86, 120);
    assert!(
        left_center[2] > 200 && left_center[0] < 50,
        "expected Compose(ColorFilter::Matrix swap_rb, Blur) to produce blue-dominant interior at (86, 120), got {left_center:?}"
    );
    // And left edge (40, 120) is smoothly blurred.
    let left_edge = get_pixel(&rgba, width, 40, 120);
    assert!(
        left_edge[2] > bg_u8[2] + 40 && left_edge[2] < left_center[2] - 30,
        "expected blurred blue edge at (40, 120), got {left_edge:?}"
    );

    // Right RRect shifted by +(20, 16) means (188, 64) inside unshifted top-left is background,
    // while shifted interior (242, 120) and shifted bottom-right (276, 160) are bright lime!
    let unshifted_tl = get_pixel(&rgba, width, 188, 64);
    let shifted_br = get_pixel(&rgba, width, 276, 160);
    assert!(
        unshifted_tl[1] < 60,
        "expected unshifted top-left interior (188, 64) to be vacated by Matrix translation, got {unshifted_tl:?}"
    );
    assert!(
        shifted_br[1] > 180,
        "expected shifted bottom-right (276, 160) to contain translated+blurred lime RRect, got {shifted_br:?}"
    );
}

// =============================================================================
// Group E: 5 BlendMode scenes (aiks_dl_blend_unittests.cc)
// =============================================================================

/// C++ source: impeller/display_list/aiks_dl_blend_unittests.cc:ColorFilterBlend
#[test]
fn aiks_dl_21_porter_duff_compositing_modes() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.08, 0.09, 0.12, 1.0);
    let dst_color = Color::new(1.0, 0.78, 0.12, 0.85);
    let src_color = Color::new(0.12, 0.72, 1.0, 0.85);

    let modes = [
        BlendMode::SrcOver,
        BlendMode::SrcIn,
        BlendMode::SrcOut,
        BlendMode::DstIn,
        BlendMode::DstOut,
        BlendMode::Xor,
        BlendMode::Plus,
    ];

    let mut builder = DisplayListBuilder::new();
    for (i, &mode) in modes.iter().enumerate() {
        let col = (i % 4) as Scalar;
        let row = (i / 4) as Scalar;
        let x0 = 14.0 + col * 75.0;
        let y0 = 18.0 + row * 106.0;
        let cell = Rect::from_xywh(x0, y0, 66.0, 92.0);

        // Isolate each Porter-Duff cell in a true offscreen transparent SaveLayer
        // (alpha < 1.0 - 1e-4 prevents identity-layer elision so backdrop is transparent).
        builder.save_layer(
            Some(cell),
            Some(Paint::fill(Color::new(1.0, 1.0, 1.0, 0.995))),
        );
        builder.draw_rect(
            Rect::from_xywh(x0 + 4.0, y0 + 6.0, 40.0, 50.0),
            Paint::fill(dst_color),
        );
        builder.draw_rect(
            Rect::from_xywh(x0 + 22.0, y0 + 30.0, 40.0, 50.0),
            Paint::fill(src_color).with_blend_mode(mode),
        );
        builder.restore();
    }

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_21_porter_duff_compositing_modes",
        width,
        height,
        bg,
        &dl,
    );

    for (i, &mode) in modes.iter().enumerate() {
        let col = (i % 4) as u32;
        let row = (i / 4) as u32;
        let x0 = 14 + col * 75;
        let y0 = 18 + row * 106;

        // Overlap center (x0 + 32, y0 + 42) has both dst_color and src_color.
        let overlap_px = get_pixel(&rgba, width, x0 + 32, y0 + 42);
        let expected_overlap =
            color_to_rgba8(bg.blend(dst_color.blend(src_color, mode), BlendMode::SrcOver));
        assert_pixel_near(
            overlap_px,
            expected_overlap,
            5,
            &format!("Porter-Duff {mode:?} overlap"),
        );

        // Src-only region (x0 + 52, y0 + 68) has transparent dst and src_color.
        let src_only_px = get_pixel(&rgba, width, x0 + 52, y0 + 68);
        let expected_src_only = color_to_rgba8(bg.blend(
            Color::TRANSPARENT.blend(src_color, mode),
            BlendMode::SrcOver,
        ));
        assert_pixel_near(
            src_only_px,
            expected_src_only,
            5,
            &format!("Porter-Duff {mode:?} src-only"),
        );
    }
}

/// C++ source: impeller/display_list/aiks_dl_blend_unittests.cc:ColorFilterAdvancedBlend
#[test]
fn aiks_dl_22_separable_advanced_blends_multiply_screen_overlay() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.08, 0.09, 0.13, 1.0);
    let dst_color = Color::new(0.76, 0.44, 0.24, 1.0);
    let src_color = Color::new(0.24, 0.68, 0.86, 1.0);

    let modes = [
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
    ];

    let mut builder = DisplayListBuilder::new();
    for (i, &mode) in modes.iter().enumerate() {
        let x0 = 12.0 + (i as Scalar) * 60.0;
        builder.draw_rect(
            Rect::from_xywh(x0, 42.0, 46.0, 104.0),
            Paint::fill(dst_color),
        );
        builder.draw_rect(
            Rect::from_xywh(x0 + 6.0, 92.0, 40.0, 104.0),
            Paint::fill(src_color).with_blend_mode(mode),
        );
    }

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_22_separable_advanced_blends_multiply_screen_overlay",
        width,
        height,
        bg,
        &dl,
    );

    for (i, &mode) in modes.iter().enumerate() {
        let x_overlap = 12 + (i as u32) * 60 + 24;
        let y_overlap = 118;
        let actual = get_pixel(&rgba, width, x_overlap, y_overlap);
        let expected = color_to_rgba8(dst_color.blend(src_color, mode));
        assert_pixel_near(
            actual,
            expected,
            4,
            &format!("separable blend {mode:?} overlap"),
        );
    }

    // Verify Multiply is strictly darker than Screen in all RGB channels.
    let mul_px = get_pixel(&rgba, width, 36, 118);
    let scr_px = get_pixel(&rgba, width, 96, 118);
    assert!(
        mul_px[0] < scr_px[0] && mul_px[1] < scr_px[1] && mul_px[2] < scr_px[2],
        "expected Multiply ({mul_px:?}) to be darker than Screen ({scr_px:?})"
    );
}

/// C++ source: impeller/display_list/aiks_dl_blend_unittests.cc:ForegroundBlendSubpassCollapseOptimization
#[test]
fn aiks_dl_23_dodge_burn_and_light_blends() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.07, 0.08, 0.12, 1.0);
    let dst_color = Color::new(0.62, 0.48, 0.36, 1.0);
    let src_color = Color::new(0.45, 0.72, 0.32, 1.0);

    let modes = [
        BlendMode::ColorDodge,
        BlendMode::ColorBurn,
        BlendMode::HardLight,
        BlendMode::SoftLight,
    ];

    let mut builder = DisplayListBuilder::new();
    for (i, &mode) in modes.iter().enumerate() {
        let x0 = 16.0 + (i as Scalar) * 74.0;
        builder.draw_rect(
            Rect::from_xywh(x0, 40.0, 56.0, 104.0),
            Paint::fill(dst_color),
        );
        builder.draw_rect(
            Rect::from_xywh(x0 + 8.0, 92.0, 48.0, 104.0),
            Paint::fill(src_color).with_blend_mode(mode),
        );
    }

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_23_dodge_burn_and_light_blends",
        width,
        height,
        bg,
        &dl,
    );

    for (i, &mode) in modes.iter().enumerate() {
        let x_overlap = 16 + (i as u32) * 74 + 30;
        let y_overlap = 118;
        let actual = get_pixel(&rgba, width, x_overlap, y_overlap);
        let expected = color_to_rgba8(dst_color.blend(src_color, mode));
        assert_pixel_near(
            actual,
            expected,
            4,
            &format!("dodge/burn/light blend {mode:?} overlap"),
        );
    }

    let dodge_px = get_pixel(&rgba, width, 46, 118);
    let burn_px = get_pixel(&rgba, width, 120, 118);
    assert!(
        dodge_px[0] > burn_px[0] && dodge_px[1] > burn_px[1],
        "expected ColorDodge ({dodge_px:?}) to be brighter than ColorBurn ({burn_px:?})"
    );
}

/// C++ source: impeller/display_list/aiks_dl_blend_unittests.cc:CanRenderAdvancedBlendColorFilterWithSaveLayer
#[test]
fn aiks_dl_24_difference_and_exclusion_blends() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.07, 0.08, 0.11, 1.0);
    let dst_color = Color::new(0.84, 0.62, 0.22, 1.0);
    let src_color = Color::new(0.36, 0.78, 0.68, 1.0);

    let mut builder = DisplayListBuilder::new();

    // Left: BlendMode::Difference.
    builder.draw_rect(
        Rect::from_xywh(24.0, 42.0, 96.0, 108.0),
        Paint::fill(dst_color),
    );
    builder.draw_rect(
        Rect::from_xywh(52.0, 88.0, 96.0, 108.0),
        Paint::fill(src_color).with_blend_mode(BlendMode::Difference),
    );

    // Right: BlendMode::Exclusion.
    builder.draw_rect(
        Rect::from_xywh(176.0, 42.0, 96.0, 108.0),
        Paint::fill(dst_color),
    );
    builder.draw_rect(
        Rect::from_xywh(204.0, 88.0, 96.0, 108.0),
        Paint::fill(src_color).with_blend_mode(BlendMode::Exclusion),
    );

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_24_difference_and_exclusion_blends",
        width,
        height,
        bg,
        &dl,
    );

    let diff_actual = get_pixel(&rgba, width, 86, 118);
    let diff_expected = color_to_rgba8(dst_color.blend(src_color, BlendMode::Difference));
    assert_pixel_near(diff_actual, diff_expected, 4, "Difference overlap");

    let excl_actual = get_pixel(&rgba, width, 238, 118);
    let excl_expected = color_to_rgba8(dst_color.blend(src_color, BlendMode::Exclusion));
    assert_pixel_near(excl_actual, excl_expected, 4, "Exclusion overlap");

    assert!(
        max_channel_delta(diff_actual, excl_actual) > 25,
        "expected Difference ({diff_actual:?}) and Exclusion ({excl_actual:?}) to produce distinct midtone contrast"
    );
}

/// C++ source: impeller/display_list/aiks_dl_blend_unittests.cc:CanDrawPaintWithAdvancedBlend
#[test]
fn aiks_dl_25_hslc_nonseparable_blends() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.07, 0.08, 0.12, 1.0);
    let dst_color = Color::new(0.82, 0.28, 0.24, 1.0);
    let src_color = Color::new(0.16, 0.86, 0.44, 1.0);

    let modes = [
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
    ];

    let mut builder = DisplayListBuilder::new();
    for (i, &mode) in modes.iter().enumerate() {
        let x0 = 16.0 + (i as Scalar) * 74.0;
        builder.draw_rect(
            Rect::from_xywh(x0, 40.0, 56.0, 104.0),
            Paint::fill(dst_color),
        );
        builder.draw_rect(
            Rect::from_xywh(x0 + 8.0, 92.0, 48.0, 104.0),
            Paint::fill(src_color).with_blend_mode(mode),
        );
    }

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_25_hslc_nonseparable_blends",
        width,
        height,
        bg,
        &dl,
    );

    for (i, &mode) in modes.iter().enumerate() {
        let x_overlap = 16 + (i as u32) * 74 + 30;
        let y_overlap = 118;
        let actual = get_pixel(&rgba, width, x_overlap, y_overlap);
        let expected = color_to_rgba8(dst_color.blend(src_color, mode));
        assert_pixel_near(
            actual,
            expected,
            5,
            &format!("HSLC non-separable blend {mode:?} overlap"),
        );
    }
}

// =============================================================================
// Group F: 5 Path, Stroke, Vertices & RSuperellipse scenes
// =============================================================================

fn build_star_and_bezier_ring(center: Point, radius: Scalar, fill_type: FillType) -> Path {
    let mut pb = PathBuilder::new();
    // 5-pointed self-intersecting star: connect every 2nd vertex (0 -> 2 -> 4 -> 1 -> 3 -> close)
    // so the central pentagon has winding number 2.
    let order = [0usize, 2, 4, 1, 3];
    for (idx, &k) in order.iter().enumerate() {
        let angle = -std::f32::consts::FRAC_PI_2 + (k as Scalar) * 2.0 * std::f32::consts::PI / 5.0;
        let pt = Point::new(
            center.x + radius * angle.cos(),
            center.y + radius * angle.sin(),
        );
        if idx == 0 {
            pb.move_to(pt);
        } else {
            pb.line_to(pt);
        }
    }
    pb.close();

    // Cubic & quadratic Bezier accent arch beneath each star.
    let y_arch = center.y + radius + 16.0;
    pb.move_to(Point::new(center.x - 48.0, y_arch));
    pb.cubic_curve_to(
        Point::new(center.x - 24.0, y_arch - 22.0),
        Point::new(center.x + 24.0, y_arch - 22.0),
        Point::new(center.x + 48.0, y_arch),
    );
    pb.quadratic_curve_to(
        Point::new(center.x, y_arch + 18.0),
        Point::new(center.x - 48.0, y_arch),
    );
    pb.close();

    pb.build(fill_type)
}

/// C++ source: impeller/display_list/aiks_dl_path_unittests.cc:CanRenderDifferencePaths
#[test]
fn aiks_dl_26_cubic_and_quad_paths_evenodd_vs_nonzero() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.06, 0.07, 0.10, 1.0);
    let bg_u8 = color_to_rgba8(bg);

    let mut builder = DisplayListBuilder::new();
    let cyan = Color::new(0.14, 0.86, 0.98, 1.0);
    let orange = Color::new(1.0, 0.54, 0.14, 1.0);

    let star_odd = build_star_and_bezier_ring(Point::new(88.0, 102.0), 58.0, FillType::Odd);
    builder.draw_path(star_odd, Paint::fill(cyan));

    let star_nonzero =
        build_star_and_bezier_ring(Point::new(232.0, 102.0), 58.0, FillType::NonZero);
    builder.draw_path(star_nonzero, Paint::fill(orange));

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_26_cubic_and_quad_paths_evenodd_vs_nonzero",
        width,
        height,
        bg,
        &dl,
    );

    // Center of EvenOdd star (88, 102) has winding=2 -> hollow hole matching background!
    assert_pixel_near(
        get_pixel(&rgba, width, 88, 102),
        bg_u8,
        3,
        "EvenOdd star center hole",
    );
    // Top spike of EvenOdd star (88, 56) has winding=1 -> filled cyan!
    assert_pixel_near(
        get_pixel(&rgba, width, 88, 56),
        color_to_rgba8(cyan),
        4,
        "EvenOdd star top spike",
    );

    // Center of NonZero star (232, 102) has winding=2 != 0 -> solidly filled orange!
    assert_pixel_near(
        get_pixel(&rgba, width, 232, 102),
        color_to_rgba8(orange),
        4,
        "NonZero star solid center",
    );
    // Top spike of NonZero star (232, 56) is also filled orange.
    assert_pixel_near(
        get_pixel(&rgba, width, 232, 56),
        color_to_rgba8(orange),
        4,
        "NonZero star top spike",
    );

    // Cubic/quad Bezier arch below left star at (88, 174) is filled cyan.
    let arch_px = get_pixel(&rgba, width, 88, 174);
    assert!(
        arch_px[2] > 200,
        "expected cubic/quad Bezier arch at (88, 174) to be filled cyan, got {arch_px:?}"
    );
}

/// C++ source: impeller/display_list/aiks_dl_path_unittests.cc:StrokeCapsAndJoins
#[test]
fn aiks_dl_27_stroke_joins_miter_round_bevel() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.06, 0.07, 0.10, 1.0);
    let bg_u8 = color_to_rgba8(bg);

    let joins = [
        (64.0, Join::Miter, Color::new(1.0, 0.42, 0.32, 1.0)),
        (160.0, Join::Round, Color::new(0.22, 0.92, 0.58, 1.0)),
        (256.0, Join::Bevel, Color::new(0.32, 0.68, 1.0, 1.0)),
    ];

    let mut builder = DisplayListBuilder::new();
    for &(cx, join, color) in &joins {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(cx - 32.0, 172.0));
        pb.line_to(Point::new(cx, 92.0));
        pb.line_to(Point::new(cx + 32.0, 172.0));
        let path = pb.build(FillType::NonZero);

        let paint = Paint::stroke(color, 24.0)
            .with_stroke_join(join)
            .with_stroke_cap(Cap::Butt)
            .with_stroke_miter(6.0);
        builder.draw_path(path, paint);
    }

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_27_stroke_joins_miter_round_bevel",
        width,
        height,
        bg,
        &dl,
    );

    // Apex is at y=92, stroke half-width = 12:
    // - Miter tip reaches up to y ~ 60 -> at y=70 (22px above apex), ONLY Miter has ink!
    assert!(
        max_channel_delta(get_pixel(&rgba, width, 64, 70), bg_u8) > 150,
        "expected Miter tip to extend to (64, 70)"
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 160, 70),
        bg_u8,
        3,
        "Round join empty at y=70",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 256, 70),
        bg_u8,
        3,
        "Bevel join empty at y=70",
    );

    // - Round join arc reaches up to y = 92 - 12 = 80 -> at y=83 (9px above apex), Miter & Round have ink, Bevel is empty!
    assert!(
        max_channel_delta(get_pixel(&rgba, width, 64, 83), bg_u8) > 150,
        "expected Miter join filled at (64, 83)"
    );
    assert!(
        max_channel_delta(get_pixel(&rgba, width, 160, 83), bg_u8) > 150,
        "expected Round join filled at (160, 83)"
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 256, 83),
        bg_u8,
        3,
        "Bevel join flat-cut empty at y=83",
    );

    // - At y=90 (2px above apex), all three joins (Miter, Round, Bevel) have ink!
    assert!(
        max_channel_delta(get_pixel(&rgba, width, 256, 90), bg_u8) > 150,
        "expected Bevel join filled just above apex at (256, 90)"
    );
}

/// C++ source: impeller/display_list/aiks_dl_basic_unittests.cc:CanRenderFilledRoundSuperellipses
#[test]
fn aiks_dl_28_round_superellipse_cupertino_squircles() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.06, 0.07, 0.11, 1.0);
    let bg_u8 = color_to_rgba8(bg);

    let mut builder = DisplayListBuilder::new();

    // 1. Left: Filled uniform Cupertino squircle.
    let blue = Color::new(0.04, 0.52, 1.0, 1.0);
    let rse_uniform =
        RoundSuperellipse::make_rect_radius(Rect::from_xywh(20.0, 48.0, 92.0, 92.0), 28.0);
    builder.draw_round_superellipse(rse_uniform, Paint::fill(blue));

    // 2. Center: Filled asymmetric-radii Cupertino squircle.
    let green = Color::new(0.19, 0.82, 0.35, 1.0);
    let rse_asym = RoundSuperellipse::make_rect_radii(
        Rect::from_xywh(126.0, 44.0, 84.0, 148.0),
        RoundingRadii {
            top_left: Size::new(36.0, 36.0),
            top_right: Size::new(6.0, 6.0),
            bottom_left: Size::new(6.0, 6.0),
            bottom_right: Size::new(36.0, 36.0),
        },
    );
    builder.draw_round_superellipse(rse_asym, Paint::fill(green));

    // 3. Right: Stroked Cupertino squircle with hollow interior.
    let red = Color::new(1.0, 0.27, 0.23, 1.0);
    let rse_stroked =
        RoundSuperellipse::make_rect_radius(Rect::from_xywh(228.0, 56.0, 72.0, 128.0), 22.0);
    builder.draw_round_superellipse(rse_stroked, Paint::stroke(red, 8.0));

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_28_round_superellipse_cupertino_squircles",
        width,
        height,
        bg,
        &dl,
    );

    // Uniform squircle center is Cupertino blue; bounding-box corner (22, 50) is rounded away.
    assert_pixel_near(
        get_pixel(&rgba, width, 66, 94),
        color_to_rgba8(blue),
        3,
        "uniform squircle center",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 22, 50),
        bg_u8,
        3,
        "uniform squircle rounded corner exterior",
    );

    // Asymmetric squircle: large-radius top-left (130, 48) is background, small-radius top-right (205, 49) is filled green!
    assert_pixel_near(
        get_pixel(&rgba, width, 130, 48),
        bg_u8,
        3,
        "asymmetric squircle large-radius top-left exterior",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 205, 49),
        color_to_rgba8(green),
        4,
        "asymmetric squircle small-radius top-right interior",
    );

    // Stroked squircle: top border (264, 56) is red, interior center (264, 120) is hollow background.
    assert_pixel_near(
        get_pixel(&rgba, width, 264, 56),
        color_to_rgba8(red),
        4,
        "stroked squircle top edge",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 264, 120),
        bg_u8,
        3,
        "stroked squircle hollow center",
    );
}

fn make_quadrant_texture(id: u64) -> ImageTexture {
    let w = 64u32;
    let h = 64u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let color = match (x >= 32, y >= 32) {
                (false, false) => [250u8, 200, 30, 255], // Top-Left: Gold
                (true, false) => [235u8, 50, 190, 255],  // Top-Right: Magenta
                (false, true) => [40u8, 220, 110, 255],  // Bottom-Left: Emerald
                (true, true) => [50u8, 140, 250, 255],   // Bottom-Right: Azure
            };
            let idx = ((y * w + x) as usize) * 4;
            data[idx..idx + 4].copy_from_slice(&color);
        }
    }
    ImageTexture::new(id, w, h, data)
}

/// C++ source: impeller/display_list/aiks_dl_vertices_unittests.cc:DrawVerticesImageSourceWithTextureCoordinatesAndColorBlending
#[test]
fn aiks_dl_29_draw_vertices_colored_and_textured_mesh() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.07, 0.08, 0.11, 1.0);

    let mut builder = DisplayListBuilder::new();

    // Left mesh: Indexed per-vertex colored diamond mesh.
    let coral = Color::new(0.95, 0.26, 0.20, 1.0);
    let teal = Color::new(0.14, 0.80, 0.96, 1.0);
    let positions_left = vec![
        Point::new(84.0, 44.0),   // 0: top
        Point::new(146.0, 120.0), // 1: right
        Point::new(22.0, 120.0),  // 2: left
        Point::new(22.0, 120.0),  // 3: left (bottom tri)
        Point::new(146.0, 120.0), // 4: right (bottom tri)
        Point::new(84.0, 196.0),  // 5: bottom
    ];
    let colors_left = vec![coral, coral, coral, teal, teal, teal];
    let indices_left = vec![0u16, 1, 2, 3, 4, 5];
    builder.draw_vertices(
        positions_left,
        None,
        Some(colors_left),
        indices_left,
        BlendMode::Dst,
        Paint::fill(Color::WHITE),
    );

    // Right mesh: Indexed textured quad with explicit UV coordinates.
    let tex = make_quadrant_texture(2901);
    let positions_right = vec![
        Point::new(176.0, 56.0),
        Point::new(300.0, 56.0),
        Point::new(300.0, 180.0),
        Point::new(176.0, 180.0),
    ];
    let tex_coords_right = vec![
        Point::new(0.0, 0.0),
        Point::new(64.0, 0.0),
        Point::new(64.0, 64.0),
        Point::new(0.0, 64.0),
    ];
    let indices_right = vec![0u16, 1, 2, 0, 2, 3];
    let tex_paint = Paint::fill(Color::WHITE).with_color_source(ColorSource::Image {
        texture: tex,
        tile_mode_x: TileMode::Clamp,
        tile_mode_y: TileMode::Clamp,
        matrix: Matrix::IDENTITY,
        linear_filter: false,
    });
    builder.draw_vertices(
        positions_right,
        Some(tex_coords_right),
        None,
        indices_right,
        BlendMode::Dst,
        tex_paint,
    );

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_29_draw_vertices_colored_and_textured_mesh",
        width,
        height,
        bg,
        &dl,
    );

    // Verify colored mesh top (coral) and bottom (teal) triangles.
    assert_pixel_near(
        get_pixel(&rgba, width, 84, 92),
        color_to_rgba8(coral),
        4,
        "colored vertices top coral triangle",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 84, 148),
        color_to_rgba8(teal),
        4,
        "colored vertices bottom teal triangle",
    );

    // Verify textured mesh quadrants.
    assert_pixel_near(
        get_pixel(&rgba, width, 206, 86),
        [250, 200, 30, 255],
        3,
        "textured vertices top-left Gold",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 268, 86),
        [235, 50, 190, 255],
        3,
        "textured vertices top-right Magenta",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 206, 148),
        [40, 220, 110, 255],
        3,
        "textured vertices bottom-left Emerald",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 268, 148),
        [50, 140, 250, 255],
        3,
        "textured vertices bottom-right Azure",
    );
}

fn make_sprite_atlas_texture(id: u64) -> ImageTexture {
    let w = 64u32;
    let h = 64u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let color = match (x >= 32, y >= 32) {
                (false, false) => [240u8, 55, 70, 255], // Sprite 0: Ruby
                (true, false) => [65u8, 230, 95, 255],  // Sprite 1: Lime
                (false, true) => [55u8, 145, 250, 255], // Sprite 2: Electric Blue
                (true, true) => [250u8, 180, 40, 255],  // Sprite 3: Amber
            };
            let idx = ((y * w + x) as usize) * 4;
            data[idx..idx + 4].copy_from_slice(&color);
        }
    }
    ImageTexture::new(id, w, h, data)
}

/// C++ source: impeller/display_list/aiks_dl_atlas_unittests.cc:DrawAtlasNoColor
#[test]
fn aiks_dl_30_draw_atlas_and_text_typography() {
    let width = 320;
    let height = 240;
    let bg = Color::new(0.06, 0.07, 0.11, 1.0);

    let mut builder = DisplayListBuilder::new();
    let atlas = make_sprite_atlas_texture(3001);

    // Top half: 4 sprites drawn via draw_atlas (3 scaled axis-aligned + 1 rotated).
    let rst_transforms = vec![
        [1.5, 0.0, 20.0, 28.0],  // Sprite 0: 48x48 at (20..68, 28..76)
        [1.5, 0.0, 88.0, 28.0],  // Sprite 1: 48x48 at (88..136, 28..76)
        [1.5, 0.0, 156.0, 28.0], // Sprite 2: 48x48 at (156..204, 28..76)
        [1.0, 0.5, 248.0, 24.0], // Sprite 3: rotated/scaled RSXform
    ];
    let tex_rects = vec![
        Rect::from_xywh(0.0, 0.0, 32.0, 32.0),
        Rect::from_xywh(32.0, 0.0, 32.0, 32.0),
        Rect::from_xywh(0.0, 32.0, 32.0, 32.0),
        Rect::from_xywh(32.0, 32.0, 32.0, 32.0),
    ];
    builder.draw_atlas(
        atlas,
        rst_transforms,
        tex_rects,
        None,
        BlendMode::SrcOver,
        None,
        None,
        false,
    );

    // Bottom half: Typography banner with crisp RobotoBold and RobotoRegular glyph runs.
    let banner_color = Color::new(0.11, 0.13, 0.19, 1.0);
    let banner_u8 = color_to_rgba8(banner_color);
    builder.draw_rrect(
        Rect::from_xywh(16.0, 122.0, 288.0, 102.0),
        RoundingRadii::from_single_radius(10.0),
        Paint::fill(banner_color),
    );
    builder.draw_text(
        "IMPELLER ATLAS",
        TextStyle {
            font: BuiltinFont::RobotoBold,
            font_size: 26.0,
            line_height: 30.0,
            color: Color::new(0.96, 0.98, 1.0, 1.0),
            max_width: None,
        },
        Point::new(28.0, 136.0),
    );
    builder.draw_text(
        "WGPU GLYPH BATCH",
        TextStyle {
            font: BuiltinFont::RobotoRegular,
            font_size: 18.0,
            line_height: 22.0,
            color: Color::new(0.35, 0.88, 1.0, 1.0),
            max_width: None,
        },
        Point::new(28.0, 178.0),
    );

    let dl = builder.build();
    let rgba = render_dl_and_save_golden(
        "aiks_dl_30_draw_atlas_and_text_typography",
        width,
        height,
        bg,
        &dl,
    );

    // Verify draw_atlas sprites 0..3.
    assert_pixel_near(
        get_pixel(&rgba, width, 44, 52),
        [240, 55, 70, 255],
        3,
        "atlas sprite 0 Ruby",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 112, 52),
        [65, 230, 95, 255],
        3,
        "atlas sprite 1 Lime",
    );
    assert_pixel_near(
        get_pixel(&rgba, width, 180, 52),
        [55, 145, 250, 255],
        3,
        "atlas sprite 2 Electric Blue",
    );
    // Sprite 3 rotated center: (tx=248, ty=24) + (scos=1.0, ssin=0.5) * (16, 16) => x = 248 + 16 - 8 = 256, y = 24 + 8 + 16 = 48.
    assert_pixel_near(
        get_pixel(&rgba, width, 256, 48),
        [250, 180, 40, 255],
        4,
        "atlas sprite 3 rotated Amber",
    );

    // Verify crisp glyph rendering in both text rows.
    let title_ink = count_non_bg_in_rect(&rgba, width, [26, 134, 290, 172], banner_u8, 80);
    let subtitle_ink = count_non_bg_in_rect(&rgba, width, [26, 176, 290, 212], banner_u8, 60);
    assert!(
        title_ink > 350,
        "expected >= 350 high-contrast glyph pixels in 'IMPELLER ATLAS' row, got {title_ink}"
    );
    assert!(
        subtitle_ink > 200,
        "expected >= 200 high-contrast glyph pixels in 'WGPU GLYPH BATCH' row, got {subtitle_ink}"
    );
}
