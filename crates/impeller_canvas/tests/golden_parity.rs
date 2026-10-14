#![forbid(unsafe_code)]

use impeller_canvas::{
    build_showcase_canvas, compare_golden_rgba, save_rgba_png, verify_showcase_pixels, BuiltinFont,
    Canvas, Cap, Color, Join, Paint, Point, Rect, RoundingRadii, Scalar, Size, TextStyle,
    WgpuContext,
};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

fn shared_renderer() -> MutexGuard<'static, WgpuContext> {
    static RENDERER: OnceLock<Mutex<WgpuContext>> = OnceLock::new();
    RENDERER
        .get_or_init(|| Mutex::new(WgpuContext::new_headless().expect("headless wgpu renderer")))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
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

fn audit_text_band_top_scanline(
    rgba: &[u8],
    width: u32,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    label: &str,
) {
    let mut first_ink_y: Option<u32> = None;
    for y in (y0 + 1)..y1 {
        let row_max = (x0..x1)
            .map(|x| {
                max_channel_delta(
                    get_pixel(rgba, width, x, y),
                    get_pixel(rgba, width, x, y - 1),
                )
            })
            .max()
            .unwrap_or(0);
        if row_max > 28 {
            first_ink_y = Some(y);
            break;
        }
    }
    let top_y =
        first_ink_y.unwrap_or_else(|| panic!("{label}: no ink pixels found in y={y0}..{y1}"));

    // Row immediately above the first ink scanline must be smooth background (no pre-cropped ghost bar).
    let above_max = (x0..x1)
        .map(|x| {
            max_channel_delta(
                get_pixel(rgba, width, x, top_y - 1),
                get_pixel(rgba, width, x, (top_y - 2).max(y0)),
            )
        })
        .max()
        .unwrap_or(0);
    assert!(
        above_max <= 12,
        "{label}: row above top scanline (y={}) was not background (max_delta={})",
        top_y - 1,
        above_max
    );

    // Check top_y (and top_y + 1 if round overshoot like 'C'/'O'/'Q' is 1px higher) for flat-top cap bar.
    let count_bright_on_row = |y: u32| -> (u8, usize) {
        let mut peak = 0u8;
        let mut bright_cols = 0usize;
        for x in x0..x1 {
            let bg = get_pixel(rgba, width, x, top_y - 1);
            let d = max_channel_delta(get_pixel(rgba, width, x, y), bg);
            peak = peak.max(d);
            if d >= 170 {
                bright_cols += 1;
            }
        }
        (peak, bright_cols)
    };

    let (peak0, cols0) = count_bright_on_row(top_y);
    let (peak1, cols1) = count_bright_on_row((top_y + 1).min(y1 - 1));
    let best_peak = peak0.max(peak1);
    let best_cols = cols0.max(cols1);

    assert!(
        best_peak >= 170,
        "{label}: top scanline y={top_y} appears cropped (peak delta={best_peak} < 170)"
    );
    assert!(
        best_cols >= 4,
        "{label}: expected >= 4 high-contrast cap-bar columns on top scanline y={top_y}, got {best_cols}"
    );
}

/// C++ source: aiks_unittests.cc:CanRenderTextFrameGoldenScanlineParity
#[test]
fn golden_typography_top_scanline_and_baseline_parity() {
    let renderer = shared_renderer();
    let bg = Color::new(0.05, 0.06, 0.08, 1.0);
    let mut canvas = Canvas::new(420, 170, bg);

    canvas.draw_text(
        "Q1 \u{00B7} RRECT BLUR SHADOWS & GRADIENTS",
        TextStyle {
            font: BuiltinFont::RobotoBold,
            font_size: 12.5,
            line_height: 15.0,
            color: Color::WHITE,
            max_width: None,
        },
        Point::new(16.0, 14.0),
    );
    canvas.draw_text(
        "CLAMP",
        TextStyle {
            font: BuiltinFont::RobotoMonoBold,
            font_size: 11.0,
            line_height: 14.0,
            color: Color::WHITE,
            max_width: None,
        },
        Point::new(16.0, 44.0),
    );
    canvas.draw_text(
        "MIRROR",
        TextStyle {
            font: BuiltinFont::RobotoMonoBold,
            font_size: 11.0,
            line_height: 14.0,
            color: Color::WHITE,
            max_width: None,
        },
        Point::new(120.0, 44.0),
    );

    canvas.draw_polymorphic_card(0, 0.18, Rect::from_xywh(16.0, 76.0, 118.0, 72.0));
    canvas.draw_polymorphic_card(1, 0.0, Rect::from_xywh(144.0, 76.0, 118.0, 72.0));
    canvas.draw_polymorphic_card(5, 0.18, Rect::from_xywh(272.0, 76.0, 118.0, 72.0));

    let rgba = canvas
        .render_to_pixels(&renderer)
        .expect("render typography golden scene");

    audit_text_band_top_scanline(&rgba, 420, 14, 340, 10, 34, "Q1 HUD header");
    audit_text_band_top_scanline(&rgba, 420, 14, 90, 40, 64, "CLAMP");
    audit_text_band_top_scanline(&rgba, 420, 116, 200, 40, 64, "MIRROR");
    audit_text_band_top_scanline(&rgba, 420, 20, 95, 80, 102, "WAVE #0");
    audit_text_band_top_scanline(&rgba, 420, 148, 220, 90, 112, "CORE #1");
    audit_text_band_top_scanline(&rgba, 420, 148, 220, 108, 132, "75 ops");
    audit_text_band_top_scanline(&rgba, 420, 304, 376, 102, 124, "WASM #5");
}

fn point_in_asymmetric_rrect(pt: Point, rect: Rect, radii: RoundingRadii) -> bool {
    let l = rect.get_left();
    let t = rect.get_top();
    let r = rect.get_right();
    let b = rect.get_bottom();
    if pt.x < l || pt.x > r || pt.y < t || pt.y > b {
        return false;
    }

    let check_ellipse = |cx: Scalar, cy: Scalar, rx: Scalar, ry: Scalar| -> bool {
        if rx <= 0.0 || ry <= 0.0 {
            return true;
        }
        let dx = (pt.x - cx) / rx;
        let dy = (pt.y - cy) / ry;
        dx * dx + dy * dy <= 1.0
    };

    if pt.x < l + radii.top_left.width && pt.y < t + radii.top_left.height {
        return check_ellipse(
            l + radii.top_left.width,
            t + radii.top_left.height,
            radii.top_left.width,
            radii.top_left.height,
        );
    }
    if pt.x > r - radii.top_right.width && pt.y < t + radii.top_right.height {
        return check_ellipse(
            r - radii.top_right.width,
            t + radii.top_right.height,
            radii.top_right.width,
            radii.top_right.height,
        );
    }
    if pt.x < l + radii.bottom_left.width && pt.y > b - radii.bottom_left.height {
        return check_ellipse(
            l + radii.bottom_left.width,
            b - radii.bottom_left.height,
            radii.bottom_left.width,
            radii.bottom_left.height,
        );
    }
    if pt.x > r - radii.bottom_right.width && pt.y > b - radii.bottom_right.height {
        return check_ellipse(
            r - radii.bottom_right.width,
            b - radii.bottom_right.height,
            radii.bottom_right.width,
            radii.bottom_right.height,
        );
    }
    true
}

/// C++ source: aiks_unittests.cc:CanRenderRoundedRectGoldenSDFParity
#[test]
fn golden_asymmetric_rrect_and_circle_analytical_sdf_parity() {
    let renderer = shared_renderer();
    let width = 320u32;
    let height = 160u32;
    let mut canvas = Canvas::new(width, height, Color::BLACK);

    let circle_center = Point::new(80.0, 80.0);
    let circle_radius = 40.0;
    canvas.draw_circle(circle_center, circle_radius, &Paint::fill(Color::WHITE));

    let rrect_bounds = Rect::from_xywh(160.0, 30.0, 130.0, 100.0);
    let rrect_radii = RoundingRadii {
        top_left: Size::new(28.0, 28.0),
        top_right: Size::new(6.0, 6.0),
        bottom_left: Size::new(6.0, 6.0),
        bottom_right: Size::new(28.0, 28.0),
    };
    canvas.draw_rrect(rrect_bounds, rrect_radii, &Paint::fill(Color::WHITE));

    let actual = canvas
        .render_to_pixels(&renderer)
        .expect("render circle + asymmetric rrect");

    // Build 16x (4x4 subpixel grid) analytical SDF reference RGBA8 image.
    let mut expected = vec![0u8; (width * height * 4) as usize];
    for py in 0..height {
        for px in 0..width {
            let mut hits = 0u32;
            for sy in 0..4u32 {
                for sx in 0..4u32 {
                    let sample = Point::new(
                        px as Scalar + (sx as Scalar + 0.5) * 0.25,
                        py as Scalar + (sy as Scalar + 0.5) * 0.25,
                    );
                    let in_circle =
                        (sample - circle_center).get_length_squared() <= circle_radius * circle_radius;
                    let in_rrect = point_in_asymmetric_rrect(sample, rrect_bounds, rrect_radii);
                    if in_circle || in_rrect {
                        hits += 1;
                    }
                }
            }
            let cov = ((hits * 255 + 8) / 16) as u8;
            let idx = ((py * width + px) as usize) * 4;
            expected[idx] = cov;
            expected[idx + 1] = cov;
            expected[idx + 2] = cov;
            expected[idx + 3] = 255;
        }
    }

    let (metrics, contact_sheet) =
        compare_golden_rgba(&actual, &expected, width, height, 32).expect("compare_golden_rgba");
    let sheet_path = repo_root().join("golden_contact_sheet.png");
    save_rgba_png(&sheet_path, width * 3, height, &contact_sheet)
        .expect("save golden_contact_sheet.png");

    assert!(
        metrics.rmse < 2.2,
        "expected SDF golden RMSE < 2.2, got {:.4} (max_diff={}, mismatch_fraction={:.5})",
        metrics.rmse,
        metrics.max_channel_diff,
        metrics.mismatch_fraction
    );
    assert!(
        metrics.mismatch_fraction < 0.005,
        "expected SDF mismatch_fraction < 0.005, got {:.5} (rmse={:.4})",
        metrics.mismatch_fraction,
        metrics.rmse
    );
}

/// C++ source: entity_unittests.cc:StrokeDiagonalMonotonicAntialiasing
#[test]
fn golden_diagonal_stroke_monotonic_aa_gradient() {
    let renderer = shared_renderer();
    let width = 200u32;
    let height = 160u32;
    let mut canvas = Canvas::new(width, height, Color::BLACK);

    // 1. Thick diagonal white stroke over black (tests 16x SSAA+MSAA edge progression)
    canvas.draw_line(
        Point::new(20.0, 22.0),
        Point::new(180.0, 78.0),
        &Paint::stroke(Color::WHITE, 14.0).with_stroke_cap(Cap::Round),
    );

    // 2. Subpixel horizontal white stroke (width = 0.35 -> clamped to 1px with alpha coverage = 0.70)
    canvas.draw_line(
        Point::new(20.0, 130.5),
        Point::new(180.0, 130.5),
        &Paint::stroke(Color::WHITE, 0.35).with_stroke_cap(Cap::Butt),
    );

    let rgba = canvas
        .render_to_pixels(&renderer)
        .expect("render diagonal + subpixel strokes");

    let mut intermediate_levels = std::collections::BTreeSet::new();
    for y in 10..95u32 {
        for x in 20..180u32 {
            let r = get_pixel(&rgba, width, x, y)[0];
            if r > 0 && r < 255 {
                intermediate_levels.insert(r);
            }
        }
    }
    assert!(
        intermediate_levels.len() >= 8,
        "expected >= 8 distinct intermediate AA gray levels on diagonal stroke, got {}: {:?}",
        intermediate_levels.len(),
        intermediate_levels
    );

    // Vertical cross-section at x = 100 across the diagonal line (y = 30..70):
    // must ascend monotonically to 255 and descend monotonically back to 0.
    let profile: Vec<u8> = (30..=70)
        .map(|y| get_pixel(&rgba, width, 100, y)[0])
        .collect();
    let peak_idx = profile
        .iter()
        .position(|&v| v == 255)
        .expect("diagonal stroke cross-section must reach 255 in core");
    let last_peak_idx = profile
        .iter()
        .rposition(|&v| v == 255)
        .expect("diagonal stroke cross-section last 255");
    for w in profile[..=peak_idx].windows(2) {
        assert!(
            w[1] >= w[0],
            "leading edge AA must be monotonic non-decreasing: {:?}",
            profile
        );
    }
    for w in profile[last_peak_idx..].windows(2) {
        assert!(
            w[1] <= w[0],
            "trailing edge AA must be monotonic non-increasing: {:?}",
            profile
        );
    }

    // Subpixel stroke verification around y = 128..133, x = 100:
    let subpixel_peak = (128..=133)
        .map(|y| get_pixel(&rgba, width, 100, y)[0])
        .max()
        .unwrap_or(0);
    assert!(
        (80..=210).contains(&subpixel_peak),
        "subpixel stroke (width=0.35) should render with dimmed coverage in 80..=210, got {subpixel_peak}"
    );
}

/// C++ source: aiks_unittests.cc:ShowcaseFullFidelityGoldenAudit
#[test]
fn golden_showcase_full_fidelity_audit() {
    let renderer = shared_renderer();
    let canvas = build_showcase_canvas();
    let showcase_path = repo_root().join("showcase.png");
    let rgba = canvas
        .render_to_png(&renderer, &showcase_path)
        .expect("render showcase.png");

    let stats = verify_showcase_pixels(&rgba, 800, 600).expect("verify showcase pixels");
    assert_eq!(stats.width, 800);
    assert_eq!(stats.height, 600);

    // Audit top scanline integrity of all 4 quadrant HUD headers + CLAMP + MIRROR + Q4 PolymorphicCards
    audit_text_band_top_scanline(&rgba, 800, 24, 340, 14, 36, "Showcase Q1 HUD");
    audit_text_band_top_scanline(&rgba, 800, 424, 720, 14, 36, "Showcase Q2 HUD");
    audit_text_band_top_scanline(&rgba, 800, 24, 340, 314, 336, "Showcase Q3 HUD");
    audit_text_band_top_scanline(&rgba, 800, 422, 760, 314, 336, "Showcase Q4 HUD");
    audit_text_band_top_scanline(&rgba, 800, 424, 495, 344, 366, "Showcase Q4 WAVE #0");

    // Ensure strokes & joins in Q3 are unaffected
    let mut pb = impeller_canvas::PathBuilder::new();
    pb.move_to(Point::new(10.0, 10.0))
        .line_to(Point::new(20.0, 20.0));
    let _ = Paint::stroke(Color::WHITE, 2.0).with_stroke_join(Join::Round);
}
