#![forbid(unsafe_code)]

use impeller_canvas::{build_showcase_canvas, verify_showcase_pixels, WgpuContext};
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let out_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("showcase.png"));

    let renderer = WgpuContext::new_headless()?;
    let canvas = build_showcase_canvas();
    let rgba = canvas.render_to_png(&renderer, &out_path)?;
    let stats = verify_showcase_pixels(&rgba, canvas.width(), canvas.height())?;

    println!(
        "Impeller Rust+wgpu Showcase rendered on {} ({}) -> {} ({}x{}, {} commands)",
        renderer.adapter_name,
        renderer.backend_name,
        out_path.display(),
        stats.width,
        stats.height,
        canvas.commands().len()
    );
    for (idx, q) in stats.quadrants.iter().enumerate() {
        println!(
            "  Quadrant {}: {}/{} non-bg pixels, {} color buckets, mean RGB=({:.1}, {:.1}, {:.1})",
            idx + 1,
            q.non_background_pixels,
            q.total_pixels,
            q.distinct_color_buckets,
            q.mean_r,
            q.mean_g,
            q.mean_b
        );
    }

    Ok(())
}
