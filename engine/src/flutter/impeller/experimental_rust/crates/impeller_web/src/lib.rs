#![forbid(unsafe_code)]

use impeller_canvas::{
    draw_bouncy_workload, draw_morphing_grid_workload, draw_showcase_scene, Canvas, Color, FontId,
    TypographerContext, WgpuRenderer,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Embedded `Roboto-Bold.ttf` bytes (`assets/fonts/Roboto-Bold.ttf`).
pub const ROBOTO_BOLD_BYTES: &[u8] = include_bytes!("../../../assets/fonts/Roboto-Bold.ttf");
/// Embedded `DejaVuSansMono-Bold.ttf` bytes (`assets/fonts/DejaVuSansMono-Bold.ttf`).
pub const DEJAVU_MONO_BOLD_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/DejaVuSansMono-Bold.ttf");
/// Embedded `MaterialIcons-Regular.otf` bytes (`assets/fonts/MaterialIcons-Regular.otf`).
pub const MATERIAL_ICONS_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/MaterialIcons-Regular.otf");

/// Dispatches `"grid"`, `"bouncy"`, or `"showcase"` onto `canvas` scaled by `dpr`.
#[allow(clippy::too_many_arguments)]
pub fn render_workload_to_canvas(
    canvas: &mut Canvas,
    typographer: &mut TypographerContext,
    font_id: FontId,
    mono_font_id: FontId,
    icon_font_id: FontId,
    workload: &str,
    width: f32,
    height: f32,
    dpr: f32,
    node_count: u32,
    anim_value: f32,
) {
    let safe_w = width.max(1.0);
    let safe_h = height.max(1.0);
    let safe_dpr = dpr.clamp(0.5, 4.0);

    canvas.save();
    canvas.scale(safe_dpr, safe_dpr);
    match workload {
        "bouncy" => {
            draw_bouncy_workload(
                canvas,
                typographer,
                font_id,
                mono_font_id,
                icon_font_id,
                safe_w,
                safe_h,
                node_count as usize,
                anim_value,
            );
        }
        "showcase" => {
            canvas.scale(safe_w / 800.0, safe_h / 600.0);
            let taken = std::mem::take(typographer);
            canvas.set_typographer(taken);
            draw_showcase_scene(canvas);
            if let Some(returned) = canvas.take_typographer() {
                *typographer = returned;
            }
        }
        _ => {
            draw_morphing_grid_workload(
                canvas,
                typographer,
                font_id,
                mono_font_id,
                icon_font_id,
                safe_w,
                safe_h,
                node_count as usize,
                anim_value,
            );
        }
    }
    canvas.restore();
}

/// Cross-platform headless workload driver used by host unit tests and benchmarks.
pub struct HeadlessWebWorkloadEngine {
    pub renderer: WgpuRenderer,
    pub typographer: TypographerContext,
    pub font_id: FontId,
    pub mono_font_id: FontId,
    pub icon_font_id: FontId,
}

impl HeadlessWebWorkloadEngine {
    pub fn new(renderer: WgpuRenderer) -> Self {
        Self {
            renderer,
            typographer: TypographerContext::new(),
            font_id: FontId::RobotoBold,
            mono_font_id: FontId::RobotoMonoBold,
            icon_font_id: FontId::MaterialIcons,
        }
    }

    pub fn render_frame_rgba(
        &mut self,
        workload: &str,
        width: f32,
        height: f32,
        dpr: f32,
        node_count: u32,
        anim_value: f32,
    ) -> Result<(Vec<u8>, usize), String> {
        let safe_dpr = dpr.clamp(0.5, 4.0);
        let phys_w = ((width.max(1.0) * safe_dpr).round() as u32).max(1);
        let phys_h = ((height.max(1.0) * safe_dpr).round() as u32).max(1);
        let clear_color = Color::make_rgba8(0x09, 0x0C, 0x10, 0xFF);
        let mut canvas = Canvas::new(phys_w, phys_h, clear_color);
        render_workload_to_canvas(
            &mut canvas,
            &mut self.typographer,
            self.font_id,
            self.mono_font_id,
            self.icon_font_id,
            workload,
            width,
            height,
            safe_dpr,
            node_count,
            anim_value,
        );
        let cmd_count = canvas.commands().len();
        let rgba = canvas.render_to_pixels(&self.renderer)?;
        Ok((rgba, cmd_count))
    }
}

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or_else(js_sys::Date::now)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct ImpellerWebEngine {
    renderer: WgpuRenderer,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    typographer: TypographerContext,
    font_id: FontId,
    mono_font_id: FontId,
    icon_font_id: FontId,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl ImpellerWebEngine {
    pub async fn create(
        canvas_id: String,
        prefer_webgpu: bool,
    ) -> Result<ImpellerWebEngine, JsValue> {
        let window =
            web_sys::window().ok_or_else(|| JsValue::from_str("Missing browser window"))?;
        let document = window
            .document()
            .ok_or_else(|| JsValue::from_str("Missing DOM document"))?;
        let element = document
            .get_element_by_id(&canvas_id)
            .ok_or_else(|| JsValue::from_str(&format!("Canvas element #{canvas_id} not found")))?;
        let canvas: web_sys::HtmlCanvasElement = element
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| {
                JsValue::from_str(&format!(
                    "Element #{canvas_id} is not an HtmlCanvasElement"
                ))
            })?;

        let (renderer, surface, config) =
            WgpuRenderer::new_for_canvas_with_backend(canvas, prefer_webgpu)
                .await
                .map_err(|e| JsValue::from_str(&e))?;

        Ok(Self {
            renderer,
            surface,
            config,
            typographer: TypographerContext::new(),
            font_id: FontId::RobotoBold,
            mono_font_id: FontId::RobotoMonoBold,
            icon_font_id: FontId::MaterialIcons,
        })
    }

    pub fn backend_name(&self) -> String {
        self.renderer.backend_name.clone()
    }

    pub fn adapter_name(&self) -> String {
        self.renderer.adapter_name.clone()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.config.width != w || self.config.height != h {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(self.renderer.device(), &self.config);
        }
    }

    pub fn render_frame(
        &mut self,
        workload: &str,
        width: f32,
        height: f32,
        dpr: f32,
        node_count: u32,
        anim_value: f32,
    ) -> Vec<f32> {
        let t0 = now_ms();
        let safe_dpr = dpr.clamp(0.5, 4.0);
        let phys_w = ((width.max(1.0) * safe_dpr).round() as u32).max(1);
        let phys_h = ((height.max(1.0) * safe_dpr).round() as u32).max(1);
        self.resize(phys_w, phys_h);

        let clear_color = Color::make_rgba8(0x09, 0x0C, 0x10, 0xFF);
        let mut canvas = Canvas::new(phys_w, phys_h, clear_color);
        render_workload_to_canvas(
            &mut canvas,
            &mut self.typographer,
            self.font_id,
            self.mono_font_id,
            self.icon_font_id,
            workload,
            width,
            height,
            safe_dpr,
            node_count,
            anim_value,
        );
        let cmd_count = canvas.commands().len();
        let t1 = now_ms();
        let build_ms = (t1 - t0) as f32;

        if let Ok(frame) = self.surface.get_current_texture() {
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let _ = self.renderer.render_to_texture_view(
                &view,
                None,
                phys_w,
                phys_h,
                self.config.format,
                1,
                clear_color,
                canvas.commands(),
            );
            frame.present();
        }

        let t2 = now_ms();
        let raster_ms = (t2 - t1) as f32;
        let total_ms = (t2 - t0) as f32;
        vec![build_ms, raster_ms, cmd_count as f32, total_ms]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C++ source: aiks_unittests.cc:EmbeddedFontAssetsNonEmpty
    #[test]
    fn test_embedded_font_assets_non_empty() {
        assert!(ROBOTO_BOLD_BYTES.len() > 10_000);
        assert!(DEJAVU_MONO_BOLD_BYTES.len() > 10_000);
        assert!(MATERIAL_ICONS_BYTES.len() > 1_000);
    }

    /// C++ source: aiks_unittests.cc:WebWorkloadEngineRendersGridBouncyAndShowcase
    #[test]
    fn test_web_workload_engine_renders_grid_bouncy_and_showcase() {
        let renderer = WgpuRenderer::new_headless().expect("headless wgpu renderer");
        let mut engine = HeadlessWebWorkloadEngine::new(renderer);

        for workload in ["grid", "bouncy", "showcase"] {
            let (rgba, cmd_count) = engine
                .render_frame_rgba(workload, 400.0, 300.0, 1.0, 64, 0.25)
                .expect("render web workload");
            assert!(
                cmd_count > 0,
                "workload {workload} should emit draw commands"
            );
            assert_eq!(rgba.len(), 400 * 300 * 4);
            let non_bg = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|px| {
                    (px[0] as i32 - 0x09).abs()
                        + (px[1] as i32 - 0x0C).abs()
                        + (px[2] as i32 - 0x10).abs()
                        > 10
                })
                .count();
            assert!(
                non_bg > 20_000,
                "workload {workload} should render >20k non-bg pixels, got {non_bg}"
            );
        }
    }
}
