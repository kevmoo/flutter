Pure-Rust port of Flutter's Impeller 2D graphics engine targeting native (`wgpu` Metal/Vulkan/DX12) and WebAssembly (`wasm32-unknown-unknown` WebGPU and WebGL2), including a drop-in 157-function `skwasm` C-ABI runtime for unmodified `dart2wasm` Flutter Web applications.

> [!NOTE]
>
> All 8 core crates enforce `#![forbid(unsafe_code)]`; `unsafe` is strictly isolated to the `extern "C"` FFI boundary in `crates/impeller_skwasm_ffi`.

## Workspace Crates

| Crate | Safety | Responsibility |
| :--- | :--- | :--- |
| `crates/impeller_geometry` | `#![forbid(unsafe_code)]` | 2D/3D points, vectors, matrices, rectangles, radii, colors, and analytic `RoundRectGeometry` / `DiffRoundRectGeometry` |
| `crates/impeller_path` | `#![forbid(unsafe_code)]` | Path verbs, `f64` quadratic/cubic Bezier subdivision, convexity tracking, and AABB bounds |
| `crates/impeller_core` | `#![forbid(unsafe_code)]` | Blend modes, color sources, tile modes, sampler descriptors, and vertex/uniform layouts |
| `crates/impeller_tessellator` | `#![forbid(unsafe_code)]` | $O(V)$ convex fast-path, disjoint-convex fan fast-path, $O(V \cdot R)$ reflex-culled ear-clipping PSLG fill, stroke expansion, and exact 2D boolean `combine_paths` |
| `crates/impeller_typographer` | `#![forbid(unsafe_code)]` | `cosmic-text` + `swash` shaping, line wrapping, bidi, fake-italic synthesis, and incremental `etagere` shelf-packed `GlyphAtlas` |
| `crates/impeller_wgpu` | `#![forbid(unsafe_code)]` | Consolidated `FrameArena` (`<= 3` buffer uploads/frame), `4x MSAA` WebGPU / `2x SSAA` WebGL2, on-demand `Depth24PlusStencil8` clipping, pooled child-pass-first `SaveLayerPool`, and WGSL shaders (`Linear`/`Radial`/`Sweep`/`Conical` gradients, separable Gaussian `Blur`, `Dilate`/`Erode`, and `ColorFilter`) |
| `crates/impeller_canvas` | `#![forbid(unsafe_code)]` | Stateful `Canvas`, `DisplayList` / `DisplayListBuilder`, `CanCollapseIntoParentPass` `SaveLayer` elision, 9-slice `draw_image_nine`, 1-hit animated path cache filter, and `SrcOver` draw batching |
| `crates/impeller_web` | `#![forbid(unsafe_code)]` | Standalone `wasm-bindgen` browser canvas demo harness |
| `crates/impeller_skwasm_ffi` | `#![allow(unsafe_code)]` | 157-export `skwasm` C-ABI implementation, shared Wasm memory bundler (`bundle_skwasm_ffi`), and sub-stage browser telemetry |

## Build & Verification

Run the 4-layer workspace quality gate (`cargo fmt --check`, `cargo clippy -- -D warnings`, `340` unit and golden parity tests, and native showcase rendering):

```bash
scripts/verify_gate.sh all
```

Build the release `wasm32-unknown-unknown` Skwasm FFI bundle (`impeller_skwasm_ffi_bg.wasm` + `impeller_skwasm_ffi.js` + `impeller_skwasm_loader.js`) and verify all 157 `skwasm` C-ABI exports:

```bash
scripts/build_skwasm_ffi.sh
```

## Wasm Bundle Size (`wasm32-unknown-unknown --release`)

| Bundle Variant | Raw Bytes | `gzip -9` Bytes | `brotli -11` (`br`) Bytes | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `impeller_skwasm_ffi_bg.wasm` (as built) | `7,454,883` | `2,627,110` | `1,877,073` | No `wasm-opt` or Rust LTO yet; includes `2.32 MB` raw (`750 KB` `br`) of 4 static fallback fonts |
| `impeller_skwasm_ffi_bg.wasm` (excluding 4 embedded fonts) | `~5,132,995` | `~1,754,000` | `~1,127,180` | Pure Rust engine + `wgpu` (WebGPU + WebGL2 + `naga`) + `cosmic-text` |
| Reference: `skwasm.wasm` (C++ Skia WebGL2) | `3,609,880` | `1,561,586` | `1,215,118` | Emscripten + Binaryen `wasm-opt` + LTO |
| Reference: `wimp.wasm` (C++ Web Impeller) | `3,607,181` | `1,472,609` | `1,139,261` | Emscripten + Binaryen `wasm-opt` + LTO |
