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
| `crates/impeller_tessellator` | `#![forbid(unsafe_code)]` | $O(V)$ convex fast-path, disjoint-convex fan fast-path, $O(V \cdot R)$ reflex-culled ear-clipping PSLG fill, and stroke expansion |
| `crates/impeller_typographer` | `#![forbid(unsafe_code)]` | `cosmic-text` + `swash` shaping, line wrapping, bidi, fake-italic synthesis, and incremental `etagere` shelf-packed `GlyphAtlas` |
| `crates/impeller_wgpu` | `#![forbid(unsafe_code)]` | Consolidated `FrameArena` (`<= 3` buffer uploads/frame), `4x MSAA` WebGPU / `2x SSAA` WebGL2 pipelines, and WGSL shaders |
| `crates/impeller_canvas` | `#![forbid(unsafe_code)]` | Stateful `Canvas`, `DisplayList` / `DisplayListBuilder`, 1-hit animated path cache filter, and `SrcOver` draw batching |
| `crates/impeller_web` | `#![forbid(unsafe_code)]` | Standalone `wasm-bindgen` browser canvas demo harness |
| `crates/impeller_skwasm_ffi` | `#![allow(unsafe_code)]` | 157-export `skwasm` C-ABI implementation, shared Wasm memory bundler (`bundle_skwasm_ffi`), and sub-stage browser telemetry |

## Build & Verification

Run the 4-layer workspace quality gate (`cargo fmt --check`, `cargo clippy -- -D warnings`, `324` unit and golden parity tests, and native showcase rendering):

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
| `impeller_skwasm_ffi_bg.wasm` (as built) | `7,329,878` | `2,582,641` | `1,853,017` | No `wasm-opt` or Rust LTO yet; includes `2.32 MB` raw (`750 KB` `br`) of 4 static fallback fonts |
| `impeller_skwasm_ffi_bg.wasm` (excluding 4 embedded fonts) | `~5,007,990` | `~1,710,000` | `~1,103,124` | Pure Rust engine + `wgpu` (WebGPU + WebGL2 + `naga`) + `cosmic-text` |
| Reference: `skwasm.wasm` (C++ Skia WebGL2) | `3,609,880` | `1,561,586` | `1,215,118` | Emscripten + Binaryen `wasm-opt` + LTO |
| Reference: `wimp.wasm` (C++ Web Impeller) | `3,607,181` | `1,472,609` | `1,139,261` | Emscripten + Binaryen `wasm-opt` + LTO |

## Core Rendering Roadmap

- **GPU stencil clipping (`clipRRect` & `clipPath`)**: Replace AABB scissor fallback with hardware depth-stencil (`Depth24PlusStencil8`) clip stack supporting `ClipOperation::Intersect` and `Difference`.
- **Offscreen `saveLayer` & `ImageFilter`**: Route `saveLayer` with `ImageFilter` (`Blur`, `Dilate`, `Erode`, `Matrix`, `Compose`) or non-associative `BlendMode` / overlapping alpha children through pooled offscreen textures and separable Gaussian blur passes.
- **`SweepGradient` & two-point `ConicalGradient`**: Replace radial-gradient fallback in `impeller_skwasm_ffi` with dedicated WGSL fragment shaders.
- **9-slice `drawImageNine` & boolean `path_combine`**: Implement 9-patch UV/rect splitting and polygon boolean operations (`Union`, `Intersect`, `Difference`, `Xor`).
