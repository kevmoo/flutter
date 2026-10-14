#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let mut args = std::env::args().skip(1);
    let wasm_path = args
        .next()
        .expect("usage: bundle_wasm <input_wasm_path> <out_dir>");
    let out_dir = args
        .next()
        .expect("usage: bundle_wasm <input_wasm_path> <out_dir>");

    std::fs::create_dir_all(&out_dir).expect("failed to create output directory");

    let mut bindgen = wasm_bindgen_cli_support::Bindgen::new();
    bindgen
        .input_path(&wasm_path)
        .web(true)
        .expect("failed to configure --target web")
        .typescript(true)
        .keep_debug(false)
        .remove_name_section(true)
        .remove_producers_section(true)
        .generate(&out_dir)
        .expect("wasm-bindgen bundle generation failed");

    println!("Bundled wasm-bindgen ES module into {out_dir}");
}

#[cfg(target_arch = "wasm32")]
fn main() {}
