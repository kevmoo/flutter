#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
fn read_leb128_u32(bytes: &[u8], mut pos: usize) -> (u32, usize) {
    let mut result = 0u32;
    let mut shift = 0u32;
    loop {
        let byte = bytes[pos];
        pos += 1;
        result |= u32::from(byte & 0x7F) << shift;
        if (byte & 0x80) == 0 {
            return (result, pos);
        }
        shift += 7;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn write_leb128_u32(mut value: u32, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn patch_wasm_memory_to_shared(wasm_bytes: &[u8]) -> Vec<u8> {
    assert!(
        wasm_bytes.len() >= 8 && &wasm_bytes[0..4] == b"\0asm",
        "invalid Wasm header"
    );
    let mut out = Vec::with_capacity(wasm_bytes.len() + 8);
    out.extend_from_slice(&wasm_bytes[0..8]);

    let mut pos = 8usize;
    let mut patched_memory = false;
    while pos < wasm_bytes.len() {
        let sec_start = pos;
        let sec_id = wasm_bytes[pos];
        pos += 1;
        let (sec_len, payload_start) = read_leb128_u32(wasm_bytes, pos);
        let payload_end = payload_start + (sec_len as usize);

        if sec_id == 5 {
            let (mem_count, p_after_count) = read_leb128_u32(wasm_bytes, payload_start);
            assert_eq!(mem_count, 1, "expected exactly 1 Wasm memory in section 5");
            let flags = wasm_bytes[p_after_count];
            let (min_pages, _p_after_min) = read_leb128_u32(wasm_bytes, p_after_count + 1);
            let _ = flags;

            let mut new_payload = Vec::new();
            write_leb128_u32(1, &mut new_payload);
            new_payload.push(0x03); // shared (0x02) | has_max (0x01)
            write_leb128_u32(min_pages, &mut new_payload);
            write_leb128_u32(32768, &mut new_payload); // 2 GiB max pages (matches main.dart.wasm import)

            out.push(5);
            write_leb128_u32(new_payload.len() as u32, &mut out);
            out.extend_from_slice(&new_payload);
            patched_memory = true;
        } else {
            out.extend_from_slice(&wasm_bytes[sec_start..payload_end]);
        }

        pos = payload_end;
    }

    assert!(patched_memory, "Wasm Section 5 (Memory) not found");
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn patch_js_for_shared_array_buffer(js_src: &str) -> String {
    let mut patched = js_src.replace(
        "cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len))",
        "cachedTextDecoder.decode(getUint8ArrayMemory0().slice(ptr, ptr + len))",
    );
    assert!(
        patched != js_src,
        "failed to patch decodeText subarray -> slice in impeller_skwasm_ffi.js"
    );

    let before_encode = patched.clone();
    patched = patched.replace(
        "const ret = cachedTextEncoder.encodeInto(arg, view);",
        "const buf = cachedTextEncoder.encode(arg); view.set(buf); const ret = { written: buf.length };",
    );
    assert!(
        patched != before_encode,
        "failed to patch passStringToWasm0 encodeInto in impeller_skwasm_ffi.js"
    );

    let typed_caches = [
        "cachedFloat32ArrayMemory0",
        "cachedFloat64ArrayMemory0",
        "cachedInt8ArrayMemory0",
        "cachedInt16ArrayMemory0",
        "cachedInt32ArrayMemory0",
        "cachedUint8ArrayMemory0",
        "cachedUint16ArrayMemory0",
        "cachedUint32ArrayMemory0",
        "cachedBigInt64ArrayMemory0",
        "cachedBigUint64ArrayMemory0",
    ];
    let mut replaced_any_cache = false;
    for name in typed_caches {
        let needle = format!("{name}.byteLength === 0");
        let replacement = format!("{name}.buffer !== wasm.memory.buffer");
        if patched.contains(&needle) {
            patched = patched.replace(&needle, &replacement);
            replaced_any_cache = true;
        }
    }
    assert!(
        replaced_any_cache,
        "failed to patch cached*ArrayMemory0.byteLength === 0 in impeller_skwasm_ffi.js"
    );

    patched
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let mut args = std::env::args().skip(1);
    let wasm_path = args
        .next()
        .expect("usage: bundle_skwasm_ffi <input_wasm_path> <out_dir>");
    let out_dir = args
        .next()
        .expect("usage: bundle_skwasm_ffi <input_wasm_path> <out_dir>");

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

    let out_wasm_path = std::path::Path::new(&out_dir).join("impeller_skwasm_ffi_bg.wasm");
    let raw_wasm = std::fs::read(&out_wasm_path).expect("failed to read generated bg.wasm");
    let shared_wasm = patch_wasm_memory_to_shared(&raw_wasm);
    std::fs::write(&out_wasm_path, shared_wasm).expect("failed to write shared-memory bg.wasm");

    let out_js_path = std::path::Path::new(&out_dir).join("impeller_skwasm_ffi.js");
    let raw_js = std::fs::read_to_string(&out_js_path).expect("failed to read generated .js");
    let shared_js = patch_js_for_shared_array_buffer(&raw_js);
    std::fs::write(&out_js_path, shared_js).expect("failed to write SharedArrayBuffer-patched .js");

    println!(
        "Bundled impeller_skwasm_ffi wasm-bindgen ES module (shared memory, max=32768 pages) into {out_dir}"
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {}
