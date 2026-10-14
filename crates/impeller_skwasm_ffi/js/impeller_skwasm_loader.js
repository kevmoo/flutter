import __wbg_init, {
  skwasm_register_callback_js,
  skwasm_surface_set_canvas_js,
} from './impeller_skwasm_ffi.js';

export async function loadImpellerSkwasm(options = {}) {
  const preferWebGpu = options.preferWebGpu !== false;
  globalThis.__impellerPreferWebGpu = preferWebGpu;

  const wasmUrl =
    options.wasmUrl ?? new URL('./impeller_skwasm_ffi_bg.wasm', import.meta.url);
  const rawExports = await __wbg_init({ module_or_path: wasmUrl });

  const wasmExports = Object.assign({}, rawExports, {
    surface_setCanvas: (surfaceHandle, canvas) =>
      skwasm_surface_set_canvas_js(surfaceHandle, canvas),
  });

  const instance = {
    wasmExports,
    wasmMemory: rawExports.memory,
    addFunction: (fn) => skwasm_register_callback_js(fn),
  };

  if (typeof window !== 'undefined') {
    window._flutter_skwasmInstance = instance;
  }

  return instance;
}

export default loadImpellerSkwasm;
