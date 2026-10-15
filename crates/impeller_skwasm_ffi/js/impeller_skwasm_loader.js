import __wbg_init, {
  skwasm_image_create_from_texture_source_js,
  skwasm_register_callback_js,
  skwasm_set_texture_extractor_js,
  skwasm_surface_set_canvas_js,
} from './impeller_skwasm_ffi.js';

export async function loadImpellerSkwasm(options = {}) {
  const preferWebGpu = options.preferWebGpu !== false;
  globalThis.__impellerPreferWebGpu = preferWebGpu;

  const wasmUrl =
    options.wasmUrl ?? new URL('./impeller_skwasm_ffi_bg.wasm', import.meta.url);
  const rawExports = await __wbg_init({ module_or_path: wasmUrl });

  let decodeCanvas = null;
  let decodeCtx = null;
  const extractTextureSourcePixels = (frame, width, height, dstPtr) => {
    if (!frame) return false;
    const w = Math.max(1, width | 0);
    const h = Math.max(1, height | 0);
    try {
      if (!decodeCanvas) {
        if (typeof OffscreenCanvas !== 'undefined') {
          decodeCanvas = new OffscreenCanvas(w, h);
        } else if (typeof document !== 'undefined') {
          decodeCanvas = document.createElement('canvas');
        }
        if (decodeCanvas) {
          decodeCtx = decodeCanvas.getContext('2d', { willReadFrequently: true });
        }
      }
      if (!decodeCanvas || !decodeCtx) return false;
      if (decodeCanvas.width !== w) decodeCanvas.width = w;
      if (decodeCanvas.height !== h) decodeCanvas.height = h;
      decodeCtx.clearRect(0, 0, w, h);
      decodeCtx.drawImage(frame, 0, 0, w, h);
      const imgData = decodeCtx.getImageData(0, 0, w, h);
      const dst = new Uint8Array(rawExports.memory.buffer, dstPtr >>> 0, w * h * 4);
      dst.set(imgData.data);
      return true;
    } catch (_) {
      return false;
    }
  };
  skwasm_set_texture_extractor_js(extractTextureSourcePixels);

  const wasmExports = Object.assign({}, rawExports, {
    surface_setCanvas: (surfaceHandle, canvas) =>
      skwasm_surface_set_canvas_js(surfaceHandle, canvas),
    image_createFromTextureSource: (frame, width, height, surfaceHandle) =>
      skwasm_image_create_from_texture_source_js(frame, width, height, surfaceHandle),
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
