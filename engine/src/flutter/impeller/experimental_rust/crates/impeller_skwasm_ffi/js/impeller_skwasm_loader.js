import __wbg_init, {
  skwasm_image_create_from_texture_source_js,
  skwasm_on_worker_render_done_js,
  skwasm_on_worker_size_set_js,
  skwasm_on_worker_surface_initialized_js,
  skwasm_register_callback_js,
  skwasm_register_worker_bridge_js,
  skwasm_set_multi_threaded_js,
  skwasm_set_texture_extractor_js,
  skwasm_surface_set_canvas_js,
} from './impeller_skwasm_ffi.js';

export async function loadImpellerSkwasm(options = {}) {
  const preferWebGpu = options.preferWebGpu !== false;
  const singleThreaded = !!options.singleThreaded;
  globalThis.__impellerPreferWebGpu = preferWebGpu;

  const wasmUrl =
    options.wasmUrl ?? new URL('./impeller_skwasm_ffi_bg.wasm', import.meta.url);
  const rawExports = await __wbg_init({ module_or_path: wasmUrl });

  skwasm_set_multi_threaded_js(false);

  const canMultiThread =
    !singleThreaded &&
    globalThis.crossOriginIsolated === true &&
    typeof Worker !== 'undefined' &&
    typeof OffscreenCanvas !== 'undefined';

  if (canMultiThread) {
    try {
      const workerUrl = new URL('./impeller_worker.js', import.meta.url);
      const worker = new Worker(workerUrl, {
        type: 'module',
        name: 'impeller_raster_worker',
      });

      const workerReady = await new Promise((resolve) => {
        const onMsg = (ev) => {
          if (ev.data && ev.data.type === 'ready') {
            worker.removeEventListener('message', onMsg);
            worker.removeEventListener('error', onErr);
            resolve(true);
          }
        };
        const onErr = (err) => {
          console.warn(
            '[impeller_skwasm_loader] Dedicated raster worker init failed; falling back to ST:',
            err,
          );
          worker.removeEventListener('message', onMsg);
          worker.removeEventListener('error', onErr);
          resolve(false);
        };
        worker.addEventListener('message', onMsg);
        worker.addEventListener('error', onErr);
        worker.postMessage({
          type: 'init',
          wasmUrl: String(wasmUrl),
        });
      });

      if (workerReady) {
        worker.onmessage = (ev) => {
          const d = ev.data;
          if (!d || typeof d.type !== 'string') return;
          switch (d.type) {
            case 'surfaceInitialized':
              skwasm_on_worker_surface_initialized_js(
                d.surfaceHandle >>> 0,
                d.callbackId >>> 0,
                d.contextLostCallbackId >>> 0,
              );
              break;
            case 'sizeSet':
              skwasm_on_worker_size_set_js(d.surfaceHandle >>> 0, d.callbackId >>> 0);
              break;
            case 'renderDone':
              skwasm_on_worker_render_done_js(
                d.surfaceHandle >>> 0,
                d.callbackId >>> 0,
                Array.isArray(d.imageBitmaps) ? d.imageBitmaps : [],
                Number(d.workerElapsedMs || 0),
                Number(d.atlasSyncMs || 0),
                Number(d.wgpuRenderMs || 0),
                Number(d.bitmapTransferMs || 0),
              );
              break;
            default:
              break;
          }
        };

        const postSetCanvas = (surfaceHandle, canvas, meta) => {
          const width = meta[0] >>> 0;
          const height = meta[1] >>> 0;
          const pWebGpu = !!meta[2];
          const callbackId = meta[3] >>> 0;
          const contextLostCallbackId = meta[4] >>> 0;
          worker.postMessage(
            {
              type: 'setCanvas',
              surfaceHandle: surfaceHandle >>> 0,
              canvas,
              width,
              height,
              preferWebGpu: pWebGpu,
              callbackId,
              contextLostCallbackId,
            },
            [canvas],
          );
        };

        const postSetSize = (surfaceHandle, callbackId, dims) => {
          worker.postMessage({
            type: 'setSize',
            surfaceHandle: surfaceHandle >>> 0,
            callbackId: callbackId >>> 0,
            width: dims[0] >>> 0,
            height: dims[1] >>> 0,
          });
        };

        const postRenderFrame = (surfaceHandle, callbackId, packetAb) => {
          worker.postMessage(
            {
              type: 'renderFrame',
              surfaceHandle: surfaceHandle >>> 0,
              callbackId: callbackId >>> 0,
              packet: packetAb,
            },
            [packetAb],
          );
        };

        const postDestroySurface = (surfaceHandle) => {
          worker.postMessage({
            type: 'destroySurface',
            surfaceHandle: surfaceHandle >>> 0,
          });
        };

        skwasm_register_worker_bridge_js(
          postSetCanvas,
          postSetSize,
          postRenderFrame,
          postDestroySurface,
        );
        skwasm_set_multi_threaded_js(true);
        globalThis.__impellerWorker = worker;
      } else {
        worker.terminate();
      }
    } catch (err) {
      console.warn(
        '[impeller_skwasm_loader] Failed to spawn dedicated raster worker; falling back to ST:',
        err,
      );
    }
  }

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
