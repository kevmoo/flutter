import __wbg_init, {
  skwasm_worker_destroy_surface_js,
  skwasm_worker_init_surface_js,
  skwasm_worker_render_frame_js,
  skwasm_worker_set_size_js,
} from './impeller_skwasm_ffi.js';

let initialized = false;
let queue = Promise.resolve();

async function handleWorkerMessage(msg) {
  switch (msg.type) {
    case 'init': {
      const wasmUrl =
        msg.wasmUrl || new URL('./impeller_skwasm_ffi_bg.wasm', import.meta.url).href;
      await __wbg_init({ module_or_path: wasmUrl });
      initialized = true;
      self.postMessage({ type: 'ready' });
      break;
    }

    case 'setCanvas': {
      const surfaceHandle = msg.surfaceHandle >>> 0;
      const callbackId = msg.callbackId >>> 0;
      const contextLostCallbackId = msg.contextLostCallbackId >>> 0;
      if (!initialized || !msg.canvas) {
        self.postMessage({
          type: 'surfaceInitialized',
          surfaceHandle,
          callbackId,
          contextLostCallbackId,
          ok: false,
        });
        return;
      }
      const ok = await skwasm_worker_init_surface_js(
        surfaceHandle,
        msg.canvas,
        msg.width >>> 0,
        msg.height >>> 0,
        !!msg.preferWebGpu,
      );
      self.postMessage({
        type: 'surfaceInitialized',
        surfaceHandle,
        callbackId,
        contextLostCallbackId,
        ok: !!ok,
      });
      break;
    }

    case 'setSize': {
      const surfaceHandle = msg.surfaceHandle >>> 0;
      const callbackId = msg.callbackId >>> 0;
      if (initialized) {
        skwasm_worker_set_size_js(surfaceHandle, msg.width >>> 0, msg.height >>> 0);
      }
      self.postMessage({
        type: 'sizeSet',
        surfaceHandle,
        callbackId,
      });
      break;
    }

    case 'renderFrame': {
      const surfaceHandle = msg.surfaceHandle >>> 0;
      const callbackId = msg.callbackId >>> 0;
      if (!initialized || !(msg.packet instanceof ArrayBuffer)) {
        self.postMessage({
          type: 'renderDone',
          surfaceHandle,
          callbackId,
          imageBitmaps: [],
          workerElapsedMs: 0,
          atlasSyncMs: 0,
          wgpuRenderMs: 0,
          bitmapTransferMs: 0,
        });
        return;
      }
      const packetBytes = new Uint8Array(msg.packet);
      const res = skwasm_worker_render_frame_js(surfaceHandle, packetBytes);
      const imageBitmaps = Array.isArray(res?.imageBitmaps) ? res.imageBitmaps : [];
      const transferList = imageBitmaps.filter(
        (bmp) => bmp && typeof ImageBitmap !== 'undefined' && bmp instanceof ImageBitmap,
      );
      self.postMessage(
        {
          type: 'renderDone',
          surfaceHandle,
          callbackId,
          imageBitmaps,
          workerElapsedMs: Number(res?.workerElapsedMs || 0),
          atlasSyncMs: Number(res?.atlasSyncMs || 0),
          wgpuRenderMs: Number(res?.wgpuRenderMs || 0),
          bitmapTransferMs: Number(res?.bitmapTransferMs || 0),
        },
        transferList,
      );
      break;
    }

    case 'destroySurface': {
      if (initialized) {
        skwasm_worker_destroy_surface_js(msg.surfaceHandle >>> 0);
      }
      break;
    }

    default:
      break;
  }
}

self.onmessage = (event) => {
  const msg = event.data;
  if (!msg || typeof msg.type !== 'string') return;
  queue = queue
    .then(() => handleWorkerMessage(msg))
    .catch((err) => {
      console.error('[impeller_worker] Unhandled message error:', err);
    });
};
