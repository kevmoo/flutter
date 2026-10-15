#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/pkg_skwasm}"
TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/impeller_target_skwasm_bundle}"
ABI_CHECKER="/usr/local/google/home/kevmoo/.gemini/jetski/brain/aaf8f361-7fe2-4fe7-8caa-40ddbfdb8335/scratch/verify_skwasm_abi.py"

cd "${ROOT_DIR}"

echo "=== Building impeller_skwasm_ffi (wasm32-unknown-unknown --release) ==="
CARGO_TARGET_DIR="${TARGET_DIR}" cargo build --locked --target wasm32-unknown-unknown --release -p impeller_skwasm_ffi --lib

WASM_FILE="${TARGET_DIR}/wasm32-unknown-unknown/release/impeller_skwasm_ffi.wasm"
if [[ ! -f "${WASM_FILE}" ]]; then
  echo "ERROR: Expected Wasm artifact at ${WASM_FILE}" >&2
  exit 1
fi

WASM_OPT="/usr/local/google/home/kevmoo/github/flutter/bin/cache/dart-sdk/bin/utils/wasm-opt"
if [[ -x "${WASM_OPT}" ]]; then
  echo "=== Optimizing raw Wasm with wasm-opt -Oz --traps-never-happen --converge ==="
  "${WASM_OPT}" -Oz --traps-never-happen --converge \
    --enable-bulk-memory \
    --enable-threads \
    --enable-nontrapping-float-to-int \
    --enable-sign-ext \
    --enable-mutable-globals \
    "${WASM_FILE}" -o "${WASM_FILE}"
fi

echo "=== Generating wasm-bindgen ES module into ${OUT_DIR} ==="
CARGO_TARGET_DIR="${TARGET_DIR}" cargo run --locked --release -p impeller_skwasm_ffi --bin bundle_skwasm_ffi -- "${WASM_FILE}" "${OUT_DIR}"

cp "${ROOT_DIR}/crates/impeller_skwasm_ffi/js/impeller_skwasm_loader.js" "${OUT_DIR}/impeller_skwasm_loader.js"
cp "${ROOT_DIR}/crates/impeller_skwasm_ffi/js/impeller_worker.js" "${OUT_DIR}/impeller_worker.js"

if [[ -f "${ABI_CHECKER}" ]]; then
  echo "=== Verifying 157 skwasm C-ABI exports against main.dart.wasm ==="
  python3 "${ABI_CHECKER}" "${OUT_DIR}/impeller_skwasm_ffi_bg.wasm"
fi

ls -lh "${OUT_DIR}/impeller_skwasm_ffi.js" "${OUT_DIR}/impeller_skwasm_ffi_bg.wasm" "${OUT_DIR}/impeller_skwasm_loader.js" "${OUT_DIR}/impeller_worker.js"
