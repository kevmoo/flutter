#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/pkg}"
TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/impeller_target_wasm_bundle}"

cd "${ROOT_DIR}"

echo "=== Building impeller_web (wasm32-unknown-unknown --release) ==="
CARGO_TARGET_DIR="${TARGET_DIR}" cargo build --locked --target wasm32-unknown-unknown --release -p impeller_web --lib

WASM_FILE="${TARGET_DIR}/wasm32-unknown-unknown/release/impeller_web.wasm"
if [[ ! -f "${WASM_FILE}" ]]; then
  echo "ERROR: Expected Wasm artifact at ${WASM_FILE}" >&2
  exit 1
fi

echo "=== Generating wasm-bindgen ES module into ${OUT_DIR} ==="
CARGO_TARGET_DIR="${TARGET_DIR}" cargo run --locked --release -p impeller_web --bin bundle_wasm -- "${WASM_FILE}" "${OUT_DIR}"

ls -lh "${OUT_DIR}/impeller_web.js" "${OUT_DIR}/impeller_web_bg.wasm"
