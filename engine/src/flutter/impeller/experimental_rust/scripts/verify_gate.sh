#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CRATE="${1:-all}"
TARGET_DIR="/tmp/impeller_rs_gate_${CRATE}"

check_crate_hygiene() {
  local crate_name="$1"
  local min_tests="$2"
  local crate_dir="$REPO_ROOT/crates/$crate_name"

  echo "=== Checking hygiene, C++ provenance, & wasm32 compile for $crate_name (min $min_tests C++ tests) ==="

  if [[ "$crate_name" != "impeller_skwasm_ffi" && "$crate_name" != "impeller_interop_c" ]]; then
    if ! grep -q '#!\[forbid(unsafe_code)\]' "$crate_dir/src/lib.rs"; then
      echo "FAIL [$crate_name]: Missing #![forbid(unsafe_code)] in src/lib.rs"
      exit 1
    fi
  fi

  if grep -rnE 'todo!\(|unimplemented!\(|#\[allow\(dead_code\)\]|#\[allow\(unused' "$crate_dir/src/"; then
    echo "FAIL [$crate_name]: Found forbidden todo!/unimplemented!/allow(dead_code)/allow(unused) in $crate_dir/src/"
    exit 1
  fi

  local test_count
  test_count=$(grep -rn '#\[test\]' "$crate_dir/" | wc -l | tr -d ' ')
  local cpp_source_count
  cpp_source_count=$(grep -rn '/// C++ source:' "$crate_dir/" | wc -l | tr -d ' ')

  echo "  -> Found $test_count #[test] functions and $cpp_source_count '/// C++ source:' provenance tags"
  if (( test_count < min_tests )); then
    echo "FAIL [$crate_name]: Only $test_count #[test] functions found (required >= $min_tests)"
    exit 1
  fi
  if (( cpp_source_count < min_tests )); then
    echo "FAIL [$crate_name]: Only $cpp_source_count '/// C++ source:' tags found (required >= $min_tests)"
    exit 1
  fi

  CARGO_TARGET_DIR="$TARGET_DIR" cargo clippy --locked -p "$crate_name" --all-targets -- -D warnings
  CARGO_TARGET_DIR="$TARGET_DIR" cargo clippy --locked --target wasm32-unknown-unknown -p "$crate_name" --lib -- -D warnings
  CARGO_TARGET_DIR="$TARGET_DIR" cargo test --locked -p "$crate_name"
  echo "PASS [$crate_name]"
}

case "$CRATE" in
  impeller_geometry)    check_crate_hygiene impeller_geometry 40 ;;
  impeller_path)        check_crate_hygiene impeller_path 25 ;;
  impeller_core)        check_crate_hygiene impeller_core 15 ;;
  impeller_tessellator) check_crate_hygiene impeller_tessellator 25 ;;
  impeller_typographer) check_crate_hygiene impeller_typographer 20 ;;
  impeller_wgpu)        check_crate_hygiene impeller_wgpu 26 ;;
  impeller_canvas)      check_crate_hygiene impeller_canvas 22 ;;
  impeller_web)         check_crate_hygiene impeller_web 2 ;;
  impeller_skwasm_ffi)  check_crate_hygiene impeller_skwasm_ffi 10 ;;
  impeller_interop_c)   check_crate_hygiene impeller_interop_c 12 ;;
  all)
    check_crate_hygiene impeller_geometry 40
    check_crate_hygiene impeller_path 25
    check_crate_hygiene impeller_core 15
    check_crate_hygiene impeller_tessellator 25
    check_crate_hygiene impeller_typographer 20
    check_crate_hygiene impeller_wgpu 26
    check_crate_hygiene impeller_canvas 22
    check_crate_hygiene impeller_web 2
    check_crate_hygiene impeller_skwasm_ffi 10
    check_crate_hygiene impeller_interop_c 12
    CARGO_TARGET_DIR="$TARGET_DIR" cargo test --locked -p impeller_canvas --test golden_parity
    CARGO_TARGET_DIR="$TARGET_DIR" cargo run --locked -p impeller_canvas --bin showcase -- "$REPO_ROOT/showcase.png"
    ;;
  *)
    echo "Usage: $0 [impeller_geometry|impeller_path|impeller_core|impeller_tessellator|impeller_typographer|impeller_wgpu|impeller_canvas|impeller_web|impeller_skwasm_ffi|impeller_interop_c|all]"
    exit 1
    ;;
esac
