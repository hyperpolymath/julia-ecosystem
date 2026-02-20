#!/usr/bin/env bash
# SPDX-License-Identifier: PMPL-1.0-or-later

set -Eeuo pipefail

STRICT_JULIA="${READINESS_STRICT_JULIA:-0}"
if [ "${CI:-}" = "true" ]; then
  STRICT_JULIA=1
fi

failures=0

pass() {
  printf '[readiness] PASS: %s\n' "$1"
}

warn() {
  printf '[readiness] WARN: %s\n' "$1"
}

fail() {
  printf '[readiness] FAIL: %s\n' "$1"
  failures=$((failures + 1))
}

require_file() {
  local path="$1"
  if [ -f "$path" ]; then
    pass "required file exists: $path"
  else
    fail "required file missing: $path"
  fi
}

check_project_version() {
  local version
  version="$(sed -n 's/^version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' Project.toml | head -n1)"
  if [ -z "$version" ]; then
    fail "Project.toml version not found"
    return
  fi

  if printf '%s' "$version" | rg -q '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    pass "Project.toml semantic version: $version"
  else
    fail "Project.toml version is not semver: $version"
  fi
}

run_julia_smoke() {
  if ! command -v julia >/dev/null 2>&1; then
    if [ "$STRICT_JULIA" = "1" ]; then
      fail "julia not found on PATH"
    else
      warn "julia not found on PATH (non-strict mode)"
    fi
    return
  fi

  local cmd
  cmd='using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.build(); Pkg.test(); pkg=Pkg.project().name; Core.eval(Main, Expr(:using, Symbol(pkg))); println(pkg * " load OK")'

  if julia --project=. -e "$cmd"; then
    pass "julia add/precompile/build/test/load smoke"
  else
    if [ "$STRICT_JULIA" = "1" ]; then
      fail "julia add/precompile/build/test/load smoke"
    else
      warn "julia smoke failed in non-strict mode"
    fi
  fi
}

main() {
  if ! command -v rg >/dev/null 2>&1; then
    fail "ripgrep (rg) is required for readiness checks"
  fi

  require_file "Project.toml"
  if [ -f "README.adoc" ] || [ -f "README.md" ]; then
    pass "README present"
  else
    fail "README missing"
  fi

  check_project_version
  run_julia_smoke

  if [ "$failures" -gt 0 ]; then
    printf '[readiness] completed with %s failure(s)\n' "$failures"
    exit 1
  fi

  printf '[readiness] all mandatory checks passed\n'
}

main "$@"
