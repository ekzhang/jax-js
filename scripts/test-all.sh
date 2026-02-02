#!/bin/bash
# Unified test runner that uses Deno for WebGPU tests when Chromium lacks WebGPU support.
#
# Vitest runs in headless Chromium which typically doesn't have WebGPU.
# On systems with a display, headed Chromium may have WebGPU via GPU process.
# Deno has native WebGPU support (wgpu-rs) that works headlessly.
#
# This script:
# 1. Runs Vitest (browser tests in headless Chromium)
# 2. If WebGPU tests were skipped, runs Deno WebGPU tests as supplement

set -e

cd "$(dirname "$0")/.."

echo "=== Running Vitest (headless Chromium) ==="
# Run vitest and capture output to check for WebGPU skip count
VITEST_OUTPUT=$(pnpm test run 2>&1) || {
    echo "$VITEST_OUTPUT"
    echo "❌ Vitest failed"
    exit 1
}
echo "$VITEST_OUTPUT"

# Check if any tests were skipped (indicates WebGPU wasn't available)
SKIPPED_COUNT=$(echo "$VITEST_OUTPUT" | grep -oP '\d+(?= skipped)' | tail -1 || echo "0")

if [ "$SKIPPED_COUNT" -gt "0" ]; then
    echo ""
    echo "=== $SKIPPED_COUNT tests skipped (likely WebGPU) ==="
    echo "=== Running Deno WebGPU tests as supplement ==="
    echo ""
    
    # Check if Deno is available
    if command -v "$HOME/.deno/bin/deno" &> /dev/null || command -v deno &> /dev/null; then
        pnpm run test:deno || {
            echo "❌ Deno WebGPU tests failed"
            exit 1
        }
        echo ""
        echo "✅ All tests passed (Vitest + Deno WebGPU)"
    else
        echo "⚠️  Deno not found, skipping WebGPU tests"
        echo "   Install Deno: curl -fsSL https://deno.land/install.sh | sh"
        echo ""
        echo "✅ Vitest tests passed (WebGPU tests skipped)"
    fi
else
    echo ""
    echo "✅ All tests passed (WebGPU available in Chromium)"
fi
