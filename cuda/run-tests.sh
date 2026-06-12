#!/bin/bash
# Test suite for the moonshine-cuda encoder. Run from cuda/ after building.
# Usage: ./run-tests.sh [build_dir]   (default: build)
#
# Gates, in order:
#   1. golden transcripts   - CUDA output matches known-good text
#   2. CPU reference        - CUDA == CPU fp32 reference (beckett, ~8s)
#   3. fused vs unfused     - byte-identical on one-shot (long/multi-tile
#                             windows) and streaming bench (short windows)
#   4. perf report          - cycle profile + ms/call vs recorded reference
#                             (informational: requires pinned clocks to
#                             compare across sessions -- sudo jetson_clocks)
set -u
cd "$(dirname "$0")"
B=${1:-build}
M=models/streaming-tiny
TA=../test-assets
FAIL=0

check() { if [ "$2" == "$3" ]; then echo "PASS $1"; else echo "FAIL $1"; echo "  got:      $2"; echo "  expected: $3"; FAIL=1; fi; }

# 1. Golden transcripts (greedy decode is deterministic)
check "golden beckett" \
  "$($B/moonshine-cuda $M $TA/beckett.wav 2>/dev/null)" \
  "Ever tried, ever failed. No matter, try again, fail again, fail better."
check "golden intent" \
  "$($B/moonshine-cuda $M $TA/intent.wav 2>/dev/null)" \
  "Hi, I'm Pete. Can you go forward please? Rotate left. Eat a ham sandwich. Move in reverse."

# 2. CPU fp32 reference agreement (ground truth for the math)
check "cpu-vs-cuda beckett" \
  "$($B/moonshine-cuda $M $TA/beckett.wav 2>/dev/null)" \
  "$($B/moonshine-ref $M $TA/beckett.wav 2>/dev/null)"

# 3. Fused vs unfused, one-shot (exercises multi-tile long windows)
for w in beckett intent two_cities_16k; do
  check "fused-vs-unfused $w" \
    "$($B/moonshine-cuda $M $TA/$w.wav 2>/dev/null)" \
    "$(MOONSHINE_NO_FUSED=1 $B/moonshine-cuda $M $TA/$w.wav 2>/dev/null)"
done
# ... and streaming (steady-state + remainder windows); strip latency fields
S1=$($B/moonshine-bench $M $TA/two_cities_16k.wav 2>/dev/null | grep '^\[' | sed 's/([0-9.]*ms)//')
S2=$(MOONSHINE_NO_FUSED=1 $B/moonshine-bench $M $TA/two_cities_16k.wav 2>/dev/null | grep '^\[' | sed 's/([0-9.]*ms)//')
check "fused-vs-unfused streaming" "$S1" "$S2"

# 4. Performance report (informational; reference: 3.92 ms/call fused,
#    kernel ~3240 cycle-us at len=137, measured 2026-06-12 @ pinned 1020MHz)
echo "--- perf (reference: fused 3.92 ms/call; pin clocks for comparison) ---"
MOONSHINE_ENC_PROF=1 $B/enc-only $M $TA/two_cities_16k.wav 30 2>&1 | tail -2

[ $FAIL -eq 0 ] && echo "ALL GATES PASSED" || echo "FAILURES PRESENT"
exit $FAIL
