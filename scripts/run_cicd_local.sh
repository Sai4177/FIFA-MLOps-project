#!/usr/bin/env bash
# EXPERIMENT 5 - replay every CI/CD practical (P1..P7) LOCALLY, including the failure demos.
# The exact same commands appear in .github/workflows/*.yml.  Usage: bash scripts/run_cicd_local.sh
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export PYTHONWARNINGS=ignore
pass=0; fail=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
expect() {   # expect <ok|fail> <description> <command...>
  local want="$1" desc="$2"; shift 2
  if "$@" >/tmp/cicd_step.log 2>&1; then got=ok; else got=fail; fi
  tail -n 6 /tmp/cicd_step.log | sed 's/^/    /'
  if [ "$want" = "$got" ]; then echo "  [ OK ] $desc (expected $want, got $got)"; pass=$((pass+1));
  else echo "  [BAD ] $desc (expected $want, got $got)"; fail=$((fail+1)); fi
}
cleanup() { rm -rf release_candidate fifa_tier_model.pkl metrics.json fifa_dataset.csv .cicd_backup; }
trap cleanup EXIT
cleanup

step "P1  First CI - unit tests of tier_logic.py"
expect ok   "P1 tests pass on correct code"        $PY -m unittest test_tier_logic -v
cp tier_logic.py /tmp/tier_logic.bak
sed -i 's/return "Elite"/return "Developing"/' tier_logic.py
expect fail "P1 tests FAIL after injected bug"      $PY -m unittest test_tier_logic -v
cp /tmp/tier_logic.bak tier_logic.py
expect ok   "P1 tests pass after fix"               $PY -m unittest test_tier_logic -v

step "P2  ML CI - train model + ML tests"
expect ok   "P2 train_model.py"                     $PY train_model.py
expect ok   "P2 test_ml_pipeline"                   $PY -m unittest test_ml_pipeline -v

step "P3  Quality gate"
expect ok   "P3 gate passes (0.85)"                 env MIN_ACCURACY=0.85 $PY quality_gate.py
expect fail "P3 gate FAILS (0.99) -> exit code 1"   env MIN_ACCURACY=0.99 $PY quality_gate.py

step "P4  Preserve artifacts (validated files must exist)"
check_files() { for f in "$@"; do [ -f "$f" ] || { echo "missing: $f"; return 1; }; done; echo "all present: $*"; }
expect ok   "P4 model+metrics+dataset exist"        check_files fifa_tier_model.pkl metrics.json fifa_dataset.csv
expect fail "P4 missing artifact detected"          check_files fifa_tier_model_missing.pkl metrics.json fifa_dataset.csv

step "P5  Continuous delivery - API tests + release candidate"
expect ok   "P5 test_app"                           $PY -m unittest test_app -v
cp test_app.py /tmp/test_app.bak
sed -i 's/"prediction"], "Elite"/"prediction"], "Developing"/' test_app.py
expect fail "P5 API test FAILS (delivery blocked)"  $PY -m unittest test_app -v
cp /tmp/test_app.bak test_app.py
prep_release() { mkdir -p release_candidate/src && cp app.py tier_logic.py fifa_tier_model.pkl metrics.json release_candidate/ \
  && cp requirements-cicd.txt Dockerfile .dockerignore release_candidate/ && cp -r src/fifa_mlops release_candidate/src/ && ls -A release_candidate; }
expect ok   "P5/P6 release candidate prepared"      prep_release

step "P7  Docker CI - container build/run/health/predict"
if docker info >/dev/null 2>&1; then
  expect ok "P7 docker build"                       docker build -t fifa-tier-ml:local .
  docker rm -f fifa-tier-ml-container >/dev/null 2>&1
  expect ok "P7 docker run"                         docker run -d --name fifa-tier-ml-container -p 5000:5000 fifa-tier-ml:local
  RUN_STOP() { docker rm -f fifa-tier-ml-container; }
else
  echo "  (Docker daemon not available here -> running the SAME Flask app from release_candidate/ as a stand-in)"
  ( cd release_candidate && PYTHONPATH=src $PY app.py >/tmp/flask_local.log 2>&1 & echo $! > /tmp/flask.pid )
  RUN_STOP() { kill "$(cat /tmp/flask.pid)" 2>/dev/null; }
fi
wait_ready() { for i in $(seq 1 6); do curl --fail --silent http://localhost:${1}/health && { echo; echo "Application is ready."; return 0; }; echo "Waiting... attempt $i"; sleep 1; done; return 1; }
expect ok   "P7 /health on port 5000"               wait_ready 5000
predict() { r=$(curl --fail --silent --show-error -X POST -H "Content-Type: application/json" \
  -d '{"previous_rank":3,"previous_points":1770.65,"association":"CONMEBOL"}' http://localhost:5000/predict); echo "Prediction response: $r"; echo "$r" | grep -q '"Elite"'; }
expect ok   "P7 /predict returns Elite"             predict
expect fail "P7 wrong health port 5001 detected"    wait_ready 5001
RUN_STOP >/dev/null 2>&1

printf '\nSUMMARY: %d checks as expected, %d unexpected\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
