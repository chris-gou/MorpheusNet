#!/bin/bash
notify() {
    url="ntfy.sh/piririri"
    content="$(curl -d "$1" $url)"
}

BASELINE_DIR="configs"
LATENCY_DIR="/mnt/truenaas_db/user/christina/morpheus"


run_test() {
    echo "=== Running: $* ==="
    local config="$1"
    local override="$2"
    echo "=== Config: $config ==="
    echo "=== Override: $override ==="
    python test.py --config "$config" --override "$override"
    if [ $? -ne 0 ]; then
        echo "FAILED: $*" | tee -a failed_runs.log
    else
        echo "OK: $*" >> completed_runs.log
    fi
}

BASE="configs/SEDF78.json"

for config in $BASELINE_DIR/ablations/*.json; do
    echo "=== Testing config: $config ==="
    if [[ "$config" == *"PHYSIO"*".json" ]]; then
        run_test "$BASE" "$config" 
    fi
done
notify "=== Done. Failed runs (if any) in failed_runs.log ==="