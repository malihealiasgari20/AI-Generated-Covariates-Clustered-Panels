#!/bin/bash
# Production runs of the R1 modules, in order. Logs in out/run_log.txt.
cd "$(dirname "$0")/code"
mkdir -p ../out
for m in A1 C D1 D2 B A2; do
  python3 -I -B r1_run.py $m --workers 2 >> ../out/run_log.txt 2>&1 || echo "MODULE $m FAILED" >> ../out/run_log.txt
done
echo "RUN COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> ../out/run_log.txt
