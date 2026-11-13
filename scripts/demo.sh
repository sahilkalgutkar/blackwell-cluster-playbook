#!/usr/bin/env bash
# Run the whole pipeline on the synthetic fixtures: topology -> validate -> report.
# Healthy fixtures must pass and the degraded one must fail, or this script exits non-zero.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=build/demo
rm -rf "${OUT}"
mkdir -p "${OUT}"

for profile in hgx-h100 dgx-b200 gb200-nvl72 dgx-b200-degraded; do
  echo "=== ${profile}"
  bcp topology --replay "fixtures/${profile}" --format mermaid > "${OUT}/${profile}.mmd"
  status=0
  bcp validate --replay "fixtures/${profile}" --out "${OUT}" > "${OUT}/${profile}.txt" 2>&1 || status=$?
  expected=0
  [[ "${profile}" == *degraded* ]] && expected=1
  if [[ "${status}" -ne "${expected}" ]]; then
    cat "${OUT}/${profile}.txt"
    echo "FAILED: ${profile} exited ${status}, expected ${expected}" >&2
    exit 1
  fi
  grep -m1 "^Verdict" "${OUT}/${profile}.txt"
done

bcp report "${OUT}"/*.json --out "${OUT}/report.md"
echo
cat "${OUT}/report.md"
echo "Demo OK. Outputs in ${OUT}/"
