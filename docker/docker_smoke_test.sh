#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-cellexlink:0.1.0}"

docker run --rm "${IMAGE}" cellexlink --help >/dev/null

docker run --rm "${IMAGE}" python - <<'PY'
import cellexlink
import pyab3p
print("CellExLink Docker smoke test OK")
PY
