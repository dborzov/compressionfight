#!/usr/bin/env bash
# Build docs/map into map/ with Hugo. Usage: ./build.sh [--local]
#   --local  baseURL http://127.0.0.1:8080/map/ (serve repo root with: python3 -m http.server 8080)
set -euo pipefail
cd "$(dirname "$0")"
BASE="https://www.borzov.ca/compressionfight/map/"
[ "${1:-}" = "--local" ] && BASE="http://127.0.0.1:8080/map/"
if command -v hugo >/dev/null 2>&1; then HUGO=(hugo); else HUGO=(mise exec -- hugo); fi
python3 site/tools/gen_map_data.py
[ -f site/assets/css/syntax.css ] || "${HUGO[@]}" gen chromastyles --style=dracula > site/assets/css/syntax.css
rm -rf map
"${HUGO[@]}" --source site --baseURL "$BASE" --quiet --gc
echo "built $(find map -name index.html | wc -l) pages into map/ (baseURL $BASE)"
