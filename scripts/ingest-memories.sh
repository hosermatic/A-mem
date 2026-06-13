#!/usr/bin/env bash
# ingest-memories.sh — Seed A-MEM with all tiered knowledge files
# Usage: bash ingest-memories.sh [--tier1-only] [--dry-run] [--reset-manifest]
set -euo pipefail

AMEM_URL="http://localhost:8101"
MANIFEST="$HOME/.agentic_memory/.ingest-manifest"
TIER1_ONLY=false
DRY_RUN=false
RESET=false

for arg in "$@"; do
  case $arg in
    --tier1-only)    TIER1_ONLY=true ;;
    --dry-run)       DRY_RUN=true ;;
    --reset-manifest) RESET=true ;;
  esac
done

# ── Health check ──────────────────────────────────────────────────────────────
curl -sf "$AMEM_URL/health" > /dev/null 2>&1 || {
  echo "⚠  A-MEM not running on $AMEM_URL"
  echo "   Start: launchctl load ~/Library/LaunchAgents/com.hodtools.a-mem.plist"
  exit 1
}

mkdir -p "$(dirname "$MANIFEST")"
[[ $RESET == true ]] && { rm -f "$MANIFEST"; echo "Manifest reset."; }
touch "$MANIFEST"

NEW=0; SKIP=0; FAIL=0

# ── Post one file to A-MEM ────────────────────────────────────────────────────
ingest_file() {
  local FILE="$1" CATEGORY="$2" TAGS="$3"

  # Skip missing or too-large (> 100KB) files
  [[ ! -f "$FILE" ]] && return
  local SIZE
  SIZE=$(wc -c < "$FILE" 2>/dev/null || echo 0)
  [[ $SIZE -gt 102400 ]] && { SKIP=$((SKIP+1)); return; }
  [[ $SIZE -lt 10 ]] && { SKIP=$((SKIP+1)); return; }

  # Idempotency: md5 hash
  local HASH
  HASH=$(md5 -q "$FILE" 2>/dev/null || md5sum "$FILE" | awk '{print $1}')
  local KEY="$FILE|$HASH"
  if grep -qF "$KEY" "$MANIFEST" 2>/dev/null; then
    SKIP=$((SKIP+1)); return
  fi

  if [[ $DRY_RUN == true ]]; then
    echo "  [dry] $(basename "$FILE") → $CATEGORY"
    NEW=$((NEW+1)); return
  fi

  # POST via Python for safe JSON encoding
  local EXIT_CODE=0
  python3 - "$FILE" "$CATEGORY" "$TAGS" "$AMEM_URL" <<'PYEOF' || EXIT_CODE=$?
import json, sys, urllib.request

file_path, category, tags_str, base_url = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(file_path, 'r', errors='replace') as f:
    content = f.read()[:8000]  # cap at 8KB per memory

payload = json.dumps({
    'content': content,
    'category': category,
    'tags': json.loads(tags_str)
}).encode()

req = urllib.request.Request(
    f'{base_url}/memory/add',
    data=payload,
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        result = json.loads(r.read())
        action = result.get('action', result.get('status', 'ok'))
        print(f"  {action}: {file_path.split('/')[-1]}")
except Exception as e:
    print(f"  FAIL: {file_path.split('/')[-1]} — {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

  if [[ $EXIT_CODE -ne 0 ]]; then
    FAIL=$((FAIL+1)); return
  fi

  echo "$KEY" >> "$MANIFEST"
  NEW=$((NEW+1))
}

# ── Ingest a directory glob ───────────────────────────────────────────────────
ingest_dir() {
  local DIR="$1" PATTERN="$2" CATEGORY="$3" TAGS="$4"
  [[ ! -d "$DIR" ]] && return
  while IFS= read -r -d '' FILE; do
    ingest_file "$FILE" "$CATEGORY" "$TAGS"
  done < <(find "$DIR" -maxdepth 1 -name "$PATTERN" -not -name "MEMORY.md" -print0 2>/dev/null | sort -z)
}

# ── Ingest latest checkpoint per project ─────────────────────────────────────
ingest_latest_checkpoints() {
  local DIR="$1"
  [[ ! -d "$DIR" ]] && return
  # Use a temp file to track seen prefixes (bash 3.2 compatible — no declare -A)
  local SEEN_FILE
  SEEN_FILE=$(mktemp /tmp/amem-seen-XXXXXX)
  find "$DIR" -maxdepth 1 -name "*-CHECKPOINT.md" -print0 2>/dev/null \
    | sort -rz | tr '\0' '\n' \
    | while IFS= read -r FILE; do
        local BASE PREFIX
        BASE=$(basename "$FILE")
        PREFIX=$(echo "$BASE" | sed 's/-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].*//')
        if ! grep -qF "$PREFIX" "$SEEN_FILE" 2>/dev/null; then
          echo "$PREFIX" >> "$SEEN_FILE"
          ingest_file "$FILE" "checkpoint" '["checkpoint","auto-ingest"]'
        fi
      done
  rm -f "$SEEN_FILE"
}

echo "=== A-MEM Ingest — $(date) ==="
echo "    Tier1-only: $TIER1_ONLY | Dry-run: $DRY_RUN"
echo ""

# ── TIER 1: Highest signal (always ingest) ───────────────────────────────────
MEMDIR="$HOME/.claude/projects/-Users-jose-Developer/memory"
echo "▶ Tier 1: memory files ($MEMDIR)"
ingest_dir "$MEMDIR" "feedback_*.md"  "behavioral-rule"    '["memory-file","feedback"]'
ingest_dir "$MEMDIR" "lessons_*.md"   "lesson"             '["memory-file","lesson"]'
ingest_dir "$MEMDIR" "project_*.md"   "project-state"      '["memory-file","project"]'
ingest_dir "$MEMDIR" "reference_*.md" "reference"          '["memory-file","reference"]'

echo "▶ Tier 1: key-facts"
ingest_dir "$HOME/Developer/shared/orchestration/key-facts" "*.md" "key-facts" '["key-facts"]'

echo "▶ Tier 1: behavioral modules"
ingest_dir "$HOME/.claude/modules" "*.md" "behavioral-module" '["module"]'

if [[ $TIER1_ONLY == false ]]; then
  echo ""
  echo "▶ Tier 2: plans"
  ingest_dir "$HOME/.claude/plans" "*.md" "plan" '["plan"]'

  echo "▶ Tier 2: latest checkpoints"
  ingest_latest_checkpoints "$HOME/Developer/shared/orchestration/checkpoints"

  echo "▶ Tier 2: production audit findings"
  PRODDIR="$HOME/Developer/shared/orchestration/knowledge/production"
  if [[ -d "$PRODDIR" ]]; then
    while IFS= read -r -d '' FILE; do
      ingest_file "$FILE" "audit-finding" '["audit","production"]'
    done < <(find "$PRODDIR" -name "*.md" -print0 2>/dev/null | sort -z)
  fi
fi

echo ""
echo "=== Summary ==="
echo "  Ingested: $NEW new"
echo "  Skipped:  $SKIP unchanged"
echo "  Failed:   $FAIL errors"
TOTAL=$(curl -sf "$AMEM_URL/health" | python3 -c "import json,sys; print(json.load(sys.stdin)['memory_count'])" 2>/dev/null || echo "?")
echo "  A-MEM total: $TOTAL memories"
