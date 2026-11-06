#!/usr/bin/env bash
# Register the video-gen-inference blueprint on Tangle.
# Release definitions and local Anvil inputs are documented in README.md.

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"
TANGLE_CORE="${TANGLE_CORE:-0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9}"

echo "=== Video-Gen Inference Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Tangle Core: $TANGLE_CORE"
echo "Payment:     ${PAYMENT_TOKEN:-<default USDC sepolia>}"
echo ""

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CORE_DIR="$(cargo metadata --manifest-path "$ROOT_DIR/Cargo.toml" --format-version 1 --locked | python3 -c 'import json,sys,pathlib; print(pathlib.Path(next(p["manifest_path"] for p in json.load(sys.stdin)["packages"] if p["name"] == "tangle-inference-core")).parent)')"
BLUEPRINT_DEFINITION="$(node "$CORE_DIR/scripts/register-blueprint.mjs" "$ROOT_DIR/deploy/definition.json" "$ROOT_DIR")"
export BLUEPRINT_DEFINITION
trap 'rm -f -- "$BLUEPRINT_DEFINITION"' EXIT
cd "$(dirname "$0")/../contracts"

# Deploy BSM AND register the blueprint in one forge-script broadcast.
DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" \
    TANGLE_CORE="${TANGLE_CORE:-}" \
    PAYMENT_TOKEN="${PAYMENT_TOKEN:-}" \
    forge script script/Deploy.s.sol:Deploy \
        --remappings "tangle-inference-core/=$CORE_DIR/contracts/" \
        --rpc-url "$RPC_URL" \
        --broadcast --slow)

echo "$DEPLOY_OUTPUT"

# Extract BSM address + blueprint ID for downstream scripts.
BSM_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_VIDEO_GEN_BSM=0x[0-9a-fA-F]+' | tail -1 | cut -d= -f2)
BLUEPRINT_ID=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_VIDEO_GEN_BLUEPRINT_ID=[0-9]+' | tail -1 | cut -d= -f2)

if [ -z "$BSM_ADDRESS" ] || [ -z "$BLUEPRINT_ID" ]; then
    echo "ERROR: failed to extract addresses from forge output"
    exit 1
fi

echo ""
echo "=== Blueprint registered ==="
echo "Blueprint ID: $BLUEPRINT_ID"
echo "VideoGenBSM:  $BSM_ADDRESS"
echo ""
echo "Next step (blueprint owner): configure a supported model on the BSM."
echo "  cast send $BSM_ADDRESS \\"
echo "    'configureModel(string,uint64,uint32,uint32)' \\"
echo "    'hunyuan-video' <pricePerSecond> <minGpuVramMib> <maxDurationSecs> \\"
echo "    --rpc-url \"$RPC_URL\" --private-key \"\$OWNER_KEY\""
