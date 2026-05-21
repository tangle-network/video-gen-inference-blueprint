#!/usr/bin/env bash
# Register the video-gen-inference blueprint on Tangle.
#
# Single-shot flow: deploys VideoGenBSM (regular, non-upgradeable) AND calls
# Tangle.createBlueprint in the same broadcast via
# `contracts/script/RegisterBlueprint.s.sol`.
#
# Replaces the prior cargo-tangle CLI flow — `cargo tangle blueprint deploy
# tangle` repeatedly failed against the deployed Base Sepolia Tangle proxy due
# to cargo-tangle's settings-file + enum-case requirements. The forge-script
# pattern below is the same one proven by sibling blueprints (image-gen, llm
# inference, ai-agent-sandbox, etc.).
#
# Prerequisites:
#   - forge (Foundry) installed
#   - Deployer wallet funded on the target network
#
# Usage (Base Sepolia, against the already-deployed Tangle protocol):
#
#   export PRIVATE_KEY=0x...
#   export RPC_URL=https://sepolia.base.org
#   export TANGLE_CORE=0xC9b0716a187072be0f38A5D972392C6479b9Cfe3
#   export PAYMENT_TOKEN=0x036CbD53842c5426634e7929541eC2318f3dCF7e  # USDC sepolia
#   ./deploy/register-blueprint.sh
#
# Local anvil (LocalTestnet snapshot):
#
#   export RPC_URL=http://127.0.0.1:8545
#   ./deploy/register-blueprint.sh   # uses anvil deployer key + Tangle/USDC defaults
#
# Outputs (parsed by downstream tooling, do not change without coordinating):
#   DEPLOY_VIDEO_GEN_BSM=<address>
#   DEPLOY_VIDEO_GEN_BLUEPRINT_ID=<u64>
#   DEPLOY_VIDEO_GEN_PAYMENT_TOKEN=<address>

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"

echo "=== Video-Gen Inference Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Deployer:    $(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Tangle Core: ${TANGLE_CORE:-<default from RegisterBlueprint.s.sol>}"
echo "Payment:     ${PAYMENT_TOKEN:-<default USDC sepolia>}"
echo ""

cd "$(dirname "$0")/../contracts"

# Deploy BSM AND register the blueprint in one forge-script broadcast.
DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" \
    TANGLE_CORE="${TANGLE_CORE:-}" \
    PAYMENT_TOKEN="${PAYMENT_TOKEN:-}" \
    forge script script/RegisterBlueprint.s.sol \
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
