#!/usr/bin/env bash
# Register the video-gen-inference blueprint on Tangle.
#
# Two-stage flow:
#   1. forge create VideoGenBSM (no constructor args — model/pricing is
#      configured post-deploy via `configureModel`).
#   2. cargo tangle blueprint deploy tangle — registers the blueprint via the
#      definition file at `deploy/definition.json` with the freshly-deployed
#      BSM address patched in as `manager`.
#
# Prerequisites:
#   - forge (Foundry) installed
#   - cargo-tangle CLI installed (`cargo install cargo-tangle`)
#   - jq installed
#   - Deployer wallet funded on the target network
#   - Keystore with the deployer key at ./keystore (or set KEYSTORE_PATH)
#
# Usage (Base Sepolia, against the deployed Tangle protocol):
#
#   export PRIVATE_KEY=0x...
#   export RPC_URL=https://sepolia.base.org
#   export WS_URL=wss://base-sepolia-rpc.publicnode.com
#   export TANGLE_CORE=0xC9b0716a187072be0f38A5D972392C6479b9Cfe3
#   # Payment token is informational here — VideoGenBSM does not bind it at
#   # construction time. Set it so post-deploy `configureModel` callers know
#   # which token the `pricePerSecond` is denominated in (USDC sepolia by
#   # default on Base Sepolia).
#   export PAYMENT_TOKEN=0x036CbD53842c5426634e7929541eC2318f3dCF7e
#   export KEYSTORE_PATH=./keystore
#   ./deploy/register-blueprint.sh
#
# Optional:
#   BSM_ADDRESS — skip the forge create step if the BSM is already deployed
#                 (definition.json gets patched with this address instead).

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"
: "${TANGLE_CORE:?Set TANGLE_CORE}"
: "${WS_URL:?Set WS_URL (ws://… or wss://…)}"
: "${KEYSTORE_PATH:=./keystore}"
: "${PAYMENT_TOKEN:=0x036CbD53842c5426634e7929541eC2318f3dCF7e}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFINITION_FILE="$REPO_ROOT/deploy/definition.json"
CONTRACTS_DIR="$REPO_ROOT/contracts"

echo "=== Video-Gen-Inference Blueprint Registration ==="
echo "Network:        $(cast chain-id --rpc-url "$RPC_URL")"
echo "Deployer:       $(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Tangle Core:    $TANGLE_CORE"
echo "Payment token:  $PAYMENT_TOKEN (used post-deploy by configureModel)"
echo "Definition:     $DEFINITION_FILE"
echo ""

# Stage 1 — Deploy VideoGenBSM if no address was supplied. The BSM extends
# BlueprintServiceManagerBase with no constructor args; the Tangle core
# address is bound the first time the core calls into the BSM (see
# BlueprintServiceManagerBase.onBlueprintCreated).
if [ -z "${BSM_ADDRESS:-}" ]; then
    echo "Stage 1: deploying VideoGenBSM …"
    BSM_ADDRESS=$(forge create \
        --root "$CONTRACTS_DIR" \
        --rpc-url "$RPC_URL" \
        --private-key "$PRIVATE_KEY" \
        --broadcast \
        "$CONTRACTS_DIR/src/VideoGenBSM.sol:VideoGenBSM" \
        --json | jq -r '.deployedTo')
    echo "VideoGenBSM deployed at: $BSM_ADDRESS"
else
    echo "Stage 1 skipped — reusing existing BSM at $BSM_ADDRESS"
fi
echo ""

# Stage 2 — Patch deploy/definition.json with the BSM address and call
# cargo-tangle's canonical deploy flow. The patched file is written to a
# temp path so the in-tree file stays untouched (its `manager: 0x0…0` is
# the template).
PATCHED_DEFINITION=$(mktemp --suffix=-video-gen-blueprint.json)
# cargo-tangle's `--settings-file` defaults to ./settings.env and errors out
# when it isn't present, even though every required value is supplied via
# CLI flags below. Generate an empty placeholder so the loader is happy.
SETTINGS_FILE=$(mktemp --suffix=-video-gen-blueprint.env)
trap 'rm -f "$PATCHED_DEFINITION" "$SETTINGS_FILE"' EXIT
jq --arg mgr "$BSM_ADDRESS" '.manager = $mgr' "$DEFINITION_FILE" > "$PATCHED_DEFINITION"
printf '# auto-generated empty settings file (values come from CLI flags)\n' \
    > "$SETTINGS_FILE"

echo "Stage 2: cargo tangle blueprint deploy tangle …"
cargo tangle blueprint deploy tangle \
    --network testnet \
    --settings-file "$SETTINGS_FILE" \
    --definition "$PATCHED_DEFINITION" \
    --http-rpc-url "$RPC_URL" \
    --ws-rpc-url "$WS_URL" \
    --tangle-contract "$TANGLE_CORE" \
    --keystore-path "$KEYSTORE_PATH"

echo ""
echo "=== Blueprint registered ==="
echo "VideoGenBSM: $BSM_ADDRESS"
echo "(blueprint ID is logged by cargo-tangle above)"
echo ""
echo "Next: configure supported models on the BSM, e.g."
echo "  cast send $BSM_ADDRESS \\"
echo "    'configureModel(string,uint64,uint32,uint32)' \\"
echo "    'hunyuan-video' 50000 49152 60 \\"
echo "    --rpc-url \"$RPC_URL\" --private-key \"\$PRIVATE_KEY\""
