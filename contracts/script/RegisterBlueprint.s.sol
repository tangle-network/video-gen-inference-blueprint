// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Script, console2 } from "forge-std/Script.sol";
import { Types } from "tnt-core/libraries/Types.sol";
import { VideoGenBSM } from "../src/VideoGenBSM.sol";

/// @notice Minimal interface for Tangle blueprint registration.
interface ITangle {
    function createBlueprint(Types.BlueprintDefinition calldata def) external returns (uint64);
}

/// @title RegisterBlueprint
/// @notice Deploys VideoGenBSM and registers the video-gen blueprint on Tangle
///         in a single broadcast.
/// @dev    VideoGenBSM is a regular (non-upgradeable) BlueprintServiceManagerBase
///         contract with no constructor args, so it is deployed directly
///         without an ERC1967Proxy. Mirrors the proven sibling pattern shipping
///         across the Tangle blueprint repos.
///         Run via: `forge script contracts/script/RegisterBlueprint.s.sol
///         --rpc-url $RPC_URL --broadcast --slow`
contract RegisterBlueprint is Script {
    // ─────────────────────────────────────────────────────────────────────────
    // Defaults — overridable via env vars for non-anvil chains.
    // ─────────────────────────────────────────────────────────────────────────

    // Anvil well-known deployer key (default when no PRIVATE_KEY env is set).
    uint256 constant DEFAULT_DEPLOYER_KEY =
        0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;

    // Tangle protocol address on a LocalTestnet anvil snapshot. For real
    // chains (Base Sepolia, mainnet) pass TANGLE_CORE via env.
    address constant DEFAULT_TANGLE = 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9;

    // USDC on Base Sepolia. Per-second-of-video billing settles in this token.
    // For other networks pass PAYMENT_TOKEN via env. The address is captured
    // here purely for visibility in deployment logs — VideoGenBSM does not
    // consume it at construction time. The blueprint owner configures pricing
    // post-deploy via `VideoGenBSM.configureModel`.
    address constant DEFAULT_PAYMENT_TOKEN = 0x036CbD53842c5426634e7929541eC2318f3dCF7e;

    function run() external {
        uint256 deployerKey = vm.envOr("PRIVATE_KEY", DEFAULT_DEPLOYER_KEY);
        address tangleAddr = vm.envOr("TANGLE_CORE", DEFAULT_TANGLE);
        address paymentToken = vm.envOr("PAYMENT_TOKEN", DEFAULT_PAYMENT_TOKEN);

        ITangle tangle = ITangle(tangleAddr);

        vm.startBroadcast(deployerKey);

        // ── Deploy VideoGenBSM (non-upgradeable, no constructor args) ───────
        VideoGenBSM bsm = new VideoGenBSM();

        // ── Register on Tangle ──────────────────────────────────────────────
        uint64 blueprintId = tangle.createBlueprint(_buildDefinition(address(bsm)));

        vm.stopBroadcast();

        // ── Output for bash wrapper parsing ─────────────────────────────────
        console2.log("DEPLOY_VIDEO_GEN_BSM=%s", vm.toString(address(bsm)));
        console2.log("DEPLOY_VIDEO_GEN_BLUEPRINT_ID=%s", vm.toString(blueprintId));
        console2.log("DEPLOY_VIDEO_GEN_PAYMENT_TOKEN=%s", vm.toString(paymentToken));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Blueprint Definition builder
    //
    // Values mined from deploy/definition.json (cargo-tangle format). On-chain
    // representation flattens metadata + jobs into the Types.BlueprintDefinition
    // struct expected by Tangle.createBlueprint.
    // ═════════════════════════════════════════════════════════════════════════

    function _buildDefinition(address manager) internal pure returns (Types.BlueprintDefinition memory def) {
        def.metadataUri = "https://github.com/tangle-network/video-gen-inference-blueprint";
        // metadataHash is a digest of the canonical metadata JSON. Until that
        // payload is pinned via IPFS, derive it from the metadataUri so the
        // value is deterministic + traceable.
        def.metadataHash = keccak256(bytes(def.metadataUri));
        def.manager = manager;
        def.masterManagerRevision = 0;
        def.hasConfig = true;

        // Event-driven pricing: operators are paid per second of generated
        // video rather than on a fixed subscription cadence. Dynamic
        // membership lets new GPU operators join after registration. Mirrors
        // definition.json (`membership: Dynamic, min_operators: 1,
        // max_operators: 0`).
        def.config = Types.BlueprintConfig({
            membership: Types.MembershipModel.Dynamic,
            pricing: Types.PricingModel.EventDriven,
            minOperators: 1,
            maxOperators: 0, // unbounded
            subscriptionRate: 0,
            subscriptionInterval: 0,
            eventRate: 0 // operators set price-per-second via configureModel
         });

        def.metadata = Types.BlueprintMetadata({
            name: "Video Generation Blueprint",
            description: "Video and image generation operator via ComfyUI workflows on Tangle",
            author: "Tangle Network",
            category: "AI/Inference",
            codeRepository: "https://github.com/tangle-network/video-gen-inference-blueprint",
            logo: "",
            website: "https://tangle.tools",
            license: "MIT OR Apache-2.0",
            profilingData: "{\"execution_profile\":{\"gpu\":{\"policy\":\"required\",\"min_count\":1,\"min_vram_gb\":24}}}"
        });

        def.jobs = _buildJobs();

        def.registrationSchema = "";
        def.requestSchema = "";

        def.sources = new Types.BlueprintSource[](1);
        Types.BlueprintBinary[] memory bins = new Types.BlueprintBinary[](1);
        bins[0] = Types.BlueprintBinary({
            arch: Types.BlueprintArchitecture.Amd64,
            os: Types.BlueprintOperatingSystem.Linux,
            name: "video-gen-operator",
            // Placeholder digest mirrors definition.json (all-zeros). Operator
            // binaries are pinned via the GitHub release at runtime, so the
            // sha here is informational until the release is published.
            sha256: bytes32(uint256(0xdeadbeef))
        });
        def.sources[0] = Types.BlueprintSource({
            kind: Types.BlueprintSourceKind.Native,
            container: Types.ImageRegistrySource("", "", ""),
            wasm: Types.WasmSource(Types.WasmRuntime.Unknown, Types.BlueprintFetcherKind.None, "", ""),
            native: Types.NativeSource(
                Types.BlueprintFetcherKind.None,
                "file:///target/release/video-gen-operator",
                "./video-gen-operator"
            ),
            testing: Types.TestingSource("video-gen-inference", "video-gen-operator", "."),
            binaries: bins
        });

        // definition.json declares supported_memberships = [Dynamic, Fixed].
        def.supportedMemberships = new Types.MembershipModel[](2);
        def.supportedMemberships[0] = Types.MembershipModel.Dynamic;
        def.supportedMemberships[1] = Types.MembershipModel.Fixed;
    }

    function _buildJobs() internal pure returns (Types.JobDefinition[] memory jobs) {
        jobs = new Types.JobDefinition[](1);
        // Job 0: generate_video
        //   inputs:  (string prompt, string model, uint32 durationSecs,
        //            uint32 width, uint32 height, uint32 fps)
        //   outputs: (bytes videoData, string mimeType, uint32 durationSecs,
        //            uint32 width, uint32 height)
        // The Rust operator enforces the ABI; on-chain schemas stay empty here
        // to match the pattern used across sibling inference blueprints.
        jobs[0] = Types.JobDefinition({
            name: "generate_video",
            description: "Generate a video clip from a prompt via the operator's diffusion/ComfyUI backend",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });
    }
}
