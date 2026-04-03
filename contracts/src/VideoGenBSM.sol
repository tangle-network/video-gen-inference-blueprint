// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @title VideoGenBSM -- Blueprint Service Manager for Video Generation
/// @author Tangle Network
/// @notice Manages operator registration, capability validation, and per-second-of-video
///         pricing for the video generation blueprint.
///
/// @dev Pricing model:
///      Unlike LLM inference (per-token), video generation is priced per-second of output
///      video. Operators declare their price_per_second and clients pre-authorize based on
///      requested duration.
///
/// @dev VRAM requirements:
///      Video generation models require significantly more VRAM than LLMs:
///      - Hunyuan Video: 48GB+ (A100 80GB recommended)
///      - LTX-Video: 24GB+ (A100 40GB minimum)
///      - CogVideoX: 40GB+
///      The contract enforces minimum VRAM thresholds per model class.

import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

contract VideoGenBSM is BlueprintServiceManagerBase {

    // ═══════════════════════════════════════════════════════════════════════
    // TYPES
    // ═══════════════════════════════════════════════════════════════════════

    struct OperatorCapabilities {
        /// Video model identifier (e.g. "hunyuan-video", "ltx-video")
        string model;
        /// Number of GPUs available
        uint32 gpuCount;
        /// Total VRAM across all GPUs in MiB
        uint32 totalVramMib;
        /// GPU model name (e.g. "NVIDIA A100-SXM4-80GB")
        string gpuModel;
        /// Maximum video duration this operator supports (seconds)
        uint32 maxDurationSecs;
        /// Supported output resolutions
        string[] supportedResolutions;
        /// Operator's HTTP endpoint
        string endpoint;
        /// Whether the operator is currently active
        bool active;
    }

    struct ModelConfig {
        /// Price per second of generated video (tsUSD base units, 6 decimals)
        uint64 pricePerSecond;
        /// Minimum GPU VRAM required in MiB
        uint32 minGpuVramMib;
        /// Maximum allowed duration for this model class
        uint32 maxDurationSecs;
        /// Whether this model is enabled for registration
        bool enabled;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════

    /// operator address => capabilities
    mapping(address => OperatorCapabilities) public operatorCaps;

    /// keccak256(model name) => model configuration
    mapping(bytes32 => ModelConfig) public modelConfigs;

    /// Registered operator addresses for enumeration
    address[] internal _operators;

    // ═══════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════

    event OperatorRegistered(
        address indexed operator,
        string model,
        uint32 gpuCount,
        uint32 totalVramMib,
        uint32 maxDurationSecs
    );

    event ModelConfigured(
        string model,
        uint64 pricePerSecond,
        uint32 minGpuVramMib,
        uint32 maxDurationSecs
    );

    // ═══════════════════════════════════════════════════════════════════════
    // ADMIN
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Configure a supported video model class with pricing and VRAM requirements.
    /// @param model Model identifier (e.g. "hunyuan-video")
    /// @param pricePerSecond Price per second of video in tsUSD base units
    /// @param minGpuVramMib Minimum VRAM required in MiB
    /// @param maxDuration Maximum video duration in seconds
    function configureModel(
        string calldata model,
        uint64 pricePerSecond,
        uint32 minGpuVramMib,
        uint32 maxDuration
    ) external onlyBlueprintOwner {
        modelConfigs[keccak256(bytes(model))] = ModelConfig({
            pricePerSecond: pricePerSecond,
            minGpuVramMib: minGpuVramMib,
            maxDurationSecs: maxDuration,
            enabled: true
        });

        emit ModelConfigured(model, pricePerSecond, minGpuVramMib, maxDuration);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // REGISTRATION (BSM lifecycle)
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Called when an operator registers for this blueprint.
    /// @dev Validates GPU capabilities meet minimum requirements for the declared model.
    /// @param operator The operator's address
    /// @param registrationInputs ABI-encoded:
    ///   (string model, uint32 gpuCount, uint32 totalVramMib, string gpuModel,
    ///    uint32 maxDurationSecs, string[] supportedResolutions, string endpoint)
    function onRegister(
        address operator,
        bytes calldata registrationInputs
    ) external payable override onlyFromTangle {
        (
            string memory model,
            uint32 gpuCount,
            uint32 totalVramMib,
            string memory gpuModel,
            uint32 maxDur,
            string[] memory supportedResolutions,
            string memory endpoint
        ) = abi.decode(
            registrationInputs,
            (string, uint32, uint32, string, uint32, string[], string)
        );

        // Validate model is configured and enabled
        bytes32 modelKey = keccak256(bytes(model));
        ModelConfig memory cfg = modelConfigs[modelKey];
        require(cfg.enabled, "VideoGenBSM: model not supported");

        // Validate VRAM meets minimum for this model class
        require(
            totalVramMib >= cfg.minGpuVramMib,
            "VideoGenBSM: insufficient VRAM"
        );

        // Validate max duration does not exceed model limit
        if (cfg.maxDurationSecs > 0) {
            require(
                maxDur <= cfg.maxDurationSecs,
                "VideoGenBSM: max duration exceeds model limit"
            );
        }

        operatorCaps[operator] = OperatorCapabilities({
            model: model,
            gpuCount: gpuCount,
            totalVramMib: totalVramMib,
            gpuModel: gpuModel,
            maxDurationSecs: maxDur,
            supportedResolutions: supportedResolutions,
            endpoint: endpoint,
            active: true
        });

        _operators.push(operator);

        emit OperatorRegistered(operator, model, gpuCount, totalVramMib, maxDur);
    }

    /// @notice Called when an operator unregisters
    function onUnregister(address operator) external override onlyFromTangle {
        operatorCaps[operator].active = false;
    }

    /// @notice Called when an operator updates preferences (e.g. endpoint change)
    function onUpdatePreferences(
        address operator,
        bytes calldata preferences
    ) external payable override onlyFromTangle {
        (string memory newEndpoint) = abi.decode(preferences, (string));
        operatorCaps[operator].endpoint = newEndpoint;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // QUERIES
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Get all registered operator addresses
    function getOperators() external view returns (address[] memory) {
        return _operators;
    }

    /// @notice Check if an operator is active
    function isOperatorActive(address op) external view returns (bool) {
        return operatorCaps[op].active;
    }

    /// @notice Get pricing and endpoint for an operator
    function getOperatorPricing(address op) external view returns (
        uint64 pricePerSecond,
        string memory endpoint
    ) {
        OperatorCapabilities memory c = operatorCaps[op];
        ModelConfig memory cfg = modelConfigs[keccak256(bytes(c.model))];
        return (cfg.pricePerSecond, c.endpoint);
    }

    /// @notice Calculate cost for a video generation request
    /// @param op Operator address
    /// @param durationSecs Requested video duration in seconds
    /// @return cost Total cost in tsUSD base units
    function calculateCost(address op, uint32 durationSecs) external view returns (uint64 cost) {
        ModelConfig memory cfg = modelConfigs[keccak256(bytes(operatorCaps[op].model))];
        return cfg.pricePerSecond * durationSecs;
    }

    /// @notice Validate a video request against operator capabilities
    function validateRequest(
        address operator,
        uint32 durationSecs,
        string calldata resolution
    ) external view returns (bool) {
        OperatorCapabilities storage caps = operatorCaps[operator];
        require(caps.active, "VideoGenBSM: operator not active");
        require(durationSecs <= caps.maxDurationSecs, "VideoGenBSM: duration exceeds max");

        // Check resolution is supported
        bool found = false;
        for (uint i = 0; i < caps.supportedResolutions.length; i++) {
            if (keccak256(bytes(caps.supportedResolutions[i])) == keccak256(bytes(resolution))) {
                found = true;
                break;
            }
        }
        require(found, "VideoGenBSM: unsupported resolution");

        return true;
    }

    /// @notice Get active operators serving a specific model
    function getActiveOperators(string calldata model) external view returns (address[] memory) {
        bytes32 modelKey = keccak256(bytes(model));
        uint count = 0;
        for (uint i = 0; i < _operators.length; i++) {
            if (
                operatorCaps[_operators[i]].active &&
                keccak256(bytes(operatorCaps[_operators[i]].model)) == modelKey
            ) {
                count++;
            }
        }

        address[] memory result = new address[](count);
        uint idx = 0;
        for (uint i = 0; i < _operators.length; i++) {
            if (
                operatorCaps[_operators[i]].active &&
                keccak256(bytes(operatorCaps[_operators[i]].model)) == modelKey
            ) {
                result[idx++] = _operators[i];
            }
        }
        return result;
    }
}
