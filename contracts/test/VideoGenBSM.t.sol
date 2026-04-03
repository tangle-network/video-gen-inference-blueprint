// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Test } from "forge-std/Test.sol";
import { VideoGenBSM } from "../src/VideoGenBSM.sol";
import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

contract VideoGenBSMTest is Test {
    VideoGenBSM public bsm;

    address public tangleCore = address(0xC0DE);
    address public owner = address(0xBEEF);
    address public operator1 = address(0x1111);
    address public operator2 = address(0x2222);
    address public user = address(0x3333);

    function setUp() public {
        bsm = new VideoGenBSM();
        bsm.onBlueprintCreated(1, owner, tangleCore);

        vm.prank(owner);
        bsm.configureModel(
            "hunyuan-video",
            uint64(100_000), // price per second (0.10 USD)
            uint32(48_000),  // min VRAM
            uint32(30)       // max 30 seconds
        );
    }

    // --- Initialization ---

    function test_initialization() public view {
        assertEq(bsm.blueprintId(), 1);
        assertEq(bsm.blueprintOwner(), owner);
        assertEq(bsm.tangleCore(), tangleCore);
    }

    function test_cannotReinitialize() public {
        vm.expectRevert(BlueprintServiceManagerBase.AlreadyInitialized.selector);
        bsm.onBlueprintCreated(2, owner, tangleCore);
    }

    // --- Model Configuration ---

    function test_configureModel() public view {
        (uint64 price, uint32 minVram, uint32 maxDur, bool enabled) =
            bsm.modelConfigs(keccak256("hunyuan-video"));
        assertEq(price, 100_000);
        assertEq(minVram, 48_000);
        assertEq(maxDur, 30);
        assertTrue(enabled);
    }

    function test_configureModel_onlyOwner() public {
        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyBlueprintOwnerAllowed.selector, user, owner)
        );
        bsm.configureModel("test-model", 100, 8000, 10);
    }

    // --- Operator Registration ---

    function test_registerOperator() public {
        _registerOperator(operator1);

        assertTrue(bsm.isOperatorActive(operator1));

        address[] memory ops = bsm.getOperators();
        assertEq(ops.length, 1);
        assertEq(ops[0], operator1);
    }

    function test_registerOperator_unsupportedModel() public {
        string[] memory resolutions = new string[](1);
        resolutions[0] = "1280x720";

        bytes memory regData = abi.encode(
            "nonexistent-model",
            uint32(2),
            uint32(80_000),
            "NVIDIA A100",
            uint32(10),
            resolutions,
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        vm.expectRevert("VideoGenBSM: model not supported");
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_insufficientVram() public {
        string[] memory resolutions = new string[](1);
        resolutions[0] = "1280x720";

        bytes memory regData = abi.encode(
            "hunyuan-video",
            uint32(1),
            uint32(24_000), // below 48_000 minimum
            "NVIDIA RTX 4090",
            uint32(10),
            resolutions,
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        vm.expectRevert("VideoGenBSM: insufficient VRAM");
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_durationExceedsLimit() public {
        string[] memory resolutions = new string[](1);
        resolutions[0] = "1280x720";

        bytes memory regData = abi.encode(
            "hunyuan-video",
            uint32(2),
            uint32(80_000),
            "NVIDIA A100",
            uint32(60), // exceeds 30 second limit
            resolutions,
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        vm.expectRevert("VideoGenBSM: max duration exceeds model limit");
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_onlyTangle() public {
        string[] memory resolutions = new string[](1);
        resolutions[0] = "1280x720";

        bytes memory regData = abi.encode(
            "hunyuan-video",
            uint32(2),
            uint32(80_000),
            "NVIDIA A100",
            uint32(10),
            resolutions,
            "https://op1.example.com"
        );

        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyTangleAllowed.selector, user, tangleCore)
        );
        bsm.onRegister(operator1, regData);
    }

    // --- Pricing ---

    function test_getOperatorPricing() public {
        _registerOperator(operator1);

        (uint64 pricePerSecond, string memory endpoint) = bsm.getOperatorPricing(operator1);
        assertEq(pricePerSecond, 100_000);
        assertEq(keccak256(bytes(endpoint)), keccak256(bytes("https://op1.example.com")));
    }

    function test_calculateCost() public {
        _registerOperator(operator1);

        uint64 cost = bsm.calculateCost(operator1, 10);
        assertEq(cost, 1_000_000); // 100_000 * 10 seconds
    }

    // --- Duration Validation ---

    function test_validateRequest() public {
        _registerOperator(operator1);

        bool valid = bsm.validateRequest(operator1, 10, "1280x720");
        assertTrue(valid);
    }

    function test_validateRequest_exceedsDuration() public {
        _registerOperator(operator1);

        vm.expectRevert("VideoGenBSM: duration exceeds max");
        bsm.validateRequest(operator1, 60, "1280x720");
    }

    function test_validateRequest_unsupportedResolution() public {
        _registerOperator(operator1);

        vm.expectRevert("VideoGenBSM: unsupported resolution");
        bsm.validateRequest(operator1, 5, "4096x2160");
    }

    // --- Unregistration ---

    function test_unregisterOperator() public {
        _registerOperator(operator1);

        vm.prank(tangleCore);
        bsm.onUnregister(operator1);

        assertFalse(bsm.isOperatorActive(operator1));
    }

    // --- Active Operators Query ---

    function test_getActiveOperators() public {
        _registerOperator(operator1);
        _registerOperator(operator2);

        address[] memory active = bsm.getActiveOperators("hunyuan-video");
        assertEq(active.length, 2);
    }

    // --- Helpers ---

    function _registerOperator(address op) internal {
        string[] memory resolutions = new string[](2);
        resolutions[0] = "1280x720";
        resolutions[1] = "1920x1080";

        bytes memory regData = abi.encode(
            "hunyuan-video",
            uint32(2),
            uint32(80_000),
            "NVIDIA A100-SXM4-80GB",
            uint32(20),
            resolutions,
            "https://op1.example.com"
        );
        vm.prank(tangleCore);
        bsm.onRegister(op, regData);
    }
}
