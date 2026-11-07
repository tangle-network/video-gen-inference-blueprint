// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Test } from "forge-std/Test.sol";
import { Deploy } from "../script/Deploy.s.sol";
import { VideoGenBSM } from "../src/VideoGenBSM.sol";

/// The deployed manager accepts the example operator's own registration payload.
contract DeployTest is Test {
    function test_deployConfiguresTheOperatorsModel() public {
        vm.setEnv("TANGLE_CORE", vm.toString(address(this)));
        VideoGenBSM manager = new Deploy().run();
        manager.onBlueprintCreated(1, manager.blueprintOwner(), address(this));

        string memory config = vm.readFile("../deploy/config.example.json");
        string memory model = vm.parseJsonString(config, ".video.model");
        manager.onRegister(
            address(0x1111),
            abi.encode(
                model,
                uint32(vm.parseJsonUint(config, ".gpu.expected_gpu_count")),
                uint32(vm.parseJsonUint(config, ".gpu.min_vram_mib")),
                "unknown",
                uint32(vm.parseJsonUint(config, ".video.max_duration_secs")),
                new string[](0),
                "http://0.0.0.0:8080"
            )
        );
        assertTrue(manager.isOperatorActive(address(0x1111)));
        (uint64 price,,, bool enabled) = manager.modelConfigs(keccak256(bytes(model)));
        assertEq(price, vm.parseJsonUint(config, ".video.price_per_compute_second"));
        assertTrue(enabled);
    }
}
