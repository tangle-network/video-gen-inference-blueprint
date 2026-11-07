// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import { Script, console2 } from "forge-std/Script.sol";
import { VideoGenBSM } from "../src/VideoGenBSM.sol";

/// Model-specific manager deployment; cargo-tangle registers the definition.
contract Deploy is Script {
    function run() external returns (VideoGenBSM manager) {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        // The operator config is the one source of the model, price and the VRAM and duration bounds it registers with.
        string memory config = vm.readFile(vm.envOr("MODEL_CONFIG", string("../deploy/config.example.json")));
        vm.startBroadcast();
        manager = new VideoGenBSM(registry);
        manager.configureModel(
            vm.parseJsonString(config, ".video.model"),
            uint64(vm.parseJsonUint(config, ".video.price_per_compute_second")),
            uint32(vm.parseJsonUint(config, ".gpu.min_vram_mib")),
            uint32(vm.parseJsonUint(config, ".video.max_duration_secs"))
        );
        vm.stopBroadcast();
        console2.log("DEPLOY_MANAGER=%s", vm.toString(address(manager)));
    }
}
