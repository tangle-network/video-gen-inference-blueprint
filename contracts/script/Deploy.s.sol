// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";
import {VideoGenBSM} from "../src/VideoGenBSM.sol";

/// Model-specific manager deployment; cargo-tangle registers the definition.
contract Deploy is Script {
    function run() external {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        vm.startBroadcast();
        address manager = address(new VideoGenBSM(registry));
        vm.stopBroadcast();
        console2.log("DEPLOY_VIDEO_GEN_BSM=%s", vm.toString(manager));
    }
}
