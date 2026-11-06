// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {RegisterBlueprint} from "tangle-inference-core/script/RegisterBlueprint.s.sol";
import {console2} from "forge-std/Script.sol";
import {VideoGenBSM} from "../src/VideoGenBSM.sol";

contract Deploy is RegisterBlueprint {
    function _deployManager() internal override returns (address) {
        return address(new VideoGenBSM());
    }

    function _report(address manager, uint64 blueprintId) internal override {
        console2.log("DEPLOY_VIDEO_GEN_BSM=%s", vm.toString(manager));
        console2.log("DEPLOY_VIDEO_GEN_BLUEPRINT_ID=%s", vm.toString(blueprintId));
    }
}
