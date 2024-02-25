import { NS } from "@ns";

import { nukeAll } from "../lib.js";
import { getBestServerListCheap } from "../bestServer.js";
import { weakenServer } from "./weakenAlgo.js";
import { growServer } from "./growingAlgo.js";
import { hackServer } from "./hackingAlgo.js";

// Text color
const reset = "\x1b[0m";
const black = "\x1b[30m";
const red = "\x1b[31m";
const green = "\x1b[32m";
const yellow = "\x1b[33m";
const blue = "\x1b[34m";
const magenta = "\x1b[35m";
const cyan = "\x1b[36m";
const white = "\x1b[37m";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    let lastTarget = "";
    // steps: WGWH-WGWH-..
    while (true) {
        // find the server with the most available money
        let target: string = getBestServerListCheap(ns, true)[0].name;
        ns.print("target: " + target);

        if (lastTarget != target) {
            nukeAll(ns);
            ns.print("found new best Server: " + target);
        }
        lastTarget = target;

        // Debug
        // target = "n00dles";

        ns.print(cyan + "------------ WEAKENING ------------" + reset);
        await weakenServer(ns, target);

        ns.print(cyan + "------------- GROWING -------------" + reset);
        await growServer(ns, target);

        ns.print(cyan + "------------ WEAKENING ------------" + reset);
        await weakenServer(ns, target);

        ns.print(cyan + "------------- HACKING -------------" + reset);
        await hackServer(ns, target, 0.8);
    }
}

/**
 notes:
 weaken removes 0.05 sec lvl
 grow adds 0.004 sec lvl

 */
