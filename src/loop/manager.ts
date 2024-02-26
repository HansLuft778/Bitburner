import { NS } from "@ns";

import { Colors, nukeAll } from "../lib.js";
import { getBestServerListCheap } from "../bestServer.js";
import { weakenServer } from "./weakenAlgo.js";
import { growServer } from "./growingAlgo.js";
import { hackServer } from "./hackingAlgo.js";
import { prepareServer } from "./prepareServer.js";

let lastTarget = "";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    // steps: WGWH-WGWH-..
    while (true) {
        await loopCycle(ns);
    }
}

export async function loopCycle(ns: NS, target: string = "", threshold: number = 0.8, shouldHack: boolean = true) {
    // find the server with the most available money
    if (target == "") {
        target = getBestServerListCheap(ns, true)[0].name;
    }

    ns.print("target: " + target);

    if (lastTarget != target) {
        nukeAll(ns);
        ns.print("found new best Server: " + target);
    }
    lastTarget = target;

    // ns.print(cyan + "------------ WEAKENING ------------" + reset);
    // await weakenServer(ns, target);

    // ns.print(cyan + "------------- GROWING -------------" + reset);
    // await growServer(ns, target);

    // ns.print(cyan + "------------ WEAKENING ------------" + reset);
    // await weakenServer(ns, target);

    ns.print(Colors.cyan + "------------ PREPARING ------------" + Colors.reset);
    await prepareServer(ns, target, threshold);

    if (shouldHack) {
        ns.print(Colors.cyan + "------------- HACKING -------------" + Colors.reset);
        await hackServer(ns, target, threshold);
    }
}

/**
 notes:
 weaken removes 0.05 sec lvl
 grow adds 0.004 sec lvl

 */
