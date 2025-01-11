
import { getBestServerListCheap } from "../bestServer.js";
import { Colors, nukeAll } from "../lib.js";
import { hackServer } from "./hackingAlgo.js";
import { prepareServer } from "./prepareServer.js";

let lastTarget = "";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    // steps: WGWH-WGWH-..
    while (true) {
        const target = getBestServerListCheap(ns, true)[0].name;
        if (lastTarget != target) {
            nukeAll(ns);
            ns.print("found new best Server: " + target);
        }
        lastTarget = target;
        await loopCycle(ns, target, 0.8);
    }
}

export async function loopCycle(ns: NS, target: string, threshold: number) {
    ns.print(Colors.CYAN + "------------ PREPARING ------------" + Colors.RESET);
    await prepareServer(ns, target, threshold);

    ns.print(Colors.CYAN + "------------- HACKING -------------" + Colors.RESET);
    await hackServer(ns, target, threshold);
}

/**
 notes:
 weaken removes 0.05 sec lvl
 grow adds 0.004 sec lvl

 */
