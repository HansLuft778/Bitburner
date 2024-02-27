import { NS } from "@ns";

import { nukeAll, getTimeH, Colors } from "../lib.js";
import { getBestServer } from "../bestServer.js";
import { weakenServer } from "./weakenAlgo.js";
import { growServer } from "./growingAlgo.js";
import { hackServer } from "./hackingAlgo.js";
import { printServerStats } from "../serverStats.js";

const GROW_HOST = "home";
const WEAK_HOST = "home";
const HACK_HOST = "home";

const DELAY_MARGIN_MS = 1000;
let lastTarget = "";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    // timing order (always same): weaken > grow > hack
    // for now, each has own server:    aws-0   aws-    aws-2   aws-3
    //                                  weak    weak    grow    hack

    // steps: WGWH-WGWH-..
    while (true) {
        await parallelCycle(ns);
    }
}

export async function parallelCycle(ns: NS, target: string = "", hackThreshold: number = 0.8) {
    // find the server with the most available money
    if (target == "") {
        target = getBestServer(ns);
    }

    if (lastTarget != target) {
        nukeAll(ns);
        ns.print("found new best Server: " + target);
        printServerStats(ns, target, hackThreshold);
    }
    lastTarget = target;

    // debug
    // target = "phantasy"

    // get execution times:
    const weakTime = ns.getWeakenTime(target);
    const growTime = ns.getGrowTime(target);
    const hackTime = ns.getHackTime(target);

    // --------------------------------------
    // weak I

    ns.print("Attempting to start Weak I at " + getTimeH());
    const weak1Dispatched = weakenServer(ns, target, 1);

    // --------------------------------------
    // weak II

    let weak2delay = 2 * DELAY_MARGIN_MS;
    ns.print("Attempting to start Weak II at " + getTimeH(Date.now() + weak2delay));
    weakenServer(ns, target, 2, weak2delay);

    // --------------------------------------
    // grow

    let growDelay = weakTime - growTime + DELAY_MARGIN_MS;
    ns.print("Attempting to start Grow at " + getTimeH());
    growServer(ns, target, growDelay);

    // --------------------------------------
    // hacking

    // hack normal
    const hackStartTime = weakTime - hackTime + 3 * DELAY_MARGIN_MS;
    ns.print("Attempting to start Hack at " + getTimeH());
    hackServer(ns, target, HACK_HOST, hackThreshold, hackStartTime);

    ns.print(Colors.green + "Cycle done. Beginning new cycle.." + Colors.reset);
    printServerStats(ns, target, hackThreshold);

    await ns.sleep(weakTime + 4 * DELAY_MARGIN_MS);

    // await ns.sleep(hackTime + DELAY_MARGIN_MS);

    // hacking start logic, for further time optimizations
    // note: when weak2 fails, the grow must also fail (and vice versa: when grow fails, weak2 should not have started)
    // if (weak1Dispatched == true && weak2Dispatched == false && growDispatched == false) {
    //     // scenario: weak1 works, rest skip
    //     // hack finishes 1 margin unit after weak1 ends
    //     ns.print(
    //         Colors.yellow +
    //             "Weak 2 was skipped. Did the last hack attempt fail?\nHacking is about to start earlier than planned." +
    //             Colors.reset,
    //     );
    //     const hackStartTime = weakTime + DELAY_MARGIN_MS - hackTime;
    //     await ns.sleep(hackStartTime);
    //     ns.print("Attempting to start Hack at " + getTimeH());
    //     hackServer(ns, target, HACK_HOST, hackThreshold);
    //     await ns.sleep(hackTime + DELAY_MARGIN_MS);
    // } else if (weak1Dispatched == false && weak2Dispatched == false && growDispatched == false) {
    //     // scenario: weak1 and weak2 skipped
    //     ns.print(Colors.yellow + "Weak 1 and Weak 2 were skipped? Hacking now. " + getTimeH() + Colors.reset);
    //     hackServer(ns, target, HACK_HOST, hackThreshold);
    //     await ns.sleep(hackTime + DELAY_MARGIN_MS);
    // } else if (weak1Dispatched == true && growDispatched == true && weak2Dispatched == true) {
    //     // hack normal
    //     ns.print(Colors.green + "Hack is about to start as expected" + Colors.reset);
    //     const hackStartTime = weakTime + 3 * DELAY_MARGIN_MS - hackTime;
    //     const hackDelayDiff = hackStartTime - growDelay;
    //     await ns.sleep(hackDelayDiff);
    //     ns.print("Attempting to start Hack at " + getTimeH());
    //     hackServer(ns, target, HACK_HOST, hackThreshold);
    //     await ns.sleep(hackTime + DELAY_MARGIN_MS);
    // } else if (weak1Dispatched == false && weak2Dispatched == true && growDispatched == true) {
    //     // case weak1 was skipped, but weak2 and grow were dispatched

    //     ns.print(
    //         Colors.yellow + "Weak 1 was skipped. Perhaps the server is already at the min sec lvl." + Colors.reset,
    //     );
    //     const hackStartTime = weakTime + 2 * DELAY_MARGIN_MS - hackTime;
    //     await ns.sleep(hackStartTime - growDelay);
    //     ns.print("Attempting to start Hack at " + getTimeH());
    //     hackServer(ns, target, HACK_HOST, hackThreshold);
    //     await ns.sleep(hackTime + DELAY_MARGIN_MS);
    // } else {
    //     ns.print(Colors.red + "could not start hack!" + Colors.reset);
    //     ns.print(
    //         "weak1Dispatched: " +
    //             weak1Dispatched +
    //             " | weak2Dispatched: " +
    //             weak2Dispatched +
    //             " | growDispatched: " +
    //             growDispatched,
    //     );
    //     printServerStats(ns, target, hackThreshold);
    //     return;
    // }

    // --------------------------------------
}

/**
 notes:
 weaken removes 0.05 sec lvl
 grow adds 0.004 sec lvl

 grow adds money:

 */
