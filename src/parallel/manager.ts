import { NS } from "@ns";

import { getBestServer } from "../bestServer.js";
import { Colors, getTimeH, nukeAll } from "../lib.js";
import { printServerStats } from "../serverStats.js";
import { WGHAlgorithms } from "./WGHAlgorithms.js";
import { growServer } from "./growingAlgo.js";
import { hackServer } from "./hackingAlgo.js";
import { weakenServer } from "./weakenAlgo.js";

const DELAY_MARGIN_MS = 1000;

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    // timing order (always same): weaken > grow > hack
    // for now, each has own server:    aws-0   aws-    aws-2   aws-3
    //                                  weak    weak    grow    hack

    // steps: WGWH-WGWH-..
    while (true) {
        let target = getBestServer(ns);
        await parallelCycle(ns, target, 0.8);
    }
}

export async function parallelCycle(ns: NS, target: string, hackThreshold: number = 0.8, num_batches: number = 1) {
    const weakTime = ns.getWeakenTime(target);

    if (num_batches > 1) {
        ns.print(Colors.CYAN + "------------ MULTI BATCH MODE ------------");

        for (let batchId = 0; batchId < num_batches; batchId++) {
            ns.print(Colors.CYAN + "------------ BATCH " + batchId + " ------------");
            // get execution times:
            const weakTime = ns.getWeakenTime(target);
            const growTime = ns.getGrowTime(target);
            const hackTime = ns.getHackTime(target);

            // --------------------------------------
            // hacking

            // hack normal
            // const hackDelay = weakTime - hackTime + 3 * DELAY_MARGIN_MS;
            const hackDelay = weakTime - hackTime - DELAY_MARGIN_MS;
            ns.print("Attempting to start Hack at " + getTimeH());
            WGHAlgorithms.hackServer(ns, target, hackThreshold, batchId, true, hackDelay);

            // --------------------------------------
            // weak I

            ns.print("Attempting to start Weak I at " + getTimeH());
            // weakenServer(ns, target, 1, batchId, true);
            WGHAlgorithms.weakenServer(ns, target, 1, batchId, true);

            // --------------------------------------
            // grow

            let growDelay = weakTime - growTime + DELAY_MARGIN_MS;
            ns.print("Attempting to start Grow at " + getTimeH());
            // growServer(ns, target, batchId, growDelay);
            WGHAlgorithms.growServer(ns, target, batchId, true, hackThreshold, growDelay);

            // --------------------------------------
            // weak II

            let weak2delay = 2 * DELAY_MARGIN_MS;
            ns.print("Attempting to start Weak II at " + getTimeH(Date.now() + weak2delay));
            // weakenServer(ns, target, 2, batchId, true, weak2delay);
            WGHAlgorithms.weakenServer(ns, target, 2, batchId, true, weak2delay);

            // --------------------------------------

            printServerStats(ns, target, hackThreshold);

            ns.print(Colors.GREEN + "Cycle done. Beginning new cycle.." + Colors.RESET);
            await ns.sleep(4 * DELAY_MARGIN_MS);
        }
        await ns.sleep(weakTime);
    } else {
        ns.print(Colors.CYAN + "------------ SINGLE BATCH MODE ------------");
        const weakTime = ns.getWeakenTime(target);
        const growTime = ns.getGrowTime(target);
        const hackTime = ns.getHackTime(target);
        // weak I
        ns.print("Attempting to start Weak I at " + getTimeH());
        const weak1Dispatched = weakenServer(ns, target, 1, 0);

        // --------------------------------------
        // weak II delay

        // if weak I skip, start II immediately
        let weak2StartTime = 0;
        if (weak1Dispatched == true) {
            // weak2StartTime = weakTime + 2 * DELAY_MARGIN_MS - weakTime;
            weak2StartTime = 2 * DELAY_MARGIN_MS;
            await ns.sleep(weak2StartTime);
        }
        // weak II
        ns.print("Attempting to start Weak II at " + getTimeH());
        const weak2Dispatched = weakenServer(ns, target, 2, 0);

        // --------------------------------------
        // grow delay

        let growStartTime = 0;
        if (weak2Dispatched == true) {
            growStartTime = weakTime + DELAY_MARGIN_MS - growTime;
            const growDelay = growStartTime - weak2StartTime;
            await ns.sleep(growDelay);
        }

        // grow
        ns.print("Attempting to start Grow at " + getTimeH());
        const growDispatched = growServer(ns, target, 0);

        // --------------------------------------
        // hacking

        // hacking start logic, for further time optimizations
        // note: when weak2 fails, the grow must also fail (and vice versa: when grow fails, weak2 should not have started)
        if (weak1Dispatched == true && weak2Dispatched == false && growDispatched == false) {
            // scenario: weak1 works, rest skip
            // hack finishes 1 margin unit after weak1 ends
            ns.print(
                Colors.YELLOW +
                    "Weak 2 was skipped. Did the last hack attempt fail?\nHacking is about to start earlier than planned." +
                    Colors.RESET,
            );
            const hackStartTime = weakTime + DELAY_MARGIN_MS - hackTime;
            await ns.sleep(hackStartTime);
            ns.print("Attempting to start Hack at " + getTimeH());
            hackServer(ns, target, hackThreshold, 0);
            await ns.sleep(hackTime + DELAY_MARGIN_MS);
        } else if (weak1Dispatched == false && weak2Dispatched == false && growDispatched == false) {
            // scenario: weak1 and weak2 skipped
            ns.print(Colors.YELLOW + "Weak 1 and Weak 2 were skipped? Hacking now. " + getTimeH() + Colors.RESET);
            hackServer(ns, target, hackThreshold, 0);
            await ns.sleep(hackTime + DELAY_MARGIN_MS);
        } else if (weak1Dispatched == true && growDispatched == true && weak2Dispatched == true) {
            // hack normal
            ns.print(Colors.GREEN + "Hack is about to start as expected" + Colors.RESET);
            const hackStartTime = weakTime + 3 * DELAY_MARGIN_MS - hackTime;
            const hackDelayDiff = hackStartTime - growStartTime;
            await ns.sleep(hackDelayDiff);
            ns.print("Attempting to start Hack at " + getTimeH());
            hackServer(ns, target, hackThreshold, 0);
            await ns.sleep(hackTime + DELAY_MARGIN_MS);
        } else if (weak1Dispatched == false && weak2Dispatched == true && growDispatched == true) {
            // case weak1 was skipped, but weak2 and grow were dispatched

            ns.print(
                Colors.YELLOW + "Weak 1 was skipped. Perhaps the server is already at the min sec lvl." + Colors.RESET,
            );
            const hackStartTime = weakTime + 2 * DELAY_MARGIN_MS - hackTime;
            await ns.sleep(hackStartTime - growStartTime);
            ns.print("Attempting to start Hack at " + getTimeH());
            hackServer(ns, target, hackThreshold, 0);
            await ns.sleep(hackTime + DELAY_MARGIN_MS);
        } else {
            ns.print(Colors.RED + "could not start hack!" + Colors.RESET);
            ns.print(
                "weak1Dispatched: " +
                    weak1Dispatched +
                    " | weak2Dispatched: " +
                    weak2Dispatched +
                    " | growDispatched: " +
                    growDispatched,
            );
            printServerStats(ns, target, hackThreshold);
            return;
        }
    }
}

/**
 notes:
 weaken removes 0.05 sec lvl
 grow adds 0.004 sec lvl

 grow adds money:

 */
