import { NS } from "@ns";

import { nukeAll, getTimeH } from "../lib.js";
import { getBestServer } from "../bestServer.js";
import { weakenServer } from "./weakenAlgo.js";
import { growServer } from "./growingAlgo.js";
import { hackServer } from "./hackingAlgo.js";
import { printServerStats } from "../serverStats.js";

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

const delayMarginMs = 1000;
const hackThreshold = 0.9;
let lastTarget = "";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    // timing order (always same): weaken > grow > hack
    // for now, each has own server:    aws-0   aws-    aws-2   aws-3
    //                                  weak    weak    grow    hack

    // steps: WGWH-WGWH-..
    while (true) {
        await parallelManager(ns);
    }
}

export async function parallelManager(ns: NS) {
    // find the server with the most available money
    const target = getBestServer(ns);

    if (lastTarget != target) {
        nukeAll(ns);
        ns.print("found new best Server: " + target);
        printServerStats(ns, target, hackThreshold);
    }
    lastTarget = target;

    // debug
    // target = "max-hardware"

    // get execution times:
    const weakTime = ns.getWeakenTime(target);
    const growTime = ns.getGrowTime(target);
    const hackTime = ns.getHackTime(target);

    // buyServer(ns, 32)

    // weak I
    ns.print("Attempting to start Weak I at " + getTimeH());
    const weak1Dispatched = weakenServer(ns, target, "aws-0", 1);

    // --------------------------------------
    // weak II delay

    // if weak I skip, start II imediately
    let weak2StartTime = 0;
    if (weak1Dispatched == true) {
        // weak2StartTime = weakTime + 2 * delayMarginMs - weakTime;
        weak2StartTime = 2 * delayMarginMs;
        await ns.sleep(weak2StartTime);
    }

    // weak II
    ns.print("Attempting to start Weak II at " + getTimeH());
    const weak2Dispatched = weakenServer(ns, target, "aws-1", 2);

    // --------------------------------------
    // grow delay

    let growStartTime = 0;
    if (weak2Dispatched == true) {
        growStartTime = weakTime + delayMarginMs - growTime;
        const growDelay = growStartTime - weak2StartTime;
        await ns.sleep(growDelay);
    }

    // grow
    ns.print("Attempting to start Grow at " + getTimeH());
    const growDispatched = growServer(ns, target, "aws-2");

    // --------------------------------------
    // hacking

    // hacking start logic, for further time optimizations
    // note: when weak2 fails, the grow must also fail (and vice versa: when grow fails, weak2 shouldnt have started)
    if (weak1Dispatched == true && weak2Dispatched == false && growDispatched == false) {
        // szenario: weak1 geht, rest skip
        // hack finishes 1 margin unit after weak1 ends
        ns.print(
            yellow +
                "Weak 2 was skipped. Did the last hack attempt fail?\nHacking is about to start earlier than planned." +
                reset,
        );
        const hackStartTime = weakTime + delayMarginMs - hackTime;
        await ns.sleep(hackStartTime);
        hackServer(ns, target, "aws-3", hackThreshold);
        await ns.sleep(hackTime + delayMarginMs); // wait for hack complete
    } else if (weak1Dispatched == false && weak2Dispatched == false && growDispatched == false) {
        // szenario: weak1 und weak2 skip
        // hack immediately
        ns.print(yellow + "Weak 1 and Weak 2 were skipped? Hacking now." + reset);
        hackServer(ns, target, "aws-3", hackThreshold);
        await ns.sleep(hackTime + delayMarginMs); // wait for hack complete
    } else if (weak1Dispatched == true && growDispatched == true && weak2Dispatched == true) {
        // hack normal
        ns.print(green + "Hack is about to start as expected" + reset);
        const hackStartTime = weakTime + 3 * delayMarginMs - hackTime;
        const hackDelayDiff = hackStartTime - growStartTime;
        await ns.sleep(hackDelayDiff);
        hackServer(ns, target, "aws-3", hackThreshold);
        await ns.sleep(hackTime + delayMarginMs); // wait for hack complete
    } else {
        ns.print(red + "could not start hack!" + reset);
        ns.print(
            "answerWeak1: " +
                weak1Dispatched +
                " | answerWeak2: " +
                weak2Dispatched +
                " | answerGrow: " +
                growDispatched,
        );
        printServerStats(ns, target, hackThreshold);
        return;
    }

    // --------------------------------------
    ns.print(green + "Cycle done. Beginning new cycle.." + reset);
    printServerStats(ns, target, hackThreshold);
}

/**
 notes:
 weaken removes 0.05 sec lvl
 grow adds 0.004 sec lvl

 grow adds money:

 */
