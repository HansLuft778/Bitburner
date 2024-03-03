import { NS } from "@ns";

import { Config } from "@/Config/Config.js";
import { getBestServer } from "../bestServer.js";
import { Colors, getTimeH } from "../lib.js";
import { printServerStats } from "../serverStats.js";
import { WGHAlgorithms } from "./WGHAlgorithms.js";
import { growServer } from "./growingAlgo.js";
import { hackServer } from "./hackingAlgo.js";
import { weakenServer } from "./weakenAlgo.js";

const DELAY_MARGIN_MS = Config.DELAY_MARGIN_MS;

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    // timing order (always same): weaken > grow > hack
    // for now, each has own server:    aws-0   aws-    aws-2   aws-3
    //                                  weak    weak    grow    hack

    // steps: WGWH-WGWH-..
    while (true) {
        const target = getBestServer(ns);
        await parallelCycle(ns, target, 0.8);
    }
}
// let cycleCounter = 0;
// let offset = 1;
export async function parallelCycle(ns: NS, target: string, hackThreshold = 0.8, num_batches = 1) {
    // const time = Time.getInstance();

    const weakTime = ns.getWeakenTime(target);
    const growTime = ns.getGrowTime(target);
    const hackTime = ns.getHackTime(target);

    if (num_batches > 1) {
        ns.print(Colors.CYAN + "------------ MULTI BATCH MODE ------------");

        num_batches = Math.floor((weakTime - Config.LOOP_SAFETY_MARGIN_MS) / (4 * DELAY_MARGIN_MS));

        ns.tprint(Colors.E_ORANGE + "num_batches: " + num_batches);

        for (let batchId = 0; batchId < num_batches; batchId++) {
            ns.print(Colors.CYAN + "------------ BATCH " + batchId + " ------------");

            const pids = [];
            // --------------------------------------
            // hacking
            const hackDelay = weakTime - hackTime - DELAY_MARGIN_MS;
            const hackPid = WGHAlgorithms.hackServer(ns, target, hackThreshold, true, hackDelay);
            pids.push(hackPid);

            // --------------------------------------
            // weak I
            const weak1Pid = WGHAlgorithms.weakenServer(ns, target, 1, true);
            pids.push(weak1Pid);

            // --------------------------------------
            // grow
            const growDelay = weakTime - growTime + DELAY_MARGIN_MS;
            const growPid = WGHAlgorithms.growServer(ns, target, true, growDelay);
            pids.push(growPid);

            // --------------------------------------
            // weak II
            const weak2delay = 2 * DELAY_MARGIN_MS;
            const weak2Pid = WGHAlgorithms.weakenServer(ns, target, 2, true, weak2delay);
            pids.push(weak2Pid);

            // --------------------------------------
            // check if all processes were dispatched, kill them if not
            if (weak1Pid == 0 || weak2Pid == 0 || growPid == 0 || hackPid == 0) {
                ns.print(Colors.RED + "could not start all processes, killing batch " + batchId);

                for (const pid of pids) {
                    ns.kill(pid);
                }
                break;
            }

            pids.length = 0;

            ns.print(Colors.GREEN + "Cycle done. Beginning new cycle.." + Colors.RESET);

            await ns.sleep(4 * DELAY_MARGIN_MS);
        }
        // whats happening here is a little hard to explain but ill try my best:
        // batchDeployTime should be obvious, it is the time it took to deploy all batches
        // sleepTime is the time we wait, until the next batches can be deployed
        // offset:
        // imagine the scenario: first batch is deployed, then we wait for the next batch to be deployed.
        // When the first batch finishes, we can RAM freed up, so so this is the time we can savely
        // deploy the next batch.
        // problem is, when we just wait DELAY_MARGIN_MS long, we start to deploy the next batch in almost
        // the exact moment when the first batch finishes. This is bad, due to js and timings.
        // It can happen, that the second batch is deployed, right after grow from the first batch finishe,
        // meaning there is no money and a high seclvl on the server, which increased the weaken time of
        // the second batch, which messes up everything in the long run. Thats why we add the offset
        // for every second batch, so start the second batch in the middle of first batch completions.

        // if (cycleCounter % 2 == 0) offset = DELAY_MARGIN_MS / 2;
        // else offset = 0;

        // const batchDeployTime = 4 * DELAY_MARGIN_MS * num_batches;
        const sleepTime = weakTime; // - batchDeployTime + offset;

        await ns.sleep(sleepTime);
        // cycleCounter++;
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
