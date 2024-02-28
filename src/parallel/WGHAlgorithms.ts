import { NS } from "@ns";

import { getBestHostByRam } from "@/bestServer";
import {
    Colors,
    getGrowThreads,
    getGrowThreadsFormulas,
    getHackThreadsFormulas,
    getWeakenThreads,
    getWeakenThreadsAfterGrow,
    getWeakenThreadsAfterHack,
} from "@/lib";
import { ServerManager } from "./serverManager";

export class WGHAlgorithms {
    private static currentGrowThreads = 0;
    private static currentHackThreads = 0;

    static weakenServer(
        ns: NS,
        target: string,
        order: number,
        batchId: number,
        batchMode: boolean,
        delay: number = 0,
    ): boolean {
        let totalWeakenThreadsNeeded = 0;
        // calculate weakening threads based on the order

        if (order == 1 && !batchMode) {
            // first weak has to weaken server to min from unknown sec lvl
            totalWeakenThreadsNeeded = getWeakenThreads(ns, target);
            ns.print("Actual weaken1 threads needed: " + totalWeakenThreadsNeeded);
        } else if (order == 2 && !batchMode) {
            // second weak only has to remove the sec increase from the grow before (more ram efficient)
            const growThreads = getGrowThreads(ns, target);
            const secIncrease = ns.growthAnalyzeSecurity(growThreads, target);

            totalWeakenThreadsNeeded = Math.ceil(secIncrease / ns.weakenAnalyze(1));

            ns.print("Actual weaken2 threads needed: " + totalWeakenThreadsNeeded);
        } else if (order == 1 && batchMode) {
            // weak after previous hack
            totalWeakenThreadsNeeded = getWeakenThreadsAfterHack(ns, this.currentHackThreads);
            ns.print("Actual weaken1 threads needed: " + totalWeakenThreadsNeeded);
        } else if (order == 2 && batchMode) {
            // weak after previous grow
            totalWeakenThreadsNeeded = getWeakenThreadsAfterGrow(ns, this.currentGrowThreads);
            ns.print("Actual weaken2 threads needed: " + totalWeakenThreadsNeeded);
        } else {
            throw new Error("weaken order can only be either 1 or 2!");
        }

        if (totalWeakenThreadsNeeded < 1) {
            ns.print("Weakenthreads are 0, skipping weak " + order);
            return false;
        }

        // exec weaken.js with num of threads
        const allHosts = getBestHostByRam(ns);
        const weakenScriptRam = 1.75;

        let threadsDispatched = 0;
        let threadsRemaining = totalWeakenThreadsNeeded;
        for (let i = 0; i < allHosts.length; i++) {
            if (threadsDispatched >= totalWeakenThreadsNeeded) break;
            const host = allHosts[i].name;

            const maxRam = ns.getServerMaxRam(host);
            const freeRam = maxRam - ns.getServerUsedRam(host);
            if (freeRam < weakenScriptRam) continue;
            const threadSpace = Math.floor(freeRam / weakenScriptRam);

            // if threadsRemaining is less than the threadSpace, then we can only dispatch threadsRemaining threads
            const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

            ns.exec("weaken.js", host, threadsToDispatch, target, delay);
            threadsRemaining -= threadsToDispatch;
            threadsDispatched += threadsToDispatch;
        }

        if (threadsRemaining <= 0) {
            ns.print("Done deploying weaken" + order + "!");
            return true;
        }
        ns.print(
            Colors.YELLOW +
                "There are " +
                threadsRemaining +
                " threads remaining after dispatching all threads, attempting to dispatch remaining threads on purchased server",
        );

        const neededWeakenRam = threadsRemaining * weakenScriptRam;
        const server = ServerManager.buyOrUpgradeServer(ns, neededWeakenRam, "weak", batchId);

        if (server === "") {
            ns.tprint("Error! Could not buy server to weak " + target);
            throw new Error("Error! Could not buy server to weak " + target);
        }

        ns.exec("weaken.js", server, threadsRemaining, target, delay);

        return true;
    }

    static growServer(
        ns: NS,
        target: string,
        batchId: number,
        batchMode: boolean,
        hackThreshold: number,
        delay: number,
    ): boolean {
        let totalGrowThreadsNeeded = 0;
        if (!batchMode) {
            totalGrowThreadsNeeded = getGrowThreads(ns, target);
        } else {
            totalGrowThreadsNeeded = getGrowThreadsFormulas(ns, target, hackThreshold);
            this.currentGrowThreads = totalGrowThreadsNeeded;
        }

        ns.print("Actual grow threads needed: " + totalGrowThreadsNeeded);

        if (totalGrowThreadsNeeded < 1) {
            ns.print("No grow threads needed, skipping growth process");
            return false;
        }

        // exec grow.js with num of threads
        const allHosts = getBestHostByRam(ns);
        const growingScriptRam = 1.75;

        for (let i = 0; i < allHosts.length; i++) {
            const host = allHosts[i];

            const maxThreadsOnHost = Math.floor(host.availableRam / growingScriptRam);

            if (maxThreadsOnHost >= totalGrowThreadsNeeded) {
                ns.exec("grow.js", host.name, totalGrowThreadsNeeded, target, delay);
                return true;
            }
        }

        ns.print(Colors.YELLOW + "No available host to grow " + target + ". Attempting to upgrade/buy server...");

        const neededGrowRam = totalGrowThreadsNeeded * growingScriptRam;
        const server = ServerManager.buyOrUpgradeServer(ns, neededGrowRam, "grow", batchId);

        if (server === "") {
            ns.tprint("Error! Could not buy server to grow " + target);
            throw new Error("Error! Could not buy server to grow " + target);
        }

        ns.exec("grow.js", server, totalGrowThreadsNeeded, target, delay);

        return true;
    }

    static hackServer(ns: NS, target: string, threshold: number, batchId: number, batchMode: boolean, delay: number) {
        let totalHackThreadsNeeded = 0;
        if (!batchMode) {
            totalHackThreadsNeeded = Math.ceil(threshold / ns.hackAnalyze(target));
        } else {
            totalHackThreadsNeeded = getHackThreadsFormulas(ns, target, threshold);
            this.currentHackThreads = totalHackThreadsNeeded;
        }
        ns.print("actual hack threads needed: " + totalHackThreadsNeeded);

        const allHosts = getBestHostByRam(ns);
        const hackingScriptRam = 1.7;

        for (let i = 0; i < allHosts.length; i++) {
            const host = allHosts[i];

            const maxThreadsOnHost = Math.floor(host.availableRam / hackingScriptRam);

            if (maxThreadsOnHost >= totalHackThreadsNeeded) {
                ns.exec("hack.js", host.name, totalHackThreadsNeeded, target, delay);
                return true;
            }
        }

        ns.print(Colors.YELLOW + "No available host to hack " + target + ". Buying server...");

        const neededGrowRam = totalHackThreadsNeeded * hackingScriptRam;
        const server = ServerManager.buyOrUpgradeServer(ns, neededGrowRam, "hack", batchId);

        if (server === "") {
            ns.tprint("Error! Could not buy server to hack " + target);
            throw new Error("Error! Could not buy server to hack " + target);
        }

        ns.exec("hack.js", server, totalHackThreadsNeeded, target, delay);

        return true;
    }
}
