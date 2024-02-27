import { getBestHostByRam } from "@/bestServer";
import { getWeakenThreadsEff } from "@/lib";
import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    weakenServer(ns, "foodnstuff", "hacker", 1);
}

/**
 * Weakens a server by executing the weaken.js script with the specified number of threads.
 * The number of threads is calculated based on the order of weakening and the target server's properties.
 *
 * @param ns - The NetScriptAPI object.
 * @param target - The name of the target server to weaken.
 * @param host - The name of the current server hosting the weaken script.
 * @param order - The order of weakening. Can only be either 1 or 2.
 * @returns A boolean indicating whether the weaken operation was successful.
 * @throws An error if the weaken order is not 1 or 2, or if there is not enough free RAM to execute the weaken operation.
 */
export function weakenServer(ns: NS, target: string, host: string, order: number): boolean {
    let totalWeakenThreadsNeeded = 0;
    // calculate weakening threads based on the order
    if (order == 2) {
        // second weak only has to remove the sec increase from the grow before (more ram efficient)
        const serverMaxMoney = ns.getServerMaxMoney(target);
        const serverCurrentMoney = ns.getServerMoneyAvailable(target);
        let moneyMultiplier = serverMaxMoney / serverCurrentMoney; // doesn't work when current money is 0
        if (isNaN(moneyMultiplier) || moneyMultiplier == Infinity) moneyMultiplier = 1;
        const growThreads = Math.ceil(ns.growthAnalyze(target, moneyMultiplier));

        // const maxMoney = ns.getServerMaxMoney(target);
        // const minMoney = maxMoney * (1 - 0.8);
        // const moneyMultiplier = maxMoney / minMoney;
        // const growThreads = Math.ceil(ns.growthAnalyze(target, moneyMultiplier));

        const secIncrease = ns.growthAnalyzeSecurity(growThreads, target);

        totalWeakenThreadsNeeded = Math.ceil(secIncrease / ns.weakenAnalyze(1));

        ns.print("Actual weaken2 threads needed: " + totalWeakenThreadsNeeded);
    } else if (order == 1) {
        // first weak has to weaken server to min from unknown sec lvl
        totalWeakenThreadsNeeded = getWeakenThreadsEff(ns, target);
        ns.print("Actual weaken1 threads needed: " + totalWeakenThreadsNeeded);
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

        ns.exec("weaken.js", host, threadsToDispatch, target);
        threadsRemaining -= threadsToDispatch;
        threadsDispatched += threadsToDispatch;
    }

    if (threadsRemaining > 0) {
        ns.tprint("[WEAKEN] Error! There are threads remaining after dispatching all threads");
        throw new Error("[WEAKEN] Error! There are threads remaining after dispatching all threads");
    }
    return true;
}
