import { getBestHostByRam } from "@/bestServer";
import { Colors, getGrowThreads, getWeakenThreads } from "@/lib";
import { NS } from "@ns";
import { ServerManager } from "./serverManager";

export async function main(ns: NS) {
    ns.tail();
    weakenServer(ns, "foodnstuff", 1, 0);
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
export function weakenServer(ns: NS, target: string, order: number, batchId: number, delay: number = 0): boolean {
    let totalWeakenThreadsNeeded = 0;
    // calculate weakening threads based on the order

    if (order == 1) {
        // first weak has to weaken server to min from unknown sec lvl
        totalWeakenThreadsNeeded = getWeakenThreads(ns, target);
        ns.print("Actual weaken1 threads needed: " + totalWeakenThreadsNeeded);
    } else if (order == 2) {
        // second weak only has to remove the sec increase from the grow before (more ram efficient)
        const growThreads = getGrowThreads(ns, target);
        const secIncrease = ns.growthAnalyzeSecurity(growThreads, target);

        totalWeakenThreadsNeeded = Math.ceil(secIncrease / ns.weakenAnalyze(1));

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
        const host = allHosts[i];

        const freeRam = host.availableRam;
        if (freeRam < weakenScriptRam) continue;
        const threadSpace = Math.floor(freeRam / weakenScriptRam);

        // if threadsRemaining is less than the threadSpace, then we can only dispatch threadsRemaining threads
        const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

        ns.exec("weaken.js", host.name, threadsToDispatch, target, delay);
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
