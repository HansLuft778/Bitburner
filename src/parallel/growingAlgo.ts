import { getBestHostByRam } from "@/bestServer";
import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    growServer(ns, "foodnstuff", "hacker");
}

export function growServer(ns: NS, target: string, host: string) {
    const serverMaxMoney = ns.getServerMaxMoney(target);
    const serverCurrentMoney = ns.getServerMoneyAvailable(target);
    let moneyMultiplier = serverMaxMoney / serverCurrentMoney;
    if (isNaN(moneyMultiplier) || moneyMultiplier == Infinity) moneyMultiplier = 1;
    const totalGrowThreadsNeeded = Math.ceil(ns.growthAnalyze(target, moneyMultiplier));

    if (totalGrowThreadsNeeded < 1) {
        ns.print("No grow threads needed, skipping growth process");
        return false;
    }

    // exec grow.js with num of threads
    const allHosts = getBestHostByRam(ns);
    const growingScriptRam = 1.75;

    let threadsDispatched = 0;
    let threadsRemaining = totalGrowThreadsNeeded;
    for (let i = 0; i < allHosts.length; i++) {
        if (threadsDispatched >= totalGrowThreadsNeeded) break;
        const host = allHosts[i].name;

        const maxRam = ns.getServerMaxRam(host);
        const freeRam = maxRam - ns.getServerUsedRam(host);
        if (freeRam < growingScriptRam) continue;
        const threadSpace = Math.floor(freeRam / growingScriptRam);

        // if threadsRemaining is less than the threadSpace, then we can only dispatch threadsRemaining threads
        const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

        ns.exec("grow.js", host, threadsToDispatch, target);
        threadsRemaining -= threadsToDispatch;
        threadsDispatched += threadsToDispatch;
    }

    if (threadsRemaining > 0) {
        ns.tprint("[GROW] Error! There are threads remaining after dispatching all threads");
        throw new Error("[GROW] Error! There are threads remaining after dispatching all threads");
    }
    return true;

    // const maxRam = ns.getServerMaxRam(host);
    // const freeRam = maxRam - ns.getServerUsedRam(host);

    // const maxThreadsOnHost = Math.floor(freeRam / growingScriptRam);

    // if (maxThreadsOnHost < growThreads) {
    //     ns.tprint("Error! Not enough threads to grow " + target + " on " + host);
    //     throw new Error( // need 2568 Threads, only got 2340
    //         "can't one-hit grow on server " +
    //             target +
    //             ".\nneed " +
    //             growThreads +
    //             " Threads, only got " +
    //             maxThreadsOnHost,
    //     );
    // }

    // ns.exec("grow.js", host, growThreads, target);
}
