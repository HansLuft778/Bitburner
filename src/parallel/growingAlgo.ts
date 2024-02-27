import { getBestHostByRam } from "@/bestServer";
import { NS } from "@ns";
import { ServerManager } from "./serverManager";

export async function main(ns: NS) {
    ns.tail();
    growServer(ns, "foodnstuff");
}

export function growServer(ns: NS, target: string, delay: number = 0): boolean {
    const serverMaxMoney = ns.getServerMaxMoney(target);
    const serverCurrentMoney = ns.getServerMoneyAvailable(target);
    let moneyMultiplier = serverMaxMoney / serverCurrentMoney;
    if (isNaN(moneyMultiplier) || moneyMultiplier == Infinity) moneyMultiplier = 1;
    const totalGrowThreadsNeeded = Math.ceil(ns.growthAnalyze(target, moneyMultiplier));

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
            ns.exec("grow.js", host.name, totalGrowThreadsNeeded, target);
            return true;
        }
    }

    ns.print("No available host to grow " + target + ". Buying server...");

    // buy server
    const server = ServerManager.buyServer(ns, totalGrowThreadsNeeded * growingScriptRam);

    if (server === "") {
        ns.tprint("Error! Could not buy server to grow " + target);
        throw new Error("Error! Could not buy server to grow " + target);
    }

    ns.exec("grow.js", server, totalGrowThreadsNeeded, target);

    return true;

    // let threadsDispatched = 0;
    // let threadsRemaining = totalGrowThreadsNeeded;
    // for (let i = 0; i < allHosts.length; i++) {
    //     if (threadsDispatched >= totalGrowThreadsNeeded) break;
    //     const host = allHosts[i].name;

    //     const maxRam = ns.getServerMaxRam(host);
    //     const freeRam = maxRam - ns.getServerUsedRam(host);
    //     if (freeRam < growingScriptRam) continue;
    //     const threadSpace = Math.floor(freeRam / growingScriptRam);

    //     // if threadsRemaining is less than the threadSpace, then we can only dispatch threadsRemaining threads
    //     const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

    //     ns.exec("grow.js", host, threadsToDispatch, target, delay);
    //     threadsRemaining -= threadsToDispatch;
    //     threadsDispatched += threadsToDispatch;
    // }

    // if (threadsRemaining > 0) {
    //     ns.tprint("[GROW] Error! There are threads remaining after dispatching all threads");
    //     throw new Error("[GROW] Error! There are threads remaining after dispatching all threads");
    // }
    // return true;
}
