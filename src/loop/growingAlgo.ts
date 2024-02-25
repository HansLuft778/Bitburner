import { NS } from "@ns";

import { getBestHostByRam, getBestServerListCheap } from "../bestServer";

export async function main(ns: NS) {
    ns.tail();
    await growServer(ns, "foodnstuff");
}

export async function growServer(ns: NS, target: string) {
    const safetyMarginMs = 200;

    const serverMaxMoney = ns.getServerMaxMoney(target);
    const serverCurrentMoney = ns.getServerMoneyAvailable(target);
    const moneyMult = serverMaxMoney / serverCurrentMoney;

    const targetGrowThreads = Math.ceil(ns.growthAnalyze(target, moneyMult));
    const growingScriptRam = 1.75;

    ns.print("current money: " + serverCurrentMoney + " max: " + serverMaxMoney + " mult: " + moneyMult);
    ns.print("total growing threads needed: " + targetGrowThreads);

    const allHosts = getBestHostByRam(ns);

    const totalMaxRam = allHosts.reduce((acc, server) => {
        return acc + server.maxRam;
    }, 0);
    const numRuns = Math.ceil(targetGrowThreads / totalMaxRam);

    ns.print("total RAM: " + totalMaxRam + " numRuns: " + numRuns + "\nthreads to finish: " + targetGrowThreads);

    let sumThreadsDone = 0;
    while (sumThreadsDone < targetGrowThreads) {
        const growingTime = ns.getGrowTime(target);
        for (let i = 0; i < allHosts.length; i++) {
            if (sumThreadsDone >= targetGrowThreads) break;

            const host = allHosts[i];
            const freeRam = host.maxRam - ns.getServerUsedRam(host.name);

            const numThreadsOnHost = Math.floor(freeRam / growingScriptRam);

            ns.exec("grow.js", host.name, numThreadsOnHost, target);
            sumThreadsDone += numThreadsOnHost;
        }
        await ns.sleep(growingTime + safetyMarginMs);
        ns.print("done with " + sumThreadsDone + "/" + targetGrowThreads + " weakens");
    }

    ns.print("Done growing!");
}
