import { NS } from "@ns";

import { getBestServerListCheap } from "../bestServer";

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
    const growingTime = ns.getGrowTime(target);

    ns.print("current money: " + serverCurrentMoney + " max: " + serverMaxMoney + " mult: " + moneyMult);
    ns.print("total growing threads needed: " + targetGrowThreads);

    let allHosts = getBestServerListCheap(ns, false)
        .filter((server) => {
            return server.maxRam > 2;
        })
        .sort((a, b) => {
            // TODO: sort might be unnecessary
            return b.maxRam - a.maxRam;
        });

    const totalMaxRam = allHosts.reduce((acc, server) => {
        return acc + server.maxRam;
    }, 0);
    const numRuns = Math.ceil(targetGrowThreads / totalMaxRam);

    ns.print(
        "total RAM: " +
            totalMaxRam +
            " numRuns: " +
            numRuns +
            "\nthreads to finish: " +
            targetGrowThreads +
            " time: " +
            growingTime,
    );

    let sumThreadsDone = 0;
    while (sumThreadsDone < targetGrowThreads) {
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

    // let sumThreadsDone = 0;
    // for (let i = 0; i < Math.floor(happen); i++) {
    //     const growingTime = ns.getGrowTime(target);
    //     ns.exec("grow.js", host, numThreadsOnHost, target);
    //     await ns.sleep(growingTime + safetyMarginMs);
    //     sumThreadsDone += numThreadsOnHost;
    //     ns.print("done with " + sumThreadsDone + "/" + growThreads + " growings");
    // }
    // if (sumThreadsDone < growThreads) {
    //     ns.print("need to grow " + (growThreads - sumThreadsDone) + " more time");
    //     const growingTime = ns.getGrowTime(target);
    //     ns.exec("grow.js", host, growThreads - sumThreadsDone, target);
    //     await ns.sleep(growingTime + safetyMarginMs);
    // }
    ns.print("Done growing!");
}
