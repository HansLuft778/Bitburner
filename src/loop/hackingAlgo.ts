import { getBestServerListCheap } from "@/bestServer";
import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    await hackServer(ns, "foodnstuff", 0.8);
}

export async function hackServer(ns: NS, target: string, threshold: number) {
    const safetyMarginMs = 200;

    const serverMaxMoney = ns.getServerMaxMoney(target);
    const lowerMoneyBound = serverMaxMoney * threshold;
    const hackAmount = serverMaxMoney - lowerMoneyBound;

    //const hackChance = ns.hackAnalyzeChance(target) // todo
    const targetHackThreads = Math.ceil(ns.hackAnalyzeThreads(target, hackAmount));
    const hackingScriptRam = 1.7;
    const hackingTime = ns.getHackTime(target);

    ns.print("max money: " + serverMaxMoney + " 80%: " + lowerMoneyBound + " hack amount: " + hackAmount);
    ns.print(
        "total hack threads needed: " + targetHackThreads + " money available: " + ns.getServerMoneyAvailable(target),
    );

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
    const numRuns = Math.ceil(targetHackThreads / totalMaxRam);

    ns.print(
        "total RAM: " +
            totalMaxRam +
            " numRuns: " +
            numRuns +
            "\nthreads to finish: " +
            targetHackThreads +
            " time: " +
            hackingTime,
    );

    let sumThreadsDone = 0;
    while (sumThreadsDone < targetHackThreads) {
        for (let i = 0; i < allHosts.length; i++) {
            if (sumThreadsDone >= targetHackThreads) break;

            const host = allHosts[i];
            const freeRam = host.maxRam - ns.getServerUsedRam(host.name);

            const numThreadsOnHost = Math.floor(freeRam / hackingScriptRam);

            ns.exec("hack.js", host.name, numThreadsOnHost, target);
            sumThreadsDone += numThreadsOnHost;
        }
        await ns.sleep(hackingTime + safetyMarginMs);
        ns.print("done with " + sumThreadsDone + "/" + targetHackThreads + " weakens");
    }

    // let sumThreadsDone = 0;
    // for (let i = 0; i < Math.floor(happen); i++) {
    //     const hackingTime = ns.getHackTime(target);
    //     ns.exec("hack.js", host, numThreadsOnHost, target);
    //     await ns.sleep(hackingTime + safetyMarginMs);
    //     sumThreadsDone += numThreadsOnHost;
    //     ns.print("done with " + sumThreadsDone + "/" + hackThreads + " hackings");
    // }
    // if (sumThreadsDone < hackThreads) {
    //     ns.print("need to hack " + (hackThreads - sumThreadsDone) + " more time");
    //     const hackingTime = ns.getHackTime(target);
    //     ns.exec("hack.js", host, hackThreads - sumThreadsDone, target);
    //     await ns.sleep(hackingTime + safetyMarginMs);
    // }
    ns.print("Done hacking!");
}
