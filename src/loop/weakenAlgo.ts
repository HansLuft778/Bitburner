import { NS } from "@ns";
import { getBestServerListCheap } from "../bestServer";

export async function main(ns: NS) {
    ns.tail();
    await weakenServer(ns, "foodnstuff");
}

export async function weakenServer(ns: NS, target: string) {
    const safetyMarginMs = 200;

    // weaken
    const serverSecLvl = ns.getServerSecurityLevel(target);
    const targetWeakenThreads = Math.ceil((serverSecLvl - ns.getServerMinSecurityLevel(target)) / 0.05);
    const weakenScriptRam = 1.75;
    const weakenTime = ns.getWeakenTime(target);

    ns.print("min sec: " + ns.getServerMinSecurityLevel(target) + " cur sec lvl: " + serverSecLvl);
    ns.print("total weaken threads needed: " + targetWeakenThreads);

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
    const numRuns = Math.ceil(targetWeakenThreads / totalMaxRam);

    ns.print(
        "total RAM: " +
            totalMaxRam +
            " numRuns: " +
            numRuns +
            "\nthreads to finish: " +
            targetWeakenThreads +
            " time: " +
            weakenTime,
    );

    let sumThreadsDone = 0;
    while (sumThreadsDone < targetWeakenThreads) {
        for (let i = 0; i < allHosts.length; i++) {
            if (sumThreadsDone >= targetWeakenThreads) break;

            const host = allHosts[i];
            const freeRam = host.maxRam - ns.getServerUsedRam(host.name);

            const numThreadsOnHost = Math.floor(freeRam / weakenScriptRam);

            ns.exec("weaken.js", host.name, numThreadsOnHost, target);
            sumThreadsDone += numThreadsOnHost;
        }
        await ns.sleep(weakenTime + safetyMarginMs);
        ns.print("done with " + sumThreadsDone + "/" + targetWeakenThreads + " weakens");
    }

    // let sumThreadsDone = 0;
    // for (let i = 0; i < Math.floor(weakenThreadsPerRun); i++) {
    //     const weakenTime = ns.getWeakenTime(target);
    //     ns.exec("weaken.js", host, threadSpace, target);
    //     // mngr.weaken(ns);
    //     await ns.sleep(weakenTime + safetyMarginMs);
    //     sumThreadsDone += threadSpace;
    //     ns.print("done with " + sumThreadsDone + "/" + serverWeakenThreads + " weakens");
    // }
    // if (sumThreadsDone < serverWeakenThreads) {
    //     ns.print("need to weaken " + (serverWeakenThreads - sumThreadsDone) + " more times");
    //     const weakenTime = ns.getWeakenTime(target);
    //     ns.exec("weaken.js", host, serverWeakenThreads - sumThreadsDone, target);
    //     await ns.sleep(weakenTime + safetyMarginMs);
    // }
    ns.print("Done weakening!");
}
