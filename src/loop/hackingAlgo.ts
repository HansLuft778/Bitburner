import { getBestHostByRam, getBestServerListCheap } from "@/bestServer";
import { getHackThreads } from "@/lib";
import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    await hackServer(ns, "foodnstuff", 0.8);
}

export async function hackServer(ns: NS, target: string, threshold: number) {
    const safetyMarginMs = 200;

    //const hackChance = ns.hackAnalyzeChance(target) // todo
    const targetHackThreads = getHackThreads(ns, target, threshold);
    const hackingScriptRam = 1.7;

    ns.print(
        "total hack threads needed: " + targetHackThreads + " money available: " + ns.getServerMoneyAvailable(target),
    );

    const allHosts = getBestHostByRam(ns);

    const totalMaxRam = allHosts.reduce((acc, server) => {
        return acc + server.maxRam;
    }, 0);
    const numRuns = Math.ceil(targetHackThreads / totalMaxRam);

    ns.print("total RAM: " + totalMaxRam + " numRuns: " + numRuns + "\nthreads to finish: " + targetHackThreads);

    let sumThreadsDone = 0;
    while (sumThreadsDone < targetHackThreads) {
        const hackingTime = ns.getHackTime(target);
        for (let i = 0; i < allHosts.length; i++) {
            if (sumThreadsDone >= targetHackThreads) break;

            const host = allHosts[i];
            const freeRam = host.maxRam - ns.getServerUsedRam(host.name);

            const numThreadsOnHost = Math.floor(freeRam / hackingScriptRam);

            ns.exec("hack.js", host.name, numThreadsOnHost, target, 0);
            sumThreadsDone += numThreadsOnHost;
        }
        await ns.sleep(hackingTime + safetyMarginMs);
        ns.print("done with " + sumThreadsDone + "/" + targetHackThreads + " hacks");
    }

    ns.print("Done hacking!");
}
