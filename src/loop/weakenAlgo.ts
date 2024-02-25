import { NS } from "@ns";
import { getBestHostByRam } from "../bestServer";
import { getWeakenThreadsEff } from "@/lib";

export async function main(ns: NS) {
    ns.tail();
    await weakenServer(ns, "foodnstuff");
}

export async function weakenServer(ns: NS, target: string) {
    const safetyMarginMs = 200;

    // weaken
    const targetWeakenThreads = getWeakenThreadsEff(ns, target);
    const weakenScriptRam = 1.75;

    ns.print("total weaken threads needed: " + targetWeakenThreads);

    const allHosts = getBestHostByRam(ns);

    const totalMaxRam = allHosts.reduce((acc, server) => {
        return acc + server.maxRam;
    }, 0);
    const numRuns = Math.ceil(targetWeakenThreads / totalMaxRam);

    ns.print("total RAM: " + totalMaxRam + " numRuns: " + numRuns + "\nthreads to finish: " + targetWeakenThreads);

    let sumThreadsDone = 0;
    while (sumThreadsDone < targetWeakenThreads) {
        const weakenTime = ns.getWeakenTime(target);
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

    ns.print("Done weakening!");
}
