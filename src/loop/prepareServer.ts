import { NS } from "@ns";
import { getBestHostByRam } from "../bestServer";
import { Colors, getGrowThreads, getWeakenThreadsEff } from "@/lib";
import { printServerStats } from "@/serverStats";

export async function main(ns: NS) {
    ns.tail();
    await prepareServer(ns, "foodnstuff");
}

export async function prepareServer(ns: NS, target: string, threshold: number = 0.8) {
    const safetyMarginMs = 200;

    const weakenScriptRam = 1.75;
    const growingScriptRam = 1.75;

    const allHosts = getBestHostByRam(ns);

    // TODO: use similar method as in parallel/manager.ts to let the grow finish right after the weaken
    while (true) {
        const targetWeakenThreads = getWeakenThreadsEff(ns, target);

        ns.print(Colors.cyan + "------------ WEAKENING ------------" + Colors.reset);
        ns.print("total weaken threads needed: " + targetWeakenThreads);
        // weaken to min sec lvl
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
        printServerStats(ns, target, threshold);
        const targetGrowThreads = getGrowThreads(ns, target);
        // check if grow is needed
        if (targetGrowThreads === 0) {
            ns.print("No growth needed");
            break;
        }

        ns.print(Colors.cyan + "------------- GROWING -------------" + Colors.reset);
        ns.print("total growing threads needed: " + targetGrowThreads);
        // grow one batch
        const growingTime = ns.getGrowTime(target);
        let sumGrowthThreadsDone = 0;
        for (let i = 0; i < allHosts.length; i++) {
            if (sumGrowthThreadsDone >= targetGrowThreads) break;

            const host = allHosts[i];
            const freeRam = host.maxRam - ns.getServerUsedRam(host.name);

            const numThreadsOnHost = Math.floor(freeRam / growingScriptRam);

            ns.exec("grow.js", host.name, numThreadsOnHost, target);
            sumGrowthThreadsDone += numThreadsOnHost;
        }
        await ns.sleep(growingTime + safetyMarginMs);
        ns.print("done with " + sumGrowthThreadsDone + "/" + targetGrowThreads + " grows");
        printServerStats(ns, target, threshold);
    }

    ns.print("Done preparing!");
}
