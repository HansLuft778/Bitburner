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
        const totalWeakenThreadsNeeded = getWeakenThreadsEff(ns, target);

        ns.print(Colors.CYAN + "------------ WEAKENING ------------" + Colors.reset);
        ns.print("total weaken threads needed: " + totalWeakenThreadsNeeded);
        // weaken to min sec lvl
        let threadsDispatched = 0;
        let threadsRemaining = totalWeakenThreadsNeeded;
        while (threadsDispatched < totalWeakenThreadsNeeded) {
            const weakenTime = ns.getWeakenTime(target);

            for (let i = 0; i < allHosts.length; i++) {
                if (threadsDispatched >= totalWeakenThreadsNeeded) break;

                const host = allHosts[i];
                const freeRam = host.maxRam - ns.getServerUsedRam(host.name);
                if (freeRam < weakenScriptRam) continue;
                const numThreadsOnHost = Math.floor(freeRam / weakenScriptRam);

                const threadsToDispatch = Math.min(threadsRemaining, numThreadsOnHost);

                ns.exec("weaken.js", host.name, threadsToDispatch, target);
                threadsRemaining -= threadsToDispatch;
                threadsDispatched += threadsToDispatch;
            }
            ns.print("dispatched " + threadsDispatched + " weaken threads");
            await ns.sleep(weakenTime + safetyMarginMs + 1000);
            ns.print("done with " + threadsDispatched + "/" + totalWeakenThreadsNeeded + " weakens");
        }
        printServerStats(ns, target, threshold);

        ns.print(Colors.CYAN + "------------- GROWING -------------" + Colors.reset);
        const totalGrowThreadsNeeded = getGrowThreads(ns, target);
        // check if grow is needed
        if (totalGrowThreadsNeeded === 0) {
            ns.print("No growth needed");
            break;
        }
        ns.print("total growing threads needed: " + totalGrowThreadsNeeded);

        // grow one batch
        const growingTime = ns.getGrowTime(target);
        threadsDispatched = 0;
        for (let i = 0; i < allHosts.length; i++) {
            // if (threadsDispatched >= totalGrowThreadsNeeded) break;

            const host = allHosts[i];
            const freeRam = host.maxRam - ns.getServerUsedRam(host.name);
            if (freeRam < growingScriptRam) continue;
            const numThreadsOnHost = Math.floor(freeRam / growingScriptRam);

            ns.exec("grow.js", host.name, numThreadsOnHost, target);
            threadsDispatched += numThreadsOnHost;
        }
        ns.print("dispatched " + threadsDispatched + " grow threads");
        await ns.sleep(growingTime + safetyMarginMs);
        ns.print("done with " + threadsDispatched + "/" + totalGrowThreadsNeeded + " grows");
        printServerStats(ns, target, threshold);
    }

    ns.print("Done preparing!");
}
