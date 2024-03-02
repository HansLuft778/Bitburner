import { Config } from "@/Config/Config";
import { Colors, getGrowThreads, getWeakenThreads, getWeakenThreadsAfterGrow, nukeAll } from "@/lib";
import { WGHAlgorithms } from "@/parallel/WGHAlgorithms";
import { printServerStats } from "@/serverStats";
import { NS } from "@ns";
import { getBestHostByRamOptimized } from "../bestServer";

export async function main(ns: NS) {
    ns.tail();
    await prepareServer(ns, "foodnstuff");
}

export async function prepareServer(ns: NS, target: string, threshold = 0.8) {
    // either prepare in loop or parallel mode
    let allHosts = getBestHostByRamOptimized(ns);
    const sumAvailableRam = allHosts.reduce((acc, server) => {
        return acc + server.availableRam;
    }, 0);

    // how much threads are needed: weaken from unknown to min + grow from unknown to max + weaken grow effect
    const weakenThreads = getWeakenThreads(ns, target);
    const growThreads = getGrowThreads(ns, target);
    const weakenAfterGrowThreads = getWeakenThreadsAfterGrow(ns, growThreads);

    const totalRamNeeded =
        weakenThreads * Config.WEAKEN_SCRIPT_RAM +
        growThreads * Config.GROW_SCRIPT_RAM +
        weakenAfterGrowThreads * Config.WEAKEN_SCRIPT_RAM;

    ns.print(
        "needs " + totalRamNeeded + "GB of RAM and got " + sumAvailableRam + " to running parallel mode on " + target,
    );

    if (totalRamNeeded === 0) {
        ns.print("No preparation needed");
        return;
    }

    if (totalRamNeeded < sumAvailableRam) {
        // -------------------------------- PARALLEL MODE --------------------------------
        ns.print(Colors.CYAN + "Preparing " + target + " in parallel mode");

        const weakTime = ns.getWeakenTime(target);
        const growTime = ns.getGrowTime(target);

        WGHAlgorithms.weakenServer(ns, target, 1, false, 0, false);

        const growDelay = weakTime - growTime + Config.DELAY_MARGIN_MS;
        WGHAlgorithms.growServer(ns, target, false, growDelay, false);

        const weak2delay = 2 * Config.DELAY_MARGIN_MS;
        WGHAlgorithms.weakenServer(ns, target, 2, false, weak2delay, false);

        // wait for prep to finish
        await ns.sleep(weakTime + 4 * Config.DELAY_MARGIN_MS);
    } else {
        // ---------------------------------- LOOP MODE ----------------------------------
        ns.print(Colors.CYAN + "Preparing " + target + " in loop mode");

        const safetyMarginMs = Config.DELAY_MARGIN_MS;
        let weakenMandatory = false;

        // TODO: use similar method as in parallel/manager.ts to let the grow finish right after the weaken
        while (true) {
            const totalWeakenThreadsNeeded = getWeakenThreads(ns, target);

            nukeAll(ns);
            allHosts = getBestHostByRamOptimized(ns);
            if (totalWeakenThreadsNeeded > 50 || weakenMandatory) {
                ns.print(Colors.CYAN + "------------ WEAKENING ------------");
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
                        if (freeRam < Config.WEAKEN_SCRIPT_RAM) continue;
                        const numThreadsOnHost = Math.floor(freeRam / Config.WEAKEN_SCRIPT_RAM);

                        const threadsToDispatch = Math.min(threadsRemaining, numThreadsOnHost);

                        ns.exec("weaken.js", host.name, threadsToDispatch, target, 0);
                        threadsRemaining -= threadsToDispatch;
                        threadsDispatched += threadsToDispatch;
                    }
                    ns.print("dispatched " + threadsDispatched + " weaken threads");
                    await ns.sleep(weakenTime + safetyMarginMs + 1000);
                    ns.print("done with " + threadsDispatched + "/" + totalWeakenThreadsNeeded + " weakens");
                }
                printServerStats(ns, target, threshold);

                if (weakenMandatory) {
                    break;
                }
            }

            ns.print(Colors.CYAN + "------------- GROWING -------------");
            const totalGrowThreadsNeeded = getGrowThreads(ns, target);
            // check if grow is needed
            if (totalGrowThreadsNeeded === 0) {
                ns.print("No growth needed");
                weakenMandatory = true;
                continue;
            }
            ns.print("total growing threads needed: " + totalGrowThreadsNeeded);

            // grow one batch
            const growingTime = ns.getGrowTime(target);
            let threadsDispatched = 0;
            for (let i = 0; i < allHosts.length; i++) {
                // if (threadsDispatched >= totalGrowThreadsNeeded) break;

                const host = allHosts[i];
                const freeRam = host.maxRam - ns.getServerUsedRam(host.name);
                if (freeRam < Config.GROW_SCRIPT_RAM) continue;
                const numThreadsOnHost = Math.floor(freeRam / Config.GROW_SCRIPT_RAM);

                ns.exec("grow.js", host.name, numThreadsOnHost, target, 0);
                threadsDispatched += numThreadsOnHost;
            }
            ns.print("dispatched " + threadsDispatched + " grow threads");
            await ns.sleep(growingTime + safetyMarginMs);
            ns.print("done with " + threadsDispatched + "/" + totalGrowThreadsNeeded + " grows");
            printServerStats(ns, target, threshold);
        }
    }
}
