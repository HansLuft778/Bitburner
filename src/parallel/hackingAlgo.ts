import { getBestHostByRam } from "@/bestServer";
import { NS } from "@ns";
import { ServerManager } from "./serverManager";

export async function main(ns: NS) {
    ns.tail();
    hackServer(ns, "silver-helix", "hacker", 0.8);
}

export function hackServer(ns: NS, target: string, host: string, threshold: number, delay: number = 0) {
    let totalHackThreadsNeeded = Math.ceil(threshold / ns.hackAnalyze(target));
    // ns.print("threshold: " + threshold);
    // ns.print("hackAnalyze: " + ns.hackAnalyze(target));
    // ns.print("threshold / ns.hackAnalyze(target): " + threshold / ns.hackAnalyze(target));
    ns.print("actual hack threads needed: " + totalHackThreadsNeeded);

    // hackThreads = 100;
    // const hackThreads = getHackThreads(ns, target, threshold);
    // ns.print("hackThreads: " + hackThreads);

    const allHosts = getBestHostByRam(ns);
    const hackingScriptRam = 1.7;

    for (let i = 0; i < allHosts.length; i++) {
        const host = allHosts[i];

        const maxThreadsOnHost = Math.floor(host.availableRam / hackingScriptRam);

        if (maxThreadsOnHost >= totalHackThreadsNeeded) {
            ns.exec("hack.js", host.name, totalHackThreadsNeeded, target);
            return true;
        }
    }

    ns.print("No available host to grow " + target + ". Buying server...");

    // buy server
    const server = ServerManager.buyServer(ns, totalHackThreadsNeeded * hackingScriptRam);

    if (server === "") {
        ns.tprint("Error! Could not buy server to hack " + target);
        throw new Error("Error! Could not buy server to hack " + target);
    }

    ns.exec("hack.js", server, totalHackThreadsNeeded, target);

    return true;


    // let threadsDispatched = 0;
    // let threadsRemaining = totalHackThreadsNeeded;
    // for (let i = 0; i < allHosts.length; i++) {
    //     if (threadsDispatched >= totalHackThreadsNeeded) break;
    //     const host = allHosts[i].name;

    //     const maxRam = ns.getServerMaxRam(host);
    //     const freeRam = maxRam - ns.getServerUsedRam(host);
    //     if (freeRam < hackingScriptRam) continue;
    //     const threadSpace = Math.floor(freeRam / hackingScriptRam);

    //     // if threadsRemaining is less than the threadSpace, then we can only dispatch threadsRemaining threads
    //     const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

    //     ns.exec("hack.js", host, threadsToDispatch, target, delay);
    //     threadsRemaining -= threadsToDispatch;
    //     threadsDispatched += threadsToDispatch;
    // }

    // if (threadsRemaining > 0) {
    //     ns.tprint("[HACK] Error! There are threads remaining after dispatching all threads");
    //     throw new Error("[HACK] Error! There are threads remaining after dispatching all threads");
    // }

    // const maxRam = ns.getServerMaxRam(host);
    // const freeRam = maxRam - ns.getServerUsedRam(target);

    // const maxThreadsOnHost = Math.floor(freeRam / hackingRam);

    // if (maxThreadsOnHost < hackThreads)
    //     throw new Error(
    //         "can't one-hit hack on server " +
    //             target +
    //             ".\nneed " +
    //             hackThreads +
    //             " Threads, only got " +
    //             maxThreadsOnHost,
    //     );

    // ns.exec("hack.js", host, hackThreads, target);
    // // ns.exec("hack.js", "aws-0", 1000, target);
}
