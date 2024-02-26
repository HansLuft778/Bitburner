import { getBestHostByRam } from "@/bestServer";
import { getHackThreads } from "@/lib";
import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    hackServer(ns, "silver-helix", "hacker", 0.8);
}

export function hackServer(ns: NS, target: string, host: string, threshold: number) {
    let totalHackThreadsNeeded = Math.ceil(threshold / ns.hackAnalyze(target));
    // ns.print("threshold: " + threshold);
    // ns.print("hackAnalyze: " + ns.hackAnalyze(target));
    // ns.print("threshold / ns.hackAnalyze(target): " + threshold / ns.hackAnalyze(target));
    ns.print("hackThreads: " + totalHackThreadsNeeded);

    // hackThreads = 100;
    // const hackThreads = getHackThreads(ns, target, threshold);
    // ns.print("hackThreads: " + hackThreads);

    const allHosts = getBestHostByRam(ns);
    const hackingScriptRam = 1.7;

    let threadsDispatched = 0;
    let threadsRemaining = totalHackThreadsNeeded;
    for (let i = 0; i < allHosts.length; i++) {
        if (threadsDispatched >= totalHackThreadsNeeded) break;
        const host = allHosts[i].name;

        const maxRam = ns.getServerMaxRam(host);
        const freeRam = maxRam - ns.getServerUsedRam(host);
        if (freeRam < hackingScriptRam) continue;
        const threadSpace = Math.floor(freeRam / hackingScriptRam);

        // if threadsRemaining is less than the threadSpace, then we can only dispatch threadsRemaining threads
        const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

        ns.exec("hack.js", host, threadsToDispatch, target);
        threadsRemaining -= threadsToDispatch;
        threadsDispatched += threadsToDispatch;
    }

    if (threadsRemaining > 0) {
        ns.tprint("[HACK] Error! There are threads remaining after dispatching all threads");
        throw new Error("[HACK] Error! There are threads remaining after dispatching all threads");
    }

    // const maxRam = ns.getServerMaxRam(host);
    // const freeRam = maxRam - ns.getServerUsedRam(target);

    // const maxThreadsOnHost = Math.floor(freeRam / hackingRam);

    // if (maxThreadsOnHost < hackThreads)
    //     throw new Error(
    //         "can't onehit hack on server " +
    //             target +
    //             ".\nneed " +
    //             hackThreads +
    //             " Threads, only got " +
    //             maxThreadsOnHost,
    //     );

    // ns.exec("hack.js", host, hackThreads, target);
    // // ns.exec("hack.js", "aws-0", 1000, target);
}
