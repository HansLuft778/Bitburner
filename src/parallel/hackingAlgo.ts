import { getHackThreads } from "@/lib";
import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    hackServer(ns, "foodnstuff", "hacker", 0.8);
}

export function hackServer(ns: NS, target: string, host: string, threshold: number) {
    let hackThreads = Math.ceil(threshold / ns.hackAnalyze(target));
    ns.print("hackThreads: " + hackThreads);
    hackThreads = 100;
    // const hackThreads = getHackThreads(ns, target, threshold);
    ns.print("hackThreads: " + hackThreads);

    const hackingRam = 1.7;
    const maxRam = ns.getServerMaxRam(host);
    const freeRam = maxRam - ns.getServerUsedRam(target);

    const maxThreadsOnHost = Math.floor(freeRam / hackingRam);

    if (maxThreadsOnHost < hackThreads)
        throw new Error(
            "can't onehit hack on server " +
                target +
                ".\nneed " +
                hackThreads +
                " Threads, only got " +
                maxThreadsOnHost,
        );

    ns.exec("hack.js", host, hackThreads, target);
    // ns.exec("hack.js", "aws-0", 1000, target);
}
