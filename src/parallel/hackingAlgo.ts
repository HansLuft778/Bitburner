import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    hackServer(ns, "foodnstuff", "hacker", 0.8);
}

export function hackServer(ns: NS, target: string, host: string, threshold: number) {
    const hackThreads = Math.ceil(threshold / ns.hackAnalyze(target));

    const hackingRam = 1.7;
    const maxRam = ns.getServerMaxRam(host);
    const freeRam = maxRam - ns.getServerUsedRam(target);

    const numThreadsOnHost = Math.floor(freeRam / hackingRam);

    if (numThreadsOnHost < hackThreads)
        throw new Error(
            "can't onehit hack on server " +
                target +
                ".\nneed " +
                hackThreads +
                " Threads, only got " +
                numThreadsOnHost,
        );

    ns.exec("hack.js", host, hackThreads, target);

    // const happen = hackThreads / numThreadsOnHost

    // ns.print("max ram: " + maxRam + " free ram: " + freeRam)
    // ns.print("threads on host: " + numThreadsOnHost + " happen: " + happen)

    // let sumThreadsDone = 0
    // for (let i = 0; i < Math.floor(happen); i++) {
    // 	const hackingTime = ns.getHackTime(target)
    // 	ns.exec("hack.js", host, numThreadsOnHost, target)
    // 	await ns.sleep(hackingTime + safetyMarginMs)
    // 	sumThreadsDone += numThreadsOnHost;
    // 	ns.print("done with " + sumThreadsDone + "/" + hackThreads + " hackings")
    // }
    // if (sumThreadsDone < hackThreads) {
    // 	ns.print("need to hack " + (hackThreads - sumThreadsDone) + " more time")
    // 	const hackingTime = ns.getHackTime(target)
    // 	ns.exec("hack.js", host, hackThreads - sumThreadsDone, target)
    // 	await ns.sleep(hackingTime + safetyMarginMs)
    // }
    // ns.print("Done hacking!")
}
