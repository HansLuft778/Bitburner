import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    growServer(ns, "foodnstuff", "hacker");
}

export function growServer(ns: NS, target: string, host: string) {
    const serverMaxMoney = ns.getServerMaxMoney(target);
    const serverCurrentMoney = ns.getServerMoneyAvailable(target);
    let moneyMult = serverMaxMoney / serverCurrentMoney;
    if (isNaN(moneyMult) || moneyMult == Infinity) moneyMult = 1;

    const growThreads = Math.ceil(ns.growthAnalyze(target, moneyMult));

    if (growThreads < 1) {
        ns.print("Growthreads are 0, skipping grow");
        return false;
    }

    // exec grow.js with num of threads
    const growingScriptRam = 1.75;
    const maxRam = ns.getServerMaxRam(host);
    const freeRam = maxRam - ns.getServerUsedRam(target);

    const numThreadsOnHost = Math.floor(freeRam / growingScriptRam);

    if (numThreadsOnHost < growThreads)
        throw new Error(
            "can't onehit weaken on server " +
                target +
                ".\nneed " +
                growThreads +
                " Threads, only got " +
                numThreadsOnHost,
        );

    ns.exec("grow.js", host, growThreads, target);

    return true;

    // const happen = growThreads / numThreadsOnHost

    // ns.print("max ram: " + maxRam + " free ram: " + freeRam)
    // ns.print("threads on host: " + numThreadsOnHost + " happen: " + happen)

    // let sumThreadsDone = 0
    // for (let i = 0; i < Math.floor(happen); i++) {
    // 	const growingTime = ns.getGrowTime(target)
    // 	ns.exec("grow.js", host, numThreadsOnHost, target)
    // 	await ns.sleep(growingTime + safetyMarginMs)
    // 	sumThreadsDone += numThreadsOnHost;
    // 	ns.print("done with " + sumThreadsDone + "/" + growThreads + " growings")
    // }
    // if (sumThreadsDone < growThreads) {
    // 	ns.print("need to grow " + (growThreads - sumThreadsDone) + " more time")
    // 	const growingTime = ns.getGrowTime(target)
    // 	ns.exec("grow.js", host, growThreads - sumThreadsDone, target)
    // 	await ns.sleep(growingTime + safetyMarginMs)
    // }
    // ns.print("Done growing!")
}
