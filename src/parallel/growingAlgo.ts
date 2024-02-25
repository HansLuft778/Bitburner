import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    growServer(ns, "foodnstuff", "hacker");
}

export function growServer(ns: NS, target: string, host: string) {
    const serverMaxMoney = ns.getServerMaxMoney(target);
    const serverCurrentMoney = ns.getServerMoneyAvailable(target);
    let moneyMultiplier = serverMaxMoney / serverCurrentMoney;
    if (isNaN(moneyMultiplier) || moneyMultiplier == Infinity) moneyMultiplier = 1;
    const growThreads = Math.ceil(ns.growthAnalyze(target, moneyMultiplier));

    if (growThreads < 1) {
        ns.print("No grow threads needed, skipping growth process");
        return false;
    }

    // exec grow.js with num of threads
    const growingScriptRam = 1.75;
    const maxRam = ns.getServerMaxRam(host);
    const freeRam = maxRam - ns.getServerUsedRam(host);

    const maxThreadsOnHost = Math.floor(freeRam / growingScriptRam);

    if (maxThreadsOnHost < growThreads) {
        ns.tprint("Error! Not enough threads to grow " + target + " on " + host);
        throw new Error( // need 2568 Threads, only got 2340
            "can't one-hit grow on server " +
                target +
                ".\nneed " +
                growThreads +
                " Threads, only got " +
                maxThreadsOnHost,
        );
    }

    ns.exec("grow.js", host, growThreads, target);

    return true;
}
