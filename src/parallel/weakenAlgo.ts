import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    weakenServer(ns, "foodnstuff", "hacker", 1);
}

/**
 * Weakens a server by executing the weaken.js script with the specified number of threads.
 * The number of threads is calculated based on the order of weakening and the target server's properties.
 *
 * @param ns - The NetScriptAPI object.
 * @param target - The name of the target server to weaken.
 * @param host - The name of the current server hosting the weaken script.
 * @param order - The order of weakening. Can only be either 1 or 2.
 * @returns A boolean indicating whether the weaken operation was successful.
 * @throws An error if the weaken order is not 1 or 2, or if there is not enough free RAM to execute the weaken operation.
 */
export function weakenServer(ns: NS, target: string, host: string, order: number): boolean {
    let serverWeakenThreads = 0;
    // calculate weakening threads based on the order
    if (order == 2) {
        // second weak only has to remove the sec increase from the grow before (more ram efficient)
        const serverMaxMoney = ns.getServerMaxMoney(target);
        const serverCurrentMoney = ns.getServerMoneyAvailable(target);
        let moneyMult = serverMaxMoney / serverCurrentMoney;
        if (isNaN(moneyMult) || moneyMult == Infinity) moneyMult = 1;
        const growThreads = Math.ceil(ns.growthAnalyze(target, moneyMult));

        const secIncrease = ns.growthAnalyzeSecurity(growThreads, target);

        serverWeakenThreads = Math.ceil(secIncrease / 0.05);
    } else if (order == 1) {
        // first weak has to weaken server to min from unknown sec lvl
        const serverSecLvl = ns.getServerSecurityLevel(target);
        serverWeakenThreads = Math.ceil((serverSecLvl - ns.getServerMinSecurityLevel(target)) / 0.05);
    } else {
        throw new Error("weaken order can only be either 1 or 2!");
    }

    if (serverWeakenThreads < 1) {
        ns.print("Weakenthreads are 0, skipping weak " + order);
        return false;
    }

    // exec weaken.js with num of threads
    const weakenRam = 1.75;
    const maxRam = ns.getServerMaxRam(host);
    const freeRam = maxRam - ns.getServerUsedRam(host);

    const threadSpace = Math.floor(freeRam / weakenRam);

    if (threadSpace < serverWeakenThreads)
        throw new Error(
            "can't onehit weaken on server " +
                target +
                ".\nneed " +
                serverWeakenThreads +
                " Threads, only got " +
                threadSpace,
        );

    ns.exec("weaken.js", host, serverWeakenThreads, target);
    return true;

    // const happen = serverWeakenThreads / threadSpace
    // ns.print("will weaken in " + happen + " steps")

    // let howMany = 0
    // for (let i = 0; i < Math.floor(happen); i++) {
    // 	const weakenTime = ns.getWeakenTime(target)
    // 	ns.exec("weaken.js", host, threadSpace, target)
    // 	await ns.sleep(weakenTime + safetyMarginMs)
    // 	howMany += threadSpace;
    // 	ns.print("done with " + howMany + "/" + serverWeakenThreads + " weakens")
    // }
    // if (howMany < serverWeakenThreads) {
    // 	ns.print("need to weaken " + (serverWeakenThreads - howMany) + " more times")
    // 	const weakenTime = ns.getWeakenTime(target)
    // 	ns.exec("weaken.js", host, serverWeakenThreads - howMany, target)
    // 	await ns.sleep(weakenTime + safetyMarginMs)
    // }
}
