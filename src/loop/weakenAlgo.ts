/** @param {NS} ns */
export async function main(ns) {
    ns.tail();
    await weakenServer(ns, "foodnstuff", "hacker");
}

/** @param {NS} ns */
export async function weakenServer(ns, target, host) {
    const safetyMarginMs = 200;

    // weaken
    const serverSecLvl = ns.getServerSecurityLevel(target);
    const serverWeakenThreads = Math.ceil((serverSecLvl - ns.getServerMinSecurityLevel(target)) / 0.05);
    //const serverWeakenEffect = ns.weakenAnalyze(serverWeakenThreads) //todo: debug, remove in prod

    ns.print(
        "min sec: " +
            ns.getServerMinSecurityLevel(target) +
            " cur sec lvl: " +
            serverSecLvl +
            " weaken threads: " +
            serverWeakenThreads /*+ " effect: " + serverWeakenEffect*/,
    );

    // exec weaken.js with num of threads
    const weakenRam = 1.75;
    const maxRam = ns.getServerMaxRam(host);
    const freeRam = maxRam - ns.getServerUsedRam(host);

    const threadSpace = Math.floor(freeRam / weakenRam);
    ns.print("freeRam:" + freeRam + " maxRam: " + maxRam + " threadSpace " + threadSpace);

    const happen = serverWeakenThreads / threadSpace;
    ns.print("will weaken in " + happen + " steps");

    let howMany = 0;
    for (let i = 0; i < Math.floor(happen); i++) {
        const weakenTime = ns.getWeakenTime(target);
        ns.exec("weaken.js", host, threadSpace, target);
        await ns.sleep(weakenTime + safetyMarginMs);
        howMany += threadSpace;
        ns.print("done with " + howMany + "/" + serverWeakenThreads + " weakens");
    }
    if (howMany < serverWeakenThreads) {
        ns.print("need to weaken " + (serverWeakenThreads - howMany) + " more times");
        const weakenTime = ns.getWeakenTime(target);
        ns.exec("weaken.js", host, serverWeakenThreads - howMany, target);
        await ns.sleep(weakenTime + safetyMarginMs);
    }
    ns.print("Done weakening!");
}
