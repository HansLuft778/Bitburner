import { NS } from "@ns";
import { getBestHostByRam, getBestServerListCheap } from "./bestServer";
import { parallelManager } from "./parallel/manager";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    // either start loop or parrallelize, depending on the number of servers and money the player hass

    let servers = getBestServerListCheap(ns, false);

    let target = servers[0].name;
    // is server WGH-onehit use parallel

    // grow
    const serverMaxMoney = ns.getServerMaxMoney(target);
    const serverCurrentMoney = ns.getServerMoneyAvailable(target);
    let moneyMult = serverMaxMoney / serverCurrentMoney;
    if (isNaN(moneyMult) || moneyMult == Infinity) moneyMult = 1;
    const growThreads = Math.ceil(ns.growthAnalyze(target, moneyMult));

    // weak
    const secIncrease = ns.growthAnalyzeSecurity(growThreads, target);
    const serverWeakenThreads = Math.ceil(secIncrease / 0.05);

    // hack
    const lowerMoneyBound = serverMaxMoney * 0.8;
    const hackAmount = serverMaxMoney - lowerMoneyBound;
    const targetHackThreads = Math.ceil(ns.hackAnalyzeThreads(target, hackAmount));

    const weakRamNeeded = 1.75 * serverWeakenThreads;
    const growRamNeeded = 1.75 * growThreads;
    const hackRamNeeded = 1.7 * targetHackThreads;
    const totalRamNeeded = weakRamNeeded + growRamNeeded + hackRamNeeded;

    let optimalHost = getBestHostByRam(ns);
    let totalMaxRam = optimalHost.reduce((acc, server) => {
        return acc + server.maxRam;
    }, 0);

    if (totalMaxRam > totalRamNeeded) {
        // lauch parallel
        ns.print("totalMaxRam > totalRamNeeded");
    }

    // check if the player has enough money to buy a server to use parallel
    const powerOfTwo = Math.ceil(Math.log2(totalMaxRam));
    let money = ns.getPurchasedServerCost(powerOfTwo);
    if (money < ns.getPlayer().money) {
        // ns.purchaseServer("aws", powerOfTwo);
        ns.print("bought server");
        // lauch parallel
    }

    // lauch loop
    ns.print("lauch loop");
}

function launchParallel(ns: NS) {
    while (true) {
        parallelManager(ns);
    }
}

function launchLoop(ns: NS) {
    while (true) {}
}
