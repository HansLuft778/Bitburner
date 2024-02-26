import { NS } from "@ns";
import { getBestHostByRam, getBestServerListCheap } from "./bestServer";
import { parallelCycle } from "./parallel/manager";
import { loopCycle } from "./loop/manager";
import { getGrowThreads, getHackThreads, getWeakenThreads, getWeakenThreadsEff } from "./lib";

// TODO: determine the threshold by first prepping the server to max money/min sec lvl.
// Then find the percentage of money that can be stolen safely, so it can be regrown per one-hit.
// This might be key to solve the threads problem, without having to get the FormulasAPI
const MONEY_HACK_THRESHOLD = 0.8;
const RAM_WEAKEN = 1.75;
const RAM_GROW = 1.75;
const RAM_HACK = 1.7;

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    // either start loop or parallelize, depending on the number of servers and money the player has

    let target = getBestServerListCheap(ns, false)[0].name;
    // target = "silver-helix";
    ns.print("target: " + target);

    // ----------------- PREPARE SERVER -----------------

    // TODO: find if prep is needed
    await loopCycle(ns, target, MONEY_HACK_THRESHOLD);

    // ----------------- CHECK WHICH MODE TO USE -----------------

    // now we have to find out wether to keep using loop or switch to parallel mode this should be decided dynamically

    const allHosts = getBestHostByRam(ns);
    const totalMaxRam = allHosts.reduce((acc, server) => {
        return acc + server.maxRam;
    }, 0);

    // i have to find out, how many threads of each i need, after

    // how many to weak from current sec lvl to min
    const firstWeakenThreads = getWeakenThreadsEff(ns, target);

    // how many threads i need to grow the server from (1 - Threshold) to 1
    const maxMoney = ns.getServerMaxMoney(target);
    const minMoney = maxMoney * (1 - MONEY_HACK_THRESHOLD);
    const moneyMultiplier = maxMoney / minMoney;
    const serverGrowThreads = Math.ceil(ns.growthAnalyze(target, moneyMultiplier));

    // how many threads i need to weaken security to 0 after growings
    const secIncrease = ns.growthAnalyzeSecurity(serverGrowThreads, target);
    const secondWeakenThreads = Math.ceil(secIncrease / ns.weakenAnalyze(1));

    // how many threads i need to hack the server
    const hackAmount = ns.getServerMaxMoney(target) * MONEY_HACK_THRESHOLD;
    const serverHackThreads = Math.ceil(ns.hackAnalyzeThreads(target, hackAmount));

    // this var describes the total amount of threads i need to run parallel mode
    const sumThreads = firstWeakenThreads + serverGrowThreads + secondWeakenThreads + serverHackThreads;

    ns.print(sumThreads + " threads needed to run parallel mode on " + target);
    if (sumThreads < totalMaxRam) {
        ns.print("sumThreads < totalMaxRam");
        await launchParallel(ns, target, MONEY_HACK_THRESHOLD);
    }
    ns.print("sumThreads > totalMaxRam");
}

async function launchParallel(ns: NS, target: string, moneyHackThreshold: number) {
    // TODO: instead of calling the function, it might be better to call ns.exec() to run the script in parallel to the daemon. This enables easy support for hitting multiple targets at the same time, in case spare resources are available.
    while (true) {
        await parallelCycle(ns, target, moneyHackThreshold);
    }
}

function launchLoop(ns: NS) {
    while (true) {}
}
