import { NS } from "@ns";
import { getBestHostByRam, getBestServerListCheap } from "./bestServer";
import { parallelCycle } from "./parallel/manager";
import { loopCycle } from "./loop/manager";
import { getGrowThreads, getHackThreads, getWeakenThreads } from "./lib";

const MONEY_HACK_THRESHOLD = 0.8;
const RAM_WEAKEN = 1.75;
const RAM_GROW = 1.75;
const RAM_HACK = 1.7;

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    // either start loop or parrallelize, depending on the number of servers and money the player hass

    // ----------------- PREPARE SERVER -----------------

    // TODO: find if prep is needed
    // await loopCycle(ns);

    // ----------------- CHECK WHICH MODE TO USE -----------------

    let target = getBestServerListCheap(ns, false)[0].name;

    // now we have to find out wether to keep using loop or switch to parallel mode this should be decided dynamicallys

    const allHosts = getBestHostByRam(ns);
    const totalMaxRam = allHosts.reduce((acc, server) => {
        return acc + server.maxRam;
    }, 0);

    // i have to find out, how many threads of each i need, after

    // how many to weak from current sec lvl to min
    const firstWeakenThreads = getWeakenThreads(ns, target);

    // how many threads i need to grow the server from (1 - Threshold) to 1
    const maxMoney = ns.getServerMaxMoney(target);
    const minMoney = maxMoney * (1 - MONEY_HACK_THRESHOLD);
    const moneyMult = maxMoney / minMoney;
    const serverGrowThreads = Math.ceil(ns.growthAnalyze(target, moneyMult));

    // how many threads i need to weaken security to 0 after growings

    const secIncrease = ns.growthAnalyzeSecurity(serverGrowThreads, target);
    const secondWeakenThreads = Math.ceil(secIncrease / ns.weakenAnalyze(1));

    // how many threads i need to hack the server
    const hackAmount = ns.getServerMaxMoney(target) * MONEY_HACK_THRESHOLD;
    const serverHackThreads = Math.ceil(ns.hackAnalyzeThreads(target, hackAmount));

    // this var describes the total amount of threads i need to run parallel mode
    const sumThreads = firstWeakenThreads + serverGrowThreads + secondWeakenThreads + serverHackThreads;

    ns.print(sumThreads + " threads needed");
    if (sumThreads < totalMaxRam) {
        ns.print("sumThreads < totalMaxRam");
        // launchParallel(ns);
    }
    ns.print("sumThreads > totalMaxRam");
}

function launchParallel(ns: NS) {
    while (true) {
        parallelCycle(ns);
    }
}

function launchLoop(ns: NS) {
    while (true) {}
}
