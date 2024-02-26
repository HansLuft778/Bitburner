import { NS } from "@ns";
import { getBestHostByRam, getBestServerListCheap } from "./bestServer";
import { Colors, getGrowThreadsThreshold, getWeakenThreadsEff } from "./lib";
import { loopCycle } from "./loop/manager";
import { parallelCycle } from "./parallel/manager";
import { get } from "http";

// TODO: determine the threshold by first prepping the server to max money/min sec lvl.
// Then find the percentage of money that can be stolen safely, so it can be regrown per one-hit.
// This might be key to solve the threads problem, without having to get the FormulasAPI
const MONEY_HACK_THRESHOLD = 0.5;
const RAM_WEAKEN = 1.75;
const RAM_GROW = 1.75;
const RAM_HACK = 1.7;

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    // either start loop or parallelize, depending on the number of servers and money the player has

    let target = getBestServerListCheap(ns, false)[0].name;
    // target = "harakiri-sushi";
    ns.print("target: " + target);

    // ----------------- PREPARE SERVER -----------------

    // prepare when money is not at max or sec lvl is not at min
    if (
        ns.getServerMaxMoney(target) != ns.getServerMoneyAvailable(target) ||
        ns.getServerSecurityLevel(target) != ns.getServerMinSecurityLevel(target)
    ) {
        await loopCycle(ns, target, MONEY_HACK_THRESHOLD, false);
    }

    if (
        ns.getServerMaxMoney(target) == ns.getServerMoneyAvailable(target) &&
        ns.getServerSecurityLevel(target) == ns.getServerMinSecurityLevel(target)
    ) {
        ns.print(Colors.green + "Preparation finished, starting parallel mode");
    }

    // ----------------- CHECK WHICH MODE TO USE -----------------
    while (true) {
        let hackThreshold = getOptimalHackThreshold(ns, target);
        ns.print("hackThreshold: " + hackThreshold);
        await parallelCycle(ns, target, hackThreshold);
    }
}

async function launchParallel(ns: NS, target: string, moneyHackThreshold: number) {
    while (true) {
        await parallelCycle(ns, target, moneyHackThreshold);
    }
}

function launchLoop(ns: NS) {
    while (true) {}
}

function getOptimalHackThreshold(ns: NS, target: string): number {
    const allHosts = getBestHostByRam(ns);
    const totalMaxRam = allHosts.reduce((acc, server) => {
        return acc + server.maxRam;
    }, 0);

    let hackThreshold = 0.9;
    const THRESHOLD_STEP = 0.05;
    while (true) {
        // i have to find out, how many threads of each i need, after

        // how many to weak from current sec lvl to min
        const firstWeakenThreads = getWeakenThreadsEff(ns, target);

        // how many threads i need to grow the server from (1 - Threshold) to 1
        const serverGrowThreads = getGrowThreadsThreshold(ns, target, hackThreshold);

        // how many threads i need to weaken security to 0 after growings
        const secIncrease = ns.growthAnalyzeSecurity(serverGrowThreads);
        const secondWeakenThreads = Math.ceil(secIncrease / ns.weakenAnalyze(1));

        // how many threads i need to hack the server
        const hackAmount = ns.getServerMaxMoney(target) * hackThreshold;
        const serverHackThreads = Math.ceil(ns.hackAnalyzeThreads(target, hackAmount));

        // this var describes the total amount of threads i need to run parallel mode
        const totalRamNeeded =
            firstWeakenThreads * RAM_WEAKEN +
            serverGrowThreads * RAM_GROW +
            secondWeakenThreads * RAM_WEAKEN +
            serverHackThreads * RAM_HACK;

        if (totalRamNeeded <= totalMaxRam) {
            ns.print(
                "needs " + totalRamNeeded + "GB of RAM and got " + totalMaxRam + ". Running parallel mode on" + target,
            );
            break;
        }
        hackThreshold = Math.round((hackThreshold - THRESHOLD_STEP) * 100) / 100;
        ns.print("Threshold is too high, trying with: " + hackThreshold);
    }
    return hackThreshold;
}
