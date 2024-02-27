import { NS } from "@ns";
import { getBestHostByRam, getBestServerListCheap } from "./bestServer";
import { Colors, getGrowThreadsThreshold, getWeakenThreadsAfterHack } from "./lib";
import { prepareServer } from "./loop/prepareServer";
import { parallelCycle } from "./parallel/manager";

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
    // target = "phantasy";
    ns.print("target: " + target);

    // ----------------- PREPARE SERVER -----------------

    // prepare when money is not at max or sec lvl is not at min
    if (
        ns.getServerMaxMoney(target) != ns.getServerMoneyAvailable(target) ||
        ns.getServerSecurityLevel(target) != ns.getServerMinSecurityLevel(target)
    ) {
        // await prepareServer(ns, target, MONEY_HACK_THRESHOLD);
    }

    if (
        ns.getServerMaxMoney(target) == ns.getServerMoneyAvailable(target) &&
        ns.getServerSecurityLevel(target) == ns.getServerMinSecurityLevel(target)
    ) {
        ns.print(Colors.GREEN + "Preparation finished, starting parallel mode");
    } else {
        ns.print(Colors.RED + "Preparation failed, starting loop mode");
    }

    // ----------------- CHECK WHICH MODE TO USE -----------------
    let hackThreshold = getOptimalHackThreshold(ns, target);
    hackThreshold = 0.9;
    while (true) {
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

/**
 * Needs update: check until i can buy WGH servers
 */
function getOptimalHackThreshold(ns: NS, target: string): number {
    const allHosts = getBestHostByRam(ns);
    const totalMaxRam = allHosts.reduce((acc, server) => {
        return acc + server.maxRam;
    }, 0);

    let hackThreshold = 0.9;
    const THRESHOLD_STEP = 0.05;
    while (true) {
        // how many threads i need to grow the server from (1 - Threshold) to 1
        // needs threshold grow calculation, cause when the server is at max money, it would return 0 otherwise
        const serverGrowThreads = getGrowThreadsThreshold(ns, target, hackThreshold + THRESHOLD_STEP);

        // how many threads i need to weaken security to 0 after growings
        const secIncrease = ns.growthAnalyzeSecurity(serverGrowThreads);
        const secondWeakenThreads = Math.ceil(secIncrease / ns.weakenAnalyze(1));

        // how many threads i need to hack the server
        const hackAmount = ns.getServerMaxMoney(target) * hackThreshold;
        const serverHackThreads = Math.ceil(ns.hackAnalyzeThreads(target, hackAmount));

        // how many to weak to min sec lvl after [threshold]-hack
        const firstWeakenThreads = getWeakenThreadsAfterHack(ns, serverHackThreads);

        // this var describes the total amount of threads i need to run parallel mode
        const totalRamNeeded =
            firstWeakenThreads * RAM_WEAKEN +
            serverGrowThreads * RAM_GROW +
            secondWeakenThreads * RAM_WEAKEN +
            serverHackThreads * RAM_HACK;

        // log all
        ns.print("predicted threads needed:");
        ns.print("firstWeakenThreads: " + firstWeakenThreads);
        ns.print("serverGrowThreads: " + serverGrowThreads);
        ns.print("secondWeakenThreads: " + secondWeakenThreads);
        ns.print("serverHackThreads: " + serverHackThreads);

        if (totalRamNeeded < totalMaxRam) {
            ns.print(
                "needs " + totalRamNeeded + "GB of RAM and got " + totalMaxRam + ". Running parallel mode on " + target,
            );
            break;
        }
        hackThreshold = Math.round((hackThreshold - THRESHOLD_STEP) * 100) / 100;
        ns.print("Threshold is too high, trying with: " + hackThreshold);
    }
    return hackThreshold;
}
