import { NS } from "@ns";

import { Config } from "./Config/Config";
import { Time } from "./Time";
import { getBestHostByRamOptimized, getBestServer } from "./bestServer";
import {
    Colors,
    getGrowThreadsFormulas,
    getGrowThreadsThreshold,
    getHackThreadsFormulas,
    getWeakenThreadsAfterGrow,
    getWeakenThreadsAfterHack,
    writeToPort,
} from "./lib";
import { prepareServer } from "./loop/prepareServer";
import { PlayerManager } from "./parallel/PlayerManager";
import { ServerManager } from "./parallel/ServerManager";
import { parallelCycle } from "./parallel/manager";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    // either start loop or parallelize, depending on the number of servers and money the player has

    let hackThreshold = 0.5;
    let lastTarget = "";

    const time = Time.getInstance();
    while (true) {
        time.startTime();

        PlayerManager.getInstance(ns).resetPlayer(ns);

        const target = getBestServer(ns);

        writeToPort(ns, 1, target);
        ns.print("lastTarget: " + lastTarget + " target: " + target);
        if (ns.fileExists("Formulas.exe", "home")) {
            if (lastTarget !== target) {
                // ----------------- PREPARE SERVER -----------------
                if (
                    ns.getServerMaxMoney(target) != parseFloat(ns.getServerMoneyAvailable(target).toFixed(5)) ||
                    parseFloat(ns.getServerSecurityLevel(target).toFixed(5)) != ns.getServerMinSecurityLevel(target)
                ) {
                    await prepareServer(ns, target);
                }
                hackThreshold = getHackThreshold(ns, target);
                ns.tprint(Colors.E_ORANGE + "hackThreshold: " + hackThreshold);

                if (
                    ns.getServerMaxMoney(target) == parseFloat(ns.getServerMoneyAvailable(target).toFixed(5)) ||
                    parseFloat(ns.getServerSecurityLevel(target).toFixed(5)) == ns.getServerMinSecurityLevel(target)
                ) {
                    ns.print(Colors.GREEN + "Preparation finished, starting parallel mode");
                } else {
                    ns.tprint(Colors.RED + "Preparation failed");
                    throw new Error("Preparation failed");
                }

                lastTarget = target;
            }

            // ----------------- CHECK WHICH MODE TO USE -----------------
            PlayerManager.getInstance(ns).resetPlayer(ns);
            await parallelCycle(ns, target, hackThreshold, Config.LOOP_BATCH_COUNT);
        } else {
            if (lastTarget !== target) {
                // ----------------- PREPARE SERVER -----------------

                // prepare when money is not at max or sec lvl is not at min
                if (
                    ns.getServerMaxMoney(target) != ns.getServerMoneyAvailable(target) ||
                    ns.getServerSecurityLevel(target) != ns.getServerMinSecurityLevel(target)
                ) {
                    await prepareServer(ns, target);
                }
                hackThreshold = getHackThreshold(ns, target);
                ns.print("hackThreshold: " + hackThreshold);

                if (
                    ns.getServerMaxMoney(target) == ns.getServerMoneyAvailable(target) &&
                    ns.getServerSecurityLevel(target) == ns.getServerMinSecurityLevel(target)
                ) {
                    ns.print(Colors.GREEN + "Preparation finished, starting parallel mode");
                } else {
                    ns.tprint(Colors.RED + "Preparation failed, starting loop mode");
                    throw new Error("Preparation failed, starting loop mode");
                }

                lastTarget = target;
            }
            // ----------------- CHECK WHICH MODE TO USE -----------------

            await parallelCycle(ns, target, hackThreshold);
        }
        time.endTime();
        ns.print("Cycle took: " + time.getTime(ns));
    }
}

function getHackThreshold(ns: NS, target: string): number {
    const allHosts = getBestHostByRamOptimized(ns);
    const totalMaxRam = allHosts.reduce((acc, server) => {
        return acc + server.maxRam;
    }, 0);

    let hackThreshold = 0.9;
    const THRESHOLD_STEP = Config.THRESHOLD_STEP;

    // const moneyAllowedToUse = ns.getServerMoneyAvailable("home") * (2 / 3);
    const RAM_WEAKEN = Config.WEAKEN_SCRIPT_RAM;
    const RAM_GROW = Config.GROW_SCRIPT_RAM;
    const RAM_HACK = Config.HACK_SCRIPT_RAM;

    while (true) {
        let serverHackThreads = 0;
        let serverGrowThreads = 0;
        if (ns.fileExists("Formulas.exe", "home")) {
            serverHackThreads = getHackThreadsFormulas(ns, target, hackThreshold);

            serverGrowThreads = getGrowThreadsFormulas(ns, target, serverHackThreads);
        } else {
            const hackAmount = ns.getServerMaxMoney(target) * hackThreshold;
            serverHackThreads = Math.ceil(ns.hackAnalyzeThreads(target, hackAmount));

            serverGrowThreads = getGrowThreadsThreshold(ns, target, hackThreshold + THRESHOLD_STEP);
        }
        const firstWeakenThreads = getWeakenThreadsAfterHack(ns, serverHackThreads);

        const secondWeakenThreads = getWeakenThreadsAfterGrow(ns, serverGrowThreads);

        // this var describes the total amount of threads i need to run parallel mode
        const weaken1RamNeeded = RAM_WEAKEN * firstWeakenThreads;
        const weaken2RamNeeded = RAM_WEAKEN * secondWeakenThreads;
        const growRamNeeded = RAM_GROW * serverGrowThreads;
        const hackRamNeeded = RAM_HACK * serverHackThreads;

        const totalRamNeeded = weaken1RamNeeded + weaken2RamNeeded + growRamNeeded + hackRamNeeded;

        // log all
        ns.print("predicted threads needed:");
        ns.print("firstWeakenThreads: " + firstWeakenThreads + " with " + weaken1RamNeeded + "GB of RAM");
        ns.print("serverGrowThreads: " + serverGrowThreads + " with " + growRamNeeded + "GB of RAM");
        ns.print("secondWeakenThreads: " + secondWeakenThreads + " with " + weaken2RamNeeded + "GB of RAM");
        ns.print("serverHackThreads: " + serverHackThreads + " with " + hackRamNeeded + "GB of RAM");

        if (totalRamNeeded < totalMaxRam) {
            ns.print(
                "needs " + totalRamNeeded + "GB of RAM and got " + totalMaxRam + ". Running parallel mode on " + target,
            );
            return hackThreshold;
        }
        ns.print(
            Colors.YELLOW +
                "Not enough RAM to run parallel mode on " +
                target +
                ". Attempting to upgrade/buy server...",
        );

        if (ns.fileExists("Formulas.exe", "home")) {
            const hackServer = ServerManager.buyOrUpgradeServer(ns, hackRamNeeded, Config.HACK_SERVER_NAME);
            const Server = ServerManager.buyOrUpgradeServer(ns, growRamNeeded, Config.GROW_SERVER_NAME);
            const weaken1Server = ServerManager.buyOrUpgradeServer(ns, weaken1RamNeeded, Config.WEAK_SERVER_NAME);
            const weaken2Server = ServerManager.buyOrUpgradeServer(ns, weaken2RamNeeded, Config.WEAK_SERVER_NAME);

            if (hackServer !== "" && Server !== "" && weaken1Server !== "" && weaken2Server !== "") {
                ns.print(Colors.GREEN + "Servers bought, running parallel mode on " + target);
                return hackThreshold;
            }
        }

        hackThreshold = Math.round((hackThreshold - THRESHOLD_STEP) * 100) / 100;
        if (hackThreshold < Config.MIN_HACK_THRESHOLD) {
            ns.tprint(Colors.RED + "Error! Not enough RAM to run parallel mode on " + target);
            throw new Error("Error! Not enough RAM to run parallel mode on " + target);
        }
        ns.print("Threshold is too high, trying with: " + hackThreshold);
    }
}
