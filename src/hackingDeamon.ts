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
import { parallelCycle } from "./parallel/manager";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    let hackThreshold = 0.5;
    let lastTarget = "";

    while (true) {
        // start stock manager if player has WSE account
        const stockManagerRunning: boolean = ns.ps().find((p) => p.filename === "Stock/manager.js") !== undefined;
        if (
            ns.stock.hasWSEAccount() &&
            ns.stock.has4SData() &&
            ns.stock.has4SDataTIXAPI() &&
            ns.stock.hasTIXAPIAccess() &&
            ns.getServerMoneyAvailable("home") > Config.STOCK_MARKET_MIN_HOME_MONEY &&
            !stockManagerRunning
        )
            ns.exec("Stock/manager.js", "home");

        PlayerManager.getInstance(ns).resetPlayer(ns);

        const target = ns.args[0] === undefined ? getBestServer(ns) : ns.args[0].toString();

        writeToPort(ns, 1, target);
        ns.print("lastTarget: " + lastTarget + " target: " + target);
        if (ns.fileExists("Formulas.exe", "home")) {
            if (lastTarget !== target) {
                // ----------------- PREPARE SERVER -----------------
                hackThreshold = await prepare(ns, target, hackThreshold);
                lastTarget = target;
            }

            // ----------------- CHECK WHICH MODE TO USE -----------------
            PlayerManager.getInstance(ns).resetPlayer(ns);
            await parallelCycle(ns, target, hackThreshold, Config.LOOP_BATCH_COUNT);
        } else {
            if (lastTarget !== target) {
                // ----------------- PREPARE SERVER -----------------

                hackThreshold = await prepare(ns, target, hackThreshold);
                lastTarget = target;
            }
            // ----------------- CHECK WHICH MODE TO USE -----------------

            await parallelCycle(ns, target, hackThreshold);
        }
    }
}

function getHackThreshold(ns: NS, target: string) {
    const hackingScriptRam = Config.HACK_SCRIPT_RAM;
    const weakenScriptRam = Config.WEAKEN_SCRIPT_RAM;
    const growingScriptRam = Config.GROW_SCRIPT_RAM;

    const THRESHOLD_STEP = Config.THRESHOLD_STEP;
    let hackThreshold = 0.9;

    if (hackThreshold < Config.MIN_HACK_THRESHOLD) throw new Error("Hack threshold is too low");

    ns.print(Colors.E_ORANGE + "Starting hack threshold: " + hackThreshold);

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

        const allHosts = getBestHostByRamOptimized(ns);
        // ----------------- Simulate hack -----------------
        let hackDeployed = false;
        for (let i = 0; i < allHosts.length; i++) {
            const host = allHosts[i];
            if (host.name.includes(Config.WEAK_SERVER_NAME) || host.name.includes(Config.GROW_SERVER_NAME)) continue;

            const maxThreadsOnHost = Math.floor(host.availableRam / hackingScriptRam);

            if (maxThreadsOnHost >= serverHackThreads) {
                host.availableRam -= serverHackThreads * hackingScriptRam;
                hackDeployed = true;
                break;
            }
        }

        if (!hackDeployed) {
            hackThreshold -= THRESHOLD_STEP;
            ns.print("Threshold too high, decreasing to: " + hackThreshold);
            continue;
        }

        // ----------------- Simulate weak1 -----------------

        let threadsDispatched = 0;
        let threadsRemaining = firstWeakenThreads;
        for (let i = 0; i < allHosts.length; i++) {
            if (threadsDispatched >= firstWeakenThreads) break;
            const host = allHosts[i];
            if (host.name.includes(Config.GROW_SERVER_NAME) || host.name.includes(Config.HACK_SERVER_NAME)) continue;

            const freeRam = host.availableRam;
            if (freeRam < weakenScriptRam) continue;
            const threadSpace = Math.floor(freeRam / weakenScriptRam);

            const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

            // simulate weaken
            host.availableRam -= threadsToDispatch * weakenScriptRam;
            threadsRemaining -= threadsToDispatch;
            threadsDispatched += threadsToDispatch;
        }

        if (threadsRemaining > 0) {
            hackThreshold -= THRESHOLD_STEP;
            ns.print("Threshold too high, decreasing to: " + hackThreshold);
            continue;
        }

        // ----------------- Simulate grow -----------------
        let growDeployed = false;
        for (let i = 0; i < allHosts.length; i++) {
            const host = allHosts[i];
            if (host.name.includes(Config.WEAK_SERVER_NAME) || host.name.includes(Config.HACK_SERVER_NAME)) continue;

            const maxThreadsOnHost = Math.floor(host.availableRam / growingScriptRam);

            if (maxThreadsOnHost >= serverGrowThreads) {
                host.availableRam -= serverGrowThreads * growingScriptRam;
                growDeployed = true;
            }
        }

        if (!growDeployed) {
            hackThreshold -= THRESHOLD_STEP;
            ns.print("Threshold too high, decreasing to: " + hackThreshold);
            continue;
        }
        // ----------------- Simulate weak2 -----------------

        threadsDispatched = 0;
        threadsRemaining = secondWeakenThreads;
        for (let i = 0; i < allHosts.length; i++) {
            if (threadsDispatched >= secondWeakenThreads) break;
            const host = allHosts[i];
            if (host.name.includes(Config.GROW_SERVER_NAME) || host.name.includes(Config.HACK_SERVER_NAME)) continue;

            const freeRam = host.availableRam;
            if (freeRam < weakenScriptRam) continue;
            const threadSpace = Math.floor(freeRam / weakenScriptRam);

            const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

            // simulate weaken
            host.availableRam -= threadsToDispatch * weakenScriptRam;
            threadsRemaining -= threadsToDispatch;
            threadsDispatched += threadsToDispatch;
        }

        if (threadsRemaining > 0) {
            hackThreshold -= THRESHOLD_STEP;
            ns.print("Threshold too high, decreasing to: " + hackThreshold);
            continue;
        }

        ns.print(Colors.GREEN + "All simulations passed, hackThreshold: " + hackThreshold);

        return hackThreshold;
    }
}

export async function prepare(ns: NS, target: string, hackThreshold: number) {
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

    return hackThreshold;
}
