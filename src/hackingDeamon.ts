import { NS } from "@ns";

import { Config } from "./Config/Config";
import { getBestHostByRamOptimized, getBestServer } from "./bestServer";
import {
    Colors,
    getGrowThreadsFormulas,
    getGrowThreadsThreshold,
    getHackThreadsFormulas,
    getWeakenThreadsAfterGrow,
    getWeakenThreadsAfterHack,
    isPreparationNeeded,
    killWGH,
    writeToPort,
} from "./lib";
import { prepareServer } from "./loop/prepareServer";
import { PlayerManager } from "./parallel/PlayerManager";
import { parallelCycle } from "./parallel/manager";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    // kill all active WGH scripts
    await ns.sleep(1000);
    killWGH(ns);

    ns.getPortHandle(2).clear();

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

        // start tea/party script if player has corp
        const teaPartyRunning: boolean = ns.ps().find((p) => p.filename === "Corporation/teaParty.js") !== undefined;
        if (ns.corporation.hasCorporation() && !teaPartyRunning) ns.exec("Corporation/teaParty.js", "home");

        PlayerManager.getInstance(ns).resetPlayer(ns);

        const target = ns.args[0] === undefined ? getBestServer(ns) : ns.args[0].toString();

        writeToPort(ns, 1, target);
        ns.print("lastTarget: " + lastTarget + " target: " + target);
        if (ns.fileExists("Formulas.exe", "home")) {
            if (lastTarget !== target || isPreparationNeeded(ns, target)) {
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
    let hackThreshold = 0.7;

    if (hackThreshold < Config.MIN_HACK_THRESHOLD) throw new Error("Hack threshold is too low");

    ns.print(Colors.E_ORANGE + "Starting hack threshold: " + hackThreshold);

    while (true) {
        let serverHackThreads = 0;
        let serverGrowThreads = 0;
        if (ns.fileExists("Formulas.exe", "home")) {
            PlayerManager.getInstance(ns).resetPlayer(ns);
            serverHackThreads = getHackThreadsFormulas(ns, target, hackThreshold);
            PlayerManager.getInstance(ns).addHackingExp(ns, target, serverHackThreads);
            serverGrowThreads = getGrowThreadsFormulas(ns, target, serverHackThreads);
            PlayerManager.getInstance(ns).addHackingExp(ns, target, serverGrowThreads);
            PlayerManager.getInstance(ns).resetPlayer(ns);
        } else {
            const hackAmount = ns.getServerMaxMoney(target) * hackThreshold;
            serverHackThreads = Math.ceil(ns.hackAnalyzeThreads(target, hackAmount));

            serverGrowThreads = getGrowThreadsThreshold(ns, target, hackThreshold + THRESHOLD_STEP);
        }
        const firstWeakenThreads = getWeakenThreadsAfterHack(ns, serverHackThreads);

        const secondWeakenThreads = getWeakenThreadsAfterGrow(ns, serverGrowThreads);

        ns.print(
            "serverHackThreads: " +
                serverHackThreads +
                " serverGrowThreads: " +
                serverGrowThreads +
                " firstWeakenThreads: " +
                firstWeakenThreads +
                " secondWeakenThreads: " +
                secondWeakenThreads,
        );

        const allHosts = getBestHostByRamOptimized(ns);
        // ----------------- Simulate hack -----------------
        let hackDeployed = false;
        for (let i = 0; i < allHosts.length; i++) {
            // const host = allHosts[i];
            if (
                allHosts[i].name.includes(Config.WEAK_SERVER_NAME) ||
                allHosts[i].name.includes(Config.GROW_SERVER_NAME)
            )
                continue;

            const maxThreadsOnHost = Math.floor(allHosts[i].availableRam / hackingScriptRam);

            if (maxThreadsOnHost >= serverHackThreads) {
                allHosts[i].availableRam -= serverHackThreads * hackingScriptRam;
                hackDeployed = true;
                break;
            }
        }

        if (!hackDeployed) {
            hackThreshold = parseFloat((hackThreshold - THRESHOLD_STEP).toFixed(2));
            ns.print("Threshold too high, decreasing to: " + hackThreshold);
            continue;
        }

        // ----------------- Simulate weak1 -----------------

        let threadsDispatched = 0;
        let threadsRemaining = firstWeakenThreads;
        for (let i = 0; i < allHosts.length; i++) {
            if (threadsDispatched >= firstWeakenThreads) break;
            // const host = allHosts[i];
            if (
                allHosts[i].name.includes(Config.GROW_SERVER_NAME) ||
                allHosts[i].name.includes(Config.HACK_SERVER_NAME)
            )
                continue;

            const freeRam = allHosts[i].availableRam;
            if (freeRam < weakenScriptRam) continue;
            const threadSpace = Math.floor(freeRam / weakenScriptRam);

            const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

            // simulate weaken
            allHosts[i].availableRam -= threadsToDispatch * weakenScriptRam;
            threadsRemaining -= threadsToDispatch;
            threadsDispatched += threadsToDispatch;
        }

        if (threadsRemaining > 0) {
            hackThreshold = parseFloat((hackThreshold - THRESHOLD_STEP).toFixed(2));
            ns.print("Threshold too high, decreasing to: " + hackThreshold);
            continue;
        }

        // ----------------- Simulate grow -----------------
        let growDeployed = false;
        for (let i = 0; i < allHosts.length; i++) {
            // const host = allHosts[i];
            if (
                allHosts[i].name.includes(Config.WEAK_SERVER_NAME) ||
                allHosts[i].name.includes(Config.HACK_SERVER_NAME)
            )
                continue;

            const maxThreadsOnHost = Math.floor(allHosts[i].availableRam / growingScriptRam);
            // ns.print("maxThreadsOnHost: " + maxThreadsOnHost + " serverGrowThreads: " + serverGrowThreads);
            if (maxThreadsOnHost >= serverGrowThreads) {
                allHosts[i].availableRam -= serverGrowThreads * growingScriptRam;
                growDeployed = true;
                // ns.print("HERE");
            }
        }

        if (!growDeployed) {
            hackThreshold = parseFloat((hackThreshold - THRESHOLD_STEP).toFixed(2));
            ns.print("Threshold too high, decreasing to: " + hackThreshold);
            continue;
        }
        // ----------------- Simulate weak2 -----------------

        threadsDispatched = 0;
        threadsRemaining = secondWeakenThreads;
        for (let i = 0; i < allHosts.length; i++) {
            if (threadsDispatched >= secondWeakenThreads) break;
            // const host = allHosts[i];
            if (
                allHosts[i].name.includes(Config.GROW_SERVER_NAME) ||
                allHosts[i].name.includes(Config.HACK_SERVER_NAME)
            )
                continue;

            const freeRam = allHosts[i].availableRam;
            if (freeRam < weakenScriptRam) continue;
            const threadSpace = Math.floor(freeRam / weakenScriptRam);

            const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

            // simulate weaken
            allHosts[i].availableRam -= threadsToDispatch * weakenScriptRam;
            threadsRemaining -= threadsToDispatch;
            threadsDispatched += threadsToDispatch;
        }

        if (threadsRemaining > 0) {
            hackThreshold = parseFloat((hackThreshold - THRESHOLD_STEP).toFixed(2));
            ns.print("Threshold too high, decreasing to: " + hackThreshold);
            continue;
        }

        ns.print(Colors.GREEN + "All simulations passed, hackThreshold: " + hackThreshold);

        ns.print(allHosts.map((host) => host.name + " " + host.availableRam).join("\n"));

        return hackThreshold;
    }
}

export async function prepare(ns: NS, target: string, hackThreshold: number) {
    if (isPreparationNeeded(ns, target)) {
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
