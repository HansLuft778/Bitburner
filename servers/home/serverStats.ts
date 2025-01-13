import { Colors } from "./lib.js";
import { getBestHostByRamOptimized } from "./bestServer.js";
import { Config } from "./Config/Config.js";

const BORDER_COLOR = Colors.CYAN;

let maxMoney = 0;
let curMoney = 0;
let hackingChance = 0;
let minSec = 0;
let curSec = 0;
let maxRam = 0;
let useRam = 0;
let freeRam = 0;
let moneyMultiplier = 0;
let growingThreads = 0;
let serverWeakenThreadsCur = 0;
let hackThreads = 0;
let headerString = "";
let footerString = "";
let hackingPercent = 0;

export function printServerStats(ns: NS, server: string, hackThreshold: number) {
    setStats(ns, server, hackThreshold);

    ns.print(BORDER_COLOR + headerString + Colors.RESET);

    printStatLine(ns, "Money:", false);
    printStatLine(ns, "Max Money: " + ns.formatNumber(maxMoney) + " | Current Money: " + ns.formatNumber(curMoney));
    printStatLine(
        ns,
        "Percent: " + ns.formatNumber(curMoney / maxMoney) + " | Hack Chance: " + ns.formatNumber(hackingChance),
    );

    printStatLine(ns, "Security:", false);
    printStatLine(ns, "Min Seclvl: " + minSec + " | Current Seclvl: " + ns.formatNumber(curSec));

    printStatLine(ns, "Ram:", false);
    printStatLine(ns, "Server Max Ram: " + maxRam);
    printStatLine(ns, "Used Ram: " + useRam + " | free Ram: " + freeRam);

    printStatLine(ns, "Threads:", false);
    printStatLine(ns, "Grow Threads: " + growingThreads);
    printStatLine(ns, "Weaken Threads " + serverWeakenThreadsCur);
    printStatLine(ns, "Hack Threads: " + hackThreads + " | Hack percent: " + ns.formatNumber(hackingPercent, 5));

    ns.print(BORDER_COLOR + footerString + Colors.RESET);

    return footerString.length;
}

export function printServerStatsConsole() {
    // todo
}

function setStats(ns: NS, server: string, hackThreshold: number) {
    // money
    maxMoney = ns.getServerMaxMoney(server);
    curMoney = ns.getServerMoneyAvailable(server);
    hackingChance = ns.hackAnalyzeChance(server);
    // sec lvl
    minSec = ns.getServerMinSecurityLevel(server);
    curSec = ns.getServerSecurityLevel(server);
    // ram
    maxRam = ns.getServerMaxRam(server);
    useRam = ns.getServerUsedRam(server);
    freeRam = maxRam - useRam;
    // threads

    moneyMultiplier = maxMoney / curMoney;
    if (isNaN(moneyMultiplier) || moneyMultiplier == Infinity) moneyMultiplier = 1;

    growingThreads = Math.ceil(ns.growthAnalyze(server, moneyMultiplier));

    serverWeakenThreadsCur = Math.ceil((curSec - ns.getServerMinSecurityLevel(server)) / 0.05);

    hackingPercent = ns.hackAnalyze(server);
    hackThreads = Math.ceil(hackThreshold / hackingPercent);
    if (isNaN(hackThreads) || hackThreads == Infinity) hackThreads = 0;
    if (isNaN(hackingPercent) || hackingPercent == Infinity) hackingPercent = 0;

    headerString = "┌───────────────── stats for " + server + " ─────────────────┐";
    footerString = "└" + "─".repeat(headerString.length - 2) + "┘";
}

function printStatLine(ns: NS, value: string, indent = true) {
    if (indent) value = "\t" + value;
    const offset = indent ? 8 : 3; // the offset to subtract the border and indent
    ns.print(
        BORDER_COLOR +
            "│ " +
            Colors.RESET +
            value +
            " ".repeat(headerString.length - value.length - offset) +
            BORDER_COLOR +
            "│" +
            Colors.RESET,
    );
}

export function getNumThreadsActive(ns: NS) {
    const hosts = getBestHostByRamOptimized(ns);

    let totalThreads = 0;
    for (let i = 0; i < hosts.length; i++) {
        const host = hosts[i];

        const processes = ns.ps(host.name);

        processes.forEach((process) => {
            totalThreads += process.threads;
        });
    }
    return totalThreads;
}

interface AutocompleteData {
    servers: string[];
    txts: string[];
    scripts: string[];
    flags: string[];
}

export function autocomplete(data: AutocompleteData) {
    return [...data.servers];
}

export async function main(ns: NS) {
    ns.clearLog();
    ns.tail();
    ns.disableLog("ALL");
    // while (true) {
    //     ns.clearLog();
    //     ns.print(getNumThreadsActive(ns));
    //     await ns.sleep(1000);
    // }
    if (ns.args.length == 1) {
        while (true) {
            ns.clearLog();
            const width = printServerStats(ns, ns.args[0].toString(), 0.9);
            ns.resizeTail((width - 1) * 10, 375);
            await ns.sleep(100);
        }
    } else {
        while (true) {
            ns.clearLog();
            const server = ns.peek(1);
            if (server === "NULL PORT DATA") {
                ns.print("No server found, waiting for data...");
                await ns.getPortHandle(1).nextWrite();
                continue;
            }
            const width = printServerStats(ns, server.toString(), 0.9);
            ns.resizeTail((width - 1) * 11, 375);
            await ns.sleep(Config.DELAY_MARGIN_MS);
        }
    }
}
