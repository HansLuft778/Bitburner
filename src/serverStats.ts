import { NS } from "@ns";
import { Colors } from "./lib";

let maxMoney = 0;
let curMoney = 0;
let hackingChance = 0;
let minSec = 0;
let curSec = 0;
let maxRam = 0;
let useRam = 0;
let freeRam = 0;
let moneyMult = 0;
let growingThreads = 0;
let serverWeakenThreadsCur = 0;
let lowerMoneyBound = 0;
let hackAmount = 0;
let hackThreads = 0;
let headerString = "";
let footerString = "";
let hackingPercent = 0;

export async function main(ns: NS) {
    ns.clearLog();
    ns.tail();
    ns.disableLog("ALL");
    await printServerStatsLive(ns, ns.args[0].toString(), 0.9);
}

export function printServerStats(ns: NS, server: string, hackThreshold: number) {
    setStats(ns, server, hackThreshold);

    ns.print(Colors.cyan + headerString + Colors.reset);

    ns.print("Money:");
    ns.print("\tMax Money: " + ns.formatNumber(maxMoney) + " | Current Money: " + ns.formatNumber(curMoney));
    ns.print("\tHack Chance: " + ns.formatNumber(hackingChance));

    ns.print("Security:");
    ns.print("\tMin Seclvl: " + minSec + " | Current Seclvl: " + ns.formatNumber(curSec));

    ns.print("Ram:");
    ns.print("\tServer Max Ram: " + maxRam);
    ns.print("\tUsed Ram: " + useRam + " | free Ram: " + freeRam);

    ns.print("Threads:");
    ns.print("\tGrow Threads: " + growingThreads);
    ns.print("\tWeaken Threads " + serverWeakenThreadsCur);

    ns.print(
        "\tHack Threads: " + ns.formatNumber(hackThreads) + " | Hack percent: " + ns.formatNumber(hackingPercent, 5),
    );

    ns.print(Colors.cyan + footerString + Colors.reset);
}

export function printServerStatsConsole(ns: NS, server: string) {
    // todo
}

async function printServerStatsLive(ns: NS, server: string, hackThreshold: number) {
    while (true) {
        setStats(ns, server, hackThreshold);

        ns.print(Colors.cyan + headerString + Colors.reset);

        ns.print("Money:");
        ns.print("\tMax Money: " + ns.formatNumber(maxMoney) + " | Current Money: " + ns.formatNumber(curMoney));
        ns.print("\tHack Chance: " + ns.formatNumber(hackingChance));

        ns.print("Security:");
        ns.print("\tMin Seclvl: " + minSec + " | Current Seclvl: " + ns.formatNumber(curSec));

        ns.print("Ram:");
        ns.print("\tServer Max Ram: " + maxRam);
        ns.print("\tUsed Ram: " + useRam + " | free Ram: " + freeRam);

        ns.print("Threads:");
        ns.print("\tGrow Threads: " + growingThreads);
        ns.print("\tWeaken Threads " + serverWeakenThreadsCur);

        ns.print(
            "\tHack Threads: " +
                ns.formatNumber(hackThreads) +
                " | Hack percent: " +
                ns.formatNumber(hackingPercent, 5),
        );

        ns.print(Colors.cyan + footerString + Colors.reset);
        await ns.sleep(200);
        ns.clearLog();
    }
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

    moneyMult = maxMoney / curMoney;
    if (isNaN(moneyMult) || moneyMult == Infinity) moneyMult = 1;

    growingThreads = Math.ceil(ns.growthAnalyze(server, moneyMult));

    serverWeakenThreadsCur = Math.ceil((curSec - ns.getServerMinSecurityLevel(server)) / 0.05);

    lowerMoneyBound = maxMoney * hackThreshold;
    hackAmount = maxMoney - lowerMoneyBound;
    hackingPercent = ns.hackAnalyze(server);
    hackThreads = Math.ceil(hackThreshold / hackingPercent);
    if (isNaN(hackThreads) || hackThreads == Infinity) hackThreads = 0;
    if (isNaN(hackingPercent) || hackingPercent == Infinity) hackingPercent = 0;

    headerString = "----------------- stats for " + server + " -----------------";
    footerString = buildFooterString(headerString.length);
}

function buildFooterString(len: number) {
    let footerStr = "";
    for (let i = 0; i < len; i++) {
        footerStr += "-";
    }
    return footerStr;
}
