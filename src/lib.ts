import { NS } from "@ns";

export enum Colors {
    reset = "\x1b[0m",
    black = "\x1b[30m",
    red = "\x1b[31m",
    green = "\x1b[32m",
    yellow = "\x1b[33m",
    blue = "\x1b[34m",
    magenta = "\x1b[35m",
    cyan = "\x1b[36m",
    white = "\x1b[37m",
}

export function serverScanner(ns: NS) {
    let uncheckedHosts = ["home"];
    let checkedHosts = [];

    for (let i = 0; i < uncheckedHosts.length; i++) {
        let newHosts = ns.scan(uncheckedHosts[i]);
        checkedHosts.push(uncheckedHosts[i]);

        for (let n = 0; n < newHosts.length; n++) {
            if (checkedHosts.includes(newHosts[n]) == false || uncheckedHosts.includes(newHosts[n]) == false) {
                uncheckedHosts.push(newHosts[n]);
            }
        }
    }

    return checkedHosts.sort();
}

export function isHackable(ns: NS, server: string) {
    if (
        ns.getServerNumPortsRequired(server) <= getNumHacks(ns) &&
        ns.getServerRequiredHackingLevel(server) <= ns.getHackingLevel()
    )
        return true;
    else return false;
}

export function getNumHacks(ns: NS) {
    let i = 0;
    if (ns.fileExists("BruteSSH.exe")) i++;
    if (ns.fileExists("FTPCrack.exe")) i++;
    if (ns.fileExists("HTTPWorm.exe")) i++;
    if (ns.fileExists("SQLInject.exe")) i++;
    return i;
}

export function nukeAll(ns: NS) {
    const hosts = serverScanner(ns);
    for (let i = 0; i < hosts.length; i++) {
        // check if the host is hackable
        if (isHackable(ns, hosts[i]) || ns.getPurchasedServers().includes(hosts[i])) {
            openPorts(ns, hosts[i]);
            ns.nuke(hosts[i]);

            // copy all scripts to the server
            ns.scp("hack.js", hosts[i]);
            ns.scp("grow.js", hosts[i]);
            ns.scp("weaken.js", hosts[i]);
        } else {
            continue;
        }
    }
}

export function openPorts(ns: NS, target: string) {
    if (ns.fileExists("BruteSSH.exe")) ns.brutessh(target);
    if (ns.fileExists("FTPCrack.exe")) ns.ftpcrack(target);
    if (ns.fileExists("HTTPWorm.exe")) ns.httpworm(target);
    if (ns.fileExists("SQLInject.exe")) ns.sqlinject(target);
}

export function getTimeH(timestamp?: number) {
    if (timestamp == undefined || timestamp == null) timestamp = Date.now();

    const date = new Date(timestamp);
    date.setUTCHours(date.getUTCHours() + 1);
    const hours = date.getUTCHours().toString().padStart(2, "0");
    const minutes = date.getUTCMinutes().toString().padStart(2, "0");
    const seconds = date.getUTCSeconds().toString().padStart(2, "0");
    const milliseconds = date.getUTCMilliseconds().toString().padStart(3, "0");
    const formattedTime = `${hours}:${minutes}:${seconds}:${milliseconds}`;
    return formattedTime;
}

export function getGrowThreads(ns: NS, server: string) {
    const serverMaxMoney = ns.getServerMaxMoney(server);
    const serverCurrentMoney = ns.getServerMoneyAvailable(server);
    let moneyMultiplier = serverMaxMoney / serverCurrentMoney;
    if (isNaN(moneyMultiplier) || moneyMultiplier == Infinity) moneyMultiplier = 1;
    const serverGrowThreads = Math.ceil(ns.growthAnalyze(server, moneyMultiplier));

    return serverGrowThreads;
}

export function getGrowThreadsThreshold(ns: NS, server: string, threshold: number) {
    const maxMoney = ns.getServerMaxMoney(server);
    const minMoney = maxMoney * (1 - threshold);
    const moneyMultiplier = maxMoney / minMoney;
    const serverGrowThreads = Math.ceil(ns.growthAnalyze(server, moneyMultiplier));

    return serverGrowThreads;
}

export function getWeakenThreads(ns: NS, server: string) {
    const growThreads = getGrowThreads(ns, server);
    const secIncrease = ns.growthAnalyzeSecurity(growThreads, server);
    const serverWeakenThreads = Math.ceil(secIncrease / 0.05);

    return serverWeakenThreads;
}

export function getWeakenThreadsEff(ns: NS, server: string) {
    const serverSecLvl = ns.getServerSecurityLevel(server);
    const serverWeakenThreads = Math.ceil((serverSecLvl - ns.getServerMinSecurityLevel(server)) / ns.weakenAnalyze(1));

    return serverWeakenThreads;
}

export function getHackThreads(ns: NS, server: string, moneyHackThreshold: number) {
    const serverMaxMoney = ns.getServerMaxMoney(server);
    const lowerMoneyBound = serverMaxMoney * (1 - moneyHackThreshold);
    const hackAmount = serverMaxMoney - lowerMoneyBound;
    const serverHackThreads = Math.ceil(ns.hackAnalyzeThreads(server, hackAmount));

    return serverHackThreads;
}

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    const server = "silver-helix";
    ns.print(getWeakenThreadsEff(ns, server) + " weakens needed");
    ns.print(getGrowThreads(ns, server) + " grows needed");
    ns.print(getHackThreads(ns, server, 0.9) + " hacks needed");
}
