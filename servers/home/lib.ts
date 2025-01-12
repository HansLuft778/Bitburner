import { PlayerManager } from "./parallel/PlayerManager.js";

export enum Colors {
    RESET = "\x1b[0m",
    BLACK = "\x1b[30m",
    RED = "\x1b[31m",
    GREEN = "\x1b[32m",
    YELLOW = "\x1b[33m",
    BLUE = "\x1b[34m",
    MAGENTA = "\x1b[35m",
    CYAN = "\x1b[36m",
    WHITE = "\x1b[37m",
    E_ORANGE = "\x1b[38;5;178m",
}

export function serverScanner(ns: NS) {
    const uncheckedHosts = ["home"];
    const checkedHosts = [];

    for (let i = 0; i < uncheckedHosts.length; i++) {
        const newHosts = ns.scan(uncheckedHosts[i]);
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
            // TODO: getPurchasedServers() is not needed
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

export function nukeServer(ns: NS, server: string) {
    openPorts(ns, server);
    ns.nuke(server);

    ns.scp("hack.js", server);
    ns.scp("grow.js", server);
    ns.scp("weaken.js", server);
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

export function killWGH(ns: NS) {
    const servers = serverScanner(ns);

    for (const server of servers) {
        const processes = ns.ps(server);
        processes.forEach((p) => {
            if (p.filename == "weaken.js" || p.filename == "grow.js" || p.filename == "hack.js") {
                ns.kill(p.pid);
            }
        });
    }
}

export function getGrowThreads(ns: NS, server: string) {
    const serverMaxMoney = ns.getServerMaxMoney(server);
    const serverCurrentMoney = ns.getServerMoneyAvailable(server);
    let moneyMultiplier = serverMaxMoney / serverCurrentMoney;
    if (isNaN(moneyMultiplier) || moneyMultiplier == Infinity) moneyMultiplier = serverMaxMoney;
    const serverGrowThreads = Math.ceil(ns.growthAnalyze(server, moneyMultiplier) * 1.01);

    return serverGrowThreads;
}

export function getGrowThreadsThreshold(ns: NS, server: string, threshold: number) {
    const maxMoney = ns.getServerMaxMoney(server);
    const minMoney = maxMoney * (1 - threshold);
    const moneyMultiplier = maxMoney / minMoney;
    const serverGrowThreads = Math.ceil(ns.growthAnalyze(server, moneyMultiplier) * 1.01);
    ns.print(serverGrowThreads + " grows needed");

    return serverGrowThreads;
}

export function getWeakenThreadsAfterHack(ns: NS, numHackThreads: number): number {
    const hackSecLvlIncrease = ns.hackAnalyzeSecurity(numHackThreads);

    const serverWeakenThreads = Math.ceil(hackSecLvlIncrease / ns.weakenAnalyze(1));

    return serverWeakenThreads;
}

export function getWeakenThreadsAfterGrow(ns: NS, numGrowThreads: number): number {
    const hackSecLvlIncrease = ns.growthAnalyzeSecurity(numGrowThreads);

    const serverWeakenThreads = Math.ceil(hackSecLvlIncrease / ns.weakenAnalyze(1));

    return serverWeakenThreads;
}

export function getWeakenThreads(ns: NS, server: string) {
    const serverSecLvl = ns.getServerSecurityLevel(server);
    const serverWeakenThreads = Math.ceil((serverSecLvl - ns.getServerMinSecurityLevel(server)) / ns.weakenAnalyze(1));

    return serverWeakenThreads;
}

export function getHackThreads(ns: NS, server: string, moneyHackThreshold: number) {
    const serverMaxMoney = ns.getServerMaxMoney(server);
    const lowerMoneyBound = serverMaxMoney * (1 - moneyHackThreshold);
    const hackAmount = serverMaxMoney - lowerMoneyBound;
    const serverHackThreads = Math.floor(ns.hackAnalyzeThreads(server, hackAmount));

    return serverHackThreads;
}

// ----------------- FormulasAPI -----------------
export function getGrowThreadsFormulas(ns: NS, server: string, hackThreads: number) {
    const serverObject = ns.getServer(server);
    const playerObject = PlayerManager.getInstance(ns).getPlayer();

    if (serverObject.moneyMax == undefined) return 0;

    serverObject.moneyAvailable = serverObject.moneyMax;
    serverObject.baseDifficulty = serverObject.minDifficulty;

    const percentPerHack = ns.formulas.hacking.hackPercent(serverObject, playerObject);
    const hackPercent = percentPerHack * hackThreads;

    serverObject.moneyAvailable = serverObject.moneyMax * (1 - hackPercent);
    serverObject.baseDifficulty = serverObject.minDifficulty;

    return Math.ceil(ns.formulas.hacking.growThreads(serverObject, playerObject, serverObject.moneyMax) * 1.01);
}

export function getHackThreadsFormulas(ns: NS, server: string, hackThreshold: number) {
    const serverObject = ns.getServer(server);
    const playerObject = PlayerManager.getInstance(ns).getPlayer();

    serverObject.baseDifficulty = serverObject.minDifficulty;
    serverObject.moneyAvailable = serverObject.moneyMax;

    // threads * percent == hackThreshold => threads == hackThreshold / percent
    return Math.floor(hackThreshold / ns.formulas.hacking.hackPercent(serverObject, playerObject));
}

// ----------------- Ports -----------------
export function writeToPort(ns: NS, port: number, data: string) {
    const currentData = ns.peek(port);
    if (currentData == "NULL PORT DATA") {
        ns.writePort(port, data);
    } else {
        ns.writePort(port, data);
        ns.readPort(port);
    }
}

export function isPreparationNeeded(ns: NS, target: string) {
    if (
        ns.getServerMaxMoney(target) != parseFloat(ns.getServerMoneyAvailable(target).toFixed(5)) ||
        parseFloat(ns.getServerSecurityLevel(target).toFixed(5)) != ns.getServerMinSecurityLevel(target)
    ) {
        return true;
    } else {
        return false;
    }
}

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
}
