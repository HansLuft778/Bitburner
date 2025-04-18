
import { Config } from "./Config/Config.js";
import { isHackable, nukeServer, serverScanner } from "./lib.js";

export interface Server {
    name: string;
    maxMoney?: number;
    hackingChance?: number;
    weakeningTime?: number;
    maxRam: number;
    availableRam: number;
    score: number;
}

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    getBestServerList(ns, true);
}

export function getBestServerList(ns: NS, shouldPrint: boolean) {
    const serverList = serverScanner(ns);

    const servers: Server[] = [];

    for (let i = 0; i < serverList.length; i++) {
        const serverName = serverList[i];
        if (!isHackable(ns, serverName) && !ns.hasRootAccess(serverName)) continue;

        nukeServer(ns, serverName); // 50 ms for 500 itterations

        const maxMoney = ns.getServerMaxMoney(serverName);
        const hackingChance = ns.hackAnalyzeChance(serverName);
        let weakeningTime = ns.getWeakenTime(serverName);
        const maxRam = ns.getServerMaxRam(serverName);

        // filter server with no money or the hacking level above players hacking level
        if (maxMoney < 1 || ns.getServerRequiredHackingLevel(serverName) > ns.getHackingLevel()) continue;

        // const score = (maxMoney / (weakeningTime + 3)) * hackingChance * (1 / weakeningTime)
        // const score = ns.formatNumber(((maxMoney / (weakeningTime)) * hackingChance) / 1000)

        let score = maxMoney / ns.getServerMinSecurityLevel(serverName) / 1000000;

        if (ns.fileExists("formulas.exe", "home")) {
            const server = ns.getServer(serverName);
            const player = ns.getPlayer();
            server.hackDifficulty = server.minDifficulty;
            const maxMoney = server.moneyMax == undefined ? 0 : server.moneyMax;
            weakeningTime = ns.formulas.hacking.weakenTime(server, player);
            score = ((maxMoney / weakeningTime) * ns.formulas.hacking.hackChance(server, player)) / 1000;
        } else {
            const serverLvl = ns.getServerRequiredHackingLevel(serverName);
            const playerLvl = ns.getHackingLevel();
            if (playerLvl / serverLvl < 2) score = 0;
        }

        const server: Server = {
            name: serverName,
            maxMoney: maxMoney,
            hackingChance: hackingChance,
            weakeningTime: weakeningTime,
            maxRam: maxRam,
            availableRam: maxRam - ns.getServerUsedRam(serverName),
            score: score,
        };

        servers.push(server);
    }

    servers.sort((a, b) => {
        return (b.score || 0) - (a.score || 0);
    });

    if (shouldPrint) printTable(ns, servers);

    return servers;
}

export function getBestServer(ns: NS): string {
    const best = getBestServerList(ns, false)[0].name;
    if (Config.TARGET !== "") return Config.TARGET;
    return best;
}

export function getBestHostByRamOptimized(ns: NS): Server[] {
    const allHosts: Server[] = [];
    const allServers = serverScanner(ns);
    // let homeIdx = -1;
    // let home: Server | undefined = undefined;

    for (let i = 0; i < allServers.length; i++) {
        const server = ns.getServer(allServers[i]);

        if (server.maxRam - server.ramUsed < 2) continue;
        if (!server.hasAdminRights) continue;

        const serverObj: Server = {
            name: server.hostname,
            maxRam: server.maxRam,
            availableRam: server.maxRam - server.ramUsed,
            score: 0,
        };
        if (serverObj.name === "home") {
            serverObj.availableRam -= Config.HOME_FREE_RAM;
            // homeIdx = i;
            // home = serverObj;
        }
        allHosts.push(serverObj);
    }

    // sort by ram in ascending order
    allHosts.sort((a, b) => {
        return a.availableRam - b.availableRam;
    });

    // if (home !== undefined && homeIdx !== -1) {
    //     allHosts.splice(homeIdx, 1);
    //     allHosts.unshift(home);
    // }

    return allHosts;
}

/**
 * @deprecated use getBestHostByRamOptimized instead
 */
export function getBestHostByRam(ns: NS): Server[] {
    const allHosts = getBestServerListCheap(ns, false).filter((server) => {
        return server.availableRam > 2;
    });

    const home: Server = {
        name: "home",
        maxRam: ns.getServerMaxRam("home") - Config.HOME_FREE_RAM,
        availableRam: ns.getServerMaxRam("home") - ns.getServerUsedRam("home") - Config.HOME_FREE_RAM,
        score: 0,
    }; // 10 some safety margin
    allHosts.push(home);

    const purchasedServers = ns.getPurchasedServers();
    for (let i = 0; i < purchasedServers.length; i++) {
        const server: Server = {
            name: purchasedServers[i],
            maxRam: ns.getServerMaxRam(purchasedServers[i]),
            availableRam: ns.getServerMaxRam(purchasedServers[i]) - ns.getServerUsedRam(purchasedServers[i]),
            score: 0,
        };
        if (server.maxRam > 2) {
            allHosts.push(server);
        }
    }

    // sort by ram in ascending order
    allHosts.sort((a, b) => {
        return a.availableRam - b.availableRam;
    });

    return allHosts;
}

export function getBestServerListCheap(ns: NS, shouldPrint: boolean): Server[] {
    const serverList = serverScanner(ns);

    const servers: Server[] = [];

    for (let i = 0; i < serverList.length; i++) {
        const serverName = serverList[i];
        // const so = ns.getServer(serverName);
        // const player = ns.getPlayer();

        if (!isHackable(ns, serverName)) continue;

        // const canOpenPorts = so.numOpenPortsRequired <= getNumHacks(ns);
        // const canHack = so.requiredHackingSkill <= player.skills.hacking;
        // if (!(canOpenPorts && canHack)) continue;

        const maxMoney = ns.getServerMaxMoney(serverName);
        const maxRam = ns.getServerMaxRam(serverName);

        // filter server with no money or the hacking level above players hacking level
        if (maxMoney < 1 || ns.getServerRequiredHackingLevel(serverList[i]) > ns.getHackingLevel()) continue;

        const score = maxMoney / ns.getServerMinSecurityLevel(serverName) / 1000000;

        const server: Server = {
            name: serverName,
            maxMoney: maxMoney,
            maxRam: maxRam,
            availableRam: maxRam - ns.getServerUsedRam(serverName),
            score: score,
        };
        servers.push(server);
    }

    servers.sort((a, b) => {
        return (b.score || 0) - (a.score || 0);
    });

    if (shouldPrint) printTable(ns, servers);

    return servers;
}

export function printTable(ns: NS, array: Server[]) {
    // sanity check + number formatting
    interface TableServer {
        name: string;
        maxMoney: string;
        hackingChance: string;
        weakeningTime: string;
        score: string;
    }
    const tableArray: TableServer[] = [];

    for (let i = 0; i < array.length; i++) {
        if (array[i].maxMoney === undefined) array[i].maxMoney = 0;
        if (array[i].hackingChance === undefined) array[i].hackingChance = 0;
        if (array[i].weakeningTime === undefined) array[i].weakeningTime = 0;

        const server: TableServer = {
            name: array[i].name,
            maxMoney: ns.formatNumber(Number(array[i].maxMoney)),
            hackingChance: ns.formatNumber(Number(array[i].hackingChance)),
            weakeningTime: ns.formatNumber(Number(array[i].weakeningTime) / 1000), //ns.formatNumber(Number(array[i].weakeningTime), 4),
            score: ns.formatNumber(Number(array[i].score)),
        };
        tableArray.push(server);
    }

    ns.print("╔════════════════════╦══════════╦════════╦═══════════╦═════════╗");
    ns.print("║       server       ║   Max $  ║ chance ║ Weak time ║  score  ║");
    ns.print("╠════════════════════╬══════════╬════════╬═══════════╬═════════╣");
    // polluting table with data
    for (let i = 0; i < tableArray.length; i++) {
        ns.print(
            "║ " +
                tableArray[i].name +
                space(tableArray[i].name.length, 0) +
                "║ " +
                tableArray[i].maxMoney +
                space(tableArray[i].maxMoney.length, 1) +
                "║ " +
                tableArray[i].hackingChance +
                space(tableArray[i].hackingChance.length, 2) +
                "║ " +
                tableArray[i].weakeningTime +
                space(tableArray[i].weakeningTime.length, 3) +
                "║ " +
                tableArray[i].score +
                space(tableArray[i].score.length, 4) +
                "║",
        );
    }

    ns.print("╚════════════════════╩══════════╩════════╩═══════════╩═════════╝");
}

function space(len: number, colIndex: number) {
    // with of the cell content
    const colLen = [19, 9, 7, 10, 8];
    const real = colLen[colIndex] - len;
    let str = "";
    for (let i = 0; i < real; i++) {
        str += " ";
    }
    return str;
}
