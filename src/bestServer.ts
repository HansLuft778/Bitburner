import { NS } from "@ns";

import { serverScanner, isHackable } from "./lib.js";

interface Server {
    name: string;
    maxMoney?: number;
    hackingChance?: number;
    weakeningTime?: number;
    score: number;
}

export async function main(ns: NS) {
    ns.tail();
    ns.print("gorg");
    // ns.disableLog('ALL');
    getBestServerList(ns, true);
}

export function getBestServerList(ns: NS, shouldPrint: boolean) {
    const serverList = serverScanner(ns);

    // relevant stats are max money and hack chance
    // possible formula: (Max Money / weaktime + 3) * (1 / weak time) * (chance to hack)
    const servers: Server[] = [];

    for (let i = 0; i < serverList.length; i++) {
        if (isHackable(ns, serverList[i])) {
            const maxMoney = ns.getServerMaxMoney(serverList[i]);
            const hackingChance = ns.hackAnalyzeChance(serverList[i]);
            const weakeningTime = ns.getWeakenTime(serverList[i]);

            // filter server with no money or the hacking level above players hacking level
            if (maxMoney < 1 || ns.getServerRequiredHackingLevel(serverList[i]) > ns.getHackingLevel()) continue;

            // const score = (maxMoney / (weakeningTime + 3)) * hackingChance * (1 / weakeningTime)
            // const score = ns.formatNumber(((maxMoney / (weakeningTime)) * hackingChance) / 1000)
            let score = maxMoney / ns.getServerMinSecurityLevel(serverList[i]) / 1000000;

            if (ns.fileExists("formulas.exe", "home")) {
                const server = ns.getServer(serverList[i]);
                const player = ns.getPlayer();
                server.hackDifficulty = server.minDifficulty;
                const maxMoney = server.moneyMax == undefined ? 0 : server.moneyMax;
                score =
                    ((maxMoney / ns.formulas.hacking.weakenTime(server, player)) *
                        ns.formulas.hacking.hackChance(server, player)) /
                    1000;
            }

            const server: Server = {
                name: serverList[i],
                maxMoney: maxMoney,
                hackingChance: hackingChance,
                weakeningTime: weakeningTime,
                score: score,
            };

            servers.push(server);
        }
    }

    servers.sort((a, b) => {
        return (b.score || 0) - (a.score || 0);
    });

    if (shouldPrint) printTable(ns, servers);

    return servers;
}

export function getBestServer(ns: NS): string {
    const servers = getBestServerList(ns, false);
    return servers[0].name;
}

export function getBestServerListCheap(ns: NS): Server[] {
    const serverList = serverScanner(ns);

    const servers: Server[] = [];

    for (let i = 0; i < serverList.length; i++) {
        const serverName = serverList[i];
        if (!isHackable(ns, serverName)) continue;

        const maxMoney = ns.getServerMaxMoney(serverName);

        // filter server with no money or the hacking level above players hacking level
        if (maxMoney < 1 || ns.getServerRequiredHackingLevel(serverList[i]) > ns.getHackingLevel()) continue;

        let score = maxMoney / ns.getServerMinSecurityLevel(serverName) / 1000000;

        const server: Server = {
            name: serverName,
            score: score,
        };
        servers.push(server);
    }

    servers.sort((a, b) => {
        return (b.score || 0) - (a.score || 0);
    });

    return servers;
}

function sliceIntoChunks(arr: any[], chunkSize: number) {
    const res = [];
    for (let i = 0; i < arr.length; i += chunkSize) {
        const chunk = arr.slice(i, i + chunkSize);
        res.push(chunk);
    }
    return res;
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

    ns.print("╔════════════════════╦═══════════╦════════╦═══════════╦═════════╗");
    ns.print("║       server       ║   Max $   ║ chance ║ Weak time ║  score  ║");
    ns.print("╠════════════════════╬═══════════╬════════╬═══════════╬═════════╣");
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

    ns.print("╚════════════════════╩═══════════╩════════╩═══════════╩═════════╝");
}

function space(len: number, colIndex: number) {
    // with of the cell content
    const colLen = [19, 10, 7, 10, 8];
    const real = colLen[colIndex] - len;
    let str = "";
    for (let i = 0; i < real; i++) {
        str += " ";
    }
    return str;
}
