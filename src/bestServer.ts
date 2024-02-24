import { NS } from "@ns";

import { serverScanner, isHackable } from "/src/lib.js";

interface Server {
    name: string;
    maxMoney: number;
    hackingChance: number;
    weakeningTime: number;
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

            // filter server if it has no money or the hacking level is above 2/3 of the players hacking level
            if (maxMoney < 1 || ns.getServerRequiredHackingLevel(serverList[i]) > ns.getHackingLevel()) {
                continue;
            }

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

        servers.sort((a, b) => {
            return b.score - a.score;
        });
    }

    if (shouldPrint) printTable(ns, servers);

    return servers;
}

export function getBestServer(ns: NS) {
    const servers = getBestServerList(ns, false);
    return servers[0].name;
}

export function findBestServer(ns: NS) {
    // for now only finds the server with the highest max money, this is not optimal
    const serverList = serverScanner(ns);

    let maxMoney = 0;
    const topTen: string[] = [];
    const stats: number[] = [];

    serverList.forEach((server) => {
        if (isHackable(ns, server)) {
            const serverMaxMoney = ns.getServerMaxMoney(server);
            const hackTime = ns.getHackTime(server);
            const growTime = ns.getGrowTime(server);
            const weakenTime = ns.getWeakenTime(server);

            if (serverMaxMoney > maxMoney) {
                maxMoney = serverMaxMoney;
                topTen.push(server);

                stats.push(serverMaxMoney);
                stats.push(hackTime);
                stats.push(growTime);
                stats.push(weakenTime);

                if (topTen.length > 10) {
                    topTen.shift();
                    stats.splice(0, 4);
                }
            }
        }
    });
    const statSplit = sliceIntoChunks(stats, 4);

    return [topTen, statSplit];
}

function sliceIntoChunks(arr: any[], chunkSize: number) {
    const res = [];
    for (let i = 0; i < arr.length; i += chunkSize) {
        const chunk = arr.slice(i, i + chunkSize);
        res.push(chunk);
    }
    return res;
}

// with of the cell content
const colLen = [19, 10, 7, 10, 8];

export function printTable(ns: NS, array: any[]) {
    ns.print("╔════════════════════╦═══════════╦════════╦═══════════╦═════════╗");
    ns.print("║       server       ║   Max $   ║ chance ║ Weak time ║  score  ║");
    ns.print("╠════════════════════╬═══════════╬════════╬═══════════╬═════════╣");
    // polluting table with data
    for (let i = 0; i < array.length; i++) {
        ns.print(
            "║ " +
                array[i][0] +
                space(array[i][0].length, 0) +
                "║ " +
                array[i][1] +
                space(array[i][1].length, 1) +
                "║ " +
                array[i][2] +
                space(array[i][2].length, 2) +
                "║ " +
                array[i][3] +
                space(array[i][3].length, 3) +
                "║ " +
                array[i][4] +
                space(array[i][4].length, 4) +
                "║",
        );
    }

    ns.print("╚════════════════════╩═══════════╩════════╩═══════════╩═════════╝");
}

function space(len: number, colIndex: number) {
    const real = colLen[colIndex] - len;
    let str = "";
    for (let i = 0; i < real; i++) {
        str += " ";
    }
    return str;
}
