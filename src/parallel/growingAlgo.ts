import { Config } from "@/Config/Config";
import { getBestHostByRamOptimized } from "@/bestServer";
import { Colors } from "@/lib";
import { NS } from "@ns";
import { ServerManager } from "./ServerManager";

export async function main(ns: NS) {
    ns.tail();
    growServer(ns, "foodnstuff", 0);
}

export function growServer(ns: NS, target: string, batchId: number, delay = 0): boolean {
    const serverMaxMoney = ns.getServerMaxMoney(target);
    const serverCurrentMoney = ns.getServerMoneyAvailable(target);
    let moneyMultiplier = serverMaxMoney / serverCurrentMoney;
    if (isNaN(moneyMultiplier) || moneyMultiplier == Infinity) moneyMultiplier = 1;
    const totalGrowThreadsNeeded = Math.ceil(ns.growthAnalyze(target, moneyMultiplier));

    ns.print("Actual grow threads needed: " + totalGrowThreadsNeeded);

    if (totalGrowThreadsNeeded < 1) {
        ns.print("No grow threads needed, skipping growth process");
        return false;
    }

    // exec grow.js with num of threads
    const allHosts = getBestHostByRamOptimized(ns);
    const growingScriptRam = 1.75;

    for (let i = 0; i < allHosts.length; i++) {
        const host = allHosts[i];

        const maxThreadsOnHost = Math.floor(host.availableRam / growingScriptRam);

        if (maxThreadsOnHost >= totalGrowThreadsNeeded) {
            ns.exec("grow.js", host.name, totalGrowThreadsNeeded, target, delay);
            return true;
        }
    }

    ns.print(Colors.YELLOW + "No available host to grow " + target + ". Attempting to upgrade/buy server...");

    const neededGrowRam = totalGrowThreadsNeeded * growingScriptRam;
    const server = ServerManager.buyOrUpgradeServer(ns, neededGrowRam, Config.GROW_SERVER_NAME);

    if (server === "") {
        ns.tprint("Error! Could not buy server to grow " + target);
        throw new Error("Error! Could not buy server to grow " + target);
    }

    ns.exec("grow.js", server, totalGrowThreadsNeeded, target, delay);

    return true;
}
