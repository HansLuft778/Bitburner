import { getBestHostByRam } from "@/bestServer";
import { NS } from "@ns";
import { ServerManager } from "./serverManager";
import { Colors } from "@/lib";

export async function main(ns: NS) {
    ns.tail();
    hackServer(ns, "silver-helix", 0.8, 0);
}

export function hackServer(ns: NS, target: string, threshold: number, batchId: number, delay: number = 0) {
    const totalHackThreadsNeeded = Math.ceil(threshold / ns.hackAnalyze(target));
    ns.print("actual hack threads needed: " + totalHackThreadsNeeded);

    const allHosts = getBestHostByRam(ns);
    const hackingScriptRam = 1.7;

    for (let i = 0; i < allHosts.length; i++) {
        const host = allHosts[i];

        const maxThreadsOnHost = Math.floor(host.availableRam / hackingScriptRam);

        if (maxThreadsOnHost >= totalHackThreadsNeeded) {
            ns.exec("hack.js", host.name, totalHackThreadsNeeded, target, delay);
            return true;
        }
    }

    ns.print(Colors.YELLOW + "No available host to grow " + target + ". Buying server...");

    const neededGrowRam = totalHackThreadsNeeded * hackingScriptRam;
    const server = ServerManager.buyOrUpgradeServer(ns, neededGrowRam, "hack", batchId);

    if (server === "") {
        ns.tprint("Error! Could not buy server to hack " + target);
        throw new Error("Error! Could not buy server to hack " + target);
    }

    ns.exec("hack.js", server, totalHackThreadsNeeded, target, delay);

    return true;
}
