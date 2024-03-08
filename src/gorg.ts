import { NS } from "@ns";
import { PlayerManager } from "./parallel/PlayerManager";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    // PlayerManager.getInstance(ns).resetPlayer(ns);

    const pids = [];
    pids.push(1);
    pids.push(2);
    pids.push(3);
    pids.push(4);

    ns.print(pids[pids.length - 1]);
}
