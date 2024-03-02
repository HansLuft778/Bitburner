import { NS } from "@ns";
import { getHackThreads, getHackThreadsFormulas, isHackable } from "./lib";
import { PlayerManager } from "./parallel/PlayerManager";
import { getBestServer } from "./bestServer";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    PlayerManager.getInstance(ns).resetPlayer(ns);

    ns.print(isHackable(ns, "I.I.I.I"));

    const target = getBestServer(ns);
}
