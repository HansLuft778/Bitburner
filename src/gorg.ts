import { NS } from "@ns";
import { PlayerManager } from "./parallel/PlayerManager";

class Gorg {
    name = "";

    constructor() {
        //
    }
}

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    PlayerManager.getInstance(ns).resetPlayer(ns);

    const gorg = new Gorg();
    gorg.name = "Gorg";
}
