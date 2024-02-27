import { NS } from "@ns";
import { readFile } from "fs";

 
/**
 * Represents the settings for the application.
 * 
 * To modify the settings, update the class properties accordingly.
 */
class Settings {

    /**
     * The target the daemon should hack, leave empty to hack the best server.
     * Default: ""
     */
    public target = "";

    /**
     * The maximum amount of money the daemon is allowed to use to buy servers.
     * Default: 0
     */
    public maxMoneyToBuy = 0;

    /**
     * The maximum amount of money the daemon is allowed to hack from a server, leave at 0 to let the daemon decide.
     * Default: 0
     */
    public hackThreshold = 0;
}

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    const file = "/Settings/settings.txt";

    const data = ns.read(file);
    ns.print(data)

    const settings: Settings = JSON.parse(data);

    ns.print(settings.hackThreshold);
}
