import { NS } from "@ns";

export class ServerManager {
    static buyServer(ns: NS, ram: number): string {
        const limit = ns.getPurchasedServerLimit();
        const all = ns.getPurchasedServers();
        if (all.length >= limit) {
            ns.print("[server Manager] attempted to buy a new server, but the limit has been reached");
            return "";
        }

        // round ram up to the next power of 2
        const exponent = Math.ceil(Math.log2(ram));
        ram = Math.pow(2, exponent);

        const name = ns.purchaseServer("aws", ram);
        return name;
    }

    static upgradeServer(ns: NS, ram: number, name: string) {
        // get current ram
        const servers = ns.getPurchasedServers();
        if (!servers.includes(name)) {
            ns.print("[server Manager] attempted to upgrade Server, but the server does not exist");
            return false;
        }

        const price = ns.getPurchasedServerUpgradeCost(name, ram);

        ns.upgradePurchasedServer(name, ram);

        return true;
    }
}

function getNextServerName(ns: NS) {
    const allServers = ns.getPurchasedServers();
    const awsServers = allServers.filter((str) => str.startsWith("aws-"));

    // get highest number and construct the new name
    const numbers = awsServers.map((server) => parseInt(server.split("-")[1]));
    const max = Math.max(...numbers);
    const newName = "aws-" + (max + 1);
    return newName;
}
