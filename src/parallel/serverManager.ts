import { NS } from "@ns";

export async function main(ns: NS) {
    //
}

export function buyServer(ns: NS, ram: number) {
    const limit = ns.getPurchasedServerLimit();
    const all = ns.getPurchasedServers();
    if (all.length >= limit) {
        ns.print("[server Manager] attempted to buy a new server, but the limit has been reached");
        return false;
    }

    const price = ns.getPurchasedServerCost(ram);
    const name = getNextServerName(ns);
    ns.purchaseServer(name, ram);

    return true;
}

export function upgradeServer(ns: NS, ram: number, name: string) {
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

function getNextServerName(ns: NS) {
    const allServers = ns.getPurchasedServers();
    const filteredStrings = allServers.filter((str) => str.startsWith("aws-"));

    // get highest number and construct the new name
    const numbers = filteredStrings.map((server) => parseInt(server.split("-")[1]));
    const max = Math.max(...numbers);
    const newName = "aws-" + (max + 1);
    return newName;
}
