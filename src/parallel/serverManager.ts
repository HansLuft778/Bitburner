import { Colors, nukeAll, nukeServer } from "@/lib";
import { NS } from "@ns";

export class ServerManager {
    /**
     * Buys or upgrades a server based on the desired RAM and server type.
     * If the server exists and can be upgraded successfully, it returns the upgraded server name.
     * If the server cannot be upgraded, it attempts to buy a new server with the desired RAM and name.
     *
     * @param ns - The Bitburner namespace object.
     * @param desiredRam - How much RAM needs the server at least.
     * @param serverType - The type of server to buy or upgrade (`weak`, `grow`, `hack`).
     * @param batchId - The batch ID of the server.
     * @returns The name of the upgraded server or an empty string if a new server cannot be bought/upgraded.
     */
    static buyOrUpgradeServer(ns: NS, desiredRam: number, serverType: string, batchId: number): string {
        const serverName = serverType + batchId;
        const upgradeSuccessful = this.upgradeServer(ns, desiredRam, serverName);

        if (upgradeSuccessful) {
            return serverName;
        }

        const exponent = Math.ceil(Math.log2(desiredRam));
        desiredRam = Math.pow(2, exponent);

        let buyCost = ns.getPurchasedServerCost(desiredRam);
        if (buyCost > ns.getServerMoneyAvailable("home")) {
            ns.print(Colors.RED + "[server Manager] Player has not enough money to buy a new server!");
            return "";
        }

        const name = this.buyServer(ns, desiredRam, serverName);
        nukeAll(ns);
        return name;
    }

    static buyServer(ns: NS, ram: number, serverName: string): string {
        const purchasedServerLimit = ns.getPurchasedServerLimit();
        const purchasedServers = ns.getPurchasedServers();
        if (purchasedServers.length >= purchasedServerLimit) {
            ns.print(Colors.RED + "[server Manager] attempted to buy a new server, but the limit has been reached");
            return "";
        }

        const exponent = Math.ceil(Math.log2(ram));
        ram = Math.pow(2, exponent);

        nukeServer(ns, serverName);

        const name = ns.purchaseServer(serverName, ram);
        return name;
    }

    static upgradeServer(ns: NS, desiredRam: number, serverName: string): boolean {
        const purchasedServers = ns.getPurchasedServers();
        if (!purchasedServers.includes(serverName)) {
            ns.print(Colors.YELLOW + "[server Manager] attempted to upgrade Server, but the server does not exist");
            return false;
        }

        const serverMaxRam = ns.getServerMaxRam(serverName);
        let totalRequiredRam = desiredRam + serverMaxRam;

        const exponent = Math.ceil(Math.log2(totalRequiredRam));
        totalRequiredRam = Math.pow(2, exponent);

        const cost = ns.getPurchasedServerUpgradeCost(serverName, totalRequiredRam);

        if (!ns.upgradePurchasedServer(serverName, totalRequiredRam)) {
            ns.print(Colors.RED + "[server Manager] attempted to upgrade Server, but the upgrade failed");
            return false;
        }
        return true;
    }

    /**
     * Upgrades an existing server to meet the desired RAM requirement, that is cheapest to upgrade.
     *
     * @param ns - The NetScriptJS object.
     * @param desiredRam - How much RAM needs the server at least.
     * @param name - The name of the server to upgrade.
     * @returns The name of the upgraded server, or an empty string if the upgrade failed.
     */
    static getBestServerToUpgrade(ns: NS, desiredRam: number, name: string) {
        const purchasedServers = ns.getPurchasedServers().filter((server) => server.includes(name));

        let minUpgradeCost = Number.MAX_VALUE;
        let serverToUpgrade = "";

        let totalRequiredRam = 0;

        for (let i = 0; i < purchasedServers.length; i++) {
            const server = purchasedServers[i];

            const serverMaxRam = ns.getServerMaxRam(server);
            totalRequiredRam = desiredRam + serverMaxRam;

            const exponent = Math.ceil(Math.log2(totalRequiredRam));
            totalRequiredRam = Math.pow(2, exponent);

            const cost = ns.getPurchasedServerUpgradeCost(server, totalRequiredRam);

            if (cost < minUpgradeCost) {
                minUpgradeCost = cost;
                serverToUpgrade = server;
            }
        }
        return {
            minUpgradeCost,
            serverToUpgrade,
            totalRequiredRam,
        };
    }
}
