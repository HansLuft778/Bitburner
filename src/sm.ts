import { NS } from "@ns";

export async function main(ns: NS) {
    let isUpgrade = false;
    let upgradeName = "";
    let upgradeRam = 0;

    let isBuy = false;
    let buyName = "";
    let buyRam = 0;
    let buyUnit = "";

    if (ns.args.length == 3 || ns.args.length == 0) {
        if (ns.args[0] == "-u") {
            isUpgrade = true;
            upgradeName = ns.args[1].toString();
            upgradeRam = Number(ns.args[2]);
        }
        if (ns.args[0] == "-b") {
            isBuy = true;
            buyName = ns.args[1].toString();
            buyRam = Number(String(ns.args[2]).slice(0, -1));
            buyUnit = String(ns.args[2]).slice(-1);
        }
    } else {
        ns.tprint("\nusage: sm.js [options]\n\nOptions:\n\t-u <Name> <Ram><G|T|P>\n\t-b <Name> <Ram><G|T|P>");
        return;
    }

    if (isUpgrade) {
        // get current ram
        const servers = ns.getPurchasedServers();
        if (!servers.includes(upgradeName)) {
            ns.tprint("You do not own a server called " + upgradeName);
        }

        const price = ns.getPurchasedServerUpgradeCost(upgradeName, upgradeRam);

        const answer = await ns.prompt(
            "upgrading the server (" +
                upgradeName +
                ") to " +
                upgradeRam +
                "GB of Ram, will cost " +
                ns.formatNumber(price),
        );
        if (answer) ns.upgradePurchasedServer(upgradeName, upgradeRam);
    } else if (isBuy) {
        buyRam = getGBfromAnyUnit(ns, buyRam, buyUnit);
        if (buyRam < 1) return;

        const price = ns.getPurchasedServerCost(buyRam);

        const answer = await ns.prompt(
            "buying the server (" + buyName + ") with " + buyRam + "GB of Ram, will cost " + ns.formatNumber(price),
        );
        if (answer) ns.purchaseServer(buyName, buyRam);
    } else {
        const playerMoney = ns.getServerMoneyAvailable("home");
        let ramSize = 16;
        while (ns.getPurchasedServerCost(ramSize) < playerMoney) {
            ramSize *= 2;
        }
        ramSize = ramSize / 2;
        ns.tprint(
            "you can buy a server with " +
                ramSize +
                "GB of ram. This costs " +
                ns.formatNumber(ns.getPurchasedServerCost(ramSize)),
        );

        const answer = await ns.prompt("buy Server for " + ns.formatNumber(ns.getPurchasedServerCost(ramSize)) + "?");

        if (answer) {
            ns.purchaseServer("hacker", ramSize);
        }
    }
}

function getGBfromAnyUnit(ns: NS, ram: number, unit: string): number {
    if (unit == "G") {
        return ram;
    } else if (unit == "T") {
        return ram * 1024;
    } else if (unit == "P") {
        return ram * 1024 * 1024;
    } else {
        ns.tprint("\nusage: sm.js [options]\n\nOptions:\n\t-u <Name> <Ram><G|T|P>\n\t-b <Name> <Ram><G|T|P>");
        return 0;
    }
}
