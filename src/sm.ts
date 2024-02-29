import { NS } from "@ns";

export async function main(ns: NS) {
    let isUpgrade = false;
    let upgradeName = "";
    let upgradeRam = 0;
    let upgradeUnit = "";

    let isBuy = false;
    let buyName = "";
    let buyRam = 0;
    let buyUnit = "";

    let isRename = false;
    let oldName = "";
    let newName = "";

    let isDelete = false;
    let deleteName = "";

    if (ns.args.length == 3 || ns.args.length == 0 || ns.args.length == 2) {
        if (ns.args[0] == "-u") {
            isUpgrade = true;
            upgradeName = ns.args[1].toString();
            upgradeRam = Number(String(ns.args[2]).slice(0, -1));
            upgradeUnit = String(ns.args[2]).slice(-1);
        }
        if (ns.args[0] == "-b") {
            isBuy = true;
            buyName = ns.args[1].toString();
            buyRam = Number(String(ns.args[2]).slice(0, -1));
            buyUnit = String(ns.args[2]).slice(-1);
        }
        if (ns.args[0] == "-r") {
            isRename = true;
            oldName = ns.args[1].toString();
            newName = ns.args[2].toString();
        }
        if (ns.args[0] == "-d") {
            isDelete = true;
            deleteName = ns.args[1].toString();
        }
    } else {
        ns.tprint(
            "\nusage: sm.js [options]\n\nOptions:" +
                "\n\t-u <Name> <Ram><G|T|P>" +
                "\n\t-b <Name> <Ram><G|T|P>" +
                "\n\t-r <old name> <new name>" +
                "\n\t-d <server name>",
        );
        return;
    }

    if (isUpgrade) {
        // get current ram
        const exponent = Math.ceil(Math.log2(upgradeRam));
        upgradeRam = Math.pow(2, exponent);

        upgradeRam = getGBfromAnyUnit(ns, upgradeRam, upgradeUnit);
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
    } else if (isRename) {
        if (!ns.getPurchasedServers().includes(oldName)) {
            ns.tprint("You do not own a server called " + oldName);
        }
        ns.renamePurchasedServer(oldName, newName);
    } else if (isDelete) {
        ns.deleteServer(deleteName);
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
        ns.tprint(
            "\nusage: sm.js [options]\n\nOptions:\n\t-u <Name> <Ram><G|T|P>\n\t-b <Name> <Ram><G|T|P>\n\t-r <old name> <new name>",
        );
        return 0;
    }
}
