import { NS } from "@ns";

interface AutocompleteData {
    servers: string[];
    txts: string[];
    scripts: string[];
    flags: string[];
}
export function autocomplete(data: AutocompleteData) {
    return [...data.servers];
}

export async function main(ns: NS) {
    let primaryName = "";

    let isUpgrade = false;
    let upgradeRam = 0;
    let upgradeUnit = "";

    let isBuy = false;
    let buyRam = 0;
    let buyUnit = "";

    let isRename = false;
    let newName = "";

    let isDelete = false;

    let isKill = false;

    let isOverview = false;

    if (ns.args.length > 0) {
        if (ns.args[0] == "-u" && ns.args.length == 3) {
            isUpgrade = true;
            primaryName = ns.args[1].toString();
            upgradeRam = Number(String(ns.args[2]).slice(0, -1));
            upgradeUnit = String(ns.args[2]).slice(-1);
        }
        if (ns.args[0] == "-b" && ns.args.length == 3) {
            isBuy = true;
            primaryName = ns.args[1].toString();
            buyRam = Number(String(ns.args[2]).slice(0, -1));
            buyUnit = String(ns.args[2]).slice(-1);
        }
        if (ns.args[0] == "-r" && ns.args.length == 3) {
            isRename = true;
            primaryName = ns.args[1].toString();
            newName = ns.args[2].toString();
        }
        if (ns.args[0] == "-d" && ns.args.length == 2) {
            isDelete = true;
            primaryName = ns.args[1].toString();
        }
        if (ns.args[0] == "-k" && ns.args.length == 2) {
            isKill = true;
            primaryName = ns.args[1].toString();
        }
        if (ns.args[0] == "-o" && ns.args.length == 1) {
            isOverview = true;
        }
    } else {
        ns.tprint(
            "\nusage: sm.js [options]\n\nOptions:" +
                "\n\t-u <Name> <Ram><G|T|P>" +
                "\n\t-b <Name> <Ram><G|T|P>" +
                "\n\t-r (<old name> <new name>" +
                "\n\t-d (delete) <server name>" +
                "\n\t-k (kill) <server name>" +
                "\n\t-o (overview)",
        );
        return;
    }

    if (isUpgrade) {
        // get current ram
        const exponent = Math.ceil(Math.log2(upgradeRam));
        upgradeRam = Math.pow(2, exponent);

        upgradeRam = getGBfromAnyUnit(ns, upgradeRam, upgradeUnit);
        const servers = ns.getPurchasedServers();
        if (!servers.includes(primaryName)) {
            ns.tprint("You do not own a server called " + primaryName);
        }

        const price = ns.getPurchasedServerUpgradeCost(primaryName, upgradeRam);

        const answer = await ns.prompt(
            "upgrading the server (" +
                primaryName +
                ") to " +
                upgradeRam +
                "GB of Ram, will cost " +
                ns.formatNumber(price),
        );
        if (answer) ns.upgradePurchasedServer(primaryName, upgradeRam);
    } else if (isBuy) {
        buyRam = getGBfromAnyUnit(ns, buyRam, buyUnit);
        if (buyRam < 1) return;

        const price = ns.getPurchasedServerCost(buyRam);

        const answer = await ns.prompt(
            "buying the server (" + primaryName + ") with " + buyRam + "GB of Ram, will cost " + ns.formatNumber(price),
        );
        if (answer) ns.purchaseServer(primaryName, buyRam);
    } else if (isRename) {
        if (!ns.getPurchasedServers().includes(primaryName)) {
            ns.tprint("You do not own a server called " + primaryName);
        }
        ns.renamePurchasedServer(primaryName, newName);
    } else if (isDelete) {
        ns.deleteServer(primaryName);
    } else if (isKill) {
        ns.tprint("Killing server " + primaryName);
        ns.killall(primaryName);
    } else if (isOverview) {
        ns.tail();
        ns.disableLog("ALL");
        const servers = ns.getPurchasedServers();
        for (let i = 0; i < servers.length; i++) {
            const ramPercent = ns.formatNumber(ns.getServerUsedRam(servers[i]) / ns.getServerMaxRam(servers[i]));
            ns.print(servers[i] + "\t" + ns.getServerMaxRam(servers[i]) + "GB\t" + ramPercent + "%");
        }
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
