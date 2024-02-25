import { NS } from "@ns";

export async function main(ns: NS) {
    let upgrade = false;
    let upgradeName = '';
    let upgradeRam = 0;

    let buy = false;
    let buyName = '';
    let buyRam = 0;

    if (ns.args.length == 3 || ns.args.length == 0) {
        if (ns.args[0] == '-u') {
            upgrade = true;
            upgradeName = ns.args[1].toString();
            upgradeRam = Number(ns.args[2]);
        }
        if (ns.args[0] == '-b') {
            buy = true;
            buyName = ns.args[1].toString();
            buyRam = Number(ns.args[2]);
        }
    } else {
        ns.tprint('\nusage: sm.js [options]\n\nOptions:\n\t-u <Name> <Ram>\n\t-b <Name> <Ram>');
        return;
    }

    if (upgrade) {
        // get current ram
        const servers = ns.getPurchasedServers();
        if (!servers.includes(upgradeName)) {
            ns.tprint('You do not own a server called ' + upgradeName);
        }

        let price = ns.getPurchasedServerUpgradeCost(upgradeName, upgradeRam);

        const answer = await ns.prompt(
            'upgrading the server (' +
                upgradeName +
                ') to ' +
                upgradeRam +
                'GB of Ram, will cost ' +
                ns.formatNumber(price),
        );
        if (answer) ns.upgradePurchasedServer(upgradeName, upgradeRam);
    } else if (buy) {
        let price = ns.getPurchasedServerCost(buyRam);

        const answer = await ns.prompt(
            'buying the server (' + buyName + ') with ' + buyRam + 'GB of Ram, will cost ' + ns.formatNumber(price),
        );
        if (answer) ns.purchaseServer(buyName, buyRam);
    } else {
        const playerMoney = ns.getServerMoneyAvailable('home');
        let ramSize = 16;
        while (ns.getPurchasedServerCost(ramSize) < playerMoney) {
            ramSize *= 2;
        }
        ramSize = ramSize / 2;
        ns.tprint(
            'you can buy a server with ' +
                ramSize +
                'GB of ram. This costs ' +
                ns.formatNumber(ns.getPurchasedServerCost(ramSize)),
        );

        const answer = await ns.prompt('buy Server for ' + ns.formatNumber(ns.getPurchasedServerCost(ramSize)) + '?');

        if (answer) {
            ns.purchaseServer('hacker', ramSize);
        }
    }
}
