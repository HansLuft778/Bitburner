import {
    buyBoostMaterial,
    customSmartSupply,
    developNewProduct,
    Divisions
} from "./lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");
    ns.clearLog();

    const corp = ns.corporation;

    // customSmartSupply(ns);

    while (true) {
        ns.print(
            corp.getCorporation().prevState +
                " " +
                ns.formatNumber(corp.getInvestmentOffer().funds)
        );
        await corp.nextUpdate();
    }
}
