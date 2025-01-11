import { getOptimalBoostMaterialQuantities } from "./lib";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const corp = ns.corporation;

    const productName = "gorg";
    let productId = 30;

    while (true) {
        const playerMoney = corp.getCorporation().funds;
        const currentProducts = corp.getDivision("Tobacco").products;

        for (const product of currentProducts) {
            corp.getProduct("Tobacco", "Sector-12", product).desiredSellPrice;
        }

        if (currentProducts.length === 3) {
            corp.discontinueProduct("Tobacco", currentProducts[0]);
            break;
        }

        corp.makeProduct("Tobacco", "Sector-12", productName + productId, playerMoney * 0.01, playerMoney * 0.01);

        productId++;
        await corp.nextUpdate();
    }
}
