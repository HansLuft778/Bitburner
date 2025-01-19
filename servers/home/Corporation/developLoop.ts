import { getOptimalBoostMaterialQuantities } from "./lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const corp = ns.corporation;

    const productName = "gorg-";

    // find latest product id
    let ids: number[] = [];
    for (const product of corp.getDivision("Tobacco").products) {
        const parts = product.split("-");
        ids.push(parseInt(parts[1]));
    }
    let productId = Math.max(...ids) + 1;

    while (true) {
        const corpFunds = corp.getCorporation().funds;
        const currentProducts = corp.getDivision("Tobacco").products;

        if (currentProducts.length === 3) {
            corp.discontinueProduct("Tobacco", currentProducts[0]);
            break;
        }

        corp.makeProduct(
            "Tobacco",
            "Sector-12",
            productName + productId,
            corpFunds * 0.01,
            corpFunds * 0.01
        );

        productId++;
        await corp.nextUpdate();
    }
}
