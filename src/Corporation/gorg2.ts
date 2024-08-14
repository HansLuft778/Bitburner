import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const corp = ns.corporation;

    const currentProducts = corp.getDivision("Tobacco").products;
    corp.getProduct("Tobacco", "Sector-12", currentProducts[0]).desiredSellPrice = 1e6;

    for (const product of currentProducts) {
        const demand = corp.getProduct("Tobacco", "Sector-12", product).actualSellAmount;
        const comp = corp.getProduct("Tobacco", "Sector-12", product).competition;
        const rating = corp.getProduct("Tobacco", "Sector-12", product).rating;
        const gorg = corp.getProduct("Tobacco", "Sector-12", product).stats.performance;
        ns.print(`${product} demand: ${demand}, competition: ${comp}, rating: ${rating}, score: ${gorg}`);
    }

    // await buyBoostMaterial(ns, "KFC");
    // ChangeEmployeeRatio(ns, "Tobacco", CityName.Sector12, Ratio.profitRatio);
}
