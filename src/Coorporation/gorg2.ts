import { NS } from "@ns";
import { getOptimalBoostMaterialQuantities } from "./lib";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const industryData = ns.corporation.getIndustryData("Chemical");
    for (const city of ["Aevum", "Chongqing", "Ishima", "New Tokyo", "Sector-12", "Volhaven"] as const) {
        ns.print(ns.corporation.getWarehouse("BASF", city).level);
    }
    const qunatities = getOptimalBoostMaterialQuantities(industryData, 300);

    ns.print(qunatities);
}
