import { NS } from "@ns";
import { ChangeEmployeeRatio, CityName, Ratio, buyBoostMaterial, getOptimalBoostMaterialQuantities } from "./lib";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const corp = ns.corporation;
    const data = corp.getIndustryData("Tobacco");
    const warehouseSpace = corp.getWarehouse("Tobacco", "Sector-12").size * 0.8;
    const neededMaterial = getOptimalBoostMaterialQuantities(data, warehouseSpace);
    ns.print(neededMaterial);
    const gorg = corp.getProduct("Tobacco", "Sector-12", "gorg22").competition;
    const gorg1 = corp.getProduct("Tobacco", "Sector-12", "gorg22").demand;
    ns.print(`competition 22: ${gorg}`);
    ns.print(`demand 22: ${gorg1}`);
    const gorg2 = corp.getProduct("Tobacco", "Sector-12", "gorg23").competition;
    const gorg21 = corp.getProduct("Tobacco", "Sector-12", "gorg23").demand;
    ns.print(`competition 23: ${gorg2}`);
    ns.print(`demand 23: ${gorg21}`);

    // await buyBoostMaterial(ns, "KFC");
    // ChangeEmployeeRatio(ns, "Tobacco", CityName.Sector12, Ratio.profitRatio);
}
