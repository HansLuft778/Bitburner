import { NS } from "@ns";
import { buyBoostMaterial, getOptimalBoostMaterialQuantities } from "./lib";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const corp = ns.corporation;
    const ad = corp.getHireAdVertCount("Lettuce begin");
    ns.print(ad);
    // const quant = getOptimalBoostMaterialQuantities(corp.getIndustryData("Agriculture"), 5250 * 0.8);
    // ns.print(quant);
    // const mat = corp.getMaterial("BASF", "Chongqing", "Hardware").stored;
    // ns.print(mat);
    // await buyBoostMaterial(ns, "Lettuce begin");
}
