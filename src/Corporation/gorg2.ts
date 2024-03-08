import { CorpMaterialName, NS } from "@ns";
import { CityName, buyResourceOnce, getOptimalBoostMaterialQuantities } from "./lib";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const corp = ns.corporation;

    const data = corp.getIndustryData("Chemical");
    const count = getOptimalBoostMaterialQuantities(data, 350);
    ns.print(count);

    const materials: CorpMaterialName[] = ["AI Cores", "Hardware", "Real Estate", "Robots"];

    let state = corp.getCorporation().nextState;
    ns.print("state: " + state);
    while (state !== "PURCHASE") {
        await corp.nextUpdate();
        state = corp.getCorporation().nextState;
        ns.print("waiting for PURCHASE, next state: " + state);
    }
    ns.print("state: " + state);
    const cities = Object.values(CityName);
    if (state === "PURCHASE") {
        for (const city of cities) {
            for (let i = 0; i < materials.length; i++) {
                corp.buyMaterial("BASF", city, materials[i], count[i] / 10);
            }
        }
    }
    await corp.nextUpdate();
    ns.print("cancle all orders...");
    // cancle all orders
    for (const city of cities) {
        for (let i = 0; i < materials.length; i++) {
            corp.buyMaterial("BASF", city, materials[i], 0);
        }
    }
}
