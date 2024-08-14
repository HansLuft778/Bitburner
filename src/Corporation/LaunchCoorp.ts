import { NS } from "@ns";
import { Division, initializeDivision } from "./lib";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    const corp = ns.corporation;

    if (corp.hasCorporation()) throw new Error("You already have a corporation");

    const res = corp.createCorporation("Apple", false);
    if (!res) throw new Error("Failed to create corporation");

    // --------------------- PHASE 1 ---------------------
    // 1. create argi division
    // check if division already exists
    try {
        const div = ns.corporation.getDivision(Division.Agriculture.name);
        ns.print(div);
    } catch (error) {
        ns.print("Division does not exist, creating...");
        await initializeDivision(ns, Division.Agriculture);
    }
}

export async function launchCorp(ns: NS) {
    const corp = ns.corporation;

    const res = corp.createCorporation("Apple", false);
    if (!res) throw new Error("Failed to create corporation");

    // --------------------- PHASE 1 ---------------------
    // 1. create argi division
    await initializeDivision(ns, Division.Agriculture);
}
