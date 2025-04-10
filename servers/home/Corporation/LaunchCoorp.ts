import { Divisions, initializeDivision } from "./lib.js";

// debug, run manager.ts
export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    const corp = ns.corporation;

    if (!corp.hasCorporation()) {
        const res = corp.createCorporation("Apple", false);
        if (!res) throw new Error("Failed to create corporation");
    }

    // --------------------- PHASE 1 ---------------------
    // 1. create argi division
    // check if division already exists
    try {
        const div = ns.corporation.getDivision(Divisions.Agriculture.name);
        ns.print(div);
    } catch (error) {
        ns.print("Division does not exist, creating...");
        await initializeDivision(ns, Divisions.Agriculture);
    }
}

export async function launchCorp(ns: NS) {
    const corp = ns.corporation;

    // --------------------- PHASE 1 ---------------------
    // 1. create argi division
    try {
        corp.getDivision(Divisions.Agriculture.name);
        ns.print("Agri division exists, assuming its correct lol")
    } catch (error) {
        ns.print("no agri division found, creating...");
        await initializeDivision(ns, Divisions.Agriculture);
        ns.print("agri division created!");
    }
}
