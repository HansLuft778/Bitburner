import { Division, initializeDivision } from "./lib.js";

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
        const div = ns.corporation.getDivision(Division.Agriculture.name);
        ns.print(div);
    } catch (error) {
        ns.print("Division does not exist, creating...");
        await initializeDivision(ns, Division.Agriculture);
    }
}

export async function launchCorp(ns: NS) {
    const corp = ns.corporation;

    // if (!corp.hasCorporation()) {
    //     const res = corp.createCorporation("Apple", false);
    //     if (!res) throw new Error("Failed to create corporation");
    // }

    // --------------------- PHASE 1 ---------------------
    // 1. create argi division
    await initializeDivision(ns, Division.Agriculture);
}
