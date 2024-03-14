import { NS } from "@ns";
import { launchCorp } from "./LaunchCoorp";

export async function main(ns: NS) {
    const corp = ns.corporation;

    if (corp.hasCorporation()) await launchCorp(ns);

    // check for phase 2
    let hasChemical = false;
    try {
        corp.getDivision("BASF");
        hasChemical = true;
    } catch (error) {
        hasChemical = false;
    }

    if (!hasChemical) {
        corp.expandIndustry("Chemical", "BASF");
        if (!res) throw new Error("Failed to expand industry");
    }

    if (corp.getDivision("Lettuce begin") === undefined) {
        let lastState = "";
        while (true) {
            lastState = await corp.nextUpdate();
        }
    }
}
