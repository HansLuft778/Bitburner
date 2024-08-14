import { NS } from "@ns";
import { launchCorp } from "./LaunchCoorp";
import { Division, initializeDivision } from "./lib";

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
        await initializeDivision(ns, Division.Chemical);
    }

    if (corp.getDivision("Lettuce begin") === undefined) {
        let lastState = "";
        while (true) {
            lastState = await corp.nextUpdate();
        }
    }
}
