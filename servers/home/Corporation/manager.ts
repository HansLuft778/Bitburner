import { launchCorp } from "./LaunchCoorp.js";
import { Division, initializeDivision } from "./lib.js";

export async function main(ns: NS) {
    const corp = ns.corporation;

    await launchCorp(ns);

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
