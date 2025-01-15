import {
    convertDivisionToSupportOffices,
    Divisions,
    OfficeRatios
} from "./lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");
    ns.clearLog();

    const corp = ns.corporation;

    // customSmartSupply(ns);

    // await buyBoostMaterial(ns, "GOrg");

    convertDivisionToSupportOffices(
        ns,
        corp.getDivision(Divisions.Agriculture.name),
        OfficeRatios.progressRatio
    );
}
