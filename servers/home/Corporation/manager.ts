import { CorporationState } from "./CorpState.js";
import { launchCorp } from "./LaunchCoorp.js";
import {
    acceptBestInvestmentOffer,
    buyBoostMaterial,
    CityName,
    Divisions,
    initializeDivision,
    unemployEveryone
} from "./lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();

    const corp = ns.corporation;

    await launchCorp(ns);

    // wait for phase two
    while (true) {
        await corp.nextUpdate();
    }
    const offer = await acceptBestInvestmentOffer(ns);
    ns.print("accepted offer: " + ns.formatNumber(offer));

    // upgrade agri
    const divisionNameAgri = Divisions.Agriculture.name;
    const divisionNameChem = Divisions.Chemical.name;

    const cities: CityName[] = Object.values(CityName);
    for (const city of cities) {
        const size = corp.getOffice(divisionNameAgri, city).size;
        if (size < 8) {
            corp.upgradeOfficeSize(divisionNameAgri, city, 8 - size);
            ns.print(
                `upgradeded office size of ${divisionNameAgri} in Sector-12 to ${
                    corp.getOffice(divisionNameAgri, city).size
                }`
            );
        }
        for (let i = 0; i < 8 - size; i++) {
            corp.hireEmployee(divisionNameAgri, city);
        }

        // assign everyone to RnD
        unemployEveryone(ns, divisionNameAgri, city);
        corp.setAutoJobAssignment(
            divisionNameAgri,
            city,
            "Research & Development",
            corp.getOffice(divisionNameAgri, city).size
        );

        // upgrade warehouse
        const currentWarehouse = corp.getWarehouse(
            divisionNameAgri,
            city
        ).level;
        if (currentWarehouse < 16)
            corp.upgradeWarehouse(
                divisionNameAgri,
                city,
                16 - currentWarehouse
            );
    }

    for (let i = 0; i < 8 - corp.getHireAdVertCount(divisionNameAgri); i++) {
        corp.hireAdVert(divisionNameAgri);
    }

    // Create Chemical
    try {
        corp.getDivision("BASF");
    } catch (error) {
        ns.print("No chemical devision present, creating...");
        await initializeDivision(ns, Divisions.Chemical);
    }

    ns.print("upgrading Smart Storage to lvl 25");
    const smLvl = corp.getUpgradeLevel("Smart Storage");
    for (let i = 0; i < 25 - smLvl; i++) {
        corp.levelUpgrade("Smart Storage");
    }

    // wait for RP
    CorporationState.getInstance().addDivisionToSet(divisionNameChem);
    ns.print("waiting for RP to increse...");
    while (
        corp.getDivision(divisionNameAgri).researchPoints <
        Divisions.Agriculture.minResearch
    ) {
        await corp.nextUpdate();
    }
    CorporationState.getInstance().removeDivisionFromSet(divisionNameChem);

    ns.print("reassigning jobs");
    const divisionAgri = corp.getDivision(divisionNameAgri);
    for (const city of divisionAgri.cities) {
        unemployEveryone(ns, divisionNameAgri, city);
        corp.setAutoJobAssignment(divisionNameAgri, city, "Operations", 3);
        corp.setAutoJobAssignment(divisionNameAgri, city, "Engineer", 1);
        corp.setAutoJobAssignment(divisionNameAgri, city, "Business", 2);
        corp.setAutoJobAssignment(divisionNameAgri, city, "Management", 2);
    }
    const divisionChem = corp.getDivision(divisionNameChem);
    for (const city of divisionChem.cities) {
        unemployEveryone(ns, divisionNameChem, city);
        corp.setAutoJobAssignment(divisionNameAgri, city, "Operations", 1);
        corp.setAutoJobAssignment(divisionNameAgri, city, "Engineer", 1);
        corp.setAutoJobAssignment(divisionNameAgri, city, "Business", 1);
    }

    ns.print("buying boost materials");
    buyBoostMaterial(ns, divisionNameAgri);
    buyBoostMaterial(ns, divisionNameChem);

    ns.print("DONE :D");
}
