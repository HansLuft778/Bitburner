import { launchCorp } from "./LaunchCoorp.js";
import {
    CityName,
    Division,
    initializeDivision,
    unemployEveryone
} from "./lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();

    const corp = ns.corporation;

    await launchCorp(ns);

    // wait for phase two
    while (corp.getInvestmentOffer().funds < 310e9) {
        await corp.nextUpdate();
    }

    let result = corp.acceptInvestmentOffer();
    if (!result) {
        ns.print("Error accepting investment offer");
        return;
    }

    // upgrade agri
    const divisionName = Division.Agriculture.name;
    const cities: CityName[] = Object.values(CityName);
    for (const city of cities) {
        const size = corp.getOffice(divisionName, city).size;
        if (size < 8) {
            corp.upgradeOfficeSize(divisionName, city, 8 - size);
            ns.print(
                `upgradeded office size of ${divisionName} in Sector-12 to ${
                    corp.getOffice(divisionName, city).size
                }`
            );
        }
        for (let i = 0; i < 8 - size; i++) {
            corp.hireEmployee(divisionName, city);
        }

        // assign everyone to RnD
        unemployEveryone(ns, divisionName, city);
        corp.setAutoJobAssignment(
            divisionName,
            city,
            "Research & Development",
            corp.getOffice(divisionName, city).size
        );

        // upgrade warehouse
        const currentWarehouse = corp.getWarehouse(divisionName, city).level;
        if (currentWarehouse < 16)
            corp.upgradeWarehouse(divisionName, city, 16 - currentWarehouse);
    }

    for (let i = 0; i < 8 - corp.getHireAdVertCount(divisionName); i++) {
        corp.hireAdVert(divisionName);
    }

    // Create Chemical
    try {
        corp.getDivision("BASF");
    } catch (error) {
        ns.print("No chemical devision present, creating...");
        await initializeDivision(ns, Division.Chemical);
    }

    ns.print("upgrading Smart Storage to lvl 25")
    const smLvl = corp.getUpgradeLevel("Smart Storage");
    for (let i = 0; i < 25 - smLvl; i++) {
        corp.levelUpgrade("Smart Storage");
    }

    ns.print("DONE! :D");

    if (corp.getDivision("Lettuce begin") === undefined) {
        let lastState = "";
        while (true) {
            lastState = await corp.nextUpdate();
        }
    }
}
