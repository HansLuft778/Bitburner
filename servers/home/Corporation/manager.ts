import { launchCorp } from "./LaunchCoorp.js";
import {
    acceptBestInvestmentOffer,
    buyBoostMaterial,
    CityName,
    convertDivisionToSupportOffices,
    Divisions,
    emptyAllWarehouses,
    initializeDivision,
    OfficeRatios,
    officeUpgradeCostFromAtoB,
    ProductDivisionFunds,
    unemployEveryone,
    wilsonAdvertisingOptimizer
} from "./lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();

    const corp = ns.corporation;

    await launchCorp(ns);

    const offer = await acceptBestInvestmentOffer(ns, 500e9);
    ns.print("accepted offer: " + ns.formatNumber(offer));

    ns.print("shutting down custom smart supply cuase it sucks ass");
    ns.scriptKill("Corporation/smartSupply.js", "home");
    await emptyAllWarehouses(ns);

    ns.print("buying the real smart supply cause its good");
    corp.purchaseUnlock("Smart Supply");
    for (const divisionName of corp.getCorporation().divisions) {
        for (const city of corp.getDivision(divisionName).cities) {
            corp.setSmartSupply(divisionName, city, true);
        }
    }

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
        const currentWarehouse = corp.getWarehouse(divisionNameAgri, city).level;
        if (currentWarehouse < 16)
            corp.upgradeWarehouse(divisionNameAgri, city, 16 - currentWarehouse);
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
        corp.setAutoJobAssignment(divisionNameChem, city, "Operations", 1);
        corp.setAutoJobAssignment(divisionNameChem, city, "Engineer", 1);
        corp.setAutoJobAssignment(divisionNameChem, city, "Business", 1);
    }

    // Smart Factories to lvl 25
    for (let i = 0; i < 25 - corp.getUpgradeLevel("Smart Factories"); i++) {
        corp.levelUpgrade("Smart Smart Factories");
    }

    ns.print("buying boost materials");
    await buyBoostMaterial(ns, divisionNameAgri);
    await buyBoostMaterial(ns, divisionNameChem);

    // ----------------------------------------------------------------------------------
    // Round 3
    await acceptBestInvestmentOffer(ns, 9.5e12);

    // upgrade Agri (~500b)
    for (const city of corp.getDivision(Divisions.Agriculture.name).cities) {
        // offices +18 (~222b)
        corp.upgradeOfficeSize(Divisions.Agriculture.name, city, 18);

        // warehouse to 7000 (~90b)
        corp.upgradeWarehouse(Divisions.Agriculture.name, city, 4);
    }
    convertDivisionToSupportOffices(
        ns,
        corp.getDivision(Divisions.Agriculture.name),
        OfficeRatios.progressRatio
    );

    // upgrade Chemical (~100b)
    // office 3 -> 12 (90b)
    for (const city of corp.getDivision(Divisions.Chemical.name).cities) {
        corp.upgradeOfficeSize(Divisions.Chemical.name, city, 9);
    }
    convertDivisionToSupportOffices(
        ns,
        corp.getDivision(Divisions.Chemical.name),
        OfficeRatios.progressRatio
    );

    // create Tobacco division
    const fundsa = corp.getCorporation().funds;
    const TobaccoBudget = fundsa - 600e9;

    const divisionNameTobacco = "Tobacco";
    corp.expandIndustry("Tobacco", divisionNameTobacco);
    const officeBudget =
        (TobaccoBudget * ProductDivisionFunds.beforeFocusOnAdvert.office) / cities.length;
    for (const city of cities) {
        if (city != "Sector-12") corp.expandCity(divisionNameTobacco, city);

        corp.purchaseWarehouse(divisionNameTobacco, city);

        // number of office levels that fit budget
        let cost = 0;
        let targetSize = corp.getOffice(divisionNameTobacco, city).size + 1;
        while (cost < officeBudget) {
            cost = officeUpgradeCostFromAtoB(
                corp.getOffice(divisionNameTobacco, city).size,
                targetSize
            );
            targetSize++;
        }

        ns.print(`upgrading Tobacco office at ${city} to ${targetSize}`);
        corp.upgradeOfficeSize(
            divisionNameTobacco,
            city,
            targetSize - corp.getOffice(divisionNameTobacco, city).size
        );
    }
    // upgrade upgrades
    const rawProductionBudget =
        TobaccoBudget * ProductDivisionFunds.beforeFocusOnAdvert.rawProduction;

    // level smart factorier/storage
    let moneySpent = 0;
    while (moneySpent < rawProductionBudget) {
        const upgradeCost =
            corp.getUpgradeLevelCost("Smart Factories") + corp.getUpgradeLevelCost("Smart Storage");

        corp.levelUpgrade("Smart Factories");
        corp.levelUpgrade("Smart Storage");
        moneySpent += upgradeCost;
    }

    // buy wilson advert
    const wilsonAdvertBudget =
        TobaccoBudget * ProductDivisionFunds.beforeFocusOnAdvert.wilsonAdvert;

    const data = wilsonAdvertisingOptimizer(
        ns,
        wilsonAdvertBudget,
        corp.getDivision(divisionNameTobacco)
    );

    ns.print("buying wilson/advert...");
    for (let i = 0; i < data[0]; i++) {
        corp.levelUpgrade("Wilson Analytics");
        await ns.sleep(0);
    }
    for (let i = 0; i < data[0]; i++) {
        corp.hireAdVert(divisionNameTobacco);
        await ns.sleep(0);
    }

    const employeeStatUpgradeBudget =
        TobaccoBudget * ProductDivisionFunds.beforeFocusOnAdvert.employeStatUpgrades;

    const employeeStatUpgrades = [
        "Neural Accelerators",
        "Nuoptimal Nootropic Injector Implants",
        "Speech Processor Implants"
    ];
    moneySpent = 0;
    while (moneySpent < employeeStatUpgradeBudget) {
        const upgradeCost = employeeStatUpgrades.reduce(
            (sum, upgrade) => sum + corp.getUpgradeLevelCost(upgrade),
            0
        );
        employeeStatUpgrades.map((upgrade) => corp.levelUpgrade(upgrade));
        moneySpent += upgradeCost;
    }

    const salesBotBudget = TobaccoBudget * ProductDivisionFunds.beforeFocusOnAdvert.salesBot;
    moneySpent = 0;
    while (moneySpent < salesBotBudget) {
        const upgradeCost = corp.getUpgradeLevelCost("ABC SalesBots");
        corp.levelUpgrade("ABC SalesBots");
        moneySpent += upgradeCost;
    }

    const projectInsightBudget =
        TobaccoBudget * ProductDivisionFunds.beforeFocusOnAdvert.projectInsight;
    moneySpent = 0;
    while (moneySpent < projectInsightBudget) {
        const upgradeCost = corp.getUpgradeLevelCost("Project Insight");
        corp.levelUpgrade("Project Insight");
        moneySpent += upgradeCost;
    }

    convertDivisionToSupportOffices(
        ns,
        corp.getDivision(divisionNameTobacco),
        OfficeRatios.progressRatio
    );

    await buyBoostMaterial(ns, divisionNameAgri);
    await buyBoostMaterial(ns, divisionNameAgri);
    await buyBoostMaterial(ns, divisionNameTobacco);

    ns.print("DONE :D");
}
