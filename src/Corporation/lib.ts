import { CorpIndustryData, CorpIndustryName, CorpMaterialName, NS } from "@ns";

export enum CityName {
    Aevum = "Aevum",
    Chongqing = "Chongqing",
    Sector12 = "Sector-12",
    NewTokyo = "New Tokyo",
    Ishima = "Ishima",
    Volhaven = "Volhaven",
}

enum BoostMaterialsSizes {
    AiCores = 0.1,
    Hardware = 0.06,
    RealEstate = 0.005,
    Robots = 0.5,
}

export function getOptimalBoostMaterialQuantities(
    industryData: CorpIndustryData,
    spaceConstraint: number,
    round = true,
): number[] {
    const { aiCoreFactor, hardwareFactor, realEstateFactor, robotFactor } = industryData;
    if (
        aiCoreFactor === undefined ||
        hardwareFactor === undefined ||
        realEstateFactor === undefined ||
        robotFactor === undefined
    ) {
        throw new Error("Industry data is missing factors");
    }
    const boostMaterialCoefficients = [aiCoreFactor, hardwareFactor, realEstateFactor, robotFactor];
    const boostMaterialSizes = [
        BoostMaterialsSizes.AiCores,
        BoostMaterialsSizes.Hardware,
        BoostMaterialsSizes.RealEstate,
        BoostMaterialsSizes.Robots,
    ];

    const calculateOptimalQuantities = (matCoefficients: number[], matSizes: number[]): number[] => {
        const sumOfCoefficients = matCoefficients.reduce((a, b) => a + b, 0);
        const sumOfSizes = matSizes.reduce((a, b) => a + b, 0);
        const result = [];
        for (let i = 0; i < matSizes.length; ++i) {
            let matCount =
                (spaceConstraint -
                    500 *
                        ((matSizes[i] / matCoefficients[i]) * (sumOfCoefficients - matCoefficients[i]) -
                            (sumOfSizes - matSizes[i]))) /
                (sumOfCoefficients / matCoefficients[i]) /
                matSizes[i];
            if (matCoefficients[i] <= 0 || matCount < 0) {
                return calculateOptimalQuantities(matCoefficients.toSpliced(i, 1), matSizes.toSpliced(i, 1)).toSpliced(
                    i,
                    0,
                    0,
                );
            } else {
                if (round) {
                    matCount = Math.round(matCount);
                }
                result.push(matCount);
            }
        }
        return result;
    };
    return calculateOptimalQuantities(boostMaterialCoefficients, boostMaterialSizes);
}

export async function buyBoostMaterial(ns: NS, divisionName: string) {
    const corp = ns.corporation;

    const division = corp.getDivision(divisionName);

    const data = corp.getIndustryData(division.type);

    const materials: CorpMaterialName[] = ["AI Cores", "Hardware", "Real Estate", "Robots"];

    let state = corp.getCorporation().nextState;
    // wait for PURCHASE state
    while (state !== "PURCHASE") {
        await corp.nextUpdate();
        state = corp.getCorporation().nextState;
        ns.print("waiting for PURCHASE, next state: " + state);
    }
    ns.print("state: " + state);
    const cities = division.cities;
    if (state === "PURCHASE") {
        for (const city of cities) {
            for (let i = 0; i < materials.length; i++) {
                const material = materials[i];
                const warehouseSpace = corp.getWarehouse(divisionName, city).size * 0.8;
                const neededMaterial = getOptimalBoostMaterialQuantities(data, warehouseSpace);
                const storedMaterial = corp.getMaterial(divisionName, city, material).stored;
                const toBuy = Math.max(neededMaterial[i] - storedMaterial, 0) / 10;
                ns.print(`Buying ${toBuy}/${neededMaterial[i]} ${material} in ${city}`);
                corp.buyMaterial(divisionName, city, material, toBuy);
            }
        }
    }
    await corp.nextUpdate();

    // cancel all orders
    for (const city of cities) {
        for (let i = 0; i < materials.length; i++) {
            corp.buyMaterial(divisionName, city, materials[i], 0);
        }
    }
}

export class Ratio {
    static progressRatio = {
        Operations: 0.037,
        Engineer: 0.513,
        Business: 0.011,
        Management: 0.44,
    };

    static profitRatio = {
        Operations: 0.032,
        Engineer: 0.462,
        Business: 0.067,
        Management: 0.439,
    };

    static profitRatioEndgame = {
        Operations: 0.064,
        Engineer: 0.317,
        Business: 0.298,
        Management: 0.321,
    };
}

interface DivisionType {
    name: string;
    industry: CorpIndustryName;
    minResearch: number;
    totalAdVerts: number;
    warehouseLvl: number;
    smartStorageLvl: number;
}

export class Division {
    static Agriculture: DivisionType = {
        name: "Lettuce begin",
        industry: "Agriculture",
        minResearch: 30,
        totalAdVerts: 2,
        warehouseLvl: 6,
        smartStorageLvl: 9,
    };

    static Chemical: DivisionType = {
        name: "BASF",
        industry: "Chemical",
        minResearch: 700,
        totalAdVerts: 2,
        warehouseLvl: 6,
        smartStorageLvl: 9,
    };
}

export function unemployEveryone(ns: NS, divisionName: string, city: CityName) {
    const corp = ns.corporation;
    const office = corp.getOffice(divisionName, city);
    Object.entries(office.employeeJobs).map(([job]) => {
        return corp.setAutoJobAssignment(divisionName, city, job, 0);
    });
}

export function ChangeEmployeeRatio(ns: NS, divisionName: string, city: CityName, ratio: Ratio) {
    const corp = ns.corporation;

    unemployEveryone(ns, divisionName, city);
    const office = corp.getOffice(divisionName, city);
    const employees = office.numEmployees;

    Object.entries(ratio).map(([job, ratio]) => {
        return corp.setAutoJobAssignment(divisionName, city, job, Math.round(employees * ratio));
    });
}

export async function initializeDivision(ns: NS, type: DivisionType) {
    const corp = ns.corporation;

    const divisionName = type.name;

    corp.expandIndustry(type.industry, divisionName);
    // 2. expand argo to all cities
    const cities: CityName[] = Object.values(CityName);
    for (const city of cities) {
        corp.expandCity(divisionName, city);
        const office = corp.getOffice(divisionName, city);
        // 3. assign three employees in each city to R&D until 100(?) RP WHILE giving Tea/Party
        for (let i = 0; i < office.size; i++) {
            corp.hireEmployee(divisionName, city, "Unassigned");
        }

        unemployEveryone(ns, divisionName, city);
        Object.entries(office.employeeJobs).map(() => {
            return corp.setAutoJobAssignment(divisionName, city, "Research & Development", office.size);
        });
    }
    // maximize morale and energy

    while (true) {
        const divisionCities = corp.getDivision(divisionName).cities;

        let canStop = true;
        for (const city of divisionCities) {
            const office = corp.getOffice(divisionName, city);
            const energy = office.avgEnergy;
            const morale = office.avgMorale;

            if (energy < 0.99) {
                corp.buyTea(divisionName, city);
                canStop = false;
            }
            if (morale < 0.99) {
                corp.throwParty(divisionName, city, 200_000);
                canStop = false;
            }
        }

        if (canStop) {
            break;
        }

        await ns.corporation.nextUpdate();
    }

    // 4. wait for 30 RP
    while (corp.getDivision(divisionName).researchPoints < type.minResearch) await corp.nextUpdate();

    // 5. buy some adVert
    while (corp.getHireAdVertCount(divisionName) < type.totalAdVerts) corp.hireAdVert(divisionName);

    // 6. upgrade all warehouses to lvl 6 and smart storage to lvl 9
    for (const city of corp.getDivision(divisionName).cities) {
        const lvl = corp.getWarehouse(divisionName, city).level;
        const targetLvl = type.warehouseLvl;
        if (lvl < targetLvl) {
            for (let i = 0; i < targetLvl - lvl; i++) {
                corp.upgradeWarehouse(divisionName, city);
            }
        }

        const smartStorageLvl = corp.getUpgradeLevel("Smart Storage");
        if (smartStorageLvl < type.smartStorageLvl)
            for (let i = 0; i < type.smartStorageLvl - smartStorageLvl; i++) {
                corp.levelUpgrade("Smart Storage");
            }
    }

    // 7. assign one employee each to: operations, engineer, and business
    for (const city of cities) {
        unemployEveryone(ns, divisionName, city);
        corp.setAutoJobAssignment(divisionName, "Aevum", "Operations", 1);
        corp.setAutoJobAssignment(divisionName, "Aevum", "Engineer", 1);
        corp.setAutoJobAssignment(divisionName, "Aevum", "Business", 1);
    }

    // 8. buy boost materials
    await buyBoostMaterial(ns, divisionName);
}

export async function developNewProduct(ns: NS) {
    const productBaseName = "gorg";

    const division = ns.corporation.getDivision("Tobacco");

    // get next product number
    const products = division.products;

    let minId = Infinity;
    let maxId = 0;
    products.forEach((product) => {
        const id = parseInt(product.replace(productBaseName, ""));
        if (id < minId) minId = id;
        if (id > maxId) maxId = id;
    });

    ns.print(minId);
    ns.print(maxId);

    const newProductName = productBaseName + (maxId + 1);

    ns.corporation.discontinueProduct("Tobacco", productBaseName + minId);
    ns.corporation.makeProduct("Tobacco", "Sector-12", newProductName, 10000000000000, 10000000000000);

    let developmentProgress = 0;
    while (developmentProgress < 100) {
        ns.print(developmentProgress);
        await ns.corporation.nextUpdate();
        developmentProgress = ns.corporation.getProduct("Tobacco", "Sector-12", newProductName).developmentProgress;
    }

    ns.corporation.sellProduct("Tobacco", "Sector-12", newProductName, "MAX", "MP", true);
    ns.corporation.setProductMarketTA2("Tobacco", newProductName, true);
}
