import {
    CorpIndustryData,
    CorpIndustryName,
    CorpMaterialName,
    Division
} from "@/NetscriptDefinitions.js";
import { Colors } from "../lib.js";

export enum CityName {
    Aevum = "Aevum",
    Chongqing = "Chongqing",
    Sector12 = "Sector-12",
    NewTokyo = "New Tokyo",
    Ishima = "Ishima",
    Volhaven = "Volhaven"
}

enum BoostMaterialsSizes {
    AiCores = 0.1,
    Hardware = 0.06,
    RealEstate = 0.005,
    Robots = 0.5
}

export function forEveryDivisionAndCity(
    ns: NS,
    func: (division: Division, city: string) => void
) {
    for (const divisionName of ns.corporation.getCorporation().divisions) {
        const division = ns.corporation.getDivision(divisionName);
        for (const city of division.cities) {
            func(division, city);
        }
    }
}

export function getOptimalBoostMaterialQuantities(
    industryData: CorpIndustryData,
    spaceConstraint: number,
    round = true
): number[] {
    const { aiCoreFactor, hardwareFactor, realEstateFactor, robotFactor } =
        industryData;
    if (
        aiCoreFactor === undefined ||
        hardwareFactor === undefined ||
        realEstateFactor === undefined ||
        robotFactor === undefined
    ) {
        throw new Error("Industry data is missing factors");
    }
    const boostMaterialCoefficients = [
        aiCoreFactor,
        hardwareFactor,
        realEstateFactor,
        robotFactor
    ];
    const boostMaterialSizes = [
        BoostMaterialsSizes.AiCores,
        BoostMaterialsSizes.Hardware,
        BoostMaterialsSizes.RealEstate,
        BoostMaterialsSizes.Robots
    ];

    const calculateOptimalQuantities = (
        matCoefficients: number[],
        matSizes: number[]
    ): number[] => {
        const sumOfCoefficients = matCoefficients.reduce((a, b) => a + b, 0);
        const sumOfSizes = matSizes.reduce((a, b) => a + b, 0);
        const result = [];
        for (let i = 0; i < matSizes.length; ++i) {
            let matCount =
                (spaceConstraint -
                    500 *
                        ((matSizes[i] / matCoefficients[i]) *
                            (sumOfCoefficients - matCoefficients[i]) -
                            (sumOfSizes - matSizes[i]))) /
                (sumOfCoefficients / matCoefficients[i]) /
                matSizes[i];
            if (matCoefficients[i] <= 0 || matCount < 0) {
                return calculateOptimalQuantities(
                    matCoefficients.toSpliced(i, 1),
                    matSizes.toSpliced(i, 1)
                ).toSpliced(i, 0, 0);
            } else {
                if (round) {
                    matCount = Math.round(matCount);
                }
                result.push(matCount);
            }
        }
        return result;
    };
    return calculateOptimalQuantities(
        boostMaterialCoefficients,
        boostMaterialSizes
    );
}

export async function buyBoostMaterial(ns: NS, divisionName: string) {
    const corp = ns.corporation;

    const division = corp.getDivision(divisionName);

    const data = corp.getIndustryData(division.type);

    const materials: CorpMaterialName[] = [
        "AI Cores",
        "Hardware",
        "Real Estate",
        "Robots"
    ];

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
                const warehouseSpace =
                    corp.getWarehouse(divisionName, city).size * 0.8;
                const neededMaterial = getOptimalBoostMaterialQuantities(
                    data,
                    warehouseSpace
                );
                const storedMaterial = corp.getMaterial(
                    divisionName,
                    city,
                    material
                ).stored;
                const toBuy =
                    Math.max(neededMaterial[i] - storedMaterial, 0) / 10;
                ns.print(
                    `Buying ${toBuy}/${neededMaterial[i]} ${material} in ${city}`
                );
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

export interface OfficeRatio {
    Operations: number;
    Engineer: number;
    Business: number;
    Management: number;
}
export class OfficeRatios {
    /**
     * My setup for round 3
     */
    static progressRatio: OfficeRatio = {
        Operations: 0.18,
        Engineer: 0.5,
        Business: 0.15,
        Management: 0.17
    };

    /**
     * switch to it by end of round to increase offer
     */
    static profitRatio: OfficeRatio = {
        Operations: 0.032,
        Engineer: 0.462,
        Business: 0.067,
        Management: 0.439
    };
}

// Round 3+ budgets and ratios
export class DivisionBudget {
    static agriculture = 500e9;
    static chemical = 100e9;
}

export class ProductDivisionFunds {
    static beforeFocusOnAdvert = {
        rawProduction: 1 / 23,
        wilsonAdvert: 4 / 23,
        office: 8 / 23,
        employeStatUpgrades: 8 / 23,
        salesBot: 1 / 23,
        projectInsight: 1 / 23
    };
    static afterFocusOnAdvert = {
        rawProduction: 1 / 19,
        wilsonAdvert: 0,
        office: 8 / 19,
        employeStatUpgrades: 1 / 19,
        salesBot: 1 / 19,
        projectInsight: 1 / 19
    };
}

export const MAIN_OFFICE: CityName = "Sector-12" as CityName;

/**
 * For support division, funds ratio of warehouse to offices: 10-90.
 */
export const SUPPORT_DIVISION_WAREHOUSE_RATIO = 0.1;

/**
 * Funds ratio of main office to support offices: Round 3: 75-25
 */
export const MAIN_TO_SUPPORT_OFFICE_RATIO_R3 = 0.75;

/**
 * Funds ratio of main office to support offices: Round 4+: 50-50.
 */
export const MAIN_TO_SUPPORT_OFFICE_RATIO_R4 = 0.5;

interface DivisionType {
    name: string;
    industry: CorpIndustryName;
    minResearch: number;
    totalAdVerts: number;
    warehouseLvl: number;
    smartStorageLvl: number;
    officeSize: number;
}

export class Divisions {
    static Agriculture: DivisionType = {
        name: "Lettuce begin",
        industry: "Agriculture",
        minResearch: 55,
        totalAdVerts: 2,
        warehouseLvl: 6,
        smartStorageLvl: 6,
        officeSize: 4
    };

    static Chemical: DivisionType = {
        name: "BASF",
        industry: "Chemical",
        minResearch: 400,
        totalAdVerts: 0,
        warehouseLvl: 2,
        smartStorageLvl: 0,
        officeSize: 3
    };
}

export function unemployEveryone(ns: NS, divisionName: string, city: CityName) {
    const corp = ns.corporation;
    const office = corp.getOffice(divisionName, city);
    Object.entries(office.employeeJobs).map(([job]) => {
        return corp.setAutoJobAssignment(divisionName, city, job, 0);
    });
}

export function ChangeEmployeeRatio(
    ns: NS,
    divisionName: string,
    city: CityName,
    ratio: OfficeRatio
) {
    const corp = ns.corporation;

    unemployEveryone(ns, divisionName, city);
    const office = corp.getOffice(divisionName, city);
    const employees = office.numEmployees;

    Object.entries(ratio).map(([job, ratio]) => {
        return corp.setAutoJobAssignment(
            divisionName,
            city,
            job,
            Math.round(employees * ratio)
        );
    });
}

export async function initializeDivision(ns: NS, type: DivisionType) {
    const corp = ns.corporation;

    const divisionName = type.name;

    corp.expandIndustry(type.industry, divisionName);
    // 2. expand division to all cities
    const cities: CityName[] = Object.values(CityName);
    for (const city of cities) {
        if (city != "Sector-12") corp.expandCity(divisionName, city);

        if (corp.getOffice(type.name, city).size < type.officeSize) {
            corp.upgradeOfficeSize(
                divisionName,
                city,
                type.officeSize - corp.getOffice(type.name, city).size
            );
            ns.print(
                `upgradeded office size of ${divisionName} in ${city} to ${
                    corp.getOffice(type.name, city).size
                }`
            );
        }

        corp.purchaseWarehouse(type.name, city);

        const office = corp.getOffice(divisionName, city);
        // 3. assign every employees in each city to R&D
        for (let i = 0; i < office.size; i++) {
            corp.hireEmployee(divisionName, city, "Unassigned");
        }

        ns.print("assigning emloyees to RnD");
        unemployEveryone(ns, divisionName, city);
        Object.entries(office.employeeJobs).map(() => {
            return corp.setAutoJobAssignment(
                divisionName,
                city,
                "Research & Development",
                office.size
            );
        });
    }

    // maximize morale and energy
    ns.print("waiting for energy/morale to improve...");
    while (true) {
        const divisionCities = corp.getDivision(divisionName).cities;

        let canStop = true;
        for (const city of divisionCities) {
            const office = corp.getOffice(divisionName, city);
            const energy = office.avgEnergy;
            const morale = office.avgMorale;

            if (energy < 99) {
                corp.buyTea(divisionName, city);
                canStop = false;
            }
            if (morale < 99) {
                corp.throwParty(divisionName, city, 200_000);
                canStop = false;
            }
        }

        if (canStop) {
            break;
        }

        await ns.corporation.nextUpdate();
    }

    while (corp.getDivision(divisionName).researchPoints < type.minResearch) {
        await corp.nextUpdate();
    }

    // 5. buy some adVert
    ns.print(`buying ${type.totalAdVerts} adverts`);
    while (corp.getHireAdVertCount(divisionName) < type.totalAdVerts)
        corp.hireAdVert(divisionName);

    // 6. upgrade all warehouses and smart storage
    for (const city of corp.getDivision(divisionName).cities) {
        const lvl = corp.getWarehouse(divisionName, city).level;
        const targetLvl = type.warehouseLvl;
        if (lvl < targetLvl) {
            corp.upgradeWarehouse(divisionName, city, targetLvl - lvl);
        }

        const smartStorageLvl = corp.getUpgradeLevel("Smart Storage");
        if (smartStorageLvl < type.smartStorageLvl) {
            for (let i = 0; i < type.smartStorageLvl - smartStorageLvl; i++) {
                corp.levelUpgrade("Smart Storage");
            }
        }
    }

    if (type.name == Divisions.Agriculture.name) {
        // 7. assign one employee each to: operations, engineer, and business
        for (const city of cities) {
            unemployEveryone(ns, divisionName, city);
            corp.setAutoJobAssignment(divisionName, city, "Operations", 1);
            corp.setAutoJobAssignment(divisionName, city, "Engineer", 1);
            corp.setAutoJobAssignment(divisionName, city, "Business", 1);
            corp.setAutoJobAssignment(divisionName, city, "Management", 1);
        }

        for (const city of cities) {
            corp.sellMaterial(divisionName, city, "Plants", "MAX", "MP");
            corp.sellMaterial(divisionName, city, "Food", "MAX", "MP");
        }
    }
    if (type.name == Divisions.Chemical.name) {
        if (!corp.hasUnlock("Export")) corp.purchaseUnlock("Export");
        ns.print("Setting up export routes");
        for (const divisionName of corp.getCorporation().divisions) {
            const div = corp.getDivision(divisionName);
            for (const city of div.cities) {
                if (div.type == "Agriculture") {
                    // Plants from Agri to Chem
                    corp.exportMaterial(
                        Divisions.Agriculture.name,
                        city,
                        type.name,
                        city,
                        "Plants",
                        "(IPROD+IINV/10)*(-1)"
                    );
                } else if (div.type == "Chemical") {
                    // Chemicals from Chem to Agri
                    corp.exportMaterial(
                        type.name,
                        city,
                        Divisions.Agriculture.name,
                        city,
                        "Chemicals",
                        "(IPROD+IINV/10)*(-1)"
                    );
                }
            }
        }
    }
    // 8. buy boost materials
    await buyBoostMaterial(ns, divisionName);

    if (type.name == Divisions.Agriculture.name) {
        await corp.nextUpdate();
        // enable custom smart supply as late as possible cause it sucks
        if (!ns.scriptRunning("Corporation/smartSupply.js", "home")) {
            ns.run("Corporation/smartSupply.js");
        }
    }
}

export async function developNewProduct(ns: NS) {
    const productBaseName = "gorg-";

    const division = ns.corporation.getDivision("Tobacco");

    // get next product number
    const products = division.products;

    let ids: number[] = [];
    for (const product of ns.corporation.getDivision("Tobacco").products) {
        const parts = product.split("-");
        ids.push(parseInt(parts[1]));
    }
    const productName = productBaseName + Math.max(...ids) + 1;

    if (products.length === division.maxProducts) {
        ns.corporation.discontinueProduct("Tobacco", division.products[0]);
    }
    const totalFunds = ns.corporation.getCorporation().funds;
    ns.print(totalFunds);
    ns.corporation.makeProduct(
        "Tobacco",
        "Sector-12",
        productName,
        totalFunds * 0.01,
        totalFunds * 0.01
    );

    let developmentProgress = 0;
    while (developmentProgress < 100) {
        await ns.corporation.nextUpdate();
        developmentProgress = ns.corporation.getProduct(
            "Tobacco",
            "Sector-12",
            productName
        ).developmentProgress;
    }

    ns.corporation.sellProduct(
        "Tobacco",
        "Sector-12",
        productName,
        "MAX",
        "MP",
        true
    );
    ns.corporation.setProductMarketTA2("Tobacco", productName, true);
}

export async function acceptBestInvestmentOffer(
    ns: NS,
    minOffer: number
): Promise<number> {
    let previousOffer = 0;

    while (true) {
        if (ns.corporation.getCorporation().prevState != "START") {
            await ns.corporation.nextUpdate();
            continue;
        }
        const offer = ns.corporation.getInvestmentOffer().funds;
        if (offer < previousOffer && offer > minOffer) {
            if (!ns.corporation.acceptInvestmentOffer()) {
                ns.tprint(
                    Colors.RED +
                        "[ERROR] Investment offer could not be accepted for some reason"
                );
                ns.exit();
            }
            return offer;
        }
        previousOffer = offer;
        await ns.corporation.nextUpdate();
    }
}

export async function emptyAllWarehouses(ns: NS) {
    const corp = ns.corporation;
    while (corp.getCorporation().nextState != "SALE") {
        await corp.nextUpdate();
    }

    for (const divisionName of corp.getCorporation().divisions) {
        const division = corp.getDivision(divisionName);
        for (const city of division.cities) {
            const requiredMaterials = corp.getIndustryData(
                division.type
            ).requiredMaterials;

            for (const materialName of Object.keys(requiredMaterials)) {
                ns.tprint(`selling ${materialName}`);
                // Clear purchase
                ns.corporation.buyMaterial(
                    division.name,
                    city,
                    materialName,
                    0
                );
                // Discard stored input material
                ns.corporation.sellMaterial(
                    division.name,
                    city,
                    materialName,
                    "MAX",
                    "0"
                );
            }
        }
    }

    await corp.nextUpdate();

    for (const divisionName of corp.getCorporation().divisions) {
        const division = corp.getDivision(divisionName);
        for (const city of division.cities) {
            const requiredMaterials = corp.getIndustryData(
                division.type
            ).requiredMaterials;
            for (const materialName of Object.keys(requiredMaterials)) {
                ns.tprint(`stop selling ${materialName}`);
                const material = ns.corporation.getMaterial(
                    division.name,
                    city,
                    materialName
                );
                if (material.desiredSellAmount !== 0) {
                    // Stop discarding input material
                    ns.corporation.sellMaterial(
                        division.name,
                        city,
                        materialName,
                        "0",
                        "0"
                    );
                }
            }
        }
    }
}

export function convertDivisionToSupportOffices(
    ns: NS,
    division: Division,
    ratio: OfficeRatio
) {
    const corp = ns.corporation;

    for (const city of division.cities) {
        const officeSize = corp.getOffice(division.name, city).size;
        if (city == MAIN_OFFICE) {
            // apply ratio
            const numOp = Math.round(officeSize * ratio.Operations);
            const numEng = Math.round(officeSize * ratio.Engineer);
            const numMn = Math.round(officeSize * ratio.Management);
            const numBus = officeSize - (numOp + numEng + numMn);

            ns.print(city + " " + officeSize);
            ns.print(numOp);
            ns.print(numEng);
            ns.print(numMn);
            ns.print(numBus);

            ns.print("sum: " + (numOp + numBus + numEng + numMn));

            unemployEveryone(ns, division.name, city);

            corp.setAutoJobAssignment(division.name, city, "Operations", numOp);
            corp.setAutoJobAssignment(division.name, city, "Engineer", numEng);
            corp.setAutoJobAssignment(division.name, city, "Business", numBus);
            corp.setAutoJobAssignment(division.name, city, "Management", numMn);
        } else {
            // 1 to each Operation, Engineer, Business
            // rest to RnD

            unemployEveryone(ns, division.name, city);
            corp.setAutoJobAssignment(division.name, city, "Operations", 1);
            corp.setAutoJobAssignment(division.name, city, "Engineer", 1);
            corp.setAutoJobAssignment(division.name, city, "Business", 1);
            corp.setAutoJobAssignment(
                division.name,
                city,
                "Research & Development",
                officeSize - 3
            );
        }
    }
}

export function officeUpgradeCostFromAtoB(a: number, b: number) {
    return 4e9 * ((1.09 ** (b / 3) - 1.09 ** (a / 3)) / 0.09);
}
