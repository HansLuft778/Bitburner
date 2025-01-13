import {
    CorpIndustryData,
    CorpIndustryName,
    CorpMaterialName,
    Division,
    Material,
    Product,
    Warehouse
} from "@/NetscriptDefinitions.js";
import { CorporationState } from "./CorpState.js";
import { Colors } from "../lib.js";

type PartialRecord<K extends string, V> = Partial<Record<K, V>>;

const smartSupplyData: Map<string, number> = new Map<string, number>();

const setOfDivisionsWaitingForRP: Set<string> = new Set<string>();

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

export class Ratio {
    static progressRatio = {
        Operations: 0.037,
        Engineer: 0.513,
        Business: 0.011,
        Management: 0.44
    };

    static profitRatio = {
        Operations: 0.032,
        Engineer: 0.462,
        Business: 0.067,
        Management: 0.439
    };

    static profitRatioEndgame = {
        Operations: 0.064,
        Engineer: 0.317,
        Business: 0.298,
        Management: 0.321
    };
}

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
        minResearch: 700,
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
    ratio: Ratio
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
    setOfDivisionsWaitingForRP.add(divisionName);
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

    setOfDivisionsWaitingForRP.delete(divisionName);

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

        // sell products
        if (!ns.scriptRunning("Corporation/smartSupply.js", "home")) {
            ns.run("Corporation/smartSupply.js");
        }
        // if (!corp.hasUnlock("Smart Supply"))
        //     corp.purchaseUnlock("Smart Supply");
        for (const city of cities) {
            corp.sellMaterial(divisionName, city, "Plants", "MAX", "MP");
            corp.sellMaterial(divisionName, city, "Food", "MAX", "MP");
            // corp.setSmartSupply(divisionName, city, true);
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

    const newProductName = productBaseName + (maxId + 1);

    if (products.length === division.maxProducts) {
        ns.corporation.discontinueProduct("Tobacco", productBaseName + minId);
    }
    const totalFunds = ns.corporation.getCorporation().funds;
    ns.print(totalFunds);
    ns.corporation.makeProduct(
        "Tobacco",
        "Sector-12",
        newProductName,
        totalFunds * 0.01,
        totalFunds * 0.01
    );

    let developmentProgress = 0;
    while (developmentProgress < 100) {
        await ns.corporation.nextUpdate();
        developmentProgress = ns.corporation.getProduct(
            "Tobacco",
            "Sector-12",
            newProductName
        ).developmentProgress;
    }

    ns.corporation.sellProduct(
        "Tobacco",
        "Sector-12",
        newProductName,
        "MAX",
        "MP",
        true
    );
    ns.corporation.setProductMarketTA2("Tobacco", newProductName, true);
}

export async function customSmartSupply(
    ns: NS,
    warehouseCongestionData: Map<string, number>
) {
    const corp = ns.corporation;
    if (corp.getCorporation().nextState != "PURCHASE") return;

    for (const divisionName of corp.getCorporation().divisions) {
        const div = corp.getDivision(divisionName);
        for (const city of div.cities) {
            const industryData = corp.getIndustryData(div.type);

            const office = corp.getOffice(divisionName, city);

            // check for warehouse congestion
            let isCongested = false;
            if (
                !setOfDivisionsWaitingForRP.has(divisionName) &&
                office.employeeJobs["Research & Development"] !==
                    office.numEmployees
            )
                isCongested = detectWarehouseCongestion(
                    ns,
                    div,
                    industryData,
                    city,
                    warehouseCongestionData
                );
            if (isCongested) return;

            const warehouse = ns.corporation.getWarehouse(divisionName, city);
            const inputMaterials: PartialRecord<
                CorpMaterialName,
                {
                    requiredQuantity: number;
                    coefficient: number;
                }
            > = {};

            for (const materialName of Object.keys(
                industryData.requiredMaterials
            )) {
                inputMaterials[materialName] = {
                    requiredQuantity: 0,
                    coefficient: industryData.requiredMaterials[materialName]
                };
            }

            // Find required quantity of input materials to produce material/product
            for (const inputMaterialData of Object.values(inputMaterials)) {
                const requiredQuantity =
                    (smartSupplyData.get(`${divisionName}|${city}`) ?? 0) *
                    inputMaterialData.coefficient;
                inputMaterialData.requiredQuantity += requiredQuantity;
            }

            // Limit the input material units to max number of units that we can store in warehouse's free space
            for (const materialName of Object.keys(inputMaterials)) {
                const inputMaterialData = inputMaterials[materialName];
                const materialData = ns.corporation.getMaterialData(
                    materialName as CorpMaterialName
                );
                const maxAcceptableQuantity = Math.floor(
                    (warehouse.size - warehouse.sizeUsed) / materialData.size
                );
                const limitedRequiredQuantity = Math.min(
                    inputMaterialData.requiredQuantity,
                    maxAcceptableQuantity
                );
                if (limitedRequiredQuantity > 0) {
                    inputMaterialData.requiredQuantity =
                        limitedRequiredQuantity;
                }
            }

            // Find which input material creates the least number of output units
            let leastAmountOfOutputUnits = Number.MAX_VALUE;
            for (const { requiredQuantity, coefficient } of Object.values(
                inputMaterials
            )) {
                const amountOfOutputUnits = requiredQuantity / coefficient;
                if (amountOfOutputUnits < leastAmountOfOutputUnits) {
                    leastAmountOfOutputUnits = amountOfOutputUnits;
                }
            }

            // Align all the input materials to the smallest amount
            for (const inputMaterialData of Object.values(inputMaterials)) {
                inputMaterialData.requiredQuantity =
                    leastAmountOfOutputUnits * inputMaterialData.coefficient;
            }

            // Calculate the total size of all input materials we are trying to buy
            let requiredSpace = 0;
            for (const materialName of Object.keys(inputMaterials)) {
                requiredSpace +=
                    inputMaterials[materialName].requiredQuantity *
                    ns.corporation.getMaterialData(
                        materialName as CorpMaterialName
                    ).size;
            }

            // If there is not enough free space, we apply a multiplier to required quantity to not overfill warehouse
            const freeSpace = warehouse.size - warehouse.sizeUsed;
            if (requiredSpace > freeSpace) {
                const constrainedStorageSpaceMultiplier =
                    freeSpace / requiredSpace;
                for (const inputMaterialData of Object.values(inputMaterials)) {
                    inputMaterialData.requiredQuantity = Math.floor(
                        inputMaterialData.requiredQuantity *
                            constrainedStorageSpaceMultiplier
                    );
                }
            }

            // Deduct the number of stored input material units from the required quantity
            for (const materialName of Object.keys(inputMaterials)) {
                const inputMaterialData = inputMaterials[materialName];
                const material = ns.corporation.getMaterial(
                    divisionName,
                    city,
                    materialName
                );
                inputMaterialData.requiredQuantity = Math.max(
                    0,
                    inputMaterialData.requiredQuantity - material.stored
                );
            }

            // Buy input materials
            for (const materialName of Object.keys(inputMaterials)) {
                ns.corporation.buyMaterial(
                    divisionName,
                    city,
                    materialName,
                    inputMaterials[materialName].requiredQuantity / 10
                );
            }
        }
    }
}

export function setSmartSupplyData(ns: NS): void {
    // Only set smart supply data after "PURCHASE" state
    if (ns.corporation.getCorporation().prevState !== "PURCHASE") {
        return;
    }

    const corp = ns.corporation;
    for (const divisionName of corp.getCorporation().divisions) {
        const div = corp.getDivision(divisionName);
        for (const city of div.cities) {
            const division = ns.corporation.getDivision(divisionName);
            const industrialData = ns.corporation.getIndustryData(
                division.type
            );
            const warehouse = ns.corporation.getWarehouse(division.name, city);
            let totalRawProduction = 0;

            if (industrialData.makesMaterials) {
                totalRawProduction += getLimitedRawProduction(
                    ns,
                    division,
                    city,
                    industrialData,
                    warehouse,
                    false,
                    -1
                );
            }

            if (industrialData.makesProducts) {
                for (const productName of division.products) {
                    const product = ns.corporation.getProduct(
                        divisionName,
                        city,
                        productName
                    );
                    if (product.developmentProgress < 100) {
                        continue;
                    }
                    totalRawProduction += getLimitedRawProduction(
                        ns,
                        division,
                        city,
                        industrialData,
                        warehouse,
                        true,
                        product.size
                    );
                }
            }

            smartSupplyData.set(`${divisionName}|${city}`, totalRawProduction);
        }
    }
}

function getRawProduction(
    ns: NS,
    divisionName: string,
    city: CityName,
    isProduct: boolean
) {
    const corp = ns.corporation;
    const office = corp.getOffice(divisionName, city);

    const OpProduction = office.employeeProductionByJob.Operations;
    const EngProduction = office.employeeProductionByJob.Engineer;
    const MngmtProduction = office.employeeProductionByJob.Management;

    const mngmtFactor =
        1 +
        MngmtProduction /
            (1.2 * (OpProduction + EngProduction + MngmtProduction));

    const emplProdMult =
        (OpProduction ** 0.4 + office.employeeProductionByJob.Engineer ** 0.3) *
        mngmtFactor;

    let officeMultiplier = 0.05 * emplProdMult;
    if (isProduct) officeMultiplier *= 0.5;

    let divisionProductionMult = corp.getDivision(divisionName).productionMult;

    let upgradeMult = 1 + corp.getUpgradeLevel("Smart Factories") * 0.03;

    let researchMult = 1;
    if (corp.hasResearched(divisionName, "Self-Correcting Assemblers"))
        researchMult *= 1.1;
    if (corp.hasResearched(divisionName, "Drones - Assembly"))
        researchMult *= 1.2;
    if (isProduct && corp.hasResearched(divisionName, "uPgrade: Fulcrum"))
        researchMult *= 1.05;

    return (
        officeMultiplier * divisionProductionMult * upgradeMult * researchMult
    );
}

function getLimitedRawProduction(
    ns: NS,
    division: Division,
    city: CityName,
    industrialData: CorpIndustryData,
    warehouse: Warehouse,
    isProduct: boolean,
    productSize?: number
) {
    let rawProduction = getRawProduction(ns, division.name, city, isProduct);

    // Calculate required storage space of each output unit. It is the net change in warehouse's storage space when
    // producing an output unit.
    let requiredStorageSpaceOfEachOutputUnit = 0;
    if (isProduct) {
        requiredStorageSpaceOfEachOutputUnit += productSize!;
    } else {
        for (const outputMaterialName of industrialData.producedMaterials!) {
            requiredStorageSpaceOfEachOutputUnit +=
                ns.corporation.getMaterialData(outputMaterialName).size;
        }
    }
    for (const requiredMaterialName of Object.keys(
        industrialData.requiredMaterials
    )) {
        const requiredMaterialCoefficient =
            industrialData.requiredMaterials[requiredMaterialName];
        requiredStorageSpaceOfEachOutputUnit -=
            ns.corporation.getMaterialData(
                requiredMaterialName as CorpMaterialName
            ).size * requiredMaterialCoefficient;
    }
    // Limit the raw production if needed
    if (requiredStorageSpaceOfEachOutputUnit > 0) {
        const maxNumberOfOutputUnits = Math.floor(
            (warehouse.size - warehouse.sizeUsed) /
                requiredStorageSpaceOfEachOutputUnit
        );
        rawProduction = Math.min(rawProduction, maxNumberOfOutputUnits);
    }

    rawProduction = Math.max(rawProduction, 0);
    return rawProduction;
}

function detectWarehouseCongestion(
    ns: NS,
    division: Division,
    industrialData: CorpIndustryData,
    city: CityName,
    warehouseCongestionData: Map<string, number>
): boolean {
    const requiredMaterials = industrialData.requiredMaterials;

    let isWarehouseCongested = false;
    const warehouseCongestionDataKey = `${division.name}|${city}`;
    const items: (Material | Product)[] = [];
    if (industrialData.producedMaterials) {
        for (const materialName of industrialData.producedMaterials) {
            items.push(
                ns.corporation.getMaterial(division.name, city, materialName)
            );
        }
    }
    if (industrialData.makesProducts) {
        for (const productName of division.products) {
            const product = ns.corporation.getProduct(
                division.name,
                city,
                productName
            );
            if (product.developmentProgress < 100) {
                continue;
            }
            items.push(product);
        }
    }
    for (const item of items) {
        if (item.productionAmount !== 0) {
            warehouseCongestionData.set(warehouseCongestionDataKey, 0);
            continue;
        }
        // item.productionAmount === 0 means that division does not produce material/product last cycle.
        let numberOfCongestionTimes =
            warehouseCongestionData.get(warehouseCongestionDataKey)! + 1;
        if (Number.isNaN(numberOfCongestionTimes)) {
            numberOfCongestionTimes = 0;
        }
        warehouseCongestionData.set(
            warehouseCongestionDataKey,
            numberOfCongestionTimes
        );
        break;
    }
    // If that happens more than 5 times, the warehouse is very likely congested.
    if (warehouseCongestionData.get(warehouseCongestionDataKey)! > 5) {
        isWarehouseCongested = true;
    }
    // We need to mitigate this situation. Discarding stored input material is the simplest solution.
    if (isWarehouseCongested) {
        ns.tprint(
            `Warehouse may be congested. Division: ${division.name}, city: ${city}.`
        );
        for (const materialName of Object.keys(requiredMaterials)) {
            ns.tprint(`selling ${materialName}`);
            // Clear purchase
            ns.corporation.buyMaterial(division.name, city, materialName, 0);
            // Discard stored input material
            ns.corporation.sellMaterial(
                division.name,
                city,
                materialName,
                "MAX",
                "0"
            );
        }
        warehouseCongestionData.set(warehouseCongestionDataKey, 0);
    } else {
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
    return isWarehouseCongested;
}

export async function acceptBestInvestmentOffer(ns: NS): Promise<number> {
    let previousOffer = 0;

    const numCycles = 10;
    let currentCycle = 0;

    while (true) {
        if (ns.corporation.getCorporation().prevState != "START") {
            await ns.corporation.nextUpdate();
            continue;
        }
        const offer = ns.corporation.getInvestmentOffer().funds;
        if (offer < previousOffer && currentCycle >= numCycles) {
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
        currentCycle++;
        await ns.corporation.nextUpdate();
    }
}
