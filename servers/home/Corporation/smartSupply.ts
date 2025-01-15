import {
    CityName,
    CorpIndustryData,
    CorpMaterialName,
    Division,
    Material,
    Product,
    Warehouse
} from "@/NetscriptDefinitions.js";

const smartSupplyData: Map<string, number> = new Map<string, number>();
const setOfDivisionsWaitingForRP: Set<string> = new Set<string>();
type PartialRecord<K extends string, V> = Partial<Record<K, V>>;

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
                    false
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
    let rawProduction =
        getRawProduction(ns, division.name, city, isProduct) * 10;

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
        ns.print(`size: ${warehouse.size} used: ${warehouse.sizeUsed}`);
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

export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();
    const corp = ns.corporation;

    // CorporationState.getInstance().resetState();

    let smartSupplyHasBeenEnabledEverywhere = false;
    const warehouseCongestionData = new Map<string, number>();
    while (true) {
        for (const divisionName of corp.getCorporation().divisions) {
            const div = corp.getDivision(divisionName);
            for (const city of div.cities) {
                if (!smartSupplyHasBeenEnabledEverywhere) {
                    // Enable Smart Supply everywhere if we have unlocked this feature
                    setSmartSupplyData(ns);
                    customSmartSupply(ns, warehouseCongestionData);
                }
            }
        }

        await corp.nextUpdate();
    }
}
