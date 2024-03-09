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

    const neededMaterial = getOptimalBoostMaterialQuantities(data, 700);
    ns.print(neededMaterial);

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
