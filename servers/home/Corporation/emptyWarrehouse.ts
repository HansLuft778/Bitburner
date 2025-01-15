export async function main(ns: NS) {
    ns.tail()
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
