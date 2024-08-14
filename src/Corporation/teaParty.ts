import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    const corp = ns.corporation;

    const COST_PER_EMPOLYEE = 200_000;
    const ENERGY_THRESHOLD = 109;
    const MORALE_THRESHOLD = 109;

    let lastState = "SALE";
    while (true) {
        if (lastState !== "SALE") {
            lastState = await corp.nextUpdate();
            continue;
        }

        const divisions = corp.getCorporation().divisions;

        for (const division of divisions) {
            const divisionCities = corp.getDivision(division).cities;
            for (const city of divisionCities) {
                const office = corp.getOffice(division, city);
                const energy = office.avgEnergy;
                const morale = office.avgMorale;

                ns.print(`City: ${city}, Energy: ${ns.formatNumber(energy)}, Morale: ${ns.formatNumber(morale)}`);

                if (energy < ENERGY_THRESHOLD) corp.buyTea(division, city);
                if (morale < MORALE_THRESHOLD) corp.throwParty(division, city, COST_PER_EMPOLYEE);
            }
        }

        lastState = await corp.nextUpdate();
    }
}
