import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    // ns.corporation.throwParty();
    // ns.corporation.buyTea("lettuce begin", "Sector-12");

    // ns.corporation.getCorporation();
    const divisions: string[] = ["Lettuce begin", "BASF"];

    const COST_PER_EMPOLYEE = 200_000;
    const ENERGY_THRESHOLD = 99;
    const MORALE_THRESHOLD = 99;

    while (true) {
        // const coorp = ns.corporation.getCorporation();

        for (const division of divisions) {
            const divisionCities = ns.corporation.getDivision(division).cities;
            for (const city of divisionCities) {
                const office = ns.corporation.getOffice(division, city);
                const energy = office.avgEnergy;
                const morale = office.avgMorale;

                ns.print(`City: ${city}, Energy: ${ns.formatNumber(energy)}, Morale: ${ns.formatNumber(morale)}`);

                if (energy < ENERGY_THRESHOLD) ns.corporation.buyTea(division, city);
                if (morale < MORALE_THRESHOLD) ns.corporation.throwParty(division, city, COST_PER_EMPOLYEE);
            }
        }

        const state = await ns.corporation.nextUpdate();

        ns.print(state);
    }
}
