import { Division } from "@/NetscriptDefinitions.js";

function doForEveryting(
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

const doAll = function* (ns: NS): Generator<[Division, string]> {
    for (const divisionName of ns.corporation.getCorporation().divisions) {
        const division = ns.corporation.getDivision(divisionName);
        for (const city of division.cities) {
            yield [division, city];
        }
    }
};

export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();

    // doForEveryting(ns, (division, city) => {
    //     ns.print(division, " ", city);
    // });

    // for (const [division, city] of doAll(ns)) {
    //     ns.print(division.name, " ", city);
    // }

    ns.print("\nabc".length)
}
