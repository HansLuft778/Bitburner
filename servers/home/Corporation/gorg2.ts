import { Divisions, wilsonAdvertisingOptimizer } from "./lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");
    ns.clearLog();

    const corp = ns.corporation;

    const data = wilsonAdvertisingOptimizer(
        ns,
        // corp.getCorporation().funds,
        1e30,
        corp.getDivision(Divisions.Agriculture.name)
    );

    for (const entry of data.slice(0, 10)) {
        ns.print(
            `wilson: ${entry[0]} advert: ${ns.formatNumber(entry[1])} cost: ${ns.formatNumber(
                entry[2]
            )} advertFactor: ${ns.formatNumber(entry[3])} ratio: ${ns.formatNumber(
                entry[4]
            )} cost per factor: ${ns.formatNumber(entry[5])}`
        );
    }

    ns.print("done");
}
