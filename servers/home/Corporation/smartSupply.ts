import { customSmartSupply, setSmartSupplyData } from "./lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();
    const corp = ns.corporation;

    let smartSupplyHasBeenEnabledEverywhere = false;
    const warehouseCongestionData = new Map<string, number>();
    while (true) {
        for (const divisionName of corp.getCorporation().divisions) {
            ns.print("asdsda");
            const div = corp.getDivision(divisionName);
            for (const city of div.cities) {
                if (!smartSupplyHasBeenEnabledEverywhere) {
                    // Enable Smart Supply everywhere if we have unlocked this feature
                    setSmartSupplyData(ns);
                    customSmartSupply(
                        ns,
                        divisionName,
                        warehouseCongestionData
                    );
                }
            }
        }

        await corp.nextUpdate();
    }
}
