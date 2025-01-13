import { customSmartSupply, developNewProduct, Division } from "./lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");
    ns.clearLog()

    const corp = ns.corporation;

    // customSmartSupply(ns);

    const req = corp.getIndustryData("Agriculture").requiredMaterials;
    ns.print(req)
    // print(Object.keys(req))
    // Object.keys(req).forEach((material) => {
    //     const size = req[material];
    //     ns.print(size);
    // });
}
