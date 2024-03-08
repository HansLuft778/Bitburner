import { CorpIndustryName, NS } from "@ns";
import { CityName } from "./lib";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    const corp = ns.corporation;
    // ns.corporation.throwParty();
    // ns.corporation.buyTea("lettuce begin", "Sector-12");

    // ns.corporation.getCorporation();

    if (corp.hasCorporation()) throw new Error("You already have a corporation");

    const res = corp.createCorporation("Apple", false);
    if (!res) throw new Error("Failed to create corporation");

    // --------------------- PHASE 1 ---------------------
    // 1. create argi division
    const divisionName = "Lettuce begin";
    // check if division already exists
    try {
        const div = ns.corporation.getDivision(divisionName);
        ns.print(div);
    } catch (error) {
        ns.print("Division does not exist, creating...");
        createDivision(ns, divisionName, "Agriculture");
    }
}

function createDivision(ns: NS, divisionName: string, industry: CorpIndustryName) {
    const corp = ns.corporation;

    corp.expandIndustry(industry, divisionName);
    // 2. expand argo to all cities
    const cities: CityName[] = Object.values(CityName);
    for (const city of cities) {
        corp.expandCity(divisionName, city);
        const office = corp.getOffice(divisionName, city);
        // 3. assign three employees in each city to R&D until 100(?) RP WHILE giving Tea/Party
        for (let i = 0; i < office.size; i++) {
            corp.hireEmployee(divisionName, city, "Unassigned");
        }
        Object.entries(office.employeeJobs).map(([name]) => {
            return ns.corporation.setAutoJobAssignment(divisionName, city, name, 0);
        });
    }
    // 4. assign one employee each to: operations, engineer, and business
    corp.setAutoJobAssignment(divisionName, "Aevum", "Operations", 1);
    corp.setAutoJobAssignment(divisionName, "Aevum", "Engineer", 1);
    corp.setAutoJobAssignment(divisionName, "Aevum", "Business", 1);
    // 5. warehouse in each city to 2k?
    // 6.
}
