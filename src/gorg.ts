import { NS } from "@ns";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const start = Date.now();
    for (let index = 0; index < 400; index++) {
        const exists = ns.fileExists("Formulas.exe", "home");
    }
    const end = Date.now();

    ns.print("Time: " + (end - start));
}
