import { NS } from "@ns";
import { getBestHostByRam, getBestHostByRamOptimized } from "./bestServer";
import { Colors } from "./lib";
export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    const start = Date.now();
    for (let i = 0; i < 400; i++) {
        const element = getBestHostByRam(ns);
    }
    const end = Date.now();
    for (let i = 0; i < 400; i++) {
        const element = getBestHostByRamOptimized(ns);
    }
    const end2 = Date.now();

    ns.formulas.skills.calculateExp(ns.getHackingLevel() + 1);
    ns.formulas.skills.calculateSkill;

    ns.print(Colors.GREEN + "normal: " + (end - start));
    ns.print(Colors.GREEN + "optimized: " + (end2 - end));
}
