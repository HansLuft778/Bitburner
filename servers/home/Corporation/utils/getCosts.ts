import { Divisions } from "../lib.js";

export function officeUpgradeCostFromAtoB(a: number, b: number) {
    return 4e9 * ((1.09 ** (b / 3) - 1.09 ** (a / 3)) / 0.09);
}

export async function main(ns: NS) {
    // office upgrade cost from a to b
    ns.tprint(ns.formatNumber(officeUpgradeCostFromAtoB(3, 30)));
}
