import { NS } from "@ns";
export async function main(ns: NS) {
    if (typeof ns.args[0] !== "string") throw new TypeError("First argument must be a string");
    await ns.weaken(ns.args[0]);
}
