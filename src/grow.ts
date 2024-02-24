import { NS } from "@ns";
export async function main(ns: NS) {
    if (typeof ns.args[0] !== "string") throw new TypeError("First argument must be a string");
    await ns.grow(ns.args[0]);
}
