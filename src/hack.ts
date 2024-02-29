import { NS } from "@ns";
export async function main(ns: NS) {
    if (typeof ns.args[0] !== "string") throw new TypeError("[HACK] First argument must be a string");
    if (typeof ns.args[1] !== "number") throw new TypeError("[HACK] Second argument must be a number");
    await ns.hack(ns.args[0], { additionalMsec: ns.args[1] });
}
