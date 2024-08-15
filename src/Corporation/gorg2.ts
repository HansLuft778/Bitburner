import { NS } from "@ns";
import { developNewProduct } from "./lib";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const corp = ns.corporation;

    await developNewProduct(ns);
}
