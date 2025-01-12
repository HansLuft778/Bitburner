import { developNewProduct } from "./lib.js";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const corp = ns.corporation;

    await developNewProduct(ns);
}
