import { NS } from "@ns";

export async function main(ns: NS, host: string) {
    await ns.weaken(host);
}
