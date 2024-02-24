import { NS } from "@ns";

export async function main(ns: NS) {
    const hostname = ns.getHostname();
    const growThreashold = 0.1;

    // first weaken the server
    const minSecLvl = ns.getServerMinSecurityLevel(hostname);
    const curSecLvl = ns.getServerSecurityLevel(hostname);
    const numWeakens = Math.ceil(curSecLvl - minSecLvl) / 0.05;

    if (numWeakens != 0) {
        // weaken the server
    }

    while (true) {
        await ns.hack(hostname);
    }
}
