/** @param {NS} ns */
export async function main(ns, hostname) {
    await ns.weaken(hostname);
}
