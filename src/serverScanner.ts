import { NS } from "@ns";

export function serverScanner(ns: NS): string[] {
    let uncheckedHosts = ["home"];
    let checkedHosts = [];

    for (let i = 0; i < uncheckedHosts.length; i++) {
        let newHosts = ns.scan(uncheckedHosts[i]);
        checkedHosts.push(uncheckedHosts[i]);

        for (let n = 0; n < newHosts.length; n++) {
            if (checkedHosts.includes(newHosts[n]) == false || uncheckedHosts.includes(newHosts[n]) == false) {
                uncheckedHosts.push(newHosts[n]);
            }
        }
    }

    return checkedHosts.sort();
}

export function isHackable(ns: NS, server: string): boolean {
    ns.print(server);
    if (
        ns.getServerNumPortsRequired(server) <= getNumHacks(ns) &&
        ns.getServerRequiredHackingLevel(server) <= ns.getHackingLevel()
    )
        return true;
    else return false;
}

export function getNumHacks(ns: NS): number {
    let i = 0;
    if (ns.fileExists("BruteSSH.exe")) i++;
    if (ns.fileExists("FTPCrack.exe")) i++;
    if (ns.fileExists("HTTPWorm.exe")) i++;
    if (ns.fileExists("SQLInject.exe")) i++;
    return i;
}
