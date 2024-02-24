import { NS } from "@ns";

export function serverScanner(ns: NS) {
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

export function isHackable(ns: NS, server: string) {
    if (
        ns.getServerNumPortsRequired(server) <= getNumHacks(ns) &&
        ns.getServerRequiredHackingLevel(server) <= ns.getHackingLevel()
    )
        return true;
    else return false;
}

export function getNumHacks(ns: NS) {
    let i = 0;
    if (ns.fileExists("BruteSSH.exe")) i++;
    if (ns.fileExists("FTPCrack.exe")) i++;
    if (ns.fileExists("HTTPWorm.exe")) i++;
    if (ns.fileExists("SQLInject.exe")) i++;
    return i;
}

export function nukeAll(ns: NS) {
    const hosts = serverScanner(ns);
    for (let i = 0; i < hosts.length; i++) {
        // check if the host is hackable
        if (isHackable(ns, hosts[i])) {
            openPorts(ns, hosts[i]);
            ns.nuke(hosts[i]);

            // copy all scripts to the server
            ns.scp("hack.js", hosts[i]);
            ns.scp("grow.js", hosts[i]);
            ns.scp("weaken.js", hosts[i]);
        } else {
            continue;
        }
    }
}

export function openPorts(ns: NS, target: string) {
    if (ns.fileExists("BruteSSH.exe")) ns.brutessh(target);
    if (ns.fileExists("FTPCrack.exe")) ns.ftpcrack(target);
    if (ns.fileExists("HTTPWorm.exe")) ns.httpworm(target);
    if (ns.fileExists("SQLInject.exe")) ns.sqlinject(target);
}

export function getTimeH(timestamp: number) {
    if (timestamp == undefined || timestamp == null) timestamp = Date.now();

    const date = new Date(timestamp);
    date.setUTCHours(date.getUTCHours() + 1);
    const hours = date.getUTCHours().toString().padStart(2, "0");
    const minutes = date.getUTCMinutes().toString().padStart(2, "0");
    const seconds = date.getUTCSeconds().toString().padStart(2, "0");
    const milliseconds = date.getUTCMilliseconds().toString().padStart(3, "0");
    const formattedTime = `${hours}:${minutes}:${seconds}:${milliseconds}`;
    return formattedTime;
}
