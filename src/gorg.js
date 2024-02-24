/** @param {NS} ns */
export async function main(ns) {
	ns.tail()

	let hosts = ns.scan()
	let numHacks = getNumHacks()


	ns.tprint(hosts)

	for (let i = 0; i <= hosts.length; i++) {
		ns.tprint("current: " + hosts[i])
		if (!ns.hasRootAccess(hosts[i]) && ns.getServerNumPortsRequired() <= numHacks) {
			openPorts(ns, hosts[i])
			ns.nuke(hosts[i])
		}

		ns.scp("/src/hacker.js", hosts[i])
		ns.exec("/src/hacker.js", hosts[i], 1)

	}



	function openPorts(host) {
		if (ns.fileExists("BruteSSH.exe")) ns.brutessh(host)
		if (ns.fileExists("FTPCrack.exe")) ns.ftpcrack(host)
		if (ns.fileExists("HTTPWorm.exe")) ns.httpworm(host)
		if (ns.fileExists("SQLInject.exe")) ns.sqlinject(host)
	}

	function getNumHacks() {
		let i;
		if (ns.fileExists("BruteSSH.exe")) i++
		if (ns.fileExists("FTPCrack.exe")) i++
		if (ns.fileExists("HTTPWorm.exe")) i++
		if (ns.fileExists("SQLInject.exe")) i++
		return i;
	}
}