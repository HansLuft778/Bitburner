/** @param {NS} ns */
export async function main(ns) {

	const hostname = ns.getHostname()
	let growThreashold = 0.1
	
	// first weaken the server
	let minSecLvl = ns.getServerMinSecurityLevel(hostname)
	let curSecLvl = ns.getServerSecurityLevel(hostname);
	let numWeakens = Math.ceil(curSecLvl - minSecLvl) / 0.05

	

	if (numWeakens != 0) {

	}

	while(true) {
		await ns.hack(hostname);
	}
}