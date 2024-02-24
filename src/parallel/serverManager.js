/** @param {NS} ns */
export async function main(ns) {

}

/** @param {NS} ns */
export function buyServer(ns, ram) {

	const limit = ns.getPurchasedServerLimit()
	const all = ns.getPurchasedServers()
	if (all.length >= limit) {
		ns.print("[server Manager] attempted to buy a new server, but the limit has been reached")
		return false
	}

	let price = ns.getPurchasedServerCost(ram)
	const name = getNextServerName(ns)
	ns.purchaseServer(name, buyRam)

	return true
}

/** @param {NS} ns */
export function upgradeServer(ns, ram, name) {
	// get current ram
	const servers = ns.getPurchasedServers();
	if (!servers.includes(upgradeName)) {
		ns.print("[server Manager] attempted to upgrade Server, but the server does not exist")
		return false
	}

	let price = ns.getPurchasedServerUpgradeCost(upgradeName, upgradeRam)

	ns.upgradePurchasedServer(upgradeName, upgradeRam)

	return true
}

/** @param {NS} ns */
function getNextServerName(ns) {
	const allServers = ns.getPurchasedServers()
	const filteredStrings = allServers.filter(str => str.startsWith("aws-"));

	// get highest number and construct the new name
	const numbers = filteredStrings.map(server => parseInt(server.split("-")[1]))
	const max = Math.max(...numbers)
	const newName = "aws-" + (max + 1)
	return newName
}