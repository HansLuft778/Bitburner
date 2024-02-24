/** @param {NS} ns */
export async function main(ns) {
	ns.tail()
	weakenServer(ns, "foodnstuff", "hacker", 1)
}

/** @param {NS} ns */
export function weakenServer(ns, target, host, order) {

	let serverWeakenThreads = 0
	// calculate weakening threads based on the order
	if (order == 2) {
		// second weak only has to remove the sec increase from the grow before (more ram efficient)
		const serverMaxMoney = ns.getServerMaxMoney(target)
		const serverCurrentMoney = ns.getServerMoneyAvailable(target)
		let moneyMult = serverMaxMoney / serverCurrentMoney
		if (isNaN(moneyMult) || moneyMult == Infinity)
			moneyMult = 1
		const growThreads = Math.ceil(ns.growthAnalyze(target, moneyMult))

		const secIncrease = ns.growthAnalyzeSecurity(growThreads, target)

		serverWeakenThreads = Math.ceil(secIncrease / 0.05)
	} else if (order == 1) {
		// first weak has to weaken server to min from unknown sec lvl
		const serverSecLvl = ns.getServerSecurityLevel(target)
		serverWeakenThreads = Math.ceil((serverSecLvl - ns.getServerMinSecurityLevel(target)) / 0.05)
	} else {
		throw new Error("weaken order can only be either 1 or 2!")
	}

	if (serverWeakenThreads < 1) {
		ns.print("Weakenthreads are 0, skipping weak " + order)
		return false
	}

	// exec weaken.js with num of threads
	const weakenRam = 1.75
	const maxRam = ns.getServerMaxRam(host)
	const freeRam = maxRam - ns.getServerUsedRam(host)

	const threadSpace = Math.floor(freeRam / weakenRam)

	if (threadSpace < serverWeakenThreads)
		throw new Error("can't onehit weaken on server " + target + ".\nneed " + serverWeakenThreads + " Threads, only got " + threadSpace)

	ns.exec("weaken.js", host, serverWeakenThreads, target)
	return true

	// const happen = serverWeakenThreads / threadSpace
	// ns.print("will weaken in " + happen + " steps")

	// let howMany = 0
	// for (let i = 0; i < Math.floor(happen); i++) {
	// 	const weakenTime = ns.getWeakenTime(target)
	// 	ns.exec("weaken.js", host, threadSpace, target)
	// 	await ns.sleep(weakenTime + safetyMarginMs)
	// 	howMany += threadSpace;
	// 	ns.print("done with " + howMany + "/" + serverWeakenThreads + " weakens")
	// }
	// if (howMany < serverWeakenThreads) {
	// 	ns.print("need to weaken " + (serverWeakenThreads - howMany) + " more times")
	// 	const weakenTime = ns.getWeakenTime(target)
	// 	ns.exec("weaken.js", host, serverWeakenThreads - howMany, target)
	// 	await ns.sleep(weakenTime + safetyMarginMs)
	// }
}