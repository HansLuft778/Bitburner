/** @param {NS} ns */
export async function main(ns) {
	ns.tail()
	await hackServer(ns, "foodnstuff", "hacker", 0.8)
}

/** @param {NS} ns */
export async function hackServer(ns, target, host, threshold) {

	const safetyMarginMs = 200

	const serverMaxMoney = ns.getServerMaxMoney(target)
	const lowerMoneyBound = serverMaxMoney * threshold
	const hackAmount = serverMaxMoney - lowerMoneyBound
	
	//const hackChance = ns.hackAnalyzeChance(target) // todo
	const hackThreads = Math.ceil(ns.hackAnalyzeThreads(target, hackAmount))

	ns.print("max money: " + serverMaxMoney + " 80%: " + lowerMoneyBound + " hack amount: " + hackAmount)
	ns.print("hack threads: " + hackThreads + " money available: " + ns.getServerMoneyAvailable(target))

	const hackingRam = 1.7
	const maxRam = ns.getServerMaxRam(host)
	const freeRam = maxRam - ns.getServerUsedRam(target)

	const numThreadsOnHost = Math.floor(freeRam / hackingRam)

	const happen = hackThreads / numThreadsOnHost

	ns.print("max ram: " + maxRam + " free ram: " + freeRam)
	ns.print("threads on host: " + numThreadsOnHost + " happen: " + happen)

	let sumThreadsDone = 0
	for (let i = 0; i < Math.floor(happen); i++) {
		const hackingTime = ns.getHackTime(target)
		ns.exec("hack.js", host, numThreadsOnHost, target)
		await ns.sleep(hackingTime + safetyMarginMs)
		sumThreadsDone += numThreadsOnHost;
		ns.print("done with " + sumThreadsDone + "/" + hackThreads + " hackings")
	}
	if (sumThreadsDone < hackThreads) {
		ns.print("need to hack " + (hackThreads - sumThreadsDone) + " more time")
		const hackingTime = ns.getHackTime(target)
		ns.exec("hack.js", host, hackThreads - sumThreadsDone, target)
		await ns.sleep(hackingTime + safetyMarginMs)
	}
	ns.print("Done hacking!")

	
}