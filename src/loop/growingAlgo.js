/** @param {NS} ns */
export async function main(ns) {
	ns.tail()
	await growServer(ns, "foodnstuff", "hacker")
}

/** @param {NS} ns */
export async function growServer(ns, target, host) {

	ns.print(target + " " + host)

	const safetyMarginMs = 200

	const serverMaxMoney = ns.getServerMaxMoney(target)
	const serverCurrentMoney = ns.getServerMoneyAvailable(target)
	const moneyMult = serverMaxMoney / serverCurrentMoney

	const growThreads = Math.ceil(ns.growthAnalyze(target, moneyMult))

	ns.print("current money: " + serverCurrentMoney + " max: " + serverMaxMoney + " mult: " + moneyMult)
	ns.print("threads: " + growThreads)

	// exec grow.js with num of threads
	const growingScriptRam = 1.75
	const maxRam = ns.getServerMaxRam(host)
	const freeRam = maxRam - ns.getServerUsedRam(target)

	const numThreadsOnHost = Math.floor(freeRam / growingScriptRam)

	const happen = growThreads / numThreadsOnHost

	ns.print("max ram: " + maxRam + " free ram: " + freeRam)
	ns.print("threads on host: " + numThreadsOnHost + " happen: " + happen)

	let sumThreadsDone = 0
	for (let i = 0; i < Math.floor(happen); i++) {
		const growingTime = ns.getGrowTime(target)
		ns.exec("grow.js", host, numThreadsOnHost, target)
		await ns.sleep(growingTime + safetyMarginMs)
		sumThreadsDone += numThreadsOnHost;
		ns.print("done with " + sumThreadsDone + "/" + growThreads + " growings")
	}
	if (sumThreadsDone < growThreads) {
		ns.print("need to grow " + (growThreads - sumThreadsDone) + " more time")
		const growingTime = ns.getGrowTime(target)
		ns.exec("grow.js", host, growThreads - sumThreadsDone, target)
		await ns.sleep(growingTime + safetyMarginMs)
	}
	ns.print("Done growing!")

}