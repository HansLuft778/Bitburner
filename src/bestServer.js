import { serverScanner, isHackable } from '/src/lib.js'
import { printTable } from '/src/table.js'

/** @param {NS} ns */
export async function main(ns) {
	ns.tail()
	ns.disableLog("ALL")
	getBestServerList(ns, true)
}

/** @param {NS} ns */
export function getBestServerList(ns, shouldPrint) {

	const serverList = serverScanner(ns)

	// relevant stats are max money and hack chance
	// possible formula: (Max Money / weaktime + 3) * (1 / weak time) * (chance to hack)

	let servers = []

	for (let i = 0; i < serverList.length; i++) {
		if (isHackable(ns, serverList[i])) {

			let maxMoney = ns.getServerMaxMoney(serverList[i])
			let hackingChance = ns.hackAnalyzeChance(serverList[i])
			let weakeningTime = ns.getWeakenTime(serverList[i])

			// filter server if it has no money or the hacking level is above 2/3 of the players hacking level
			if (maxMoney < 1 || ns.getServerRequiredHackingLevel(serverList[i]) > ns.getHackingLevel() * (1 / 1)) {
				continue
			}

			// const score = (maxMoney / (weakeningTime + 3)) * hackingChance * (1 / weakeningTime)
			// const score = ns.formatNumber(((maxMoney / (weakeningTime)) * hackingChance) / 1000)
			let score = ns.formatNumber((maxMoney / (ns.getServerMinSecurityLevel(serverList[i]))) / 1000000, 3)

			if (ns.fileExists("formulas.exe", "home")) {
				let so = ns.getServer(serverList[i])
				let po = ns.getPlayer()
				so.hackDifficulty = so.minDifficulty;
				score = ns.formatNumber(((so.moneyMax / ns.formulas.hacking.weakenTime(so, po)) * ns.formulas.hacking.hackChance(so, po) / 1000), 3)
			}

			maxMoney = ns.formatNumber(maxMoney)
			hackingChance = ns.formatNumber(hackingChance)
			weakeningTime = (weakeningTime / 1000).toFixed(2)

			let serverData = [
				serverList[i],
				maxMoney,
				hackingChance,
				weakeningTime,
				score
			];

			servers.push(serverData)
		}

		servers.sort((a, b) => {
			return b[4] - a[4]
		})

	}

	if (shouldPrint)
		printTable(ns, servers)

	return servers
}

/** @param {NS} ns */
export function getBestServer(ns) {
	const servers = getBestServerList(ns, false)
	return servers[0]
}



/** @param {NS} ns */
export function findBestServer(ns) {
	// for now only finds the server with the highest max money, this is not optimal
	const serverList = serverScanner(ns)

	let maxMoney = 0
	let topTen = []
	let stats = []

	serverList.forEach((e) => {
		if (isHackable(ns, e)) {
			const serverMaxMoney = ns.getServerMaxMoney(e)
			const hackTime = ns.getHackTime(e)
			const growTime = ns.getGrowTime(e)
			const weakenTime = ns.getWeakenTime(e)

			if (serverMaxMoney > maxMoney) {
				maxMoney = serverMaxMoney
				topTen.push(e)

				stats.push(serverMaxMoney)
				stats.push(hackTime)
				stats.push(growTime)
				stats.push(weakenTime)

				if (topTen.length > 10) {
					topTen.shift()
					stats.splice(0, 4)
				}
			}
		}
	})
	const statSplit = sliceIntoChunks(stats, 4)

	return [topTen, statSplit]
}

function sliceIntoChunks(arr, chunkSize) {
	const res = [];
	for (let i = 0; i < arr.length; i += chunkSize) {
		const chunk = arr.slice(i, i + chunkSize);
		res.push(chunk);
	}
	return res;
}