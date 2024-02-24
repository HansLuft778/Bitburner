import { serverScanner, getNumHacks } from '/src/lib.js'
import { findBestServer } from '/src/bestServer.js'
import { weakenServer } from '/src/loop/weakenAlgo.js'
import { growServer } from '/src/loop/growingAlgo.js'
import { hackServer } from '/src/loop/hackingAlgo.js'

// Text color
const reset = "\x1b[0m";
const black = "\x1b[30m";
const red = "\x1b[31m";
const green = "\x1b[32m";
const yellow = "\x1b[33m";
const blue = "\x1b[34m";
const magenta = "\x1b[35m";
const cyan = "\x1b[36m";
const white = "\x1b[37m";

/** @param {NS} ns */
export async function main(ns) {
	ns.tail()
	ns.disableLog("ALL");

	const hosts = serverScanner(ns)
	nukeAll(ns, hosts)

	let lastTarget = ""
	// steps: WGWH-WGWH-..
	while (true) {

		// find the server with the most available money
		const bestServers = findBestServer(ns)
		// const target = bestServers[0][bestServers[0].length - 1]
		const target = "phantasy"

		if (lastTarget != target) {
			nukeAll(ns, serverScanner(ns))
		}
		lastTarget = target

		ns.print(cyan + "------------ WEAKENING ------------" + reset)
		// await weakenServer(ns, target, "aws-0")

		ns.print(cyan + "------------- GROWING -------------" + reset)
		await growServer(ns, target, "aws-1")

		ns.print(cyan + "------------ WEAKENING ------------" + reset)
		// await weakenServer(ns, target, "aws-2")

		ns.print(cyan + "------------- HACKING -------------" + reset)
		// await hackServer(ns, target, "aws-3", 0.5)
	}
}

/** @param {NS} ns */
function openPorts(ns, target) {
	if (ns.fileExists("BruteSSH.exe")) ns.brutessh(target)
	if (ns.fileExists("FTPCrack.exe")) ns.ftpcrack(target)
	if (ns.fileExists("HTTPWorm.exe")) ns.httpworm(target)
	if (ns.fileExists("SQLInject.exe")) ns.sqlinject(target)
}

/** @param {NS} ns */
function nukeAll(ns, hosts) {
	for (let i = 0; i < hosts.length; i++) {
		// check if the host is hackable
		if (ns.getServerNumPortsRequired(hosts[i]) <= getNumHacks(ns) && ns.getServerRequiredHackingLevel(hosts[i]) <= ns.getHackingLevel()) {// !ns.hasRootAccess(hosts[i]) && 
			ns.print(cyan + "------------" + hosts[i] + "------------" + reset)

			openPorts(ns, hosts[i])
			ns.nuke(hosts[i])

			// copy all scripts to the server
			ns.scp("/src/loop/hack.js", hosts[i])
			ns.scp("/src/loop/grow.js", hosts[i])
			ns.scp("/src/loop/weaken.js", hosts[i])

			const serverGrowTime = ns.getGrowTime(hosts[i])
			const serverHackTime = ns.getHackTime(hosts[i])
			const serverWeakTime = ns.getWeakenTime(hosts[i])

			ns.print("hack-time: " + serverHackTime + " grow-time: " + serverGrowTime + " weaken-time: " + serverWeakTime)

			// grow
			const serverMaxMoney = ns.getServerMaxMoney(hosts[i])
			const serverCurMoney = ns.getServerMoneyAvailable(hosts[i])
			let serverMoneyMultiplier = serverMaxMoney / serverCurMoney
			if (isNaN(serverMoneyMultiplier))
				serverMoneyMultiplier = 1;

			const serverGrowThreads = ns.growthAnalyze(hosts[i], serverMoneyMultiplier)

			ns.print("max Money: " + serverMaxMoney + " current Money: " + serverCurMoney)
			ns.print("mult: " + serverMoneyMultiplier + " grow threads: " + serverGrowThreads)

			// weaken
			const serverSecLvl = ns.getServerSecurityLevel(hosts[i])
			const serverWeakenThreads = (serverSecLvl - ns.getServerMinSecurityLevel(hosts[i])) / 0.05
			const serverWeakenEffect = ns.weakenAnalyze(serverWeakenThreads)

			ns.print("min sec: " + ns.getServerMinSecurityLevel(hosts[i]) + " cur sec lvl: " + serverSecLvl)
			ns.print("weaken threads: " + serverWeakenThreads + " weaken effect: " + serverWeakenEffect)

		} else {
			continue
		}
	}
}


/**
 notes:
 weaken removes 0.05 sec lvl
 grow adds 0.004 sec lvl

 grow adds money:

 */