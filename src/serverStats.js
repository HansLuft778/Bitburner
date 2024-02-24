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

let maxMoney = 0;
let curMoney = 0;
let hackingChance = 0;
let minSec = 0;
let curSec = 0;
let maxRam = 0;
let useRam = 0;
let freeRam = 0;
let moneyMult = 0;
let growingThreads = 0;
let serverWeakenThreadsCur = 0;
let lowerMoneyBound = 0;
let hackAmount = 0;
let hackThreads = 0;
let headerString = "";
let footerString = "";
let hackingPercent = 0



/** @param {NS} ns */
export async function main(ns) {
	ns.clearLog()
	ns.tail()
	ns.disableLog("ALL")
	await printServerStatsLive(ns, ns.args[0], 0.9)
}

/** @param {NS} ns */
export function printServerStats(ns, server, hackThreshold) {

	setStats(ns, server, hackThreshold)

	ns.print(cyan + headerString + reset)

	ns.print("Money:")
	ns.print("\tMax Money: " + ns.formatNumber(maxMoney) + " | Current Money: " + ns.formatNumber(curMoney))
	ns.print("\tHack Chance: " + ns.formatNumber(hackingChance))

	ns.print("Security:")
	ns.print("\tMin Seclvl: " + minSec + " | Current Seclvl: " + ns.formatNumber(curSec))

	ns.print("Ram:")
	ns.print("\tServer Max Ram: " + maxRam)
	ns.print("\tUsed Ram: " + useRam + " | free Ram: " + freeRam)

	ns.print("Threads:")
	ns.print("\tGrow Threads: " + growingThreads)
	ns.print("\tWeaken Threads " + serverWeakenThreadsCur)

	ns.print("\tHack Threads: " + ns.formatNumber(hackThreads) + " | Hack percent: " + ns.formatNumber(hackingPercent, 5))

	ns.print(cyan + footerString + reset)
}

/** @param {NS} ns */
export function printServerStatsConsole(ns, server) {
	// todo
}

/** @param {NS} ns */
async function printServerStatsLive(ns, server, hackThreshold) {
	while (true) {
		setStats(ns, server, hackThreshold)


		ns.print(cyan + headerString + reset)

		ns.print("Money:")
		ns.print("\tMax Money: " + ns.formatNumber(maxMoney) + " | Current Money: " + ns.formatNumber(curMoney))
		ns.print("\tHack Chance: " + ns.formatNumber(hackingChance))

		ns.print("Security:")
		ns.print("\tMin Seclvl: " + minSec + " | Current Seclvl: " + ns.formatNumber(curSec))

		ns.print("Ram:")
		ns.print("\tServer Max Ram: " + maxRam)
		ns.print("\tUsed Ram: " + useRam + " | free Ram: " + freeRam)

		ns.print("Threads:")
		ns.print("\tGrow Threads: " + growingThreads)
		ns.print("\tWeaken Threads " + serverWeakenThreadsCur)

		ns.print("\tHack Threads: " + ns.formatNumber(hackThreads) + " | Hack percent: " + ns.formatNumber(hackingPercent, 5))

		ns.print(cyan + footerString + reset)
		await ns.sleep(200)
		ns.clearLog()
	}
}

/** @param {NS} ns */
function setStats(ns, server, hackThreshold) {
	// money
	maxMoney = ns.getServerMaxMoney(server)
	curMoney = ns.getServerMoneyAvailable(server)
	hackingChance = ns.hackAnalyzeChance(server)
	// sec lvl
	minSec = ns.getServerMinSecurityLevel(server)
	curSec = ns.getServerSecurityLevel(server)
	// ram
	maxRam = ns.getServerMaxRam(server)
	useRam = ns.getServerUsedRam(server)
	freeRam = maxRam - useRam
	// threads

	moneyMult = maxMoney / curMoney
	if (isNaN(moneyMult) || moneyMult == Infinity)
		moneyMult = 1

	growingThreads = Math.ceil(ns.growthAnalyze(server, moneyMult))

	serverWeakenThreadsCur = Math.ceil((curSec - ns.getServerMinSecurityLevel(server)) / 0.05)

	lowerMoneyBound = maxMoney * hackThreshold
	hackAmount = maxMoney - lowerMoneyBound
	hackingPercent = ns.hackAnalyze(server)
	hackThreads = Math.ceil(hackThreshold / hackingPercent)
	if (isNaN(hackThreads) || hackThreads == Infinity)
		hackThreads = 0
	if (isNaN(hackingPercent) || hackingPercent == Infinity)
		hackingPercent = 0

	headerString = "----------------- stats for " + server + " -----------------"
	footerString = buildFooterString(headerString.length)
}

function buildFooterString(len) {
	let footerStr = ""
	for (let i = 0; i < len; i++) {
		footerStr += "-";
	}
	return footerStr
}