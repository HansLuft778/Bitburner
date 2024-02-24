/** @param {NS} ns */
export async function main(ns) {
	createTable(ns, test)
}

// with of the cell content
const colLen = [19, 10, 7, 10, 8]

/** @param {NS} ns */
export function printTable(ns, array) {
	ns.print("╔════════════════════╦═══════════╦════════╦═══════════╦═════════╗")
	ns.print("║       server       ║   Max $   ║ chance ║ Weak time ║  score  ║")
	ns.print("╠════════════════════╬═══════════╬════════╬═══════════╬═════════╣")
	// polluting table with data
	for (let i = 0; i < array.length; i++) {
		ns.print("║ " + array[i][0] + space(array[i][0].length, 0) + "║ " + array[i][1] + space(array[i][1].length, 1) + 
				 "║ " + array[i][2] + space(array[i][2].length, 2) + "║ " + array[i][3] + space(array[i][3].length, 3) + "║ " + array[i][4] + space(array[i][4].length, 4) + "║")
	}

	ns.print("╚════════════════════╩═══════════╩════════╩═══════════╩═════════╝")
}

function space(len, colIndex) {
	let real = colLen[colIndex] - len
	let str = ""
	for (let i = 0; i < real; i++) {
		str += " ";
	}
	return str
}