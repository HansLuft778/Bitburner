import { NS } from "@ns";
import { algorithmicStockTraderIII } from "./manager";

function sumCombinations(target: number, current: number[] = [], start = 1, result: number[][] = []) {
    if (target === 0) {
        result.push(current.slice()); // Push a copy of the current combination to the result
        return;
    }

    for (let i = start; i <= target; i++) {
        current.push(i);
        sumCombinations(target - i, current, i, result);
        current.pop();
    }

    return result;
}

export function generateSwitchCase(ns: NS) {
    let switchCase = "";
    switchCase += "switch (contractType) {\n";
    for (const contractType of ns.codingcontract.getContractTypes()) {
        switchCase += `case "${contractType}":\n`;
        switchCase += `break;\n`;
    }
    switchCase += "default:\n";
    switchCase += "break;\n";
    switchCase += "}\n";
    return switchCase;
}

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");

    algorithmicStockTraderIII(ns, "", "");
}
