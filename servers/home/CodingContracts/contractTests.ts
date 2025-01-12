import { algorithmicStockTraderIII } from "./manager.js";

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
