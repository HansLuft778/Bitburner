export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();
    ns.disableLog("ALL");
    ns.print("\n");

    const stock = ns.stock;

    stock
        .getSymbols()
        .sort()
        .forEach((symbol) => {
            ns.print(symbol);
        });
}
