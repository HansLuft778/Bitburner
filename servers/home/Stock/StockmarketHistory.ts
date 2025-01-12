import { StockConfig } from "./Config.js";
import { Stock } from "./model.js";

export class StockmarketHistory {
    /**
     * Stock market history is array where each entry is a stock. Each stock is an array of timesteps
     */
    private history: number[][] = [];

    constructor(ns: NS) {
        const hostoryCSV = ns.read(StockConfig.STOCK_HISTORY_NAME);
        if (hostoryCSV === "") this.history = [];
        else this.history = this.csvToArray(hostoryCSV);
    }

    saveHistory(ns: NS) {
        if (this.history.length > 0 && this.history[0].length > 0) {
            const csv = this.arrayToCSV(this.history);
            ns.write(StockConfig.STOCK_HISTORY_NAME, csv, "w");
        }
    }

    updateHistory(ns: NS, stocks: Stock[]) {
        const sortedStocks = stocks.sort((a, b) =>
            a.symbol.localeCompare(b.symbol)
        );

        for (let i = 0; i < sortedStocks.length; i++) {
            const stock = sortedStocks[i];
            if (this.history[i] == null) this.history[i] = [stock.price];
            else this.history[i].push(stock.price);

            if (this.history[i].length > StockConfig.STOCK_HISTORY_LENGTH)
                this.history[i].shift();
        }

        this.saveHistory(ns);
    }

    getStockHistory() {
        return this.history;
    }

    private arrayToCSV(data: number[][]): string {
        let csv = "";
        for (let i = 0; i < data.length; i++) {
            csv += data[i].join(",");
            csv += "\n";
        }
        return csv;
    }

    private csvToArray(csvData: string): number[][] {
        const lines = csvData.split("\n");
        const result = [];
        for (let i = 0; i < lines.length; i++) {
            // Skip empty lines
            if (lines[i].trim() === "") continue;

            const values = lines[i]
                .split(",")
                .map((x) => x.trim())
                .filter((x) => x !== "") // Remove empty values
                .map((x) => {
                    const num = parseFloat(x);
                    return isNaN(num) ? 0 : num; // Convert NaN to 0
                });

            if (values.length > 0) {
                result.push(values);
            }
        }
        return result;
    }
}

export async function main(ns: NS) {
    const smh = new StockmarketHistory(ns);
}
