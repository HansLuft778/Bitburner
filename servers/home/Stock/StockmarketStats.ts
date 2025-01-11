export interface Stats {
    totalProfit: number;
}

export class StockMarketStats {
    private stats: Stats;

    constructor(ns: NS) {
        this.loadStats(ns);
    }

    public loadStats(ns: NS): Stats {
        const data = ns.read("Stock/stats.txt");
        if (data === "") {
            this.stats = { totalProfit: 0 };
            ns.write("Stock/stats.txt", JSON.stringify(this.stats), "w");
            return this.stats;
        }
        this.stats = JSON.parse(data);
        return this.stats;
    }

    public getStats(): Stats {
        return this.stats;
    }

    public saveStats(ns: NS): void {
        ns.write("Stock/stats.txt", JSON.stringify(this.stats), "w");
    }

    public resetStats(ns: NS): void {
        this.stats = { totalProfit: 0 };
        this.saveStats(ns);
    }

    public addProfit(ns: NS, profit: number): void {
        this.stats.totalProfit += profit;
        this.saveStats(ns);
    }

    public profit(): number {
        return this.stats.totalProfit;
    }
}
