export interface Stock {
    symbol: string;
    organization: string;
    observedMinPrice: number;
    observedMaxPrice: number;
    longShares: number;
    longPrice: number;
    shortShares: number;
    shortPrice: number;

    /**
     * maximum price at which someone will buy a stock
     */
    bidPrice: number;
    /**
     * minimum price at which someone will sell a stock
     */
    askPrice: number;
    /**
     * Average price of bid and ask price
     */
    price: number;
    previousPrice: number;
    forecast: number;
    volatility: number;
    maxShares: number;
    profit: number;
    cost: number;
    profitPotential: number;
}
