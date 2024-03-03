import { Colors } from "@/lib";
import { NS } from "@ns";

interface Stock {
    symbol: string;
    organization: string;
    observedMinPrice: number;
    observedMaxPrice: number;
    longShares: number;
    longPrice: number;
    shortShares: number;
    shortPrice: number;
    bidPrice: number;
    askPrice: number;
    price: number;
    previousPrice: number;
    forecast: number;
    volatility: number;
    maxShares: number;
    profit: number;
    cost: number;
    profitPotential: number;
}

const COMMISSION_FEE = 100_000;

function getAllStocks(ns: NS): Stock[] {
    const stocks: Stock[] = [];
    const symbols = ns.stock.getSymbols();

    for (const symbol of symbols) {
        const position = ns.stock.getPosition(symbol);
        const stock: Stock = {
            symbol: symbol,
            organization: ns.stock.getOrganization(symbol),
            observedMinPrice: -1,
            observedMaxPrice: -1,
            longShares: position[0],
            longPrice: position[1],
            shortShares: position[2],
            shortPrice: position[3],
            bidPrice: ns.stock.getBidPrice(symbol),
            askPrice: ns.stock.getAskPrice(symbol),
            price: ns.stock.getPrice(symbol),
            previousPrice: -1,
            forecast: ns.stock.getForecast(symbol),
            volatility: ns.stock.getVolatility(symbol),
            maxShares: ns.stock.getMaxShares(symbol),
            profit: -1,
            cost: -1,
            profitPotential: -1,
        };

        stock.profit = stock.longShares * (stock.bidPrice - stock.longPrice) - 2 * COMMISSION_FEE;
        stock.cost = stock.longShares * stock.longPrice + stock.shortShares * stock.shortPrice;
        stock.profitPotential = 2 * Math.abs(stock.forecast - 0.5) * stock.volatility;

        stocks.push(stock);
    }

    return stocks;
}

function updateStocks(ns: NS, stocks: Stock[]) {
    for (const stock of stocks) {
        const position = ns.stock.getPosition(stock.symbol);
        stock.longShares = position[0];
        stock.longPrice = position[1];
        stock.shortShares = position[2];
        stock.shortPrice = position[3];
        stock.bidPrice = ns.stock.getBidPrice(stock.symbol);
        stock.askPrice = ns.stock.getAskPrice(stock.symbol);
        stock.price = ns.stock.getPrice(stock.symbol);
        stock.forecast = ns.stock.getForecast(stock.symbol);
        stock.volatility = ns.stock.getVolatility(stock.symbol);
        stock.maxShares = ns.stock.getMaxShares(stock.symbol);
    }

    return stocks;
}

function saveMarket(ns: NS, stocks: Stock[]) {
    const str = JSON.stringify(stocks);
    ns.write("Stock/market.txt", str, "w");
}

function loadMarket(ns: NS): Stock[] {
    const str = ns.read("Stock/market.txt");
    try {
        const data = JSON.parse(str);
        return data;
    } catch (error) {
        return [];
    }
}

function getPercentage(money: number) {
    return 1 / (1 + Math.exp(0.0000000005 * money)) + 0.1;
}

function logPositions(ns: NS, stock: Stock, isBuy: boolean, profit = 0) {
    if (isBuy)
        ns.write(
            "Stock/log.txt",
            "buying " +
                ns.formatNumber(stock.longShares) +
                " shares of " +
                stock.symbol +
                " for " +
                ns.formatNumber(stock.longPrice * stock.longShares) +
                "\n",
            "a",
        );
    else
        ns.write(
            "Stock/log.txt",
            "selling " +
                ns.formatNumber(stock.longShares) +
                " shares of " +
                stock.symbol +
                " for " +
                ns.formatNumber(profit) +
                "\n",
            "a",
        );
}

export async function main(ns: NS) {
    ns.print("asd");
    ns.tail();
    ns.disableLog("ALL");
    const s = ns.stock;

    let initialStocks: Stock[] = getAllStocks(ns);

    ns.print(initialStocks.length);

    let totalProfit = 0;
    while (true) {
        ns.clearLog();

        const stocks: Stock[] = updateStocks(ns, initialStocks);

        for (const stock of stocks) {
            if (stock.observedMinPrice === -1 || stock.price < stock.observedMinPrice) {
                stock.observedMinPrice = stock.price;
            }
            if (stock.observedMaxPrice === -1 || stock.price > stock.observedMaxPrice) {
                stock.observedMaxPrice = stock.price;
            }
            if (stock.previousPrice === -1) {
                stock.previousPrice = stock.price;
            }

            // buy long stock
            if (stock.forecast >= 0.6 && stock.longShares === 0) {
                const money = getPercentage(ns.getServerMoneyAvailable("home")) * ns.getServerMoneyAvailable("home");
                const numShares = Math.min(Math.round(money / stock.bidPrice), stock.maxShares);

                const buyPrice = s.buyStock(stock.symbol, numShares);
                stock.longPrice = buyPrice;
                stock.longShares = numShares;
                ns.print(Colors.E_ORANGE + "buying " + numShares + " shares of " + stock.symbol + " for " + buyPrice);
                logPositions(ns, stock, true);
            }
            // sell long stock
            if (stock.longShares > 0 && stock.forecast <= 0.5) {
                const sellPrice = s.sellStock(stock.symbol, stock.longShares);
                const profit = sellPrice * stock.longShares - stock.longPrice * stock.longShares - 2 * COMMISSION_FEE;
                stock.longShares = 0;
                stock.longPrice = 0;
                totalProfit += profit;
                ns.print(Colors.E_ORANGE + "selling " + stock.symbol + " for " + profit);
                logPositions(ns, stock, false, profit);
            }

            const color = stock.longShares > 0 ? Colors.E_ORANGE : "";
            const arrow = stock.price > stock.previousPrice ? "↗" : stock.price < stock.previousPrice ? "↘" : "→";

            ns.print(
                color +
                    `${stock.symbol}:\t${ns.formatNumber(stock.price)} (min: ${ns.formatNumber(
                        stock.observedMinPrice,
                    )}, max: ${ns.formatNumber(stock.observedMaxPrice)})\tforecast: ${ns.formatNumber(
                        stock.forecast,
                    )} ${arrow}`,
            );

            stock.previousPrice = stock.price;
            initialStocks = stocks;
        }
        const color = totalProfit < 0 ? Colors.RED : Colors.GREEN;
        ns.print(color + "Total profit: " + ns.formatNumber(totalProfit));

        saveMarket(ns, stocks);
        await s.nextUpdate();
        ns.print("\n");
    }
}
