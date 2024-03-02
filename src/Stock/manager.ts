import { NS } from "@ns";
import { Colors } from "@/lib";

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
    forecast: number;
    volatility: number;
    maxShares: number;
    profit: number;
    cost: number;
    profitPotential: number;
}

const BUY_VALUE = 1_000_000;
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

export async function main(ns: NS) {
    ns.print("asd");
    ns.tail();
    ns.disableLog("ALL");
    const s = ns.stock;

    const stocks: Stock[] = loadMarket(ns);
    if (stocks.length === 0) {
        ns.print(Colors.YELLOW + "Unable to load market, fetching new data");
        const allStocks = getAllStocks(ns);
        stocks.push(...allStocks);
    }

    ns.print(stocks.length);

    let totalProfit = 0;
    while (true) {
        ns.clearLog();

        const stocks: Stock[] = getAllStocks(ns);

        for (const stock of stocks) {
            if (stock.observedMinPrice === -1 || stock.price < stock.observedMinPrice) {
                stock.observedMinPrice = stock.price;
            }
            if (stock.observedMaxPrice === -1 || stock.price > stock.observedMaxPrice) {
                stock.observedMaxPrice = stock.price;
            }

            // buy long stock
            if (stock.forecast >= 0.6 && stock.longShares === 0) {
                const numShares = Math.round(BUY_VALUE / stock.bidPrice);
                const buyPrice = s.buyStock(stock.symbol, numShares);
                stock.longPrice = buyPrice;
                stock.longShares = numShares;
                ns.print(Colors.E_ORANGE + "buying " + stock.symbol + " for " + buyPrice);
            }
            // sell long stock
            if (stock.longShares > 0 && stock.forecast <= 0.5) {
                const sellPrice = s.sellStock(stock.symbol, stock.longShares);
                totalProfit += sellPrice * stock.longShares - stock.longPrice * stock.longShares - 2 * COMMISSION_FEE;
                stock.longShares = 0;
                stock.longPrice = 0;
                ns.print(Colors.E_ORANGE + "selling " + stock.symbol + " for " + totalProfit);
            }

            const color = stock.longShares > 0 ? Colors.E_ORANGE : "";

            ns.print(
                color +
                    `${stock.symbol}:\t${ns.formatNumber(stock.price)} (min: ${ns.formatNumber(
                        stock.observedMinPrice,
                    )}, max: ${ns.formatNumber(stock.observedMaxPrice)})\tforecast: ${stock.forecast}`,
            );
        }
        const color = totalProfit < 0 ? Colors.RED : Colors.GREEN;
        ns.print(color + "Total profit: " + ns.formatNumber(totalProfit));

        saveMarket(ns, stocks);
        await s.nextUpdate();
        ns.print("\n");
    }
}
