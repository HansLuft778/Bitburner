import Chart from "chart.js/auto";
import { createWindowApp, NetscriptContext } from "../WindowApp.js";

import React, { useContext, useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";
import { StockmarketHistory } from "./StockmarketHistory.js";

interface ChartData {
    labels: string[];
    datasets: {
        label: string;
        data: number[];
        backgroundColor: string;
        borderColor: string;
        borderWidth: number;
        hidden: boolean;
    }[];
}
export function LinePlot() {
    const chartRef = useRef<HTMLCanvasElement>(null);
    const chartInstance = useRef<Chart | null>(null);
    const ns: NS = useContext(NetscriptContext);

    const [stockHistory, setStockHistory] = useState<number[][]>([]);

    // Create chart once on mount
    useEffect(() => {
        if (!chartRef.current) return;

        chartInstance.current = new Chart(chartRef.current, {
            type: "line",
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                animation: {
                    duration: 0.1
                }
            }
        });

        // Cleanup on unmount
        return () => {
            if (chartInstance.current) {
                chartInstance.current.destroy();
            }
        };
    }, []);

    // Fetch data periodically
    useEffect(() => {
        // Initial fetch
        const history = new StockmarketHistory(ns).getStockHistory();
        console.log("Initial history:", history);
        setStockHistory(history);

        // Set up interval for periodic updates
        const intervalId = setInterval(() => {
            const updatedHistory = new StockmarketHistory(ns).getStockHistory();
            setStockHistory(updatedHistory);
        }, 1000);

        return () => clearInterval(intervalId);
    }, [ns]);

    // Update chart data
    useEffect(() => {
        if (!chartInstance.current || stockHistory.length === 0) return;

        const symbols = ns.stock.getSymbols().sort();

        const labels = Array.from(Array(stockHistory[0].length).keys());
        let new_datasets = [];
        for (let i = 0; i < stockHistory.length; i++) {
            const stock = stockHistory[i];
            let obj = {
                label: symbols[i],
                data: stock,
                backgroundColor: "rgba(255, 99, 132, 0.2)",
                borderColor: "rgba(255, 99, 132, 1)",
                hidden: !chartInstance.current.isDatasetVisible(i)
            };
            new_datasets.push(obj);
        }

        chartInstance.current.data.labels = labels;
        chartInstance.current.data.datasets = new_datasets;
        chartInstance.current.update(); // Update without animation
    }, [stockHistory]);

    return (
        <div style={{ position: "absolute", width: "100%", height: "100%" }}>
            <canvas style={{ width: "100%", height: "100%" }} ref={chartRef} />
        </div>
    );
}

function MyApp() {
    return (
        <div>
            <LinePlot />
        </div>
    );
}

/** @param {NS} ns */
export async function main(ns: NS) {
    const WindowApp = createWindowApp(ns);

    ns.atExit(() => {
        WindowApp.cleanup();
    });

    ns.print("Hello from MyApp!");

    return WindowApp.mount(<MyApp></MyApp>);
}
