import Chart from "chart.js/auto";
import { createWindowApp } from "../WindowApp.js";

import React, { useContext, useEffect, useRef } from "react";
import ReactDOM from "react-dom";

function MyButton() {
    return <button>MyButton</button>;
}

export function LinePlot() {
    const chartRef = useRef<HTMLCanvasElement>(null);
    const chartInstance = useRef<Chart | null>(null);

    useEffect(() => {
        if (!chartRef.current) return;

        // Destroy existing chart
        if (chartInstance.current) {
            chartInstance.current.destroy();
        }

        // Create new chart
        chartInstance.current = new Chart(chartRef.current, {
            type: "line",
            data: {
                labels: ["Red", "Blue", "Yellow", "Green", "Purple", "Orange"],
                datasets: [
                    {
                        label: "# of Votes",
                        data: [12, 19, 3, 5, 2, 3],
                        backgroundColor: "rgba(255, 99, 132, 0.2)",
                        borderColor: "rgba(255, 99, 132, 1)",
                        borderWidth: 1
                    },
                    {
                        label: "gorg",
                        data: [1, 2, 3, 4, 5, 6, 7, 8],
                        backgroundColor: "rgba(14, 160, 26, 0.2)",
                        borderColor: "rgb(19, 116, 196)",
                        borderWidth: 1
                    }
                ]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        // Cleanup on unmount
        return () => {
            if (chartInstance.current) {
                chartInstance.current.destroy();
            }
        };
    }, []); // Empty dependency array means this effect runs once on mount

    return (
        <div style={{ width: "100%", height: "400px" }}>
            <canvas style={{ width: "100%", height: "100%" }} ref={chartRef} />
        </div>
    );
}

function MyApp() {
    return (
        <div>
            {/* <MyButton /> */}
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
