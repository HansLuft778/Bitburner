import { NS } from "@ns";

import { Config } from "@/Config/Config";
import { Server, getBestHostByRamOptimized } from "@/bestServer";
import {
    Colors,
    getGrowThreads,
    getGrowThreadsFormulas,
    getHackThreadsFormulas,
    getWeakenThreads,
    getWeakenThreadsAfterGrow,
    getWeakenThreadsAfterHack,
} from "@/lib";
import { ServerManager } from "./ServerManager";
import { PlayerManager } from "./PlayerManager";

export class WGHAlgorithms {
    private static currentGrowThreads = 0;
    private static currentHackThreads = 0;

    /**
     * Weakens a server by executing the weaken.js script with the specified number of threads.
     * The number of threads dispatched depends on the order, batch mode, and available resources.
     *
     * @param ns - The NetScriptJS object.
     * @param target - The name of the target server to weaken.
     * @param order - wether it is weaken I or weaken II.
     * @param batchId - The ID of the parallel batch.
     * @param batchMode - Set to true, of more than one batch should run in parallel mode.
     * @param delay - Time in ms, by how much the weaken script should be delayed to enable precise parallel batch mode timing (default: 0).
     * @returns A number representing the PID of the script that was executed, or 0 if no script was executed.
     * @throws An error if the weaken order is not 1 or 2.
     */
    static weakenServer(
        ns: NS,
        target: string,
        order: number,
        batchMode: boolean,
        delay = 0,
        isLastWeaken = false,
        filterNotAllowedHosts = true,
    ): number {
        ns.print("----------------- WEAKEN " + target + " -----------------");
        let totalWeakenThreadsNeeded = 0;
        // calculate weakening threads based on the order

        if (order == 1 && !batchMode) {
            // first weak has to weaken server to min from unknown sec lvl
            totalWeakenThreadsNeeded = getWeakenThreads(ns, target);
        } else if (order == 2 && !batchMode) {
            // second weak only has to remove the sec increase from the grow before (more ram efficient)
            const growThreads = getGrowThreads(ns, target);

            totalWeakenThreadsNeeded = getWeakenThreadsAfterGrow(ns, growThreads);
        } else if (order == 1 && batchMode) {
            // weak after previous hack
            totalWeakenThreadsNeeded = getWeakenThreadsAfterHack(ns, this.currentHackThreads);
        } else if (order == 2 && batchMode) {
            // weak after previous grow
            totalWeakenThreadsNeeded = getWeakenThreadsAfterGrow(ns, this.currentGrowThreads);
        } else {
            throw new Error("weaken order can only be either 1 or 2!");
        }

        if (totalWeakenThreadsNeeded < 1) {
            ns.print("Weakenthreads are 0, skipping weak " + order);
            return 0;
        }

        // hack xp
        const pm = PlayerManager.getInstance(ns);
        pm.addHackingExp(ns, target, totalWeakenThreadsNeeded);

        // exec weaken.js with num of threads
        let allHosts: Server[] = getBestHostByRamOptimized(ns);
        if (filterNotAllowedHosts) {
            allHosts = allHosts.filter(
                (host) => !host.name.includes(Config.GROW_SERVER_NAME) && !host.name.includes(Config.HACK_SERVER_NAME),
            );
        }
        const weakenScriptRam = Config.WEAKEN_SCRIPT_RAM;
        let pid = 0;

        let threadsDispatched = 0;
        let threadsRemaining = totalWeakenThreadsNeeded;
        for (let i = 0; i < allHosts.length; i++) {
            if (threadsDispatched >= totalWeakenThreadsNeeded) break;
            const host = allHosts[i];

            const freeRam = host.availableRam;
            if (freeRam < weakenScriptRam) continue;
            const threadSpace = Math.floor(freeRam / weakenScriptRam);

            // if threadsRemaining is less than the threadSpace, then we can only dispatch threadsRemaining threads
            const threadsToDispatch = Math.min(threadsRemaining, threadSpace);

            pid = ns.exec("weaken.js", host.name, threadsToDispatch, target, delay, isLastWeaken);
            threadsRemaining -= threadsToDispatch;
            threadsDispatched += threadsToDispatch;
        }

        if (threadsRemaining <= 0) {
            ns.print(
                "Done deploying " +
                    threadsDispatched +
                    "/" +
                    totalWeakenThreadsNeeded +
                    " weaken" +
                    order +
                    " threads!",
            );
            return pid;
        }
        ns.print(
            Colors.YELLOW +
                "There are " +
                threadsRemaining +
                " threads remaining after dispatching all threads, attempting to dispatch remaining threads on purchased server",
        );

        const neededWeakenRam = threadsRemaining * weakenScriptRam;
        const server = ServerManager.buyOrUpgradeServer(ns, neededWeakenRam, Config.WEAK_SERVER_NAME);

        if (server === "") return 0;

        pid = ns.exec("weaken.js", server, threadsRemaining, target, delay, isLastWeaken);
        ns.print("Done deploying " + totalWeakenThreadsNeeded + " weaken" + order + " threads on " + server + "!");
        return pid;
    }

    /**
     * Grows the specified server by executing the "grow.js" script with the specified number of threads.
     * If batchMode is enabled, it calculates the number of threads needed using the getGrowThreadsFormulas function.
     * Otherwise, it uses the getGrowThreads function to determine the number of threads needed.
     * If there are no threads needed, the growth process is skipped.
     * If there is an available host with enough RAM to execute the "grow.js" script, it is executed immediately.
     * Otherwise, it attempts to upgrade or buy a server with enough RAM to execute the script.
     *
     * @param ns - The NetScript instance.
     * @param target - The name of the server to grow.
     * @param batchId - The ID of the batch.
     * @param batchMode - Set to true, of more than one batch should run in parallel mode.
     * @param delay - Time in ms, by how much the grow script should be delayed to enable precise parallel batch mode timing (default: 0).
     * @returns A number representing the PID of the script that was executed, or 0 if no script was executed.
     */
    static growServer(ns: NS, target: string, batchMode: boolean, delay: number, filterNotAllowedHosts = true): number {
        ns.print("----------------- GROWING " + target + " -----------------");
        let totalGrowThreadsNeeded = 0;
        if (!batchMode) {
            totalGrowThreadsNeeded = getGrowThreads(ns, target);
        } else {
            totalGrowThreadsNeeded = getGrowThreadsFormulas(ns, target, this.currentHackThreads);
            this.currentGrowThreads = totalGrowThreadsNeeded;
        }

        if (totalGrowThreadsNeeded < 1) {
            ns.print("No grow threads needed, skipping growth process");
            return 0;
        }

        // hack xp
        const pm = PlayerManager.getInstance(ns);
        pm.addHackingExp(ns, target, totalGrowThreadsNeeded);

        // exec grow.js with num of threads
        let allHosts = getBestHostByRamOptimized(ns);
        if (filterNotAllowedHosts) {
            allHosts = allHosts.filter(
                (host) => !host.name.includes(Config.WEAK_SERVER_NAME) && !host.name.includes(Config.HACK_SERVER_NAME),
            );
        }
        const growingScriptRam = Config.GROW_SCRIPT_RAM;

        for (let i = 0; i < allHosts.length; i++) {
            const host = allHosts[i];

            const maxThreadsOnHost = Math.floor(host.availableRam / growingScriptRam);

            if (maxThreadsOnHost >= totalGrowThreadsNeeded) {
                const pid = ns.exec("grow.js", host.name, totalGrowThreadsNeeded, target, delay);
                ns.print("Done deploying " + totalGrowThreadsNeeded + " grow threads on " + host.name + "!");
                return pid;
            }
        }

        ns.print(Colors.YELLOW + "No available host to grow " + target + ". Attempting to upgrade/buy server...");

        const neededGrowRam = totalGrowThreadsNeeded * growingScriptRam;
        const server = ServerManager.buyOrUpgradeServer(ns, neededGrowRam, Config.GROW_SERVER_NAME);

        if (server === "") return 0;

        const pid = ns.exec("grow.js", server, totalGrowThreadsNeeded, target, delay);
        ns.print("Done deploying " + totalGrowThreadsNeeded + " grow threads on " + server + "!");
        return pid;
    }

    /**
     * Hacks a given server by executing the "hack.js" script with the specified number of threads, on certain hosts.
     *
     * @param ns - The NetScript object.
     * @param target - The name of the server to hack.
     * @param threshold - The hacking threshold for the server.
     * @param batchId - The ID of the current hacking batch.
     * @param batchMode - Set to true, of more than one batch should run in parallel mode.
     * @param delay - Time in ms, by how much the hack script should be delayed to enable precise parallel batch mode timing (default: 0).
     * @returns A number representing the PID of the script that was executed, or 0 if no script was executed.
     */
    static hackServer(ns: NS, target: string, threshold: number, batchMode: boolean, delay: number): number {
        ns.print("----------------- HACKING " + target + " -----------------");
        let totalHackThreadsNeeded = 0;
        if (!batchMode) {
            totalHackThreadsNeeded = Math.ceil(threshold / ns.hackAnalyze(target));
        } else {
            totalHackThreadsNeeded = getHackThreadsFormulas(ns, target, threshold);
            this.currentHackThreads = totalHackThreadsNeeded;
        }

        // hack xp
        const pm = PlayerManager.getInstance(ns);
        pm.addHackingExp(ns, target, totalHackThreadsNeeded);

        const allHosts = getBestHostByRamOptimized(ns).filter(
            (host) => !host.name.includes(Config.WEAK_SERVER_NAME) && !host.name.includes(Config.GROW_SERVER_NAME),
        );
        const hackingScriptRam = Config.HACK_SCRIPT_RAM;

        for (let i = 0; i < allHosts.length; i++) {
            const host = allHosts[i];

            const maxThreadsOnHost = Math.floor(host.availableRam / hackingScriptRam);

            if (maxThreadsOnHost >= totalHackThreadsNeeded) {
                const pid = ns.exec("hack.js", host.name, totalHackThreadsNeeded, target, delay);
                ns.print("Done deploying " + totalHackThreadsNeeded + " hack threads on " + host.name + "!");
                return pid;
            }
        }

        ns.print(Colors.YELLOW + "No available host to hack " + target + ". Buying server...");

        const neededGrowRam = totalHackThreadsNeeded * hackingScriptRam;
        const server = ServerManager.buyOrUpgradeServer(ns, neededGrowRam, Config.HACK_SERVER_NAME);

        if (server === "") return 0;

        const pid = ns.exec("hack.js", server, totalHackThreadsNeeded, target, delay);
        ns.print("Done deploying " + totalHackThreadsNeeded + " hack threads!");
        return pid;
    }
}
