import { NS } from "@ns";
import { getBestServerListCheap } from "./bestServer";
import { Server } from "./bestServer";

export class HostsManager {
    private ns: NS;
    private allHosts: Server[] = [];
    private serverWeakenThreads: number;
    private target: string;

    constructor(ns: NS, serverWeakenThreads: number, target: string) {
        this.ns = ns;

        this.allHosts = getBestServerListCheap(ns, false);
        this.allHosts
            .filter((server) => {
                return server.maxRam > 2;
            })
            .sort((a, b) => {
                // TODO: sort might be unnecessary
                return b.maxRam - a.maxRam;
            });

        this.serverWeakenThreads = serverWeakenThreads;
        this.target = target;
    }

    /*
    Idea: There are x many servers that can help weaken the target server.
    We know how many threads we need in total and how many threads we can run on each server.

    Then we blindly run the weaken script on each server, until the total threads are reached.


    for each server:
        check if more weaken threads are needed
        calculate how many threads can be run on this server
        run weaken script on this server
    */
    async weaken(ns: NS) {
        const weakenTime = ns.getWeakenTime(this.target);
        const weakenRam = 1.75;
        const totalMaxRam = this.allHosts.reduce((acc, server) => {
            return acc + server.maxRam;
        }, 0);
        ns.print("maxRam: " + totalMaxRam);

        const numRuns = Math.ceil(this.serverWeakenThreads / totalMaxRam);

        let sumThreadsDone = 0;
        while (sumThreadsDone < this.serverWeakenThreads) {
            for (let i = 0; i < this.allHosts.length; i++) {
                if (sumThreadsDone >= this.serverWeakenThreads) break;

                const host = this.allHosts[i];
                const freeRam = host.maxRam - ns.getServerUsedRam(host.name);

                const numThreadsOnHost = Math.floor(freeRam / weakenRam);

                ns.exec("weaken.js", host.name, numThreadsOnHost, this.target);
                sumThreadsDone += numThreadsOnHost;
            }
            await ns.sleep(weakenTime + 200);
            ns.print("done with " + sumThreadsDone + "/" + this.serverWeakenThreads + " weakens");
        }
    }

    grow(ns: NS) {}
}
