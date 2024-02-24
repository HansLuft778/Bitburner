import { NS } from "@ns";
import { getBestServerListCheap } from "./bestServer";


export class HostsManager {
    private ns: NS;
    private bestServers: Server[] = [];

    constructor(ns: NS) {
        this.ns = ns;
        const bs = getBestServerListCheap(ns, false);
    }


    sleep(ns: NS) {

    }
}