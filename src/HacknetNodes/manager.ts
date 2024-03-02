import { NS } from "@ns";

const EXPECTED_TURNOVER_TIME = 1000 * 60 * 60 * 2; // 2 hours

export async function main(ns: NS) {
    const purchaseCost = ns.hacknet.getPurchaseNodeCost();
    const stats = ns.hacknet.getNodeStats(0);
}
