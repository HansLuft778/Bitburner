import { NS } from "@ns";
import { PlayerManager } from "./parallel/PlayerManager";

export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    ns.print("\n");

    const target = "phantasy";

    const hackThreads = 10000; //getGrowThreads(ns, target);

    const pm = PlayerManager.getInstance(ns);
    pm.resetPlayer(ns);
    const playerObject = pm.getPlayer();

    const serverObject = ns.getServer(target);

    // how much hack xp will be gained from one HGW-Thread
    const hackXp = ns.formulas.hacking.hackExp(serverObject, playerObject);
    const totalExpGain = hackXp * hackThreads;
    ns.print("totalHackXp gained: " + totalExpGain);

    const currentHackExp = playerObject.exp.hacking;
    const currentHackLvl = playerObject.skills.hacking;

    const lvlAfterHack = ns.formulas.skills.calculateSkill(currentHackExp + totalExpGain, playerObject.mults.hacking);
    ns.print("currentHackLvl: " + currentHackLvl + " -> " + lvlAfterHack);

    const nextLevelExp = ns.formulas.skills.calculateExp(currentHackLvl + 1, playerObject.mults.hacking);
    const nextLevelExpAfter = ns.formulas.skills.calculateExp(lvlAfterHack + 1, playerObject.mults.hacking);

    ns.print(
        "hack xp remaining now: " +
            (nextLevelExp - currentHackExp) +
            " -> " +
            (nextLevelExpAfter - (currentHackExp + totalExpGain)),
    );

    // const growTime = ns.getGrowTime(target);
    // WGHAlgorithms.growServer(ns, target, -1, false, 0);
    // await ns.sleep(growTime + Config.DELAY_MARGIN_MS);

    // ns.print("-------------------");
    // const playerAfter = ns.getPlayer();
    // const lvlAfterGrow = playerAfter.skills.hacking;
    // ns.print("lvlAfterGrow: " + lvlAfterGrow);

    // const expAfterGrow = playerAfter.exp.hacking;
    // ns.print("expAfterGrow: " + expAfterGrow);

    // const afterNextLevelExp = ns.formulas.skills.calculateExp(lvlAfterGrow + 1, playerAfter.mults.hacking);
    // ns.print("exp remaining after grow: " + (afterNextLevelExp - expAfterGrow));
}
