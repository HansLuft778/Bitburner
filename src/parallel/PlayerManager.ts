import { NS, Player } from "@ns";

export class PlayerManager {
    private static instance: PlayerManager;
    private player: Player;

    public static getInstance(ns: NS) {
        if (!PlayerManager.instance) {
            PlayerManager.instance = new PlayerManager(ns);
        }
        return PlayerManager.instance;
    }

    getPlayer() {
        return this.player;
    }

    resetPlayer(ns: NS) {
        const player = ns.getPlayer();
        this.player = player;
    }

    addHackingExp(ns: NS, target: string, threadCount: number) {
        if (!ns.fileExists("Formulas.exe", "home")) {
            return;
        }

        const serverObject = ns.getServer(target);

        const hackXp = ns.formulas.hacking.hackExp(serverObject, this.player);
        const totalExpGain = hackXp * threadCount;

        const lvlAfterHack = ns.formulas.skills.calculateSkill(
            this.player.exp.hacking + totalExpGain,
            this.player.mults.hacking,
        );

        if (lvlAfterHack > this.player.skills.hacking) {
            ns.tprint(
                `Hacking level up! Exp: ${this.player.exp.hacking} -> ${
                    this.player.exp.hacking + totalExpGain
                } | Lvl: ${this.player.skills.hacking} -> ${lvlAfterHack}`,
            );
        }

        // set player exp and lvl after hack for simulated player
        this.player.exp.hacking += totalExpGain;
        this.player.skills.hacking = lvlAfterHack;
    }

    private constructor(ns: NS) {
        this.player = ns.getPlayer();
    }
}
