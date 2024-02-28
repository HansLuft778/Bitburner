/**
 * Represents the settings for the application.
 *
 * To modify the settings, update the class properties accordingly.
 */
export class Config {
    /**
     * The target the daemon should hack, leave empty to hack the best server.
     * Default: ""
     */
    public static readonly TARGET = "";

    /**
     * The maximum amount of money the daemon is allowed to use to buy servers.
     * Default: 0
     */
    public static readonly MAX_MONEY_TO_BUY = 0;

    /**
     * The maximum amount of money the daemon is allowed to hack from a server, leave at 0 to let the daemon decide.
     * Default: 0
     */
    public static readonly HACK_THRESHOLD = 0;

    /**
     * The step value which is used to decrease the hack threshold to calculate the optimal HACK_THRESHOLD.
     * Default: 0.05
     */
    public static readonly THRESHOLD_STEP = 0.05;

    /**
     * The maximum amount of RAM the daemon should leave free on the Home server.
     * Default: 50
     */
    public static readonly HOME_FREE_RAM = 50;

    /**
     * The delay time in milliseconds.
     * This constant represents the amount of time to add as a margin when calling weak/grow/hack in parallel mode.
     * Default: 1000
     */
    public static readonly DELAY_MARGIN_MS = 1000;

    /**
     * The number of batches to use in parallel mode.
     * Default: 2
     */
    public static readonly NUM_BATCHES = 2;

    /**
     * The amount of RAM (in gigabytes) required by the weaken, grow and hack script.
     * Default: 1.75, 1.75, 1.7
     */
    public static readonly WEAKEN_SCRIPT_RAM = 1.75;
    public static readonly GROW_SCRIPT_RAM = 1.75;
    public static readonly HACK_SCRIPT_RAM = 1.7;
}
