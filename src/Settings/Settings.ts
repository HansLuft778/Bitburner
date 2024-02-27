/**
 * Represents the settings for the application.
 *
 * To modify the settings, update the class properties accordingly.
 */
export class Settings {
    /**
     * The target the daemon should hack, leave empty to hack the best server.
     * Default: ""
     */
    public static target = "";

    /**
     * The maximum amount of money the daemon is allowed to use to buy servers.
     * Default: 0
     */
    public static maxMoneyToBuy = 0;

    /**
     * The maximum amount of money the daemon is allowed to hack from a server, leave at 0 to let the daemon decide.
     * Default: 0
     */
    public static hackThreshold = 0;

    /**
     * The maximum amount of RAM the daemon should leave free on the Home server.
     * Default: 50
     */
    public static homeFreeRam = 50;

    /**
     * The delay time in milliseconds.
     * This constant represents the amount of time to add as a margin when calling weak/grow/hack in parallel mode.
     * Default: 1000
     */
    public static DELAY_MARGIN_MS = 1000;
}
