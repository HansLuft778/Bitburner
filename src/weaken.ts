import { NS } from "@ns";
export async function main(ns: NS) {
    if (typeof ns.args[0] !== "string") throw new TypeError("[WEAKEN] First argument must be a string");
    if (typeof ns.args[1] !== "number") throw new TypeError("[WEAKEN] Second argument must be a number");

    await ns.weaken(ns.args[0], { additionalMsec: ns.args[1] });

    const port = ns.getPortHandle(2);
    const data = port.peek();
    // ns.tprint(data + " | " + ns.pid);

    if (ns.args[2] === true || data === ns.pid) {
        port.clear();
        ns.atExit(() => {
            port.write(ns.pid);
        });
    }
}
