declare global {
    var webpackChunkbitburner: any[];
}

/** @param {NS} ns */
export async function main(ns: NS) {
    globalThis.webpack_require ??
        webpackChunkbitburner.push([
            [-1],
            {},
            (w) => (globalThis.webpack_require = w)
        ]);
    Object.keys(globalThis.webpack_require.m).forEach((k) =>
        Object.values(globalThis.webpack_require(k)).forEach((p: any) =>
            p?.toPage?.("Dev")
        )
    );
    // globalThis.webpack_require ??
    //     webpackChunkbitburner.push([
    //         [-1],
    //         {},
    //         (w) => (globalThis.webpack_require = w)
    //     ]);
    // let player;
    // Object.keys(globalThis.webpack_require.m).forEach((k) => {
    //     Object.values(globalThis.webpack_require(k)).forEach((p) => {
    //         // Check if the object looks like the player object by inspecting its properties
    //         try {
    //             Object.keys(p).forEach((key) => {
    //                 if (key.includes("hp")) {
    //                     ns.tprint(p);
    //                 }
    //             });
    //         } catch {
    //             ns.tprint("error reading prop for");
    //         }
    //     });
    // });
    // if (player) {
    //     ns.tprint(`Player object found: ${JSON.stringify(player)}`);
    // } else {
    //     ns.tprint("Player object not found.");
    // }
}
