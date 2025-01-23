interface EnemyMove {
    type: "move" | "pass" | "gameOver";
    x?: number | null;
    y?: number | null;
}

export function printBoard(ns: NS, board: string[], rotate = true) {
    let chars = board.map((row) => row.split(""));
    if (rotate) {
        chars = chars[0].map((val, index) => chars.map((row) => row[row.length - 1 - index]));
    }

    ns.print("╔═══════════╗");
    for (const row of chars) {
        let str = "║ ";
        for (const ch of row) {
            str += ch + " ";
        }
        ns.print(str + "║");
    }
    ns.print("╚═══════════╝");
}

export function getNumKillsOf_prev_prevprev(ns: NS, killed: string) {
    const history = ns.go.getMoveHistory();

    const prev_state = history[0];
    const prevprev_state = history[1];

    // printBoard(ns, prevprev_state, false);
    // printBoard(ns, prev_state, false);
    // printBoard(ns, currentBoard, false);

    let num_kills = 0;
    for (let row_idx = 0; row_idx < 5; row_idx++) {
        for (let col_idx = 0; col_idx < 5; col_idx++) {
            const prev_move = prev_state[row_idx][col_idx];
            const prevprev_move = prevprev_state[row_idx][col_idx];

            if (prevprev_move === "O" && prev_move === ".") {
                num_kills++;
                // ns.print("X killed " + row_idx + " " + col_idx);
            }
        }
    }
    return num_kills;
}

export function getNumKillsOf_current_prev(ns: NS, killed: string) {
    const history = ns.go.getMoveHistory();
    const currentBoard = ns.go.getBoardState();
    const prevBoard = history[0];

    // printBoard(ns, prevBoard, false);
    // printBoard(ns, currentBoard, false);

    let num_kills = 0;
    for (let row_idx = 0; row_idx < 5; row_idx++) {
        for (let col_idx = 0; col_idx < 5; col_idx++) {
            if (prevBoard[row_idx][col_idx] === killed && currentBoard[row_idx][col_idx] === ".") {
                num_kills++;
                // ns.print(`killed ${row_idx} ${col_idx}`);
            }
        }
    }
    return num_kills;
}

export function getRewardForKillPlayer(ns: NS, enemyMove: EnemyMove) {
    let res = 0;
    if (enemyMove.type == "pass") {
        res = getNumKillsOf_current_prev(ns, "O");
    } else {
        res = getNumKillsOf_prev_prevprev(ns, "O");
    }

    return res;
}

export function getRewardForKillEnemy(ns: NS) {
    let res = 0;
    res = getNumKillsOf_current_prev(ns, "X");

    return res;
}

export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();

    const go = ns.go;

    ns.print(`X killed (enemy moved): ${getRewardForKillPlayer(ns, { type: "move" })}`);
    ns.print(`X killed: (enemy passed): ${getRewardForKillPlayer(ns, { type: "pass" })}`);
    ns.print(`O killed: ${getRewardForKillEnemy(ns)}`);
}
