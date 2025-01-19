function hasValidMoves(validMoves: boolean[][]) {
    for (let i = 0; i < validMoves.length; i++) {
        for (let j = 0; j < validMoves[i].length; j++) {
            if (validMoves[i][j]) return true;
        }
    }
    return false;
}

export async function main(ns: NS) {
    ns.clearLog();
    ns.tail();

    const go = ns.go;

    ns.print(go.analysis.getValidMoves());
    ns.print(go.getBoardState());
    go.makeMove(1, 3);

    // do {
    //     const state = go.getBoardState();
    //     const validMoves = go.analysis.getValidMoves(state);
    //     ns.print(validMoves)
    //     if (!hasValidMoves(validMoves)) {
    //         await go.passTurn();
    //         continue;
    //     }

    //     let isValid = true;
    //     let x: number, y: number;
    //     do {
    //         x = Math.round(Math.random() * 4);
    //         y = Math.round(Math.random() * 4);
    //         ns.print(`chose move: ${x} ${y}`);
    //         isValid = validMoves[x][y];
    //     } while (!isValid);

    //     let res = await go.makeMove(x, y);
    //     if (res.type == "gameOver") break;
    //     if (res.type == "pass") {
    //         await go.passTurn();
    //         break;
    //     }
    //     ns.print(res);
    // } while (true);

    // ns.print(go.getGameState());
}
