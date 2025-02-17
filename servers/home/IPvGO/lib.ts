export interface GoScore {
    black: number;
    white: number;
}

export function setBoardPosition(x: number, y: number, state: string[], to: string): string[] {
    return state.map((row, i) => {
        if (i === x) {
            return row.slice(0, y) + to + row.slice(y + 1);
        }
        return row;
    });
}

export function get_state_after_move(
    ns: NS,
    x: number,
    y: number,
    state: string[],
    isWhite: boolean
): string[] {
    let stateAfterMove = setBoardPosition(x, y, state, isWhite ? "O" : "X");
    const libsAfterMove = ns.go.analysis.getLiberties(stateAfterMove);

    for (let row = 0; row < state.length; row++) {
        for (let col = 0; col < state[0].length; col++) {
            if (
                libsAfterMove[row][col] === -1 &&
                stateAfterMove[row][col] !== "." &&
                stateAfterMove[row][col] !== "#"
            ) {
                // if (ns.go.analysis.getLiberties(state)[row][col] !== -1 && libsAfterMove[row][col] === -1) {
                stateAfterMove = setBoardPosition(row, col, stateAfterMove, ".");
            }
        }
    }

    return stateAfterMove;
}

export function getScores(ns: NS, boardState: string[]): GoScore {
    const komi = 5.5;
    const emptyNodes = ns.go.analysis.getControlledEmptyNodes(boardState);
    const eyesX = emptyNodes.reduce((sum, row) => sum + (row.match(/X/g) || []).length, 0);
    const eyesO = emptyNodes.reduce((sum, row) => sum + (row.match(/O/g) || []).length, 0);

    const controlledX = boardState.reduce((sum, row) => sum + (row.match(/X/g) || []).length, 0);
    const controlledO = boardState.reduce((sum, row) => sum + (row.match(/O/g) || []).length, 0);

    return {
        black: controlledX + eyesX,
        white: controlledO + eyesO + komi
    };
}

export async function main(ns: NS) {
    ns.clearLog();
    ns.tail();

    const board = ["OXX..", ".....", "..#..", "...XX", "...X."];
    const score = getScores(ns, board);
}
