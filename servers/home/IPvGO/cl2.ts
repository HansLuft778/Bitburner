import {
    getNumKillsOf_current_prev,
    getNumKillsOf_prev_prevprev,
    getRewardForKillEnemy as getNumForKillEnemy,
    getRewardForKillPlayer as getNumForKillPlayer
} from "./captureDetection.js";

interface RequestData {
    command: string;
    x?: number;
    y?: number;
    noAI?: boolean;
    playAsWhite: boolean;
}

export async function sendDataAndWaitForResponse(
    socket: WebSocket,
    data: unknown,
    ns: NS
): Promise<unknown> {
    return new Promise((resolve, reject) => {
        function handleResponse(event: MessageEvent) {
            const response = JSON.parse(event.data);
            ns.print("Received from server:", response);
            socket.removeEventListener("message", handleResponse);
            resolve(response);
        }

        socket.addEventListener("message", handleResponse);
        socket.send(JSON.stringify(data));
    });
}

function waitForOpen(socket: WebSocket, ns: NS) {
    return new Promise<void>((resolve) => {
        socket.onopen = function (event) {
            ns.print("Connected to Python RL server.");
            resolve();
        };
    });
}

async function getNextMove(ns: NS) {
    // Create a WebSocket connection to our server
    const socket = new WebSocket("ws://localhost:8765");
    await waitForOpen(socket, ns);

    const gameState = {
        board: ns.go.getBoardState(),
        current_player: "X",
        history: ns.go.getMoveHistory()
    };
    let res = await sendDataAndWaitForResponse(socket, gameState, ns);
    ns.print(res);

    socket.close();
}

function resetGame(ns: NS) {

}

/**
 * function that receives commands from the server and plays them.
 * @param ns
 */
export async function waitForIncomingRequests(ns: NS) {
    const socket = new WebSocket("ws://localhost:8765");
    await waitForOpen(socket, ns);

    let num_wins_black = 0;
    let num_games = 0;

    while (true) {
        let requestData: RequestData = await new Promise((resolve) => {
            function onMessage(event: MessageEvent) {
                socket.removeEventListener("message", onMessage);
                resolve(JSON.parse(event.data));
            }
            socket.addEventListener("message", onMessage);
        });

        console.log(requestData);

        if (requestData.command === "get_state") {
            const gameState = {
                board: ns.go.getBoardState(),
                current_player: "X",
                history: ns.go.getMoveHistory()
            };
            socket.send(JSON.stringify({ state: gameState }));
        } else if (requestData.command === "get_valid_moves") {
            const validMoves = ns.go.analysis.getValidMoves(undefined, undefined, requestData.playAsWhite);
            const canPass = ns.go.getGameState().currentPlayer == "None" ? false : true;
            socket.send(JSON.stringify({ validMoves, canPass }));
        } else if (requestData.command === "make_move") {
            console.log("make move");

            if (requestData.x != null && requestData.y != null) {
                const isWhite = requestData.playAsWhite
                ns.print(`${isWhite ? "White" : "Black"}: Make Move`)
                ns.go.makeMove(requestData.x, requestData.y, isWhite);
                // wait for the one move to complete, result contains the move above
                const result = await ns.go.opponentNextTurn(false, !isWhite)
                ns.print(result)

                // reward shaping: reward agent when it captures enemy stones, normalized
                // let reward = getNumForKillPlayer(ns, result) / 25;
                // reward -= getNumForKillEnemy(ns) / 25;
                let reward = 0;
                let done = false;
                const state = ns.go.getGameState();
                if (result.type == "gameOver") {
                    done = true;
                    let score = state.blackScore > state.whiteScore ? 1 : -1;
                    score = isWhite ? score * -1 : score;
                    reward += score;
                    if (score === 1) num_wins_black++;
                } else {
                    // reward shaping
                    // if agent made move with better score, reward it
                    if (state.blackScore > state.whiteScore) {
                        reward += (isWhite ? -0.1 : 0.1);
                    } else if (state.blackScore < state.whiteScore) {
                        reward += (isWhite ? 0.1 : -0.1);
                    }
                }

                socket.send(
                    JSON.stringify({ board: ns.go.getBoardState(), reward: reward, done: done })
                );
            } else {
                throw new Error(
                    "there was a move supposed to be made, but the provided x and y coords are invalid: " +
                    requestData.x +
                    " " +
                    requestData.y
                );
            }
        } else if (requestData.command === "pass_turn") {
            ns.print(`${requestData.playAsWhite ? "White" : "Black"}: Pass turn`)
            ns.go.passTurn(requestData.playAsWhite);
            const result = await ns.go.opponentNextTurn(true, !requestData.playAsWhite)

            ns.print(result)

            let reward = 0;
            let done = false;
            if (result.type == "gameOver") {
                const state = ns.go.getGameState();
                done = true;
                let score = state.blackScore > state.whiteScore ? 1 : -1;
                score = requestData.playAsWhite ? score * -1 : score;
                reward += score;

                if (score === 1) num_wins_black++;
            }

            socket.send(
                JSON.stringify({ board: ns.go.getBoardState(), reward: reward, done: done })
            );
        } else if (requestData.command == "reset_game") {
            if (num_games > 0) {
                ns.print(`black winrate: ${num_wins_black}/${num_games} = ${num_wins_black / num_games}`);
                ns.print(`white winrate: ${num_games - num_wins_black}/${num_games} = ${(num_games - num_wins_black) / num_games}`);
            }
            num_games++;

            const randNum = Math.random();
            let opponent: GoOpponent = "No AI";
            // randomly choose enemy
            if (requestData.noAI != true) {
                if (randNum < 1 / 3) opponent = "Netburners";
                else if (randNum < 2 / 3) opponent = "Slum Snakes";
                else opponent = "The Black Hand";
            }

            const board = ns.go.resetBoardState(opponent, 5);
            socket.send(JSON.stringify({ board: board }));
        } else if (requestData.command == "get_history") {
            const history = ns.go.getMoveHistory();
            socket.send(JSON.stringify({ history: history }));
        } else {
            socket.send(JSON.stringify({ status: "unknown_command" }));
        }
    }
}

export async function main(ns: NS) {
    ns.clearLog();
    ns.tail();

    await waitForIncomingRequests(ns);
}
