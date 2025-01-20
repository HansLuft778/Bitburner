import { GoOpponent } from "@/NetscriptDefinitions.js";

interface RequestData {
    command: string;
    x?: number;
    y?: number;
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

/**
 * function that receives commands from the server and plays them.
 * @param ns
 */
export async function waitForIncomingRequests(ns: NS) {
    const socket = new WebSocket("ws://localhost:8765");
    await waitForOpen(socket, ns);

    let num_wins = 0;
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

        // Send only one response per request
        if (requestData.command === "get_state") {
            const gameState = {
                board: ns.go.getBoardState(),
                current_player: "X",
                history: ns.go.getMoveHistory()
            };
            socket.send(JSON.stringify({ state: gameState }));
        } else if (requestData.command === "get_valid_moves") {
            const validMoves = ns.go.analysis.getValidMoves();
            const canPass = ns.go.getGameState().currentPlayer == "None" ? false : true;
            socket.send(JSON.stringify({ validMoves, canPass }));
        } else if (requestData.command === "make_move") {
            console.log("make move");

            if (requestData.x != null && requestData.y != null) {
                console.log("all set");

                const result = await ns.go.makeMove(requestData.x, requestData.y);

                ns.go.getBoardState;

                let reward = 0;
                let done = false;
                const state = ns.go.getGameState();
                if (result.type == "gameOver") {
                    done = true;
                    reward = state.blackScore > state.whiteScore ? 1 : -1;
                    if (reward === 1) num_wins++;
                } else {
                    // reward shaping
                    // if agent made move with better score, reward it
                    reward = state.blackScore > state.whiteScore ? 0.01 : 0;
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
            const result = await ns.go.passTurn();

            let reward = 0;
            let done = false;
            if (result.type == "gameOver") {
                const state = ns.go.getGameState();
                done = true;
                reward = state.blackScore > state.whiteScore ? 1 : -1;
                if (reward === 1) num_wins++;
            }

            socket.send(
                JSON.stringify({ board: ns.go.getBoardState(), reward: reward, done: done })
            );
        } else if (requestData.command == "reset_game") {
            if (num_games > 0)
                ns.print(`winrate: ${num_wins}/${num_games} = ${num_wins / num_games}`);
            num_games++;

            const randNum = Math.random();
            let opponent: GoOpponent;
            if (randNum < 1 / 3) opponent = "Netburners";
            else if (randNum < 2 / 3) opponent = "Slum Snakes";
            else opponent = "The Black Hand";

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
