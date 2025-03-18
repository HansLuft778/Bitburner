// interface RequestData {
//     command: string;
//     x?: number;
//     y?: number;
//     opponent?: string;
//     state?: string[];
//     prev_state?: string[];
//     playAsWhite: boolean;
// }

// export interface GoScore {
//     black: number;
//     white: number;
// }

// export function getScores(ns: NS, boardState: string[]): GoScore {
//     const komi = 5.5;
//     const emptyNodes = ns.go.analysis.getControlledEmptyNodes(boardState);
//     const eyesX = emptyNodes.reduce((sum, row) => sum + (row.match(/X/g) || []).length, 0);
//     const eyesO = emptyNodes.reduce((sum, row) => sum + (row.match(/O/g) || []).length, 0);

//     const controlledX = boardState.reduce((sum, row) => sum + (row.match(/X/g) || []).length, 0);
//     const controlledO = boardState.reduce((sum, row) => sum + (row.match(/O/g) || []).length, 0);

//     return {
//         black: controlledX + eyesX,
//         white: controlledO + eyesO + komi
//     };
// }

// function setBoardPosition(x: number, y: number, state: string[], to: string): string[] {
//     return state.map((row, i) => {
//         if (i === x) {
//             return row.slice(0, y) + to + row.slice(y + 1)
//         }
//         return row
//     })
// }

// function get_state_after_move(ns: NS, x: number, y: number, state: string[], isWhite: boolean): string[] {
//     let stateAfterMove = setBoardPosition(x, y, state, (isWhite ? "O" : "X"))
//     const libsAfterMove = ns.go.analysis.getLiberties(stateAfterMove)

//     for (let row = 0; row < state.length; row++) {
//         for (let col = 0; col < state[0].length; col++) {
//             if (libsAfterMove[row][col] === -1 && stateAfterMove[row][col] !== "." && stateAfterMove[row][col] !== "#") {
//                 // if (ns.go.analysis.getLiberties(state)[row][col] !== -1 && libsAfterMove[row][col] === -1) {
//                 stateAfterMove = setBoardPosition(row, col, stateAfterMove, ".")
//             }
//         }
//     }

//     return stateAfterMove
// }

// export async function sendDataAndWaitForResponse(
//     socket: WebSocket,
//     data: unknown,
//     ns: NS
// ): Promise<unknown> {
//     return new Promise((resolve, reject) => {
//         function handleResponse(event: MessageEvent) {
//             const response = JSON.parse(event.data);
//             ns.print("Received from server:", response);
//             socket.removeEventListener("message", handleResponse);
//             resolve(response);
//         }

//         socket.addEventListener("message", handleResponse);
//         socket.send(JSON.stringify(data));
//     });
// }

// function waitForOpen(socket: WebSocket, ns: NS) {
//     return new Promise<void>((resolve) => {
//         socket.onopen = function (event) {
//             ns.print("Connected to Python RL server.");
//             resolve();
//         };
//     });
// }


// /**
//  * function that receives commands from the server and plays them.
//  * @param ns
//  */
// export async function waitForIncomingRequests(ns: NS) {
//     const socket = new WebSocket("ws://localhost:8765");
//     await waitForOpen(socket, ns);

//     let num_wins_black = 0;
//     let num_games = 0;
//     let has_ai_opponent = false;

//     while (true) {
//         let requestData: RequestData = await new Promise((resolve) => {
//             function onMessage(event: MessageEvent) {
//                 socket.removeEventListener("message", onMessage);
//                 resolve(JSON.parse(event.data));
//             }
//             socket.addEventListener("message", onMessage);
//         });

//         console.log(requestData);

//         if (requestData.command === "get_state") {
//             const gameState = {
//                 board: ns.go.getBoardState(),
//                 current_player: "X",
//                 history: ns.go.getMoveHistory()
//             };
//             socket.send(JSON.stringify({ state: gameState }));
//         } else if (requestData.command === "get_valid_moves") {
//             let validMoves = []
//             if (requestData.state!.length > 0) {
//                 validMoves = ns.go.analysis.getValidMoves(requestData.state, requestData.prev_state, requestData.playAsWhite);
//             } else {
//                 validMoves = ns.go.analysis.getValidMoves(undefined, undefined, requestData.playAsWhite);
//             }
//             const canPass = ns.go.getGameState().currentPlayer == "None" ? false : true;
//             socket.send(JSON.stringify({ validMoves, canPass }));
//         } else if (requestData.command === "make_move") {
//             console.log("make move");

//             if (requestData.x != null && requestData.y != null) {
//                 const isWhite = requestData.playAsWhite
//                 ns.print(`${isWhite ? "White" : "Black"}: Make Move`)
//                 ns.go.makeMove(requestData.x, requestData.y, isWhite);

//                 // wait for the one move to complete, result contains the move above
//                 let result;
//                 if (!has_ai_opponent) result = await ns.go.opponentNextTurn(false, !isWhite)
//                 else result = await ns.go.opponentNextTurn(true, isWhite)

//                 let outcome = 0;
//                 let done = false;
//                 const state = ns.go.getGameState();
//                 if (result.type == "gameOver") {
//                     done = true;

//                     outcome = state.blackScore > state.whiteScore ? 1 : -1
//                     if (state.blackScore > state.whiteScore) num_wins_black++;
//                 }

//                 socket.send(
//                     JSON.stringify({ board: ns.go.getBoardState(), outcome: outcome, done: done, opponent_move_type: result.type, x: result.x, y: result.y })
//                 );
//             } else {
//                 throw new Error(
//                     "there was a move supposed to be made, but the provided x and y coords are invalid: " +
//                     requestData.x +
//                     " " +
//                     requestData.y
//                 );
//             }
//         } else if (requestData.command === "pass_turn") {
//             ns.print(`${requestData.playAsWhite ? "White" : "Black"}: Pass turn`)
//             // ns.go.passTurn(requestData.playAsWhite);
//             // const result = await ns.go.opponentNextTurn(true, !requestData.playAsWhite)

//             const promise = ns.go.passTurn(requestData.playAsWhite);
//             const promoise2 = ns.go.opponentNextTurn(true, !requestData.playAsWhite)

//             const result = await Promise.any([promise, promoise2])

//             // ns.print(result)

//             let outcome = 0;
//             let done = false;
//             if (result.type == "gameOver") {
//                 const state = ns.go.getGameState();
//                 done = true;
//                 // const rootColor = requestData.playAsWhite
//                 // outcome = state.blackScore > state.whiteScore ? 1 : -1;
//                 // outcome *= rootColor ? -1 : 1
//                 outcome = state.whiteScore > state.blackScore ? -1 : 1
//                 if (state.blackScore > state.whiteScore) num_wins_black++;
//             }

//             socket.send(
//                 JSON.stringify({ board: ns.go.getBoardState(), outcome: outcome, done: done })
//             );
//         } else if (requestData.command == "reset_game") {
//             if (num_games > 0) {
//                 ns.print(`black winrate: ${num_wins_black}/${num_games} = ${num_wins_black / num_games}`);
//                 ns.print(`white winrate: ${num_games - num_wins_black}/${num_games} = ${(num_games - num_wins_black) / num_games}`);
//             }
//             num_games++;

//             let opponent: string = requestData.opponent!;
//             has_ai_opponent = true;
//             if (opponent == "No AI") {
//                 has_ai_opponent = false;
//             }

//             // randomly choose enemy
//             if (requestData.opponent == "random") {
//                 const randNum = Math.random();
//                 if (randNum < 1 / 3) opponent = "Netburners";
//                 else if (randNum < 2 / 3) opponent = "Slum Snakes";
//                 else opponent = "The Black Hand";
//             }

//             const board = ns.go.resetBoardState(opponent as GoOpponent, 5);
//             socket.send(JSON.stringify({ board: board }));
//         } else if (requestData.command == "get_history") {
//             const history = ns.go.getMoveHistory();
//             socket.send(JSON.stringify({ history: history }));
//         } else if (requestData.command == "get_state_after_move") {
//             const x = requestData.x!
//             const y = requestData.y!
//             const state = requestData.state!
//             const isWhite = requestData.playAsWhite!
//             let stateAfterMove = get_state_after_move(ns, x, y, state, isWhite)
//             socket.send(JSON.stringify({ state: stateAfterMove }));
//         } else if (requestData.command == "get_score") {
//             const scores = getScores(ns, requestData.state!);
//             socket.send(JSON.stringify(scores));
//         }
//         else {
//             socket.send(JSON.stringify({ status: "unknown_command" }));
//         }
//     }
// }

// export async function main(ns: NS) {
//     ns.clearLog();
//     ns.tail();

//     await waitForIncomingRequests(ns);
// }
