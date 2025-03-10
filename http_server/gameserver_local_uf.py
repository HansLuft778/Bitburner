import asyncio
import json
from typing import Any

import numpy as np
import websockets

# from Go.Go import Go
from Go.Go_uf_copy import Go_uf, UnionFind

State = np.ndarray[Any, np.dtype[np.int8]]


class GameServerGo:
    def __init__(self, board_size: int) -> None:
        self.client_connected = asyncio.Event()
        self.recv_lock = asyncio.Lock()
        self.message_queue: asyncio.Queue[str] = asyncio.Queue()
        self.server = None  # Reference to the server task

        self.go = Go_uf(
            board_size, np.zeros((board_size, board_size), dtype=np.int8), 0
        )
    async def handle_client(self, websocket: Any):
        self.websocket = websocket
        self.client_connected.set()
        async for message in websocket:
            await self.message_queue.put(message)

    async def send_request(self, command: dict[str, Any]) -> dict[str, Any]:
        try:
            await self.websocket.send(json.dumps(command))
            async with self.recv_lock:
                response = await self.message_queue.get()
            return json.loads(response)
        except asyncio.TimeoutError:
            print("Request timed out")
            raise TimeoutError
        except Exception as e:
            print(f"Error waiting for response: {e}")
            raise e

    async def reset_game(self, opponent: str) -> tuple[State, float]:
        res = await self.send_request({"command": "reset_game", "opponent": opponent})
        enc_s = self.go.encode_state(res["board"])
        return enc_s, res["komi"]

    def request_valid_moves(
        self,
        is_white: bool,
        state: State,
        uf: UnionFind,
        history: list[State] = [],
    ) -> np.ndarray[Any, np.dtype[np.bool_]]:
        assert state.shape == (
            self.go.board_height,
            self.go.board_height,
        ), f"Array must be 5x5: {state}"
        assert np.all(
            np.isin(state, [0, 1, 2, 3])
        ), f"Array must only contain values 0, 1, 2, or 3: {state}"

        v = self.go.get_valid_moves(state, uf, is_white, history)
        return np.append(v, True)
    

    async def make_move(
        self, action: tuple[int, int], action_idx: int, is_white: bool
    ) -> tuple[State, int, bool]:
        # make move in bitburner
        if action == (-1, -1):  # pass
            res = await self.send_request(
                {"command": "pass_turn", "playAsWhite": is_white}
            )
        else:
            x, y = action
            res = await self.send_request(
                {"command": "make_move", "x": x, "y": y, "playAsWhite": is_white}
            )

        # make same move locally
        s, r, d = self.go.make_move(action_idx, is_white)
        print(res)
        next_state = res.get("board", [])
        reward = res.get("outcome", 0)
        done = res.get("done", False)

        # DEBUGGING ONLY: check both are equal
        assert np.array_equal(
            s, self.go.encode_state(next_state)
        ), f"State mismatch: left: {s}, right: {self.go.encode_state(next_state)}"
        assert r == reward, f"Reward mismatch: left: {r}, right: {reward}"
        assert d == done, f"Done flag mismatch: left: {d}, right: {done}"

        return s, r, d

    async def make_move_eval(
        self, action: tuple[int, int], action_idx: int, is_white: bool
    ) -> tuple[State, float, bool]:
        # make move in bitburner
        if action == (-1, -1):  # pass
            res = await self.send_request(
                {"command": "pass_turn", "playAsWhite": is_white}
            )
        else:
            x, y = action
            res = await self.send_request(
                {"command": "make_move", "x": x, "y": y, "playAsWhite": is_white}
            )

        # make same move locally
        self.go.make_move(action_idx, is_white)
        print(res)

        next_state = res["board"]
        outcome = res["outcome"]
        done = res["done"]

        enc_state = self.go.encode_state(next_state)
        self.go.state = enc_state

        self.go.history.append(self.go.state)

        print(f"state: {enc_state} reward: {outcome} done: {done}")
        return enc_state, outcome, done

    def get_game_history(self) -> list[State]:
        history = self.go.get_history()
        return history

    def get_state_after_move(
        self,
        action: int,
        state: State,
        is_white: bool,
        uf: UnionFind,
        additional_history: list[State] = [],
    ) -> tuple[State, UnionFind]:
        res = self.go.state_after_action(
            action, is_white, state, uf, additional_history
        )
        assert res[0].shape == (
            self.go.board_height,
            self.go.board_height,
        ), f"Array must be 5x5: action: {action} state: {res[0]}"
        assert np.all(
            np.isin(res[0], [0, 1, 2, 3])
        ), f"Array must only contain values 0, 1, 2, or 3: {res[0]}"
        return res

    def get_score(self) -> dict[str, dict[str, float]]:
        scores = self.go.get_score(self.go.komi)
        return scores

    async def get_state(self) -> list[str]:
        state = await self.send_request({"command": "get_state"})
        return state["state"]["board"]

    async def start_server(self):
        self.server = await websockets.serve(self.handle_client, "localhost", 8765)
        print("Server started on ws://localhost:8765")

    async def wait_for_client(self):
        print("Waiting for client to connect...")
        await self.client_connected.wait()
        print("Client connected. Server is running.")

    async def wait(self):
        # Start the server as a background task
        asyncio.create_task(self.start_server())
        # Wait for the client to connect
        await self.wait_for_client()
