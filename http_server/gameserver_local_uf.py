import asyncio
import json
from typing import Any

import numpy as np
import websockets

# from Go.Go import Go
from Go.Go_uf import Go_uf, UnionFind

State = np.ndarray[Any, np.dtype[np.int8]]


class GameServerGo:
    def __init__(self, board_size: int) -> None:
        self.client_connected = asyncio.Event()
        self.recv_lock = asyncio.Lock()
        self.message_queue: asyncio.Queue[str] = asyncio.Queue()
        self.server = None  # Reference to the server task

        self.go = Go_uf(board_size, np.zeros((board_size, board_size), dtype=np.int8), 0)

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

    async def reset_game(self, opponent: str, is_white: bool = False) -> tuple[State, float]:
        res = await self.send_request(
            {"command": "reset_game", "opponent": opponent, "boardSize": self.go.board_height, "playAsWhite": is_white}
        )
        enc_s = self.go.encode_state(res["board"])
        return enc_s, res["komi"]

    def request_valid_moves(
        self,
        is_white: bool,
        uf: UnionFind,
        history: list[np.uint64] = [],
    ) -> np.ndarray[Any, np.dtype[np.bool_]]:
        v = self.go.get_valid_moves(uf, is_white, history)
        return v

    async def make_move(self, action: tuple[int, int], action_idx: int, is_white: bool) -> tuple[UnionFind, int, bool]:
        # make move in bitburner
        if action == (-1, -1):  # pass
            res = await self.send_request({"command": "pass_turn", "playAsWhite": is_white})
        else:
            x, y = action
            res = await self.send_request({"command": "make_move", "x": x, "y": y, "playAsWhite": is_white})
        print(res)
        assert res != -2, f"This was an invalid move somehow: {res}"

        # make same move locally
        uf, r, d = self.go.make_move(action_idx, is_white)

        next_state = res.get("board", [])
        reward = res.get("outcome", 0)
        done = res.get("done", False)

        # DEBUGGING ONLY: check both are equal
        assert np.array_equal(
            uf.state, self.go.encode_state(next_state)
        ), f"State mismatch: left: {uf}, right: {self.go.encode_state(next_state)}"
        assert r == reward, f"Reward mismatch: left: {r}, right: {reward}"
        assert d == done, f"Done flag mismatch: left: {d}, right: {done}"

        return uf, r, d

    async def make_move_eval(
        self, action: tuple[int, int], action_idx: int, is_white: bool
    ) -> tuple[UnionFind, float, bool]:
        # make move in bitburner
        if action == (-1, -1):  # pass
            res = await self.send_request({"command": "pass_turn", "playAsWhite": is_white})
        else:
            x, y = action
            res = await self.send_request({"command": "make_move", "x": x, "y": y, "playAsWhite": is_white})
        print(res)

        # make same move locally
        self.go.make_move(action_idx, is_white)

        next_state = res["board"]
        outcome = res["outcome"]
        done = res["done"]
        pos = res.get("pos", -1)

        if pos != -1:
            uf, r, d = self.go.make_move(pos, not is_white)
        else:
            uf, r, d = self.go.make_move(self.go.board_size, not is_white)

        assert np.array_equal(
            uf.state, self.go.encode_state(next_state)
        ), f"State mismatch: left: {uf.state}, right: {self.go.encode_state(next_state)}"
        assert r == outcome, f"Reward mismatch: left: {r}, right: {outcome}"
        assert d == done, f"Done flag mismatch: left: {d}, right: {done}"

        return uf, outcome, done

    def get_game_history(self) -> list[State]:
        history = self.go.get_history()
        return history

    def get_hash_history(self) -> list[np.uint64]:
        hash_history = self.go.get_hash_history()
        return hash_history

    def get_score(self, uf: UnionFind) -> dict[str, dict[str, float]]:
        scores = self.go.get_score(uf, self.go.komi)
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
