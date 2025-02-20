import asyncio
import json
import numpy as np
import websockets
from Go.Go import Go


class GameServerGo:
    def __init__(self) -> None:
        self.client_connected = asyncio.Event()
        self.websocket = None
        self.recv_lock = asyncio.Lock()
        self.message_queue: asyncio.Queue[str] = asyncio.Queue()
        self.server = None  # Reference to the server task

        self.go = Go(5, 5, ["#####", "#####", "#####", "#####", "#####"])

    async def handle_client(self, websocket):
        self.websocket = websocket
        self.client_connected.set()
        async for message in websocket:
            await self.message_queue.put(message)

    async def send_request(self, command):
        try:
            await self.websocket.send(json.dumps(command))
            async with self.recv_lock:
                response = await self.message_queue.get()
                # response = await asyncio.wait_for(
                #     self.message_queue.get(), timeout=10
                # )  # Added timeout
            return json.loads(response)
        except asyncio.TimeoutError:
            print("Request timed out")
            return None
        except Exception as e:
            print(f"Error waiting for response: {e}")
            return None

    async def reset_game(self, no_ai: bool) -> list[str]:
        res = await self.send_request({"command": "reset_game", "noAI": no_ai})
        return res["board"]

    def request_valid_moves(
        self, is_white: bool, state: list[str], history: list[list[str]] = []
    ):
        enc_h = [self.go.encode_state(e) for e in history]
        enc_s = self.go.encode_state(state)
        v = self.go.get_valid_moves(enc_s, is_white, enc_h)
        return np.append(v, True)

    async def make_move(
        self, action: tuple[int, int], action_idx: int, is_white: bool
    ) -> tuple[list[str], float, bool]:
        if action == (-1, -1):  # pass
            res = await self.send_request(
                {"command": "pass_turn", "playAsWhite": is_white}
            )
        else:
            x, y = action
            res = await self.send_request(
                {"command": "make_move", "x": x, "y": y, "playAsWhite": is_white}
            )

        self.go.make_move(action_idx, is_white)
        print(res)
        next_state = res.get("board", [])
        reward = res.get("outcome", 0.0)
        done = res.get("done", False)
        return next_state, reward, done

    def get_game_history(self) -> list[list[str]]:
        history = self.go.get_history_decoded()
        return history

    def get_state_after_move(self, action: int, state: list[str], is_white: bool):
        res = self.go.state_after_action(action, is_white, state)
        dec_s = self.go.decode_state(res)
        return dec_s

    def get_score(self, state: list[str]):
        scores = self.go.get_score()
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
        game_state = await self.get_state()
        self.go = Go(5, 5, game_state)
