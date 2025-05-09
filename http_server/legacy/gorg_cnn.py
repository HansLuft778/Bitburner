import asyncio
import datetime
import json
import os

import numpy as np
import torch
import websockets

from legacy.agent_cnn import DQNAgentCNN
from plotter import Plotter


class GameServer:
    def __init__(self, board_width: int, board_height: int):
        self.client_connected = asyncio.Event()
        self.websocket = None
        self.recv_lock = asyncio.Lock()
        self.message_queue: asyncio.Queue[str] = asyncio.Queue()

        self.plotter = Plotter()
        self.agent = DQNAgentCNN(board_width, board_height, self.plotter)

        self.episodes = 2000
        self.steps_per_episode = 50
        self.target_update_freq = 20  # update target net every X episodes

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
            return json.loads(response)
        except asyncio.TimeoutError:
            print("Request timed out")
            return None
        except Exception as e:
            print(f"Error waiting for response: {e}")
            return None

    async def reset_game(self) -> list[str]:
        res = await self.send_request({"command": "reset_game"})
        return res["board"]

    async def request_valid_moves(self):
        valid_moves = await self.send_request({"command": "get_valid_moves"})
        print(valid_moves)
        return np.append(
            np.array(valid_moves["validMoves"]).flatten(),
            valid_moves["canPass"],
        )

    async def make_move(self, action) -> tuple[list[str], float, bool]:
        print(f"making move :{action}:")
        if action == "pass":
            res = await self.send_request({"command": "pass_turn"})
        else:
            x, y = action
            res = await self.send_request({"command": "make_move", "x": x, "y": y})

        print(res)
        next_state = res["board"]
        reward = res["reward"]
        done = res["done"]
        return next_state, reward, done

    async def get_game_history(self) -> list[list[str]]:
        histroy = await self.send_request({"command": "get_history"})
        return histroy["history"]

    async def train_loop(self) -> None:
        """
        Main training loop:
        - For each episode:
          - reset the game
          - step through until done or step limit
          - do DQN updates
        """

        for ep in range(self.episodes):
            board = await self.reset_game()
            done = False
            step_count = 0

            while not done and step_count < self.steps_per_episode:
                # 1. Agent selects action
                valid_moves: list[bool] = await self.request_valid_moves()
                print(f"invalid moves: {valid_moves}")
                game_history = await self.get_game_history()
                action_idx = self.agent.select_action(board, valid_moves, game_history)

                action_decoded = self.agent.decode_action(action_idx)
                print(f"chose {action_idx} which corresponds to {action_decoded}")

                # 2. Current state tensor
                state_tensor = self.agent.preprocess_state(board, game_history)

                # 3. Execute in environment
                next_board, reward, done = await self.make_move(action_decoded)
                if done and reward > 0:
                    self.plotter.update_value_loss(1)
                elif done and reward < 0:
                    self.plotter.update_value_loss(0)

                self.plotter.update_wins_black(reward)

                next_valid_moves = await self.request_valid_moves()
                print(f"next valid moves: {next_valid_moves}")

                # 4. Next state
                next_game_history = await self.get_game_history()
                next_state_tensor = self.agent.preprocess_state(
                    next_board, next_game_history
                )

                # 5. Store transition in replay
                self.agent.replay_buffer.push(
                    state_tensor,
                    action_idx,
                    reward,
                    next_state_tensor,
                    done,
                    next_valid_moves,
                )
                print(f"replay Buffer: {len(self.agent.replay_buffer)}")

                # 6. Train step
                print("before train step")
                self.agent.train_step()
                print("after train step")

                board = next_board
                step_count += 1

            # Episode done
            self.agent.decay_epsilon()

            # Periodically update target network
            if ep % self.target_update_freq == 0:
                self.agent.update_target_network()

            # torch.save(self.agent.policy_net.state_dict(), "models/model_cnn.pt")
            torch.save(
                {
                    "model_state_dict": self.agent.policy_net.state_dict(),
                    "optimizer_state_dict": self.agent.optimizer.state_dict(),
                    "epsilon": self.agent.epsilon,
                },
                f"models/checkpoints/checkpoint_{datetime.datetime.now().isoformat().replace(":", "-").split(".")[0]}.pt",
            )
            files = os.listdir("models/checkpoints/")
            if len(files) > 10:
                for file in range(len(files) - 10):
                    os.remove(f"models/checkpoints/{files[file]}")

            print(f"[Episode {ep}] done, epsilon={self.agent.epsilon:.2f}")

    async def run_server(self):
        async with websockets.serve(self.handle_client, "localhost", 8765):
            print("Server started on ws://localhost:8765")
            print("Waiting for client to connect...")
            await self.client_connected.wait()
            print("Client connected. Starting training loop.")
            await self.train_loop()
            # print(await self.get_game_history())


def main():
    server = GameServer(5, 5)
    # server = GameServer(board_width=5, board_height=5)
    asyncio.run(server.run_server())


if __name__ == "__main__":
    main()
