import asyncio
import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import websockets

from legacy.agent_cnn import DQNAgentCNN
from legacy.agent_cnn_self import DQNAgentCNNShared
from plotter import Plotter


class SelfPlayServer:
    def __init__(self, board_width: int, board_height: int):
        self.client_connected = asyncio.Event()
        self.websocket = None
        self.recv_lock = asyncio.Lock()
        self.message_queue: asyncio.Queue[str] = asyncio.Queue()
        self.plotter = Plotter()

        # We might have a single agent controlling both colors,
        # or separate agents for black and white.
        # self.agent_black = ...
        # self.agent_white = ...
        # or shared
        self.agent_shared = DQNAgentCNNShared(board_width, board_height, self.plotter)

        self.episodes = 2000
        self.steps_per_episode = 50
        self.target_update_freq = 10  # update target net every X episodes

        self.last_black_transition = None
        self.last_white_transition = None

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

    async def reset_game(self):
        """
        Tells the environment to reset a 5x5 board in 'no AI' or 'self-play' mode
        so we can supply *both* black and white moves externally.
        """
        resp = await self.send_request({"command": "reset_game", "noAI": True})
        return resp["board"]

    async def make_move(self, action, self_play: bool, play_as_white=False):
        if self_play:
            if action == "pass":
                res = await self.send_request(
                    {
                        "command": "pass_turn",
                        "playAsWhite": play_as_white,
                    }
                )
            else:
                x, y = action
                res = await self.send_request(
                    {
                        "command": "make_move",
                        "x": x,
                        "y": y,
                        "playAsWhite": play_as_white,
                    }
                )

            next_state = res["board"]
            reward = res["reward"]
            done = res["done"]
            return next_state, reward, done
        else:
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

    async def request_valid_moves(self, play_as_white: bool):
        valid_moves = await self.send_request(
            {"command": "get_valid_moves", "playAsWhite": play_as_white}
        )
        return np.append(
            np.array(valid_moves["validMoves"]).flatten(),
            valid_moves["canPass"],
        )

    async def get_game_history(self) -> list[list[str]]:
        histroy = await self.send_request({"command": "get_history"})
        return histroy["history"]

    # ==========================================================================

    async def play_one_turn(self, board, color_is_white):
        """
        Let the correct agent pick a move, do 'make_move' and then
        'opponent_next_turn' to finalize.
        """

        # Agent picks a move
        valid_moves = await self.request_valid_moves(color_is_white)
        print(f"invalid moves: {valid_moves}")
        game_history = await self.get_game_history()
        action_idx = self.agent_shared.select_action(
            board, valid_moves, game_history, color_is_white
        )
        action_decoded = self.agent_shared.decode_action(action_idx)
        print(f"chose {action_idx} which corresponds to {action_decoded}")

        # make the move
        next_board, reward, done = await self.make_move(
            action_decoded, True, play_as_white=color_is_white
        )

        # Update replay buffer
        # state before move was made
        state_tensor = self.agent_shared.preprocess_state(
            board, game_history, color_is_white
        )

        # states after move
        next_valid_moves = await self.request_valid_moves(color_is_white)
        next_game_history = await self.get_game_history()
        next_state_tensor = self.agent_shared.preprocess_state(
            next_board, next_game_history, color_is_white
        )

        transition = (
            state_tensor,
            action_idx,
            0.0,
            next_state_tensor,
            done,
            next_valid_moves,
        )
        if done:
            if color_is_white:
                # white won/lost
                if reward > 0:
                    self.finalize_transitions(white_reward=+1, black_reward=-1)
                else:
                    self.finalize_transitions(white_reward=-1, black_reward=+1)
            else:
                # black won/lost
                if reward > 0:
                    self.finalize_transitions(white_reward=-1, black_reward=+1)
                else:
                    self.finalize_transitions(white_reward=+1, black_reward=-1)
        else:
            # game not done yet, store transition for later
            if color_is_white:
                if self.last_white_transition is not None:
                    self.agent_shared.replay_buffer.push(*self.last_white_transition)
                    self.agent_shared.train_step()
                self.last_white_transition = transition
            else:
                if self.last_black_transition is not None:
                    self.agent_shared.replay_buffer.push(*self.last_black_transition)
                    self.agent_shared.train_step()
                self.last_black_transition = transition

        # self.agent_shared.replay_buffer.push(
        #     state_tensor, action_idx, reward, next_state_tensor, done, next_valid_moves
        # )

        # self.agent_shared.train_step()

        return next_board, reward, done

    def finalize_transitions(self, white_reward, black_reward):
        """Finalize the last transitions for both players."""
        self.plotter.update_wins_black(black_reward)
        self.plotter.update_wins_white(white_reward)

        # Process both transitions before training
        transitions_to_add = []

        if self.last_white_transition:
            s, a, _, ns, _, mask = self.last_white_transition
            transitions_to_add.append((s, a, white_reward, ns, True, mask))
            self.last_white_transition = None

        if self.last_black_transition:
            s, a, _, ns, _, mask = self.last_black_transition
            transitions_to_add.append((s, a, black_reward, ns, True, mask))
            self.last_black_transition = None

        # Add both transitions
        for transition in transitions_to_add:
            self.agent_shared.replay_buffer.push(*transition)

        # Train only once after both transitions are added
        if transitions_to_add:
            self.agent_shared.train_step()

    async def run_self_play_episode(self):
        # Reset
        board = await self.reset_game()
        done = False
        total_reward_black = 0
        total_reward_white = 0

        color_white_turn = False

        step_count = 0

        while not done and step_count < self.steps_per_episode:
            board, reward, done = await self.play_one_turn(board, color_white_turn)

            # color_white_turn = not color_white_turn

            step_count += 1

        # Episode done
        self.agent_shared.decay_epsilon()

        torch.save(
            {
                "model_state_dict": self.agent_shared.policy_net.state_dict(),
                "optimizer_state_dict": self.agent_shared.optimizer.state_dict(),
                "epsilon": self.agent_shared.epsilon,
            },
            f"models/checkpoints/checkpoint_{datetime.datetime.now().isoformat().replace(":", "-").split(".")[0]}.pt",
        )
        files = os.listdir("models/checkpoints/")
        if len(files) > 10:
            for file in range(len(files) - 10):
                os.remove(f"models/checkpoints/{files[file]}")

        print(
            "Episode done, black reward:",
            total_reward_black,
            "white reward:",
            total_reward_white,
        )

    async def play_one_turn_eval(self, board, color_is_white):
        """
        Evaluation version of play_one_turn that doesn't do any training
        and uses the model deterministically.
        """
        # Set model to evaluation mode
        self.agent_shared.policy_net.eval()

        with torch.no_grad():  # Disable gradient calculations
            # Get valid moves and game history
            valid_moves = await self.request_valid_moves(color_is_white)
            game_history = await self.get_game_history()

            # Get action deterministically (no epsilon-greedy)
            action_idx = self.agent_shared.select_action_eval(
                board, valid_moves, game_history, color_is_white
            )
            action_decoded = self.agent_shared.decode_action(action_idx)
            print(f"chose {action_idx} which corresponds to {action_decoded}")

            # Make the move
            next_board, reward, done = await self.make_move(action_decoded, False)

        # Don't forget to set back to train mode if you plan to train later
        self.agent_shared.policy_net.train()

        return next_board, reward, done

    async def load_and_eval(self, model_path):
        # Load the model
        checkpoint = torch.load(model_path)
        self.agent_shared.policy_net.load_state_dict(checkpoint["model_state_dict"])

        # Set to eval mode
        self.agent_shared.policy_net.eval()
        self.agent_shared.epsilon = 0

        # Run evaluation games
        board = await self.reset_game()
        done = False
        while not done:
            board, reward, done = await self.play_one_turn_eval(
                board, color_is_white=False
            )
            # if not done:
            #     board, reward, done = await self.play_one_turn_eval(
            #         board, color_is_white=True
            #     )

    async def train_loop(self):
        for ep in range(self.episodes):
            await self.run_self_play_episode()

            # Periodically update target network
            if ep % self.target_update_freq == 0:
                self.agent_shared.update_target_network()

    async def run_server(self):
        async with websockets.serve(self.handle_client, "localhost", 8765):
            print("Server started on ws://localhost:8765")
            print("Waiting for client to connect...")
            await self.client_connected.wait()
            print("Client connected. Starting training loop.")
            # await self.train_loop()
            while True:
                await self.load_and_eval(
                    "models/checkpoints/checkpoint_2025-01-22T20-29-49.pt"
                )


def main():
    server = SelfPlayServer(5, 5)
    asyncio.run(server.run_server())


if __name__ == "__main__":
    main()
