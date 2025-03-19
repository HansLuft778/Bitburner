import math
import time
from collections import defaultdict
from typing import Any, Union

import numpy as np
import torch

from agent_cnn_zero import AlphaZeroAgent
from gameserver_local_uf import GameServerGo
from Go.Go_uf import Go_uf, UnionFind
from plotter import Plotter
from TreePlotter import TreePlot  # pyright: ignore

State = np.ndarray[Any, np.dtype[np.int8]]


def get_ucb_value(child: "Node", parent_visit_count: int, c_puct: float = 2.0) -> float:
    """
    child.win_sum is from the childs perspective, so from the parent's perspective we must negate it
      Q(s,a) = child.win_sum/child.visit_cnt
      U(s,a) = c_puct * P(s,a)* sqrt(parent.visit_cnt)/(1+child.visit_cnt)
    """
    if child.visit_cnt == 0:
        q_value = 0.0
    else:
        q_value = -(child.win_sum / child.visit_cnt)
    u_value = c_puct * child.selected_policy * math.sqrt(parent_visit_count) / (1 + child.visit_cnt)
    return q_value + u_value


class Node:
    def __init__(
        self,
        # state: State,
        uf: UnionFind,
        server: GameServerGo,
        is_white: bool,
        agent: AlphaZeroAgent,
        parent: Union["Node", None] = None,
        action: int | None = None,
        selected_policy: float = 0,
        visit_count: int = 0,
    ):
        assert uf.state.shape == (
            agent.board_width,
            agent.board_width,
        ), f"Array must be 5x5: {uf.state}"
        assert np.all(np.isin(uf.state, [0, 1, 2, 3])), f"Array must only contain values 0, 1, 2, or 3: {uf.state}"

        # self.state = state
        self.uf: UnionFind = uf
        self.parent = parent
        self.action = action
        self.server = server
        self.is_white = is_white
        self.agent = agent
        self.selected_policy = selected_policy
        self.valid_moves: np.ndarray[Any, np.dtype[np.bool_]] | None = None

        self.children: list[Node] = []

        self.visit_cnt = visit_count
        self.win_sum = 0.0

    def is_fully_expanded(self):
        # enough to check for childs > 0 since we expand to all possible states at once
        return len(self.children) > 0

    def get_valid_moves(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        if self.valid_moves is None:
            if self.parent is not None:
                history = self.get_hash_history()
                self.valid_moves = self.server.request_valid_moves(self.is_white, self.uf, history)
            else:
                self.valid_moves = self.server.request_valid_moves(self.is_white, self.uf)
        return self.valid_moves

    # returns score based on current node
    # maybe should be based on root node (current player/who initiated the search)
    def get_score(self) -> float:
        score: dict[str, dict[str, float]] = self.server.get_score(self.uf)
        score_white = score["white"]["sum"]
        score_black = score["black"]["sum"]
        diff = score_white - score_black

        # normalize score, so
        max_score = self.agent.board_width * self.agent.board_height
        normalized = diff / max_score

        return normalized if self.is_white else -normalized

    def has_game_ended(self) -> tuple[bool, float]:
        board_size = self.agent.board_width * self.agent.board_height
        # both pass back-to-back
        if (
            self.action
            and self.parent
            and self.parent.action
            and self.action == board_size
            and self.parent.action == board_size
        ):
            return (True, self.get_score())

        # one pass, next player has no valid moves
        if self.parent:
            valid_moves = np.sum(self.get_valid_moves())
            if self.parent.action == board_size and valid_moves == 0:
                print(f"node.parent.action: {self.parent.action}, valid_moves: {valid_moves}")
                return (True, self.get_score())

        # board full
        space_available = np.any(self.uf.state == 0)
        if not space_available:
            return (True, self.get_score())

        return (False, 0)

    def get_history_ref(self) -> list[State]:
        history: list[State] = []
        current = self
        while self:
            history.append(self.uf.state)
            if current.parent is not None:
                current = current.parent
            else:
                break
        return history[::-1]  # Reverse for chronological order

    def get_hash_history(self) -> list[np.uint64]:
        history: list[np.uint64] = []
        current = self
        while self:
            history.append(self.uf.hash)
            if current.parent is not None:
                current = current.parent
            else:
                break
        return history[::-1]  # Reverse for chronological order

    def next(self) -> "Node":
        best: tuple[Node | None, float] = (None, -999999999)
        for c in self.children:
            score = get_ucb_value(c, self.visit_cnt, c_puct=1.8)
            if score > best[1]:
                best = (c, score)

        assert best[0] is not None
        return best[0]

    def expand(self, q_values: torch.Tensor) -> None:
        board_size = self.agent.board_width * self.agent.board_height
        for action in range(board_size + 1):
            if action == board_size:
                next_uf = self.uf.copy()
                # Penalize passing if board is mostly empty
                empty_cells = np.sum(self.uf.state == 0)
                empty_percentage = empty_cells / board_size
                if empty_percentage > 0.5:
                    q_values[action] *= (1.0 - empty_percentage) * 0.5
                # same hash, but different player to move
                next_uf.hash = self.server.go.zobrist.update_hash(
                    self.uf.hash, action, 0, 0, self.is_white, not self.is_white
                )
            else:
                color = 2 if self.is_white else 1
                is_legal, undo = self.server.go.simulate_move(self.uf, action, color, self.get_hash_history())
                if not is_legal:
                    continue

                next_uf = self.uf.copy()
                self.uf.undo_move_changes(undo, self.server.go.zobrist)

            child = Node(
                next_uf,
                self.server,
                not self.is_white,
                self.agent,
                self,
                action,
                q_values[action].item(),
            )
            self.children.append(child)

    def backprop(self, score: float):
        self.visit_cnt += 1
        self.win_sum += score

        if self.parent:
            self.parent.backprop(-score)


class MCTS:
    def __init__(
        self,
        server: GameServerGo,
        plotter: Plotter,
        agent: AlphaZeroAgent,
        search_iterations: int,
    ):
        self.search_iterations = search_iterations
        self.server = server

        self.plotter = plotter
        self.agent = agent
        self.timing_stats: defaultdict[str, float] = defaultdict(float)
        self.iterations_stats: defaultdict[str, int] = defaultdict(int)

    @torch.no_grad()  # pyright: ignore
    def search(self, uf: UnionFind, is_white: bool):
        uf.hash = self.server.go.zobrist.compute_hash(uf.state, is_white)
        root: Node = Node(uf, self.server, is_white, self.agent)

        self.timing_stats.clear()
        self.iterations_stats.clear()
        search_start = time.time()

        for _ in range(self.search_iterations):
            node = root
            # selection
            select_start = time.time()
            while node.is_fully_expanded():
                node = node.next()
            self.timing_stats["selection"] += time.time() - select_start
            self.iterations_stats["selection"] += 1

            # expansion
            end_check_start = time.time()
            done, value = node.has_game_ended()
            self.timing_stats["end_check"] += time.time() - end_check_start

            if not done:
                prep_start = time.time()
                valid_moves = node.get_valid_moves()
                history = node.get_history_ref()
                history.extend(self.server.get_game_history())
                self.timing_stats["move_prep"] += time.time() - prep_start

                inference_start = time.time()
                logits, value = self.agent.get_actions_eval(node.uf.state, valid_moves, history, node.is_white)
                if node.parent is None:
                    print(f"Value estimate: {value:.3f} (from {'white' if node.is_white else 'black'}'s perspective)")

                self.timing_stats["nn_inference"] += time.time() - inference_start
                self.iterations_stats["nn_inference"] += 1

                policy_start = time.time()
                valid_mask = torch.tensor(valid_moves, device=self.agent.device, dtype=torch.bool)
                logits[~valid_mask] = -1e9
                policy = torch.softmax(logits, dim=0)

                # Apply dirichlet noise
                if node.parent is None:
                    alpha = 0.2
                    dir_noise = np.random.dirichlet([alpha] * len(policy))
                    dir_noise_tensor = torch.tensor(dir_noise, device=policy.device, dtype=policy.dtype)
                    epsilon = 0.2  # noise weight
                    policy = (1 - epsilon) * policy + epsilon * dir_noise_tensor

                    # Renormalize
                    policy = policy / torch.sum(policy)
                self.timing_stats["policy_compute"] += time.time() - policy_start

                expand_start = time.time()
                node.expand(policy)
                self.timing_stats["expansion"] += time.time() - expand_start

            # backpropagation
            backprop_start = time.time()
            node.backprop(value)
            self.timing_stats["backprop"] += time.time() - backprop_start
            # plot tree from root node for debugging
            # if iter == 999:
            #     TreePlot(root).create_tree()

        final_policy_start = time.time()

        props = torch.zeros(
            self.agent.board_height * self.agent.board_height + 1,
            device=self.agent.device,
        )
        for c in root.children:
            props[c.action] = c.visit_cnt
        props /= props.sum()
        self.timing_stats["final_policy"] += time.time() - final_policy_start
        total_time = time.time() - search_start
        self.timing_stats["total"] = total_time

        # Print timing statistics
        print("\nMCTS Timing Statistics:")
        print(f"Total search time: {total_time:.3f}s")
        for key, val in self.timing_stats.items():
            if key != "total":
                percentage = (val / total_time) * 100
                avg_time = val / float(self.iterations_stats.get(key, self.search_iterations))
                print(f"{key}: {val:.3f}s ({percentage:.1f}%) - Avg: {avg_time*1000:.2f}ms")

        return props


def choose_action(pi: torch.Tensor, episode_length: int) -> int:
    if episode_length < 4:
        return int(np.random.choice(torch.nonzero(pi[:-1]).flatten().cpu()))
    return int(torch.argmax(pi).item())


async def main() -> None:
    torch.manual_seed(0)  # pyright: ignore
    np.random.seed(0)

    board_size = 5
    server = GameServerGo(board_size)
    await server.wait()
    print("GameServer ready and client connected")

    plotter = Plotter()
    agent = AlphaZeroAgent(board_size, plotter)
    # agent.load_checkpoint("checkpoint_69.pth")
    mcts = MCTS(server, plotter, agent, search_iterations=1000)

    NUM_EPISODES = 600
    outcome = 0
    for iter in range(NUM_EPISODES):
        state, komi = await server.reset_game("Debug")
        server.go = Go_uf(board_size, state, komi)

        buffer: list[tuple[State, bool, torch.Tensor, list[State]]] = []  # score: dict[str, dict[str, Any]]]
        is_white = False
        done = False
        game_history = [state]
        mcts.agent.policy_net.eval()
        episode_length = 0
        while not done:
            before = time.time()
            pi_mcts = mcts.search(server.go.uf, is_white)
            after = time.time()
            print(f"TOOK: {after-before}s")
            print(pi_mcts)
            # best_move = int(torch.argmax(pi_mcts).item())
            best_move = choose_action(pi_mcts, episode_length)
            print(f"{best_move}, {pi_mcts[best_move]}")
            action = mcts.agent.decode_action(best_move)
            print(f"make move: {action}")

            buffer.append((state, is_white, pi_mcts, game_history[: mcts.agent.num_past_steps]))

            # outcome is: 1 if black won, -1 is white won
            next_state, outcome, done = await server.make_move(action, best_move, is_white)
            game_history.insert(0, next_state)

            is_white = not is_white
            state = next_state
            episode_length += 1

        assert outcome != 0, "outcome should not be 0 after a game ended"

        mcts.agent.plotter.update_wins_white(1 if outcome == -1 else -1)
        mcts.agent.plotter.update_wins_black(1 if outcome == 1 else -1)

        mcts.agent.policy_net.train()
        for state, was_white, pi, history in buffer:
            # Flip if the buffer entry belongs to the opposite color
            #  - opposite of player who moves
            z = outcome if not was_white else -outcome
            mcts.agent.augment_state(state, pi, z, history, was_white)

        if iter < 2:
            print("Skipping training")
            continue
        game_length = len(game_history)
        min_train_steps = 10  # Minimum number of training steps
        max_train_steps = 40  # Maximum number of training steps

        # Calculate training steps - scales with game length
        train_steps = min(max_train_steps, max(min_train_steps, int(game_length * 0.75)))
        train_steps = 5
        print(f"Game length: {game_length}, performing {train_steps} training steps")
        for _ in range(train_steps):
            mcts.agent.train_step()
        mcts.agent.save_checkpoint(f"checkpoint_{iter}.pth")


async def main_eval():
    server = GameServerGo(5)
    await server.wait()
    print("GameServer ready and client connected")

    plotter = Plotter()
    agent = AlphaZeroAgent(7, plotter, batch_size=128)
    agent.load_checkpoint("checkpoint_56.pth")
    mcts = MCTS(server, plotter, agent, search_iterations=1000)

    NUM_EPISODES = 100
    outcome = 0
    for _ in range(NUM_EPISODES):
        state, komi = await server.reset_game("Netburners")
        server.go = Go_uf(7, state, komi)

        is_white = False
        done = False
        mcts.agent.policy_net.eval()
        while not done:
            pi_mcts = mcts.search(server.go.uf, is_white)
            print(pi_mcts)
            best_move = int(torch.argmax(pi_mcts).item())
            print(f"{best_move}, {pi_mcts[best_move]}")
            action = mcts.agent.decode_action(best_move)
            print(f"make move: {action}")

            # outcome is: 1 if black won, -1 is white won
            next_state, outcome, done = await server.make_move_eval(action, best_move, is_white)
            state = next_state

        assert outcome != 0, "outcome should not be 0 after a game ended"

        mcts.agent.plotter.update_wins_white(1 if outcome == -1 else -1)
        mcts.agent.plotter.update_wins_black(1 if outcome == 1 else -1)


# Run the main coroutine
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
