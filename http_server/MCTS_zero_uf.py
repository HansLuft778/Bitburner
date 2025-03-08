import math
import time
from collections import defaultdict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from agent_cnn_zero import AlphaZeroAgent
from gameserver_local_uf import GameServerGo
from Go.Go_uf_copy import Go_uf, UnionFind
from plotter import Plotter
from TreePlotter import TreePlot


# returns score based on current node
# maybe should be based on root node (current player/who initiated the search)
def get_score(server: GameServerGo, node: "Node"):
    score = server.get_score()
    score_white = score["white"]["sum"]
    score_black = score["black"]["sum"]
    diff = score_white - score_black

    # normalize score, so
    max_score = node.agent.board_width * node.agent.board_height
    normalized = diff / max_score

    return normalized if node.is_white else -normalized


def has_game_ended(server: GameServerGo, node: "Node") -> tuple[bool, int]:
    # both pass back-to-back
    if (
        node.action
        and node.parent
        and node.parent.action
        and node.action == server.go.board_height * server.go.board_height
        and node.parent.action == server.go.board_height * server.go.board_height
    ):
        return (True, get_score(server, node))

    # one pass, next player has no valid moves
    if node.parent:
        valid_moves = np.sum(node.get_valid_moves())
        if (
            node.parent.action == server.go.board_height * server.go.board_height
            and valid_moves == 0
        ):
            print(
                f"node.parent.action: {node.parent.action}, valid_moves: {valid_moves}"
            )
            return (True, get_score(server, node))

    # # board full
    space_available = np.any(node.state == 0)
    if not space_available:
        return (True, get_score(server, node))

    return (False, 0)


def get_ucb_value(child: "Node", parent_visit_count: int, c_puct=2.0) -> float:
    """
    child.win_sum is from the parent's perspective.
      Q(s,a) = child.win_sum/child.visit_cnt
      U(s,a) = c_puct * P(s,a)* sqrt(parent.visit_cnt)/(1+child.visit_cnt)
    """
    if child.visit_cnt == 0:
        q_value = 0.0
    else:
        q_value = -(child.win_sum / child.visit_cnt)
    u_value = (
        c_puct
        * child.selected_policy
        * math.sqrt(parent_visit_count)
        / (1 + child.visit_cnt)
    )
    return q_value + u_value


class Node:
    def __init__(
        self,
        state: np.ndarray,
        uf: UnionFind,
        server: GameServerGo,
        is_white: bool,
        agent: AlphaZeroAgent,
        parent: Union["Node", None] = None,
        action: int | None = None,
        selected_policy: float = 0,
        visit_count: int = 0,
    ):
        assert state.shape == (
            agent.board_width,
            agent.board_width,
        ), f"Array must be 5x5: {state}"
        assert np.all(
            np.isin(state, [0, 1, 2, 3])
        ), f"Array must only contain values 0, 1, 2, or 3: {state}"

        self.state = state
        self.uf = uf
        self.parent = parent
        self.action = action
        self.server = server
        self.is_white = is_white
        self.agent = agent
        self.selected_policy = selected_policy
        self.valid_moves: np.ndarray | None = None

        self.children: list[Node] = []

        self.visit_cnt = visit_count
        self.win_sum = 0.0

    def is_fully_expanded(self):
        # enough to check for childs > 0 since we expand to all possible states at once
        return len(self.children) > 0

    def get_valid_moves(self) -> np.ndarray:
        if self.valid_moves is None:
            if self.parent is not None:
                history = self.get_history()
                self.valid_moves = self.server.request_valid_moves(
                    self.is_white, self.state, self.uf, history
                )
            else:
                self.valid_moves = self.server.request_valid_moves(
                    self.is_white, self.state, self.uf
                )
        return self.valid_moves

    def get_history(self) -> list[np.ndarray]:
        history = []
        current = self
        while self:
            history.append(self.state)
            if current.parent is not None:
                current = current.parent
            else:
                break
        return history[::-1]  # Reverse for chronological order

    def next(self) -> "Node":
        best: tuple[Node | None, float] = (None, -999999999)
        for c in self.children:
            score = get_ucb_value(c, self.visit_cnt, c_puct=1.5)
            if score > best[1]:
                best = (c, score)

        assert best[0] is not None
        return best[0]

    def expand(self, q_values: torch.Tensor) -> None:
        valid_moves = self.get_valid_moves().flatten()
        empty_cells = np.sum(self.state == 0)
        board_size = self.agent.board_width * self.agent.board_height
        # Penalize passing if board is mostly empty
        empty_percentage = empty_cells / board_size
        if empty_percentage > 0.5:
            q_values[self.agent.board_width * self.agent.board_width] *= (
                1.0 - empty_percentage
            ) * 0.5

        for action, q_value in enumerate(q_values):
            if not valid_moves[action]:
                continue
            if action != self.agent.board_width * self.agent.board_width:
                next_state, next_uf = self.server.get_state_after_move(
                    action, self.state, self.is_white, self.uf, self.get_history()
                )
                assert next_state.shape == (
                    self.agent.board_width,
                    self.agent.board_width,
                ), f"Array must be 5x5: action:{action} state: {next_state}"
                assert np.all(
                    np.isin(next_state, [0, 1, 2, 3])
                ), f"Array must only contain values 0, 1, 2, or 3: {next_state}"
            else:
                next_state = self.state

            child = Node(
                next_state,
                next_uf,
                self.server,
                not self.is_white,
                self.agent,
                self,
                action,
                q_value.item(),
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

    @torch.no_grad()
    def search(self, state: np.ndarray, uf: UnionFind, is_white: bool):

        # assert state.shape == (5, 5), f"Array must be 5x5: {state}"
        assert np.all(
            np.isin(state, [0, 1, 2, 3])
        ), f"Array must only contain values 0, 1, 2, or 3: {state}"

        root: Node = Node(state, uf, self.server, is_white, self.agent)

        self.timing_stats.clear()
        self.iterations_stats.clear()
        search_start = time.time()

        for iter in range(self.search_iterations):
            node = root
            # selection
            select_start = time.time()
            while node.is_fully_expanded():
                node = node.next()
            self.timing_stats["selection"] += time.time() - select_start
            self.iterations_stats["selection"] += 1

            # expansion
            end_check_start = time.time()
            done, value = has_game_ended(self.server, node)
            self.timing_stats["end_check"] += time.time() - end_check_start

            if not done:
                prep_start = time.time()
                valid_moves = node.get_valid_moves()
                history = node.get_history()
                history.extend(self.server.get_game_history())
                self.timing_stats["move_prep"] += time.time() - prep_start

                inference_start = time.time()
                logits, value = self.agent.get_actions_eval(
                    node.state, valid_moves, history, node.is_white
                )
                if node.parent is None:
                    print(
                        f"Value estimate: {value:.3f} (from {'white' if node.is_white else 'black'}'s perspective)"
                    )

                self.timing_stats["nn_inference"] += time.time() - inference_start
                self.iterations_stats["nn_inference"] += 1

                policy_start = time.time()
                logits[~valid_moves] = -1e9
                policy = torch.softmax(logits, dim=0)

                # Apply dirichlet noise
                if node.parent is None:
                    alpha = 0.15
                    dir_noise = np.random.dirichlet([alpha] * len(policy))
                    dir_noise_tensor = torch.tensor(
                        dir_noise, device=policy.device, dtype=policy.dtype
                    )
                    epsilon = 0.15  # noise weight
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
                avg_time = val / float(
                    self.iterations_stats.get(key, self.search_iterations)
                )
                print(
                    f"{key}: {val:.3f}s ({percentage:.1f}%) - Avg: {avg_time*1000:.2f}ms"
                )

        return props


async def main():
    server = GameServerGo(5)
    await server.wait()
    print("GameServer ready and client connected")

    plotter = Plotter()
    agent = AlphaZeroAgent(5, plotter)
    # agent.load_checkpoint("checkpoint_69.pth")
    mcts = MCTS(server, plotter, agent, search_iterations=1000)

    NUM_EPISODES = 1000

    for iter in range(NUM_EPISODES):
        state, komi = await server.reset_game("No AI")
        server.go = Go_uf(5, state, komi)

        buffer = []
        is_white = False
        done = False
        game_history = [state]
        mcts.agent.policy_net.eval()
        while not done:
            before = time.time()
            pi_mcts = mcts.search(state, server.go.uf, is_white)
            after = time.time()
            print(f"TOOK: {after-before}s")
            print(pi_mcts)
            best_move = torch.argmax(pi_mcts).item()
            print(f"{best_move}, {pi_mcts[best_move]}")
            action = mcts.agent.decode_action(best_move)
            print(f"make move: {action}")
            scores = server.get_score()

            buffer.append(
                (
                    state,
                    is_white,
                    pi_mcts,
                    game_history[: mcts.agent.num_past_steps],
                    scores,
                )
            )

            # outcome is: 1 if black won, -1 is white won
            next_state, outcome, done = await server.make_move(
                action, best_move, is_white
            )
            game_history.insert(0, next_state)

            is_white = not is_white
            state = next_state

        assert outcome != 0, "outcome should not be 0 after a game ended"

        mcts.agent.plotter.update_wins_white(1 if outcome == -1 else -1)
        mcts.agent.plotter.update_wins_black(1 if outcome == 1 else -1)

        mcts.agent.policy_net.train()
        for state, was_white, pi, history, scores in buffer:
            # Flip if the buffer entry belongs to the opposite color
            #  - opposite of player who moves

            z = outcome if not was_white else -outcome

            # add bonus for territory + pieces, without komi
            white_territory = (
                scores["white"]["territory"] * 1.2 + scores["white"]["pieces"] * 0.8
            )
            black_territory = (
                scores["black"]["territory"] * 1.2 + scores["black"]["pieces"] * 0.8
            )

            territory_bonus = (
                (black_territory - white_territory)
                / (mcts.agent.board_width * mcts.agent.board_width)
                * 0.5
            )

            # z += territory_bonus if not was_white else -territory_bonus

            # move_count_bonus = min(len(game_history) / 15.0, 0.5)  # Cap at 0.5
            # z += move_count_bonus if not was_white else -move_count_bonus
            mcts.agent.augment_state(state, pi, z, history, was_white)
            # mcts.agent.train_buffer.push(state, pi, z, history, was_white)

        game_length = len(game_history)
        min_train_steps = 10  # Minimum number of training steps
        max_train_steps = 40  # Maximum number of training steps

        # Calculate training steps - scales with game length
        train_steps = min(
            max_train_steps, max(min_train_steps, int(game_length * 0.75))
        )
        print(f"Game length: {game_length}, performing {train_steps} training steps")
        for _ in range(train_steps):
            mcts.agent.train_step()
        mcts.agent.save_checkpoint(f"checkpoint_{iter}.pth")


async def main_eval():
    server = GameServerGo()
    await server.wait()
    print("GameServer ready and client connected")

    plotter = Plotter()
    agent = AlphaZeroAgent(5, plotter, batch_size=128)
    agent.load_checkpoint("checkpoint_56.pth")
    mcts = MCTS(server, plotter, agent, search_iterations=1000)

    NUM_EPISODES = 100

    for _ in range(NUM_EPISODES):
        state = await server.reset_game("Netburners")
        server.go = Go_uf(5, state)

        is_white = False
        done = False
        mcts.agent.policy_net.eval()
        while not done:
            pi_mcts = mcts.search(state, is_white)
            print(pi_mcts)
            best_move = torch.argmax(pi_mcts).item()
            print(f"{best_move}, {pi_mcts[best_move]}")
            action = mcts.agent.decode_action(best_move)
            print(f"make move: {action}")

            # outcome is: 1 if black won, -1 is white won
            next_state, outcome, done = await server.make_move_eval(
                action, best_move, is_white
            )
            state = next_state

        assert outcome != 0, "outcome should not be 0 after a game ended"

        mcts.agent.plotter.update_wins_white(1 if outcome == -1 else -1)
        mcts.agent.plotter.update_wins_black(1 if outcome == 1 else -1)


# Run the main coroutine
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
