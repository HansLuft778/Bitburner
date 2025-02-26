import math
import time
from collections import defaultdict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from agent_cnn_zero import AlphaZeroAgent
from gameserver_local import GameServerGo
from Go.Go import Go
from plotter import Plotter
from TreePlotter import TreePlot


# returns score based on current node
# maybe should be based on root node (current player/who initiated the search)
# or based on who started (black)?
def get_score(server: GameServerGo, node: "Node"):
    score = server.get_score()
    # 1 means "current node is winning," -1 means "current node is losing"
    score_white = score["white"]["sum"]
    score_black = score["black"]["sum"]
    diff = score_white - score_black

    # normalize score, so
    max_score = len(node.state) ** 2
    normalized = diff / max_score

    if score_white > score_black:
        return normalized if node.is_white else -normalized
    else:
        return -normalized if node.is_white else normalized


def has_game_ended(server: GameServerGo, node: "Node") -> tuple[bool, int]:
    # both pass back-to-back
    # if (
    #     node.action
    #     and node.parent
    #     and node.parent.action
    #     and node.action == 25
    #     and node.parent.action == 25
    # ):
    #     return (True, await get_score(server, node))

    # one pass, next player has no valid moves
    if node.parent:
        valid_moves = np.sum(node.get_valid_moves())
        if node.parent.action == 25 and valid_moves == 0:
            print(
                f"node.parent.action: {node.parent.action}, valid_moves: {valid_moves}"
            )
            if node.parent.action == 25 and valid_moves == 0:
                return (True, get_score(server, node))

    # # board full
    space_available = np.any(node.state == 3)
    if not space_available:
        return (True, get_score(server, node))

    return (False, 0)


def get_ucb_value(child: "Node", parent_visit_count: int, c_puct=2.0) -> float:
    """
    child.win_sum is from the parent's perspective? No, by flipping sign in backprop,
    child.win_sum is actually from the parent's perspective if done correctly.
    But let's do the standard:
      Q(s,a) = child.win_sum/child.visit_cnt   (already parent's perspective)
      U(s,a) = c_puct * P(s,a)* sqrt(parent.visit_cnt)/(1+child.visit_cnt)
    """
    if child.visit_cnt == 0:
        q_value = 0.0
    else:
        q_value = child.win_sum / child.visit_cnt
        # q_value = 1 - ((child.win_sum / child.visit_cnt) + 1) / 2
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
        server: GameServerGo,
        is_white: bool,
        agent: AlphaZeroAgent,
        parent: Union["Node", None] = None,
        action: int | None = None,
        selected_policy: float = 0,
        visit_count: int = 0,
    ):
        assert state.shape == (5, 5), f"Array must be 5x5: {state}"
        assert np.all(
            np.isin(state, [0, 1, 2, 3])
        ), f"Array must only contain values 0, 1, 2, or 3: {state}"

        self.state = state
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
                    self.is_white, self.state, history
                )
            else:
                self.valid_moves = self.server.request_valid_moves(
                    self.is_white, self.state
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
            score = get_ucb_value(c, self.visit_cnt, c_puct=2.0)
            if score > best[1]:
                best = (c, score)

        assert best[0] is not None
        return best[0]

    def expand(self, q_values: torch.Tensor) -> None:
        valid_moves = self.get_valid_moves().flatten()
        for action, q_value in enumerate(q_values):
            if not valid_moves[action]:
                continue
            if action != 25:
                print(
                    f"self.server.get_state_after_move({action}, {self.state}, {self.is_white})"
                )
                next_state = self.server.get_state_after_move(
                    action, self.state, self.is_white, self.get_history()
                )
                print(f"next_state = {next_state}")
                assert next_state.shape == (
                    5,
                    5,
                ), f"Array must be 5x5: action:{action} state: {next_state}"
                assert np.all(
                    np.isin(next_state, [0, 1, 2, 3])
                ), f"Array must only contain values 0, 1, 2, or 3: {next_state}"
            else:
                next_state = self.state

            child = Node(
                next_state,
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
    def __init__(self, server: GameServerGo, search_iterations: int):
        self.search_iterations = search_iterations
        self.server = server

        self.plotter = Plotter()
        self.agent = AlphaZeroAgent(5, 5, self.plotter)
        self.timing_stats = defaultdict(float)
        self.iterations_stats = defaultdict(int)

    @torch.no_grad()
    def search(self, state: np.ndarray, is_white: bool):

        assert state.shape == (5, 5), f"Array must be 5x5: {state}"
        assert np.all(
            np.isin(state, [0, 1, 2, 3])
        ), f"Array must only contain values 0, 1, 2, or 3: {state}"

        root: Node = Node(state, self.server, is_white, self.agent)

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
            # value *= -1
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
                self.timing_stats["nn_inference"] += time.time() - inference_start
                self.iterations_stats["nn_inference"] += 1

                policy_start = time.time()
                logits[~valid_moves] = -1e9
                policy = torch.softmax(logits, dim=0)
                policy = policy / policy.sum()
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
        props = torch.zeros(26, device=self.agent.device)
        for c in root.children:
            props[c.action] = c.visit_cnt
        props /= props.sum()
        self.timing_stats["final_policy"] += time.time() - final_policy_start
        total_time = time.time() - search_start
        self.timing_stats["total"] = total_time

        # Print timing statistics
        print("\nMCTS Timing Statistics:")
        print(f"Total search time: {total_time:.3f}s")
        for key, value in self.timing_stats.items():
            if key != "total":
                percentage = (value / total_time) * 100
                avg_time = value / self.iterations_stats.get(
                    key, self.search_iterations
                )
                print(
                    f"{key}: {value:.3f}s ({percentage:.1f}%) - Avg: {avg_time*1000:.2f}ms"
                )

        return props


async def main():
    server = GameServerGo()
    await server.wait()
    print("GameServer ready and client connected")

    mcts = MCTS(server, search_iterations=100)

    NUM_EPISODES = 100

    for _ in range(NUM_EPISODES):
        state = await server.reset_game(True)
        server.go = Go(5, 5, state)

        buffer = []
        is_white = False
        done = False
        game_history = [state]
        mcts.agent.policy_net.eval()
        while not done:
            before = time.time()
            pi_mcts = mcts.search(state, is_white)
            after = time.time()
            print(f"TOOK: {after-before}s")
            print(pi_mcts)
            best_move = torch.argmax(pi_mcts).item()
            print(f"{best_move}, {pi_mcts[best_move]}")
            action = mcts.agent.decode_action(best_move)
            print(f"make move: {action}")

            buffer.append(
                (
                    state,
                    is_white,
                    pi_mcts,
                    game_history[: mcts.agent.num_past_steps],
                )
            )

            # outcome is from current players/root perspective
            # above maybe wrong
            # outcome is: 1 if black won, -1 is white won (TODO: change this in client)
            next_state, outcome, done = await server.make_move(
                action, best_move, is_white
            )
            game_history.insert(0, next_state)

            is_white = not is_white
            state = next_state

        # blacks move   => z = -1
        # whites move   => z = 1
        # white won -> final = -1

        # blacks move   => z = 1
        # whites move   => z = -1
        # black won -> final = 1

        # white move    => z = 1
        # black move    => z = -1
        # white won -> final = -1

        # white move    => z = -1
        # black move    => z = 1
        # black won -> final = 1

        assert outcome != 0, "outcome should not be 0 after a game ended"
        final_outcome = outcome

        mcts.agent.plotter.update_wins_white(1 if final_outcome == 1 else -1)
        mcts.agent.plotter.update_wins_black(1 if final_outcome == -1 else -1)

        mcts.agent.policy_net.train()
        for state, was_white, pi, history in buffer:
            # Flip if the buffer entry belongs to the opposite color
            # what is opposite color?:
            #  - opposite of last player (player who won)
            #  - opposite of player who starts (always black) <- i chose this

            z = final_outcome if not was_white else -final_outcome
            mcts.agent.train_buffer.push(state, pi, z, history, was_white)

        for _ in range(20):
            mcts.agent.train_step()


# Run the main coroutine
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


"""
using string list:
selection: 0.001s (0.1%) - Avg: 0.01ms 
end_check: 0.650s (36.8%) - Avg: 6.50ms 
move_prep: 0.012s (0.7%) - Avg: 0.12ms 
nn_inference: 0.467s (26.4%) - Avg: 4.67ms 
policy_compute: 0.020s (1.1%) - Avg: 0.20ms 
expansion: 0.615s (34.8%) - Avg: 6.15ms 
backprop: 0.000s (0.0%) - Avg: 0.00ms 
final_policy: 0.001s (0.1%) - Avg: 0.01ms 
TOOK: 1.766082763671875s

numppy array:
selection: 0.013s (0.1%) - Avg: 0.01ms
end_check: 7.162s (42.3%) - Avg: 7.16ms
move_prep: 0.009s (0.1%) - Avg: 0.01ms
nn_inference: 3.260s (19.2%) - Avg: 3.26ms
policy_compute: 0.150s (0.9%) - Avg: 0.15ms
expansion: 6.351s (37.5%) - Avg: 6.35ms
backprop: 0.001s (0.0%) - Avg: 0.00ms
final_policy: 0.001s (0.0%) - Avg: 0.00ms
TOOK: 16.951805353164673s

"""
