import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from agent_cnn_zero import AlphaZeroAgent
from gameserver_local import GameServerGo
from plotter import Plotter
from TreePlotter import TreePlot
import time


# returns score based on current node
# maybe should be based on root node (current player/who initiated the search)
# or based on who started (black)?
def get_score(server: GameServerGo, node: "Node"):
    score = server.get_score(node.state)
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


def get_history(node: "Node") -> list[list[str]]:
    def _get_history_helper(node: "Node", history: list[list[str]]) -> list[list[str]]:
        history.append(node.state)
        if node.parent:
            return _get_history_helper(node.parent, history)
        return history

    return _get_history_helper(node, [])


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
    space_available = any([True for row in node.state if "." in row])
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
        state: list[str],
        server: GameServerGo,
        is_white: bool,
        agent: AlphaZeroAgent,
        parent: Union["Node", None] = None,
        action: int | None = None,
        selected_policy: float = 0,
        visit_count: int = 0,
    ):
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
                h = get_history(self)
                self.valid_moves = self.server.request_valid_moves(
                    self.is_white, self.state, h
                )
            else:
                self.valid_moves = self.server.request_valid_moves(
                    self.is_white, self.state
                )
        return self.valid_moves

    def get_history(self):
        # TODO
        raise NotImplementedError("Not yet implemented")

    def next(self) -> "Node":
        best: tuple[Node | None, float] = (None, -999999999)
        for c in self.children:
            score = get_ucb_value(c, self.visit_cnt, c_puct=2.0)
            if score > best[1]:
                best = (c, score)

        assert best[0] is not None
        return best[0]

    def expand(self, q_values: torch.Tensor) -> None:
        valid_moves = np.array(self.get_valid_moves()).flatten()
        for action, q_value in enumerate(q_values):
            if not valid_moves[action]:
                continue
            if action != 25:
                next_state = self.server.get_state_after_move(
                    action, self.state, self.is_white
                )
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

    @torch.no_grad()
    def search(self, state: list[str], is_white: bool):
        root: Node = Node(state, self.server, is_white, self.agent)

        # valid_moves = await root.get_valid_moves()
        # history = get_history(root, [])
        # history.extend(await self.server.get_game_history())
        # policy, _ = self.agent.get_actions_eval(state, valid_moves, history, is_white)
        # policy1 = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        # policy1 = (1 - 0.25) * policy1 + 0.25 * np.random.dirichlet([0.3] * (25 + 1))

        # policy1 *= valid_moves
        # policy1 /= np.sum(policy1)

        # await root.expand(policy.squeeze(0))

        for iter in range(self.search_iterations):
            node = root
            # selection
            while node.is_fully_expanded():
                node = node.next()

            # expansion
            done, value = has_game_ended(self.server, node)
            # value *= -1
            if not done:
                valid_moves = node.get_valid_moves()
                history = get_history(node)
                history.extend(self.server.get_game_history())
                logits, value = self.agent.get_actions_eval(
                    node.state, valid_moves, history, node.is_white
                )

                # mask out invalid moves and normalize to 1
                # action_policy = torch.softmax(action_policy, dim=1).squeeze(0)
                # action_policy = action_policy * torch.tensor(
                #     valid_moves, device=action_policy.device

                # )
                # action_policy = action_policy / action_policy.sum()

                logits[~valid_moves] = -1e9
                policy = torch.softmax(logits, dim=0)
                policy = policy / policy.sum()
                node.expand(policy)

            # backpropagation
            node.backprop(value)
            # plot tree from root node for debugging
            # if iter == 999:
            #     TreePlot(root).create_tree()

        props = torch.zeros(26, device=self.agent.device)
        for c in root.children:
            props[c.action] = c.visit_cnt
        props /= props.sum()
        return props


async def main():
    server = GameServerGo()
    await server.wait()
    print("GameServer ready and client connected")

    mcts = MCTS(server, search_iterations=1000)

    NUM_EPISODES = 100

    for _ in range(NUM_EPISODES):
        state = await server.reset_game(True)

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
