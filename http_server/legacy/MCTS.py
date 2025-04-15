# raw mcts with no ml model

import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from legacy.agent_cnn import DQNAgentCNN
from legacy.gameserver import GameServer
from plotter import Plotter


async def get_score(server, state, is_white):
    score = await server.get_score(state)
    value = 1 if score["black"] > score["white"] else -1
    value *= -1 if is_white else 1
    return value


def is_duplicate_state(state: list[str], history: list[list[str]]):
    # b = set([str(string) for string in a])
    history_set = set([str(h) for h in history])
    assert len(history) == len(history_set)
    history_size = len(history_set)
    history_set.add(str(state))

    return len(history_set) == history_size


def get_history(node: "Node", history: list[list[str]]) -> list[list[str]]:
    history.append(node.state)
    if node.parent:
        return get_history(node.parent, history)
    return history


def decode_action(action_idx: int):
    """
    Convert the action index back to (x, y) or 'pass'.
    If action_idx == board_size, then 'pass'.
    """
    board_size = 25
    if action_idx == board_size:
        return "pass"
    else:
        x = action_idx // 5
        y = action_idx % 5
        return (x, y)


async def has_game_ended(
    server: GameServer,
    state: list[str],
    action: int,
    node: "Node",
    is_white: bool,
) -> tuple[bool, int]:

    # both pass back-to-back
    if node and node.parent and action == 25 and node.parent.action == 25:
        return (True, await get_score(server, state, is_white))

    # one pass, next player has no valid moves

    action_decoded = decode_action(action)
    if action_decoded != "pass":
        state_after_move = await server.get_state_after_move(
            action_decoded[0], action_decoded[1], state, is_white
        )
    else:
        state_after_move = state
    if node is not None:
        history = get_history(node, [])
        history.extend(await server.get_game_history())
    else:
        history = []
    move_creates_duplicate_state = is_duplicate_state(state_after_move, history)

    valid_moves = np.sum(
        (await server.request_valid_moves(is_white, state, history[0]))[:-1]
    )
    if (
        node
        and node.parent
        and node.parent.action == "pass"
        and (valid_moves == 0 or move_creates_duplicate_state)
    ):
        return (True, await get_score(server, state, is_white))

    # board full
    space_available = any([True for row in state if "." in row])
    if not space_available:
        return (True, await get_score(server, state, is_white))

    return (False, 0)


async def has_game_ended_sim(
    server: GameServer,
    state: list[str],
    action: int,
    prev_action: int | None,
    is_white: bool,
    history: list[list[str]],
) -> tuple[bool, int]:

    # both pass back-to-back
    if prev_action and action == 25 and prev_action == 25:
        return (True, await get_score(server, state, is_white))

    # one pass, next player has no valid moves
    action_decoded = decode_action(action)
    if action_decoded != "pass":
        state_after_move = await server.get_state_after_move(
            action_decoded[0], action_decoded[1], state, is_white
        )
    else:
        state_after_move = state

    move_creates_duplicate_state = is_duplicate_state(state_after_move, history)

    valid_moves = np.sum(
        (await server.request_valid_moves(is_white, state, history[0]))[:-1]
    )
    if (
        prev_action
        and prev_action == 26
        and (valid_moves == 0 or move_creates_duplicate_state)
    ):
        return (True, await get_score(server, state, is_white))

    # board full
    space_available = any([True for row in state if "." in row])
    if not space_available:
        return (True, await get_score(server, state, is_white))

    return (False, 0)


class Node:
    def __init__(
        self,
        state: list[str],
        server: GameServer,
        is_white: bool,
        agent: DQNAgentCNN,
        parent: Union["Node", None] = None,
        action: int | None = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.server = server
        self.is_white = is_white
        self.agent = agent

        self.children: list[Node] = []

        self.visit_cnt = 0
        self.win_cnt = 0

    async def is_leaf(self) -> bool:
        if len(self.children) == 0:
            return True

        if self.action and self.parent:
            done, _ = await has_game_ended(
                self.server, self.state, self.action, self.parent, self.is_white
            )
            return done

        return False

    async def is_fully_expanded(self):
        if self.parent:
            valid = await self.server.request_valid_moves(
                self.is_white, self.state, self.parent.state
            )
            return np.sum(valid) == self.children and self.children > 0
        else:
            valid = await self.server.request_valid_moves(self.is_white)
            return sum(valid) == len(self.children) and self.children > 0

    def get_ucb_value(
        self,
        num_wins: int,
        num_visited: int,
        parent_visit_cnt: int,
        exploration=math.sqrt(2),
    ) -> float:
        # negative since the child is the opponent player -> node wants child to lose
        return (num_wins / num_visited) + exploration * math.sqrt(
            math.log(parent_visit_cnt) / num_visited
        )

    def next(self) -> Union["Node", None]:
        best: tuple[Node | None, float] = (None, -999999999)
        for c in self.children:
            score = self.get_ucb_value(c.win_cnt, c.visit_cnt, self.visit_cnt)
            if score > best[1]:
                best = (c, score)

        return best[0]

    async def expand(self, state, server: GameServer) -> "Node":
        history = get_history(self, [])
        history.extend(await self.server.get_game_history())
        action = self.agent.select_action_eval(
            state,
            await self.server.request_valid_moves(self.is_white, state, history[0]),
            # [self.parent.state, self.parent.parent.state],
            history,
            self.is_white,
        )
        action_decoded = self.agent.decode_action(action)
        if action_decoded != "pass":
            next_state = await self.server.get_state_after_move(
                action_decoded[0], action_decoded[1], state, self.is_white
            )
        else:
            next_state = state

        child = Node(next_state, server, not self.is_white, self.agent, self, action)
        self.children.append(child)
        return child

    async def simulate(self, root_is_white: bool) -> int:
        assert self.action
        done, score = await has_game_ended(
            self.server,
            self.state,
            self.action,
            self,
            self.is_white,
        )

        if done:
            return score

        sim_state = self.state.copy()
        sim_is_white = self.is_white
        sim_history: list[list[str]] = get_history(self, [])
        sim_history.extend(await self.server.get_game_history())
        assert self.parent
        sim_prev_action = self.parent.action
        while True:
            valid = await self.server.request_valid_moves(
                sim_is_white, sim_state, sim_history[1]
            )
            action = self.agent.select_action_eval(
                self.state, valid, sim_history, sim_is_white
            )
            action_decoded = self.agent.decode_action(action)

            if action_decoded != "pass":
                sim_state = await self.server.get_state_after_move(
                    action_decoded[0], action_decoded[1], sim_state, sim_is_white
                )
                sim_history.insert(0, sim_state)

            done, score = await has_game_ended_sim(
                self.server,
                sim_state,
                action,
                sim_prev_action,
                root_is_white,
                sim_history,
            )

            if done:
                # normalize score such that it is from roots perspective
                return score
            sim_is_white = not sim_is_white
            sim_prev_action = action

    def backprop(self, score: int):
        self.visit_cnt += 1
        self.win_cnt += score

        if self.parent:
            self.parent.backprop(-score)


class MCTS:
    def __init__(self, search_iterations):
        self.search_iterations = search_iterations
        self.gameserver: GameServer = GameServer()

        self.plotter = Plotter()
        self.agent = DQNAgentCNN(5, 5, self.plotter)

    @classmethod
    async def create(cls, search_iterations=50):
        self = cls(search_iterations)
        await self.gameserver.wait()  # Wait for a client to connect
        print("GameServer ready and client connected")
        return self

    async def search(self, state: list[str], is_white: bool):
        root: Node = Node(state, self.gameserver, is_white, self.agent)

        for _ in range(self.search_iterations):
            node = root
            # selection
            while await node.is_fully_expanded():
                ret = node.next()
                if ret is not None:
                    node = ret

            # expansion
            ## get q-values for board possition
            if node.action and node.parent:
                done, score = await has_game_ended(
                    self.gameserver,
                    node.state,
                    node.action,
                    node.parent,
                    node.is_white,
                )
            else:
                done = False
            if not done:
                node = await node.expand(node.state, self.gameserver)
                # simulation
                score = await node.simulate(root.is_white)

            # backpropagation
            node.backprop(score)

        props = torch.zeros(26)
        for c in root.children:
            props[c.action] = c.visit_cnt
        props = F.softmax(props)
        return props

    def get_updated_ucb_value(
        self,
        num_wins: int,
        num_visited: int,
        num_simulations: int,
        propability: float,
        exploration=math.sqrt(2),
    ) -> float:
        return (num_wins / num_visited) + propability * exploration * math.sqrt(
            math.log(num_simulations)
        ) / (1 + num_visited)


async def main():
    mcts = await MCTS.create(search_iterations=10)
    state = (await mcts.gameserver.get_state())["board"]
    print(await mcts.search(state, False))


# Run the main coroutine
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
