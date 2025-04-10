import time
from collections import defaultdict
from typing import Any, Union

import numpy as np
import torch

from agent_cnn_zero import AlphaZeroAgent
from gameserver_local_uf import GameServerGo
from Go.Go_uf import Go_uf, UnionFind
from plotter import Plotter, ModelOverlay  # type: ignore
from TreePlotter import TreePlot  # pyright: ignore
from Buffer import BufferElement

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
    u_value = c_puct * child.prior * parent_visit_count**0.5 / (1 + child.visit_cnt)
    return q_value + u_value


def get_num_forced_playouts(child: "Node", k: int = 2):
    assert child.parent is not None, "Child must have a parent to calculate number of forced playouts"
    return (k * child.prior * child.parent.visit_cnt) ** 0.5


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
        prior: float = 0,
        visit_count: int = 0,
    ):
        # self.state = state
        self.uf: UnionFind = uf
        self.parent = parent
        self.action = action
        self.server = server
        self.is_white = is_white
        self.agent = agent
        self.prior = prior
        self.valid_moves: np.ndarray[Any, np.dtype[np.bool_]] | None = None
        self.depth: int = 0 if parent is None else parent.depth + 1

        self.children: list[Node] = []
        self.policy: torch.Tensor | None = None
        self.value: float | None = None

        self.done: bool | None = None

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
        while current:
            history.append(current.uf.state)
            if current.parent is not None:
                current = current.parent
            else:
                break
        return history[::-1]  # Reverse for chronological order

    def get_hash_history(self) -> list[np.uint64]:
        history: list[np.uint64] = []
        current = self
        while current:
            history.append(current.uf.hash)
            if current.parent is not None:
                current = current.parent
            else:
                break
        return history[::-1]  # Reverse for chronological order

    def next(self) -> "Node":
        best: tuple[Node | None, float] = (None, -99999)
        for c in self.children:
            score = get_ucb_value(c, self.visit_cnt, c_puct=1.8)
            if score > best[1]:
                best = (c, score)

        assert best[0] is not None
        return best[0]

    def expand(self, q_values: torch.Tensor) -> None:
        board_size = self.agent.board_width * self.agent.board_height
        policy_cpu: np.ndarray[Any, np.dtype[np.float32]] = q_values.cpu().numpy()  # type: ignore
        for action in range(board_size + 1):
            if action == board_size:
                next_uf = self.uf.copy()
                # Penalize passing if board is mostly empty
                # empty_cells = np.sum(self.uf.state == 0)
                # empty_percentage = empty_cells / board_size
                # if empty_percentage > 0.5:
                #     policy_cpu[action] *= (1.0 - empty_percentage) * 0.5
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
                policy_cpu[action],
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
        agent: AlphaZeroAgent,
        search_iterations: int,
    ):
        self.search_iterations = search_iterations
        self.server = server

        self.agent = agent
        self.timing_stats: defaultdict[str, float] = defaultdict(float)
        self.iterations_stats: defaultdict[str, int] = defaultdict(int)
        self.max_depth = 0

    @torch.no_grad()  # pyright: ignore
    def search(self, uf: UnionFind, is_white: bool, best_move: int, eval_mode: bool = False) -> torch.Tensor:
        if best_move == -1:
            self.root = Node(uf, self.server, is_white, self.agent)
        else:
            child_idx = find_child_index(self.root, best_move)
            self.root = self.root.children[child_idx]
            self.root.parent = None

        self.timing_stats.clear()
        self.iterations_stats.clear()
        search_start = time.time()

        for iter in range(self.search_iterations):  # type: ignore
            node = self.root
            # selection
            select_start = time.time()
            while node.is_fully_expanded():
                next_node = None
                if node.parent is None:
                    for c in node.children:
                        if c.visit_cnt < get_num_forced_playouts(c) and (
                            next_node is None or c.prior > next_node.prior
                        ):
                            next_node = c
                if next_node is None:
                    node = node.next()
                    self.max_depth = max(self.max_depth, node.depth)
                else:
                    node = next_node

            self.timing_stats["selection"] += time.time() - select_start
            self.iterations_stats["selection"] += 1

            # expansion
            end_check_start = time.time()
            if node.done is None:
                node.done, node.value = node.has_game_ended()
            assert node.value is not None, "Value should not be None"
            value = node.value

            self.timing_stats["end_check"] += time.time() - end_check_start

            if not node.done:
                prep_start = time.time()
                valid_moves = node.get_valid_moves()
                history = node.get_history_ref()
                history.extend(self.server.get_game_history())
                self.timing_stats["move_prep"] += time.time() - prep_start

                inference_start = time.time()
                if node.policy is None:
                    raw_logits, raw_value = self.agent.get_actions_eval(node.uf, history, node.is_white)
                    valid_mask = torch.tensor(valid_moves, device=self.agent.device, dtype=torch.bool)
                    raw_logits[~valid_mask] = -1e9
                    final_probs = torch.softmax(raw_logits, dim=0)

                    # Apply dirichlet noise
                    if node.parent is None and not eval_mode:
                        alpha = 0.2
                        dir_noise = np.random.dirichlet([alpha] * len(final_probs))
                        dir_noise_tensor = torch.tensor(dir_noise, device=final_probs.device, dtype=final_probs.dtype)
                        epsilon = 0.25
                        final_probs = (1 - epsilon) * final_probs + epsilon * dir_noise_tensor

                        # Renormalize
                        final_probs = final_probs / torch.sum(final_probs)
                    node.policy = final_probs
                    node.value = raw_value

                policy = node.policy
                assert node.value is not None, "Value should not be None"
                value = node.value

                self.timing_stats["nn_inference"] += time.time() - inference_start
                self.iterations_stats["nn_inference"] += 1

                if iter == 0:
                    print(f"Value estimate: {value:.3f} (from {'white' if node.is_white else 'black'}'s perspective)")

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
        for c in self.root.children:
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


def find_child_index(root_node: Node, chosen_action: int) -> int:
    for i, child in enumerate(root_node.children):
        if child.action == chosen_action:
            return i
    raise RuntimeError(f"No child found with action={chosen_action}")


async def main() -> None:
    # torch.manual_seed(0)  # pyright: ignore
    # np.random.seed(0)

    board_size = 5
    server = GameServerGo(board_size)
    await server.wait()
    print("GameServer ready and client connected")
    plotter = Plotter(4, 3)
    plotter.add_plot(  # type: ignore
        "cumulative_reward_black",
        plotter.axes[0, 0],  # type: ignore
        "Cumulative Wins Over Time",
        "Updates",
        "Cumulative Wins",
        label="Black",
    )
    plotter.add_plot(  # type: ignore
        "cumulative_reward_white",
        plotter.axes[0, 0],  # type: ignore
        "Cumulative Wins Over Time",
        "Updates",
        "Cumulative Wins",
        label="White",
    )
    plotter.add_plot("loss", plotter.axes[0, 1], "Training Loss Over Time", "Updates", "Loss")  # type: ignore
    plotter.add_plot("depth", plotter.axes[0, 2], "MCTS Depth", "Iteration", "Depth", label="tree depth")  # type: ignore
    plotter.add_plot("episode_length", plotter.axes[0, 2], "Episode Length", "Iteration", "Length", label="episode length")  # type: ignore

    plotter.add_plot("policy_loss_own", plotter.axes[1, 0], "Own Policy Loss Over Time", "Updates", "Policy Loss")  # type: ignore
    plotter.add_plot("policy_loss_opp", plotter.axes[1, 1], "Opponent Policy Loss Over Time", "Updates", "Policy Loss")  # type: ignore
    plotter.add_plot("value_loss", plotter.axes[1, 2], "Value Loss Over Time", "Updates", "Value Loss")  # type: ignore

    plotter.add_plot("ownership_loss", plotter.axes[2, 1], "Ownership Loss Over Time", "Updates", "Ownership Loss")  # type: ignore
    plotter.add_plot("score_pdf_loss", plotter.axes[2, 2], "Score PDF Loss Over Time", "Updates", "Score PDF Loss")  # type: ignore

    plotter.add_plot("score_cdf_loss", plotter.axes[3, 0], "Score CDF Loss Over Time", "Updates", "Score CDF Loss")  # type: ignore
    plotter.add_plot("score_mean_loss", plotter.axes[3, 1], "Score Mean Loss Over Time", "Updates", "Score Mean Loss")  # type: ignore
    plotter.add_plot("score_std_loss", plotter.axes[3, 2], "Score Std Dev Loss Over Time", "Updates", "Score Std Dev Loss")  # type: ignore

    agent = AlphaZeroAgent(board_size, plotter)
    # agent.load_checkpoint("checkpoint_126.pth")
    mcts = MCTS(server, agent, search_iterations=1000)

    NUM_EPISODES = 1000
    outcome = 0
    for iter in range(NUM_EPISODES):
        is_white = False
        state, komi = await server.reset_game("No AI", is_white)
        uf = UnionFind.get_uf_from_state(state, server.go.zobrist)
        server.go = Go_uf(board_size, state, komi)

        buffer: list[BufferElement] = []
        done = False
        game_history = [state]
        mcts.agent.policy_net.eval()
        episode_length = 0
        previous_move = -1
        start_time = time.time()
        while not done:
            before = time.time()
            pi_mcts = mcts.search(server.go.uf, is_white, previous_move)
            after = time.time()
            print(f"TOOK: {after-before}s")
            print(pi_mcts)
            # best_move = int(torch.argmax(pi_mcts).item())
            best_move = choose_action(pi_mcts, episode_length)
            print(f"{best_move}, {pi_mcts[best_move]}")
            action = mcts.agent.decode_action(best_move)
            print(f"make move: {action}")

            # add move response to buffer
            if len(buffer) > 0:
                buffer[-1].pi_mcts_response = pi_mcts

            buffer.append(BufferElement(uf, is_white, pi_mcts, game_history[: mcts.agent.num_past_steps]))

            # outcome is: 1 if black won, -1 is white won
            next_uf, outcome, done = await server.make_move(action, best_move, is_white)
            game_history.insert(0, next_uf.state)

            is_white = not is_white
            uf = next_uf
            previous_move = best_move
            episode_length += 1

        # last move has no response, so set it to zero
        buffer[-1].pi_mcts_response = torch.zeros(mcts.agent.board_height * mcts.agent.board_height + 1, device=mcts.agent.device)  # type: ignore

        plotter.update_stat("depth", mcts.max_depth)  # type: ignore
        plotter.update_stat("episode_length", episode_length)  # type: ignore
        mcts.max_depth = 0
        print(f"Episode length: {episode_length}, took {time.time()-start_time:.3f}s")
        print("================================================================================")

        assert outcome != 0, "outcome should not be 0 after a game ended"

        mcts.agent.plotter.update_wins_white(1 if outcome == -1 else -1)
        mcts.agent.plotter.update_wins_black(1 if outcome == 1 else -1)

        mcts.agent.policy_net.train()

        ownership_mask = np.zeros((board_size, board_size), dtype=np.int8)
        black_territory = np.zeros((board_size, board_size), dtype=np.int8)
        black_stones = np.zeros((board_size, board_size), dtype=np.int8)
        white_territory = np.zeros((board_size, board_size), dtype=np.int8)
        white_stones = np.zeros((board_size, board_size), dtype=np.int8)

        visited: np.ndarray[Any, np.dtype[np.bool_]] = np.zeros((board_size, board_size), dtype=np.bool_)
        for x in range(board_size):
            for y in range(board_size):
                if server.go.uf.state[x, y] == 0 and not visited[x, y]:
                    color, territory = server.go.flood_fill_territory(server.go.uf.state, x, y, visited)
                    if color is not None:
                        if color == 1:  # black
                            ownership_mask[territory] = 1
                            black_territory[territory] = 1
                        elif color == 2:  # white
                            ownership_mask[territory] = -1
                            white_territory[territory] = 1
                elif server.go.uf.state[x, y] == 1:
                    black_stones[x, y] = 1
                    ownership_mask[x, y] = 1
                    visited[x, y] = True
                elif server.go.uf.state[x, y] == 2:
                    white_stones[x, y] = 1
                    ownership_mask[x, y] = -1
                    visited[x, y] = True

        black_score = np.count_nonzero(black_stones) + np.count_nonzero(black_territory)
        white_score = np.count_nonzero(white_stones) + np.count_nonzero(white_territory) + komi
        score = black_score - white_score  # black leads with

        mo = ModelOverlay()
        for i in [0, 1, len(buffer) // 2, len(buffer) // 2 + 1, -1, -2]:
            be = buffer[i]
            state_tensor = agent.preprocess_state(be.uf, be.history, be.is_white)
            out = agent.policy_net(state_tensor)
            mo.heatmap(be.uf, out, be.is_white, server, True, f"model_overlay_ep_{iter}_{i}.png")

        for be in buffer:
            # Flip if the outcome from neutrals perspective to players perspective
            z = outcome if not be.is_white else -outcome
            ownership_corrected = ownership_mask * (-1 if be.is_white else 1)
            score_corrected = score * (-1 if be.is_white else 1)
            mcts.agent.augment_state(be, z, ownership_corrected, score_corrected)

        if iter < 12:
            print("Skipping training")
            continue

        train_steps = 15
        print(f"Game length: {episode_length}, performing {train_steps} training steps")
        for _ in range(train_steps):
            mcts.agent.train_step()
        mcts.agent.save_checkpoint(f"checkpoint_{iter}.pth")


async def main_eval():

    board_size = 5
    server = GameServerGo(board_size)
    await server.wait()
    print("GameServer ready and client connected")

    plotter = Plotter()
    agent = AlphaZeroAgent(board_size, plotter, checkpoint_dir="models")
    agent.load_checkpoint("checkpoint_129_katago_v1.pth")
    mcts = MCTS(server, agent, search_iterations=1000)

    plotter.add_plot(  # type: ignore
        "cumulative_reward_black",
        plotter.axes[0, 0],  # type: ignore
        "Cumulative Wins Over Time",
        "Updates",
        "Cumulative Wins",
        label="Black",
    )
    plotter.add_plot(  # type: ignore
        "cumulative_reward_white",
        plotter.axes[0, 0],  # type: ignore
        "Cumulative Wins Over Time",
        "Updates",
        "Cumulative Wins",
        label="White",
    )

    NUM_EPISODES = 100
    outcome = 0
    for _ in range(NUM_EPISODES):
        state, komi = await server.reset_game("Slum Snakes")
        server.go = Go_uf(board_size, state, komi)

        is_white = False
        done = False
        mcts.agent.policy_net.eval()
        while not done:
            pi_mcts = mcts.search(server.go.uf, is_white, -1, eval_mode=True)
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
