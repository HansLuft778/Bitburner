import time
from collections import defaultdict
from typing import Any, Union

import numpy as np
import torch

from agent_cnn_zero import AlphaZeroAgent, State
from gameserver_local_uf import GameServerGo
from Go.Go_uf import Go_uf, UnionFind
from plotter import Plotter, ModelOverlay, cleanup_out_folder  # type: ignore
from TreePlotter import TreePlot  # pyright: ignore
from Buffer import BufferElement
from LookupTable import LookupTable

C_PUCT = 1.8
NUM_EPISODES = 1000
C_SCORE = 0.5


def get_puct_value(child: "Node", parent_visit_count: int, c_puct: float = 2.0) -> float:
    """
    child.win_sum is from the childs perspective, so from the parent's perspective we must negate it
      Q(s,a) = child.win_sum/child.visit_cnt
      U(s,a) = c_puct * P(s,a)* sqrt(parent.visit_cnt)/(1+child.visit_cnt)
    """
    if child.visit_cnt == 0:
        q_value = 0.0
    else:
        q_value = -(child.utility_sum / child.visit_cnt)
    u_value = c_puct * child.prior * parent_visit_count**0.5 / (1 + child.visit_cnt)
    return q_value + u_value


def get_num_forced_playouts(child: "Node", k: int = 2) -> int:
    assert child.parent is not None, "Child must have a parent to calculate number of forced playouts"
    return int((k * child.prior * child.parent.visit_cnt) ** 0.5)


def get_explore_selection_value_inverse(
    explore_selection_value: float, explore_scaling: float, prior: float, child_utility: float
) -> float:
    if prior < 0:
        return 0

    value_component = child_utility

    explore_component = explore_selection_value - value_component
    explore_component_scaling = explore_scaling * prior

    if explore_component < 1e-9 or explore_component_scaling <= 1e-9:
        return 0.0

    child_weight = (explore_component_scaling / explore_component) - 1.0

    return max(0.0, child_weight)


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
        prior: float = 0.0,
        visit_count: int = 0,
    ):
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

        self.win_utility: float | None = None
        self.score_utility: float | None = None

        self.mu_s: float | None = None
        self.sigma_s: float | None = None

        self.done: bool | None = None

        self.visit_cnt = visit_count
        self.utility_sum = 0.0

    def is_fully_expanded(self):
        # enough to check for childs > 0 since we expand to all possible states at once
        return len(self.children) > 0

    def get_valid_moves(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        if self.valid_moves is None:
            history_hashes = self.get_hash_history()
            self.valid_moves = self.server.request_valid_moves(self.is_white, self.uf, history_hashes)
        return self.valid_moves

    def get_predictions(self, table: LookupTable, x_0: float) -> tuple[torch.Tensor, float, float, float]:
        if self.policy is None:
            assert self.mu_s is None and self.sigma_s is None
            history_states = self.get_history_ref()
            history_states.extend(self.server.get_game_history())
            valid_moves = self.get_valid_moves()
            policy, value, mu, sigma = self.agent.predict_eval(self.uf, history_states, valid_moves, self.is_white)

            self.policy = policy
            self.win_utility = value * 2 - 1
            self.mu_s = mu
            self.sigma_s = sigma

            self.score_utility = table.get_expected_uscore(mu=mu, sigma=sigma, x_0=x_0).item()

        assert self.win_utility is not None and self.mu_s is not None and self.sigma_s is not None
        return self.policy, self.win_utility, self.mu_s, self.sigma_s

    def get_win_utility(self, score_white: float, score_black: float) -> float:
        outcome = 1.0 if score_black > score_white else -1.0
        return -outcome if self.is_white else outcome

    def get_score_utility(self, score_white: float, score_black: float, x_0: float, c_score: float) -> float:
        score_diff = (score_white - score_black) if self.is_white else (score_black - score_white)
        return c_score * ((2 / np.pi) * np.arctan((score_diff - x_0) / self.server.go.board_width))

    def get_utility(self, x_0: float) -> tuple[float, float]:
        score: dict[str, dict[str, float]] = self.server.get_score(self.uf)
        score_white = score["white"]["sum"]
        score_black = score["black"]["sum"]

        u_win = self.get_win_utility(score_white, score_black)
        u_score = self.get_score_utility(score_white, score_black, x_0, C_SCORE)

        return u_win, u_score

    def has_game_ended(self) -> bool:
        board_size = self.agent.board_width * self.agent.board_height
        # both pass back-to-back
        if self.action == board_size and self.parent is not None and self.parent.action == board_size:
            return True

        # technically, passing is always legal, so these two are not needed?
        # when the board is full, you can still pass

        # # board full (technically not needed?)
        # space_available = np.any(self.uf.state == 0)
        # if not space_available:
        #     return True

        # one pass, next player has no valid moves
        # if self.parent is not None and self.parent.action == board_size:
        #     if self.valid_moves is not None:
        #         has_valid = np.any(self.valid_moves)
        #     else:
        #         history_hashes = self.get_hash_history()
        #         has_valid = self.server.go.has_any_valid_moves(self.uf, self.is_white, history_hashes)
        #     if not has_valid:
        #         return True

        return False

    def get_history_ref(self) -> list[State]:
        history: list[State] = []
        if self.parent is None:
            return history

        current = self.parent
        while current:
            history.append(current.uf.state)
            if current.parent is not None:
                current = current.parent
            else:
                break
        return history

    def get_hash_history(self) -> list[np.uint64]:
        history: list[np.uint64] = []
        if self.parent is None:
            return history

        current = self.parent
        while current:
            history.append(current.uf.hash)
            if current.parent is not None:
                current = current.parent
            else:
                break
        return history

    def next(self) -> "Node":
        best: tuple[Node | None, float] = (None, -99999)
        for c in self.children:
            score = get_puct_value(c, self.visit_cnt, c_puct=C_PUCT)
            if score > best[1]:
                best = (c, score)

        assert best[0] is not None
        return best[0]

    def expand(self, policy: torch.Tensor) -> None:
        assert self.policy is not None, "Policy must be predicted before expansion"

        board_size = self.agent.board_width * self.agent.board_height
        policy_cpu: np.ndarray[Any, np.dtype[np.float32]] = policy.cpu().numpy()  # type: ignore
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
        self.utility_sum += score

        if self.parent:
            self.parent.backprop(-score)


class MCTS:
    def __init__(self, server: GameServerGo, agent: AlphaZeroAgent, search_iterations: int, table: LookupTable):
        self.search_iterations = search_iterations
        self.server = server
        self.table = table

        self.agent = agent
        self.timing_stats: defaultdict[str, float] = defaultdict(float)
        self.iterations_stats: defaultdict[str, int] = defaultdict(int)
        self.max_depth = 0

    @torch.no_grad()  # pyright: ignore
    def search(self, uf: UnionFind, is_white: bool, best_move: int, eval_mode: bool = False) -> torch.Tensor:
        search_init_start = time.time()
        if best_move == -1 or (
            best_move == self.agent.board_width * self.agent.board_width and len(self.root.children) == 0
        ):
            self.root = Node(uf, self.server, is_white, self.agent)
        else:
            child_idx = find_child_index(self.root, best_move)
            self.root = self.root.children[child_idx]
            self.root.parent = None

        self.timing_stats.clear()
        self.iterations_stats.clear()

        self.root.get_predictions(self.table, 0.0)
        x_0 = self.root.mu_s
        assert self.root.mu_s is not None and self.root.sigma_s is not None and x_0 is not None
        self.root.score_utility = self.table.get_expected_uscore(self.root.mu_s, self.root.sigma_s, x_0).item()

        self.timing_stats["search_init"] += time.time() - search_init_start
        search_start = time.time()
        for iter in range(self.search_iterations):  # type: ignore
            node = self.root

            # selection with forced playouts
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
                node.done = node.has_game_ended()
            assert node.done is not None

            if node.done:
                node.win_utility, node.score_utility = node.get_utility(x_0)
                assert node.win_utility is not None and node.score_utility is not None, "Values should not be None"
                self.timing_stats["end_check"] += time.time() - end_check_start
            else:
                self.timing_stats["end_check"] += time.time() - end_check_start
                inference_start = time.time()

                if node.policy is None:
                    # This sets leaf_node.win_utility and leaf_node.score_utility
                    final_probs, _, _, _ = node.get_predictions(self.table, x_0)

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

                policy = node.policy
                assert node.win_utility is not None and node.score_utility is not None, "Values should not be None"

                self.timing_stats["nn_inference"] += time.time() - inference_start
                self.iterations_stats["nn_inference"] += 1

                if iter == 0:
                    print(
                        f"Value estimate: {node.win_utility:.3f} (from {'white' if node.is_white else 'black'}'s perspective)"
                    )

                expand_start = time.time()
                node.expand(policy)
                self.timing_stats["expansion"] += time.time() - expand_start

            # backpropagation
            backprop_start = time.time()
            assert node.win_utility is not None and node.score_utility is not None
            node.backprop(node.win_utility + node.score_utility)
            self.timing_stats["backprop"] += time.time() - backprop_start

            # plot tree from root node for debugging
            # if iter == 999:
            #     TreePlot(self.root).create_tree()

        # calculate final policy distribution
        final_policy_start = time.time()

        # Handles case where the root is terminal / no expansion occurred
        if not self.root.children or len(self.root.children) == 0:
            print("WARNING: MCTS root has no children after search. Likely a terminal state. Returning uniform policy.")

            # return policy based purely on root's NN eval if available, else uniform
            if self.root.policy is not None:
                props = self.root.policy.clone()
            else:  # Need a fallback if NN wasn't even run
                props = torch.ones(
                    self.agent.board_height * self.agent.board_height + 1,
                    device=self.agent.device,
                    dtype=torch.float32,
                ) / (self.agent.board_height * self.agent.board_height + 1)

            self.timing_stats["final_policy"] += time.time() - final_policy_start
            total_time = time.time() - search_start
            self.timing_stats["total"] = total_time

            print("\nMCTS Timing Statistics:")
            print(f"Total search time: {total_time:.3f}s")
            for key, val in self.timing_stats.items():
                if key != "total":
                    percentage = (val / total_time) * 100
                    avg_time = val / float(self.iterations_stats.get(key, self.search_iterations))
                    print(f"{key}: {val:.3f}s ({percentage:.1f}%) - Avg: {avg_time*1000:.2f}ms")

            return props

        # policy target pruning
        c_star = max(self.root.children, key=lambda c: c.visit_cnt)
        assert c_star.parent is not None, "c_start must have a parent"
        c_star_puct = get_puct_value(c_star, c_star.parent.visit_cnt, c_puct=C_PUCT)
        explore_scaling = C_PUCT * self.root.visit_cnt**0.5
        pruned_visit_counts = {c.action: float(c.visit_cnt) for c in self.root.children}

        for c in self.root.children:
            action = c.action
            # use action to differentiate between c and c_star, since action is unique for every child
            if action is None or action == c_star.action:
                continue

            n_orig = float(c.visit_cnt)
            wins = c.utility_sum
            prior = c.prior

            # set visit count to 0 if c has no visits
            if n_orig <= 0:
                pruned_visit_counts[action] = 0.0
                continue

            child_q_value = wins / n_orig
            child_utility_from_parent = -child_q_value

            wanted_weight = get_explore_selection_value_inverse(
                c_star_puct, explore_scaling, prior, child_utility_from_parent
            )

            actual_weight = n_orig
            pruned_weight = min(wanted_weight, actual_weight)

            if pruned_weight < 1.0:
                print(f"Pruning child {action} from {actual_weight:.3f} to {pruned_weight:.3f}")
                pruned_weight = 0.0

            pruned_visit_counts[action] = pruned_weight

        props = torch.zeros(
            self.agent.board_height * self.agent.board_height + 1,
            device=self.agent.device,
            dtype=torch.float32,
        )
        total_pruned_weight = 0.0
        for action, weight in pruned_visit_counts.items():
            if action is not None:
                props[action] = float(weight)
                total_pruned_weight += weight

        if total_pruned_weight > 1e-9:
            props /= total_pruned_weight
        else:
            # If all visit counts are zero, set the policy to uniform distribution
            print("WARNING: All moves pruned, returning uniform policy")
            props = torch.ones_like(props) / (self.agent.board_height * self.agent.board_height + 1)

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
    table = LookupTable(board_size, C_SCORE)

    server = GameServerGo(board_size)
    await server.wait()
    print("GameServer ready and client connected")
    print("initializing MCTS...")

    plotter = Plotter(4, 3)
    cleanup_out_folder("D:/AProgramming/Bitburner/bb-external-editor/http_server/out")
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
    plotter.update_wins_black(0, draw=False)
    plotter.update_wins_white(0, draw=False)
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
    # agent.load_checkpoint("checkpoint_55.pth")
    mcts = MCTS(server, agent, search_iterations=1000, table=table)

    print("Done! Starting MCTS...")

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
            print(f"================================ {episode_length} ================================")
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

            valid_moves = server.go.get_valid_moves(uf, is_white, server.go.hash_history)
            buffer.append(BufferElement(uf, is_white, pi_mcts, game_history[: mcts.agent.history_length], valid_moves))

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

        mcts.agent.plotter.update_wins_white(1 if outcome == -1 else -1, draw=False)
        mcts.agent.plotter.update_wins_black(1 if outcome == 1 else -1, draw=False)
        mcts.agent.plotter.draw_and_flush()

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

        print("saving model overlay...")
        mo = ModelOverlay()
        len_buffer = len(buffer)
        buffer_indices = [0, 1, len_buffer // 2, len_buffer // 2 + 1, len_buffer - 2, len_buffer - 1]
        # buffer_indices = range(len_buffer)  # for testing
        for i in buffer_indices:
            be = buffer[i]
            state_tensor, state_vector = agent.preprocess_state(be.uf, be.history, be.valid_moves, be.is_white, "cpu")
            logits = agent.policy_net(state_tensor, state_vector)
            next_uf = buffer[i + 1].uf if i + 1 < len_buffer else None
            next_next_uf = buffer[i + 2].uf if i + 2 < len_buffer else None
            mo.heatmap(
                be.uf,
                next_uf,
                next_next_uf,
                logits,
                be.is_white,
                server,
                score,
                True,
                f"model_overlay_ep_{iter}_{i}.png",
            )
        print("done!")

        for be in buffer:
            # Flip if the outcome from neutrals perspective to players perspective
            z = outcome if not be.is_white else -outcome
            ownership_corrected = ownership_mask * (-1 if be.is_white else 1)
            score_corrected = score * (-1 if be.is_white else 1)
            mcts.agent.augment_state(be, z, ownership_corrected, score_corrected)

        if iter < 12:
            print("Skipping training")
            continue

        mcts.agent.policy_net.train()
        train_steps = 15
        print(f"Game length: {episode_length}, performing {train_steps} training steps")
        for _ in range(train_steps):
            mcts.agent.train_step()
        mcts.agent.save_checkpoint(f"checkpoint_{iter}.pth")


async def main_eval():
    board_size = 5
    table = LookupTable(board_size, C_SCORE)
    server = GameServerGo(board_size)
    await server.wait()
    print("GameServer ready and client connected")

    plotter = Plotter()
    agent = AlphaZeroAgent(board_size, plotter, checkpoint_dir="models")
    agent.load_checkpoint("checkpoint_129_katago_v1.pth")
    mcts = MCTS(server, agent, search_iterations=1000, table=table)

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
