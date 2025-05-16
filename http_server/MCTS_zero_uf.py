import time
from collections import defaultdict
from typing import Any, Union
import math

import numpy as np
from numpy.typing import NDArray
import torch

from agent_cnn_zero import AlphaZeroAgent
from gameserver_local_uf import GameServerGo
from Go.Go_uf import Go_uf, UnionFind, MoveResult
from Go.Go_state_generator import GoStateGenerator
from plotter import Plotter, ModelOverlay, GameStatePlotter, cleanup_out_folder, TensorBoardPlotter  # type: ignore
from TreePlotter import TreePlot  # pyright: ignore
from Buffer import BufferElement
from LookupTable import LookupTable
from go_types import State
import matplotlib.pyplot as plt

C_PUCT = 1.1
NUM_EPISODES = 1000
C_SCORE = 0.5

INITIAL_TEMPERATURE = 1.0
FINAL_TEMPERATURE = 0.1
TEMPERATURE_DECAY_MOVES = 14

FORCED_PLAYOUTS_K = 2

USE_BITBURNER = False

MCTS_BATCH_SIZE = 8
MCTS_VIRTUAL_LOSS_VALUE = 1


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
        self.action = action  # Action that led to this node
        self.server = server
        self.is_white = is_white  # Player *to move* from this node
        self.agent = agent
        self.prior = prior
        self._valid_moves: np.ndarray[Any, np.dtype[np.bool_]] | None = None
        self.depth: int = 0 if parent is None else parent.depth + 1

        # self.children: list[Node] = []
        self.children: dict[int, Node] = {}
        self.policy: torch.Tensor | None = None

        self.win_utility: float | None = None
        self.score_utility: float | None = None

        self.mu_s: float | None = None
        self.sigma_s: float | None = None

        self.done: bool | None = None

        self.visit_cnt = visit_count
        self.utility_sum = 0.0

        self.queued_for_inference = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.uf.hash == other.uf.hash

    def is_fully_expanded(self):
        # node is fully expanded, if every valid move has a child
        valid = self.get_valid_moves()
        num_valid = np.count_nonzero(valid)

        return num_valid == len(self.children)

    def get_valid_moves(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """returns the valid moves including pass (always legal) in shape `(board_size ** 2 + 1,)`
        Returns:
            np.ndarray[Any, np.dtype[np.bool_]]: the valid moves
        """
        if self._valid_moves is None:
            history_hashes = self.get_hash_history()
            self._valid_moves = self.server.request_valid_moves(self.is_white, self.uf, history_hashes)
        return self._valid_moves

    def get_predictions(self, table: LookupTable, x_0: float) -> tuple[torch.Tensor, float, float, float]:
        if self.policy is None:
            assert self.mu_s is None and self.sigma_s is None
            history_states = self.get_history_ref()
            history_states.extend(self.server.get_game_history())
            valid_moves = self.get_valid_moves()
            policy, value, mu, sigma = self.agent.predict_eval(
                self.uf, history_states, valid_moves, self.is_white, self.server.go.white_starts
            )

            self.policy = policy.cpu()
            self.win_utility = value  # * 2.0 - 1.0
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
        #     self.done = True
        #     return True

        # one pass, next player has no valid moves
        # if self.parent is not None and self.parent.action == board_size:
        #     if self._valid_moves is not None:
        #         has_valid = True#np.any(self._valid_moves)
        #     else:
        #         # history_hashes = self.get_hash_history()
        #         has_valid = True #self.server.go.has_any_valid_moves(self.uf, self.is_white, history_hashes)
        #     if not has_valid:
        #         self.done = True
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

    def get_complete_history_tensor(self, limit: int = -1) -> NDArray[np.int8]:
        go_history = self.server.go.get_history()
        go_history_length = len(go_history)

        history_length = limit if limit != -1 else self.depth + go_history_length
        history = np.zeros((history_length, self.agent.board_width, self.agent.board_width), dtype=np.int8)
        if self.parent is None:
            return history

        # collect MCTS history
        current = self.parent
        depth = 0
        while current:
            if depth >= history_length:
                return history

            history[depth] = current.uf.state
            if current.parent is not None:
                current = current.parent
            else:
                break
            depth += 1

        # collect go history
        for h in go_history:
            if depth >= history_length:
                return history
            history[depth] = h
            depth += 1

        return history

    def next(self) -> int:
        assert self.policy is not None
        valid_moves = self.get_valid_moves()
        only_valid = np.where(valid_moves == True)[0]

        parent_visit_cnt_sqrt = self.visit_cnt**0.5  # faster than sqrt

        best: tuple[int, float] = (-1, -99999)
        for action in only_valid:
            c = self.get_child_if_exists(action)
            child_prior = self.policy[action].item()
            if c is not None:
                q_value = 0.0 if c.visit_cnt == 0 else -(c.utility_sum / c.visit_cnt)
                u_value = C_PUCT * child_prior * parent_visit_cnt_sqrt / (1 + c.visit_cnt)
                score = q_value + u_value
            else:
                q_value = 0.0
                u_value = C_PUCT * child_prior * parent_visit_cnt_sqrt  # / (1 + 0)
                score = q_value + u_value
            if score > best[1]:
                best = (action, score)

        assert best[0] != -1
        return best[0]

    def get_child_if_exists(self, action: int) -> "Node | None":
        return self.children.get(action)

    def create_child(self, action: int) -> "Node":
        assert self.policy is not None, "Policy must be predicted before expansion"

        board_size = self.server.go.board_size
        policy_cpu = self.policy.numpy()  # type: ignore # TODO: maybe dont need .numpy()

        if action == board_size:
            next_uf = self.uf.copy()
        else:
            # simulate the move to retrieve the resulting board state
            color = 2 if self.is_white else 1
            result, undo = self.server.go.simulate_move(self.uf, action, color, self.get_hash_history())
            assert result == MoveResult.SUCCESS, "Illegal move cannot be lazily expanded"

            next_uf = self.uf.copy()
            self.uf.undo_move_changes(undo, self.server.go.zobrist)

        child = Node(
            next_uf,
            self.server,
            not self.is_white,
            self.agent,
            self,
            action,
            prior=policy_cpu[action],
        )
        self.children[action] = child

        return child

    def backprop(self, score: float):
        node = self
        sign = 1.0
        while node is not None:
            node.visit_cnt += 1
            node.utility_sum += sign * score
            node = node.parent
            sign *= -1.0

    def apply_virtual_loss(self, amount: float):
        node = self
        sign = 1.0
        while node is not None:
            node.visit_cnt += 1
            node.utility_sum -= sign * amount
            node = node.parent
            sign *= -1.0

    def revert_virtual_loss(self, amount: float):
        node = self
        sign = 1.0
        while node is not None:
            node.visit_cnt -= 1
            node.utility_sum += sign * amount
            node = node.parent
            sign *= -1.0


class MCTS:
    def __init__(
        self,
        server: GameServerGo,
        agent: AlphaZeroAgent,
        full_search_iterations: int,
        fast_search_iterations: int,
        full_seach_prop: float,
        table: LookupTable,
        eval_mode: bool = False,
    ):
        self.server = server
        self.table = table

        self.full_search_iterations = full_search_iterations
        self.fast_search_iterations = fast_search_iterations
        self.full_seach_prop = full_seach_prop

        self.eval_mode = eval_mode

        self.agent = agent
        self.timing_stats: defaultdict[str, float] = defaultdict(float)
        self.iterations_stats: defaultdict[str, int] = defaultdict(int)
        self.max_depth = 0

        self.root: Node | None = None

    @torch.no_grad()  # pyright: ignore
    def search(
        self,
        uf: UnionFind,
        is_white: bool,
        best_move: int,
    ) -> tuple[torch.Tensor, bool]:
        # ------------------------------------------------ init ------------------------------------------------
        search_init_start = time.time()
        # tree reuse
        if self.root is None or best_move == -1:  # case first search iteration
            self.root = Node(uf, self.server, is_white, self.agent)
        else:
            next_root = self.root.children.get(best_move)
            if next_root is None:
                self.root = Node(uf, self.server, is_white, self.agent)
            else:
                self.root = next_root
                self.root.parent = None

        # playout cap randomization
        if self.eval_mode:
            is_full_search = True
            playout_cap = self.full_search_iterations
            exploration_features_disabled = True
        else:
            is_full_search = np.random.rand() < self.full_seach_prop
            playout_cap = self.full_search_iterations if is_full_search else self.fast_search_iterations
            exploration_features_disabled = False
        print(f"starting {"full" if is_full_search else "fast"} search...")

        # reset stats
        self.timing_stats.clear()
        self.iterations_stats.clear()

        # get x0 for MCTS score target
        if self.root.policy is None:
            self.root.get_predictions(self.table, 0.0)
        x_0 = self.root.mu_s
        assert self.root.mu_s is not None and self.root.sigma_s is not None and x_0 is not None
        assert self.root.policy is not None

        self.root.score_utility = self.table.get_expected_uscore(self.root.mu_s, self.root.sigma_s, x_0).item()

        # Apply dirichlet noise
        print(
            f"Value estimate: {self.root.win_utility:.3f} (from {'white' if self.root.is_white else 'black'}'s perspective)"
        )
        if not self.eval_mode and self.root.visit_cnt == 0:
            alpha = (
                0.03 * self.agent.board_width * self.agent.board_width / np.count_nonzero(self.root.get_valid_moves())
            )
            dir_noise = np.random.dirichlet([alpha] * len(self.root.policy))
            dir_noise_tensor = torch.tensor(dir_noise, device=self.root.policy.device, dtype=self.root.policy.dtype)
            epsilon = 0.25
            self.root.policy = (1 - epsilon) * self.root.policy + epsilon * dir_noise_tensor
            self.root.policy /= torch.sum(self.root.policy)

        # ------------------------------------------------ start search ------------------------------------------------
        self.timing_stats["search_init"] += time.time() - search_init_start
        search_start = time.time()
        for batch_id in range(math.ceil(playout_cap / MCTS_BATCH_SIZE)):  # type: ignore

            # selection with forced playouts and lazy expand

            leaf_node_batch: list[Node] = []

            for i in range(MCTS_BATCH_SIZE):
                node = self.root
                expand_to = -1
                # force_action_taken = False
                select_start = time.time()
                while True:
                    # --- FORCED PLAYOUT LOGIC ---
                    if node.parent is None and not exploration_features_disabled:
                        assert node.policy is not None
                        policy = node.policy

                        child_forced_playouts = torch.floor((FORCED_PLAYOUTS_K * policy * node.visit_cnt) ** 0.5)
                        valid = node.get_valid_moves()
                        only_valid = np.where(valid == True)[0]

                        action_to_force = -1
                        next_node_prior: torch.Tensor | None = None
                        for action in only_valid:
                            child = node.get_child_if_exists(action)
                            child_prior = policy[action]
                            child_visit_cnt = 0 if child is None else child.visit_cnt

                            # find node to force by choosing the one with the largest prior
                            if child_visit_cnt < child_forced_playouts[action] and (
                                next_node_prior is None or child_prior > next_node_prior
                            ):
                                next_node_prior = child_prior
                                action_to_force = action  # child with larger prior found, override it

                        # if a child to force is found set it to the next child, otherwise continue without any forcing
                        if action_to_force != -1:
                            # force_action_taken = True
                            forced_child = node.get_child_if_exists(action_to_force)
                            if forced_child is None:
                                # Need to expand this forced action, it does not yet exist
                                expand_to = action_to_force
                                break
                            else:
                                node = forced_child
                                # force_action_taken = False
                                continue

                    # if force_action_taken and expand_to == -1:  # Check expand_to in case force decided to expand
                    #     force_action_taken = False
                    #     continue

                    if node.queued_for_inference:
                        node = None
                        break

                    is_node_done = node.has_game_ended()
                    if is_node_done:
                        win_util, score_util = node.get_utility(x_0)
                        combined_util = win_util + score_util
                        node.backprop(combined_util)
                        node = None
                        break

                    if node.policy is None:
                        leaf_node_batch.append(node)
                        node.queued_for_inference = True
                        node.apply_virtual_loss(MCTS_VIRTUAL_LOSS_VALUE)
                        node = None
                        break

                    # node is fully expanded and not none
                    child_idx = node.next()
                    child = node.get_child_if_exists(child_idx)
                    if child is None:
                        # found a node to expand
                        expand_to = child_idx
                        break
                    else:
                        node = child
                        continue

                if node is None:
                    # hit a terminal node, no reason to process it further
                    continue

                self.timing_stats["selection"] += time.time() - select_start
                self.iterations_stats["selection"] += 1

                if expand_to != -1:
                    expand_start = time.time()
                    new_child = node.create_child(expand_to)
                    self.max_depth = max(self.max_depth, new_child.depth)

                    leaf_node_batch.append(new_child)
                    new_child.queued_for_inference = True
                    new_child.apply_virtual_loss(MCTS_VIRTUAL_LOSS_VALUE)
                    self.timing_stats["expansion"] += time.time() - expand_start
                    self.iterations_stats["expansion"] += 1
                else:
                    print("here")

            if len(leaf_node_batch) == 0:
                continue  # TODO: break cause when a batch could not be filled the game is most likely over?

            inference_start = time.time()
            props_batch, value_batch, mu_batch, sigma_batch = self.agent.predict_eval_batch(leaf_node_batch)
            self.timing_stats["nn_inference"] += time.time() - inference_start
            self.iterations_stats["nn_inference"] += 1

            for i, node in enumerate(leaf_node_batch):
                node.revert_virtual_loss(MCTS_VIRTUAL_LOSS_VALUE)
                node.queued_for_inference = False

                # fill child with data
                end_check_start = time.time()
                if node.done is None:
                    node.done = node.has_game_ended()
                assert node.done is not None
                self.timing_stats["end_check"] += time.time() - end_check_start
                self.iterations_stats["end_check"] += 1

                evaluation_start_time = time.time()
                if node.done:
                    node.win_utility, node.score_utility = node.get_utility(x_0)
                    assert node.win_utility is not None and node.score_utility is not None, "Values should not be None"
                else:
                    if node.policy is None:
                        # This sets leaf_node.win_utility and leaf_node.score_utility
                        # final_probs, _, _, _ = node.get_predictions(self.table, x_0)
                        node.policy = props_batch[i].cpu()
                        node.win_utility = value_batch[i].item()
                        node.mu_s = mu_batch[i].item()
                        node.sigma_s = sigma_batch[i].item()
                        node.score_utility = self.table.get_expected_uscore(
                            mu=node.mu_s, sigma=node.sigma_s, x_0=x_0
                        ).item()

                    assert node.win_utility is not None and node.score_utility is not None, "Values should not be None"
                self.timing_stats["evaluation"] += time.time() - evaluation_start_time
                self.iterations_stats["evaluation"] += 1

                # backpropagation
                backprop_start = time.time()
                assert node.win_utility is not None and node.score_utility is not None
                node.backprop(node.win_utility + node.score_utility)
                self.timing_stats["backprop"] += time.time() - backprop_start
                self.iterations_stats["backprop"] += 1

            # plot tree from root node for debugging
            # if iter == playout_cap - 1:
            #     TreePlot(self.root).create_tree()

        # calculate final policy distribution
        final_policy_start = time.time()

        if exploration_features_disabled:
            props = torch.zeros(
                self.agent.board_height * self.agent.board_height + 1,
                device=self.agent.device,
                dtype=torch.float32,
            )
            for c in self.root.children.values():
                # i use c.action so it **should** be the same as the action index on the board
                props[c.action] = c.visit_cnt

            props /= props.sum()

            self.timing_stats["final_policy"] += time.time() - final_policy_start
            total_time = time.time() - search_start
            self.timing_stats["total"] = total_time

            print("\nMCTS Timing Statistics:")
            print(f"Total search time: {total_time:.3f}s")
            for key, val in self.timing_stats.items():
                if key != "total":
                    percentage = (val / total_time) * 100
                    avg_time = val / float(self.iterations_stats.get(key, playout_cap))
                    print(f"{key}: {val:.3f}s ({percentage:.1f}%) - Avg: {avg_time*1000:.2f}ms")

            return props, is_full_search

        # policy target pruning
        c_star = max(self.root.children.values(), key=lambda c: c.visit_cnt)
        assert c_star.parent is not None, "c_start must have a parent"
        c_star_puct = get_puct_value(c_star, c_star.parent.visit_cnt, c_puct=C_PUCT)
        explore_scaling = C_PUCT * self.root.visit_cnt**0.5
        pruned_visit_counts = {c.action: float(c.visit_cnt) for c in self.root.children.values()}

        for c in self.root.children.values():
            action = c.action
            # use action to differentiate between c and c_star, since action is unique for every child
            if action is None or action == c_star.action:
                continue

            n_orig = float(c.visit_cnt)
            wins = c.utility_sum
            child_prior = c.prior

            # set visit count to 0 if c has no visits
            if n_orig <= 0:
                pruned_visit_counts[action] = 0.0
                continue

            child_q_value = wins / n_orig
            child_utility_from_parent = -child_q_value

            wanted_weight = get_explore_selection_value_inverse(
                c_star_puct, explore_scaling, child_prior, child_utility_from_parent
            )

            actual_weight = n_orig
            pruned_weight = min(wanted_weight, actual_weight)

            if pruned_weight < 1.0:
                # print(f"Pruning child {action} from {actual_weight:.3f} to {pruned_weight:.3f}")
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
                avg_time = val / float(self.iterations_stats.get(key, playout_cap))
                print(f"{key}: {val:.3f}s ({percentage:.1f}%) - Avg: {avg_time*1000:.2f}ms")

        return props, is_full_search


def choose_action_temperature(pi: torch.Tensor, temperature: float) -> int:
    safe_temperature = max(temperature, 1e-6)

    if abs(safe_temperature - 0.0) < 1e-6:
        return int(torch.argmax(pi).item())
    else:
        pi_temp = pi ** (1.0 / safe_temperature)
        pi_temp_sum = pi_temp.sum()
        # Handle edge case where sum is zero (e.g., pi was all zeros)
        if pi_temp_sum < 1e-9:
            print("Warning: Sum of temperature-scaled probabilities is near zero. Falling back to argmax.")
            return int(torch.argmax(pi).item())

        pi_temp /= pi_temp_sum
        pi_np = pi_temp.cpu().numpy().astype(np.float64)  # type: ignore

        pi_np = np.maximum(pi_np, 0)
        pi_np /= pi_np.sum()

        chosen = np.random.choice(len(pi_np), p=pi_np)
        return int(chosen)


def temperature_decay(episode_length: int) -> float:
    if episode_length < TEMPERATURE_DECAY_MOVES:
        # Linear decay
        current_temp = INITIAL_TEMPERATURE + (FINAL_TEMPERATURE - INITIAL_TEMPERATURE) * (
            episode_length / TEMPERATURE_DECAY_MOVES
        )
        return max(current_temp, FINAL_TEMPERATURE)
    else:
        return FINAL_TEMPERATURE


async def main() -> None:
    # torch.manual_seed(0)  # pyright: ignore
    # np.random.seed(0)
    board_size = 7
    komi = 5.5

    server = GameServerGo(board_size)
    if USE_BITBURNER:
        await server.wait()
    print("GameServer ready and client connected")
    print("initializing MCTS...")

    cleanup_out_folder("D:/AProgramming/Bitburner/bb-external-editor/http_server/out")
    plotter = TensorBoardPlotter()
    mo = ModelOverlay(board_size, komi)

    agent = AlphaZeroAgent(board_size, komi, plotter, batch_size=256)
    # agent.load_checkpoint("checkpoint_51.pth", True)

    table = LookupTable(board_size, C_SCORE, komi)
    mcts = MCTS(
        server, agent, full_search_iterations=1000, fast_search_iterations=200, full_seach_prop=0.25, table=table
    )

    game_state_plotter = GameStatePlotter(board_size)
    generator = GoStateGenerator(board_size)

    print("Done! Starting MCTS...")

    temperature = 0.8
    outcome = 0
    for iter in range(NUM_EPISODES):
        if USE_BITBURNER:
            white_starts = False
        else:
            white_starts = iter % 2 == 0  # white even, black odd
        white_komi = 0 if white_starts else komi
        black_komi = komi if white_starts else 0

        if USE_BITBURNER:
            state, _ = await server.reset_game("No AI", white_starts)
        else:
            state = generator.convert_state_to_MCTS(generator.generate_board_state())

        uf = UnionFind.get_uf_from_state(state, server.go.zobrist)
        server.go = Go_uf(board_size, state, komi, white_starts)

        buffer: list[BufferElement] = []
        done = False
        game_history = [state]
        mcts.agent.policy_net.eval()
        episode_length = 0
        previous_move = -1
        is_white = white_starts
        start_time = time.time()
        while not done:
            print(f"================================ {episode_length} ================================")
            before = time.time()
            pi_mcts, is_full_search = mcts.search(server.go.uf, is_white, previous_move)
            after = time.time()
            print(f"TOOK: {after-before}s")
            print(pi_mcts)
            if plotter is not None: # type: ignore
                entropy = -sum(pi_mcts * np.log2(pi_mcts))
                kl_divergence = sum(pi_mcts * np.log(pi_mcts / mcts.root.policy)) # type: ignore
                plotter.update_stat("mcts/pi_mcts_entropy", entropy)
                plotter.update_stat("mcts/kl_divergence_root", kl_divergence)

            best_move = choose_action_temperature(pi_mcts, temperature)
            temperature = temperature_decay(episode_length)

            print(f"{best_move}, {pi_mcts[best_move]}")
            action = mcts.agent.decode_action(best_move)
            print(f"make move: {action}")

            # add move response to buffer
            if len(buffer) > 0:
                buffer[-1].pi_mcts_response = pi_mcts

            valid_moves = server.go.get_valid_moves(uf, is_white, server.go.hash_history)
            buffer.append(
                BufferElement(
                    uf, is_white, pi_mcts, game_history[: mcts.agent.history_length], valid_moves, is_full_search
                )
            )

            # outcome is: 1 if black won, -1 is white won
            if USE_BITBURNER:
                next_uf, outcome, done = await server.make_move(action, best_move, is_white)
            else:
                next_uf, outcome, done = server.make_move_local(best_move, is_white, game_state_plotter)
            game_history.insert(0, next_uf.state)

            is_white = not is_white
            uf = next_uf
            previous_move = best_move
            episode_length += 1

        assert outcome != 0, "outcome should not be 0 after a game ended"

        if mcts.agent.plotter is not None:
            mcts.agent.plotter.update_stat_dict(
                "misc/cumulative_wins_over_time",
                {"white": 1 if outcome == -1 else -1, "black": 1 if outcome == 1 else -1},
                cumulative=True,
            )
            # plotter.update_stat("depth", mcts.max_depth)  # type: ignore
            # plotter.update_stat("episode_length", episode_length)  # type: ignore:
            mcts.agent.plotter.update_stat_dict(
                "misc/depth and episode length", {"tree depth": mcts.max_depth, "episode length": episode_length}
            )

        # last move has no response, so set it to zero
        buffer[-1].pi_mcts_response = torch.zeros(mcts.agent.board_height * mcts.agent.board_height + 1, device=mcts.agent.device)  # type: ignore

        mcts.max_depth = 0
        print(f"Episode length: {episode_length}, took {time.time()-start_time:.3f}s")
        print("================================================================================")

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

        black_score = np.count_nonzero(black_stones) + np.count_nonzero(black_territory) + black_komi
        white_score = np.count_nonzero(white_stones) + np.count_nonzero(white_territory) + white_komi
        score = black_score - white_score  # black leads with

        print("saving model overlay...")
        len_buffer = len(buffer)
        buffer_indices = [0, 1, len_buffer // 2, len_buffer // 2 + 1, len_buffer - 2, len_buffer - 1]
        # buffer_indices = range(len_buffer)  # for testing
        for i in buffer_indices:
            if i >= len_buffer:
                continue
            be = buffer[i]
            state_tensor, state_vector = agent.preprocess_state(
                be.uf.state, be.history, be.valid_moves, be.is_white, white_starts, "cpu"
            )
            logits = agent.policy_net(state_tensor, state_vector)
            next_uf = buffer[i + 1].uf if i + 1 < len_buffer else None
            next_next_uf = buffer[i + 2].uf if i + 2 < len_buffer else None
            # TODO: also log out: actual played move, pi_mcts, model out argmax
            fig = mo.heatmap(
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
            # plotter.update_figure("model_overlay", fig)
            plt.close(fig)
        print("done!")

        for be in buffer:
            # Flip if the outcome from neutrals perspective to players perspective
            z = outcome if not be.is_white else -outcome
            ownership_corrected = ownership_mask * (-1 if be.is_white else 1)
            score_corrected = score * (-1 if be.is_white else 1)
            mcts.agent.augment_state(be, z, ownership_corrected, score_corrected, white_starts)

        if iter < 12 or iter % 2 == 0:
            print("Skipping training")
            continue

        mcts.agent.policy_net.train()
        train_steps = 15
        print(f"Game length: {episode_length}, performing {train_steps} training steps")
        for _ in range(train_steps):
            mcts.agent.train_step()
        mcts.agent.save_checkpoint(f"checkpoint_{iter}.pth")


async def main_eval():
    board_size = 7
    komi = 5.5

    table = LookupTable(board_size, C_SCORE, komi)
    server = GameServerGo(board_size)
    await server.wait()
    print("GameServer ready and client connected")

    plotter = TensorBoardPlotter("eval_run")
    agent = AlphaZeroAgent(board_size, komi, plotter, checkpoint_dir="models")
    agent.load_checkpoint("checkpoint_21_7x7_but_broken.pth", False)
    mcts = MCTS(server, agent, 1000, 100, 0.25, table=table, eval_mode=True)

    NUM_EPISODES = 100
    outcome = 0
    for _ in range(NUM_EPISODES):
        state, _ = await server.reset_game("Daedalus")
        server.go = Go_uf(board_size, state, komi, False)

        is_white = False
        done = False
        mcts.agent.policy_net.eval()
        while not done:
            pi_mcts, _ = mcts.search(server.go.uf, is_white, -1)
            print(pi_mcts)
            best_move = int(torch.argmax(pi_mcts).item())
            print(f"{best_move}, {pi_mcts[best_move]}")
            action = mcts.agent.decode_action(best_move)
            print(f"make move: {action}")

            # outcome is: 1 if black won, -1 is white won
            next_state, outcome, done = await server.make_move_eval(action, best_move, is_white)
            state = next_state

        assert outcome != 0, "outcome should not be 0 after a game ended"

        if mcts.agent.plotter is not None:
            mcts.agent.plotter.update_stat_dict(
                "cumulative_wins_over_time",
                {"white": 1 if outcome == -1 else -1, "black": 1 if outcome == 1 else -1},
                cumulative=True,
            )


# Run the main coroutine
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
