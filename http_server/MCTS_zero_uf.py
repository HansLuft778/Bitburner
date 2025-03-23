import time
from collections import defaultdict
from typing import Any, Union

import numpy as np
import torch

from agent_cnn_zero import AlphaZeroAgent
from gameserver_local_uf import GameServerGo
from Go.Go_uf import Go_uf, UnionFind
from plotter import Plotter  # type: ignore
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
    u_value = c_puct * child.selected_policy * parent_visit_count**0.5 / (1 + child.visit_cnt)
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
        # self.state = state
        self.uf: UnionFind = uf
        self.parent = parent
        self.action = action
        self.server = server
        self.is_white = is_white
        self.agent = agent
        self.selected_policy = selected_policy
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
        best: tuple[Node | None, float] = (None, -999999999)
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
                empty_cells = np.sum(self.uf.state == 0)
                empty_percentage = empty_cells / board_size
                if empty_percentage > 0.5:
                    policy_cpu[action] *= (1.0 - empty_percentage) * 0.5
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
                node = node.next()
                self.max_depth = max(self.max_depth, node.depth)
            self.timing_stats["selection"] += time.time() - select_start
            self.iterations_stats["selection"] += 1

            # expansion
            end_check_start = time.time()
            if node.done is None:
                node.done, node.value = node.has_game_ended()
            assert node.value is not None, "Value should not be None"
            value = node.value
            done = node.done

            self.timing_stats["end_check"] += time.time() - end_check_start

            if not done:
                prep_start = time.time()
                valid_moves = node.get_valid_moves()
                history = node.get_history_ref()
                history.extend(self.server.get_game_history())
                self.timing_stats["move_prep"] += time.time() - prep_start

                inference_start = time.time()
                if node.policy is None:
                    raw_logits, raw_value = self.agent.get_actions_eval(
                        node.uf.state, valid_moves, history, node.is_white
                    )
                    valid_mask = torch.tensor(valid_moves, device=self.agent.device, dtype=torch.bool)
                    raw_logits[~valid_mask] = -1e9
                    final_probs = torch.softmax(raw_logits, dim=0)

                    # Apply dirichlet noise
                    if node.parent is None and not eval_mode:
                        alpha = 0.2
                        dir_noise = np.random.dirichlet([alpha] * len(final_probs))
                        dir_noise_tensor = torch.tensor(dir_noise, device=final_probs.device, dtype=final_probs.dtype)
                        epsilon = 0.2  # noise weight
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

                if node.parent is None:
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
    plotter = Plotter(2, 3)
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
    plotter.add_plot("policy_loss", plotter.axes[1, 0], "Policy Loss Over Time", "Updates", "Policy Loss")  # type: ignore
    plotter.add_plot("value_loss", plotter.axes[1, 1], "Value Loss Over Time", "Updates", "Value Loss")  # type: ignore
    plotter.add_plot("depth", plotter.axes[0, 2], "MCTS Depth", "Iteration", "Depth")  # type: ignore

    agent = AlphaZeroAgent(board_size, plotter)
    # agent.load_checkpoint("checkpoint_15.pth")
    mcts = MCTS(server, agent, search_iterations=2000)

    NUM_EPISODES = 1000
    outcome = 0
    for iter in range(NUM_EPISODES):
        state, komi = await server.reset_game("No AI")
        server.go = Go_uf(board_size, state, komi)

        buffer: list[tuple[State, bool, torch.Tensor, list[State]]] = []  # score: dict[str, dict[str, Any]]]
        is_white = False
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
            # child_idx = get_child_idx(pi_mcts, best_move)
            print(f"{best_move}, {pi_mcts[best_move]}")
            action = mcts.agent.decode_action(best_move)
            print(f"make move: {action}")

            buffer.append((state, is_white, pi_mcts, game_history[: mcts.agent.num_past_steps]))

            # outcome is: 1 if black won, -1 is white won
            next_state, outcome, done = await server.make_move(action, best_move, is_white)
            game_history.insert(0, next_state)

            is_white = not is_white
            state = next_state
            previous_move = best_move
            episode_length += 1

        plotter.update_stat("depth", mcts.max_depth)  # type: ignore
        mcts.max_depth = 0
        print(f"Episode length: {episode_length}, took {time.time()-start_time:.3f}s")
        print("================================================================================")

        assert outcome != 0, "outcome should not be 0 after a game ended"

        mcts.agent.plotter.update_wins_white(1 if outcome == -1 else -1)
        mcts.agent.plotter.update_wins_black(1 if outcome == 1 else -1)

        mcts.agent.policy_net.train()
        for state, was_white, pi, history in buffer:
            # Flip if the buffer entry belongs to the opposite color
            #  - opposite of player who moves
            z = outcome if not was_white else -outcome
            mcts.agent.augment_state(state, pi, z, history, was_white)

        if iter < 16:
            print("Skipping training")
            continue
        game_length = len(game_history)
        min_train_steps = 10  # Minimum number of training steps
        max_train_steps = 40  # Maximum number of training steps

        # Calculate training steps - scales with game length
        train_steps = min(max_train_steps, max(min_train_steps, int(game_length * 0.75)))
        train_steps = 10
        print(f"Game length: {game_length}, performing {train_steps} training steps")
        for _ in range(train_steps):
            mcts.agent.train_step()
        mcts.agent.save_checkpoint(f"checkpoint_{iter}.pth")


async def main_eval():
    board_size = 5
    server = GameServerGo(board_size)
    await server.wait()
    print("GameServer ready and client connected")

    plotter = Plotter()
    agent = AlphaZeroAgent(board_size, plotter)
    agent.load_checkpoint("checkpoint_69.pth")
    mcts = MCTS(server, agent, search_iterations=1000)

    NUM_EPISODES = 100
    outcome = 0
    for _ in range(NUM_EPISODES):
        state, komi = await server.reset_game("Netburners")
        server.go = Go_uf(board_size, state, komi)

        is_white = False
        done = False
        mcts.agent.policy_net.eval()
        while not done:
            pi_mcts = mcts.search(server.go.uf, is_white, True)
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


def debug_sign_flips():
    """
    Debug function: sets up a simple 5x5 board with a known final pattern of stones,
    forcibly ends the game, prints out the internal final scoring and outcome from your code.
    """
    server = GameServerGo(5)

    # 1) Reset the server to an empty board of size 5x5
    board_size = 5
    state, komi = server.go.encode_state(["#..O#", "#..X.", "#..OX", "#XXXX", "#.X.."]), 5.5
    server.go = Go_uf(board_size, state, komi)

    # 2) Force a few moves (or directly set board state)
    # For example, let's place two black stones and one white stone in known positions.
    # You can skip MCTS and do something like:
    # black_stone_positions = [(0, 0), (2, 2)]  # black stones
    # white_stone_positions = [(0, 4)]  # white stone

    # # We'll pretend these moves were made so the board is partially filled:
    # color_black = 1
    # color_white = 2
    # for x, y in black_stone_positions:
    #     server.go.uf.state[x, y] = color_black
    # for x, y in white_stone_positions:
    #     server.go.uf.state[x, y] = color_white

    # If needed, recalc Zobrist hash so the server's union-find is consistent:
    server.go.uf = UnionFind.get_uf_from_state(state, server.go.zobrist)

    # 3) Now forcibly end the game (simulate passes or you can directly check scoring).
    #    The simplest way: call your "get_score" logic from MCTS or from the server:
    score_dict = server.get_score(server.go.uf)
    score_white = score_dict["white"]["sum"]
    score_black = score_dict["black"]["sum"]
    diff = score_white - score_black

    print("DEBUG BOARD (Black=1, White=2):\n", server.go.uf.state)
    print(f"Score black: {score_black}, score white: {score_white}, difference (white - black)={diff}")

    # 4) Now ask: from black's perspective, is that a winning difference or not?
    #    If you do normalized difference in your Node code, do that here as well:
    max_score = board_size * board_size
    normalized_diff = diff / max_score
    print(f"Normalized difference: {normalized_diff}")

    # 5) Suppose we interpret that if white leads, the raw outcome is -1 (meaning "white wins").
    #    If black leads, raw outcome is +1. Let's see:
    outcome_manual = +1 if (diff < 0) else -1 if (diff > 0) else 0  # depends if you do "white minus black"
    # ^ Note that if diff = (white - black), then diff>0 => white is winning => outcome should be -1.
    #   Adjust that logic to match your code exactly.

    print(f"Manual outcome (based on 'white - black' diff): {outcome_manual}")

    # 6) If you want to fully replicate your MCTS perspective logic, you can do:
    is_white = False
    perspective = "white" if is_white else "black"
    node = Node(
        uf=server.go.uf.copy(),
        server=server,
        is_white=is_white,  # let's say it's black's turn, for debugging
        agent=AlphaZeroAgent(5, Plotter()),  # or pass a dummy agent
    )
    print(f"score: {node.get_score()} from {perspective}'s perspective")
    done, value = node.has_game_ended()
    print(f"has_game_ended says: done={done}, value={value} (from {perspective}'s perspective)")

    # If done is True, 'value' should be:
    #   +something if black is winning,
    #   -something if white is winning.
    # Then you see if that matches your manual logic above.

    print("DONE. Compare these results with your manual expectation.")


def debug_perspective_inversion():
    agent = AlphaZeroAgent(5, Plotter())
    server = GameServerGo(5)

    state = server.go.encode_state(["#..#.", "#....", "#....", "#....", "#...."])
    server.go = Go_uf(5, state, 5.5)

    uf = UnionFind.get_uf_from_state(state, server.go.zobrist)
    root = Node(uf, server, False, agent, None, None)

    moves = [1, 2, 7, 6, 11, 4, 9, 8, 13, 14, 19, 21]
    node = root
    for move in moves:
        # Simulate the move
        uf_after = server.go.state_after_action(move, node.is_white, node.uf)
        assert uf_after is not None

        # node is: state was reached by move, next is is_white's turn
        node.children.append(Node(uf_after, server, not node.is_white, agent, node, move))
        node = node.children[-1]

        value = node.get_score()
        node.backprop(value)

    # Suppose we pass some "score=+0.7" meaning White is quite happy.

    print("Parent (black) win_sum :")


# Run the main coroutine
if __name__ == "__main__":
    # import asyncio

    # asyncio.run(main())
    debug_perspective_inversion()
