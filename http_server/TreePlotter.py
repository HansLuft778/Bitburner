import plotly.graph_objects as go
from igraph import EdgeSeq, Graph  # type: ignore
from typing import TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from MCTS_zero_uf import Node


def rotate_state(state: np.ndarray[Any, np.dtype[np.int8]]) -> list[str]:
    rotated_state: list[str] = []

    for i in range(len(state)):
        tmp = ""
        for j in range(len(state)):
            tmp += str(state[j][i])

        rotated_state.append(tmp)
    rotated_state.reverse()
    return rotated_state


def beatify_state(state: list[str], delim: str = "<br>") -> str:
    beautified_state: str = ""
    for i in range(len(state)):
        for j in range(len(state)):
            beautified_state += f"{state[i][j]} "
        beautified_state += delim
    return beautified_state


def rotate_and_beatify(state: np.ndarray[Any, np.dtype[np.int8]], delim: str = "<br>") -> str:
    return beatify_state(rotate_state(state), delim)


class TreePlot:
    def __init__(self, root: "Node"):
        self.root = root
        self.nodes: list["Node"] = []
        self.edges: list[tuple[int, int]] = []
        self.labels: list[str] = []

    def create_tree(self):
        num_nodes = 1

        def traverse(node: "Node", parent_idx: int | None = None):
            nonlocal num_nodes
            idx = len(self.nodes)
            self.nodes.append(node)
            self.labels.append(
                f"win util: {node.win_utility}<br>score util: {node.score_utility}<br>visit: {node.visit_cnt}<br>white: {node.is_white}<br>{rotate_and_beatify(node.uf.state)}"
            )
            if parent_idx is not None:
                self.edges.append((parent_idx, idx))
            for c in node.children:
                num_nodes += 1
                traverse(c, idx)

        traverse(self.root)
        print("num_nodes: ", num_nodes)
        self.create_plot()

    def create_plot(self):
        G = Graph(edges=self.edges, directed=True)
        lay = G.layout_reingold_tilford(root=[0], mode="out")  # type: ignore
        # Convert positions to Plotly points
        Xn, Yn, Xe, Ye = [], [], [], []
        for i, pos in enumerate(lay):  # type: ignore
            Xn.append(pos[0])  # type: ignore
            Yn.append(pos[1])  # type: ignore

        for e in G.es:  # type: ignore
            Xe += [lay[e.source][0], lay[e.target][0], None]  # type: ignore
            Ye += [lay[e.source][1], lay[e.target][1], None]  # type: ignore

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Xe, y=Ye, mode="lines"))  # type: ignore
        fig.add_trace(  # type: ignore
            go.Scatter(
                x=Xn,  # type: ignore
                y=Yn,  # type: ignore
                mode="markers",
                text=self.labels,
                hoverinfo="text",
                hoverlabel=dict(
                    font=dict(
                        family="Courier New, monospace",  # monospaced font
                        color="black",  # text color
                    ),
                    bgcolor="white",  # hover background color
                    bordercolor="black",
                ),
            )
        )
        fig.update_yaxes(autorange="reversed")
        # fig.show()
        fig.write_html(f"out/plot{1}.html")  # type: ignore
