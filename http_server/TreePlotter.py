import igraph  # type: ignore
import plotly.graph_objects as go  # type: ignore
from igraph import EdgeSeq, Graph

# from MCTS_zero import Node


def rotate_state(state: list[str]) -> list[str]:
    rotated_state: list[str] = []

    for i in range(len(state)):
        tmp = ""
        for j in range(len(state)):
            tmp += state[j][i]

        rotated_state.append(tmp)
    rotated_state.reverse()
    return rotated_state


def beatify_state(state: list[str]) -> str:
    beautified_state: str = ""
    for i in range(len(state)):
        for j in range(len(state)):
            beautified_state += f"{state[i][j]} "
        beautified_state += "<br>"
    return beautified_state


class TreePlot:
    def __init__(self, root):
        self.root = root
        self.nodes = []
        self.edges = []
        self.labels = []

    def create_tree(self):
        def traverse(node, parent_idx=None):
            idx = len(self.nodes)
            self.nodes.append(node)
            self.labels.append(
                f"win: {node.win_sum}<br>visit: {node.visit_cnt}<br>white: {node.is_white}<br>{beatify_state(rotate_state(node.state))}"
            )
            if parent_idx is not None:
                self.edges.append((parent_idx, idx))
            for c in node.children:
                traverse(c, idx)

        traverse(self.root)
        self.create_plot()

    def create_plot(self):
        G = Graph(edges=self.edges, directed=True)
        lay = G.layout_reingold_tilford(root=[0], mode="out")
        # Convert positions to Plotly points
        Xn, Yn, Xe, Ye = [], [], [], []
        for i, pos in enumerate(lay):
            Xn.append(pos[0])
            Yn.append(pos[1])
            
        for e in G.es:
            Xe += [lay[e.source][0], lay[e.target][0], None]
            Ye += [lay[e.source][1], lay[e.target][1], None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Xe, y=Ye, mode="lines"))
        fig.add_trace(
            go.Scatter(
                x=Xn,
                y=Yn,
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
        fig.show()


# if __name__ == "__main__":
#     root = Node()
#     child1 = Node(root)
#     child2 = Node(root)
#     root.children = [child1, child2]
#     grandchild = Node(child1)
#     child1.children = [grandchild]

#     tp = TreePlot(root)
#     tp.create_tree()
