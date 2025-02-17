import dash
from dash import html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__)

# Sample data: Replace this with your actual tree data
nodes = [
    {
        "data": {
            "id": "root",
            "label": "Root\nScore: 10",
            "board": "X|O|X\nO|X|O\nX|O|X",
        }
    },
    {
        "data": {
            "id": "child1",
            "label": "Child 1\nScore: 5",
            "board": "O|X|O\nX|O|X\nO|X|O",
        }
    },
    {
        "data": {
            "id": "child2",
            "label": "Child 2\nScore: 3",
            "board": "X|X|O\nO|O|X\nX|O|X",
        }
    },
]

edges = [
    {"data": {"source": "root", "target": "child1"}},
    {"data": {"source": "root", "target": "child2"}},
]

elements = nodes + edges

# Define the layout and styles
app.layout = html.Div(
    [
        cyto.Cytoscape(
            id="game-tree",
            elements=elements,
            layout={"name": "breadthfirst"},
            style={"width": "100%", "height": "800px"},
            stylesheet=[
                {
                    "selector": "node",
                    "style": {
                        "content": "data(label)",
                        "text-valign": "center",
                        "text-halign": "center",
                        "shape": "rectangle",
                        "width": "label",
                        "height": "label",
                        "padding": "10px",
                        "background-color": "#BFD7B5",
                        "font-size": "12px",
                        "border-color": "#000",
                        "border-width": 1,
                    },
                },
                {
                    "selector": "edge",
                    "style": {
                        "width": 2,
                        "line-color": "#A3C4BC",
                        "target-arrow-color": "#A3C4BC",
                        "target-arrow-shape": "triangle",
                    },
                },
                {
                    "selector": ":selected",
                    "style": {
                        "background-color": "#FF5722",
                        "line-color": "#FF5722",
                        "target-arrow-color": "#FF5722",
                        "source-arrow-color": "#FF5722",
                    },
                },
            ],
            # Enable panning and zooming
            minZoom=0.2,
            maxZoom=2,
            zoom=1,
            # Add all possible event listeners
            responsive=True,
            # Add clientside event callbacks
            autoungrabify=False,
            autounselectify=False,
        ),
        html.Div(id="node-details", style={"padding": "20px", "border": "1px solid #ccc"}),
        html.Div(id="click-data", style={"padding": "20px"}),
        # Add a hidden div for debugging
        html.Div(id="debug-output", style={"display": "none"}),
    ]
)

# Add multiple callbacks for different events
@app.callback(
    [Output("node-details", "children"), Output("click-data", "children")],
    [Input("game-tree", "tapNodeData"),
     Input("game-tree", "selectedNodeData"),
     Input("game-tree", "mouseoverNodeData")]
)
def update_node_data(tap_data, select_data, hover_data):
    logger.debug(f"Tap data: {tap_data}")
    logger.debug(f"Select data: {select_data}")
    logger.debug(f"Hover data: {hover_data}")
    
    ctx = dash.callback_context
    logger.debug(f"Triggered by: {ctx.triggered[0]['prop_id']}")
    
    # Use the first non-None data we find
    node_data = tap_data or select_data or hover_data
    
    if not node_data:
        return "Interact with a node to see details.", "No interaction detected."
    
    if isinstance(node_data, list):
        node_data = node_data[0]
    
    try:
        data = node_data.get('data', node_data)
        node_id = data.get('id', 'Unknown')
        label = data.get('label', 'No label')
        board = data.get('board', 'No board data')
        
        details = html.Div([
            html.H4(f"Node: {node_id}"),
            html.P(label),
            html.Pre(board),
        ])
        
        debug_info = f"Last interaction - Node: {node_id}"
        return details, debug_info
    
    except Exception as e:
        logger.error(f"Error processing node data: {e}")
        return f"Error: {str(e)}", "Error occurred"

if __name__ == "__main__":
    logger.info("Starting server...")
    app.run_server(debug=True)
