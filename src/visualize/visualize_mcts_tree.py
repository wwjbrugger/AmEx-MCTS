import networkx as nx
import matplotlib.pyplot as plt


# import pygraphviz
def visualize_mcts(mcts):
    G = nx.Graph()

    max_visits_edge = max([count for count in mcts.times_edge_s_a_was_visited.values()])
    labels, state = add_edges_to_graph(G, max_visits_edge, mcts)
    node_sizes = [G.nodes[node]['size'] for node in G.nodes]
    edge_color = [G.edges[edge]['color'] for edge in G.edges]
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")

    fig = create_figure(G, edge_color, labels, max_visits_edge, node_sizes, pos, state)
    fig.show(block=True)


def add_edge(G, key, max_visits_edge, mcts, state):
    color_edge = round(mcts.times_edge_s_a_was_visited[key] / max_visits_edge, 2)
    G.add_edge(u_of_edge=state.hash,
               v_of_edge=state.previous_state.hash,
               color=color_edge)


def add_node_size(G, key, mcts, state):
    size = float(mcts.Rsa[key])
    if size < 0:
        nx.set_node_attributes(G, {state.hash: 0}, name="size")
    elif size == 0:
        nx.set_node_attributes(G, {state.hash: 10}, name="size")
    else:
        nx.set_node_attributes(G, {state.hash: 20 + size * 100}, name="size")


def add_node_labels(labels, state):
    if state.done:
        if hasattr(state, 'found_equation'):
            labels[state.hash] = state.found_equation
        else:
            labels[state.hash] = state.observation['current_tree_representation_str']


def add_start_node(G, mcts):
    start_key = list(mcts.times_s_was_visited)[0]
    nx.set_node_attributes(G, {start_key: 120}, name="size")


def add_edges_to_graph(G, max_visits_edge, mcts):
    labels = {}
    for key, state in mcts.Ssa.items():
        add_edge(G, key, max_visits_edge, mcts, state)
        add_node_size(G, key, mcts, state)
        add_node_labels(labels, state)
    # for start node:
    add_start_node(G, mcts)
    return labels, state


def create_figure(G, edge_color, labels, max_visits_edge, node_sizes, pos, state):
    fig, (ax1) = plt.subplots(figsize=(20, 20), ncols=1)
    nx.draw_networkx(G, pos,
                     ax=ax1,
                     node_size=node_sizes,
                     alpha=0.5,
                     # node_color=node_color,
                     with_labels=True,
                     arrows=True,
                     font_size=10,
                     edge_color=edge_color,
                     labels=labels,
                     cmap='turbo',
                     width=5,
                     vmin=0,
                     vmax=1,
                     edge_vmin=0,
                     edge_vmax=1,
                     )
    add_colorbar(ax1, fig, max_visits_edge)
    fig.suptitle(f"true equation {state.observation['true_equation']}", fontsize=40)
    return fig


def add_colorbar(ax1, fig, max_visits_edge):
    cbar = fig.colorbar(None, ax=ax1, location='right', shrink=0.7,
                        ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['1', str(0.5 * max_visits_edge), str(max_visits_edge)])
