import networkx as nx
import matplotlib.pyplot as plt
from definitions import ROOT_DIR
import numpy as np

# import pygraphviz
def visualize_mcts(mcts):
    G = nx.Graph()

    max_visits_edge = 100# max([count for count in mcts.times_edge_s_a_was_visited.values()])
    labels, state = add_edges_to_graph(G, max_visits_edge, mcts)
    node_sizes = [G.nodes[node]['size'] for node in G.nodes]
    node_color = [G.nodes[node]['color'] for node in G.nodes]
    symbols = [G.nodes[node]['symbol'] for node in G.nodes]
    edge_color = [G.edges[edge]['color'] for edge in G.edges]




    fig = create_figure(G, edge_color, labels, max_visits_edge, node_sizes,
                        state,node_color, args = mcts.args, symbols=symbols)
    plt.tight_layout()
    fig.savefig(fname = ROOT_DIR / 'plots' / f"{mcts.args.mcts_engine}_{mcts.args.num_MCTS_sims}{'_labels' if mcts.args.with_labels else ''}.svg",
                transparent=None, dpi='figure', format=None,
        metadata=None, bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto', backend=None,
       )

def add_edges_to_graph(G, max_visits_edge, mcts):
    labels = {}
    for key, state in mcts.Ssa.items():
        add_edge(G, key, max_visits_edge, mcts, state)
        add_node_size(G, key, mcts, state)
        add_node_labels(labels, state)
    # for start node:
    add_start_node(G, mcts, labels)
    return labels, state

def add_edge(G, key, max_visits_edge, mcts, state):
    color_edge = round(mcts.times_edge_s_a_was_visited[key] / max_visits_edge, 2)
    G.add_edge(u_of_edge=state.hash,
               v_of_edge=state.previous_state.hash,
               color=color_edge)


def add_node_size(G, key, mcts, state):
    size = float(mcts.Qsa[key])
    nx.set_node_attributes(G, {state.hash: 1.8 ** (size + 10) + 20}, name="size")
    nx.set_node_attributes(G, {state.hash: 'o'}, name="symbol")
    if state.reward > 0.99:
        nx.set_node_attributes(G, {state.hash: 'cyan'}, name="color")
    else:
        nx.set_node_attributes(G, {state.hash: 'brown'}, name="color")
    # if size < 0:
    #     nx.set_node_attributes(G, {state.hash: 0}, name="size")
    # elif size == 0:
    #     nx.set_node_attributes(G, {state.hash: 10}, name="size")
    # else:
    #     nx.set_node_attributes(G, {state.hash: 20 + size * 100}, name="size")


def add_node_labels(labels, state):
    labels[state.hash] = state.syntax_tree.rearrange_equation_infix_notation(-1)[1]
    # if state.done:
    #     if hasattr(state, 'found_equation'):
    #         labels[state.hash] = state.found_equation
    #     else:
    #         labels[state.hash] = state.syntax_tree.rearrange_equation_infix_notation(-1)[1]


def add_start_node(G, mcts, labels):
    start_key = list(mcts.times_s_was_visited)[0]
    nx.set_node_attributes(G, {start_key: 600}, name="size")
    nx.set_node_attributes(G, {start_key: 'black'}, name="color")
    nx.set_node_attributes(G, {start_key: 'x'}, name="symbol")
    labels[start_key] = 'S'





def create_figure(G, edge_color, labels, max_visits_edge, node_sizes,
                  state, node_color, args, symbols):
    fig, (ax1) = plt.subplots(figsize=(20, 16), ncols=1, dpi=300)
    start_node_boolean=  [True if sNode[1]['symbol'] == 'x' else False for sNode in G.nodes(data=True)]  # list with keys
    start_node_key = [sNode[0] for sNode in filter(lambda x: x[1]["symbol"] == 'x', G.nodes(data=True))]
    other_node_boolean = [True if sNode[1]['symbol'] == 'o' else False for sNode in G.nodes(data=True)]  # list with keys
    other_node_key = [sNode[0] for sNode in filter(lambda x: x[1]["symbol"] == 'o', G.nodes(data=True))]
    #nodePos = nx.layout.spring_layout(G)
    # start_node
    nodePos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
    nx.draw_networkx_nodes(
        G=G,
        pos=nodePos,
        nodelist= start_node_key,
        node_size=np.array(node_sizes)[start_node_boolean],
        node_color=np.array(node_color)[start_node_boolean],
        node_shape='x',
        alpha=1,
        vmin=0,
        vmax=1,
        ax=ax1,
        linewidths=None,
        edgecolors=None,
        label=True if args.with_labels else False,
        margins=None
    )
    # other nodes
    nx.draw_networkx_nodes(
        G=G,
        pos=nodePos,
        nodelist=other_node_key,
        node_size=np.array(node_sizes)[other_node_boolean],
        node_color=np.array(node_color)[other_node_boolean],
        node_shape='o',
        alpha=0.5,
        vmin=0,
        vmax=1,
        ax=ax1,
        linewidths=None,
        edgecolors=None,
        label=True if args.with_labels else False,
        margins=None
    )
    nx.draw_networkx_edges(
        G=G,
        pos=nodePos,
        width=8,
        edge_color=edge_color,
        style='solid',
        alpha=0.5,
        arrowstyle=None,
        arrowsize=10,
        edge_cmap=plt.get_cmap('copper'),
        edge_vmin=0,
        edge_vmax=1,
        ax=ax1,
        arrows=True,
        node_size=np.array(node_sizes),
        nodelist=None,
        node_shape='o',
        connectionstyle='arc3',
        min_source_margin=10,
        min_target_margin=10
    )

    # nx.draw_networkx(G, pos,
    #                  ax=ax1,
    #                  node_size=node_sizes,
    #                  alpha=0.5,
    #                  node_color=node_color,
    #                  node_shape=symbols,
    #                  with_labels=True if args.with_labels else False,
    #                  arrows=True,
    #                  font_size=20,
    #                  edge_color=edge_color,
    #                  labels=labels,
    #                  cmap='copper',
    #                  width=8,
    #                  vmin=0,
    #                  vmax=1,
    #                  edge_vmin=0,
    #                  edge_vmax=1,
    #                  )
    if args.with_labels:
        nx.draw_networkx_labels(
            G=G,
            pos=nodePos,
            labels=labels,
            font_size=12,
            font_color='k',
            font_family='sans-serif',
            font_weight='normal',
            alpha=0.5,
            bbox=None,
            horizontalalignment='center',
            verticalalignment='center',
            ax=ax1,
            clip_on=True
        )

    add_colorbar(ax1, fig, max_visits_edge,
                 args= args)
    if args.with_labels:
        fig.suptitle(f"true equation {state.observation['true_equation']}", fontsize=40)
    return fig


def add_colorbar(ax1, fig, max_visits_edge, args):
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='copper'), ax=ax1, location='right', shrink=0.7,
                        ticks=[0, 0.5, 1])
    if args.with_labels:
        cbar.ax.set_yticklabels(['1', str(max(0.5 * max_visits_edge, 1)), str(max_visits_edge)])
    else:
        cbar.ax.set_yticklabels(['', '', ''])

