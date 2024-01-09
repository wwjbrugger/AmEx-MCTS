import copy
from nltk.grammar import is_terminal
import bisect


class Node():
    def __init__(self, tree, node_id, parent_node, depth):
        self.tree = tree
        bisect.insort(self.tree.nodes_to_expand, node_id)
        self.tree.dict_of_nodes[node_id] = self
        self.node_id = node_id
        self.parent_node = parent_node
        self.list_children = []
        self.depth = depth
        self.math_class = None
        self.node_symbol = None
        self.invertible = None
        self.selected_action = []
        self.selected_production = []
        if self.depth > self.tree.current_depth:
            self.tree.current_depth = self.depth
        if self.depth + 1 == self.tree.max_depth:
            self.tree.max_depth_reached = True

    def prefix_to_syntax_tree(self, prefix, recursive=True):
        self.node_symbol = prefix.pop(0)
        self.invertible = self.parent_node.invertible and \
                          self.parent_node.math_class.invertible
        if not str(self.node_symbol) in self.tree.non_terminals:
            self.tree.nodes_to_expand.remove(self.node_id)

        if self.node_symbol in self.tree.operator_to_class.keys():
            self.math_class = self.tree.operator_to_class[self.node_symbol](
                node=self
            )
            total_number_of_child_in_one_branch = geometric_sum(
                max_branching_factor=self.tree.max_branching_factor,
                n=self.tree.max_depth - (self.depth + 2)
            )
            for i in range(self.math_class.num_child):
                child_node_id = self.node_id + 1 + i * total_number_of_child_in_one_branch
                child_node = Node(node_id=child_node_id,
                                  tree=self.tree,
                                  parent_node=self,
                                  depth=self.depth + 1
                                  )
                self.list_children.append((child_node))
                if recursive:
                    prefix = child_node.prefix_to_syntax_tree(prefix=prefix)
        else:
            # not a mathematical operator
            self.math_class = self.tree.operator_to_class['terminal'](
                node=self
            )
        return prefix

    def print(self):
        ret = f"d: {self.depth}  id: {str(self.node_id):<5} " + "\t" * self.depth + str(self.node_symbol)
        print(ret)
        for child in self.list_children:
            child.print()

    def delete(self):
        self.parent_node.list_children.remove(self)
        self.math_class.delete()
        del self.tree.dict_of_nodes[self.node_id]
        if self.node_id in self.tree.nodes_to_expand:
            self.tree.nodes_to_expand.remove(self.node_id)
        for child in self.list_children:
            child.delete()

    def expand_node_with_action(self, action=None):
        # entscheide welche regel angewendet werden soll
        self.selected_action.append(action)
        self.selected_production.append(self.tree.grammar._productions[action])
        self.check_if_node_and_production_fit(self.selected_production[-1])
        rhs = copy.deepcopy(list(self.selected_production[-1].rhs()))
        rhs = [str(symbol) for symbol in rhs]
        self.prefix_to_syntax_tree(prefix=rhs)

    def check_if_node_and_production_fit(self, selected_production):
        if str(selected_production._lhs) != str(self.node_symbol):
            raise ValueError(
                f"When constructing the tree from a list of production index a error occurred. "
                f"Current node and the production rule selected at this node do not fit each other."
                f"The current node symbol is {self.node_symbol}."
                f"The selected production is {selected_production}")

    def count_nodes_in_tree(self, current_count=0):
        # goes through all nodes and its children should be the same as
        # len(syntax_tree.dict_of_nodes)
        current_count += 1
        for child in self.list_children:
            current_count = child.count_nodes_in_tree(current_count=current_count)
        return current_count


def geometric_sum(max_branching_factor, n):
    # sum_k=0^n q^k
    nominator = 1 - max_branching_factor ** (n + 1)
    denominator = 1 - max_branching_factor
    return nominator / denominator
