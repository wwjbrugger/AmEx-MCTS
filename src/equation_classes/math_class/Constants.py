import numpy as np
from src.equation_classes.math_class.AbstractOperator import AbstractOperator


class Constants(AbstractOperator):
    def __init__(self, node):
        self.num_child = 0
        self.node = node
        self.invertible = True
        self.node.node_symbol = f"c_{self.node.tree.num_constants_in_complete_tree}"
        self.node.tree.constants_in_tree[self.node.node_symbol] = {
            'node_id': self.node.node_id,
            'value': None
        }
        self.node.tree.num_constants_in_complete_tree += 1
        if self.node.tree.num_constants_in_complete_tree > self.node.tree.args.max_constants_in_tree:
            self.node.tree.max_constants_reached = True

    def prefix_notation(self, call_node_id):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.prefix_notation(
                call_node_id=self.node.node_id)
        else:
            return 'c'

    def infix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.infix_notation(call_node_id=self.node.node_id, kwargs=kwargs)
        elif len(list(kwargs.keys())):
            return f"{kwargs[self.node.node_symbol]['value']:.4f}"
        else:
            return f"{self.node.node_symbol}"


    def evaluate_subtree(self, call_node_id, dataset, kwargs):
        symbol = self.node.node_symbol
        return np.full(
            shape=dataset.shape[0],
            fill_value=kwargs[symbol]['value'],
            dtype=np.float64
        )

    def delete(self):
        self.node.tree.num_constants_in_complete_tree -= 1
