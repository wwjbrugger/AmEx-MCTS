import numpy as np
from src.equation_classes.math_class.AbstractOperator import AbstractOperator

class Terminal(AbstractOperator):
    def __init__(self, node):
        self.num_child = 0
        self.node = node
        self.invertible = True

    def prefix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.prefix_notation(
                call_node_id=self.node.node_id, kwargs=kwargs)
        else:
            return f"{self.node.node_symbol}"

    def infix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.infix_notation(
                call_node_id=self.node.node_id,
                kwargs=kwargs
            )
        else:
            return f"{self.node.node_symbol}"

    def evaluate_subtree(self, call_node_id, dataset, kwargs):
        symbol = self.node.node_symbol
        if symbol in dataset.columns:
            return dataset.loc[:, symbol].to_numpy(dtype=np.float64)
        else:
            return np.full(
                shape=dataset.shape[0],
                fill_value=symbol,
                dtype=np.float64
            )

    def delete(self):
        pass
