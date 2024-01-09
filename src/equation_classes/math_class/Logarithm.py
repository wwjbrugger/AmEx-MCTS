from src.equation_classes.node import Node
import numpy as np


class Logarithm():
    def __init__(self, node):
        self.num_child = 1
        self.node = node
        self.invertible = True
        self.neutral_element = 1

    def prefix_notation(self, call_node_id):
        if call_node_id == self.node.node_id:
            p_str = self.node.parent_node.math_class.prefix_notation(
                call_node_id=self.node.node_id)
            return p_str
        elif call_node_id == self.node.parent_node.node_id or call_node_id is None:
            c_0_str = self.node.list_children[0].math_class.prefix_notation(self.node.node_id)
            return f' log  {c_0_str} '
        elif call_node_id == self.node.list_children[0].node_id:
            p_str = self.node.parent_node.math_class.prefix_notation(
                call_node_id=self.node.node_id)
            return f' ** {p_str} 10   '

    def infix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            p_str = self.node.parent_node.math_class.infix_notation(call_node_id=self.node.node_id, kwargs=kwargs)
            return p_str
        elif call_node_id == self.node.parent_node.node_id or call_node_id is None:
            c_0_str = self.node.list_children[0].math_class.infix_notation(self.node.node_id, kwargs)
            return f' log ( {c_0_str} )'
        elif call_node_id == self.node.list_children[0].node_id:
            p_str = self.node.parent_node.math_class.infix_notation(call_node_id=self.node.node_id, kwargs=kwargs)
            return f' 10 **  {p_str}  '
    def evaluate_subtree(self, call_node_id, dataset, kwargs):
        c_0 = self.node.list_children[0].math_class.evaluate_subtree(self.node.node_id, dataset, kwargs)
        return np.log10(c_0, dtype=np.float64)

    def delete(self):
        pass
