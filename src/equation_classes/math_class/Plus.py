from src.equation_classes.node import Node
import numpy as np


class Plus():
    def __init__(self, node):
        self.num_child = 2
        self.node = node
        self.invertible = True
        self.neutral_element = 0

    def prefix_notation(self, call_node_id):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.prefix_notation(
                call_node_id=self.node.node_id)
        elif call_node_id == self.node.parent_node.node_id or call_node_id is None:
            return f' + {self.node.list_children[0].math_class.prefix_notation(self.node.node_id)}' \
                   f' {self.node.list_children[1].math_class.prefix_notation(self.node.node_id)} '
        elif call_node_id == self.node.list_children[0].node_id:
            return f' - {self.node.parent_node.math_class.prefix_notation(self.node.node_id)}' \
                   f' {self.node.list_children[1].math_class.prefix_notation(self.node.node_id)} '
        elif call_node_id == self.node.list_children[1].node_id:
            return f' - {self.node.parent_node.math_class.prefix_notation(self.node.node_id)}' \
                   f' {self.node.list_children[0].math_class.prefix_notation(self.node.node_id)} '

    def infix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.parent_node.math_class.infix_notation(call_node_id=self.node.node_id, kwargs=kwargs)
        elif call_node_id == self.node.parent_node.node_id or call_node_id is None:
            return f'( {self.node.list_children[0].math_class.infix_notation(self.node.node_id, kwargs)} + ' \
                   f' {self.node.list_children[1].math_class.infix_notation(self.node.node_id, kwargs)} ) '
        elif call_node_id == self.node.list_children[0].node_id:
            return f'(  {self.node.parent_node.math_class.infix_notation(self.node.node_id, kwargs)} - ' \
                   f' {self.node.list_children[1].math_class.infix_notation(self.node.node_id, kwargs)} ) '
        elif call_node_id == self.node.list_children[1].node_id:
            return f'( {self.node.parent_node.math_class.infix_notation(self.node.node_id, kwargs)} - ' \
                   f' {self.node.list_children[0].math_class.infix_notation(self.node.node_id, kwargs)} ) '

    def evaluate_subtree(self, call_node_id, dataset, kwargs):
        c_0 = self.node.list_children[0].math_class.evaluate_subtree(self.node.node_id, dataset, kwargs)
        c_1 = self.node.list_children[1].math_class.evaluate_subtree(self.node.node_id, dataset, kwargs)
        return np.add(c_0, c_1, dtype=np.float64)
