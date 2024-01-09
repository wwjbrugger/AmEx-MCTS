import numpy as np


class Y():
    def __init__(self, node):
        self.num_child = 1
        self.node = node
        self.invertible = True

    def prefix_notation(self, call_node_id):
        if call_node_id == self.node.node_id:
            return self.node.list_children[0].math_class.prefix_notation(
                call_node_id=self.node.node_id)
        else:
            return f"{self.node.node_symbol}"

    def infix_notation(self, call_node_id, kwargs):
        if call_node_id == self.node.node_id:
            return self.node.list_children[0].math_class.infix_notation(call_node_id=self.node.node_id, kwargs=kwargs)
        else:
            return f"{self.node.node_symbol}"

    def evaluate_subtree(self, call_node_id, dataset, kwargs):
        return self.node.list_children[0].math_class.evaluate_subtree(self.node.node_id, dataset, kwargs)

    def delete(self):
        pass
