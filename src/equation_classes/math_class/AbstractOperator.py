from abc import ABC, abstractmethod


class AbstractOperator(ABC):

    @abstractmethod
    def __init__(self, node):
        self.num_child = None   # number of child nodes for the operator
        self.node = node
        self.invertible = None  # is the operation invertible
        self.node.node_symbol = '' # str representation of node

    @abstractmethod
    def prefix_notation(self, call_node_id, kwargs):
        """
        Specifies the node-level behavior for printing
         the syntax tree in prefix order.
        :param call_node_id: int
        :return: str
        """
        pass

    @abstractmethod
    def infix_notation(self, call_node_id, kwargs):
        """
        Specifies the node-level behavior for printing
         the syntax tree in infix order.
        :param call_node_id: int
        :param call_node_id: dict with node information
        :return: str
        """
        pass

    @abstractmethod
    def evaluate_subtree(self, call_node_id, dataset, kwargs):
        """
         Specifies the node-level behavior for evaluating
         the syntax tree.
        :param call_node_id:
        :param dataset:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def delete(self):
        """
        Specifies the node-level behavior for deleting
         the syntax tree.
        :return:
        """
