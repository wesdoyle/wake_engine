import numpy as np


class Node:
    """
    Represents a game state in which all of the valid moves for a player are represented
    """

    def __init__(self, position):
        self.position = position
        self.valid_moves = []

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def children(self):
        return []


def build_position_tree(initial_position) -> Node:
    pass


class PositionEvaluation:
    def __init__(self, node: Node):
        self.score = 0


class Negamax:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def search(self, node: Node, depth, color) -> PositionEvaluation:
        original_color = color

        if depth == 0 or node.is_leaf:
            return self.evaluate(node, original_color)

        max_score = -np.ninf

        for child in node.children:
            max_score = max(max_score, -self.search(child, depth - 1, not color).score)

        return max_score

    def evaluate(self, node: Node, color: bool) -> PositionEvaluation:
        return self.evaluator.evaluate(node, color)
