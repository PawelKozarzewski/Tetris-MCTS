import numpy as np
from numba import njit
from time import perf_counter

from evaluator import Evaluator
from tqueue import TQueue
from utility import get_legal_actions, is_game_over, get_next_state


class MctsNode:
    evaluator: Evaluator = None
    t_queue: TQueue = None
    c_param = 0.0

    def __init__(self, state: np.ndarray = None, t_index: int = 0, parent=None, parent_action=None):
        self.state = state
        self.t_index = t_index
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits = 0
        self.qavg = np.random.uniform(low=0.0, high=1.0, size=None)
        self.__children_data = None

    def initialize_state(self):
        t_type = MctsNode.t_queue.get(self.t_index)
        self.state = get_next_state(self.parent.state, t_type, self.parent_action)

    def select(self):
        current_node = self
        while current_node.is_fully_expanded():
            current_node = current_node.best_child(MctsNode.c_param)
        return current_node

    def is_terminal_node(self):
        return is_game_over(self.state)

    def is_fully_expanded(self):
        return len(self.children) != 0

    def expand(self, q_values=None):
        if self.is_terminal_node():
            return

        legal_actions = get_legal_actions(self.state)
        for action in legal_actions:
            self.children.append(MctsNode(t_index=self.t_index + 1, parent=self, parent_action=action))
        self.__children_data = np.zeros((len(self.children), 2), dtype='float32')

        if q_values is not None:
            for i in range(len(self.children)):
                self.children[i].qavg = q_values[i]

    def best_child(self, c_param: float = 0.0):
        for i, c in enumerate(self.children):
            self.__children_data[i][0] = c.qavg
            self.__children_data[i][1] = c.visits
        return self.children[MctsNode.calc_weights(self.__children_data, self.visits, c_param)]

    @staticmethod
    @njit(cache=True)
    def calc_weights(ch_data: np.ndarray, selfn: int, c_param: float) -> int:
        choices_weights = ch_data[:, 0] + c_param * np.sqrt(np.log(selfn) / (ch_data[:, 1] + 1))
        return np.argmax(choices_weights)

    def evaluation(self) -> int:
        return MctsNode.evaluator.evaluate(self.state, self.t_index)

    def backpropagate(self, q: float):
        self.visits += 1
        self.qavg += (q - self.qavg) / self.visits

        if self.parent is not None:
            self.parent.backpropagate(q)


class MctsTree:
    def __init__(self, node: MctsNode = None):
        self.root = node

    def set_root(self, node: MctsNode) -> None:
        self.root = node

    @staticmethod
    def decrement_indices(node: MctsNode):
        if node is not None:
            for c in node.children:
                MctsTree.decrement_indices(c)
            node.t_index -= 1

    def best_action(self, simulation_no: int, time_budget: float = None) -> MctsNode:
        sim_start_time = perf_counter()

        # if time_budget is not None:
        #     simulation_no = np.inf

        while self.root.visits < simulation_no:
            if time_budget is not None:
                if perf_counter() - sim_start_time >= time_budget:
                    break

            node = self.root.select()
            if node.state is None:
                node.initialize_state()

            q, q_vector = node.evaluation()
            if not node.is_terminal_node():
                node.expand(q_vector)

            node.backpropagate(q)

        return self.root.best_child()
