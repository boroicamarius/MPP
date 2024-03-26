from typing import Sequence, Tuple
from environments.utils import add_coords
import numpy as np
import torch


class Map:

    def __init__(self, num_robots: int, context_window: int):
        self.num_robots = num_robots
        self.moves = [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ]

        self.states_rewards = []
        self.states_map = []
        self.states_path = []
        self.states_actions = []
        self.context_window = context_window

    def load(self, filename: str):
        with open(filename, 'r') as f:
            lines = f.readlines()
            loaded_map = [[1.0 if e == '1' else 0.0 for e in line.replace('\n', '')] for line in lines]

        self.map = loaded_map
        self.width = len(self.map[0])
        self.height = len(self.map)

        self.robots_positions = []
        self.targets_positions = []

        for _ in range(self.num_robots):
            path = self.__generate_positions(self.robots_positions, self.map)
            self.robots_positions.append(path)

        for _ in range(self.num_robots):
            path = self.__generate_positions(self.targets_positions, self.map)
            self.targets_positions.append(path)

        self.states_rewards = [[0]]
        self.states_map = [self.map]
        self.states_path = [self.get_paths()]
        self.states_actions = [[0] * self.num_robots]
        return self

    def get_dims(self):
        return (self.height, self.width)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def move_robots(self, actions: Sequence[int]):
        """
        TODO:
            - What happens if robots collide after actions took place?
            - make everything using numpy
        """

        results = [0 for _ in range(self.num_robots)]
        future_positions = [add_coords(self.robots_positions[index], self.moves[int(actions[index])]) for index in
                            range(self.num_robots)]

        for index, position in enumerate(future_positions):
            y, x = position

            if x >= self.width or y >= self.height or x < 0 or y < 0:
                results[index] = -1
                continue

            if future_positions.count(position) > 1:
                results[index] = -2
                continue

            if self.map[y][x] == 1:
                results[index] = -3
                continue

            if future_positions[index] == self.targets_positions[index]:
                results[index] = 1
                continue

        for index in range(self.num_robots):
            if results[index] < 0:
                continue

            self.robots_positions[index] = future_positions[index]

        self.states_actions.append(actions)
        self.states_map.append(self.map)
        self.states_rewards.append([results.count(1) - 2 * results.count(-2) - 3 * results.count(-3)])
        self.states_path.append(self.get_paths())

        self.states_actions = self.states_actions[-self.context_window:]
        self.states_map = self.states_map[-self.context_window:]
        self.states_rewards = self.states_rewards[-self.context_window:]
        self.states_path = self.states_path[-self.context_window:]

    def get_states(self):
        return (torch.tensor(self.states_rewards, dtype=torch.float32, device='cuda'),
                torch.tensor(self.states_map, dtype=torch.float32, device='cuda').flatten(-2, -1),
                torch.tensor(self.states_path, dtype=torch.float32, device='cuda').flatten(-2, -1),
                torch.tensor(self.states_actions, dtype=torch.float32, device='cuda'))

    def get_robots(self):
        return self.robots_positions

    def get_targets(self):
        return self.targets_positions

    def get_map(self):
        return self.map

    def get_area(self):
        return self.width * self.height

    def get_nr_robots(self):
        return self.num_robots

    def get_paths(self):
        return [
            [*self.robots_positions[index], *self.targets_positions[index]]
            for index in range(self.num_robots)
        ]

    def __generate_positions(self, previous: Sequence[Tuple[int, int]], map):
        y, x = np.random.randint(self.height), np.random.randint(self.width)
        while [y, x] in previous or map[y][x] == 1:
            y, x = np.random.randint(self.height), np.random.randint(self.width)
        return [y, x]
