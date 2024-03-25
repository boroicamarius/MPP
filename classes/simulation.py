from typing import Sequence, Tuple

import numpy as np


class Map:

    def __init__(self, num_robots: int) -> None:
        self.num_robots = num_robots

    def load(self, filename: str):
        with open(filename, 'r') as f:
            lines = f.readlines()
            loaded_map = [[1.0 if e == '1' else 0.0 for e in line.replace('\n', '')] for line in lines]
            
        self.map = np.array(loaded_map)
        self.width = self.map.shape[1]
        self.height = self.map.shape[0]

        self.robots_positions = []
        self.targets_positions = []

        for _ in range(self.num_robots):
            path = self.__generate_positions(self.robots_positions, self.map)
            self.robots_positions.append(path)

        for _ in range(self.num_robots):
            path = self.__generate_positions(self.targets_positions, self.map)
            self.targets_positions.append(path)

        return self

    def move_robots(self, actions: Sequence[int]) -> None:
        moves = [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ]

        results = [0 for _ in range(self.num_robots)]
        for index, action in enumerate(actions):
            px, py = self.robots_positions[index]
            mx, my = moves[action]
            cx, cy = px + mx, py + my

            if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height or self.map[cx][cy] == 1:
                results[index] = 1
                continue

            tx, ty = self.targets_positions[index]
            if cx == tx and cy == ty:
                results[index] = 2

            self.robots_positions[index] = (cx, cy)
        return results

    def get_map(self):
        return self.map

    def get_flattened_map(self):
        return self.map.flatten()

    def get_area(self):
        return self.width * self.height

    def get_nr_robots(self):
        return self.num_robots

    def get_paths(self):
        return np.array([
            [*self.robots_positions[index], *self.targets_positions[index]]
            for index in range(self.num_robots)
        ])

    def __generate_positions(self, previous: Sequence[Tuple[int, int]], map: np.ndarray):
        positions = np.random.randint(map.shape[0]), np.random.randint(map.shape[1])
        while positions in previous:
            positions = np.random.randint(map.shape[0]), np.random.randint(map.shape[1])
        return positions
