import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from tqdm import tqdm



# https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU
def tsp_opt(points):
    """
    Dynamic programing solution for TSP - O(2^n*n^2)
    https://gist.github.com/mlalevic/6222750

    :param points: List of (x, y) points
    :return: Optimal solution
    """

    def length(x_coord, y_coord):
        return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))

    # Calculate all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # Initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx+1]), idx+1): (dist, [0, idx+1]) for idx, dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
            for j in S - {0}:
                # This will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                B[(S, j)] = min([(A[(S-{j}, k)][0] + all_distances[k][j], A[(S-{j}, k)][1] + [j])
                                 for k in S if k != 0 and k != j])
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    return np.asarray(res[1])


class TSPDataset(Dataset):
    """
    Random TSP dataset

    """
    def __init__(self, data_size, seq_len, solver=tsp_opt, solve=True, data_file_path=None):
        self.data_size = data_size
        self.seq_len = seq_len
        self.solve = solve
        self.solver = solver
        self.data_file_path = data_file_path

        if self.data_file_path is not None:
            self.data = self._load_data(self.data_file_path)
        else:
            self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['Points_List'][idx]).float()
        solution = torch.from_numpy(self.data['Solutions'][idx]).long() if self.solve else None

        sample = {'Points':tensor, 'Solution': solution}

        return sample

    def _load_data(self, data_file_path):
        num_line, num_points = self._get_data_shape(data_file_path)
        num_line = min(num_line, self.data_size)

        self.seqs = np.empty((num_line, num_points, 2))
        self.len_seq = np.empty(num_line, dtype=np.long)
        self.labels = np.empty((num_line, num_points), dtype=np.long)

        with open(data_file_path, 'r') as f:
            for idx in range(num_line):
                line = f.readline()
                if not line:
                    break
                self.seqs[idx, ...], self.len_seq[idx], self.labels[idx, ...] = self._get_data_from_line(line)

        return {'Points_List': self.seqs[idx, ...], 'Solutions': self.labels[idx, ...]}


    def _get_data_from_line(self, line):
        linesep =line.split()
        path = []
        pos = 0

        while linesep[pos] != 'output':
            path.append([float(linesep[pos]), float(linesep[pos+1])])
            pos += 2

        gt = [ int(idx)-1 for idx in linesep[pos+1:] ]
        return np.array(path), len(path), np.array(gt, dtype=np.long)[:-1]

    def _get_data_shape(self, path):
        num_lines = 0
        with open(path, 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                num_lines += 1

        with open(path, 'r') as f:
            line = f.readline()
            data = self._get_data_from_line(line)

        return num_lines, len(data[0])



    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        points_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data points %i/%i' % (i+1, self.data_size))
            points_list.append(np.random.random((self.seq_len, 2)))
        solutions_iter = tqdm(points_list, unit='solve')
        if self.solve:
            for i, points in enumerate(solutions_iter):
                solutions_iter.set_description('Solved %i/%i' % (i+1, len(points_list)))
                solutions.append(self.solver(points))
        else:
            solutions = None

        return {'Points_List': points_list, 'Solutions':solutions}

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec
