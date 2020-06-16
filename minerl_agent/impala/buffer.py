from collections import defaultdict
from typing import List

import numpy as np
import tensorflow as tf


class Buffer(object):
    def __init__(self, size, batch_size_in, batch_size_out, unroll_length, shapes: List[tuple], dtypes: List[tf.DType]):
        assert size % batch_size_in == 0, f"buffer size must be a multiple of batch_size_in, {size} and {batch_size_in} given"
        self.size = size
        self.batch_size_in = batch_size_in
        self.batch_size_out = batch_size_out
        self.unroll_length = unroll_length
        self.shapes = list(shapes)
        self.dtypes = [dtype.as_numpy_dtype for dtype in dtypes]
        self.buffers = [np.empty((size,) + tuple(shape), dtype) for shape, dtype in zip(self.shapes, self.dtypes)]
        self.next_idx = 0
        self.num_in_buffer = 0

    def __len__(self):
        return self.num_in_buffer

    def put(self, trajectories):
        for buffer, features in zip(self.buffers, trajectories):
            buffer[self.next_idx:self.next_idx+self.batch_size_in] = features
        self.next_idx = (self.next_idx + self.batch_size_in) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + self.batch_size_in)

    def get(self):
        idx = np.random.randint(0, self.num_in_buffer, self.batch_size_out)
        return [buffer[idx] for buffer in self.buffers], idx
