from collections import deque

import numpy as np
import torch
from skimage import color


def prepare_raw_frame(frame):
    return np.expand_dims(color.rgb2gray(frame), 0)


class Memory:
    def __init__(self, memory_size, history_images, image_dims):
        self.memory_size = memory_size
        self.image_dims = image_dims
        self.history_images = history_images

        self.frames = [self.get_empty_frame() for _ in range(self.history_images)]

        self.buffer = deque(maxlen=self.memory_size)

    def get_state(self, frame, to_tensor=False):
        frame = prepare_raw_frame(frame)
        out_np = np.append(np.array(self.frames[1:]), frame, axis=0).astype(np.float32)
        if to_tensor:
            return torch.tensor(out_np, dtype=torch.float32).unsqueeze(0)
        return out_np

    def get_empty_frame(self):
        return np.zeros(self.image_dims, dtype=np.float32)

    def add_experience(self, state_frame, action, reward, next_state_frame, done):
        state_frame = prepare_raw_frame(state_frame)
        next_state_frame = prepare_raw_frame(next_state_frame)
        self.frames.append(state_frame[0])
        if len(self.frames) == self.history_images + 1:
            self.frames = self.frames[1:]

        curr_state = np.array(self.frames, dtype=np.float32)

        next_state = np.append(np.array(self.frames[1:]), next_state_frame, axis=0).astype(np.float32)

        self.buffer.append({
            'state': curr_state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def clear(self):
        self.frames = [self.get_empty_frame() for _ in range(self.history_images)]
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.buffer[idx].items()}
