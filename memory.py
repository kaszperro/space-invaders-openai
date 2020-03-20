from collections import deque

import torch
from torchvision import transforms


# import torchvision.transforms


class Memory:
    def __init__(self, memory_size, history_images, image_dims, device=torch.device('cpu')):
        self.memory_size = memory_size
        self.image_dims = image_dims
        self.history_images = history_images
        self.device = device

        self.frames = [self.get_empty_frame() for _ in range(self.history_images)]
        self.buffer = deque(maxlen=self.memory_size)

        self.to_pil = transforms.ToPILImage()
        self.to_grayscale = transforms.Grayscale()
        self.resize = transforms.Resize(self.image_dims)
        self.to_tensor = transforms.ToTensor()

    def prepare_raw_frame(self, frame, to_numpy=False):
        frame = self.to_pil(frame)
        frame = self.to_grayscale(frame)
        frame = self.resize(frame)
        tensor = self.to_tensor(frame).to(self.device)

        if to_numpy:
            return tensor.cpu().numpy()
        return tensor

    def get_state(self, frame, to_numpy=False, as_batch=True):

        with torch.no_grad():
            frame = self.prepare_raw_frame(frame)
            out_tensor = torch.cat([*self.frames[1:], frame], dim=0)
            if as_batch:
                out_tensor = out_tensor.unsqueeze(0)
            if to_numpy:
                return out_tensor.cpu().numpy()
            return out_tensor

    def get_empty_frame(self):
        frame_dims = (1, self.image_dims[0], self.image_dims[1])
        return torch.zeros(frame_dims, dtype=torch.float32, device=self.device)

    def add_experience(self, state_frame, action, reward, next_state_frame, done):
        with torch.no_grad():
            state_frame = self.prepare_raw_frame(state_frame)
            next_state_frame = self.prepare_raw_frame(next_state_frame)

            self.frames.append(state_frame)
            if len(self.frames) == self.history_images + 1:
                self.frames = self.frames[1:]

            curr_state = torch.cat(self.frames, dim=0)
            next_state = torch.cat([*self.frames[1:], next_state_frame], dim=0)

            self.buffer.append({
                'state': curr_state,
                'action': action.to(self.device, dtype=torch.long),
                # torch.tensor(action, dtype=torch.long, device=self.store_device),
                'reward': torch.tensor(reward, dtype=torch.float32, device=self.device),
                'next_state': next_state,
                'done': torch.tensor(done, dtype=torch.bool, device=self.device)
            })

    def clear(self):
        self.frames = [self.get_empty_frame() for _ in range(self.history_images)]
        self.buffer.clear()

    def __len__(self):
        # return self.memory_size
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]  # {k: v.to(self.train_device) for k, v in self.buffer[idx].items()}
