"""
Blackfyre Dataset - loads neighboring frames with enhanced metadata

Loads: F-1, F0, F+1 frames + enhanced metadata (16 channels)
"""

import random
import torch
from torch.utils.data import Dataset
from pathlib import Path


class BlackfyreDataset(Dataset):
    """
    Dataset that loads neighboring frames with enhanced metadata
    for Blackfyre temporal attention network
    """
    
    def __init__(
        self,
        data_dir: str = "data/precomputed",
        patch_size: int = 132,
        split: str = "train",
        train_ratio: float = 0.8,
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.split = split
        
        # Group frames by video and check sizes efficiently
        self.video_frames = {}
        self.video_sizes = {}
        for video_dir in self.data_dir.iterdir():
            if video_dir.is_dir():
                frames = sorted(video_dir.glob("poc_*.pt"), 
                             key=lambda x: int(x.stem.split("_")[1]))
                if len(frames) >= 3:
                    # Load one frame to get size (cached per video)
                    test_data = torch.load(frames[0], weights_only=True)
                    _, h, w = test_data["decoded"].shape
                    self.video_frames[video_dir.name] = frames
                    self.video_sizes[video_dir.name] = (h, w)
        
        # Create valid indices - filter videos too small for patch_size
        self.samples = []
        for video_name, frames in self.video_frames.items():
            h, w = self.video_sizes[video_name]
            if h >= self.patch_size and w >= self.patch_size:
                for i in range(1, len(frames) - 1):
                    self.samples.append((video_name, frames[i-1], frames[i], frames[i+1]))
        
        # Shuffle and split
        random.seed(42)
        random.shuffle(self.samples)
        
        n = len(self.samples)
        if split == "train":
            self.samples = self.samples[:int(n * train_ratio)]
        elif split == "val":
            self.samples = self.samples[int(n * train_ratio):int(n * (train_ratio + 0.1))]
        else:
            self.samples = self.samples[int(n * (train_ratio + 0.1)):]
        
        print(f"Loaded {len(self.samples)} frame triplets for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_name, prev_pt, curr_pt, next_pt = self.samples[idx]
        
        # Load frames
        prev_data = torch.load(prev_pt, weights_only=True)
        curr_data = torch.load(curr_pt, weights_only=True)
        next_data = torch.load(next_pt, weights_only=True)
        
        # Extract YUV frames
        prev_frame = prev_data["decoded"]
        curr_frame = curr_data["decoded"]
        next_frame = next_data["decoded"]
        original = curr_data["original"]
        
        # Enhanced features (now 16 channels instead of 8)
        features = curr_data["features"]  # [16, H, W]
        
        # Random crop
        _, h, w = curr_frame.shape
        th, tw = self.patch_size, self.patch_size
        
        if h > th:
            top = random.choice(range(0, h - th + 1, 8))
        else:
            top = 0
        if w > tw:
            left = random.choice(range(0, w - tw + 1, 8))
        else:
            left = 0
        
        prev_frame = prev_frame[:, top:top+th, left:left+tw]
        curr_frame = curr_frame[:, top:top+th, left:left+tw]
        next_frame = next_frame[:, top:top+th, left:left+tw]
        original = original[:, top:top+th, left:left+tw]
        features = features[:, top:top+th, left:left+tw]
        
        return (prev_frame, curr_frame, next_frame), original, features, {
            "video": video_name,
            "poc": curr_data["poc"]
        }
