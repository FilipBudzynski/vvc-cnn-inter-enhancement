"""
Blackfyre Dataset - fixed for 256x256 with min_size filter
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path


class BlackfyreDataset(Dataset):
    def __init__(self, data_dir: str, patch_size: int = 256, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.split = split
        
        self.videos = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        # Filter videos smaller than patch_size
        valid_videos = []
        for video_dir in self.videos:
            frames = sorted(video_dir.glob("poc_*.pt"))
            if not frames:
                continue
            sample = torch.load(frames[0], map_location='cpu')
            h, w = sample['decoded'].shape[-2:]
            if h >= patch_size and w >= patch_size:
                valid_videos.append(video_dir)
            else:
                print(f"Skipping {video_dir.name} (size {h}x{w} < {patch_size}x{patch_size})")
        
        self.videos = valid_videos
        print(f"Valid videos for {split}: {len(self.videos)}")
        
        self.frame_triplets = []
        for video_dir in self.videos:
            frames = sorted(video_dir.glob("poc_*.pt"))
            
            if len(frames) < 3:
                continue
                
            for poc in range(1, len(frames) - 1):
                self.frame_triplets.append({
                    'prev': frames[poc - 1],
                    'curr': frames[poc],
                    'next': frames[poc + 1],
                    'video': video_dir.name,
                    'poc': poc
                })
        
        print(f"Loaded {len(self.frame_triplets)} frame triplets for {split}")
        
    def __len__(self):
        return len(self.frame_triplets)
    
    def __getitem__(self, idx):
        triplet = self.frame_triplets[idx]
        
        prev_data = torch.load(triplet['prev'], map_location='cpu')
        curr_data = torch.load(triplet['curr'], map_location='cpu')
        next_data = torch.load(triplet['next'], map_location='cpu')
        
        prev_frame = prev_data['decoded']
        curr_frame = curr_data['decoded']
        next_frame = next_data['decoded']
        
        original = curr_data['original']
        features = curr_data['features']
        
        # Fixed crop from top-left
        curr_frame = curr_frame[:, :self.patch_size, :self.patch_size]
        prev_frame = prev_frame[:, :self.patch_size, :self.patch_size]
        next_frame = next_frame[:, :self.patch_size, :self.patch_size]
        original = original[:, :self.patch_size, :self.patch_size]
        features = features[:, :self.patch_size, :self.patch_size]
        
        return (prev_frame, curr_frame, next_frame), original, features, triplet['video']
