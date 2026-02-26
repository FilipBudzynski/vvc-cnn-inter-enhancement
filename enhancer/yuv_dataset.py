import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os

from .config import SubDatasetConfig, DatasetConfig, DataloaderConfig
from .features_parser.parser import VTMParser
from .features_generator.generator import FeatureMapGenerator


class YUVFrame:
    """Represents a single YUV frame"""
    def __init__(self, y_data: np.ndarray, u_data: np.ndarray, v_data: np.ndarray):
        self.y = y_data
        self.u = u_data  
        self.v = v_data
        
    def get_rgb_tensor(self, frame_size: Tuple[int, int]) -> torch.Tensor:
        """Convert YUV to RGB tensor"""
        # Stack YUV channels
        yuv = np.stack([self.y, self.u, self.v], axis=0)
        
        # Convert to RGB (simplified YUV to RGB conversion)
        # Using cv2 for conversion
        y_uv = np.stack([self.y, self.u, self.v], axis=-1)
        rgb = cv2.cvtColor(y_uv, cv2.COLOR_YUV2RGB)
        rgb = rgb.transpose(2, 0, 1)  # HWC to CHW
        
        return torch.from_numpy(rgb).float() / 255.0


class YUVLoader:
    """Loads YUV files and provides frame access"""
    
    def __init__(self, yuv_file: str, width: int, height: int):
        self.yuv_file = yuv_file
        self.width = width
        self.height = height
        self.frame_size = width * height
        
        # Calculate file positions for frames
        self.y_size = self.frame_size
        self.u_size = self.frame_size // 4
        self.v_size = self.frame_size // 4
        
    def load_frame(self, frame_idx: int) -> YUVFrame:
        """Load specific frame from YUV file"""
        with open(self.yuv_file, 'rb') as f:
            # Y plane
            y_offset = frame_idx * self.frame_size
            f.seek(y_offset)
            y_data = np.frombuffer(f.read(self.y_size), dtype=np.uint8)
            y_data = y_data.reshape(self.height, self.width)
            
            # U plane  
            u_offset = self.frame_size * self.height + frame_idx * self.u_size
            f.seek(u_offset)
            u_data = np.frombuffer(f.read(self.u_size), dtype=np.uint8)
            u_data = u_data.reshape(self.height // 2, self.width // 2)
            
            # V plane
            v_offset = self.frame_size * self.height + self.frame_size * self.height // 4 + frame_idx * self.v_size
            f.seek(v_offset)  
            v_data = np.frombuffer(f.read(self.v_size), dtype=np.uint8)
            v_data = v_data.reshape(self.height // 2, self.width // 2)
            
            # Upsample U and V to match Y dimensions  
            u_upsampled = np.kron(u_data, np.ones((2, 2)))
            v_upsampled = np.kron(v_data, np.ones((2, 2)))
            
        return YUVFrame(y_data, u_upsampled, v_upsampled)


class YUVChunkDataset(torch.utils.data.Dataset):
    """Dataset for YUV chunks with VTM features"""
    
    def __init__(
        self,
        files: List[Path],
        csv_files: Dict[str, Path],
        chunk_size: Tuple[int, int],
        frame_size: Tuple[int, int],
        parser,
        feature_generator,
        vtm_cache: Dict
    ):
        self.files = files
        self.csv_files = csv_files
        self.chunk_size = chunk_size
        self.frame_size = frame_size
        self.parser = parser
        self.feature_generator = feature_generator
        self.vtm_cache = vtm_cache
        
        # Generate all chunks
        self.chunks = self._generate_chunks()
    
    def _generate_chunks(self):
        """Generate all chunks from files"""
        chunks = []
        
        for yuv_file in self.files:
            # Load VTM features
            csv_file = self.csv_files.get(yuv_file.stem)
            if csv_file and csv_file not in self.vtm_cache:
                with open(csv_file, 'r') as f:
                    content = f.readlines()
                    self.vtm_cache[csv_file] = self.parser.parse(content)
            
            vtm_features = self.vtm_cache.get(csv_file, {}) if csv_file else {}
            
            # Get YUV loader
            loader = YUVLoader(str(yuv_file), *self.frame_size)
            
            # Extract chunks from first few frames (for demo)
            num_frames = min(10, 100)  # Limit frames for memory
            for frame_idx in range(num_frames):
                try:
                    frame = loader.load_frame(frame_idx)
                    frame_chunks = self._extract_chunks_from_frame(frame, vtm_features, frame_idx, yuv_file)
                    chunks.extend(frame_chunks)
                except Exception as e:
                    print(f"Error loading frame {frame_idx} from {yuv_file}: {e}")
                    continue
        
        return chunks
    
    def _extract_chunks_from_frame(self, frame: YUVFrame, vtm_features: Dict, frame_idx: int, yuv_file: Path):
        """Extract chunks from a frame"""
        h, w = self.frame_size
        chunk_h, chunk_w = self.chunk_size
        
        chunks = []
        
        # Extract frame-level VTM features
        frame_features = vtm_features.get(frame_idx, {})
        
        # Generate feature maps for this frame
        if frame_features:
            feature_maps = self.feature_generator.generate_maps_for_frame(frame_features)
        else:
            feature_maps = {}
        
        # Extract chunks
        num_chunks_h = h // chunk_h
        num_chunks_w = w // chunk_w
        
        for i in range(num_chunks_h):
            for j in range(num_chunks_w):
                # Extract chunk from frame
                y_chunk = frame.y[i*chunk_h:(i+1)*chunk_h, j*chunk_w:(j+1)*chunk_w]
                u_chunk = frame.u[i*chunk_h:(i+1)*chunk_h, j*chunk_w:(j+1)*chunk_w] 
                v_chunk = frame.v[i*chunk_h:(i+1)*chunk_h, j*chunk_w:(j+1)*chunk_w]
                
                # Convert to RGB tensor
                yuv_chunk = YUVFrame(y_chunk, u_chunk, v_chunk)
                rgb_tensor = yuv_chunk.get_rgb_tensor(self.chunk_size)
                
                # Extract VTM features for this chunk region
                chunk_vtm_features = self._extract_vtm_chunk_features(feature_maps, i*chunk_h, j*chunk_w, chunk_h, chunk_w)
                
                # Create metadata tensor (similar to original config)
                metadata = torch.zeros(6)  # Default metadata size
                
                chunks.append({
                    'chunk': rgb_tensor,
                    'metadata': metadata,
                    'vtm_features': chunk_vtm_features,
                    'path': str(yuv_file),
                    'frame_idx': frame_idx,
                    'position': (j*chunk_w, i*chunk_h)
                })
        
        return chunks
    
    def _extract_vtm_chunk_features(self, feature_maps: Dict, y_start: int, x_start: int, h: int, w: int):
        """Extract VTM features for chunk region"""
        if not feature_maps:
            return torch.zeros((6, h, w))  # Return 6 channels (QP, PredMode, Depth, MVL0_X, MVL0_Y, MVL1_X, MVL1_Y)
        
        chunk_features = []
        feature_order = ['QP', 'PredMode', 'Depth', 'MVL0_X', 'MVL0_Y', 'MVL1_X', 'MVL1_Y']
        
        for feature_name in feature_order:
            if feature_name in feature_maps:
                feature_map = feature_maps[feature_name]
                # Extract chunk region
                chunk_data = feature_map[y_start:y_start+h, x_start:x_start+w]
                chunk_features.append(torch.from_numpy(chunk_data).float())
            else:
                chunk_features.append(torch.zeros((h, w)))
        
        # Stack to create channels x height x width tensor
        result = torch.stack(chunk_features[:6], dim=0)  # Limit to 6 channels
        
        # Pad with zeros if not enough features
        if result.shape[0] < 6:
            padding = torch.zeros((6 - result.shape[0], h, w))
            result = torch.cat([result, padding], dim=0)
        
        return result
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk_data = self.chunks[idx]
        
        return {
            'chunk': chunk_data['chunk'],
            'metadata': chunk_data['metadata'],
            'vtm_features': chunk_data['vtm_features'],
            'path': chunk_data['path'],
            'frame_idx': chunk_data['frame_idx'],
            'position': chunk_data['position']
        }


class YUVVTMDataset(pl.LightningDataModule):
    """
    Dataset that loads YUV files with VTM decoder statistics
    """
    
    def __init__(
        self,
        dataset_config: DatasetConfig,
        dataloader_config: DataloaderConfig,
        data_dir: str = "output",
        frame_size: Tuple[int, int] = (1920, 1080),
        test_full_frames: bool = False,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.data_dir = Path(data_dir)
        self.frame_size = frame_size
        self.chunk_size = (dataset_config.train.chunk_height, dataset_config.train.chunk_width)
        self.test_full_frames = test_full_frames
        
        # Initialize parser and generator
        self.parser = VTMParser()
        self.feature_generator = FeatureMapGenerator(*self.frame_size)
        
        # Find YUV and CSV files
        self.yuv_files = sorted(list(Path(data_dir).glob("*.yuv")))
        self.csv_files = {f.stem: f for f in Path(data_dir).parent.glob("decoded/*.csv")}
        
        print(f"Found {len(self.yuv_files)} YUV files")
        print(f"Found {len(self.csv_files)} CSV files")
        
        # Cache for loaded data
        self.yuv_loaders = {}
        self.vtm_cache = {}
        
    def setup(self, stage=None):
        """Setup dataset for training/validation/test"""
        if stage == "fit":
            # Use first 80% for training, 20% for validation
            num_files = len(self.yuv_files)
            train_split = int(num_files * 0.8)
            
            self.train_files = self.yuv_files[:train_split]
            self.val_files = self.yuv_files[train_split:]
            
            # Create datasets
            self.dataset_train = self._create_dataset(self.train_files)
            self.dataset_val = self._create_dataset(self.val_files)
            
            print(f"Training files: {len(self.train_files)}")
            print(f"Validation files: {len(self.val_files)}")
            
        if stage in ("test", "predict"):
            self.test_files = self.yuv_files
            self.dataset_test = self._create_dataset(self.test_files)
        
    def _create_dataset(self, files: List[Path]):
        """Create dataset from file list"""
        return YUVChunkDataset(
            files=files,
            csv_files=self.csv_files,
            chunk_size=self.chunk_size,
            frame_size=self.frame_size,
            parser=self.parser,
            feature_generator=self.feature_generator,
            vtm_cache=self.vtm_cache
        )

    def load_vtm_features(self, yuv_file: Path) -> Optional[Dict]:
        """Load VTM features for a YUV file"""
        csv_file = self.csv_files.get(yuv_file.stem)
        if not csv_file:
            return None
            
        if csv_file not in self.vtm_cache:
            with open(csv_file, 'r') as f:
                content = f.readlines()
                self.vtm_cache[csv_file] = self.parser.parse(content)
        
        # Generate feature maps for each frame
        features_by_frame = self.vtm_cache[csv_file]
        enhanced_features = {}
        
        for frame_idx, tokens in features_by_frame.items():
            enhanced_features[frame_idx] = self.feature_generator.generate_maps_for_frame(tokens)
        
        return enhanced_features
        
    def extract_chunks_from_frame(self, frame: YUVFrame, vtm_features: Dict, frame_idx: int):
        """Extract multiple chunks from a full frame"""
        h, w = self.frame_size
        chunk_h, chunk_w = self.chunk_size
        
        # Calculate number of chunks
        num_chunks_h = h // chunk_h
        num_chunks_w = w // chunk_w
        
        chunks = []
        for i in range(num_chunks_h):
            for j in range(num_chunks_w):
                # Extract chunk from frame
                y_chunk = frame.y[i*chunk_h:(i+1)*chunk_h, j*chunk_w:(j+1)*chunk_w]
                u_chunk = frame.u[i*chunk_h:(i+1)*chunk_h, j*chunk_w:(j+1)*chunk_w] 
                v_chunk = frame.v[i*chunk_h:(i+1)*chunk_h, j*chunk_w:(j+1)*chunk_w]
                
                # Convert to RGB tensor
                yuv_chunk = YUVFrame(y_chunk, u_chunk, v_chunk)
                rgb_tensor = yuv_chunk.get_rgb_tensor(self.chunk_size)
                
                # Extract VTM features for this chunk region
                vtm_chunk_features = self.extract_vtm_chunk_features(
                    vtm_features, i*chunk_h, j*chunk_w, chunk_h, chunk_w
                )
                
                chunks.append({
                    'rgb': rgb_tensor,
                    'vtm_features': vtm_chunk_features,
                    'position': (j*chunk_w, i*chunk_h),
                    'frame_idx': frame_idx
                })
        
        return chunks
        
    def extract_vtm_chunk_features(self, vtm_features: Dict, y_start: int, x_start: int, h: int, w: int):
        """Extract VTM features for a specific chunk region"""
        if not vtm_features:
            return torch.zeros((1, h, w))
            
        chunk_features = []
        feature_names = sorted(vtm_features.keys())
        
        for feature_name in feature_names:
            if feature_name in vtm_features:
                feature_map = vtm_features[feature_name]
                # Extract chunk region
                chunk_data = feature_map[
                    y_start:y_start+h, x_start:x_start+w
                ]
                chunk_features.append(torch.from_numpy(chunk_data).float())
        
        if chunk_features:
            return torch.stack(chunk_features, dim=0)
        else:
            return torch.zeros((1, h, w))
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.dataloader_config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.dataloader_config.val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.dataloader_config.test_batch_size if not self.test_full_frames else 1,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
    
    def predict_dataloader(self):
        return self.test_dataloader()


def create_yuv_vtm_dataloader(
    data_dir: str,
    batch_size: int = 4,
    chunk_size: Tuple[int, int] = (132, 132)
):
    """Create and return YUV+VTM dataloader"""
    # Create default configs
    dataset_config = DatasetConfig()
    dataloader_config = DataloaderConfig()
    dataloader_config.batch_size = batch_size
    dataloader_config.val_batch_size = batch_size
    
    dataset = YUVVTMDataset(
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        data_dir=data_dir,
        frame_size=(1920, 1080),  # Adjust based on your video resolution
    )
    dataset.setup("fit")
    return dataset.train_dataloader(), dataset.val_dataloader()