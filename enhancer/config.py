from pydantic import BaseModel
from enum import Enum
from typing import Tuple, List, Optional, Any, Dict
import yaml


class DataloaderConfig(BaseModel):
    n_step: int = 1000
    val_n_step: int = 5
    test_n_step: int = 5

    batch_size: int = 8
    val_batch_size: int = 64
    test_batch_size: int = 64  # full video


class SubDatasetConfig(BaseModel):
    decoded_yuv_filepath: str = ""
    original_yuv_file_path: str = ""
    vtm_trace_filepath: str = ""
    width: int = 0
    height: int = 0


class NetworkImplementation(Enum):
    DENSE = "dense"
    RES = "res"
    CONV = "conv"


class TransitionMode(Enum):
    same = "same"
    down = "down"


class TransitionConfig(BaseModel):
    kernel_size: int = 1
    padding: int = 0
    stride: int = 1

    mode: TransitionMode


class BlockConfig(BaseModel):
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1

    num_layers: int = 1
    features: int = 16

    dropout: float = 0.0

    transition: Optional[TransitionConfig] = None


class ClassifierConfig(BlockConfig):
    sigmoid: bool = True


class OutputBlockConfig(BlockConfig):
    tanh: bool = False


class FeaturesConfig(BlockConfig):
    dense: bool = False
    res: bool = False
    pool: bool = False


class StructureConfig(BaseModel):
    blocks: List[BlockConfig] = [BlockConfig()]


class NetworkConfig(BaseModel):
    implementation: NetworkImplementation = NetworkImplementation.CONV
    reflect_padding: bool = True
    activation: str = "prelu"
    bn_size: float = 2

    structure: StructureConfig = StructureConfig()
    features: Optional[FeaturesConfig] = None
    classifier: Optional[ClassifierConfig] = None
    output_block: Optional[OutputBlockConfig] = None

    load_from: Optional[str] = None
    save_to: Optional[str] = None

    input_shape: Tuple[int, int, int] = (132, 132, 3)


class EnhancerConfig(NetworkConfig):
    metadata_size: int = 7
    metadata_features: int = 32
    with_mask: bool = True

    # input_shape is 8 for 1 luma channel and 7 additional VTM features
    input_shape: Tuple[int, int, int] = (8, 132, 132)


class DiscriminatorConfig(NetworkConfig):
    pass


class DatasetConfig(BaseModel):
    train: SubDatasetConfig = SubDatasetConfig()
    val: SubDatasetConfig = SubDatasetConfig()
    test: SubDatasetConfig = SubDatasetConfig()


class TrainingMode(Enum):
    GAN = "gan"
    ENHANCER = "enhancer"
    DISCRIMINATOR = "discriminator"


class ModeTrainingConfig(BaseModel):
    epochs: int = 100

    enhancer_lr: float = 1e-4
    betas: Tuple[float, float] = (0.5, 0.999)

    discriminator_lr: float = 1e-4
    momentum: float = 0.9

    num_samples: int = 6

    probe: int = 10
    enhancer_min_loss: float = 0.35
    discriminator_min_loss: float = 0.20

    enhancer_scheduler: bool = True
    discriminator_scheduler: bool = True

    enhancer_scheduler_gamma: float = 0.1
    discriminator_scheduler_gamma: float = 0.1

    enhancer_scheduler_milestones: List[int] = [40, 80]
    discriminator_scheduler_milestones: List[int] = [20, 50, 100, 150]

    saved_chunk_folder: str = "enhanced"


class TrainerConfig(BaseModel):
    mode: TrainingMode = TrainingMode.GAN

    gan: ModeTrainingConfig = ModeTrainingConfig()
    enhancer: ModeTrainingConfig = ModeTrainingConfig()
    discriminator: ModeTrainingConfig = ModeTrainingConfig()

    separation_epochs: int = 10
    # channels_grad_scales: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    channels_grad_scales: tuple[float, float, float] = (2 / 3, 1 / 6, 1 / 6)
    # channels_grad_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def current(self) -> ModeTrainingConfig:
        if self.mode == TrainingMode.GAN:
            return self.gan
        elif self.mode == TrainingMode.ENHANCER:
            return self.enhancer
        elif self.mode == TrainingMode.DISCRIMINATOR:
            return self.discriminator
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


class Config(BaseModel):
    dataloader: DataloaderConfig = DataloaderConfig()
    dataset: DatasetConfig = DatasetConfig()
    enhancer: EnhancerConfig = EnhancerConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
    test_full_frames: bool = False

    trainer: TrainerConfig = TrainerConfig()

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        print("\n--- DEBUG: RAW YAML DATA ---")
        print(data) 
        print("---------------------------\n")

        model = cls.model_validate(data)
        
        print("--- DEBUG: POJECTED CONFIG OBJECT ---")
        print(model.dataset.train)
        print("------------------------------------\n")

        model = cls.model_validate(data)
        return model
