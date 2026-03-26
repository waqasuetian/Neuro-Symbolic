from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

@dataclass
class ModelConfig: 
    num_classes: int = 2
    batch_size: int = 32
    input_shape: tuple = (1024, 19)
    shuffle_buffer: int = 10000 

@dataclass
class DataConfig:
    fs: int = 256
    epoch_seconds: int = 4
    lowcut: float = 0.5
    highcut: float = 40.0
    notch_freq: float = 50.0
    n_channels: int = 19
    min_seizure_duration: int = 2
    event_mapping: dict = field(default_factory=lambda: {
        'bckg': 0,
        'sz_foc_ia': 1
    })

    @property
    def inverse_mapping(self):
        return {v: k for k, v in self.event_mapping.items()}

@dataclass
class PathConfig:
    base_dir: Path = Path(r"D:\1\Code and Ablation Study\hyb_Cnn-Lstm\CompleteModel_CNNandLSTM")
    train_data: Path = Path(r"D:\1\Code and Ablation Study\hyb_Cnn-Lstm\dataset")
    test_data: Path = Path(r"D:\1\Code and Ablation Study\hyb_Cnn-Lstm\dataset\test")
    processed_train: Path = base_dir / "processed_train"
    processed_test: Path = base_dir / "processed_test"
    output_dir: Path = Path(r"D:\1\Code and Ablation Study\hyb_Cnn-Lstm\BIDSDATAFINAL - Copy\results")


class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
        
        # Initialize directories
        self.paths.processed_train.mkdir(parents=True, exist_ok=True)
        self.paths.processed_test.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)