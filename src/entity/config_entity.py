from dataclasses import dataclass, field

from from_root import from_root
import os

@dataclass
class DataIngestionConfig:
    PREFIX:str = 'images/'
    RAW:str = 'data/raw'
    SPLIT:float = "data/splitted"
    BUCKET:str = "cr-img-search-engine"
    SEED:int=1331
    RATIO:float = (0.8,0.1,0.1) #[training, validation, test]


@dataclass
class s3Config:
    ACCESS_KEY_ID:str = os.getenv('AWS_ACCESS_KEY_ID')
    SECRET_KEY:str = os.getenv('AWS_SECRET_ACCESS_KEY')
    REGION_NAME:str = 'us-east-01'
    BUCKET_NAME:str = 'cr-img-search-engine'
    KEY:str = "model"
    ZIP_NAME:str = "artifacts.tar.gz"
    ZIP_PATHS: list = field(
                            default_factory=lambda: [
                                (os.path.join(from_root(), "data", "embeddings", "embeddings.json"), "embeddings.json"),
                                (os.path.join(from_root(), "data", "embeddings", "embeddings.ann"), "embeddings.ann"),
                                (os.path.join(from_root(), "model", "finetuned", "model.pth"), "model.pth")
                            ]
                        )
    
    


