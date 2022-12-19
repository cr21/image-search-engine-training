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
class DataProcessingConfig:
    BATCH_SIZE:int = 32
    IMAGE_SIZE:int = 256
    TRAIN_DATA_PATH:str=os.path.join(from_root(),"data","splitted","train")
    VALID_DATA_PATH:str=os.path.join(from_root(),"data","splitted","val")
    TEST_DATA_PATH:str=os.path.join(from_root(),"data","splitted","test")


@dataclass
class ModelConfig:
    LABEL:str=101
    BASE_MODEL_PATH:str = os.path.join(from_root(), "model","benchmark")
    REPOSITORY:str='pytorch/vision:v0.10.0'
    BASEMODE:str = 'resnet18'
    PRETRAINED:bool = True

@dataclass
class ModelTrainerConfig:
    MODEL_STORE_PATH:str = os.path.join(from_root(),"model","finetuned","model.pth")
    EPOCHS:int=15
    EVALUATION:bool=True

@dataclass
class DatabaseConfig:
    USERNAME:str=os.getenv("DATABASE_USERNAME")
    PASSWORD:str=os.getenv("DATABASE_PASSWORD")
    URL:str='mongodb+srv://<username>:<password>@cluster0.ufj8ovv.mongodb.net/?retryWrites=true&w=majority'
    DBNAME:str="reverse_image_search_engine"
    COLLECTION:str="embeddings"



@dataclass
class EmbeddingsConfig:
    MODEL_STORE_PATH:str = os.path.join(from_root(),"model","finetuned","model.pth")

@dataclass
class ImageRecord:
    img:str=None
    label:str=None
    s3_link:str=None

@dataclass
class AnnoyConfig:
    EMBEDDING_STORE_PATH:str = os.path.join(from_root(),"data","embeddings","embeddings.ann")
    EMBEDDING_SIZE:int = 256

@dataclass
class ImageFolderConfig:
    ROOT_DIR:str=os.path.join(from_root(),"data","raw","images")
    IMG_SIZE:int=256
    LABEL_MAP={}
    BUCKET:str='cr-img-search-engine'
    S3_LINK:str="https://{0}.s3.us-east-1.amazonaws.com/images/{1}/{2}"
    
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
    
    



