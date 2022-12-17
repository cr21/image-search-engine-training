import torch
from from_root import from_root
import annoy
from src.entity.config_entity import DataIngestionConfig, s3Config
from src.components.dataingestion import DataIngestion
import os


class Pipeline:
    def __init__(self):
        self.paths = ["data", "data/raw", "data/splitted", "data/embeddings",
                      "model", "model/benchmark", "model/finetuned"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_data_ingestion_process(self):
        # create data dirs
        for folder in self.paths:
            folder_path  = os.path.join(from_root(), folder)
            print("folder_path")
            os.makedirs(folder_path, exist_ok=True)
        data_ingest_obj = DataIngestion()
        
        data_ingest_obj.run_step()

        

    def run_pipeline(self):
        # 1. Data Ingestion Process
        self.run_data_ingestion_process()
        # 2. Data Processing Process
        # 3. Data Transformation Process
        # 4. Data Training Process
        # 5. Generate Embeddings
        # 6. Annoy Embeddings
        # 7. Push Embeddings, Models, Paths to Artifact aka push artifacts





if __name__ == '__main__':
    search_pipeline = Pipeline()
    search_pipeline.run_pipeline()
    

