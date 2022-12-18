import torch
from from_root import from_root
import annoy
from src.entity.config_entity import DataIngestionConfig, s3Config, DataProcessingConfig
from src.components.dataingestion import DataIngestion
from src.components.dataprocessing import DataProcessor
from src.components.model import SearchNet
from src.components.training import ModelTrainer
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

    def run_data_processing_steps(self):
        data_processor_obj  = DataProcessor()
        loaders = data_processor_obj.run_step()
        return loaders
    
    def initiate_model_architecture(self):
        return SearchNet()


    def initiate_model_training(self, loaders, net):
        self.trainer = ModelTrainer(loaders, self.device, net)
        self.trainer.train_model()
        self.trainer.evaluate(validate=True)
        self.trainer.save_to_model_path()
        

    def run_pipeline(self):
        # 1. Data Ingestion Process
        self.run_data_ingestion_process()
        # 2. Data Processing Process
        loaders = self.run_data_processing_steps()
        # 3. Model Building
        search_net = self.initiate_model_architecture()
        # 4. Data Training Process
        self.initiate_model_training(loaders=loaders, net=search_net)
        # 5. Generate Embeddings
        # 6. Annoy Embeddings
        # 7. Push Embeddings, Models, Paths to Artifact aka push artifacts





if __name__ == '__main__':
    search_pipeline = Pipeline()
    search_pipeline.run_pipeline()
    

