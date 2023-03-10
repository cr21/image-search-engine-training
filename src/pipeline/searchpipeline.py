import torch
from from_root import from_root
import annoy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.components.embeddings import ImageFolder, ImageRecord, EmbeddingGenerator
from src.entity.config_entity import DataIngestionConfig, s3Config, DataProcessingConfig
from src.components.dataingestion import DataIngestion
from src.components.dataprocessing import DataProcessor
from src.components.model import SearchNet
from src.components.training import ModelTrainer
from src.components.nearest_neighbours import Annoy
from src.utils.storage_handler import S3Connector
import os
from torch import nn


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

    def generate_embeddings(self,loaders:dict, net:nn.Module):
        # search_dataset = 
        search_dataset = ImageFolder(label_map=loaders["train_data_loader"][1].class_to_idx)
        dataloader = DataLoader(dataset=search_dataset, batch_size=64, shuffle=True)
        embeds = EmbeddingGenerator(model=net, device=self.device)

        for batch, values in tqdm(enumerate(dataloader)):
            img, target, link = values
            print(embeds.run_steps(batch, img, target, link))

    def create_annoy_index(self):
        ann = Annoy()
        ann.run_step()

    def push_artifacts(self):
        s3_connector = S3Connector()
        response = s3_connector.zip_files()
        return response

    def run_pipeline(self):
        # 1. Data Ingestion Process
        print("Step 1")
        print("+"*100)
        self.run_data_ingestion_process()
        # 2. Data Processing Process
        print("Step 2")
        print("+"*100)
        loaders = self.run_data_processing_steps()
        # 3. Model Building
        print("Step 3")
        print("+"*100)
        search_net = self.initiate_model_architecture()
        # 4. Data Training Process
        print("Step 4")
        print("+"*100)
        self.initiate_model_training(loaders=loaders, net=search_net)
        # 5. Generate Embeddings
        print("Step 5")
        print("+"*100)
        self.generate_embeddings(loaders=loaders, net=search_net)
        # 6. Annoy Embeddings
        print("Step 6")
        print("+"*100)
        self.create_annoy_index()
        # 7. Push Embeddings, Models, Paths to Artifact aka push artifacts
        print("Step 7")
        print("+"*100)
        self.push_artifacts()




if __name__ == '__main__':
    search_pipeline = Pipeline()
    search_pipeline.run_pipeline()
    

