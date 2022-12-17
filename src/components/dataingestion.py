
import torch
import os
from from_root import from_root
from src.entity.config_entity import DataIngestionConfig
from src.exception.customexception import CustomException
import splitfolders


class DataIngestion:
    def __init__(self):
        self.data_ingest_config = DataIngestionConfig()
    
    def run_step(self):
        # 1. download data from s3 bucket to data dir
        # sync s3 folder to current folder
        self.download_dir()
        # 2. split data to train, test , validation split
        self.split_data()

        return {"Response": "Completed Data Ingestion"}

    def download_dir(self):
        """
        params:
            - prefix : pattern to match in s3 bucket
            - raw :  data fodler 
            - bucket : s3 bucket from which we want to fetch data
            - client :  initialize s3 client object
        """
        try:

            print("\n Fetching Data From s3 Table Starts:")
            data_raw=self.data_ingest_config.RAW
            folder_prefix = self.data_ingest_config.PREFIX
            bucket = self.data_ingest_config.BUCKET
            data_path = os.path.join(from_root(),data_raw, folder_prefix)
            os.system(f"aws s3 sync s3://{bucket}/{folder_prefix}  {data_path} --no-progress")
            print("\n Fetching Data from s3 Table finished")


        except Exception as exp:
            raise CustomException(exp)



    def split_data(self):
        """
        Split data into training, validation, testing data folders
        """
        try:
            splitfolders.ratio(
                                input =os.path.join(self.data_ingest_config.RAW,
                                    self.data_ingest_config.PREFIX
                                ),
                        output=self.data_ingest_config.SPLIT,
                        seed=self.data_ingest_config.SEED,
                        ratio=self.data_ingest_config.RATIO,
                        group_prefix=None,
                        move=False
            )
            
        except Exception as exp:
            raise exp


if __name__ == '__main__':
    paths = ["data", "data/raw", "data/splitted", "data/embeddings",
                      "model", "model/benchmark", "model/finetuned"]

    
    for folder in paths:
        folder_path  = os.path.join(from_root(), folder)
        os.makedirs(folder_path, exist_ok=True)

    data_ingest_obj = DataIngestion()
    data_ingest_obj.run_step()