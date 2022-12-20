from src.entity.config_entity import ImageRecord, ImageFolderConfig, EmbeddingsConfig
from src.exception.customexception import CustomException
from src.utils.database_handler import  MongoDBClient
from src.components.dataprocessing import DataProcessor
from src.components.model import SearchNet
from torchvision import transforms
from typing import List
from tqdm import tqdm
import os
import json
from PIL import Image
from torch import nn
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ImageFolder(Dataset):
    """
    Custom Image Folder Dataset for Embedding purpose
    Returns image, target, and s3_image_url triplet
    
    """
    def __init__(self, label_map:dict) -> None:
        super().__init__()
        self.config = ImageFolderConfig()
        self.config.LABEL_MAP = label_map
        self.transforms = self.transformers()
        self.image_records: List[ImageRecord]=[]
        self.record = ImageRecord

        self.folder_list  = os.listdir(self.config.ROOT_DIR)
        # iterate over every folders and then create  Image, label, and s3 link

        for folder in self.folder_list:
            folder_path = os.path.join(self.config.ROOT_DIR, f"{folder}")
            file_list = os.listdir(folder_path)
            for file_name in file_list:
                file_path = os.path.join(folder_path, f"{file_name}" )
                #for each Image file 
                # create triplet Image, label, s3_link for image
                print("Record : => ",self.record(img=file_path,
                                                      label=self.config.LABEL_MAP[folder],
                                                      s3_link=self.config.S3_LINK.format(self.config.BUCKET, folder,
                                                                                         file_name)))
                self.image_records.append(self.record(img=file_path,
                                                      label=self.config.LABEL_MAP[folder],
                                                      s3_link=self.config.S3_LINK.format(self.config.BUCKET, folder,
                                                                                         file_name)))




    def transformers(self):
        """
        Create Pytorch Image Transformation Pipeline
            Resize
            CenterCrop
            convertToTensor
            Normalize 
        """
        try:
            TRANSFORM_OBJ = transforms.Compose(
                [
                    transforms.Resize(self.config.IMG_SIZE),
                    transforms.CenterCrop(self.config.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406],
                                        std=[0.229,0.224,0.225]
                                        )
                ]
            )

            return TRANSFORM_OBJ
        except Exception as exp:
            raise CustomException(exp)



    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, index):
        record = self.image_records[index]
        img, targets, links = record.img, record.label, record.s3_link
        # open images
        images = Image.open(img)

        if len(images.getbands()) <3:
            # convert to rgb
            images = images.convert('RGB')
        # apply transformation
        images = np.array(self.transforms(images))
        # convert to tensors
        targets = torch.from_numpy(np.array(targets))
        images = torch.from_numpy(images)
        return images, targets, links

        

        
class EmbeddingGenerator:

    """
    Generate Embeddings and store it in mongoDB
    
    """

    def __init__(self, model, device) -> None:
        self.embedding_config = EmbeddingsConfig()
        self.model = model
        self.device = device
        self.embedding_model = self.load_model()
        self.embedding_model.eval()
        self.mongo = MongoDBClient()

    def load_model(self):
        model = self.model.to(self.device)
        model.load_state_dict(
                                torch.load(
                                        self.embedding_config.MODEL_STORE_PATH,
                                        map_location=self.device
                                    )
                            )

        # scrapped last part of model

        return nn.Sequential(*list(model.children())[:-1])

    def run_steps(self,batch_size, image,target, s3_link):
        record = {}
        # put images to device
        images = image.to(self.device)
        # forward pass to model to get logits
        logits = self.embedding_model(images)
        logits = logits.detach().cpu().numpy()
        # prepare triplet ImageRecord('images','label','s3_link') triplet format
        record['images'] = logits.tolist()
        record['label'] = target.tolist()
        record['s3_link'] = s3_link
        # print("s3_link", s3_link)
        # create json document list to upload to mongodb collection
        df = pd.DataFrame(record)
        #print(df)
        # create list of dictionary object to store in mongo db collection
        records = list(json.loads(df.T.to_json()).values())
        self.mongo.insert_bulk_record(records)
        

        return {"Response": f"Completed Embeddings Generation for {batch_size}."}


        



if __name__=='__main__':
    dp = DataProcessor()
    loaders = dp.run_step()

    data = ImageFolder(label_map=loaders["valid_data_loader"][1].class_to_idx)
    dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)
    embeds = EmbeddingGenerator(model=SearchNet(), device="cpu")

    for batch, values in tqdm(enumerate(dataloader)):
        img, target, link = values
        print(link, target)
        print("+"*100)
        print(embeds.run_steps(batch, img, target, link))
