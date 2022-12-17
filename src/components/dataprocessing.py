import os
import torch
from src.entity.config_entity import DataProcessingConfig
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.exception.customexception import CustomException

class DataProcessor:
    def __init__(self) -> None:
        self.config = DataProcessingConfig()

    def transformation(self):
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
                    transforms.Resize(self.config.IMAGE_SIZE),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406],
                                        std=[0.229,0.224,0.225]
                                        )
                ]
            )

            return TRANSFORM_OBJ
        except Exception as exp:
            raise CustomException(exp)


    def create_folders(self, TRANSFORM_OBJ)-> dict:
        """
        The create_loaders method takes Transformations and create dataloaders.
        :param TRANSFORM_IMG:
        :return: Dict of train, test, valid Loaders
        """
        try:
            print("Generating Data Loaders")
            loaders = {}
            for _ in tqdm(range(1)):
                train_data = ImageFolder(root=self.config.TRAIN_DATA_PATH, transform=TRANSFORM_OBJ)
                valid_data = ImageFolder(root=self.config.VALID_DATA_PATH, transform=TRANSFORM_OBJ)
                test_data = ImageFolder(root=self.config.TEST_DATA_PATH, transform=TRANSFORM_OBJ)


                train_data_loader = DataLoader(
                                                dataset=train_data,
                                                batch_size=self.config.BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=1
                                            )

                test_data_loader = DataLoader(
                                                dataset=test_data,
                                                batch_size=self.config.BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=1
                                            )

                valid_data_loader = DataLoader(
                                                dataset=valid_data,
                                                batch_size=self.config.BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=1
                                            )

            loaders = {
                "train_data_loader":(train_data_loader, train_data),
                "valid_data_loader":(valid_data_loader, valid_data),
                "test_data_loader":(test_data_loader, test_data)
            }

            return loaders 


            pass
        except Exception as exp:
            raise CustomException(exp)
        

    def run_step(self)->dict:
        transform_obj = self.transformation()
        loaders = self.create_folders(transform_obj)
        return loaders

if __name__=='__main__':

    data_processor_obj = DataProcessor()

    data_processor_obj.run_step()