
import torch
from src.entity.config_entity import ModelTrainerConfig
from src.components.dataprocessing import DataProcessor
from src.components.model import SearchNet
from torch import nn
from tqdm import tqdm
import numpy as np
from time import time
from src.utils.common import seed_everything

class ModelTrainer:

    def __init__(self, dataloader :dict, device:str,model:torch.nn.Module) -> None:
        self.loaders = dataloader
        self.train_data_loader = self.loaders['train_data_loader'][0]
        self.test_data_loader = self.loaders['test_data_loader'][0]
        self.valid_data_loader=self.loaders['valid_data_loader'][0]
        self.criterion =  nn.CrossEntropyLoss()
        self.model = model.to(device=device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=1e-4)
        self.config = ModelTrainerConfig()
        self.evaluation = self.config.EVALUATION
        self.device = device
        seed_everything()

    def train_model(self):
        train_loss , train_accuracy = [], []
        val_loss , val_accuracy = [], []
        start = time.time()
        for epoch in range(self.config.EPOCHS):
            print(f'Training Epoch: {epoch}')
            train_epoch_loss, train_epoch_accuracy = self.fit(epoch)
            val_epoch_loss, val_epoch_accuracy = self.evaluate()
            train_loss.append(train_epoch_loss)
            train_accuracy.append(train_epoch_accuracy)
            val_loss.append(val_epoch_loss)
            val_accuracy.append(val_epoch_accuracy)

        
        end = time.time()
        print((end-start)/60, 'minutes')
        print(f"average Training Loss -> {np.mean(train_loss)}")
        print(f"average Training Accuracy -> {np.mean(train_accuracy)}")
        print(f"average Validation Loss -> {np.mean(val_loss)}")
        print(f"average Validation Accuracy -> {np.mean(val_accuracy)}")

    def fit(self):
        
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        for i, data in tqdm(enumerate(self.train_data_loader)):
            data, target = data[0].to(self.device), data[1].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == target).sum().item()
            loss.backward()
            self.optimizer.step()
            
        loss = running_loss/len(self.train_data_loader.dataset)
        accuracy = 100. * running_correct/len(self.train_data_loader.dataset)
        
        print(f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}")
        
        return loss, accuracy

    # def train_model(self):
    #     print("Start Training ...... \n")
    #     # put model on training model
    #     self.model.train()
    #     for epoch in range(self.config.EPOCHS):
    #         print(f"Epoch Number : {epoch}")
    #         running_loss = 0.0
    #         running_correct = 0

    #         # iterate over data loader
    #         for data in tqdm(self.train_data_loader):
    #             data, targets = data[0].to(self.device), data[1].to(self.device)
    #             self.optimizer.zero_grad()
    #             # forward pass
    #             outs = self.model(data)
    #             # get loss values
    #             loss_val = self.criterion(outs, targets)
    #             # running loss
    #             running_loss+=loss_val.item()
    #             # get predicted labels
    #             _, preds = torch.max(outs.data,1)
    #             # count total true predictions
    #             running_correct+=(preds ==targets).sum().item()
    #             # back propogation
    #             loss_val.backward()
    #             # optimizer step
    #             self.optimizer.step()

    #         loss = running_loss/len(self.train_data_loader.dataset)
    #         accuracy = 100 * running_correct/len(self.train_data_loader.dataset)

    #         # get validation loss and validation accuracy
    #         val_loss, val_acc = self.evaluate()
                
    #         print(f"Train Acc : {accuracy:.4f}, Train Loss : {loss:.4f}, Validation Acc {val_loss:.4f} , validation loss : {val_acc:.4f}")
    #         print("Training Complete \n")

    def evaluate(self,validate=False):
        print('Validating')
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.valid_data_loader)):
                data, target = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                running_correct += (preds == target).sum().item()
            
            loss = running_loss/len(self.valid_data_loader.dataset)
            accuracy = 100. * running_correct/len(self.valid_data_loader.dataset)
            print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f}')
            
            return loss, accuracy


    # def evaluate(self, validate=False):
    #     """After the completion of each training epoch, measure the model's performance
    #     on our validation set.
    #     """
    #     self.model.eval()
    #     val_accuracy = []
    #     val_loss = []
    #     running_loss = 0.0
    #     running_correct = 0

    #     loader = self.test_data_loader if not validate else self.valid_data_loader

    #     with torch.no_grad():
    #         for batch in tqdm(loader):
    #             img = batch[0].to(self.device)
    #             labels = batch[1].to(self.device)
    #             logits = self.model(img)
    #             loss = self.criterion(logits, labels)
    #             val_loss.append(loss.item())
    #             running_loss+=loss.item()
    #             _, preds = torch.max(logits.data,1)
    #             # count total true predictions
    #             running_correct+=(preds ==labels).sum().item()
            
    #         val_loss = running_loss/len(loader.dataset)
    #         val_accuracy = 100 * running_correct/len(loader.dataset)

                
            

    #     return val_loss, val_accuracy

    

    def save_to_model_path(self):
        model_store_path = self.config.MODEL_STORE_PATH
        print(f"saving model to Path => {model_store_path}")
        torch.save(self.model.state_dict(), model_store_path)




if __name__ == "__main__":
    dp = DataProcessor()
    loaders = dp.run_step()
    trainer = ModelTrainer(loaders, "cpu", model=SearchNet())
    trainer.train_model()
    trainer.evaluate(validate=True)
    trainer.save_to_model_path()
