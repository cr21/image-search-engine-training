
import torch
from src.entity.config_entity import ModelTrainerConfig
from src.components.dataprocessing import DataProcessor
from src.components.model import SearchNet
from torch import nn
from tqdm import tqdm
import numpy as np

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

    def train_model(self):
        print("Start Training ...... \n")
        # put model on training model
        self.model.train()
        for epoch in range(self.config.EPOCHS):
            print(f"Epoch Number : {epoch}")
            running_loss = 0.0
            running_correct = 0

            # iterate over data loader
            for data in tqdm(self.train_data_loader):
                data, targets = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                # forward pass
                outs = self.model(data)
                # get loss values
                loss_val = self.criterion(outs, targets)
                # running loss
                running_loss+=loss_val.item()
                # get predicted labels
                _, preds = torch.max(outs.data,1)
                # count total true predictions
                running_correct+=(preds ==targets).sum().item()
                # back propogation
                loss_val.backward()
                # optimizer step
                self.optimizer.step()

            loss = running_loss/len(self.train_data_loader.dataset)
            accuracy = 100 * running_correct/len(self.train_data_loader.dataset)

            # get validation loss and validation accuracy
            val_loss, val_acc = self.evaluate()
                
            print(f"Train Acc : {accuracy:.4f}, Train Loss : {loss:.4f}, Validation Acc {val_loss:.4f} , validation loss : {val_acc:.4f}")
            print("Training Complete \n")


    def evaluate(self, validate=False):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        self.model.eval()
        val_accuracy = []
        val_loss = []
        running_loss = 0.0
        running_correct = 0

        loader = self.test_data_loader if not validate else self.valid_data_loader

        with torch.no_grad():
            for batch in tqdm(loader):
                img = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                logits = self.model(img)
                loss = self.criterion(logits, labels)
                val_loss.append(loss.item())
                running_loss+=loss.item()
                _, preds = torch.max(logits.data,1)
                # count total true predictions
                running_correct+=(preds ==labels).sum().item()
                


            val_loss = running_loss/len(self.loader.dataset)
            val_accuracy = 100 * running_correct/len(self.loader.dataset)

                
            

        return val_loss, val_accuracy

    def evaluate1(self,validate=True):
        print("Evaluation Starts ... \n")
        #put model in evaluation mode
        self.model.eval()
        loader = self.valid_data_loader if validate else self.test_data_loader

        val_loss=[]
        val_acc=[]


        with torch.no_grad():

            # iterate over data loader
            for data in tqdm(loader):
                data, targets = data[0].to(self.device), data[1].to(self.device)
                # forward pass
                outs = self.model(data)
                # get loss values
                loss_val = self.criterion(outs, targets)
                # running loss
                val_loss.append(loss_val.item())
                # get predicted labels
                preds = torch.argmax(outs,dim=1).flatten()
                # count total true predictions
                accuracy=(preds ==targets).cpu().numpy().mean()*100
                val_acc.append(accuracy)
                    
        val_loss= np.mean(val_loss)
        val_acc= np.mean(val_acc)

        return val_loss, val_acc


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
