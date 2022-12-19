import torch
from from_root import from_root
from src.entity.config_entity import ModelConfig
import torch.nn  as nn

class SearchNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.config = ModelConfig()
        self.base_model = self.get_base_model()
        self.conv1 = nn.Conv2d(
                                in_channels=512,
                                out_channels=32,
                                kernel_size=(3,3), stride=(1,1), padding=(1,1)
                            )
        self.conv2 = nn.Conv2d(
                                in_channels=32,
                                out_channels=16,
                                kernel_size=(3,3), stride=(1,1), padding=(1,1)
                            )
        self.conv3 = nn.Conv2d(
                                in_channels=16,
                                out_channels=4,
                                kernel_size=(3,3), stride=(1,1), padding=(1,1)
                            )

        self.flatten = nn.Flatten()
        self.final = nn.Linear(4*8*8, self.config.LABEL)

    def get_base_model(self):
        """
        Load Basic Model from torch hub store it to custom model benchmark directory
        Remove Last two layers and then add 3 conv2 layers followed by fully connected output layer
        to get final model
        
        
        """
        torch.hub.set_dir(self.config.BASE_MODEL_PATH)
        model = torch.hub.load(
                                repo_or_dir=self.config.REPOSITORY,
                                model=self.config.BASEMODE,
                                pretrained=self.config.PRETRAINED
        )
        
        # remove last two layers
        return nn.Sequential(*list(model.children())[:-2])

    def forward(self, data):
        x = self.base_model(data)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.final(x)
        return x


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = SearchNet()
    net.to(device)



    

