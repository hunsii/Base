# +
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        if opt.model_name == "efficientnet_b0":
            self.feature_extract = models.efficientnet_b0(pretrained=True)
        elif opt.model_name == "efficientnet_b5":
            self.feature_extract = models.efficientnet_b5(pretrained=True)
        self.classifier = nn.Linear(opt.feature_dim, opt.num_classes)
        

        
    def forward(self, x):
        x = self.feature_extract(x)
        x = self.classifier(x)
        return x
