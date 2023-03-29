# +
import torch.nn as nn
import torchvision.models as models
import timm

class CrashModel(nn.Module):
    def __init__(self, num_classes=13):
        super(CrashModel, self).__init__()
        self.feature_extract = FeatureExtractor(1, 1024)
        self.classifier = Classifier(1024, 13)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x
    
class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeatureExtractor, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(3, 8, (1, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, (1, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 16, (1, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
        )
    def forward(self, input):
        batch_size = input.size(0)
        output = self.feature_extract(input)
        output = output.view(batch_size, -1)
        return output
    
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes = 13):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes), 
        )
    def forward(self, input):
        output = self.classifier(input)
        return output

class CrashModelV2_3D(nn.Module):
    def __init__(self, opt):
        super(CrashModelV2_3D, self).__init__()
        if opt.model_name == "inception_v3":
            print(f"{opt.model_name} is loading...")
            self.feature_extract = models.inception_v3(pretrained=True)
        else:
            self.feature_extract = models.video.r3d_18(pretrained=True) # https://pytorch.org/vision/0.8/models.html
        num_features = self.feature_extract.fc.in_features
        self.feature_extract.fc = nn.Linear(num_features, opt.num_classes)

#         self.classifier = Classifier(opt.feature_dim, opt.num_classes)
        
    def forward(self, x):
#         batch_size = x.size(0)
        x = self.feature_extract(x)
#         x = x.view(batch_size, -1)
#         x = self.classifier(x)
        return x

class CrashModelV2_2D(nn.Module):
    def __init__(self, opt):
        super(CrashModelV2_2D, self).__init__()
        self.feature_extract = timm.create_model('efficientnet_b0', pretrained=True, num_classes=opt.num_classes)
        
    def forward(self, x):
        x = self.feature_extract(x)
        return x
    
class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        if opt.model_name == "inception_v3":
            self.feature_extract = models.inception_v3(pretrained=True)
#         elif opt.model_name==
        # Replace the output layer with a new layer for 3-class classification
        num_ftrs = self.feature_extract.fc.in_features
        model.fc = nn.Linear(num_ftrs, opt.num_classes)

        
    def forward(self, x):
        x = self.feature_extract(x)
        return x
# -


