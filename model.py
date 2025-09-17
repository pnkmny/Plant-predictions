import torch
import torch.nn as nn
import torchvision.models as models

class PlantClassifier(nn.Module):
    def __init__(self, num_meta_features=3, num_classes=100):
        super().__init__()
        # Pretrained image backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # remove original fc layer

        # Metadata branch
        self.meta_fc = nn.Linear(num_meta_features, 128)

        # Final classifier
        self.classifier = nn.Linear(512 + 128, num_classes)

    def forward(self, x_img, x_meta):
        img_feat = self.backbone(x_img)
        meta_feat = torch.relu(self.meta_fc(x_meta))
        x = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(x)
