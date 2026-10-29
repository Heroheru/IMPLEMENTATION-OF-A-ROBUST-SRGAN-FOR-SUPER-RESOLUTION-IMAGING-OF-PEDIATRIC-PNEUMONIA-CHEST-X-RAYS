
import torch.nn as nn

class CheXNetFeatureExtractor(nn.Module):
    def __init__(self, chexnet, selected_layers):
        super().__init__()
        self.chexnet = chexnet
        self.selected_layers = selected_layers  # Contoh: ['denseblock1', 'denseblock2']

    def forward(self, x):
        features = {}
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.chexnet.features.conv0(x)
        x = self.chexnet.features.norm0(x)
        x = self.chexnet.features.relu0(x)
        x = self.chexnet.features.pool0(x)

        x = self.chexnet.features.denseblock1(x)
        if 'denseblock1' in self.selected_layers:
            features['denseblock1'] = x

        x = self.chexnet.features.transition1(x)  
        x = self.chexnet.features.denseblock2(x)
        if 'denseblock2' in self.selected_layers:
            features['denseblock2'] = x

        x = self.chexnet.features.transition2(x) 
        x = self.chexnet.features.denseblock3(x)
        if 'denseblock3' in self.selected_layers:
            features['denseblock3'] = x

        x = self.chexnet.features.transition3(x) 
        x = self.chexnet.features.denseblock4(x)
        if 'denseblock4' in self.selected_layers:
            features['denseblock4'] = x

        return features