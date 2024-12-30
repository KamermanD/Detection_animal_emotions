import torch
import numpy as np
from PIL import Image

class Extraction:
    def __init__(self, network):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network = network.eval().to(self.device)
        
    def extract(self, loader):
        data_tmp = []
        label_tmp = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
            
                outputs = self.network(x)
                data_tmp.append(outputs.view(-1, 512).cpu().numpy())
                
                label_tmp.append(y.cpu().numpy())
                
        return np.vstack(data_tmp), np.hstack(label_tmp)

def extract_features_predict(image_paths, transform, feature_extractor, device):
    features = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = feature_extractor(image)
        features.append(output.view(-1).cpu().numpy())
    return np.vstack(features)