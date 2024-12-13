import torch
from pdb import set_trace as pause

class FeatureBank:

    def __init__(self, cls_number, max_samples, features_size=(512,7,7)):
       
        self.cls_number = cls_number
        self.max_samples = max_samples
        self.features_size = features_size

        self.features = torch.rand((self.cls_number, self.max_samples, self.features_size[0], self.features_size[1], self.features_size[2]))


    def add_features(self, features):
        
        for idx, c_features in enumerate(features):
            if len(c_features) > 0:
                try:
                    self.features[idx] = torch.cat((self.features[idx][len(c_features):],c_features.cpu().detach()))
                except:
                    pause()