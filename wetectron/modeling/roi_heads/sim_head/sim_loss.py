from torch.nn import functional as F
from wetectron.utils.utils import cos_sim
import torch.nn as nn
import torch
from itertools import combinations

from wetectron.modeling.roi_heads.sim_head.feature_bank import FeatureBank


from pdb import set_trace as pause


class SupConLossV2(nn.Module):
    def __init__(self, num_classes, temperature=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.means = torch.zeros((num_classes,128)).cuda()

        self.feature_bank = FeatureBank(cls_number=num_classes,  max_samples=60)

        self.not_have_mean = [i for i in range(num_classes)]

        self.d_idx_a = []
        self.d_idx_b = []
        for ca in range(self.num_classes):
            for cb in range(self.num_classes):
                if ca != cb:
                    self.d_idx_a.append(ca)
                    self.d_idx_b.append(cb)

        self.all_initialized = False

    def forward(self, overlaps_enc, feature_extractor, model_sim, device, i_ref, cfg):


        d_v = cfg.MODEL.ROI_WEAK_HEAD.D_V 
        d_d = cfg.MODEL.ROI_WEAK_HEAD.D_D

        # compute the means of the feature bank classes: 
        feats = self.feature_bank.features
        f_shape = feats.shape
       
        bank_proj_features  = model_sim(feature_extractor.forward_neck(feats.view(-1,f_shape[2],f_shape[3],f_shape[4]).to(device)))
        mb_classes          = torch.arange(f_shape[0])[:,None].repeat(1, f_shape[1]).view(-1).to(device)


        all_means = bank_proj_features.view(f_shape[0],f_shape[1], -1).mean(dim=1)

        feat_batch = []
        cls_batch  = []

        for i, embedding_v in enumerate(overlaps_enc):
            if embedding_v.shape[0] != 0:
        
                feat_batch.append(embedding_v)
                cls_batch.append( torch.ones(len((embedding_v))) * i)
        
        feat_batch = torch.cat(feat_batch)
        cls_batch  = torch.cat(cls_batch)

        batch_proj_features = model_sim(feature_extractor.forward_neck(feat_batch.to(device)))


        all_feats = torch.cat((bank_proj_features, batch_proj_features))
        all_clas  = torch.cat((mb_classes.to(device), cls_batch.to(device))).int()

        weights = 1/all_clas.unique(return_counts=True)[1]
        
        ###### Gets the VAR loss
        l_var = weights[all_clas] * (torch.clamp( torch.norm(all_means[all_clas] - all_feats, dim=1) - d_v,  min=0 ) ** 2)

        l_var = l_var.sum() / len(all_means)



        ###### Gets the DIST loss
        l_dist = torch.clamp( 2*d_d - torch.norm(all_means[self.d_idx_a] -all_means[self.d_idx_b], dim=1),  min=0 ) ** 2
        l_dist = l_dist.sum() / (len(all_means)*(len(all_means)-1))


        # ok
        self.means = all_means.detach()

        # Update the feature_bank memory with the last top scoring samples
        self.feature_bank.add_features(overlaps_enc)

        # return l_var, l_dist, l_center
        return l_var, l_dist


    
        


       

        # #### UPDATE MEANS
        # for ca, ma in zip(all_idxs, all_means):
        #     # prop = 0.01
        #     # self.means[ca] = (1-prop)*self.means[ca] + prop*ma.detach()
        #     self.means[ca] = ma.detach()

        # return l_var, l_dist, l_center
