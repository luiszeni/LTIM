# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
import collections
import torch.nn as nn
import random
import os
import numpy as np
from torch.nn import functional as F
from pdb import set_trace as pause
from wetectron.layers import smooth_l1_loss
from wetectron.modeling import registry
from wetectron.modeling.utils import cat
from wetectron.config import cfg
from wetectron.structures.boxlist_ops import boxlist_iou, boxlist_iou_async, boxlist_nms_index
from wetectron.structures.bounding_box import BoxList
from wetectron.modeling.matcher import Matcher
from wetectron.utils.utils import to_boxlist, cal_iou, easy_nms, cos_sim, get_share_class, generate_img_label
from .pseudo_label_generator import oicr_layer, mist_layer, od_layer
from wetectron.modeling.roi_heads.sim_head.sim_loss import SupConLossV2
from wetectron.modeling.roi_heads.sim_head.sim_net import Sim_Net
import cv2

from torchvision.ops import box_iou, box_area


def compute_avg_img_accuracy(labels_per_im, score_per_im, num_classes):
    """
       the accuracy of top-k prediction
       where the k is the number of gt classes
    """
    num_pos_cls = max(labels_per_im.sum().int().item(), 1)
    cls_preds = score_per_im.topk(num_pos_cls)[1]
    accuracy_img = labels_per_im[cls_preds].mean()
    return accuracy_img


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

@registry.ROI_WEAK_LOSS.register("WSDDNLoss")
class WSDDNLossComputation(object):
    """ Computes the loss for WSDDN."""
    def __init__(self, cfg):
        self.type = "WSDDN"

    def __call__(self, class_score, det_score, ref_scores, proposals, targets, epsilon=1e-10):
        """
        Arguments:
            class_score (list[Tensor])
            det_score (list[Tensor])
        Returns:
            img_loss (Tensor)
            accuracy_img (Tensor): the accuracy of image-level classification
        """
        class_score = cat(class_score, dim=0)
        class_score = F.softmax(class_score, dim=1)

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        total_loss = 0
        accuracy_img = 0
        for final_score_per_im, targets_per_im in zip(final_score_list, targets):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            total_loss += F.binary_cross_entropy(img_score_per_im, labels_per_im)
            accuracy_img += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)

        total_loss = total_loss / len(final_score_list)
        accuracy_img = accuracy_img / len(final_score_list)
        return dict(loss_img=total_loss), dict(accuracy_img=accuracy_img)


@registry.ROI_WEAK_LOSS.register("RoILoss")
class RoILossComputation(object):
    """ Generic roi-level loss """
    def __init__(self, cfg):
        refine_p = cfg.MODEL.ROI_WEAK_HEAD.OICR_P
        self.type = "RoI_loss"
        if refine_p == 0:
            self.roi_layer = oicr_layer()
        elif refine_p > 0 and refine_p < 1:
            self.roi_layer = mist_layer(refine_p)
        else:
            raise ValueError('please use propoer ratio P.')

    def __call__(self, class_score, det_score, ref_scores, proposals, targets, epsilon=1e-8):
        """
        Arguments:
            class_score (list[Tensor])
            det_score (list[Tensor])
            ref_scores
            proposals
            targets
        Returns:
            return_loss_dict (dictionary): all the losses
            return_acc_dict (dictionary): all the accuracies of image-level classification
        """
        class_score = cat(class_score, dim=0)
        class_score = F.softmax(class_score, dim=1)

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        ref_scores = [rs.split([len(p) for p in proposals]) for rs in ref_scores]

        return_loss_dict = dict(loss_img=0)
        return_acc_dict = dict(acc_img=0)
        num_refs = len(ref_scores)
        for i in range(num_refs):
            return_loss_dict['loss_ref%d'%i] = 0
            return_acc_dict['acc_ref%d'%i] = 0

        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            # MIL loss
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1))
            # Region loss
            for i in range(num_refs):
                source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                lmda = 3 if i == 0 else 1
                pseudo_labels, loss_weights = self.roi_layer(proposals_per_image, source_score, labels_per_im, device)
                return_loss_dict['loss_ref%d'%i] += lmda * torch.mean(F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights)

            with torch.no_grad():
                return_acc_dict['acc_img'] += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)
                for i in range(num_refs):
                    ref_score_per_im = torch.sum(ref_scores[i][idx], dim=0)
                    return_acc_dict['acc_ref%d'%i] += compute_avg_img_accuracy(labels_per_im[1:], ref_score_per_im[1:], num_classes)

        assert len(final_score_list) != 0
        for l, a in zip(return_loss_dict.keys(), return_acc_dict.keys()):
            return_loss_dict[l] /= len(final_score_list)
            return_acc_dict[a] /= len(final_score_list)

        return return_loss_dict, return_acc_dict



def box_intersect(boxes):
    r1 = boxes.repeat(len(boxes),1)
    r2 = boxes.repeat(1,len(boxes)).view(len(boxes)*len(boxes), 4)


    # determine the coordinates of the intersection rectangle
    x_left = torch.stack((r1[:,0], r2[:,0]),dim=1).max(dim=1)[0]

    y_top = torch.stack((r1[:,1], r2[:,1]),dim=1).max(dim=1)[0]

    x_right = torch.stack((r1[:,2], r2[:,2]),dim=1).min(dim=1)[0]
    
    y_bottom = torch.stack((r1[:,3], r2[:,3]),dim=1).min(dim=1)[0]


    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    intersection_area[(x_right < x_left) | (y_bottom < y_top)] = 0

    intersection_area = intersection_area.view(len(boxes),len(boxes))
    
    ind = np.diag_indices(intersection_area.shape[0])

    intersection_area[ind[0], ind[1]] = torch.zeros(intersection_area.shape[0]).cuda()

    return intersection_area

@registry.ROI_WEAK_LOSS.register("RoIRegLoss")
class RoIRegLossComputation(object):
    """ Generic roi-level loss """
    def __init__(self, cfg):
        self.refine_p = cfg.MODEL.ROI_WEAK_HEAD.OICR_P
        self.contra = cfg.SOLVER.CONTRA

        if self.refine_p > 0 and self.refine_p < 1 and not self.contra:
            self.mist_layer = mist_layer(self.refine_p)
        self.oicr_layer = oicr_layer()
        self.od_layer = od_layer()

        # for regression
        self.cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
        # for partial labels
        self.roi_refine = cfg.MODEL.ROI_WEAK_HEAD.ROI_LOSS_REFINE
        self.partial_label = cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS
        assert self.partial_label in ['none', 'point', 'scribble']
        self.proposal_scribble_matcher = Matcher(
            0.5, 0.5, allow_low_quality_matches=False,
        )

        self.nms = cfg.nms
        self.sim_lmda = cfg.lmda
        self.pos_update = cfg.pos_update
        self.p_thres = cfg.thres
        self.p_iou = cfg.iou
        self.max_iter = cfg.SOLVER.MAX_ITER

        self.temp = cfg.temp
        if cfg.loss == 'supcon':
            self.sim_loss = Supcon_Loss(self.temp)
        elif cfg.loss == 'supconv2':
            self.sim_loss = [SupConLossV2(cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES -1 , self.temp) for i in range(3)]
        self.output_dir = cfg.OUTPUT_DIR

        self.colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


    def filter_pseudo_labels(self, pseudo_labels, proposal, target):
        """ refine pseudo labels according to partial labels """
        if 'scribble' in target.fields() and self.partial_label=='scribble':
            scribble = target.get_field('scribble')
            match_quality_matrix_async = boxlist_iou_async(scribble, proposal)
            _matched_idxs = self.proposal_scribble_matcher(match_quality_matrix_async)
            pseudo_labels[_matched_idxs < 0] = 0
            matched_idxs = _matched_idxs.clone().clamp(0)
            _labels = target.get_field('labels')[matched_idxs]
            pseudo_labels[pseudo_labels != _labels.long()] = 0

        elif 'click' in target.fields() and self.partial_label=='point':
            clicks = target.get_field('click').keypoints
            clicks_tiled = torch.unsqueeze(torch.cat((clicks, clicks), dim=1), dim=1)
            num_obj = clicks.shape[0]
            box_repeat = torch.cat([proposal.bbox.unsqueeze(0) for _ in range(num_obj)], dim=0)
            diff = clicks_tiled - box_repeat
            matched_ids = (diff[:,:,0] > 0) * (diff[:,:,1] > 0) * (diff[:,:,2] < 0) * (diff[:,:,3] < 0)
            matched_cls = matched_ids.float() * target.get_field('labels').view(-1, 1)
            pseudo_labels_repeat = torch.cat([pseudo_labels.unsqueeze(0) for _ in range(matched_ids.shape[0])])
            correct_idx = (matched_cls == pseudo_labels_repeat.float()).sum(0)
            pseudo_labels[correct_idx==0] = 0

        return pseudo_labels

    def __call__(self, class_score, det_score, ref_scores, ref_bbox_preds, sim_feature, clean_pooled_feats, feature_extractor, model_sim, proposals, targets, iteration=None, epsilon=1e-8):
        class_score = F.softmax(cat(class_score, dim=0), dim=1)
        class_score_list = class_score.split([len(p) for p in proposals])

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)
        detection_score_list = final_det_score.split([len(p) for p in proposals])

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])

        device = class_score.device
        num_classes = class_score.shape[1]

        ref_score = ref_scores.copy()
        for r, r_score in enumerate(ref_scores):
            ref_score[r] = F.softmax(r_score, dim=1)
        avg_score = torch.stack(ref_score).mean(0).detach()
        avg_score_split = avg_score.split([len(p) for p in proposals])

        ref_scores = [rs.split([len(p) for p in proposals]) for rs in ref_scores]
        ref_bbox_preds = [rbp.split([len(p) for p in proposals]) for rbp in ref_bbox_preds]

        return_loss_dict = dict(loss_img=0)
        return_acc_dict = dict(acc_img=0)
        num_refs = len(ref_scores)


        ### Initializes the losses. 
        for i in range(num_refs):
            return_loss_dict['loss_ref_cls%d'%i] = 0
            return_loss_dict['loss_ref_reg%d'%i] = 0
            return_acc_dict['acc_ref%d'%i] = 0

            return_loss_dict['loss_s%d'%i] = 0
            return_acc_dict['loss_l_var%d'%i] = 0
            return_acc_dict['loss_l_dist%d'%i] = 0


        pos_classes = [generate_img_label(num_classes, target.get_field('labels').unique(), device)[1:].eq(1).nonzero(as_tuple=False)[:,0] for target in targets]
        if self.contra:
            
            clean_pooled_feat = clean_pooled_feats.split([len(p) for p in proposals])

            pgt_index = [[[torch.zeros((0), dtype=torch.long, device=device) for x in range(num_classes-1)] for z in range(num_refs)] for y in range(len(targets))]
            pgt_update = [[torch.zeros((0,512,7,7), dtype=torch.float, device=device) for x in range(num_classes-1)] for ixx in range(3)]
            pgt_instance = [[[torch.zeros((0), dtype=torch.long, device=device) for x in range(num_classes-1)] for z in range(num_refs)] for y in range(len(targets))]
            
            ### Normal OICR top scoring proposals selection
            for i in range(num_refs):
                sim_feature_i = sim_feature[i].split([len(p) for p in proposals])
                for idx, (final_score_per_im, pos_classes_per_im, proposals_per_image) in enumerate(zip(final_score_list, pos_classes, proposals)):
                    source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                    proposal_score = source_score[:, 1:].clone()

                    for pos_c in pos_classes_per_im:
                        max_index = torch.argmax(proposal_score[:,pos_c])
                        overlaps, _ = cal_iou(proposals_per_image, max_index, self.p_thres)
                        
                        pgt_instance[idx][i][pos_c] = torch.cat((pgt_instance[idx][i][pos_c], max_index[None])).unique() 

                        pgt_index[idx][i][pos_c] = torch.cat((pgt_index[idx][i][pos_c], overlaps)).unique() ###

                        pgt_update[i][pos_c] = torch.cat((pgt_update[i][pos_c], clean_pooled_feat[idx][max_index[None]]))
               


            ### Discovery 
            foi_a_mais = [0.0,0.0,0.0]
            for i in range(num_refs):
                sim_feature_i = sim_feature[i].split([len(p) for p in proposals])

                for idx, (final_score_per_im, pos_classes_per_im, proposals_per_image) in enumerate(zip(final_score_list, pos_classes, proposals)):
                    d_v = cfg.MODEL.ROI_WEAK_HEAD.D_V 

                    source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                    proposal_score = source_score[:, 1:].clone()
                    for pos_c in pos_classes_per_im:
                        cls_mean =  self.sim_loss[i].means[pos_c]


                        sim_feat = sim_feature_i[idx]
                        dists = torch.norm(cls_mean - sim_feat, dim=1)

                        similar_instances = torch.where(dists < d_v)[0]

                        top_scoring_box = pgt_instance[idx][i][pos_c]

                        if len(similar_instances) > 0:
                            foi_a_mais[i] += 1
                            # print(len(proposals_per_image), proposal_score[:,pos_c].shape, similar_instances.shape, similar_instances.max())
                            sim_close = easy_nms(proposals_per_image, similar_instances, proposal_score[:,pos_c], nms_iou=self.nms)      ### operate nms

                            box_indexes = torch.cat((top_scoring_box, sim_close)).unique() 

                            #### filtering.
                            boxes_  = proposals_per_image[box_indexes].bbox
                            scores_ = proposal_score[box_indexes,pos_c]

                            # # #########
                            areas = box_area(boxes_)[None].repeat(len(boxes_),1)
                            intersections = box_intersect(boxes_)

                            diff = intersections/areas
                   
                            s_keep_fast  = torch.ones(scores_.shape, dtype=bool).cuda()
                            big_diff=diff>0.9
                            if big_diff.any():
                                positions = torch.where(big_diff)
                                bigger_guys = scores_[positions[0]] > scores_[positions[1]]
                                unkeep = torch.cat((positions[0][~bigger_guys], positions[1][bigger_guys])).unique()
                                s_keep_fast[unkeep] = False


                            # #########
                            pgt_instance[idx][i][pos_c] = box_indexes[s_keep_fast]

                        

                       


            for i_ref in range(3):

                l_var, l_dist = self.sim_loss[i_ref](pgt_update[i_ref], feature_extractor, model_sim[i_ref], device, i_ref, cfg)

                return_loss_dict['loss_s'+str(i_ref)] = (self.sim_lmda * (l_var + l_dist) ).squeeze()# + l_center)

                return_acc_dict['loss_l_var'+str(i_ref)]    = l_var.squeeze()
                return_acc_dict['loss_l_dist'+str(i_ref)]   = l_dist.squeeze()

                return_acc_dict['a_mais'+str(i_ref)] = torch.tensor(foi_a_mais[i_ref]).cuda()

        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            # MIL loss
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1))
            # Region loss
            for i in range(num_refs):
                source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                # if not self.contra and self.refine_p == 0:           ### oicr_layer ###
                #     pseudo_labels, loss_weights, regression_targets = self.oicr_layer(
                #         proposals_per_image, source_score, labels_per_im, device, return_targets=True
                #         )
                #     pause()
                # elif not self.contra and self.refine_p > 0:          ### mist layer ###
                #     pseudo_labels, loss_weights, regression_targets = self.mist_layer(
                #         proposals_per_image, source_score, labels_per_im, device, return_targets=True
                #         )
                #     pause()
                # el
                
                if self.contra and self.refine_p == 0:                ### od layer ###
                    pseudo_labels, loss_weights, regression_targets = self.od_layer(
                    proposals_per_image, source_score, labels_per_im, device, pgt_instance[idx][i], return_targets=True
                    )

                if self.roi_refine:
                    pseudo_labels = self.filter_pseudo_labels(pseudo_labels, proposals_per_image, targets_per_im)

                lmda = 3 if i == 0 else 1

                return_loss_dict['loss_ref_cls%d'%i] += lmda * torch.mean(
                    F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights
                )

                # regression
                sampled_pos_inds_subset = torch.nonzero(pseudo_labels>0, as_tuple=False).squeeze(1)
                labels_pos = pseudo_labels[sampled_pos_inds_subset]
                if self.cls_agnostic_bbox_reg:
                    map_inds = torch.tensor([4, 5, 6, 7], device=device)
                else:
                    map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

                box_regression = ref_bbox_preds[i][idx]
                reg_loss = lmda * torch.sum(smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets[sampled_pos_inds_subset],
                    beta=1, reduction=False) * loss_weights[sampled_pos_inds_subset, None]
                )
                reg_loss /= pseudo_labels.numel()
                return_loss_dict['loss_ref_reg%d'%i] += reg_loss

            with torch.no_grad():
                return_acc_dict['acc_img'] += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)
                for i in range(num_refs):
                    ref_score_per_im = torch.sum(ref_scores[i][idx], dim=0)
                    return_acc_dict['acc_ref%d'%i] += compute_avg_img_accuracy(labels_per_im[1:], ref_score_per_im[1:], num_classes)

        assert len(final_score_list) != 0
        for l in return_loss_dict.keys():
            if 'sim' in l:
                continue
            return_loss_dict[l] /= len(final_score_list)

        for a in return_acc_dict.keys():
            return_acc_dict[a] /= len(final_score_list)

        return return_loss_dict, return_acc_dict


def make_roi_weak_loss_evaluator(cfg):
    func = registry.ROI_WEAK_LOSS[cfg.MODEL.ROI_WEAK_HEAD.LOSS]
    return func(cfg)