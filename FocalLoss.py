from importlib_metadata import requires
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def calculate_focal_loss(confs,gt_labels,alpha,gamma):
    softmax = -(1-F.softmax(confs,dim=1))**gamma

    onehot = F.one_hot(gt_labels,num_classes=confs.size(dim=1))

    onehot = torch.transpose(onehot,2,1)

    log = F.log_softmax(confs,dim=1)

    weight = alpha.view(1,-1,1).cuda()

    focal_loss = weight*softmax*onehot*log
    
    return focal_loss.sum()



class FocalLoss(nn.Module):
    def __init__(self,anchors,alpha,gamma):
        super().__init__()
        self.scale_xy=1.0/anchors.scale_xy
        self.scale_wh=1.0/anchors.scale_wh

        self.sl1_loss=nn.SmoothL1Loss(reduction='none')
        self.anchors=nn.Parameter(anchors(order="xywh").transpose(0,1).unsqueeze(dim=0),requires_grad=False)

        self.alpha=torch.tensor(alpha)
        self.gamma=gamma

    def _loc_vec(self,loc):
        gxy=self.scale_xy*(loc[:, :2, :]-self.anchors[:, :2, :])/self.anchors[:,2:,]
        gwh=self.scale_wh*(loc[:,2:,:]/self.anchors[:,2:,:]).log()
        return torch.cat((gxy,gwh),dim=1).contiguous()

    def forward(self,
            bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
            gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        """
        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # reshape to [batch_size, 4, num_anchors]
        with torch.no_grad():
            to_log = - F.log_softmax(confs, dim=1)[:, 0]

        classification_loss = calculate_focal_loss(confs, gt_labels, self.alpha,self.gamma)

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        total_loss = regression_loss/num_pos + classification_loss/num_pos
        to_log = dict(
            regression_loss=regression_loss/num_pos,
            classification_loss=classification_loss/num_pos,
            total_loss=total_loss
        )
        return total_loss, to_log