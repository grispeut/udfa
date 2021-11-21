from abc import ABCMeta, abstractmethod

import torch.nn as nn
import torch


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def _map_roi_levels(self, rois, num_levels):
        scale = torch.sqrt(
            (rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1))
        target_lvls = torch.floor(torch.log2(scale / 56 + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def get_gt_mask(self, featmaps, gt_bboxes, fm_backbone=None):
        device = featmaps[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in featmaps]
        featmap_strides = self.anchor_generator.strides
        imit_range = [0, 0, 0, 0, 0]
        with torch.no_grad():
            mask_batch = []
            bb_mask_batch = []
            for batch in range(len(gt_bboxes)):
                mask_level = []
                target_lvls = self._map_roi_levels(gt_bboxes[batch], len(featmap_sizes))
                if fm_backbone is not None:
                    gt_level = gt_bboxes[batch]
                    h, w = fm_backbone.shape[-2], fm_backbone.shape[-1]
                    mask_per_img = torch.zeros([h, w], dtype=torch.double).to(device)
                    for ins in range(gt_level.shape[0]):
                        stride = featmap_strides[0][0] * featmap_sizes[0][0] / h
                        gt_level_map = gt_level[ins] / stride
                        lx = max(int(gt_level_map[0]), 0)
                        rx = min(int(gt_level_map[2]), w)
                        ly = max(int(gt_level_map[1]), 0)
                        ry = min(int(gt_level_map[3]), h)
                        if (lx == rx) or (ly == ry):
                            mask_per_img[ly, lx] += 1
                        else:
                            mask_per_img[ly:ry, lx:rx] += 1
                    mask_per_img = (mask_per_img > 0).double()
                    bb_mask_batch.append(mask_per_img)

                for level in range(len(featmap_sizes)):
                    gt_level = gt_bboxes[batch][target_lvls==level]  # gt_bboxes: BatchsizexNpointx4coordinate
                    h, w = featmap_sizes[level][0], featmap_sizes[level][1]
                    mask_per_img = torch.zeros([h, w], dtype=torch.double).to(device)
                    for ins in range(gt_level.shape[0]):
                        gt_level_map = gt_level[ins] / featmap_strides[level][0]
                        lx = max(int(gt_level_map[0]) - imit_range[level], 0)
                        rx = min(int(gt_level_map[2]) + imit_range[level], w)
                        ly = max(int(gt_level_map[1]) - imit_range[level], 0)
                        ry = min(int(gt_level_map[3]) + imit_range[level], h)
                        if (lx == rx) or (ly == ry):
                            mask_per_img[ly, lx] += 1
                        else:
                            mask_per_img[ly:ry, lx:rx] += 1
                    mask_per_img = (mask_per_img > 0).double()
                    mask_level.append(mask_per_img)
                mask_batch.append(mask_level)

            mask_batch_level = []
            for level in range(len(mask_batch[0])):
                tmp = []
                for batch in range(len(mask_batch)):
                    tmp.append(mask_batch[batch][level])
                mask_batch_level.append(torch.stack(tmp, dim=0))
            if fm_backbone is not None:
                bb_mask_batch = torch.stack(bb_mask_batch, dim=0)
                return mask_batch_level, bb_mask_batch

        return mask_batch_level

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      return_neck_mask=False,
                      return_only_fm=False,
                      fm_backbone=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        if return_neck_mask:
            neck_mask_batch, bb_mask_batch = self.get_gt_mask(x, gt_bboxes, fm_backbone)

        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)

        if return_only_fm:
            losses_cls = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, return_only_fm=True)
            return neck_mask_batch, bb_mask_batch, losses_cls, outs[0]

        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)

            if return_neck_mask:
                return losses, proposal_list, neck_mask_batch, bb_mask_batch

            return losses, proposal_list
