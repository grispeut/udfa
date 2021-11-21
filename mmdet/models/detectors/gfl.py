from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmcv.runner import auto_fp16
import torch
from mmdet.core import bbox2result


@DETECTORS.register_module()
class GFL(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 adv_cfg=None,
                 pretrained=None):
        super(GFL, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, adv_cfg, pretrained)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, return_gfl=False, return_fm=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, return_fm=return_fm, **kwargs)
        else:
            return self.forward_test(img, img_metas, return_gfl, return_fm=return_fm, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      img_clean=None,
                      return_fm=False,
                      return_only_fm=False,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        x, fm_backbone = self.extract_feat(img, return_fm=True)
        # if return_bb_fm and not return_neck_mask:
        #     return fm_backbone
        fms = {}
        fms['backbone'] = fm_backbone
        fms['neck'] = x
        if return_only_fm:
            neck_mask_batch, bb_mask_batch, losses_cls, cls_score = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                                   gt_labels, gt_bboxes_ignore, img, fm_backbone=fm_backbone, return_neck_mask=True, return_only_fm=return_only_fm)
            fms['losses_cls'] = losses_cls
            fms['cls_score'] = cls_score
        else:
            losses, neck_mask_batch, bb_mask_batch = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore, img, fm_backbone=fm_backbone, return_neck_mask=True, return_only_fm=return_only_fm)
        fms['neck_mask'] = neck_mask_batch
        fms['bb_mask'] = bb_mask_batch

        if return_only_fm:
            return fms
        if return_fm:
            return losses, fms
        return losses

    def forward_test(self, imgs, img_metas, return_gfl, return_fm, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], return_gfl, return_fm, **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_metas, return_gfl, return_fm, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        if return_fm:
            x, fm_backbone = self.extract_feat(img, return_fm=True)
        else:
            x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if return_gfl and return_fm:
            return outs, x, fm_backbone
        elif return_gfl:
            return outs
        elif return_fm:
            return x, fm_backbone
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
