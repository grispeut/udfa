from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import random
import mmcv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.utils import print_log

from mmdet.utils import get_root_logger
from functools import partial
from imagecorruptions import corrupt

def ImageCorrupt(img):
    idx1 = random.randint(-1, 18)
    ser1 = random.randint(1, 1)
    if not idx1 in [-1]:
        img = corrupt(img, corruption_number=idx1, severity=ser1)
    return img

def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status
to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')

status_list = []
status_pos = 0
def status_convert(m):
    if isinstance(m, nn.BatchNorm2d):
        global status_pos
        if status_list[status_pos]:
            m.training = not m.training
        status_pos += 1

def status_print(m, inital=False, show_status=False):
    if isinstance(m, nn.BatchNorm2d):
        if inital:
            status_list.append(m.training)
        if show_status:
            print(m.training)

status_initial = partial(status_print, inital=True)
status_show = partial(status_print, show_status=True)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def init_teacher_detector(config, checkpoint=None, device='cuda:1'):
    from mmcv.runner import load_checkpoint
    from mmdet.core import get_classes
    from mmdet.models import build_detector
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    # config.model.pretrained = None
    model = build_detector(config.model, train_cfg=config.train_cfg, test_cfg=config.test_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(torch.cuda.current_device())
    print('teacher model device:',torch.cuda.current_device())
    model.eval()
    return model

class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, adv_cfg=None):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False
        if adv_cfg is not None:
            self.adv_flag = adv_cfg.get('adv_flag', False)
            self.kdfa = adv_cfg.get('kdfa', False)
            self.ssfa = adv_cfg.get('ssfa', False)
            self.kdss = adv_cfg.get('kdss', False)
            self.clean_kd = adv_cfg.get('clean_kd', False)
            self.tod = adv_cfg.get('tod', False)
            self.mixbn = adv_cfg.get('mixbn', False)
            self.mixbn1 = adv_cfg.get('mixbn1', False)
            self.freeze_adv_bn = adv_cfg.get('freeze_adv_bn', False)
            self.first_iter = True
            if self.mixbn1:
                self.mixbn = False
            self.split_backward = adv_cfg.get('split_backward', False)

            self.kd_neck_w = adv_cfg.get('kd_neck_w', 0.)
            self.kd_neck_bg_w = adv_cfg.get('kd_neck_bg_w', 0.)
            self.kd_neck_all_w = adv_cfg.get('kd_neck_all_w', 0.)
            self.kd_bb_w = adv_cfg.get('kd_bb_w', 0.)
            self.kd_bb_fg_w = adv_cfg.get('kd_bb_fg_w', 0.)
            self.kd_bb_bg_w = adv_cfg.get('kd_bb_bg_w', 0.)
            self.ck_w = adv_cfg.get('ck_w', 0.5)

            self.low_size = adv_cfg.get('low_size', 2)
            self.high_size = adv_cfg.get('high_size', 2)
            self.epsilon = adv_cfg.get('epsilon', 8)
            self.num_steps = adv_cfg.get('num_steps', 1)
            self.all_bp = adv_cfg.get('all_bp', True)
            self.sf = adv_cfg.get('sf', 0.5)
            self.random_start = adv_cfg.get('random_start', False)

            self.num_necks = adv_cfg.get('num_necks', 5)
            self.kd_config = adv_cfg.get('kd_config', None)
            self.ss_config = adv_cfg.get('ss_config', None)
            if self.kd_config is not None:
                self.kd_weights = adv_cfg.get('kd_weights', None)
                self.kd_model = init_teacher_detector(self.kd_config, self.kd_weights)
            if self.ss_config is not None:
                self.ss_weights = adv_cfg.get('ss_weights', None)
                self.momentum = adv_cfg.get('momentum', 1.)
                self.ss_model = init_teacher_detector(self.ss_config, self.ss_weights)

            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        else:
            self.adv_flag = False
            self.mixbn = False


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            logger = get_root_logger()
            print_log(f'load model from: {pretrained}', logger=logger)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
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
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses, adv_flag=False, sf=1., tod=False):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        if tod:
            loss_cls = 0
            loss_loc = 0
            for _key, _value in log_vars.items():
                if ('loss' in _key and 'ld' not in _key and 'kd' not in _key and 'loss_ss' not in _key):
                    if 'cls' in _key:
                        loss_cls = loss_cls + _value
                    elif 'bbox' in _key or 'dfl' in _key:
                        loss_loc = loss_loc + _value
            return loss_cls, loss_loc
        elif adv_flag:
            loss = sum(_value*sf for _key, _value in log_vars.items()
                       if ('loss' in _key and 'ld' not in _key and 'kd' not in _key and 'loss_ss' not in _key))
        else:
            loss = 0.
            for _key, _value in log_vars.items():
                if 'loss' in _key:
                    if ('ld' not in _key and 'kd' not in _key and 'loss_ss' not in _key):
                        loss = loss + _value*sf
                    else:
                        loss = loss + _value

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer, epoch):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        device = data['img'].device
        if self.adv_flag:
            if self.freeze_adv_bn and self.first_iter:
                self.apply(status_initial)
                self.first_iter = False

            split_backward = self.split_backward
            vis = False
            self.step_size = random.randint(self.low_size, self.high_size)

            img_metas = data['img_metas']
            img = data['img']
            kwargs = {'gt_bboxes':data['gt_bboxes'], 'gt_labels':data['gt_labels']}
            self.pixel_means = torch.from_numpy(img_metas[0]['img_norm_cfg']['mean']).type_as(img)
            self.pixel_stds = torch.from_numpy(img_metas[0]['img_norm_cfg']['std']).type_as(img)

            im_adv = img.detach().clone()
            im_adv = im_adv.permute(0, 2, 3, 1).contiguous()
            im_adv = torch.mul(im_adv, self.pixel_stds)
            im_adv = torch.add(im_adv, self.pixel_means)
            if vis:
                im_vis = im_adv[0].detach().cpu().numpy().astype(np.uint8)
                cv2.imwrite('debug.jpg', im_vis)

            x_raw = im_adv.clone()
            im_adv.requires_grad_()
            loss_clean = torch.tensor(0).type_as(im_adv)
            for step in range(self.num_steps):
                if self.random_start and step == 0:
                    delta = torch.zeros_like(im_adv).to(device)
                    delta.uniform_(-self.epsilon/2, self.epsilon/2)
                    x = im_adv + delta
                    x = torch.clamp(x, 0, 255)
                    x = torch.sub(x, self.pixel_means)
                else:
                    x = torch.sub(im_adv, self.pixel_means)
                x = torch.div(x, self.pixel_stds)
                x = x.permute(0, 3, 1, 2).contiguous()

                if self.tod:
                    if self.mixbn:
                        self.apply(to_adv_status)
                    loss_dict = self.forward_train(x, img_metas, **kwargs)
                    loss_cls, loss_loc = self._parse_losses(loss_dict, adv_flag=True, sf=1., tod=True)
                    x_grad_cls = torch.autograd.grad(loss_cls, [im_adv], retain_graph=True)[0]
                    x_grad_loc = torch.autograd.grad(loss_loc, [im_adv], retain_graph=False)[0]
                    im_adv_cls = im_adv.data + torch.sign(x_grad_cls) * self.step_size
                    im_adv_cls.data = torch.min(torch.max(im_adv_cls.data, x_raw - self.epsilon), x_raw + self.epsilon)
                    im_adv_cls.data.clamp_(0, 255)
                    im_adv_loc = im_adv.data + torch.sign(x_grad_loc) * self.step_size
                    im_adv_loc.data = torch.min(torch.max(im_adv_loc.data, x_raw - self.epsilon), x_raw + self.epsilon)
                    im_adv_loc.data.clamp_(0, 255)
                    with torch.no_grad():
                        data_temp = data.copy()
                        im_adv_cls_temp = torch.sub(im_adv_cls, self.pixel_means)
                        im_adv_cls_temp = torch.div(im_adv_cls_temp, self.pixel_stds)
                        im_adv_cls_temp = im_adv_cls_temp.permute(0, 3, 1, 2).contiguous()
                        data_temp['img'] = im_adv_cls_temp
                        loss_dict = self(**data_temp)
                        loss_cls, _ = self._parse_losses(loss_dict, adv_flag=True, sf=1.)
                        data_temp = data.copy()
                        im_adv_loc_temp = torch.sub(im_adv_loc, self.pixel_means)
                        im_adv_loc_temp = torch.div(im_adv_loc_temp, self.pixel_stds)
                        im_adv_loc_temp = im_adv_loc_temp.permute(0, 3, 1, 2).contiguous()
                        data_temp['img'] = im_adv_loc_temp
                        loss_dict = self(**data_temp)
                        loss_loc, _ = self._parse_losses(loss_dict, adv_flag=True, sf=1.)
                        if loss_cls > loss_loc:
                            im_adv = im_adv_cls
                        else:
                            im_adv = im_adv_loc

                else:
                    if self.kdss:
                        if step == 0 and not self.mixbn:
                            if self.mixbn1:
                                self.apply(to_clean_status)
                            losses_clean, fm_ss = self.forward_train(x, img_metas, return_fm=True, **kwargs)
                            fm_ss_bk = {}
                            fm_ss_bk['neck'] = []
                            for neck_i in fm_ss['neck']:
                                fm_ss_bk['neck'].append(neck_i.detach().clone())
                            fm_ss_bk['backbone'] = fm_ss['backbone'].detach().clone()
                            loss_clean, _ = self._parse_losses(losses_clean, adv_flag=False, sf=1.)
                            loss_temp = loss_clean * 0.5
                            loss_temp.backward(retain_graph=False)
                            x_grad = im_adv.grad
                        else:
                            if self.mixbn:
                                self.apply(to_adv_status)
                            loss_dict = self.forward_train(x, img_metas, **kwargs)
                            loss_temp, _ = self._parse_losses(loss_dict, adv_flag=False, sf=1.)
                            x_grad = torch.autograd.grad(loss_temp, [im_adv], retain_graph=False)[0]

                    else:
                        if step == 0:
                            losses_clean = self.forward_train(x, img_metas, **kwargs)
                            loss_clean, _ = self._parse_losses(losses_clean, adv_flag=False, sf=1.)
                            x_grad = torch.autograd.grad(loss_clean, [im_adv], retain_graph=True)[0]
                        else:
                            loss_dict = self.forward_train(x, img_metas, **kwargs)
                            loss_temp, _ = self._parse_losses(loss_dict, adv_flag=False, sf=1.)
                            x_grad = torch.autograd.grad(loss_temp, [im_adv], retain_graph=False)[0]

                    eta = torch.sign(x_grad) * self.step_size
                    im_adv.data = im_adv.data + eta
                    im_adv.data = torch.min(torch.max(im_adv.data, x_raw - self.epsilon), x_raw + self.epsilon)
                    im_adv.data.clamp_(0, 255)
                    if not (self.kdss and self.num_steps==1):
                        optimizer.zero_grad()

            im_adv = torch.sub(im_adv, self.pixel_means)
            im_adv = torch.div(im_adv, self.pixel_stds)
            im_adv = im_adv.permute(0, 3, 1, 2).contiguous()
            data['img'] = im_adv.detach().clone()

            if self.kdss:
                s0, s1 = 1., 1.
                if self.kdfa and self.ssfa:
                    s0, s1 = 0.5, 0.5
                assert self.num_steps == 1
                data['return_fm'] = True
                if self.mixbn or self.mixbn1:
                    self.apply(to_adv_status)
                global status_pos
                if self.freeze_adv_bn:
                    status_pos = 0
                    self.apply(status_convert)
                    # print('==========')
                    # self.apply(status_show)
                losses, fm_adv = self(**data)
                if self.freeze_adv_bn:
                    status_pos = 0
                    self.apply(status_convert)
                    # print('==========')
                    # self.apply(status_show)
                loss_adv, log_vars = self._parse_losses(losses, adv_flag=False, sf=1.)
                log_vars['loss_adv'] = loss_adv.item()
                fm_adv_bb = fm_adv['backbone']
                fm_adv_neck_ori = fm_adv['neck']
                fm_adv_neck_mask_ori = fm_adv['neck_mask']
                this_neck_nums = len(fm_adv_neck_ori)
                neck_start = max(this_neck_nums-self.num_necks, 0)
                fm_adv_neck = fm_adv_neck_ori[neck_start:]
                fm_adv_neck_mask = fm_adv_neck_mask_ori[neck_start:]

                data_clean = data.copy()
                data_clean['img'] = img
                data_clean['return_only_fm'] = True
                loss_kd = torch.Tensor([0]).to(device)
                loss_ss = torch.Tensor([0]).to(device)
                loss_clean_kd = torch.Tensor([0]).to(device)
                if self.kd_bb_w > 1e-10:
                    fm_adv_bb_norm = self.global_avg_pool(fm_adv_bb)
                    fm_adv_bb_norm = fm_adv_bb_norm.view(fm_adv_bb_norm.size(0), -1)
                    fm_adv_bb_norm = F.normalize(fm_adv_bb_norm, dim=1)
                if self.kdfa:
                    with torch.no_grad():
                        fm_kd = self.kd_model(**data_clean)
                        fm_kd_bb = fm_kd['backbone']
                        fm_kd_neck_ori = fm_kd['neck']
                        fm_kd_neck = fm_kd_neck_ori[neck_start:]
                    loss_kd_bb = torch.Tensor([0]).to(device)
                    loss_kd_neck = torch.Tensor([0]).to(device)
                    loss_kd_neck_bg = torch.Tensor([0]).to(device)
                    loss_kd_neck_all = torch.Tensor([0]).to(device)
                    if self.kd_bb_w > 1e-10:
                        fm_kd_bb_norm = self.global_avg_pool(fm_kd_bb)
                        fm_kd_bb_norm = fm_kd_bb_norm.view(fm_kd_bb_norm.size(0), -1)
                        fm_kd_bb_norm = F.normalize(fm_kd_bb_norm, dim=1)
                        kd_sim = torch.einsum('nc,nc->n', [fm_adv_bb_norm, fm_kd_bb_norm])
                        kd_sim.data.clamp_(-1., 1.)
                        loss_kd_bb = (1. - kd_sim).mean().view(-1) * self.kd_bb_w

                    if self.kd_neck_w > 1e-10 or self.kd_neck_bg_w > 1e-10 or self.kd_neck_all_w > 1e-10:
                        for i, _neck_feat in enumerate(fm_adv_neck):
                            mask_hint = fm_adv_neck_mask[i]
                            mask_hint = mask_hint.unsqueeze(1).repeat(1, _neck_feat.size(1), 1, 1)
                            quality_hint = 1.
                            norms = max(1.0, mask_hint.sum()) * 8
                            neck_feat_adapt = _neck_feat
                            norms_back = max(1.0, (1 - mask_hint).sum()) * 8
                            norms_all = torch.ones_like(mask_hint).sum() * 8
                            if self.kd_neck_bg_w > 1e-10:
                                loss_kd_neck_bg += (torch.pow(neck_feat_adapt - fm_kd_neck[i], 2) * quality_hint *
                                                    (1 - mask_hint)).sum() / norms_back
                            if self.kd_neck_w > 1e-10:
                                loss_kd_neck += (torch.pow(neck_feat_adapt - fm_kd_neck[i], 2) * quality_hint * mask_hint).sum() / norms
                            if self.kd_neck_all_w > 1e-10:
                                loss_kd_neck_all += (torch.pow(neck_feat_adapt - fm_kd_neck[i], 2) * quality_hint).sum() / norms_all
                        loss_kd_neck = loss_kd_neck / len(fm_adv_neck)
                        loss_kd_neck = loss_kd_neck * self.kd_neck_w
                        loss_kd_neck_bg = loss_kd_neck_bg / len(fm_adv_neck)
                        loss_kd_neck_bg = loss_kd_neck_bg * self.kd_neck_bg_w
                        loss_kd_neck_all = loss_kd_neck_all / len(fm_adv_neck)
                        loss_kd_neck_all = loss_kd_neck_all * self.kd_neck_all_w
                    loss_kd = loss_kd_neck + loss_kd_neck_bg + loss_kd_neck_all + loss_kd_bb

                if self.ssfa:
                    loss_ss_bb = torch.Tensor([0]).to(device)
                    loss_ss_neck = torch.Tensor([0]).to(device)
                    loss_ss_neck_bg = torch.Tensor([0]).to(device)
                    loss_ss_neck_all = torch.Tensor([0]).to(device)
                    loss_ss_bb_kd = torch.Tensor([0]).to(device)
                    loss_ss_neck_kd = torch.Tensor([0]).to(device)
                    loss_ss_neck_bg_kd = torch.Tensor([0]).to(device)
                    loss_ss_neck_all_kd = torch.Tensor([0]).to(device)

                    if self.mixbn:
                        with torch.no_grad():
                            self.apply(to_clean_status)
                            fm_ss = self(**data_clean)
                            fm_ss_neck_ori = fm_ss['neck']
                            fm_ss_bb = fm_ss['backbone']
                            fm_ss_neck = fm_ss_neck_ori[neck_start:]
                    else:
                        fm_ss_neck_ori = fm_ss_bk['neck']
                        fm_ss_bb = fm_ss_bk['backbone']
                        fm_ss_neck = fm_ss_neck_ori[neck_start:]

                    if self.kd_bb_w > 1e-10:
                        fm_ss_bb_norm = self.global_avg_pool(fm_ss_bb)
                        fm_ss_bb_norm = fm_ss_bb_norm.view(fm_ss_bb_norm.size(0), -1)
                        fm_ss_bb_norm = F.normalize(fm_ss_bb_norm, dim=1)
                        ss_sim = torch.einsum('nc,nc->n', [fm_adv_bb_norm, fm_ss_bb_norm])
                        ss_sim.data.clamp_(-1., 1.)
                        loss_ss_bb = (1. - ss_sim).mean().view(-1) * self.kd_bb_w

                    if self.kd_neck_w > 1e-10 or self.kd_neck_bg_w > 1e-10 or self.kd_neck_all_w > 1e-10:
                        for i, _neck_feat in enumerate(fm_adv_neck):
                            mask_hint = fm_adv_neck_mask[i]
                            mask_hint = mask_hint.unsqueeze(1).repeat(1, _neck_feat.size(1), 1, 1)
                            quality_hint = 1.
                            norms = max(1.0, mask_hint.sum()) * 8
                            neck_feat_adapt = _neck_feat
                            norms_back = max(1.0, (1 - mask_hint).sum()) * 8
                            norms_all = torch.ones_like(mask_hint).sum() * 8
                            if self.kd_neck_bg_w > 1e-10:
                                loss_ss_neck_bg += (torch.pow(neck_feat_adapt - fm_ss_neck[i], 2) * quality_hint *
                                                    (1 - mask_hint)).sum() / norms_back
                            if self.kd_neck_w > 1e-10:
                                loss_ss_neck += (torch.pow(neck_feat_adapt - fm_ss_neck[i], 2) * quality_hint * mask_hint).sum() / norms
                            if self.kd_neck_all_w > 1e-10:
                                loss_ss_neck_all += (torch.pow(neck_feat_adapt - fm_ss_neck[i], 2) * quality_hint).sum() / norms_all
                        loss_ss_neck = loss_ss_neck / len(fm_adv_neck)
                        loss_ss_neck = loss_ss_neck * self.kd_neck_w
                        loss_ss_neck_bg = loss_ss_neck_bg / len(fm_adv_neck)
                        loss_ss_neck_bg = loss_ss_neck_bg * self.kd_neck_bg_w
                        loss_ss_neck_all = loss_ss_neck_all / len(fm_adv_neck)
                        loss_ss_neck_all = loss_ss_neck_all * self.kd_neck_all_w
                    loss_ss = loss_ss_neck + loss_ss_neck_bg + loss_ss_neck_all + loss_ss_bb

                    if self.clean_kd:
                        if split_backward:
                            loss_temp = loss_adv * 0.5 + s0 * loss_kd + s1 * loss_ss
                            loss_temp.backward()
                        assert self.kdfa
                        if self.mixbn or self.mixbn1:
                            self.apply(to_clean_status)
                        if self.mixbn:
                            data_clean['return_only_fm'] = False
                            data_clean['return_fm'] = True
                            losses_clean, fm_ss = self(**data_clean)
                            loss_clean, _ = self._parse_losses(losses_clean, adv_flag=False, sf=1.)
                        else:
                            fm_ss = self(**data_clean)
                        fm_ss_neck_kd_ori = fm_ss['neck']
                        fm_ss_neck_kd = fm_ss_neck_kd_ori[neck_start:]
                        fm_ss_bb_kd = fm_ss['backbone']

                        if self.kd_bb_w > 1e-10:
                            fm_ss_bb_kd_norm = self.global_avg_pool(fm_ss_bb_kd)
                            fm_ss_bb_kd_norm = fm_ss_bb_kd_norm.view(fm_ss_bb_kd_norm.size(0), -1)
                            fm_ss_bb_kd_norm = F.normalize(fm_ss_bb_kd_norm, dim=1)
                            ss_kd_sim = torch.einsum('nc,nc->n', [fm_ss_bb_kd_norm, fm_kd_bb_norm])
                            ss_kd_sim.data.clamp_(-1., 1.)
                            loss_ss_bb_kd = (1. - ss_kd_sim).mean().view(-1) * self.kd_bb_w

                        if self.kd_neck_w > 1e-10 or self.kd_neck_bg_w > 1e-10 or self.kd_neck_all_w > 1e-10:
                            for i, _neck_feat in enumerate(fm_kd_neck):
                                mask_hint = fm_adv_neck_mask[i]
                                mask_hint = mask_hint.unsqueeze(1).repeat(1, _neck_feat.size(1), 1, 1)
                                quality_hint = 1.
                                norms = max(1.0, mask_hint.sum()) * 8
                                neck_feat_adapt = _neck_feat
                                norms_back = max(1.0, (1 - mask_hint).sum()) * 8
                                norms_all = torch.ones_like(mask_hint).sum() * 8
                                if self.kd_neck_bg_w > 1e-10:
                                    loss_ss_neck_bg_kd += (torch.pow(neck_feat_adapt - fm_ss_neck_kd[i], 2) * quality_hint *
                                                           (1 - mask_hint)).sum() / norms_back
                                if self.kd_neck_w > 1e-10:
                                    loss_ss_neck_kd += (torch.pow(neck_feat_adapt - fm_ss_neck_kd[i], 2) * quality_hint * mask_hint).sum() / norms
                                if self.kd_neck_all_w > 1e-10:
                                    loss_ss_neck_all_kd += (torch.pow(neck_feat_adapt - fm_ss_neck_kd[i], 2) * quality_hint).sum() / norms_all
                        loss_ss_neck_kd = loss_ss_neck_kd / len(fm_adv_neck)
                        loss_ss_neck_kd = loss_ss_neck_kd * self.kd_neck_w
                        loss_ss_neck_bg_kd = loss_ss_neck_bg_kd / len(fm_adv_neck)
                        loss_ss_neck_bg_kd = loss_ss_neck_bg_kd * self.kd_neck_bg_w
                        loss_ss_neck_all_kd = loss_ss_neck_all_kd / len(fm_adv_neck)
                        loss_ss_neck_all_kd = loss_ss_neck_all_kd * self.kd_neck_all_w
                        loss_clean_kd = loss_ss_neck_kd + loss_ss_neck_bg_kd + loss_ss_neck_all_kd + loss_ss_bb_kd
                        # print(loss_clean_kd.item())

                if self.clean_kd:
                    if not split_backward:
                        if self.mixbn:
                            loss = (loss_clean + loss_adv) * 0.5 + s0 * loss_kd + s1 * loss_ss + self.ck_w * loss_clean_kd
                            log_vars['loss'] = loss.item()
                        else:
                            loss = loss_adv * 0.5 + s0 * loss_kd + s1 * loss_ss + self.ck_w * loss_clean_kd
                            log_vars['loss'] = loss.item() + (loss_clean*0.5).item()
                    else:
                        if self.mixbn:
                            loss = self.ck_w * loss_clean_kd + loss_clean * 0.5
                            log_vars['loss'] = loss.item() + (loss_adv * 0.5 + s0 * loss_kd + s1 * loss_ss).item()
                        else:
                            loss = self.ck_w * loss_clean_kd
                            log_vars['loss'] = loss.item() + (loss_clean*0.5+loss_adv * 0.5 + s0 * loss_kd + s1 * loss_ss).item()
                else:
                    loss = loss_adv * 0.5 + s0 * loss_kd + s1 * loss_ss
                    log_vars['loss'] = loss.item() + (loss_clean*0.5).item()
                # if self.mixbn:
                #     loss.backward()
                #     data_clean['return_only_fm'] = False
                #     data_clean['return_fm'] = False
                #     self.apply(to_clean_status)
                #     losses_clean = self(**data_clean)
                #     loss_clean, _ = self._parse_losses(losses_clean, adv_flag=False, sf=1.)
                #     loss = loss_clean * 0.5
                if dist.is_available() and dist.is_initialized():
                    loss_kd = loss_kd.data.clone()
                    dist.all_reduce(loss_kd.div_(dist.get_world_size()))
                    loss_ss = loss_ss.data.clone()
                    dist.all_reduce(loss_ss.div_(dist.get_world_size()))
                log_vars['loss_clean_kd'] = loss_clean_kd.item()
                log_vars['loss_kd'] = loss_kd.item()
                log_vars['loss_ss'] = loss_ss.item()
                # print(loss_kd, loss_adv, loss_clean)
                # log_vars['s0'] = s0
                # log_vars['s1'] = s1

            elif self.tod:
                data_clean = data.copy()
                data_clean['img'] = img
                if self.mixbn:
                    self.apply(to_clean_status)
                loss_dict = self(**data_clean)
                loss_clean, _ = self._parse_losses(loss_dict, adv_flag=False, sf=1.)
                if self.split_backward:
                    loss_temp = loss_clean * 0.5
                    loss_temp.backward()
                if self.mixbn:
                    self.apply(to_adv_status)
                loss_dict = self(**data)
                loss_adv, log_vars = self._parse_losses(loss_dict, adv_flag=False, sf=1.)
                if self.split_backward:
                    loss = loss_adv * 0.5
                else:
                    loss = loss_adv * 0.5 + loss_clean * 0.5
                log_vars['loss'] = loss_adv.item() * 0.5 + loss_clean.item() * 0.5

            else:
                loss_dict = self(**data)
                loss_adv, log_vars = self._parse_losses(loss_dict, adv_flag=False, sf=1.)
                loss = loss_adv * 0.5 + loss_clean * 0.5
                log_vars['loss'] = loss_adv.item() * 0.5 + loss_clean.item() * 0.5

            log_vars['loss_adv'] = loss_adv.item()
            log_vars['loss_clean'] = loss_clean.item()
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        else:
            losses = self(**data)
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        mmcv.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
