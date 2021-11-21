import os.path as osp
import pickle
import shutil
import tempfile
import time

from collections import OrderedDict

import numpy as np
import mmcv
import torch
import torch.nn as nn
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.parallel import scatter, MMDataParallel
from mmdet.core import encode_mask_results, tensor2imgs

from functools import partial
# from bbox_nms import batched_nms

def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status
to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')

def pack_detections(detections: [np.ndarray]) -> (torch.Tensor, torch.Tensor):
    all_labels = [[i] * len(bbox) for i, bbox in enumerate(detections)]
    all_labels = torch.tensor(sum(all_labels, []))  # flat
    all_detections = torch.from_numpy(np.vstack(detections))
    return all_detections, all_labels

def _parse_losses(losses, adv_flag=False, sf=1.):
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

    if adv_flag:
        loss = sum(_value * sf for _key, _value in log_vars.items()
                   if ('loss' in _key and 'ld' not in _key and 'kd' not in _key))
    else:
        loss = 0.
        for _key, _value in log_vars.items():
            if 'loss' in _key:
                if ('ld' not in _key and 'kd' not in _key and 'loss_ss' not in _key):
                    loss = loss + _value*sf
                else:
                    loss = loss + _value

        # loss = sum(_value * sf for _key, _value in log_vars.items()
        #            if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars

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

status_list = []
def status_print(m, inital=False, show_status=False):
    if isinstance(m, nn.BatchNorm2d):
        if inital:
            status_list.append(m.training)
        if show_status:
            print(m.training)
status_initial = partial(status_print, inital=True)
status_show = partial(status_print, show_status=True)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    test_adv_cfg=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    adv_flag = False
    mixbn = False
    kd_config = None
    if test_adv_cfg is not None:
        adv_flag = test_adv_cfg.adv_flag
        step_size = test_adv_cfg.step_size
        epsilon = test_adv_cfg.epsilon
        num_steps = test_adv_cfg.num_steps
        mixbn = test_adv_cfg.get('mixbn', False)
        kd_config = test_adv_cfg.get('kd_config', None)

    for i, data in enumerate(data_loader):
        if adv_flag:
            data_t = data.copy()

            sample = scatter(data_t, [torch.cuda.current_device()])[0]
            img = sample['img']
            pixel_means = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['mean']).type_as(img)
            pixel_stds = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['std']).type_as(img)

            im_adv = img.detach().clone()
            im_adv = im_adv.permute(0, 2, 3, 1).contiguous()
            im_adv = torch.mul(im_adv, pixel_stds)
            im_adv = torch.add(im_adv, pixel_means)
            vis = False
            if vis:
                import numpy as np
                im_vis = im_adv[0].detach().cpu().numpy().astype(np.uint8)
                import cv2
                cv2.imwrite('vis.jpg', im_vis)

            x_raw = im_adv.clone()
            im_adv.requires_grad_()
            for step in range(num_steps):
                if mixbn:
                    if step == 0:
                        model.apply(to_clean_status)
                    else:
                        model.apply(to_clean_status)

                x = torch.sub(im_adv, pixel_means)
                x = torch.div(x, pixel_stds)
                x = x.permute(0, 3, 1, 2).contiguous()
                sample['img'] = x
                loss_dict = model(return_loss=True, **sample)
                loss_adv, _ = _parse_losses(loss_dict, adv_flag=True, sf=1.)

                x_grad = torch.autograd.grad(loss_adv, [im_adv], retain_graph=False)[0]

                eta = torch.sign(x_grad) * step_size
                im_adv.data = im_adv.data + eta
                im_adv.data = torch.min(torch.max(im_adv.data, x_raw - epsilon), x_raw + epsilon)
                im_adv.data.clamp_(0, 255)

            im_adv = torch.sub(im_adv, pixel_means)
            im_adv = torch.div(im_adv, pixel_stds)
            im_adv = im_adv.permute(0, 3, 1, 2).contiguous()
            data['img'] = [im_adv.detach().clone()]
        # else:
        #     sample = scatter(data, [torch.cuda.current_device()])[0]
        #     data['img'] = [sample['img']]
        with torch.no_grad():
            if adv_flag:
                if kd_config is None:
                    data.pop('gt_bboxes')
                    data.pop('gt_labels')
                    data['img_metas'] = [data['img_metas']]
                if mixbn:
                    if num_steps > 0:
                        model.apply(to_clean_status)
                    else:
                        model.apply(to_clean_status)
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            # sample = scatter(data, [torch.cuda.current_device()])[0]
            # data['img'] = [sample['img']]
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            if kd_config is not None:
                img_metas = data['img_metas'][0]
            else:
                img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, test_adv_cfg=None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()

    adv_flag = False
    mixbn = False
    if test_adv_cfg is not None:
        adv_flag = test_adv_cfg.adv_flag
        step_size = test_adv_cfg.step_size
        epsilon = test_adv_cfg.epsilon
        num_steps = test_adv_cfg.num_steps
        mixbn = test_adv_cfg.get('mixbn', False)

    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):

        if adv_flag:
            data_t = data.copy()
            sample = scatter(data_t, [torch.cuda.current_device()])[0]
            img = sample['img']
            pixel_means = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['mean']).type_as(img)
            pixel_stds = torch.from_numpy(sample['img_metas'][0]['img_norm_cfg']['std']).type_as(img)

            im_adv = img.detach().clone()
            im_adv = im_adv.permute(0, 2, 3, 1).contiguous()
            im_adv = torch.mul(im_adv, pixel_stds)
            im_adv = torch.add(im_adv, pixel_means)
            vis = False
            if vis:
                import numpy as np
                im_vis = im_adv[0].detach().cpu().numpy().astype(np.uint8)
                import cv2
                cv2.imwrite('vis.jpg', im_vis)

            x_raw = im_adv.clone()
            im_adv.requires_grad_()
            for step in range(num_steps):
                if mixbn:
                    if step == 0:
                        model.apply(to_clean_status)
                    else:
                        model.apply(to_clean_status)

                x = torch.sub(im_adv, pixel_means)
                x = torch.div(x, pixel_stds)
                x = x.permute(0, 3, 1, 2).contiguous()
                sample['img'] = x
                loss_dict = model(return_loss=True, **sample)
                loss_adv, _ = _parse_losses(loss_dict, adv_flag=True, sf=1.)

                x_grad = torch.autograd.grad(loss_adv, [im_adv], retain_graph=False)[0]

                eta = torch.sign(x_grad) * step_size
                im_adv.data = im_adv.data + eta
                im_adv.data = torch.min(torch.max(im_adv.data, x_raw - epsilon), x_raw + epsilon)
                im_adv.data.clamp_(0, 255)

            im_adv = torch.sub(im_adv, pixel_means)
            im_adv = torch.div(im_adv, pixel_stds)
            im_adv = im_adv.permute(0, 3, 1, 2).contiguous()
            data['img'] = [im_adv.detach().clone()]
        # else:
        #     sample = scatter(data, [torch.cuda.current_device()])[0]
        #     data['img'] = [sample['img']]
        with torch.no_grad():
            if adv_flag:
                data.pop('gt_bboxes')
                data.pop('gt_labels')
                data['img_metas'] = [data['img_metas']]
                if mixbn:
                    if num_steps > 0:
                        model.apply(to_clean_status)
                    else:
                        model.apply(to_clean_status)
            result = model(return_loss=False, rescale=True, **data)

            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
