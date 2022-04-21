import os
import time
import numpy as np
import torch

import data
from utils import frame_utils
from utils.flow_viz import save_vis_flow_tofile

from utils.utils import InputPadder, forward_interpolate
from glob import glob


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission',
                             padding_factor=8,
                             save_vis_flow=False,
                             no_save_flo=False,
                             **kwargs,
                             ):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = data.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters,
                                      flow_init=flow_prev,
                                      test_mode=True)

            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if not no_save_flo:
                frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence

            # Save vis flow
            if save_vis_flow:
                vis_flow_file = output_file.replace('.flo', '.png')
                save_vis_flow_tofile(flow, vis_flow_file)


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission',
                            padding_factor=8,
                            save_vis_flow=False,
                            **kwargs,
                            ):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = data.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flow_pr = model(image1, image2, iters=iters,
                        flow_init=None,
                        test_mode=True)[-1]

        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)

        # Save vis flow
        if save_vis_flow:
            vis_flow_file = output_filename
            save_vis_flow_tofile(flow, vis_flow_file)
        else:
            frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model,
                    iters=24,
                    **kwargs,
                    ):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    results = {}

    val_dataset = data.FlyingChairs(split='validation')

    print('Number of validation image pairs: %d' % len(val_dataset))

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        flow_pr = model(image1, image2, iters=iters, test_mode=True)[-1]  # RAFT

        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all > 1)
    px3 = np.mean(epe_all > 3)
    px5 = np.mean(epe_all > 5)

    print("Validation Chairs EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))

    results['chairs_epe'] = epe
    results['chairs_1px'] = px1
    results['chairs_3px'] = px3
    results['chairs_5px'] = px5

    return results


@torch.no_grad()
def validate_sintel(model,
                    count_time=False,
                    padding_factor=8,
                    iters=32,
                    **kwargs,
                    ):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    if count_time:
        total_time = 0
        num_runs = 100

    for dstype in ['clean', 'final']:
        val_dataset = data.MpiSintel(split='training', dstype=dstype)

        print('Number of validation image pairs: %d' % len(val_dataset))
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            if count_time and val_id >= 5:  # 5 warmup
                torch.cuda.synchronize()
                time_start = time.perf_counter()

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)

            if count_time and val_id >= 5:
                torch.cuda.synchronize()
                total_time += time.perf_counter() - time_start

                if val_id >= num_runs + 4:
                    break

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all > 1)
        px3 = np.mean(epe_all > 3)
        px5 = np.mean(epe_all > 5)

        print("Validation Sintel (%s) EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (dstype, epe, px1, px3, px5))

        dstype = 'sintel_' + dstype

        results[dstype + '_epe'] = np.mean(epe_list)
        results[dstype + '_1px'] = px1
        results[dstype + '_3px'] = px3
        results[dstype + '_5px'] = px5

        if count_time:
            print('Time: %.3fs' % (total_time / num_runs))
            break  # only the clean pass when counting time

    return results


@torch.no_grad()
def validate_kitti(model,
                   padding_factor=8,
                   iters=24,
                   **kwargs,
                   ):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()

    val_dataset = data.KITTI(split='training')
    print('Number of validation image pairs: %d' % len(val_dataset))

    out_list, epe_list = [], []
    results = {}

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()

        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI EPE: %.3f, F1-all: %.3f" % (epe, f1))
    results['kitti_epe'] = epe
    results['kitti_f1'] = f1

    return results


@torch.no_grad()
def inference_on_dir(model, inference_dir,
                     iters=32, warm_start=False, output_path='output',
                     padding_factor=8,
                     paired_data=False,  # dir of paired data instead of a sequence
                     save_flo_flow=False,  # save as .flo for quantative evaluation
                     **kwargs,
                     ):
    """ Inference on a directory """
    model.eval()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filenames = sorted(glob(inference_dir + '/*'))
    print('%d images found' % len(filenames))

    flow_prev, sequence_prev = None, None

    stride = 2 if paired_data else 1

    if paired_data:
        assert len(filenames) % 2 == 0

    for test_id in range(0, len(filenames) - 1, stride):
        image1 = frame_utils.read_gen(filenames[test_id])
        image2 = frame_utils.read_gen(filenames[test_id + 1])

        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        if len(image1.shape) == 2:  # gray image, for example, HD1K
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        if test_id == 0:
            flow_prev = None

        padder = InputPadder(image1.shape, padding_factor=padding_factor)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flow_init = None
        flow_low, flow_pr = model(image1, image2, iters=iters,
                                  flow_init=flow_prev if flow_init is None else flow_init,
                                  test_mode=True)

        if warm_start:
            flow_prev = forward_interpolate(flow_low[0])[None].cuda()

        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_flow.png')

        # Save vis flow
        save_vis_flow_tofile(flow, output_file)

        if save_flo_flow:
            output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_pred.flo')
            frame_utils.writeFlow(output_file, flow)
