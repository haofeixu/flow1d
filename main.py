import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import os

from data import build_dataset

from flow1d.flow1d import build_model
from loss import criterion
from evaluate import (validate_chairs, validate_sintel, validate_kitti,
                      create_kitti_submission, create_sintel_submission,
                      inference_on_dir,
                      )

from utils.logger import Logger
from utils import misc


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/tmp')
    parser.add_argument('--eval', action='store_true')

    # Dataset
    parser.add_argument('--image_size', default=[368, 496], type=int, nargs='+')
    parser.add_argument('--stage', default='chairs', type=str)
    parser.add_argument('--max_flow', default=400, type=int)
    parser.add_argument('--padding_factor', default=8, type=int)
    parser.add_argument('--val_dataset', default='chairs', type=str, nargs='+')

    # Create Sintel and KITTI submission
    parser.add_argument('--submission', action='store_true',
                        help='Create submission')
    parser.add_argument('--warm_start', action='store_true')
    parser.add_argument('--output_path', default='output', type=str)
    parser.add_argument('--save_vis_flow', action='store_true')
    parser.add_argument('--no_save_flo', action='store_true')

    # Inference on a directory
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--dir_paired_data', action='store_true',
                        help='Paired data in a dir instead of a sequence')
    parser.add_argument('--save_flo_flow', action='store_true')

    # Training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--lr_warmup', default=0.05, type=float,
                        help='Percentage of lr warmup')
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=5000, type=int)
    parser.add_argument('--save_ckpt_freq', default=50000, type=int)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--no_resume_optimizer', action='store_true')
    parser.add_argument('--no_latest_ckpt', action='store_true')
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
    parser.add_argument('--freeze_bn', action='store_true')

    parser.add_argument('--train_iters', default=12, type=int)
    parser.add_argument('--val_iters', default=12, type=int)

    # Flow1D
    parser.add_argument('--downsample_factor', default=8, type=int)
    parser.add_argument('--feature_channels', default=256, type=int)
    parser.add_argument('--corr_radius', default=32, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--context_dim', default=128, type=int)
    parser.add_argument('--gamma', default=0.8, type=float,
                        help='Exponential weighting')

    parser.add_argument('--mixed_precision', action='store_true')

    # Distributed training
    parser.add_argument('--local_rank', default=0, type=int)

    # Misc
    parser.add_argument('--count_time', action='store_true')

    return parser


def main(args):
    if not args.eval and not args.submission and args.inference_dir is None:
        print('PyTorch version:', torch.__version__)
        print(args)
        misc.save_args(args)
        misc.check_path(args.checkpoint_dir)
        misc.save_command(args.checkpoint_dir)

    misc.check_path(args.output_path)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = build_model(args).to(device)

    if not args.eval:
        print('Model definition:')
        print(model)

    if torch.cuda.device_count() > 1:
        print('Use %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

        model_without_ddp = model.module
    else:
        model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params:', num_params)
    if not args.eval and not args.submission and args.inference_dir is None:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    start_step = 0

    # resume checkpoints
    if args.resume:
        print('Load checkpoint: %s' % args.resume)
        checkpoint = torch.load(args.resume)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_without_ddp.load_state_dict(weights, strict=False)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

    # evaluate
    if args.eval:
        if 'chairs' in args.val_dataset:
            validate_chairs(model_without_ddp,
                            iters=args.val_iters,
                            )
        elif 'sintel' in args.val_dataset:
            validate_sintel(model_without_ddp,
                            iters=args.val_iters,
                            padding_factor=args.padding_factor,
                            count_time=args.count_time,
                            )
        elif 'kitti' in args.val_dataset:
            validate_kitti(model_without_ddp,
                           iters=args.val_iters,
                           padding_factor=args.padding_factor,
                           )
        else:
            raise ValueError(f'Dataset type {args.val_dataset} is not supported')

        return

    # create sintel and kitti submission
    if args.submission:
        if args.val_dataset[0] == 'sintel':
            create_sintel_submission(model_without_ddp,
                                     iters=args.val_iters,
                                     warm_start=args.warm_start,
                                     output_path=args.output_path,
                                     padding_factor=args.padding_factor,
                                     save_vis_flow=args.save_vis_flow,
                                     no_save_flo=args.no_save_flo,
                                     )
        elif args.val_dataset[0] == 'kitti':
            create_kitti_submission(model_without_ddp,
                                    iters=args.val_iters,
                                    output_path=args.output_path,
                                    padding_factor=args.padding_factor,
                                    save_vis_flow=args.save_vis_flow,
                                    )
        else:
            raise ValueError(f'Not supported dataset for submission')

        return

    # inferece on a dir
    if args.inference_dir is not None:
        inference_on_dir(model_without_ddp,
                         inference_dir=args.inference_dir,
                         iters=args.val_iters,
                         warm_start=args.warm_start,
                         output_path=args.output_path,
                         padding_factor=args.padding_factor,
                         paired_data=args.dir_paired_data,
                         save_flo_flow=args.save_flo_flow,
                         )

        return

    # train datset
    train_dataset = build_dataset(args)
    print('Number of training images:', len(train_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True)

    last_epoch = start_step if args.resume and not args.no_resume_optimizer else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 10,
                                                       pct_start=args.lr_warmup, cycle_momentum=False,
                                                       anneal_strategy='linear',
                                                       last_epoch=last_epoch,
                                                       )

    if args.local_rank == 0:
        summary_writer = SummaryWriter(args.checkpoint_dir)
        logger = Logger(lr_scheduler, summary_writer, args.summary_freq,
                        start_step=start_step)

    total_steps = start_step
    epoch = start_epoch
    print('Start training')
    while total_steps < args.num_steps:
        model.train()

        # freeze BN after pretraining on chairs
        if args.freeze_bn:
            model_without_ddp.freeze_bn()

        print('Start epoch %d' % (epoch + 1))
        for i, sample in enumerate(train_loader):
            img1, img2, flow_gt, valid = [x.to(device) for x in sample]

            flow_preds = model(img1, img2, iters=args.train_iters)[0]

            loss, metrics = criterion(flow_preds, flow_gt, valid,
                                      gamma=args.gamma,
                                      max_flow=args.max_flow)

            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            lr_scheduler.step()

            if args.local_rank == 0:
                logger.push(metrics)

                logger.add_image_summary(img1, img2, flow_preds, flow_gt)

            total_steps += 1

            if total_steps % args.save_ckpt_freq == 0 or total_steps == args.num_steps:
                if args.local_rank == 0:
                    print('Save checkpoint at step: %d' % total_steps)
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_without_ddp.state_dict()
                    }, checkpoint_path)

            if total_steps % args.save_latest_ckpt_freq == 0:
                # Save lastest checkpoint after each epoch
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

                if args.local_rank == 0:
                    print('Save latest checkpoint')
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }, checkpoint_path)

            if total_steps % args.val_freq == 0:
                if args.local_rank == 0:
                    print('Start validation')

                    val_results = {}
                    # Support validation on multiple datasets
                    if 'chairs' in args.val_dataset:
                        results_dict = validate_chairs(model_without_ddp,
                                                       iters=args.val_iters,
                                                       )
                        val_results.update(results_dict)
                    if 'sintel' in args.val_dataset:
                        results_dict = validate_sintel(model_without_ddp,
                                                       iters=args.val_iters,
                                                       padding_factor=args.padding_factor,
                                                       )
                        val_results.update(results_dict)

                    if 'kitti' in args.val_dataset:
                        results_dict = validate_kitti(model_without_ddp,
                                                      iters=args.val_iters,
                                                      padding_factor=args.padding_factor,
                                                      )
                        val_results.update(results_dict)

                    logger.write_dict(val_results)

                    # Save validation results
                    val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('step: %06d\t' % total_steps)
                        # order of metrics
                        metrics = ['chairs_epe', 'chairs_1px', 'clean_epe', 'clean_1px', 'final_epe', 'final_1px',
                                   'kitti_epe', 'kitti_f1']
                        for metric in metrics:
                            if metric in val_results.keys():
                                f.write('%s: %.3f\t' % (metric, val_results[metric]))
                        f.write('\n')

                    model.train()

                    # freeze BN after pretraining on chairs
                    if args.freeze_bn:
                        model_without_ddp.freeze_bn()

            if total_steps >= args.num_steps:
                print('Training done')

                return

        epoch += 1


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
