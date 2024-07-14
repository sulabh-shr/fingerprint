import os
import csv
import yaml
import torch
import random
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
from collections import defaultdict

from torch import autocast
import torch.distributed as dist

from fingerprint.modelling import build_model
from fingerprint.data import build_dataset, build_dataloader
from fingerprint.data.transforms import get_test_transforms
from fingerprint.trainer import resume, load_cfg
from fingerprint.utils import Timer, PrintTime, get_run_name
from fingerprint.evaluation import build_evaluator

import matplotlib

matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
matplotlib.rc('text', usetex='false')
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['lines.linewidth'] = 2


def get_cast_type(x):
    if x is None:
        return None
    elif x == 'float16':
        return torch.float16
    elif x == 'float32':
        return torch.float32
    raise ValueError(f'Invalid precision: {x}')


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, nargs='+', help='folder containing images', required=True)
    parser.add_argument('--ckpts', type=str, nargs='*', help='checkpoint file(s) for resuming', default=[None])
    parser.add_argument('--out', type=str, help='output folder for final scores', required=True)
    parser.add_argument('--compile', action='store_true', help='torch.compile flag', default=False)
    parser.add_argument('--ddp', action='store_true', help='use ddp', default=False)
    parser.add_argument('--print-every', type=int, help='print iterations', default=10)
    parser.add_argument('--debug', action='store_true', help='debug', default=False)
    parser.add_argument('--opts', type=str, nargs='*', help='cfg updates', default=None)
    parser.add_argument('--fig-title', type=str, help='figure title', default=None)
    parser.add_argument('--seeds', type=int, nargs='*', help='randomness seeds', default=[1, 3, 5])
    parser.add_argument('--frames', type=int, nargs='*', help='probe frames per video', default=[2, 4, 8, 300])

    args = parser.parse_args()
    return args


def collate_as_dict_of_list(data):
    new_data = {}

    for d in data:
        for k, v in d.items():
            if k in new_data:
                new_data[k].append(v)
            else:
                new_data[k] = [v]
    return new_data


def collate_as_list_of_dict(data):
    new_data = []
    for d in data:
        new_data.append(d)
    return new_data


def main(args):
    debug = args.debug
    out_root = args.out
    ckpt_paths = args.ckpts
    compile_ = args.compile
    use_ddp = args.ddp
    print_every = args.print_every
    opts = args.opts
    fig_title = args.fig_title
    seeds = args.seeds
    frames_per_video = args.frames
    PROBE_MOVES = ["Pitch", "Roll", "Yaw", ["Pitch", "Roll", "Yaw"]]

    # Load and merge configs
    cfg_org = load_cfg(args.cfg, opts)

    # Distributed
    rank = 0
    device = torch.device('cuda')
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        print(f"Evaluating with DDP on rank: {rank}")

    if rank == 0:
        for k, v in args.__dict__.items():
            print(f'{k:-<20s} : {v}')
        print('*' * 70)
        print(yaml.dump(cfg_org))
        print('*' * 70)

    # Load and merge configs
    cfg = EasyDict(cfg_org)

    # Model
    model = build_model(cfg, device)
    model = model.to(device)
    model.eval()

    if compile_:
        model = torch.compile(model)

    # Save prefix
    final_res = ''
    sep = '#' * 70
    final_scores = []

    # Dataset Transform
    test_transforms = get_test_transforms(cfg.INPUT)

    for ckpt_path in ckpt_paths:
        
        # Load weights and checkpoint specific configs
        if ckpt_path is not None:
            resume(path=ckpt_path, rank=rank, ft=True, model=model)

            ckpt_folder_path, ckpt_name = os.path.split(ckpt_path)
            ckpt_name, _ = os.path.splitext(ckpt_name)
            ckpt_out_root = os.path.join(ckpt_folder_path, f'multi-eval-{ckpt_name}')

            # Load checkpoint specific config for respective train/val/test
            ckpt_path_i = os.path.join(ckpt_folder_path, 'config.yaml')
            ckpt_cfg = load_cfg([ckpt_path_i], opts)
            ckpt_cfg = EasyDict(ckpt_cfg)
            print(f'{sep}\nUSING CONFIG: {ckpt_path_i}\n{sep}')
        else:
            ckpt_cfg = cfg

        for frames in frames_per_video:
            ckpt_cfg.DATA.TEST.DATASET.KWARGS.frames_per_video = frames
            
            for seed in seeds:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)

                for probe_mov in PROBE_MOVES:
                    ckpt_cfg.DATA.TEST.DATASET.KWARGS.probe_movements = probe_mov

                    dataset = build_dataset(ckpt_cfg.DATA.TEST)
                    dataset.transforms1 = test_transforms['test']
                    dataset.transforms2 = test_transforms['test']
                    dataloader = build_dataloader(ckpt_cfg.DATA.TEST, dataset=dataset)

                    evaluator = build_evaluator(ckpt_cfg.EVALUATOR.TEST)
                    evaluator.set_model(model)

                    # Timers
                    print(sep)
                    print(
                        f'STARTING EVALUATION FOR: {ckpt_path} | frames: {frames} | seed: {seed} | Probe: {probe_mov}')
                    print(sep)
                    printer = PrintTime(end=len(dataloader), print_every=print_every)
                    timer = Timer('Data ', debug=debug)

                    batch_idx = 0
                    for data in dataloader:
                        batch_idx += 1
                        printer.print(f'Index: {batch_idx}')
                        timer('Model')
                        with torch.no_grad():
                            features = model(data['image'].to(device))['head']
                        evaluator.process(features=features, classes=data['class'], locations=data['location'])

                    scores, fig = evaluator.evaluate()

                    if fig_title:
                        plt.title(fig_title)

                    res = evaluator.summarize(scores)
                    print(res)
                    print('-' * 70)
                    final_res += f'{sep}\n SEED: {seed} | PROBE MOVEMENT: {probe_mov}\n{sep}\n{res}'
                    score_dict = {
                        'ckpt_path': ckpt_path,
                        'seed': seed,
                        'frames_per_video': frames,
                        'probe_movements': probe_mov
                    }
                    score_dict.update(scores)
                    print(score_dict)
                    final_scores.append(score_dict)

                    # Save scores and plots
                    os.makedirs(ckpt_out_root, exist_ok=True)
                    if isinstance(probe_mov, str):
                        run_name = f'frames {frames} seed {seed} {probe_mov}'
                    else:
                        run_name = f'frames {frames} seed {seed}'
                        for i in probe_mov:
                            run_name += ' ' + i
                    eval_text_path = os.path.join(ckpt_out_root, f'eval {run_name}.txt')
                    with open(eval_text_path, 'w') as f:
                        f.write(res)
                    eval_fig_path = os.path.join(ckpt_out_root, f'eval {run_name}.png')
                    fig.savefig(eval_fig_path, bbox_inches='tight')

                    print(f'Eval Outputs saved at: {eval_text_path}')

    final_run_name = f'{ckpt_name} Eval SEEDS'
    for i in seeds:
        final_run_name += f' {i}'
    final_text_path = os.path.join(out_root, f'{final_run_name}.txt')
    final_csv_path = os.path.join(out_root, f'{final_run_name}.csv')
    final_pickle_path = os.path.join(out_root, f'{final_run_name}.pickle')
    with open(final_text_path, 'w') as f:
        f.write(final_res)

    # Mean and Std of all runs
    mean_score_dicts = []
    fieldnames = ['frames', 'probe', 'eval', 'mean', 'std']
    for frames in frames_per_video:
        for probe_mov in PROBE_MOVES:
            for score_type in [
                'EER', 'HTER_FMR<=0.01', 'HTER_FMR<=0.001', 'AUC', 'f1',
                'FNMR<=0.01', 'FNMR<=0.001', 'FMR<=0.01', 'FMR<=0.001'
            ]:
                score_list = [
                    i[score_type] for i in final_scores if
                    i['frames_per_video'] == frames and
                    i['probe_movements'] == probe_mov
                ]
                frames_probe_eval_dict = {
                    'frames': frames,
                    'probe': probe_mov,
                    'eval': score_type,
                    'mean': round(np.mean(score_list), 2),
                    'std': round(np.std(score_list), 2)
                }
                mean_score_dicts.append(frames_probe_eval_dict)

    with open(final_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mean_score_dicts)
    print(f'All evaluations CSV saved at: {final_csv_path}')

    with open(final_pickle_path, 'wb') as f:
        pickle.dump(final_scores, f)
    print(f'All evaluations PICKLED at: {final_pickle_path}')


if __name__ == '__main__':
    _args = get_args()
    main(_args)
