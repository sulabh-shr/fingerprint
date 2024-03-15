import os
import yaml
import torch
import argparse
import matplotlib.pyplot as plt
from easydict import EasyDict

from torch import autocast
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from fingerprint.modelling import build_model
from fingerprint.data import build_dataset, build_dataloader
from fingerprint.data.transforms import get_test_transforms
from fingerprint.trainer import resume, load_cfg
from fingerprint.utils import Timer, PrintTime, DummyClass, get_run_name
from fingerprint.evaluation import FingerprintEvaluator


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
    parser.add_argument('--ckpt', type=str, help='checkpoint file for resuming', default=None)
    parser.add_argument('--out', type=str, help='output folder', required=True)
    parser.add_argument('--compile', action='store_true', help='torch.compile flag', default=False)
    parser.add_argument('--ddp', action='store_true', help='use ddp', default=False)
    parser.add_argument('--print-every', type=int, help='print iterations', default=10)
    parser.add_argument('--debug', action='store_true', help='debug', default=False)
    parser.add_argument('--save', action='store_true', help='save images and results', default=False)
    parser.add_argument('--viz', action='store_true', help='visualize images and results', default=False)

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
    ckpt_path = args.ckpt
    compile_ = args.compile
    use_ddp = args.ddp
    cfg_org = load_cfg(args.cfg)
    print_every = args.print_every
    save = args.save
    viz = args.viz

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

    # Dataset
    test_transforms = get_test_transforms(cfg.INPUT)
    dataset = build_dataset(cfg.DATA.TEST)
    dataset.transforms1 = test_transforms['test']
    dataset.transforms2 = test_transforms['test']
    dataloader = build_dataloader(cfg.DATA.TEST, dataset=dataset, collate_fn=collate_as_list_of_dict)

    # Resume
    resume(path=ckpt_path, rank=rank, ft=True, model=model)

    evaluator = FingerprintEvaluator()

    # Output paths
    out_img_root = None
    if save:
        out_img_root = os.path.join(out_root, 'inference')
        os.makedirs(out_img_root, exist_ok=True)

    # Timers
    print(f'Starting Evaluation')
    printer = PrintTime(end=len(dataloader), print_every=print_every)
    timer = Timer('Data ', debug=debug)

    batch_idx = 0
    for data in dataloader:
        batch_idx += 1
        printer.print(f'Index: {batch_idx}')

        timer('Model')
        # keys = data['key']
        # for k, v in data.items():
        #     print(k)
        #     if isinstance(v[0], torch.Tensor):
        #         print([i.shape for i in v])
        #     else:
        #         print(len(v))

        for d in data:
            with torch.no_grad():
                gallery = model(d['gallery'].to(device))['head']
                probe = model(d['probe'].to(device))['head']
                key = d['key']
            evaluator.process(gallery=gallery, probe=probe, key=key)

    scores = evaluator.evaluate()
    res = evaluator.summarize(scores)
    print(res)
    print('-' * 70)
    # out = {
    #     'per_image_conf': {n: c for n, c in zip(img_names, evaluator.each_conf_matrix)},
    #     'conf': evaluator.conf_matrix
    # }
    # os.makedirs(out_root, exist_ok=True)
    # final_out_path = os.path.join(out_root, 'eval.pkl')
    # final_eval_path = os.path.join(out_root, 'eval.txt')
    # with open(final_out_path, 'wb') as f:
    #     pickle.dump(out, f)
    # with open(final_eval_path, 'w') as f:
    #     f.write(res)


if __name__ == '__main__':
    _args = get_args()
    main(_args)
