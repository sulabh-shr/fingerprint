import os
import sys
import yaml
import math
import torch
import inspect
import argparse
import warnings
from easydict import EasyDict
import matplotlib.pyplot as plt

from torch import autocast
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../fingerprint')
from fingerprint.modelling import build_model
from fingerprint.data import build_dataset, build_dataloader
from fingerprint.data.transforms import get_train_transforms, get_test_transforms
from fingerprint.trainer import build_optimizer, build_lr_scheduler, resume, load_cfg
from fingerprint.loss import build_loss
from fingerprint.utils import Timer, PrintTime, DummyClass, get_run_name
from fingerprint.evaluation import ContrastiveEvaluator


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
    parser.add_argument('--resume', type=str, help='checkpoint file for resuming', default=None)
    parser.add_argument('--ft', action='store_true', help='only load model when resuming', default=False)
    parser.add_argument('--out', type=str, help='output folder', required=True)
    parser.add_argument('--run-name', type=str, help='run name', default=None)
    parser.add_argument('--strict-run-name', action='store_true', help='debug', default=False)
    parser.add_argument('--compile', action='store_true', help='torch.compile flag', default=False)
    parser.add_argument('--skip-ddp', action='store_true', help='skip ddp', default=False)
    parser.add_argument('--debug', action='store_true', help='debug', default=False)
    args = parser.parse_args()
    return args


def evaluate(model, evaluator, dataloader, loss_fn, rank, device, verbose=False):
    if dist.is_initialized():
        dist.barrier()

    loss = torch.Tensor([0.0]).to(device)
    model.eval()

    printer = DummyClass()
    if verbose and rank == 0:
        printer = PrintTime(start=0, end=len(dataloader), print_every=10)

    with torch.no_grad():
        batch_idx = 0
        for d in dataloader:
            batch_idx += 1
            x1 = model(d['image1'].to(device))['head']
            x2 = model(d['image2'].to(device))['head']
            loss += loss_fn(x=x1, y=x2)
            keys = d['key']
            for idx in range(len(keys)):
                evaluator.process(x1=x1[idx], x2=x2[idx], key=keys[idx])
            printer.print(f'Evaluation Iter: [{batch_idx}/{len(dataloader)}]')
            del d, x1, x2
    n_loss = len(dataloader)

    warnings.warn(f'Rank {rank}: DATALOADER COMPLETED............................')

    model.train()

    if dist.is_initialized():
        dist.barrier()
        loss_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(loss_list, loss, dist.new_group(backend="gloo"))
        loss = 0
        for i in loss_list:
            loss += i.item()
        n_loss *= dist.get_world_size()
    else:
        loss = loss.item()

    loss = loss / n_loss
    scores = evaluator.evaluate()
    scores['loss'] = loss
    res = evaluator.summarize(scores)
    print(res)
    return scores


def main(args):
    debug = args.debug
    run_name = args.run_name
    strict_run_name = args.strict_run_name
    out_root = args.out
    resume_path = args.resume
    ft = args.ft
    compile_ = args.compile
    use_ddp = not args.skip_ddp
    cfg_org = load_cfg(args.cfg)

    # Distributed
    rank = 0
    device = torch.device('cuda')
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(rank)
        print(f"Training with DDP on rank: {rank}")

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
    if use_ddp:
        model = DDP(model, device_ids=[device])
    model.train()

    if compile_:
        model = torch.compile(model)

    # Dataset
    train_transforms = get_train_transforms(cfg.INPUT, debug=False)
    dataset = build_dataset(cfg.DATA.TRAIN)
    dataset.transforms1 = train_transforms['transforms1']
    dataset.transforms2 = train_transforms['transforms2']
    print(f'Using dataset:\n {dataset}')

    # for d in dataset:
    #     fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    #     ax[0].imshow(d['image1'])
    #     ax[1].imshow(d['image2'])
    #     ax[0].set_title(d['mov1'])
    #     ax[1].set_title(d['mov2'])
    #     plt.show()
    #     plt.close()
    # raise Exception

    dataloader = build_dataloader(cfg.DATA.TRAIN, dataset=dataset)

    val_transforms = get_test_transforms(cfg.INPUT)
    val_dataset = build_dataset(cfg.DATA.VAL)
    val_dataset.transforms1 = val_transforms['test']
    val_dataset.transforms2 = val_transforms['test']
    val_dataloader = build_dataloader(cfg.DATA.VAL, dataset=val_dataset)

    # Optimizers and Loss
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_lr_scheduler(cfg, optimizer)

    # Resume
    ckpt = resume(path=resume_path, rank=rank, ft=ft, model=model, optimizer=optimizer, scheduler=scheduler)

    if use_ddp:
        dist.barrier()

    # Output paths
    if strict_run_name:
        out_root = os.path.join(out_root, run_name)
    else:
        out_root = os.path.join(out_root, get_run_name(run_name))
    os.makedirs(out_root, exist_ok=True)

    # Checkpoints and summaries
    writer = SummaryWriter(out_root) if rank == 0 else DummyClass()
    ckpt_path = os.path.join(out_root, 'checkpoint.pth')
    best_ckpt_path = os.path.join(out_root, 'best-checkpoint.pth')
    print(f'Checkpoints will be saved at : {ckpt_path}')
    if rank == 0 and (resume_path is None or ft):
        cfg_path = os.path.join(out_root, 'config.yaml')
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg_org, f)
        print(f'Config saved at : {cfg_path}')

    # Run length
    if not ft and ckpt is not None and 'global_step' in ckpt:
        global_step = ckpt['global_step']
        epoch_start = max(0, ckpt['epoch'] - 1)
        epochs = ckpt['epochs']
        max_iters = ckpt['max_iters']
        print(f'Resuming from GLOBAL step: {global_step}')
    else:
        global_step = 0
        epoch_start = 0
        epochs = cfg.PARAMS.get('EPOCHS', None)
        max_iters = cfg.PARAMS.get('ITERS', None)
    if max_iters is None:
        max_iters = len(dataloader) * epochs
    else:
        epochs = math.ceil(max_iters / len(dataloader))
    print_every = cfg.PARAMS.PRINT_EVERY
    save_every = cfg.PARAMS.SAVE_EVERY
    eval_every = cfg.PARAMS.get('EVAL_EVERY', None)

    # Train Evaluators
    evaluator = ContrastiveEvaluator()
    # Mixed Precision
    cast_dtype = get_cast_type(cfg.PARAMS.get('PRECISION', None))
    scaler = GradScaler(enabled=cast_dtype is not None)

    # Timers
    printer = PrintTime(start=global_step, end=max_iters, print_every=print_every) if rank == 0 else DummyClass()
    timer = Timer('Data ', debug=debug) if rank == 0 else DummyClass()

    val_loss = float('inf')
    best_score = 0
    best_step = 0
    for epoch in range(epoch_start, epochs):
        batch_idx = -1
        epoch_loss = 0
        for d in dataloader:
            batch_idx += 1
            global_step += 1

            timer('Model')
            optimizer.zero_grad()
            with autocast(enabled=cast_dtype is not None, dtype=cast_dtype, device_type='cuda'):
                x1 = model(d['image1'].to(device))['head']
                x2 = model(d['image2'].to(device))['head']
                loss = loss_fn(x=x1, y=x2)

            timer('Optim')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            printer.print(f'Epoch: {epoch} | Iter: [{batch_idx + 1}/{len(dataloader)}] | '
                          f'Loss: {epoch_loss / (batch_idx + 1):.3f} | lr: {current_lr:.8f}')
            writer.add_scalar('train-loss-iter', loss.item(), global_step)
            writer.add_scalar('lr-iter', current_lr, global_step)

            del d, x1, x2, loss
            timer('Data ')

            # Save checkpoint
            if rank == 0 and (global_step % save_every == 0 or global_step == max_iters - 1):
                ckpt = {
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'epochs': epochs,
                    'max_iters': max_iters
                }
                torch.save(ckpt, ckpt_path)
                print(f'Checkpoint for iter {global_step} saved at: {ckpt_path}')

            # Evaluate model
            if eval_every is not None and (global_step % eval_every == 0 or global_step == max_iters - 1):
                eval_results = evaluate(model, evaluator, val_dataloader, loss_fn, rank, device)
                val_score = eval_results['AUC']
                val_loss = eval_results['loss']
                if val_score > best_score:
                    best_score = val_score
                    best_step = global_step
                    if rank == 0:
                        ckpt = {
                            'model': model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                            'global_step': global_step,
                            'epochs': epochs,
                            'max_iters': max_iters
                        }
                        torch.save(ckpt, best_ckpt_path)
                        print(f'NEW Best checkpoint saved at: {best_ckpt_path}')

                # Train eval
                writer.add_scalar('val-score', val_score, global_step)
                writer.add_scalar('val-loss', val_loss, global_step)

                if rank == 0:
                    print(f'EVALUATION RESULTS for step [{global_step}] | '
                          f'val score: {val_score:.2f} | val loss: {val_loss:.4f} | '
                          f'best mIoU: {best_score:.2f} @ step {best_step}')

            if 'metrics' in str(inspect.signature(scheduler.step)):
                if global_step % eval_every == 0:
                    scheduler.step(metrics=val_loss)
            else:
                scheduler.step()

            if global_step >= max_iters:
                break

        writer.add_scalar('train-loss', epoch_loss / len(dataloader), epoch)
        if global_step >= max_iters:
            break

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    _args = get_args()
    main(_args)
