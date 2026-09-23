"""
A2Net Training Script (clean baseline)

Supports:
- LWGANet-L0 backbone (2.91M inference params)
- Optional AFD feature distillation (training-only, removed at deploy)
- Datasets: LEVIR-CD-256, SYSU-CD-256, WHU-CD-256, CDD-CD-256

Note: this BCD task uses the test set as the validation set (consistent with
ChangeMamba / CDMamba). Every epoch the model is evaluated on the test split and
the checkpoint with the best test F1 is kept. No separate infer/ folder is
created.
"""

import os
import sys
import time
import datetime
import argparse
import random
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from models import A2Net_LWGANet_L0, A2Net_LWGANet_L2, build_loss
from models.datasets.cd_dataset import get_loader, get_test_loader
from models.utils.metrics import ConfuseMatrixMeter
from models.utils.logger import TrainingLogger
from models.utils.scheduler import adjust_learning_rate


def parse_args():
    parser = argparse.ArgumentParser(description='A2Net Training')

    # Dataset
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['LEVIR', 'SYSU', 'WHU', 'CDD'])
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--inWidth', type=int, default=256)
    parser.add_argument('--inHeight', type=int, default=256)

    # Model
    parser.add_argument('--model_type', type=str, default='L0', choices=['L0', 'L2'])
    parser.add_argument('--use_afd', action='store_true', help='Use AFD feature distillation')
    parser.add_argument('--afd_lambda', type=float, default=0.5, help='AFD loss weight')
    parser.add_argument('--pretrained', default=True, help='Use pretrained backbone')

    # GRAFT-PLUG (train-only plug-in)
    parser.add_argument('--graft_ablation', type=str, default=None,
                        choices=['T', 'D', 'U', 'F', 'L'],
                        help='GRAFT ablation: T=task only, D=distill only, U=uniform, F=full, L=local tutor')
    parser.add_argument('--graft_task_weight', type=float, default=1.0, help='GRAFT task gradient weight')
    parser.add_argument('--graft_repr_weight', type=float, default=1.0, help='GRAFT representation gradient weight')
    parser.add_argument('--graft_repr_warmup', type=float, default=0.1,
                        help='fraction of total steps over which the repr KD weight linearly ramps 0->1')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=40000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy')
    parser.add_argument('--num_workers', type=int, default=4)

    # Logging
    parser.add_argument('--experiment', default='', help='Experiment label (e.g. R0/R1)')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--log_file', default='train_log.txt')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2333)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def make_worker_init_fn(base_seed):
    """Deterministic per-worker seeding so augmentation RNG is independent of
    model-construction RNG (GRAFT init must not shift the data stream)."""
    def _init(worker_id):
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return _init


def train_epoch(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0):
    """Train for one epoch."""
    model.train()

    salEvalVal = ConfuseMatrixMeter(n_class=2)
    loss_total = 0.0
    loss_main = 0.0
    loss_graft_task = 0.0
    loss_graft_repr = 0.0
    loss_graft_kgr = 0.0
    loss_graft_conf = 0.0
    loss_graft_gap = 0.0
    has_task = has_repr = has_kgr = False
    n_batches = len(train_loader)
    total_steps = max_batches * args.max_epochs

    for iter, batched_inputs in enumerate(train_loader):
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        global_iter = iter + cur_iter

        # Adjust learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, global_iter, max_batches, lr_factor=1.0)

        # Move to GPU
        pre_img = pre_img.cuda()
        post_img = post_img.cuda()
        target = target.cuda().float()

        # Forward pass (training -> returns (masks, aux_losses))
        masks, aux_losses = model(pre_img, post_img, target)
        output, output2, output3, output4 = masks

        # Main multi-scale loss
        l_main = (criterion(output, target) + criterion(output2, target) +
                  criterion(output3, target) + criterion(output4, target))
        loss = l_main

        # AFD auxiliary loss
        if "afd" in aux_losses:
            loss = loss + args.afd_lambda * aux_losses["afd"]

        # GRAFT auxiliary losses
        if "graft_task" in aux_losses:
            loss = loss + args.graft_task_weight * aux_losses["graft_task"]
            has_task = True
        if "graft_repr" in aux_losses:
            # representation KD 前 warmup 段线性 ramp（0 -> 1），避免早期随机 teacher 拉偏 backbone
            warmup = 1.0
            if args.graft_repr_warmup > 0:
                warmup = min(1.0, global_iter / max(1, total_steps * args.graft_repr_warmup))
            loss = loss + args.graft_repr_weight * warmup * aux_losses["graft_repr"]
            has_repr = True
        if "graft_kgr" in aux_losses:
            has_kgr = True

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()
        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())

        loss_total += loss.item()
        loss_main += l_main.item()
        if has_task:
            loss_graft_task += aux_losses["graft_task"].detach().item()
        if has_repr:
            loss_graft_repr += aux_losses["graft_repr"].detach().item()
        if has_kgr:
            loss_graft_kgr += aux_losses["graft_kgr"].detach().item()
            loss_graft_conf += aux_losses["graft_conf"].detach().item()
            loss_graft_gap += aux_losses["graft_gap"].detach().item()

        if iter % 5 == 0:
            print(f'\riteration: [{global_iter}/{max_batches * args.max_epochs}] '
                  f'f1: {f1:.3f} lr: {lr:.7f} loss: {loss.item():.3f} '
                  f'main: {l_main.item():.3f}', end='')

    scores = salEvalVal.get_scores()
    ret = {
        'total': loss_total / n_batches,
        'main': loss_main / n_batches,
    }
    if has_task:
        ret['graft_task'] = loss_graft_task / n_batches
    if has_repr:
        ret['graft_repr'] = loss_graft_repr / n_batches
    if has_kgr:
        ret['graft_kgr'] = loss_graft_kgr / n_batches
        ret['graft_conf'] = loss_graft_conf / n_batches
        ret['graft_gap'] = loss_graft_gap / n_batches
    return ret, scores, lr


@torch.no_grad()
def eval_epoch(args, test_loader, model, criterion):
    """Evaluate on the test split (used as the validation set)."""
    model.eval()

    salEvalTest = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(test_loader)

    for iter, batched_inputs in enumerate(test_loader):
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        pre_img = pre_img.cuda()
        post_img = post_img.cuda()
        target = target.cuda().float()

        # Forward pass (eval -> returns masks tuple only)
        masks = model(pre_img, post_img)
        output, output2, output3, output4 = masks

        loss = (criterion(output, target) + criterion(output2, target) +
                criterion(output3, target) + criterion(output4, target))

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()
        epoch_loss.append(loss.item())
        f1 = salEvalTest.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())

        if iter % 5 == 0:
            print(f'\r[{iter}/{total_batches}] F1: {f1:.3f} loss: {loss.item():.3f}', end='')

    average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalTest.get_scores()
    return average_epoch_loss, scores


def count_deploy_params(model):
    """Count parameters remaining after switch_to_deploy() (inference/deploy params)."""
    model.switch_to_deploy()
    return sum(p.numel() for p in model.parameters())


def count_flops(model, size=256):
    """Count deploy FLOPs at `size`x`size` via thop; return None if unavailable."""
    try:
        from thop import profile
    except ImportError:
        print("[warn] thop not installed; skipping FLOPs profiling")
        return None
    try:
        model.eval()
        x1 = torch.randn(1, 3, size, size).cuda()
        x2 = torch.randn(1, 3, size, size).cuda()
        flops, _ = profile(model, inputs=(x1, x2), verbose=False)
        return flops
    except Exception as e:
        print(f"[warn] FLOPs profiling failed: {e}")
        return None


@torch.no_grad()
def verify_deploy_equivalence(model, size=256):
    """Run the same fixed input before/after switch_to_deploy(); return max abs error.

    In eval mode GRAFT is never invoked, so the only change switch_to_deploy()
    makes is deleting the side branch. A correct plug-in must give bit-identical
    main-path outputs (max error == 0; tolerance < 1e-6 guards against fp noise).
    """
    model.eval()
    x1 = torch.randn(1, 3, size, size).cuda()
    x2 = torch.randn(1, 3, size, size).cuda()
    before = model(x1, x2)                 # eval -> masks tuple, GRAFT skipped
    model.switch_to_deploy()
    after = model(x1, x2)                  # side branch deleted
    max_err = 0.0
    for b, a in zip(before, after):
        max_err = max(max_err, float((b - a).abs().max()))
    return max_err


def _build_graft_cfg(args):
    """Build GRAFTPlug config dict from --graft_ablation."""
    if args.graft_ablation is None:
        return None
    stage_ch = (32, 64, 128, 256) if args.model_type == 'L0' else (96, 192, 384, 768)
    cfg = dict(
        stage_ch=stage_ch,
        bsee_dim=128,
        token_dim=192,
        cgr_blocks=4,
        cgr_heads=6,
        mlp_ratio=4,
        pool_sizes=(8, 4, 4, 2),
    )
    mode = args.graft_ablation
    if mode == 'T':
        cfg.update(task_grad=True, repr_grad=False, detach_teacher=False, kgr=False, local_tutor=False)
    elif mode == 'D':
        cfg.update(task_grad=True, repr_grad=True, detach_teacher=True, kgr=False, local_tutor=False)
    elif mode == 'U':
        cfg.update(task_grad=True, repr_grad=True, detach_teacher=False, kgr=False, local_tutor=False)
    elif mode == 'F':
        cfg.update(task_grad=True, repr_grad=True, detach_teacher=False, kgr=True, local_tutor=False)
    elif mode == 'L':
        cfg.update(task_grad=True, repr_grad=True, detach_teacher=False, kgr=True, local_tutor=True)
    return cfg


def main():
    args = parse_args()

    # Set GPU by directly selecting the physical device.
    torch.cuda.set_device(args.gpu_id)
    set_seed(args.seed)

    # Map dataset name to path
    if not args.data_root:
        if args.dataset_name == 'LEVIR':
            args.data_root = '/data/CD/LEVIR-CD-256'
        elif args.dataset_name == 'SYSU':
            args.data_root = '/data/CD/SYSU-CD-256'
        elif args.dataset_name == 'WHU':
            args.data_root = '/data/CD/WHU-CD-256'
        elif args.dataset_name == 'CDD':
            args.data_root = '/data/CD/CDD-CD-256'

    os.makedirs(args.save_dir, exist_ok=True)

    # Build model
    print("Building model...")
    graft_cfg = _build_graft_cfg(args)
    if args.model_type == 'L0':
        model = A2Net_LWGANet_L0(pretrained=args.pretrained, use_afd=args.use_afd, graft_cfg=graft_cfg)
    else:
        model = A2Net_LWGANet_L2(pretrained=args.pretrained, use_afd=args.use_afd, graft_cfg=graft_cfg)
    model = model.cuda()

    train_params = sum(p.numel() for p in model.parameters())
    graft_params = (sum(p.numel() for p in model.graft.parameters())
                    if getattr(model, 'use_graft', False) else 0)
    print(f"Model: A2Net_LWGANet_{args.model_type}")
    print(f"Train Params: {train_params / 1e6:.2f}M")
    if graft_params:
        print(f"GRAFT (train-only) Params: {graft_params / 1e6:.2f}M")

    # Build dataloaders (test split is reused as the validation set).
    # Independent RNG (generator + worker_init_fn) keeps the sample order and
    # augmentation stream identical across R0 / R1-* regardless of GRAFT init.
    print("Loading data...")
    train_generator = torch.Generator().manual_seed(args.seed)
    test_generator = torch.Generator().manual_seed(args.seed)
    worker_init_fn = make_worker_init_fn(args.seed)
    train_loader = get_loader(args.data_root, 'train.txt',
                              batchsize=args.batch_size, trainsize=args.inWidth,
                              shuffle=True, num_workers=args.num_workers,
                              generator=train_generator, worker_init_fn=worker_init_fn)
    test_loader = get_test_loader(args.data_root, 'test.txt',
                                  batchsize=args.batch_size, testsize=args.inWidth,
                                  num_workers=args.num_workers,
                                  generator=test_generator, worker_init_fn=worker_init_fn)

    # Criterion & optimizer
    criterion = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    max_batches = len(train_loader)
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    print(f'Training for {args.max_epochs} epochs ({max_batches} batches per epoch)')

    # Logger
    log_config = {
        'experiment': args.experiment,
        'dataset': args.dataset_name,
        'model': f'A2Net_LWGANet_{args.model_type}',
        'use_afd': args.use_afd,
        'afd_lambda': args.afd_lambda if args.use_afd else 0,
        'graft_ablation': args.graft_ablation if args.graft_ablation else 'None',
        'train_params': f'{train_params / 1e6:.2f}M',
        'graft_params': f'{graft_params / 1e6:.2f}M' if graft_params else '0',
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_steps': args.max_steps,
        'n_train': len(train_loader.dataset),
        'n_test': len(test_loader.dataset),
    }
    logger = TrainingLogger(os.path.join(args.save_dir, args.log_file), log_config)

    # Training loop
    start_epoch = 0
    cur_iter = 0
    max_F1_test = 0

    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")

    for epoch in range(start_epoch, args.max_epochs):
        lossTr, score_tr, lr = train_epoch(args, train_loader, model, criterion,
                                            optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(train_loader)
        torch.cuda.empty_cache()

        if epoch == 0:
            continue

        lossTest, score_test = eval_epoch(args, test_loader, model, criterion)
        torch.cuda.empty_cache()

        is_best = score_test['F1'] > max_F1_test
        logger.log_epoch(epoch, args.max_epochs, lossTr,
                        {'f1': score_test['F1'], 'iou': score_test['IoU'], 'kappa': score_test['Kappa'],
                         'recall': score_test['recall'], 'precision': score_test['precision'], 'oa': score_test['OA']},
                        lr,
                        torch.cuda.max_memory_allocated(args.gpu_id) / 1e9,
                        is_best)

        if is_best:
            max_F1_test = score_test['F1']
            for old in glob.glob(os.path.join(args.save_dir, 'bestF1=*_model.pth')):
                try:
                    os.remove(old)
                except OSError:
                    pass
            best_model_file_name = os.path.join(args.save_dir, f'bestF1={max_F1_test:.4f}_model.pth')
            torch.save(model.state_dict(), best_model_file_name)
            print(f"\nBEST model saved: {best_model_file_name}")

        print(f"\nEpoch {epoch}: Train Loss = {lossTr['total']:.4f} (main {lossTr['main']:.4f}), "
              f"Test Loss = {lossTest:.4f}, F1(test) = {score_test['F1']:.4f}")
        torch.cuda.empty_cache()

    end_time = datetime.datetime.now()
    all_time = end_time - start_time
    print(f"Training completed in: {all_time}")

    # Load best model and report final test metrics
    best_model_files = glob.glob(os.path.join(args.save_dir, 'bestF1=*_model.pth'))
    if not best_model_files:
        print("No best model found, skipping final test report")
        return
    best_model_file_name = best_model_files[0]
    model.load_state_dict(torch.load(best_model_file_name))

    loss_test, score_test = eval_epoch(args, test_loader, model, criterion)
    logger.log_test_results(
        {'recall': score_test['recall'], 'precision': score_test['precision'],
         'f1': score_test['F1'], 'iou': score_test['IoU'],
         'oa': score_test['OA'], 'kappa': score_test['Kappa']},
        best_model_file_name
    )
    logger.log_message(f"Total training time: {all_time}")
    logger.log_message(f"Nominal max_steps: {args.max_steps}; "
                       f"actual optimizer steps: {cur_iter}")

    # Deploy-equivalence verification: same fixed input before/after removal.
    max_err = verify_deploy_equivalence(model, size=args.inWidth)
    deploy_params = count_deploy_params(model)
    deploy_params_msg = (f"Inference Params (deploy): {deploy_params} "
                         f"({deploy_params / 1e6:.4f}M)")
    deploy_err_msg = f"Deploy max error (before/after): {max_err:.3e}"
    logger.log_message(deploy_params_msg)
    logger.log_message(deploy_err_msg)
    print(deploy_params_msg)
    print(deploy_err_msg)
    assert max_err < 1e-6, f"Deploy is NOT lossless: max error {max_err:.3e} >= 1e-6"
    if args.model_type == 'L0':
        assert deploy_params == 2_913_094, f"Unexpected deploy params: {deploy_params}"
    logger.log_message("Deploy equivalence check PASSED (max error < 1e-6)")

    deploy_flops = count_flops(model, size=args.inWidth)
    if deploy_flops is not None:
        deploy_flops_msg = (f"Inference FLOPs (deploy): {deploy_flops / 1e9:.4f}G "
                            f"@ {args.inWidth}x{args.inHeight}")
        logger.log_message(deploy_flops_msg)
        print(deploy_flops_msg)

    # Export a clean deploy-only checkpoint and verify it loads strictly into a
    # fresh no-graft model (guarantees the side branch left no residue behind).
    deploy_ckpt = os.path.join(args.save_dir, 'best_deploy_model.pth')
    torch.save(model.state_dict(), deploy_ckpt)
    if args.model_type == 'L0':
        clean = A2Net_LWGANet_L0(pretrained=False, use_afd=False, graft_cfg=None)
    else:
        clean = A2Net_LWGANet_L2(pretrained=False, use_afd=False, graft_cfg=None)
    clean.load_state_dict(torch.load(deploy_ckpt), strict=True)
    logger.log_message(f"Deploy checkpoint saved & strict-loaded: {deploy_ckpt}")
    print(f"Deploy checkpoint saved & strict-loaded: {deploy_ckpt}")

    print(f"\nTest (best_epoch): Kappa = {score_test['Kappa']:.4f}, IoU = {score_test['IoU']:.4f}, "
          f"F1 = {score_test['F1']:.4f}, R = {score_test['recall']:.4f}, P = {score_test['precision']:.4f}")

    print("\n" + "=" * 70)
    print("All tasks completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("\n" + "=" * 60)
        print("Training failed!")
        print("=" * 60)
        print(tb)
        sys.exit(1)
