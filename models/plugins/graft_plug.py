"""
GRAFT-PLUG: Gradient-Routed Auxiliary Fusion Teacher Plug-in.

训练期外挂模块，挂在 LWGANet raw stage 之外：只读 backbone 特征、输出从不进入
SWA / TFM / Decoder，`switch_to_deploy()` 后整套删除，部署参数回到 2.9131M。

它制造两条互补梯度：
  A. Global Task Gradient          GT -> global teacher -> backbone
  B. Global-to-Local Distill Gradient  detached teacher -> KGR -> probe -> backbone

四个组件（对应设计文档 §8–§14）：
  BSEE  Bi-temporal Symmetric Evidence Encoder   (sum / absdiff / product)
  CGR   Cross-stage Global Reasoner              (multi-scale token transformer)
  Teacher Decoder + heads                        (global context + local R_s -> T_s -> logits)
  LSP   Local Stage Probes                       (capacity-limited student representation)
  KGR   Knowledge-Gap Gradient Router            (reliability x gap, class-balanced)

BN / Norm：内部只用 GroupNorm（不共享、不使用主路 SyncBN），保证 deploy 等价性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels):
    """GroupNorm helper：不使用 SyncBN，避免污染主路 running stats。"""
    groups = 32
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SymmetricEvidenceEncoder(nn.Module):
    """BSEE：对称 relation basis -> 1x1 projection（交换不变）。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3 * in_ch, out_ch, 1, bias=False),
            _gn(out_ch),
            nn.GELU(),
        )

    def forward(self, f1, f2):
        rel = torch.cat([f1 + f2, (f1 - f2).abs(), f1 * f2], dim=1)
        return self.proj(rel)


class GRAFTPlug(nn.Module):
    def __init__(self,
                 stage_ch=(32, 64, 128, 256),
                 bsee_dim=128,
                 token_dim=192,
                 cgr_blocks=4,
                 cgr_heads=6,
                 mlp_ratio=4,
                 pool_sizes=(8, 4, 4, 2),
                 task_grad=True,
                 repr_grad=True,
                 detach_teacher=False,
                 kgr=True,
                 local_tutor=False):
        super().__init__()
        self.stage_ch = list(stage_ch)
        self.bsee_dim = bsee_dim
        self.token_dim = token_dim
        self.pool_sizes = list(pool_sizes)
        self.num_stages = len(stage_ch)
        self.task_grad = task_grad
        self.repr_grad = repr_grad
        self.detach_teacher = detach_teacher
        self.kgr = kgr
        self.local_tutor = local_tutor

        self.token_counts = [p * p for p in self.pool_sizes]
        self.total_tokens = sum(self.token_counts)

        # BSEE per stage
        self.bsee = nn.ModuleList([
            SymmetricEvidenceEncoder(c, bsee_dim) for c in stage_ch
        ])

        if local_tutor:
            # 每个 stage 一个 local teacher（无跨 stage 全局推理，容量对齐的反例组）
            # 加宽中间层到 204，使 train-only 参数 ≈ F 模式（~3.82M），隔离"全局信息"vs"容量"
            local_dim = 204
            self.local_teacher = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(bsee_dim, local_dim, 3, 1, 1, bias=False), _gn(local_dim), nn.GELU(),
                    nn.Conv2d(local_dim, local_dim, 3, 1, 1, bias=False), _gn(local_dim), nn.GELU(),
                    nn.Conv2d(local_dim, bsee_dim, 3, 1, 1, bias=False), _gn(bsee_dim), nn.GELU(),
                ) for _ in range(self.num_stages)
            ])
        else:
            # CGR：multi-scale token 化 -> transformer
            self.token_proj = nn.ModuleList([
                nn.Linear(bsee_dim, token_dim) for _ in range(self.num_stages)
            ])
            self.stage_embed = nn.Parameter(torch.zeros(1, self.num_stages, token_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.total_tokens, token_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=cgr_heads,
                dim_feedforward=token_dim * mlp_ratio,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                dropout=0.0,  # 显式关闭 dropout：teacher 前向确定性，且不再消耗全局 RNG
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=cgr_blocks)
            # teacher fusion: [global context (token_dim) + local R_s (bsee_dim)] -> T_s
            self.teacher_fuse = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(token_dim + bsee_dim, bsee_dim, 3, 1, 1, bias=False),
                    _gn(bsee_dim),
                    nn.GELU(),
                ) for _ in range(self.num_stages)
            ])

        # teacher change heads（task gradient 的 deep supervision）
        self.teacher_head = nn.ModuleList([
            nn.Conv2d(bsee_dim, 1, 1) for _ in range(self.num_stages)
        ])

        # LSP：容量受限的 local probe（repr gradient）
        self.probe = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3 * c, bsee_dim, 1, bias=False), _gn(bsee_dim), nn.GELU(),
                nn.Conv2d(bsee_dim, bsee_dim, 3, 1, 1, groups=bsee_dim, bias=False),
                _gn(bsee_dim), nn.GELU(),
                nn.Conv2d(bsee_dim, bsee_dim, 1, bias=False),
            ) for c in stage_ch
        ])

        # 初始化
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.trunc_normal_(p, std=0.02)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, feats1, feats2, target):
        B = feats1[0].shape[0]

        # teacher 输入（R1-D 时 detach，使 task gradient 不进 backbone）
        if self.detach_teacher:
            tfeats1 = [f.detach() for f in feats1]
            tfeats2 = [f.detach() for f in feats2]
        else:
            tfeats1, tfeats2 = feats1, feats2

        # 1. BSEE
        Rs = [self.bsee[s](tfeats1[s], tfeats2[s]) for s in range(self.num_stages)]

        # 2. teacher features T_s
        if self.local_tutor:
            Ts = [self.local_teacher[s](Rs[s]) for s in range(self.num_stages)]
        else:
            tokens = []
            for s in range(self.num_stages):
                t = F.adaptive_avg_pool2d(Rs[s], self.pool_sizes[s])  # [B,C,p,p]
                t = t.flatten(2).transpose(1, 2)                       # [B,p*p,C]
                t = self.token_proj[s](t)                              # [B,p*p,token_dim]
                t = t + self.stage_embed[:, s:s + 1, :]
                tokens.append(t)
            tokens = torch.cat(tokens, dim=1)                          # [B,100,token_dim]
            tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
            g = self.transformer(tokens)                               # [B,100,token_dim]

            Ts = []
            start = 0
            for s in range(self.num_stages):
                n = self.token_counts[s]
                gt = g[:, start:start + n, :]                          # [B,n,token_dim]
                start += n
                p = self.pool_sizes[s]
                gt = gt.transpose(1, 2).reshape(B, self.token_dim, p, p)
                gt = F.interpolate(gt, size=Rs[s].shape[-2:],
                                   mode='bilinear', align_corners=False)
                Ts.append(self.teacher_fuse[s](torch.cat([gt, Rs[s]], dim=1)))

        # 3. teacher change logits
        teacher_logits = [self.teacher_head[s](Ts[s]) for s in range(self.num_stages)]

        losses = {}

        # 4. task gradient（GT deep supervision）
        if self.task_grad:
            task = 0.0
            for s in range(self.num_stages):
                logit = F.interpolate(teacher_logits[s], size=target.shape[-2:],
                                      mode='bilinear', align_corners=False)
                task = task + self._bce_dice(torch.sigmoid(logit), target)
            losses['graft_task'] = task

        # 5. repr gradient（teacher -> probe，KGR 路由；changed/unchanged 各自等预算）
        if self.repr_grad:
            repr_loss = 0.0
            kgr_mean = 0.0
            conf_mean = 0.0
            gap_mean = 0.0
            for s in range(self.num_stages):
                ZS = self.probe[s](self._symmetric(feats1[s], feats2[s]))
                ZT = Ts[s].detach()
                d = 1.0 - F.cosine_similarity(ZS, ZT, dim=1, eps=1e-6)  # [B,H,W]
                if self.kgr:
                    # teacher reliability（绝对值门控，不做类内归一化，保留"不可靠→少教"）
                    C = (2 * torch.sigmoid(teacher_logits[s]) - 1).abs().detach()  # [B,1,H,W]
                    W = (C * d.unsqueeze(1)).detach()                                # [B,1,H,W]
                    pixel = W.squeeze(1) * d                                         # [B,H,W]
                    kgr_mean = kgr_mean + W.mean()
                    conf_mean = conf_mean + C.mean()
                    gap_mean = gap_mean + d.mean()
                else:
                    pixel = d
                # changed / unchanged 分别取均值再平均，保证两类同等梯度预算
                gt = F.interpolate(target, size=d.shape[-2:], mode='nearest').squeeze(1)  # [B,H,W]
                pos = (gt == 1.0).float()
                neg = (gt == 0.0).float()
                loss_changed = (pixel * pos).sum() / (pos.sum() + 1e-6)
                loss_unchanged = (pixel * neg).sum() / (neg.sum() + 1e-6)
                repr_loss = repr_loss + 0.5 * (loss_changed + loss_unchanged)
            losses['graft_repr'] = repr_loss
            if self.kgr:
                losses['graft_kgr'] = kgr_mean / self.num_stages
                losses['graft_conf'] = conf_mean / self.num_stages
                losses['graft_gap'] = gap_mean / self.num_stages

        return losses

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _symmetric(f1, f2):
        return torch.cat([f1 + f2, (f1 - f2).abs(), f1 * f2], dim=1)

    @staticmethod
    def _bce_dice(pred, target):
        bce = F.binary_cross_entropy(pred, target)
        inter = (pred * target).sum()
        eps = 1e-5
        dice = (2 * inter + eps) / (pred.sum() + target.sum() + eps)
        return bce + 1 - dice
