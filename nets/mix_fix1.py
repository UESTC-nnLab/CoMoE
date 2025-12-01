import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from .darknet import BaseConv
class StableHypersphericalPrototype(nn.Module):
    def __init__(self, feature_dim: int, num_prototypes: int, num_domains: int,
                 temperature: float = 0.1, momentum: float = 0.95,
                 prototype_mode: str = 'mixed',
                 projection_mode: str = 'project',
                 prototype_loss_type: str = 'ce_soft'):
        super().__init__()
        self.prototype_mode = prototype_mode.lower()
        self.projection_mode = projection_mode.lower()
        self.prototype_loss_type = prototype_loss_type.lower()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.num_domains = num_domains
        self.momentum = momentum

        self.temperature = nn.Parameter(torch.tensor(temperature))

        prototypes = torch.empty(num_domains, num_prototypes, feature_dim)
        nn.init.xavier_uniform_(prototypes)
        self.register_buffer('domain_prototypes', F.normalize(prototypes, dim=-1))

        global_prototypes = torch.empty(num_prototypes, feature_dim)
        nn.init.xavier_uniform_(global_prototypes)
        self.register_buffer('global_prototypes', F.normalize(global_prototypes, dim=-1))

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

        self.prototype_weight = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_prototypes),
            nn.Softmax(dim=-1)
        )

        self.register_buffer('update_count', torch.zeros(num_domains, num_prototypes))

    def get_feature(self, features: torch.Tensor) -> torch.Tensor:
        if self.projection_mode == 'none':
            return features
        elif self.projection_mode == 'normalize':
            return F.normalize(features, p=2, dim=-1)
        elif self.projection_mode == 'project':
            projected = self.projection_head(features)
            return F.normalize(projected, dim=-1)
        else:
            raise ValueError(f"Unknown projection_mode: {self.projection_mode}")

    def adaptive_prototype_update(self, features: torch.Tensor, domain_ids: torch.Tensor):
        if not self.training:
            return

        feats = self.get_feature(features)

        with torch.no_grad():
            for domain_id in torch.unique(domain_ids):
                domain_mask = (domain_ids == domain_id)
                if domain_mask.sum() == 0:
                    continue

                domain_features = feats[domain_mask]
                similarities = torch.mm(domain_features, self.domain_prototypes[domain_id].T)
                closest_prototypes = similarities.argmax(dim=1)

                for proto_id in torch.unique(closest_prototypes):
                    proto_mask = (closest_prototypes == proto_id)
                    if proto_mask.sum() == 0:
                        continue

                    update_lr = self.momentum ** (1 + self.update_count[domain_id, proto_id] * 0.01)
                    new_proto = domain_features[proto_mask].mean(dim=0)
                    new_proto = F.normalize(new_proto, dim=-1)

                    self.domain_prototypes[domain_id, proto_id] = \
                        update_lr * self.domain_prototypes[domain_id, proto_id] + \
                        (1 - update_lr) * new_proto

                    self.update_count[domain_id, proto_id] += 1

            for proto_id in range(self.num_prototypes):
                weights = F.softmax(-self.update_count[:, proto_id], dim=0)
                global_proto = (weights.unsqueeze(-1) * self.domain_prototypes[:, proto_id]).sum(dim=0)
                self.global_prototypes[proto_id] = F.normalize(global_proto, dim=-1)

    def compute_prototype_loss(self, features: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if self.prototype_loss_type == 'none':
            return torch.tensor(0.0, device=features.device)

        feats = self.get_feature(features)
        total_loss = 0.0
        valid_domains = 0

        for domain_id in torch.unique(domain_ids):
            domain_mask = (domain_ids == domain_id)
            if domain_mask.sum() <= 1:
                continue

            domain_features = feats[domain_mask]
            domain_prototypes = self.domain_prototypes[domain_id]
            temp = torch.clamp(self.temperature, min=0.01, max=1.0)
            similarities = torch.mm(domain_features, domain_prototypes.T) / temp

            if self.prototype_loss_type == 'ce_hard':
                targets = similarities.argmax(dim=1)
                loss = F.cross_entropy(similarities, targets)

            elif self.prototype_loss_type == 'ce_soft':
                targets = similarities.argmax(dim=1)
                smooth_targets = F.one_hot(targets, num_classes=self.num_prototypes).float()
                smooth_targets = 0.9 * smooth_targets + 0.1 / self.num_prototypes
                log_probs = F.log_softmax(similarities, dim=1)
                loss = -(smooth_targets * log_probs).sum(dim=1).mean()

            elif self.prototype_loss_type == 'infonce':
                labels = similarities.argmax(dim=1)
                log_probs = F.log_softmax(similarities, dim=1)
                loss = F.nll_loss(log_probs, labels)

            elif self.prototype_loss_type == 'triplet':
                triplet_loss = 0.0
                for i in range(domain_features.size(0)):
                    anchor = domain_features[i]
                    pos_id = similarities[i].argmax()
                    positive = domain_prototypes[pos_id]
                    neg_ids = [j for j in range(self.num_prototypes) if j != pos_id.item()]
                    if not neg_ids:
                        continue
                    neg_idx = torch.randint(len(neg_ids), (1,)).item()
                    negative = domain_prototypes[neg_ids[neg_idx]]
                    triplet_loss += F.triplet_margin_loss(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0), margin=1.0)
                loss = triplet_loss / domain_features.size(0)

            else:
                raise ValueError(f"Unknown prototype_loss_type: {self.prototype_loss_type}")

            total_loss += loss
            valid_domains += 1

        return total_loss / max(valid_domains, 1)

    def forward(self, features: torch.Tensor, domain_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.get_feature(features)
        proto_weights = self.prototype_weight(features)

        self.adaptive_prototype_update(features, domain_ids)

        enhanced_features = []
        for i, domain_id in enumerate(domain_ids):
            domain_id = min(domain_id.item(), self.num_domains - 1)

            if self.prototype_mode == 'domain':
                mixed_prototypes = self.domain_prototypes[domain_id]
            elif self.prototype_mode == 'global':
                mixed_prototypes = self.global_prototypes
            elif self.prototype_mode == 'mixed':
                mixed_prototypes = 0.6 * self.domain_prototypes[domain_id] + 0.4 * self.global_prototypes
            else:
                raise ValueError(f"Unknown prototype_mode: {self.prototype_mode}")

            prototype_feature = torch.mm(proto_weights[i:i+1], mixed_prototypes)
            enhanced_feat = feats[i:i+1] + 0.2 * prototype_feature
            enhanced_features.append(enhanced_feat)

        return torch.cat(enhanced_features, dim=0), proto_weights

class UpgradedExpert(nn.Module):
    def __init__(self, channels: int, domain_dim: int = 4, use_att=True, use_adapter=True):
        super().__init__()
        self.channels = channels
        self.use_att= use_att
        self.use_adapter = use_adapter

        self.domain_transform = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.domain_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

        self.temporal_expert = CSWF(in_channel=channels, out_channel=channels)

        if use_att:
            self.att = Res_CBAM_block(in_channels=channels, out_channels=channels)

        if use_adapter:
            self.adapter = nn.Embedding(domain_dim, channels)

    def forward(self, current_feat: torch.Tensor, ref_feat: torch.Tensor = None, domain_ids: torch.Tensor = None):
        B, C, H, W = current_feat.shape
        out = current_feat

        d_out = self.domain_transform(current_feat)
        gate1 = self.domain_gate(d_out)
        out = out + gate1 * d_out

        if ref_feat is not None:
            out = self.temporal_expert(current_feat,ref_feat)

        if self.use_att:
            out = self.att(out)

        if self.use_adapter and domain_ids is not None:
            domain_bias = self.adapter(domain_ids).view(B, C, 1, 1)
            out = out + domain_bias

        return out

class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()

        inter_planes = planes // reduce_ratio
        self.query_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out + x
        return out

class CSWF(nn.Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.conv_1 = nn.Sequential(
            BaseConv(in_channel, in_channel//2, 1, 1),
            BaseConv(in_channel//2, in_channel, 1, 1)
        )
        self.conv_2 = nn.Sequential(
            BaseConv(in_channel, in_channel//2, 1, 1),
            BaseConv(in_channel//2, out_channel, 1, 1, act="sigmoid")
        )
        self.conv = nn.Sequential(
            BaseConv(out_channel, out_channel//2, 1, 1),
            BaseConv(out_channel//2, out_channel, 1, 1)
        )

    def forward(self, r_feat, c_feat):
        m_feat = r_feat + c_feat
        m_feat = self.conv_2(self.conv_1(m_feat))
        m_feat = self.conv(c_feat*m_feat + r_feat*(1-m_feat))

        m_feat = self.conv_2(self.conv_1(m_feat))
        m_feat = self.conv(c_feat*m_feat + r_feat*(1-m_feat))

        return m_feat

class OptimizedMixtureOfExperts(nn.Module):
    def __init__(self, channels: int, num_experts: int, num_domains: int, 
                 top_k: int = 2, dropout: float = 0.1,
                 topk_strategy: str = 'soft'):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.num_domains = num_domains
        self.top_k = min(top_k, num_experts)
        self.topk_strategy = topk_strategy.lower()

        self.experts = nn.ModuleList([
            UpgradedExpert(channels=self.channels, domain_dim=num_domains) for _ in range(num_experts)
        ])

        gate_input_dim = channels * 2 + 16
        self.domain_embed = nn.Embedding(num_domains, 16)
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_experts),
            nn.Softmax(dim=-1)
        )

        self.register_buffer('expert_usage', torch.ones(num_experts) / num_experts)
        self.load_balance_weight = 0.01
        self.diversity_weight = 0.05

    def compute_load_balance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        current_usage = gate_probs.mean(dim=0)
        self.expert_usage = 0.9 * self.expert_usage + 0.1 * current_usage
        target_usage = torch.ones_like(self.expert_usage) / self.num_experts
        return F.mse_loss(self.expert_usage, target_usage)

    def compute_diversity_loss(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        B, E, C, H, W = expert_outputs.shape
        expert_flat = expert_outputs.view(B, E, -1)
        loss = 0.0
        for i in range(E):
            for j in range(i + 1, E):
                sim = F.cosine_similarity(expert_flat[:, i], expert_flat[:, j], dim=-1)
                loss += sim.mean()
        return loss / (E * (E - 1) / 2)

    def forward(self, current_feat: torch.Tensor, ref_feat: torch.Tensor = None,
                domain_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = current_feat.shape

        curr_pool = F.adaptive_avg_pool2d(current_feat, 1).flatten(1)
        ref_pool = F.adaptive_avg_pool2d(ref_feat, 1).flatten(1) if ref_feat is not None else torch.zeros_like(curr_pool)
        domain_emb = self.domain_embed(domain_ids)
        gate_input = torch.cat([curr_pool, ref_pool, domain_emb], dim=1)
        gate_probs = self.gate_network(gate_input)

        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(current_feat, ref_feat, domain_ids)
            expert_outputs.append(expert_out)
        expert_outputs = torch.stack(expert_outputs, dim=1)

        final_output = torch.zeros_like(current_feat)

        if self.topk_strategy == 'full':
            gate_probs = gate_probs / (gate_probs.sum(dim=1, keepdim=True) + 1e-8)
            for i in range(B):
                for j in range(self.num_experts):
                    weight = gate_probs[i, j].view(1, 1, 1)
                    final_output[i] += weight * expert_outputs[i, j]

        elif self.topk_strategy == 'soft':
            top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=1)
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=1, keepdim=True) + 1e-8)
            for i in range(B):
                for j, expert_idx in enumerate(top_k_indices[i]):
                    weight = top_k_probs[i, j].view(1, 1, 1)
                    final_output[i] += weight * expert_outputs[i, expert_idx]

        elif self.topk_strategy == 'hard':
            top1_index = gate_probs.argmax(dim=1)
            for i in range(B):
                final_output[i] = expert_outputs[i, top1_index[i]]

        else:
            raise ValueError(f"Unknown topk_strategy: {self.topk_strategy}")

        final_output = current_feat + 0.5 * final_output
        load_balance_loss = self.compute_load_balance_loss(gate_probs)
        diversity_loss = self.compute_diversity_loss(expert_outputs)

        return final_output, load_balance_loss, diversity_loss

class OptimizedMoE_NoTemporalNoDomainNoTopK(nn.Module):
    def __init__(self, channels: int, num_experts: int, num_domains: int, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.num_domains = num_domains

        self.experts = nn.ModuleList([
            UpgradedExpert(channels=channels, domain_dim=num_domains) for _ in range(num_experts)
        ])

        self.gate_network = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_experts),
            nn.Softmax(dim=-1)
        )

        self.register_buffer('expert_usage', torch.ones(num_experts) / num_experts)
        self.load_balance_weight = 0.01
        self.diversity_weight = 0.05

    def compute_load_balance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        current_usage = gate_probs.mean(dim=0)
        self.expert_usage = 0.9 * self.expert_usage + 0.1 * current_usage
        target_usage = torch.ones_like(self.expert_usage) / self.num_experts
        return F.mse_loss(self.expert_usage, target_usage)

    def compute_diversity_loss(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        B, E, C, H, W = expert_outputs.shape
        expert_flat = expert_outputs.view(B, E, -1)
        loss = 0.0
        for i in range(E):
            for j in range(i + 1, E):
                sim = F.cosine_similarity(expert_flat[:, i], expert_flat[:, j], dim=-1)
                loss += sim.mean()
        return loss / (E * (E - 1) / 2)

    def forward(self, current_feat: torch.Tensor, ref_feat: torch.Tensor = None,
                domain_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = current_feat.shape

        curr_pool = F.adaptive_avg_pool2d(current_feat, 1).flatten(1)
        gate_input = curr_pool
        gate_probs = self.gate_network(gate_input)

        expert_outputs = torch.stack([
            expert(current_feat, None, domain_ids) for expert in self.experts
        ], dim=1)

        final_output = torch.sum(
            gate_probs.view(B, self.num_experts, 1, 1, 1) * expert_outputs, dim=1
        )

        final_output = current_feat + 0.5 * final_output

        load_balance_loss = self.compute_load_balance_loss(gate_probs)
        diversity_loss = self.compute_diversity_loss(expert_outputs)

        return final_output, load_balance_loss, diversity_loss

class ContextAwareADFM(nn.Module):
    def __init__(self, channels: int, num_domains: int):
        super().__init__()
        self.channels = channels
        self.num_domains = num_domains

        self.domain_embed = nn.Embedding(num_domains, channels)

        self.modulation_mlp = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, channels * 2)
        )

        self.align_conv = Res_CBAM_block(channels,channels)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        domain_vec = self.domain_embed(domain_ids)
        global_feat = self.global_pool(features).flatten(1)

        context_input = torch.cat([global_feat, domain_vec], dim=1)
        gamma_beta = self.modulation_mlp(context_input).view(B, 2, C, 1, 1)
        gamma, beta = gamma_beta[:, 0], gamma_beta[:, 1]

        modulated = gamma * features + beta
        aligned = self.align_conv(modulated)

        return aligned
class AdaptiveNoiseModule2(nn.Module):
    def __init__(self, channels: int, num_domains: int, temperature: float = 0.2):
        super().__init__()
        self.channels = channels
        self.num_domains = num_domains
        self.temperature = temperature

        self.noise_scale = nn.Parameter(torch.ones(num_domains) * 0.05)

        self.projector = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B, C = z1.shape
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim_matrix = torch.mm(z1, z2.t())
        logits = sim_matrix / self.temperature
        labels = torch.arange(B, device=z1.device)

        return F.cross_entropy(logits, labels)

    def forward(self, features: torch.Tensor, domain_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return features, torch.tensor(0.0, device=features.device)

        noisy_features = []
        clean_proj, noisy_proj = [], []

        for i, domain_id in enumerate(domain_ids):
            domain_id = min(domain_id.item(), self.num_domains - 1)
            feat = features[i:i+1]

            noise_scale = torch.sigmoid(self.noise_scale[domain_id]) * 0.1
            noise = torch.randn_like(feat) * noise_scale
            noisy_feat = feat + noise
            noisy_features.append(noisy_feat)

            clean_z = self.pool(self.projector(feat)).view(1, -1)
            noisy_z = self.pool(self.projector(noisy_feat)).view(1, -1)
            clean_proj.append(clean_z)
            noisy_proj.append(noisy_z)

        enhanced_features = torch.cat(noisy_features, dim=0)
        clean_proj = torch.cat(clean_proj, dim=0)
        noisy_proj = torch.cat(noisy_proj, dim=0)

        contrastive_loss = self.contrastive_loss(clean_proj, noisy_proj)
        return enhanced_features, contrastive_loss

class AdaptiveNoiseModule(nn.Module):
    def __init__(self, channels: int, num_domains: int):
        super().__init__()
        self.channels = channels
        self.num_domains = num_domains

        self.noise_scale = nn.Parameter(torch.ones(num_domains) * 0.05)

        self.consistency_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, features: torch.Tensor, domain_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return features, torch.tensor(0.0, device=features.device)

        noisy_features = []
        consistency_loss = 0.0

        for i, domain_id in enumerate(domain_ids):
            domain_id = min(domain_id.item(), self.num_domains - 1)
            feat = features[i:i+1]

            noise_scale = torch.sigmoid(self.noise_scale[domain_id]) * 0.1
            noise = torch.randn_like(feat) * noise_scale
            noisy_feat = feat + noise

            clean_out = self.consistency_net(feat)
            noisy_out = self.consistency_net(noisy_feat)
            consistency_loss += F.mse_loss(clean_out, noisy_out)

            noisy_features.append(noisy_feat)

        enhanced_features = torch.cat(noisy_features, dim=0)
        consistency_loss = consistency_loss / len(domain_ids)

        return enhanced_features, consistency_loss

class OptimizedHyperMoENeck(nn.Module):
    def __init__(self, channels: List[int] = [128, 256, 512], num_frame: int = 5,
                 num_domains: int = 4, num_experts: int = 6, num_prototypes: int = 32,
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_frame = num_frame
        self.num_domains = num_domains
        self.channels = channels[0]

        self.hyperspherical_prototype = StableHypersphericalPrototype(
            feature_dim=self.channels,
            num_prototypes=num_prototypes,
            num_domains=num_domains,
            prototype_mode='mixed',
            projection_mode= 'project',
            prototype_loss_type='ce_soft',
        )

        self.mixture_of_experts = OptimizedMixtureOfExperts(
            channels=self.channels,
            num_experts=num_experts,
            num_domains=num_domains,
            top_k=top_k,
            dropout=dropout,
            topk_strategy='soft'
        )

        self.cross_view_alignment = ContextAwareADFM(
            channels=self.channels,
            num_domains=num_domains
        )

        self.adaptive_noise = AdaptiveNoiseModule2(
            channels=self.channels,
            num_domains=num_domains
        )

        self.prototype_enhance = Res_CBAM_block(self.channels*2, self.channels)

        self.temporal_fusion = nn.Sequential(
            BaseConv(channels[0]*num_frame, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1, act='sigmoid')
        )

        self.final_fusion = Res_CBAM_block(self.channels * 3, self.channels)
        self.final_enhance = NonLocalBlock(self.channels)

        self.loss_weights = nn.ParameterDict({
            'prototype': nn.Parameter(torch.tensor(0.1)),
            'load_balance': nn.Parameter(torch.tensor(0.01)),
            'diversity_loss':nn.Parameter(torch.tensor(0.01)),
            'consistency': nn.Parameter(torch.tensor(0.05))
        })

        self.apply(self._init_weights)

        self.conv_abla  = self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1,)
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def spatial_prototype_enhancement(self, features: torch.Tensor, 
                                    prototype_features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape

        prototype_spatial = prototype_features.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)

        enhanced_input = torch.cat([features, prototype_spatial], dim=1)
        enhanced_output = self.prototype_enhance(enhanced_input)

        return enhanced_output

    def forward(self, feats: List[torch.Tensor], domain_labels: torch.Tensor,
                return_losses: bool = True) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size = feats[0].size(0)
        device = feats[0].device

        if domain_labels is None:
            domain_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        domain_labels = torch.clamp(domain_labels, 0, self.num_domains - 1)

        losses = {}
        current_feat = feats[-1]

        global_feat = F.adaptive_avg_pool2d(current_feat, 1).squeeze(-1).squeeze(-1)
        prototype_features, proto_weights = self.hyperspherical_prototype(global_feat, domain_labels)

        if return_losses:
            prototype_loss = self.hyperspherical_prototype.compute_prototype_loss(global_feat, domain_labels)
            losses['prototype_loss'] = torch.sigmoid(self.loss_weights['prototype']) * prototype_loss

        enhanced_current = self.spatial_prototype_enhancement(current_feat, prototype_features)

        expert_outputs = []
        total_load_loss = 0
        total_diver_loss = 0

        for i, feat in enumerate(feats):
            if i < len(feats) - 1:
                expert_out, load_loss, diver_loss = self.mixture_of_experts(feat, enhanced_current, domain_labels)
            else:
                expert_out, load_loss, diver_loss = self.mixture_of_experts(enhanced_current, None, domain_labels)

            expert_outputs.append(expert_out)
            total_load_loss += load_loss
            total_diver_loss += diver_loss

        if return_losses:
            losses['load_balance_loss'] = torch.sigmoid(self.loss_weights['load_balance']) * \
                                        (total_load_loss / len(feats))
            losses['diversity_loss'] = torch.sigmoid(self.loss_weights['diversity_loss']) * \
                                        (total_diver_loss / len(feats))

        aligned_outputs = []
        for expert_out in expert_outputs:
            aligned_out = self.cross_view_alignment(expert_out, domain_labels)
            aligned_outputs.append(aligned_out)

        robust_outputs = []
        total_consistency_loss = 0

        for aligned_out in aligned_outputs:
            robust_out, consistency_loss = self.adaptive_noise(aligned_out, domain_labels)
            robust_outputs.append(robust_out)
            total_consistency_loss += consistency_loss

        if return_losses:
            losses['consistency_loss'] = torch.sigmoid(self.loss_weights['consistency']) * \
                                        (total_consistency_loss / len(aligned_outputs))

        temporal_input = torch.cat(robust_outputs, dim=1)
        temporal_weights = self.temporal_fusion(temporal_input)

        final_input = torch.cat([
            robust_outputs[-1],
            enhanced_current,
            temporal_weights * robust_outputs[-1]
        ], dim=1)
        final_features = self.final_fusion(final_input)
        out = self.final_enhance(final_features)

        return [out], losses

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out 
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

if __name__ == "__main__":
    print("🚀 Testing Optimized HyperMoE-Neck...")

    neck = OptimizedHyperMoENeck(
        channels=[128, 256, 512],
        num_frame=5,
        num_domains=4,
        num_experts=6,
        num_prototypes=32,
        top_k=2,
        dropout=0.1
    )

    feats = [torch.randn(4, 128, 64, 64) for _ in range(5)]
    domain_labels = torch.tensor([0, 1, 2, 3])

    print(f"📥 Input shapes: {[feat.shape for feat in feats]}")
    print(f"📥 Domain labels: {domain_labels}")

    try:
        neck.eval()
        with torch.no_grad():
            outputs, losses = neck(feats, domain_labels, return_losses=False)
        print(f"✅ Inference mode: Output shape {outputs[0].shape}")

        neck.train()
        outputs, losses = neck(feats, domain_labels, return_losses=True)

        print(f"✅ Training mode: Output shape {outputs[0].shape}")
        print(f"✅ Losses: {list(losses.keys())}")

        for loss_name, loss_value in losses.items():
            print(f"  📊 {loss_name}: {loss_value.item():.6f}")

        total_loss = sum(losses.values())
        total_loss.backward()

        grad_norm = 0
        for name, param in neck.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        print(f"✅ Gradient norm: {grad_norm:.6f}")
        print(f"📈 Total parameters: {sum(p.numel() for p in neck.parameters()):,}")

        optimizer = torch.optim.AdamW(neck.parameters(), lr=1e-3, weight_decay=1e-4)

        print("\n🔥 Multi-step training test:")
        for step in range(5):
            optimizer.zero_grad()
            outputs, losses = neck(feats, domain_labels, return_losses=True)
            total_loss = sum(losses.values())
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(neck.parameters(), max_norm=1.0)

            optimizer.step()
            print(f"  Step {step}: Loss = {total_loss.item():.6f}")

        print("✅ All tests passed! Model is ready for training.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
