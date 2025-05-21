# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import List, Literal, Optional

import torch


def reshape_weight_task_tensors(task_tensors, weights):
    """
    Reshapes `weights` to match the shape of `task_tensors` by unsqeezing in the remaining dimenions.

    Args:
        task_tensors (`torch.Tensor`): The tensors that will be used to reshape `weights`.
        weights (`torch.Tensor`): The tensor to be reshaped.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    weights = weights.view(new_shape)
    return weights


def magnitude_based_pruning(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """
    Prune the smallest values of the task tensors and retain the top-k values based on the specified fraction
    `density`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The tensor with the pruned weights.
    """
    mask = torch.zeros_like(tensor).reshape(-1)
    k = int(density * tensor.numel())
    top_k = torch.topk(tensor.abs().reshape(-1), k=k, largest=True)
    mask[top_k[1]] = 1
    return tensor * mask.reshape(tensor.shape)


def random_pruning(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    """
    Prune random values based on the specified fraction `density`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `torch.Tensor`: The pruned tensor.
    """
    mask = torch.bernoulli(torch.full_like(input=tensor, fill_value=density))
    pruned_tensor = tensor * mask
    if rescale:
        torch.div(input=pruned_tensor, other=density)
    return pruned_tensor


def prune(
    tensor: torch.Tensor, density: float, method: Literal["magnitude", "random"], rescale: bool = False
) -> torch.Tensor:
    """
    Prune the values of task tensors based on the `method`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        method (`str`):The method to use to prune. Should be one of ["magnitude", "random"].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `torch.Tensor`: The pruned tensor.
    """
    if density >= 1:
        warnings.warn(f"The density {density} is greater than or equal to 1, no pruning will be performed.")
        return tensor
    elif density < 0:
        raise ValueError(f"Density should be >= 0, got {density}")
    if method == "magnitude":
        return magnitude_based_pruning(tensor, density)
    elif method == "random":
        return random_pruning(tensor, density, rescale=rescale)
    else:
        raise ValueError(f"Unknown method {method}")


def calculate_majority_sign_mask(
    tensor: torch.Tensor, method: Literal["total", "frequency"] = "total"
) -> torch.Tensor:
    """
    Get the mask of the majority sign across the task tensors. Task tensors are stacked on dimension 0.

    Args:
        tensor (`torch.Tensor`):The tensor to get the mask from.
        method (`str`):The method to use to get the mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The majority sign mask.
    """

    sign = tensor.sign()
    if method == "total":
        sign_magnitude = tensor.sum(dim=0)
    elif method == "frequency":
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(task_tensors: torch.Tensor, majority_sign_mask: torch.Tensor) -> torch.Tensor:
    """
    Merge the task tensors using disjoint merge.

    Args:
        task_tensors (`torch.Tensor`):The task tensors to merge.
        majority_sign_mask (`torch.Tensor`):The mask of the majority sign across the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)

#### Todo: modify steps of merging algorithms or add new methods in merge_utils.py ####

def task_arithmetic(task_tensors: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def magnitude_prune(task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float) -> torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`): The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="magnitude") for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def ties(
    task_tensors: List[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """
    Merge the task tensors using `ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="magnitude") for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    # Disjoint Merge
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors


#### Todo: Add new methods, reuse modules in other algorithms ####
#### e.g. if you want to implement “sce” algorithm ####

def sce(
    task_tensors: List[torch.Tensor],
    valid_weights: Optional[torch.Tensor] = None,  # 改为可选参数
    density: float = 1.0,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    '''
    Merge the task tensors using `sce`. Reference: paper-"https://arxiv.org/abs/2408.07990", github-"https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/sce.py"

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    '''
    # S: select top-k variance elements in matrices (among different task vectors) v.s. TIES (pruning individually)
    # C: sum of squares of elements to obtain merging coefficient for each target LLM
    # E: filter elements with minority directions
    
    # Stack all task tensors into a single tensor of shape [num_tasks, ...]
    stacked_tensors = torch.stack(task_tensors, dim=0)
    
    # Reshape weights for broadcasting
    adapter_weights = valid_weights
    if adapter_weights.dim() < stacked_tensors.dim():
        adapter_weights = reshape_weight_task_tensors(stacked_tensors, adapter_weights)

    # Apply valid_weights directly to the input tensors
    stacked_tensors = stacked_tensors * adapter_weights

    # Apply variance-based selection mask if density < 1.0
    if density < 1.0:
        mask = sce_mask(stacked_tensors, density)
        stacked_tensors = stacked_tensors * mask
    
    # Compute majority sign agreement mask
    majority_sign_mask = calculate_majority_sign_mask(stacked_tensors, method=majority_sign_method)
    
    # Compute task-specific weights
    weights = sce_weight(stacked_tensors)
    
    # Reshape weights for broadcasting
    if weights.dim() < stacked_tensors.dim():
        weights = reshape_weight_task_tensors(stacked_tensors, weights)
    
    # Apply majority sign mask to weights (zero out weights that disagree with majority)
    masked_weights = weights * majority_sign_mask
    
    # Weighted summation of masked task tensors
    merged_tensor = torch.sum(stacked_tensors * masked_weights, dim=0)
    
    # Normalize by sum of weights (with clamping to avoid division by zero)
    weight_sum = torch.sum(masked_weights, dim=0)
    merged_tensor = merged_tensor / weight_sum.clamp(min=1e-6)
    
    return merged_tensor

    
def sce_weight(task_tensors: torch.Tensor) -> torch.Tensor:
	# Implementation of C step
    
    # Compute squared magnitude (energy) per task
    weights = torch.mean(task_tensors**2, dim=list(range(1, task_tensors.dim())))
	
    # Sum all weights to normalize
    weight_sum = torch.sum(weights).item()
    
    # Handle edge case: if all task tensors are 0, fallback to uniform weights
    if abs(weight_sum) < 1e-6:
        return torch.ones_like(weights) / weights.shape[0]
	
    # Normalize to form a probability distribution over tasks
    return weights / weight_sum

def sce_mask_orig(task_tensors: torch.Tensor, density: float, mask_dtype: Optional[torch.dtype] = None):
    # Implementation of S step (sce_mask)
    
    # If density is zero, mask out everything 
    if density <= 0:
        return torch.zeros_like(task_tensors, dtype=mask_dtype)
    
    # If density is one, keep everything
    if density >= 1:
        return torch.ones_like(task_tensors, dtype=mask_dtype)
    
    # Compute variance over the task dimension for each parameter
    var = torch.var(task_tensors, dim=0, unbiased=False)
    
    # Count how many parameters have non-zero variance
    nonzero = torch.count_nonzero(var)
    
    # Compute number of parameters to keep based on density
    k = int(nonzero * density)
    if k == 0:
        return torch.zeros_like(task_tensors, dtype=mask_dtype)
    
    # Select the indices of top-k variances
    _, indices = torch.topk(var.abs().view(-1), k=k, largest=True)
    
    # Build binary mask with 1s in selected indices 
    mask = torch.zeros_like(var, dtype=mask_dtype)
    mask.view(-1)[indices] = 1
    return mask

#分位数混合选择（自适应权重）,自动平衡方差和绝对值，分别计算方差和绝对值的分位数，然后组合
def sce_mask_threshold(task_tensors: torch.Tensor, density: float, mask_dtype: Optional[torch.dtype] = None):
    if density <= 0:
        return torch.zeros_like(task_tensors, dtype=mask_dtype)
    if density >= 1:
        return torch.ones_like(task_tensors, dtype=mask_dtype)

    var = torch.var(task_tensors, dim=0, unbiased=False)
    abs_mean = torch.mean(task_tensors.abs(), dim=0)

    # 计算方差和绝对值的分位数（前 density 比例）
    var_threshold = torch.quantile(var, 1 - density)
    abs_threshold = torch.quantile(abs_mean, 1 - density)

    # 满足任一条件的参数均被选中
    mask = ((var >= var_threshold) | (abs_mean >= abs_threshold)).to(mask_dtype)
    return mask

#自动平衡方差和绝对值, 获取方差和绝对值最大的前density参数
def sce_mask(task_tensors: torch.Tensor, density: float, mask_dtype: Optional[torch.dtype] = None):
    if density <= 0:
        return torch.zeros_like(task_tensors, dtype=mask_dtype)
    if density >= 1:
        return torch.ones_like(task_tensors, dtype=mask_dtype)

    # 计算方差和绝对均值
    var = torch.var(task_tensors, dim=0, unbiased=False)
    abs_mean = torch.mean(task_tensors, dim=0).abs()
    
    # 使用小的阈值判断非零元素
    nonzero_var = torch.count_nonzero(var)
    nonzero_mean = torch.count_nonzero(abs_mean)
    
    num_params_var = int(nonzero_var * density)
    num_params_mean = int(nonzero_mean * density)
    
    # 获取topk索引
    _, var_top_indices = torch.topk(var.abs().view(-1), k=num_params_var, largest=True)
    _, abs_top_indices = torch.topk(abs_mean.abs().view(-1), k=num_params_mean, largest=True)
    
    # 合并索引
    combined_indices = torch.cat([var_top_indices, abs_top_indices]).unique()
    
    # 创建mask
    mask = torch.zeros_like(task_tensors, dtype=mask_dtype)
    mask.view(-1)[var_top_indices] = 1
    
    return mask


def dare_linear(task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float) -> torch.Tensor:
    """
    Merge the task tensors using `dare linear`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="random", rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def dare_ties(
    task_tensors: List[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """
    Merge the task tensors using `dare ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="random", rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    # Disjoint Merge
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors
