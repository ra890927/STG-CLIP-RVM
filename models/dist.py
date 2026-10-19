"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from torch import distributed as dist


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def concat_all_gather_unique(tensor: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Concatenate all gathered tensors and remove duplicates.
    """

    gathered_labels = concat_all_gather(labels)
    gathered_tensor = concat_all_gather(tensor)

    unique_labels, inverse_indices = torch.unique(
        gathered_labels, dim=0, return_inverse=True
    )

    unique_tensor = []
    for i in range(len(unique_labels)):
        mask = (inverse_indices == i)
        unique_tensor.append(gathered_tensor[mask].mean(dim=0))

    return torch.stack(unique_tensor), unique_labels


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
