# Copyright 2026 yasutoshi-lab and The HuggingFace Inc. team. All rights reserved.
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
"""np.memmap-backed fixed-length sequence dataset for Ranunculus pre-training.

Each .bin file is a flat uint32 token array; the dataset yields fixed-length
slices so no padding / collation is required downstream.
"""

from __future__ import annotations

import numpy as np
import torch


class PackedDataset(torch.utils.data.Dataset):
    """Dataset over one or more packed .bin files of token ids.

    Args:
        bin_paths: List of .bin files produced by prepare_ranunculus_data.py.
        seq_len: Fixed sequence length (design default: 8192).
    """

    def __init__(self, bin_paths: list[str], seq_len: int = 8192):
        self.arrs = [np.memmap(p, dtype=np.uint32, mode="r") for p in bin_paths]
        self.seq_len = seq_len
        lengths = [len(a) // seq_len for a in self.arrs]
        self.offsets = np.cumsum([0, *lengths])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx = int(np.searchsorted(self.offsets[1:], index, side="right"))
        local = index - self.offsets[file_idx]
        start = local * self.seq_len
        end = start + self.seq_len
        ids = torch.from_numpy(self.arrs[file_idx][start:end].astype(np.int64))
        return {"input_ids": ids, "labels": ids}
