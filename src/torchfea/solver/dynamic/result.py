from __future__ import annotations

import numpy as np
import torch

from ..baseresult import BaseResult


class DynamicResult(BaseResult):
    """The result of a dynamic finite element analysis (FEA) simulation."""

    def __init__(self,
                 GC_list: list[torch.Tensor],
                 GV_list: list[torch.Tensor],
                 GA_list: list[torch.Tensor],
                 time_list: list[float],
                 load_params: dict[str, torch.Tensor],
                 total_time: float = 0.0,
                 time_items: dict[str, list[float]] = None,
                 model_hash: str | None = None):
        super().__init__()

        self.GC_list = [x.detach().clone() for x in GC_list]
        self.GV_list = [x.detach().clone() for x in GV_list]
        self.GA_list = [x.detach().clone() for x in GA_list]
        self.time_list = list(time_list)

        self.load_params: dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in load_params.items()}

        self.total_time = float(total_time)
        self.time_items = time_items if time_items is not None else {}
        self.model_hash = model_hash

    def save(self, path: str):
        if not self.model_hash:
            raise ValueError("Cannot save a result without a model hash.")
        save_dict = {
            'GC_list': [x.cpu().numpy() for x in self.GC_list],
            'GV_list': [x.cpu().numpy() for x in self.GV_list],
            'GA_list': [x.cpu().numpy() for x in self.GA_list],
            'time_list': np.array(self.time_list),
            'load_params': {k: v.cpu().numpy() for k, v in self.load_params.items()},
            'total_time': self.total_time,
            'time_items': self.time_items,
            'model_hash': self.model_hash,
        }
        np.savez_compressed(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> "DynamicResult":
        with np.load(path, allow_pickle=True) as data:
            load_params_np = data['load_params'].item()
            load_params = {k: torch.tensor(v) for k, v in load_params_np.items()}
            model_hash = str(np.asarray(data['model_hash']).reshape(-1)[0])
            return cls(
                GC_list=[torch.tensor(x) for x in data['GC_list']],
                GV_list=[torch.tensor(x) for x in data['GV_list']],
                GA_list=[torch.tensor(x) for x in data['GA_list']],
                time_list=data['time_list'].tolist(),
                load_params=load_params,
                total_time=float(data['total_time']),
                time_items=data['time_items'].item(),
                model_hash=model_hash,
            )
