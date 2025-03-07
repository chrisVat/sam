import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from schedule_base import Schedule
from utils import make_supervised_data_module

from utils import is_running_distributed

class Full(Schedule):
    def __init__(self,
        model,
        tokenizer,
        args=None
    ):
        super(Full, self).__init__(
            model,
            tokenizer,
            args,
        )

    def initialize_labeled_data(self):
        """initialize labeled data as full data"""
        if not is_running_distributed() or torch.distributed.get_rank() == 0:
            self.labeled_idx[:] = True

    def get_updated_train_data(self):
        """load & make round labeled data -> training data"""
        data_module = make_supervised_data_module(tokenizer=self.tokenizer, data_path=self.full_data_path)
        return data_module