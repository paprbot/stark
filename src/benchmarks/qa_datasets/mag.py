import os.path as osp
import json
import torch
from src.benchmarks.qa_datasets.stark_qa import STaRKDataset


class MAGSTaRKDataset(STaRKDataset):
    
    def __init__(self, stark_qa_dir, split_dir, human_generated_eval=False):
        super(MAGSTaRKDataset, self).__init__(stark_qa_dir, split_dir, human_generated_eval)