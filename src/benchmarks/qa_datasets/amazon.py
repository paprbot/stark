import os.path as osp
import json
import torch
from src.benchmarks.qa_datasets.stark_qa import STaRKDataset


class AmazonSTaRKDataset(STaRKDataset):
    
    def __init__(self, stark_qa_dir, split_dir, human_generated_eval=False):
        super(AmazonSTaRKDataset, self).__init__(stark_qa_dir, split_dir, human_generated_eval)