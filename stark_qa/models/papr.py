from typing import Any, Dict, List, Optional, Union
import torch
from stark_qa.models.base import ModelForSTaRKQA
import pandas as pd
import numpy as np
from tqdm import tqdm


class Papr(ModelForSTaRKQA):
    def __init__(self, skb: Any, query_emb_dir: str = '.') -> None:
        """
        Initialize the Papr model.

        Args:
            skb (Any): The knowledge base containing candidate information.
            query_emb_dir (str, optional): Directory where query embeddings are stored. Defaults to '.'.
        """
        super().__init__(skb, query_emb_dir)

    def forward(
        self,
        query: Union[str, List[str]],
        candidates: Optional[List[int]] = None,
        query_id: Optional[Union[int, List[int]]] = None,
        **kwargs: Any
    ) -> Dict[int, float]:
        """
        Compute predictions for the given query using the Papr approach.

        Args:
            query (Union[str, List[str]]): The input query or list of queries.
            candidates (Optional[List[int]], optional): List of candidate IDs to consider. Defaults to None.
            query_id (Optional[Union[int, List[int]]], optional): Query ID or list of query IDs. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[int, float]: A dictionary mapping candidate IDs to predicted scores.
        """
        # Get query embeddings
        query_emb = self.get_query_emb(query, query_id, **kwargs)
        
        # Use all candidates if none specified
        if candidates is None:
            candidates = self.candidate_ids
            
        # Get candidate embeddings and compute scores
        candidate_embs = torch.stack([self.skb.get_embedding(cid) for cid in candidates])
        scores = torch.matmul(query_emb, candidate_embs.T).squeeze()
        
        # Create prediction dictionary
        pred_dict = {cid: float(score) for cid, score in zip(candidates, scores)}
        return pred_dict 

    def evaluate_batch_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        all_results = []
        
        # Collect all unique IDs to initialize evaluator
        all_ids = set()
        for idx, row in df.iterrows():
            retrieved = {int(x) for x in row['retreived_ids'].split(',')}
            # Handle multiple golden answers
            golden = {int(x) for x in row['golden_answer_ids'].split(',')}
            all_ids.update(retrieved)
            all_ids.update(golden)
        
        eval_metrics = [
            "mrr",
            "map",
            "rprecision",
            "recall@5",
            "recall@10",
            "recall@20",
            "recall@50",
            "recall@100",
            "hit@1",
            "hit@3",
            "hit@5",
            "hit@10",
            "hit@20",
            "hit@50",
        ]
        
        results = []
        for idx, row in tqdm(df.iterrows()):
            # Process retrieved IDs
            retrieved_ids = [int(x) for x in row['retreived_ids'].split(',')]
            # Handle multiple golden answers
            golden_answer_ids = [int(x) for x in row['golden_answer_ids'].split(',')]
            
            # Create prediction scores (1/rank as score)
            pred_dict = {id_: 1/(i+1) for i, id_ in enumerate(retrieved_ids)}
            
            # Evaluate
            result = self.evaluate(
                pred_dict=pred_dict,
                answer_ids=torch.tensor(golden_answer_ids),  # Pass all golden answers
                metrics=eval_metrics
            )
            results.append(result)
        
        # Calculate average metrics
        final_metrics = {}
        for metric in eval_metrics:
            final_metrics[metric] = float(np.mean([r[metric] for r in results]))
        
        return final_metrics 