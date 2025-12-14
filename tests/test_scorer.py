import numpy as np
from src.scheduling.fairness_ledger import FairnessDebtLedger
from src.scheduling.scorer import ClientScorer

def test_scorer_shapes():
    cfg = {'scheduling': {'weights': {'energy':0.2,'channel':0.2,'data_value':0.2,'fairness_debt':0.2,'bandwidth_cost':-0.1}, 'fairness_ledger': {'debt_increase':0.1,'repay_rate':0.1,'max_debt':1.0}}, 'clients': {'num_clients': 4}}
    ledger = FairnessDebtLedger(cfg, 4)
    scorer = ClientScorer(cfg, 4, ledger)
    s = scorer.score([0.1,0.2,0.3,0.4], [0.5,0.6,0.7,0.8], [1,2,3,4], [0.2,0.1,0.3,0.4])
    assert s.shape[0]==4
