from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

class Client:
    def __init__(self, cid: int, model_fn, train_dataset, indices: List[int], cfg: Dict[str, Any], device):
        self.cid = cid
        self.model_fn = model_fn
        self.train_dataset = train_dataset
        self.indices = indices
        self.cfg = cfg
        self.device = device
        self.model = self.model_fn().to(self.device)

    def local_train(self, global_state: Dict[str, Any], scaffold_state=None):
        self.model.load_state_dict(global_state)
        self.model.train()
        loader = DataLoader(Subset(self.train_dataset, self.indices), batch_size=self.cfg['training'].get('batch_size',64), shuffle=True)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.cfg['training'].get('lr',0.01), momentum=self.cfg['training'].get('momentum',0.9))
        criterion = nn.CrossEntropyLoss()
        mu = self.cfg['training'].get('fedprox_mu', 0.0)
        for _ in range(self.cfg['training'].get('local_epochs',1)):
            for x, y in loader:
                x = x.to(self.device); y = y.to(self.device)
                opt.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                if mu > 0.0:
                    prox = 0.0
                    for p, g in zip(self.model.parameters(), global_state.values()):
                        prox += ((p - torch.tensor(g).to(self.device))**2).sum()
                    loss = loss + (mu/2.0) * prox
                loss.backward()
                opt.step()
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
