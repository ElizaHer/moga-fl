from typing import Dict, Any

class ScaffoldState:
    def __init__(self, params_keys):
        self.c_global = {k: 0.0 for k in params_keys}
        self.c_local = {}

    def init_client(self, cid, params_keys):
        self.c_local[cid] = {k: 0.0 for k in params_keys}

    def update_controls(self, cid, grad_dict, lr):
        # Simplified control variates update
        for k in grad_dict:
            self.c_local[cid][k] = self.c_local[cid][k] - lr * grad_dict[k]

