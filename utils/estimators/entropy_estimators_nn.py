import torch



class pY_W_evidence(torch.nn.Module):
    def __init__(self, evidence_dim, Y_dim, n_layers = 3, evidence_name = None, sup_type = 'fullsup', dataset = 'shapes3d', hidden_dim = 1000, linear = False):
        super(pY_W_evidence, self).__init__()
        self.evidence_name = evidence_name  
        self.hidden_fc = torch.nn.ModuleList()
        if n_layers <= 1:
            raise ValueError('Number of layers must be greater than 1. If you want to use linear layer, use linear=True instead.')
        if linear:
            self.hidden_fc.append(torch.nn.Linear(evidence_dim, Y_dim))
        else:
            self.hidden_fc.append(torch.nn.Linear(evidence_dim, hidden_dim))
            for i in range(n_layers):
                self.hidden_fc.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.hidden_fc.append(torch.nn.Linear(hidden_dim, Y_dim))
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.sup_type = sup_type
        self.dataset = dataset
        
    def forward(self, x):
        x = x.float()
        for i in range(len(self.hidden_fc)):
            x = self.hidden_fc[i](x)
            if i != len(self.hidden_fc) - 1:
                x = self.relu(x)
        
        
        #x = self.sigmoid(x)
        return x