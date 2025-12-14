import torch
import torch.nn as nn

class FlipFlopsCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Gate parameters
        self.j_x = nn.Linear(input_size, hidden_size)
        self.j_h = nn.Linear(hidden_size, hidden_size)
        self.k_x = nn.Linear(input_size, hidden_size)
        self.k_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden):
        # Compute gate values
        j = torch.sigmoid(self.j_x(x) + self.j_h(hidden))
        k = torch.sigmoid(self.k_x(x) + self.k_h(hidden))
        
        # Update state
        new_hidden = j * (1 - hidden) + (1 - k) * hidden
        return new_hidden

class FF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            FlipFlopsCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            ) for i in range(num_layers)
        ])

    def forward(self, x, hidden=None):
        # x shape: (seq_len, batch, input_size)
        seq_len, batch_size, _ = x.size()
        
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            input = x[t]
            new_hidden = []
            for layer_idx, cell in enumerate(self.cells):
                h = hidden[layer_idx] if hidden.dim() == 3 else hidden
                h = cell(input, h)
                new_hidden.append(h)
                input = h  # Use output as next layer's input
                
            hidden = torch.stack(new_hidden)
            outputs.append(input)
            
        return torch.stack(outputs), hidden
