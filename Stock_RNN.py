import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, input_dim, h_dim, n_layers, out_dim, bidirectional, dropout):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, h_dim, n_layers, bidirectional=bidirectional, 
                                                            dropout=dropout, batch_first=True)
        self.linear = nn.Linear(h_dim, out_dim)
        self.n_layers = n_layers
        self.hidden_dim = h_dim
        
    def init_weights(self):
        initrange = 0.5
        self.rnn.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        
        output, _ = self.rnn(x, (h0, c0))
        output = output[:, -1, :]
        output = self.linear(output)
        return output
    
def get_model_0(input_size, output_size, hidden_dim, n_layers, bidirectional, dropout, lr,
                                                                                       scheduler_step, gamma, log_dir):
    
    model = RNN(input_size, hidden_dim, n_layers, output_size, bidirectional, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_step, gamma=gamma)

    writer = SummaryWriter(log_dir)
    
    return model, optimizer, loss_func, scheduler, writer




