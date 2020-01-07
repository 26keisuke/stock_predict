import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

class RNNEmbed(nn.Module):
    def __init__(self, input_dim, h_dim, embed_dim, n_layers, out_dim, bidirectional, dropout):
        super(RNNEmbed, self).__init__()
        self.rnn = nn.LSTM(input_dim, h_dim, n_layers, bidirectional=bidirectional, 
                                                            dropout=dropout, batch_first=True)
        self.linear = nn.Linear(h_dim, out_dim)
        self.n_layers = n_layers
        self.hidden_dim = h_dim
        self.embed = TimeEmbedding(embed_dim)
        
    def init_weights(self):
        initrange = 0.5
        self.rnn.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, inputs):
        h0 = torch.zeros(self.n_layers, inputs.size(0), self.hidden_dim)
        c0 = torch.zeros(self.n_layers, inputs.size(0), self.hidden_dim)

        inputs = self.embed(inputs)
      
        output, _ = self.rnn(inputs, (h0, c0))
        output = output[:, -1, :]
        output = self.linear(output)
        return output

# This class is tightly coupled to the dataset

# Must start with 0 !
year = {2017: 0, 2018: 1}
month = {9: 0, 10: 1, 11: 2, 12: 3, 1: 4, 2: 5}
day_of_week = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
hour = {9: 0, 10: 1, 11:2, 12: 3, 13: 4, 14: 5, 15: 6, 16: 7}

class TimeEmbedding(nn.Module):
    def __init__(self, d_embed):
        super(TimeEmbedding, self).__init__()
        self.d_embed = d_embed
        
        self.year_embed = nn.Embedding(len(year), d_embed)
        self.month_embed = nn.Embedding(len(month), d_embed)
        self.day_of_week_embed = nn.Embedding(len(day_of_week), d_embed)
        self.hour_embed = nn.Embedding(len(hour), d_embed)
        
        ## それか、これらのfeatureをnn.Linearに通してd_modelを小さくするか
        ## concat にするか sum　にするか
        
        self.init_weights()
        
    def init_weights(self):
        init_range = 0.1
        self.year_embed.weight.data.uniform_(-init_range, init_range)
        self.month_embed.weight.data.uniform_(-init_range, init_range)
        self.day_of_week_embed.weight.data.uniform_(-init_range, init_range)
        self.hour_embed.weight.data.uniform_(-init_range, init_range)
        
    def lookup(self, df_feature, table):
        for i, elem in enumerate(df_feature):
            for j, _ in enumerate(elem):
                df_feature[i][j] = table[df_feature[i][j].item()]
        return torch.Tensor(df_feature).long()
        
    def forward(self, inputs): # inputs -> (batch, seq, input_d)

        look_up_year = self.lookup(inputs[:, :, 4], year)
        look_up_month = self.lookup(inputs[:, :, 5], month)
        look_up_day_of_week = self.lookup(inputs[:, :, 7], day_of_week)
        look_up_hour = self.lookup(inputs[:, :, 6], hour)
        
        year_embed = self.month_embed(look_up_year) # (batch, seq, d_model)
        month_embed = self.month_embed(look_up_month)
        day_of_week_embed = self.day_of_week_embed(look_up_day_of_week)
        hour_embed = self.hour_embed(look_up_hour)
            
        if self.d_embed > 1:

            time_embed = torch.cat((year_embed, month_embed, day_of_week_embed, hour_embed), axis=2)
            inputs_embed = torch.cat((inputs[:, :, 0:4], time_embed), axis=2)
            
            return inputs_embed

        return inputs[:, :, 0:4] + year_embed + month_embed + day_of_week_embed + hour_embed
    
    
class DirectionalMSELoss(nn.Module):
    def __init__(self, loss_func, categorical_index=0):
        super(DirectionalMSELoss, self).__init__()
        self.loss_func = loss_func
        self.categorical_index = categorical_index
    
    def forward(self, inputs, label, outputs, scale=1e-4):

        true_dif = inputs[:, -1, :self.categorical_index] - label[:, :] 
        pred_dif = inputs[:, -1, :self.categorical_index] - outputs[:, :]
        dif_rel = torch.abs(true_dif - pred_dif) # "relative" slope difference
        
#         For comparison, this calculates "absolute"
#         epsilon = 1e-6
#         dif_abs = (a + epsilon) * (b + epsilon)  -> absolute slope difference

        dif_sum = torch.sum(dif_rel)
        dif_sum_scaled = dif_sum * scale
        
        value_loss = self.loss_func(outputs.view(-1, 4), label)
        
        return dif_sum_scaled + value_loss
    
    
def get_model_1(input_size, output_size, embed_dim, hidden_dim, n_layers, bidirectional,
                                                    dropout, lr, scheduler_step, gamma, log_dir, categorical_index):
    
    model = RNNEmbed(input_size, hidden_dim, embed_dim, n_layers, output_size, bidirectional, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    total_loss_func = DirectionalMSELoss(mse_loss, categorical_index)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_step, gamma=gamma)

    writer = SummaryWriter(log_dir)
    
    return model, optimizer, total_loss_func, scheduler, writer



        