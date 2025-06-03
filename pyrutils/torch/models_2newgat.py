import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import torch.nn.init as init
import torch.nn.functional as F

#from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv


# ============================================================================= #
class MultiLayerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerGNN, self).__init__()
        
        
        # Define the layers
        self.convs = nn.ModuleList([
            GATConv(input_dim, hidden_dim),
            GATConv(hidden_dim, output_dim)
        ])
        
        self.fuse_conv = TemporalConv()
                
    def forward(self, x, edge_index, valid_step_index):
        
        assert len(x.shape) == 4
        assert len(edge_index.shape) == 4
        
        batch_size, time_steps, num_nodes, _ = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        
        # torch.Size([3, 200, 27, 64])
        # x = self.linear(x)
        
        batch_level_gcn_features = []
        for batch in range(batch_size):
            fused_features_list = []
            for t in range(time_steps):
                x_t = x[batch, t, :, :]
                edge_index_t = edge_index[batch, t, :, :]
        
                for conv in self.convs[:-1]:  # for all but the last layer
                    x_t = conv(x_t, edge_index_t)
                    x_t = F.gelu(x_t)
                    x_t = F.dropout(x_t, p=0.2, training=self.training)
        
                x_t = self.convs[-1](x_t, edge_index_t)  # Apply the last layer
                fused_features_list.append(x_t)
            
            fused_features = torch.stack(fused_features_list, dim=0)
            mask = torch.zeros_like(fused_features)
            mask[:valid_step_index[batch], :, :] = 1
            fused_features = fused_features * mask
            
            # Apply the 1x1 convolution across the timestep dimension to fuse the features
            fused_features = self.fuse_conv(fused_features.unsqueeze(0)).squeeze()

            batch_level_gcn_features.append(fused_features)
        
        gcn_result = torch.stack(batch_level_gcn_features, dim = 0) 
        return gcn_result


class TemporalConv(nn.Module):
    def __init__(self):
        super(TemporalConv, self).__init__()
        
        self.temporal_fusion = None
        self.seq_length = None
    
    def forward(self, x):
        if self.temporal_fusion is None or self.seq_length != x.size(1):
            self.seq_length = x.size(1)
            self.temporal_fusion = nn.Sequential(
                nn.Conv2d(self.seq_length, self.seq_length, kernel_size = 1),
                nn.GELU()).to(x.device)

        return self.temporal_fusion(x) + x


class SpatialSE(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SpatialSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # Initialize the weights of the Linear layers
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        global attention_loss
        attention_loss = 0
        
        b, n, t, c = x.size()
        y = self.avg_pool(x).view(b, n)
        y = self.fc(y).view(b, n, 1, 1)

        res = x * y.expand_as(x)

        return res
