from collections import deque
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyrutils.itertools import negative_range
from pyrutils.torch.distributions import straight_through_gumbel_sigmoid, straight_through_estimator
from pyrutils.torch.general import cat_valid_tensors
from pyrutils.torch.models import build_mlp
from pyrutils.torch.models_2newgat import MultiLayerGNN, SpatialSE
from .tools import *
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GeoVisGNN(nn.Module):
    def __init__(self, input_size: tuple, num_classes: tuple, hidden_size: int = 512, no_human: int = 2,
                 discrete_networks_num_layers: int = 1, discrete_optimization_strategy: str = 'gumbel-sigmoid',
                 filter_discrete_updates: bool = False, no_human_joints: int = 9, max_no_objects: int = 5,
                 message_humans_to_human: bool = False, message_human_to_objects: bool = True,
                 message_objects_to_human: bool = True, message_objects_to_object: bool = True,
                 message_segment: bool = False, message_type: str = 'relational', message_granularity: str = 'specific',
                 message_aggregation: str = 'attention', attention_style: str = 'concat',
                 object_segment_update_strategy: str = 'independent', update_segment_threshold: float = 0.5,
                 add_segment_length: bool = False, add_time_position: bool = False, time_position_strategy: str = 's',
                 positional_encoding_style: str = 'embedding',
                 share_level_mlps: bool = False, bias: bool = True):
        
        super(GeoVisGNN, self).__init__()
        human_input_size, object_input_size = input_size
        num_subactivities, num_affordances = num_classes
        self.no_human = no_human
        self.no_human_joints = no_human_joints
        self.max_no_objects = max_no_objects
        self.discrete_optimization_strategy = discrete_optimization_strategy
        self.filter_discrete_updates = filter_discrete_updates
        self.message_humans_to_human = message_humans_to_human
        self.message_human_to_objects = message_human_to_objects
        self.message_objects_to_human = message_objects_to_human
        self.message_objects_to_object = message_objects_to_object
        self.message_segment = message_segment
        self.message_type = message_type
        self.message_granularity = message_granularity
        self.message_aggregation = message_aggregation
        self.attention_style = attention_style
        self.object_segment_update_strategy = object_segment_update_strategy
        self.update_segment_threshold = update_segment_threshold
        self.add_segment_length = add_segment_length
        self.add_time_position = add_time_position
        self.time_position_strategy = time_position_strategy
        self.positional_encoding_style = positional_encoding_style
        
        # Initialize Interdependent Entity Graph (IEG) module
        self.spatial_GCN = SenderAggregation(2*hidden_size, 2*hidden_size)
        self.spatial_seg_GCN = SenderAggregation(hidden_size, hidden_size)
        
        # Initialize attention-based fusion module 
        self.SSE_ho = SpatialSE(2*self.no_human+2*self.max_no_objects)


        # Geometry
        self.dinamics_dim = hidden_size//4 #128
        self.masked_GCN = MultiLayerGNN(int(4), int(self.dinamics_dim//2), int(self.dinamics_dim))
        
        # Human
        self.human_visual_embedding_mlp = build_mlp([2048, hidden_size, hidden_size//2], ['relu', 'relu'], bias=bias)
        self.human_geometric_embedding_mlp = build_mlp([no_human_joints*self.dinamics_dim, hidden_size, hidden_size//2], ['relu', 'relu'], bias=bias)
        self.human_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                   bidirectional=True)
        self.human_bd_embedding_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
        human_segment_input_size = hidden_size
        if message_humans_to_human:
            human_segment_input_size += hidden_size
            if message_segment:
                human_segment_input_size += hidden_size
        if message_objects_to_human:
            human_segment_input_size += hidden_size
            if message_segment:
                human_segment_input_size += hidden_size
        if add_time_position and time_position_strategy == 's':
            human_segment_input_size += hidden_size
        if add_segment_length:
            human_segment_input_size += hidden_size
        self.human_segment_rnn_fcell = nn.GRUCell(human_segment_input_size, hidden_size, bias=bias)
        self.human_segment_rnn_bcell = nn.GRUCell(human_segment_input_size, hidden_size, bias=bias)
        # Object
        self.object_visual_embedding_mlp = build_mlp([object_input_size, hidden_size, hidden_size//2], ['relu', 'relu'], bias=bias)
        self.object_geometric_embedding_mlp = build_mlp([2*self.dinamics_dim, hidden_size, hidden_size//2], ['relu', 'relu'], bias=bias)
        self.object_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                    bidirectional=True)
        self.object_bd_embedding_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
        object_segment_input_size = hidden_size
        if message_human_to_objects:
            object_segment_input_size += hidden_size
            if message_segment:
                object_segment_input_size += hidden_size
        if message_objects_to_object:
            object_segment_input_size += hidden_size
            if message_segment:
                object_segment_input_size += hidden_size
        if add_time_position and time_position_strategy == 's':
            object_segment_input_size += hidden_size
        if add_segment_length:
            object_segment_input_size += hidden_size
        self.object_segment_rnn_fcell = nn.GRUCell(object_segment_input_size, hidden_size, bias=bias)
        self.object_segment_rnn_bcell = nn.GRUCell(object_segment_input_size, hidden_size, bias=bias)
        # Messages
        # h2h features fusion
        if message_humans_to_human:
            self.humans_to_human_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
            if message_segment:
                self.humans_to_human_segment_message_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                        bias=bias)
            
                self.humans_to_human_message_att_mlp = build_mlp([4 * hidden_size, 1], ['relu'], bias=bias)
                if message_segment:
                    self.humans_to_human_segment_message_att_mlp = build_mlp([2 * hidden_size, 1], ['relu'],
                                                                                bias=bias)
        # Human(s) to Object
        if message_human_to_objects:
            self.human_to_object_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
            if message_segment:
                self.human_to_object_segment_message_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                        bias=bias)
            
            self.humans_to_object_message_att_mlp = build_mlp([4 * hidden_size, 1], ['relu'], bias=bias)
            if message_segment:
                self.humans_to_object_segment_message_att_mlp = build_mlp([2 * hidden_size, 1], ['relu'],
                                                                            bias=bias)
        # Objects to Human
        if message_objects_to_human:
            self.objects_to_human_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
            if message_segment:
                self.objects_to_human_segment_message_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                        bias=bias)
            
            self.objects_to_human_message_att_mlp = build_mlp([4 * hidden_size, 1], ['relu'], bias=bias)
            if message_segment:
                self.objects_to_human_segment_message_att_mlp = build_mlp([2 * hidden_size, 1], ['relu'],
                                                                            bias=bias)
        # Objects to Object
        if message_objects_to_object:           
            self.objects_to_object_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
            if message_segment:
                self.objects_to_object_segment_message_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                        bias=bias)
                        
            self.objects_to_object_message_att_mlp = build_mlp([4 * hidden_size, 1], ['relu'], bias=bias)
            if message_segment:
                self.objects_to_object_segment_message_att_mlp = build_mlp([2 * hidden_size, 1], ['relu'],
                                                                            bias=bias)
        # Discrete MLPs
        update_human_segment_input_size = 2 * hidden_size
        if message_humans_to_human:
            update_human_segment_input_size += hidden_size
        if message_objects_to_human:
            update_human_segment_input_size += hidden_size
        if add_time_position and time_position_strategy == 'u':
            update_human_segment_input_size += hidden_size
        num_discrete_hidden_layers = discrete_networks_num_layers - 1
        dims = [update_human_segment_input_size] + [hidden_size] * num_discrete_hidden_layers + [1]
        activations = ['relu'] * num_discrete_hidden_layers + ['sigmoid']
        self.update_human_segment_mlp = build_mlp(dims, activations, bias=bias)
        if object_segment_update_strategy not in {'same_as_human', 'sah'}:
            update_object_segment_input_size = 2 * hidden_size
            if message_human_to_objects:
                update_object_segment_input_size += hidden_size
            if message_objects_to_object:
                update_object_segment_input_size += hidden_size
            if add_time_position and time_position_strategy == 'u':
                update_object_segment_input_size += hidden_size
            dims = [update_object_segment_input_size] + [hidden_size] * num_discrete_hidden_layers + [1]
            self.update_object_segment_mlp = build_mlp(dims, activations, bias=bias)
        # Recognition/Prediction MLPs
        label_mlps_input_size = 2 * hidden_size
        
        
        self.human_recognition_mlp = build_mlp([label_mlps_input_size, num_subactivities],
                                               [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
        self.human_prediction_mlp = build_mlp([label_mlps_input_size, num_subactivities],
                                              [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
        if num_affordances is not None:
            self.object_recognition_mlp = build_mlp([label_mlps_input_size, num_affordances],
                                                    [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
            self.object_prediction_mlp = build_mlp([label_mlps_input_size, num_affordances],
                                                   [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
        
        self.human_frame_recognition_mlp = build_mlp([2 * hidden_size, num_subactivities],
                                                        [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
        self.human_frame_prediction_mlp = build_mlp([2 * hidden_size, num_subactivities],
                                                    [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
        if num_affordances is not None:
            self.object_frame_recognition_mlp = build_mlp([2 * hidden_size, num_affordances],
                                                            [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
            self.object_frame_prediction_mlp = build_mlp([2 * hidden_size, num_affordances],
                                                            [{'name': 'logsoftmax', 'dim': -1}], bias=bias)

    def get_valid_frame(self, human_joints):
        batch_size = human_joints.shape[0]
        #human_joints_features = rearrange(human_joints, 'b t n j d -> b t (n j d)')
        valid_mask = torch.zeros_like(human_joints)
        valid_index_list = []
        for i in range(batch_size):
            valid_index = human_joints[i].sum(dim = 1).nonzero().squeeze().tolist()
            valid_mask[i, valid_index] = 1
            valid_index_list.append(valid_mask[i].nonzero()[-1].to(torch.int32)[0])
        return valid_mask, valid_index_list

    def get_graph(self, human_geometry_feature):
        assert len(human_geometry_feature.shape) == 4
        batchs, T, num_node = human_geometry_feature.shape[0], human_geometry_feature.shape[1], human_geometry_feature.shape[2]
        
        batch_wise_edges = []
        for batch in range(batchs):
            time_wise_edges = []
            for t in range(T):
                # Constructing edge_index for a fully connected graph
                edge_index_list = [(i, j) for i in range(num_node) for j in range(num_node) if i != j]
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
                time_wise_edges.append(edge_index)
            time_wise_edges_stack = torch.stack(time_wise_edges, dim=0)
            batch_wise_edges.append(time_wise_edges_stack)
        
        batch_wise_edges_stack = torch.stack(batch_wise_edges)
        return batch_wise_edges_stack.to(device)
    
    
    def forward(self, x_human, x_objects, objects_mask, human_segmentation=None, objects_segmentation=None,
                human_human_distances=None, human_object_distances=None, object_object_distances=None,
                steps_per_example=None, inspect_model=False):


        num_humans, num_objects = x_human.size(2), x_objects.size(2) # Nh, No
        xx_hs, xx_os = [[] for _ in range(num_humans)], [[] for _ in range(num_objects)]
        ux_hs, ux_os = [[] for _ in range(num_humans)], [[] for _ in range(num_objects)]  # Hard
        ux_hss, ux_oss = [[] for _ in range(num_humans)], [[] for _ in range(num_objects)]  # Soft
        ax_hf = [[] for _ in range(num_humans)]
        
        # Local Graph
        if x_human.shape[3] == 2124:                  
            x_human, x_geometry = torch.split(x_human, [2048, 76], dim=-1)
            x_geometry = x_geometry.squeeze(2)
        elif x_human.shape[3] == 2168:                 
            x_human, x_geometry = torch.split(x_human, [2048, 120], dim=-1)
            x_geometry = x_geometry[:, :, 0, :]
        elif x_human.shape[3] == 2196:              
            x_human, x_geometry = torch.split(x_human, [2048, 148], dim=-1)
            x_geometry = x_geometry[:, :, 0, :]
        else:                                        
            x_human, x_geometry = torch.split(x_human, [2048, 104], dim=-1)
            x_geometry = x_geometry[:, :, 0, :]
        
        # visual features
        human_visual_features, object_visual_features = self.human_visual_embedding_mlp(x_human), self.object_visual_embedding_mlp(x_objects) # (B, T, Nh/No, 256)
        # geometric features
        human_size = num_humans*self.no_human_joints * 4
        object_size = num_objects * 8
        assert num_objects <= self.max_no_objects, 'num_objects must smaller than max_no_objects.'
        if num_objects == self.max_no_objects:
            human_geometry, object_geometry = torch.split(x_geometry, [human_size, object_size], dim=-1)
        else:
            remaining_size = x_geometry.size(-1) - human_size - object_size
            human_geometry, object_geometry, _ = torch.split(x_geometry, [human_size, object_size, remaining_size], dim=-1)
        
        valid_frame_human, valid_index_list_human = self.get_valid_frame(human_geometry)
        valid_frame_object, valid_index_list_object = self.get_valid_frame(object_geometry)
        
        # GAT on human geometry        
        bs, t, hw = human_geometry.size()
        human_geometry = human_geometry.view(bs, t, hw//4, 4)
        human_graph_edges = self.get_graph(human_geometry)
        human_geometric_features = self.masked_GCN(human_geometry, human_graph_edges, valid_index_list_human) # (B, T, Nh*J, 128)
        human_geometric_features = human_geometric_features.view(bs, t, num_humans, self.no_human_joints *self.dinamics_dim) # (B, T, Nh, J*128)
        human_geometric_features = self.human_geometric_embedding_mlp(human_geometric_features) # (B, T, Nh, 256)
        human_vg = torch.cat([human_visual_features, human_geometric_features], dim=2) # (B, T, 2*Nh, 256)
        
        # GAT on object geometry
        bs, t, ow = object_geometry.size()
        object_geometry = object_geometry.view(bs, t, ow//4, 4)
        object_graph_edges = self.get_graph(object_geometry)
        object_geometric_features = self.masked_GCN(object_geometry, object_graph_edges, valid_index_list_object) # (B, T, No*K, 128)
        object_geometric_features = object_geometric_features.view(bs, t, num_objects, 2*self.dinamics_dim) # (B, T, No, K*128)
        object_geometric_features = self.object_geometric_embedding_mlp(object_geometric_features) # (B, T, No, 256)
        
        ovg = torch.cat([object_visual_features, object_geometric_features], dim=-1) # (B, T, No, 512)
        padding_needed = self.max_no_objects - ovg.size(2)
        object_padded_tensor = F.pad(ovg, (0, 0, 0, padding_needed, 0, 0, 0, 0)) # (B, T, max_No, 512)
        object_padded_tensor = object_padded_tensor.view(bs, t, 2*self.max_no_objects, 2*self.dinamics_dim) # (B, T, 2*max_No, 256)
        ho_vg = torch.cat([human_vg, object_padded_tensor], dim=2) # (B, T, 2*(Nh+max_No), 256)
        
        # Channel Attention
        ho_vg = ho_vg.permute(0, 2, 1, 3).contiguous() #bntc
        ho_vg = self.SSE_ho(ho_vg)
        ho_vg = ho_vg.permute(0, 2, 1, 3).contiguous() #btnc
        hvg, ovg = torch.split(ho_vg, [2*num_humans, 2*self.max_no_objects], dim=2)
        x_human = hvg.view(bs, t, num_humans, 4*self.dinamics_dim) # (B, T, Nh, 512)
        x_objects = ovg[:,:,:2*num_objects,:]
        x_objects = x_objects.view(bs, t, num_objects, 4*self.dinamics_dim) # (B, T, No, 512)
        
        # Frame temporal modelling
        h_hf, h_hfr = self.frame_temporal_modelling(x_human, self.human_bd_rnn, self.human_bd_embedding_mlp)
        h_of, h_ofr = self.frame_temporal_modelling(x_objects, self.object_bd_rnn, self.object_bd_embedding_mlp)

        # num_steps = x_human.size(1)
        num_steps=10
        
        x_time = None
        
        # Frame feature aggregation
        for t in range(num_steps):
            try:
                x_tt = x_time[:, t]
            except TypeError:
                x_tt = None
            # Human(s)
            for h in range(num_humans):
                x_hfth = x_human[:, t, h]
                h_hfth = h_hf[:, t, h]
                m_hhth = None
                if self.message_humans_to_human:
                    try:
                        hh_dists = human_human_distances[:, t]
                    except TypeError:
                        hh_dists = None
                    m_hhth = self.h2h_features_fusion(x_human[:, t], h_hf[:, t], h)
                m_ohth = None
                if self.message_objects_to_human:
                    try:
                        ho_dists = human_object_distances[:, t, h]
                    except TypeError:
                        ho_dists = None
                    m_ohth, o2h_faw = self.o2h_features_fusion(x_hfth, x_objects[:, t], h_hfth, h_of[:, t],
                                                                     objects_mask, ho_dists=ho_dists)
                    ax_hf[h].append(o2h_faw)
                try:
                    u_hsth = u_hsths = human_segmentation[:, t:t + 1, h]
                except TypeError:
                    u_hsth, u_hsths = self.hum_segment_modelling(x_hfth, h_hfth, m_hhth, m_ohth, x_tt)
                    if t == (num_steps - 1):
                        u_hsth[:] = 1.0
                ux_hs[h].append(u_hsth)
                ux_hss[h].append(u_hsths)
                x_hsth = cat_valid_tensors([h_hfth, m_hhth, m_ohth], dim=-1)
                xx_hs[h].append(x_hsth)
            # Objects
            for k in range(num_objects):
                x_oftk = x_objects[:, t, k]
                h_oftk = h_of[:, t, k]
                m_hotk = None
                if self.message_human_to_objects:
                    try:
                        oh_dists = human_object_distances[:, t, :, k]
                    except TypeError:
                        oh_dists = None
                    m_hotk = self.h2o_features_fusion(x_human[:, t], x_oftk, h_hf[:, t], h_oftk,
                                                            oh_dists=oh_dists)
                    m_hotk = m_hotk * objects_mask[:, k:k + 1]
                m_ootk = None
                if self.message_objects_to_object:
                    try:
                        oo_dists = object_object_distances[:, t]
                    except TypeError:
                        oo_dists = None
                    m_ootk = self.o2o_features_fusion(x_objects[:, t], h_of[:, t], k, objects_mask,
                                                             oo_dists=oo_dists)
                try:
                    u_ostk = u_ostks = objects_segmentation[:, t:t + 1, k]
                except TypeError:
                    u_hst = ux_hs[0][-1] if len(ux_hs) == 1 else None
                    u_hsts = ux_hss[0][-1] if len(ux_hss) == 1 else None
                    u_ostk, u_ostks = self.obj_segment_modelling(x_oftk, h_oftk, m_hotk, m_ootk, u_hst, u_hsts, x_tt)
                    if t == (num_steps - 1):
                        u_ostk[:] = 1.0
                ux_os[k].append(u_ostk)
                ux_oss[k].append(u_ostks)
                x_ostk = cat_valid_tensors([h_oftk, m_hotk, m_ootk], dim=-1)
                xx_os[k].append(x_ostk)
        
        # Block feature aggregation
        batch_size, hidden_size, dtype, device = x_human.size(0), x_human.size(-1), x_human.dtype, x_human.device
        hx_hsf, hx_hsb = [[] for _ in range(num_humans)], [deque() for _ in range(num_humans)]
        hx_osf, hx_osb = [[] for _ in range(num_objects)], [deque() for _ in range(num_objects)]
        ax_hsf, ax_hsb = [[] for _ in range(num_humans)], [deque() for _ in range(num_humans)]
        for tf, tb in zip(range(num_steps), negative_range(num_steps)):
            # Human(s)
            hx_hsf_cache, hx_hsb_cache = [], []
            for h in range(num_humans):
                x_hsthf = xx_hs[h][tf]
                if self.message_segment:
                    if self.message_humans_to_human:
                        try:
                            hh_dists = human_human_distances[:, tf]
                        except TypeError:
                            hh_dists = None
                        mg_hhthf = self.h2h_SegFeatures_fusion(hx_hsf, h, batch_size, dtype, device,
                                                                         direction='forward', hh_dists=hh_dists)
                        x_hsthf = torch.cat([x_hsthf, mg_hhthf], dim=-1)
                    if self.message_objects_to_human:
                        try:
                            ho_dists = human_object_distances[:, tf, h]
                        except TypeError:
                            ho_dists = None
                        mg_ohthf, o2h_sfaw = self.o2h_SegFeatures_fusion(hx_hsf[h], hx_osf, objects_mask,
                                                                                    'forward', ho_dists=ho_dists)
                        ax_hsf[h].append(o2h_sfaw)
                        x_hsthf = torch.cat([x_hsthf, mg_ohthf], dim=-1)
                h_hsthf = self.biGRU_message_passing(x_hsthf, ux_hs[h][tf], hx_hsf[h], 'human', 'forward')
                hx_hsf_cache.append(h_hsthf)
                x_hsthb = xx_hs[h][tb]
                if self.message_segment:
                    if self.message_humans_to_human:
                        try:
                            hh_dists = human_human_distances[:, tb]
                        except TypeError:
                            hh_dists = None
                        mg_hhthb = self.h2h_SegFeatures_fusion(hx_hsb, h, batch_size, dtype, device,
                                                                         direction='backward', hh_dists=hh_dists)
                        x_hsthb = torch.cat([x_hsthb, mg_hhthb], dim=-1)
                    if self.message_objects_to_human:
                        try:
                            ho_dists = human_object_distances[:, tb, h]
                        except TypeError:
                            ho_dists = None
                        mg_ohthb, o2h_sbaw = self.o2h_SegFeatures_fusion(hx_hsb[h], hx_osb, objects_mask,
                                                                                    'backward', ho_dists=ho_dists)
                        ax_hsb[h].appendleft(o2h_sbaw)
                        x_hsthb = torch.cat([x_hsthb, mg_ohthb], dim=-1)
                h_hsthb = self.biGRU_message_passing(x_hsthb, ux_hs[h][tb], hx_hsb[h], 'human', 'backward')
                hx_hsb_cache.append(h_hsthb)
            # Objects
            hx_osf_cache, hx_osb_cache = [], []
            for k in range(num_objects):
                x_ostkf = xx_os[k][tf]
                if self.message_segment:
                    if self.message_human_to_objects:
                        try:
                            oh_dists = human_object_distances[:, tf, :, k]
                        except TypeError:
                            oh_dists = None
                        mg_hotf = self.h2o_SegFeatures_fusion(hx_osf[k], hx_hsf, batch_size, dtype, device,
                                                                         direction='forward', oh_dists=oh_dists)
                        x_ostkf = torch.cat([x_ostkf, mg_hotf], dim=-1)
                    if self.message_objects_to_object:
                        try:
                            oo_dists = object_object_distances[:, tf]
                        except TypeError:
                            oo_dists = None
                        mg_ootf = self.o2o_SegFeatures_fusion(hx_osf, objects_mask, k, 'forward',
                                                                          oo_dists=oo_dists)
                        x_ostkf = torch.cat([x_ostkf, mg_ootf], dim=-1)
                h_ostkf = self.biGRU_message_passing(x_ostkf, ux_os[k][tf], hx_osf[k], 'object', 'forward')
                hx_osf_cache.append(h_ostkf)
                x_ostkb = xx_os[k][tb]
                if self.message_segment:
                    if self.message_human_to_objects:
                        try:
                            oh_dists = human_object_distances[:, tb, :, k]
                        except TypeError:
                            oh_dists = None
                        mg_hotb = self.h2o_SegFeatures_fusion(hx_osb[k], hx_hsb, batch_size, dtype, device,
                                                                         direction='backward', oh_dists=oh_dists)
                        x_ostkb = torch.cat([x_ostkb, mg_hotb], dim=-1)
                    if self.message_objects_to_object:
                        try:
                            oo_dists = object_object_distances[:, tb]
                        except TypeError:
                            oo_dists = None
                        mg_ootb = self.o2o_SegFeatures_fusion(hx_osb, objects_mask, k, 'backward',
                                                                          oo_dists=oo_dists)
                        x_ostkb = torch.cat([x_ostkb, mg_ootb], dim=-1)
                h_ostkb = self.biGRU_message_passing(x_ostkb, ux_os[k][tb], hx_osb[k], 'object', 'backward')
                hx_osb_cache.append(h_ostkb)
            # Commit updates
            for h, (h_hsf, h_hsb) in enumerate(zip(hx_hsf_cache, hx_hsb_cache)):
                hx_hsf[h].append(h_hsf)
                hx_hsb[h].appendleft(h_hsb)
            for k, (h_osf, h_osb) in enumerate(zip(hx_osf_cache, hx_osb_cache)):
                hx_osf[k].append(h_osf)
                hx_osb[k].appendleft(h_osb)
        hx_hs = [[torch.cat([h_hsthf, h_hsthb], dim=-1) for h_hsthf, h_hsthb in zip(hx_hshf, hx_hshb)]
                 for hx_hshf, hx_hshb in zip(hx_hsf, hx_hsb)]
        hx_os = [[torch.cat([h_ostkf, h_ostkb], dim=-1) for h_ostkf, h_ostkb in zip(hx_oskf, hx_oskb)]
                 for hx_oskf, hx_oskb in zip(hx_osf, hx_osb)]
        # Fix hidden states of human(s) and objects
        partial_hx_hs = []
        for h in range(num_humans):
            hx_hsh = torch.stack(hx_hs[h], dim=1)
            ux_hsh = torch.cat(ux_hs[h], dim=-1)
            hx_hsh = hidden_transformation(hx_hsh, ux_hsh.detach())
            partial_hx_hs.append(hx_hsh)
        hx_hs = torch.stack(partial_hx_hs, dim=2)
        partial_hx_os = []
        for k in range(num_objects):
            hx_osk = torch.stack(hx_os[k], dim=1)
            ux_osk = torch.cat(ux_os[k], dim=-1)
            hx_osk = hidden_transformation(hx_osk, ux_osk.detach())
            partial_hx_os.append(hx_osk)
        hx_os = torch.stack(partial_hx_os, dim=2)
        
        # Predictions
        y_hs = torch.stack([torch.cat(ux_hsh, dim=-1) for ux_hsh in ux_hs], dim=-1)
        y_os = torch.stack([torch.cat(ux_osk, dim=-1) for ux_osk in ux_os], dim=-1)
        y_hss = torch.stack([torch.cat(ux_hssh, dim=-1) for ux_hssh in ux_hss], dim=-1)
        y_oss = torch.stack([torch.cat(ux_ossk, dim=-1) for ux_ossk in ux_oss], dim=-1)
        y_human_frame_recognition = self.human_frame_recognition_mlp(h_hfr).permute(0, 3, 1, 2).contiguous()
        y_human_frame_prediction = self.human_frame_prediction_mlp(h_hfr).permute(0, 3, 1, 2).contiguous()
        y_human_recognition = self.human_recognition_mlp(hx_hs).permute(0, 3, 1, 2).contiguous()
        y_human_prediction = self.human_prediction_mlp(hx_hs).permute(0, 3, 1, 2).contiguous()
        try:
            y_object_frame_recognition = self.object_frame_recognition_mlp(h_ofr).permute(0, 3, 1, 2).contiguous()
            y_object_frame_prediction = self.object_frame_prediction_mlp(h_ofr).permute(0, 3, 1, 2).contiguous()
            y_object_recognition = self.object_recognition_mlp(hx_os).permute(0, 3, 1, 2).contiguous()
            y_object_prediction = self.object_prediction_mlp(hx_os).permute(0, 3, 1, 2).contiguous()
        except AttributeError:
            output = [y_hs, y_hss,
                      y_human_frame_recognition, y_human_frame_prediction,
                      y_human_recognition, y_human_prediction]
        else:
            output = [y_hs, y_os, y_hss, y_oss,
                      y_human_frame_recognition, y_human_frame_prediction,
                      y_object_frame_recognition, y_object_frame_prediction,
                      y_human_recognition, y_human_prediction, y_object_recognition, y_object_prediction]
        if inspect_model:
            ax_hf = torch.stack([torch.stack(ax_hfh, dim=1) for ax_hfh in ax_hf], dim=1)
            ax_hsf = torch.stack([torch.stack(ax_hsfh, dim=1) for ax_hsfh in ax_hsf], dim=1)
            ax_hsb = torch.stack([torch.stack(list(ax_hsbh), dim=1) for ax_hsbh in ax_hsb], dim=1)
            attention_scores = [ax_hf, ax_hsf, ax_hsb]
            return output, attention_scores
        return output


    @staticmethod
    def frame_temporal_modelling(x, rnn, embedding_mlp):
        h_f, num_entities = [], x.size(2)
        for e in range(num_entities):
            h_fe, _ = rnn(x[:, :, e])
            h_f.append(h_fe)
        h_fr = torch.stack(h_f, dim=2)
        h_f = embedding_mlp(h_fr)
        return h_f, h_fr

    def h2h_features_fusion(self, x_hft, h_hft, h):
    
        x_hfth = x_hft[:, h]
        h_hfth = h_hft[:, h]
        receiver = torch.cat([x_hfth, h_hfth], dim=-1)
        x_hft = torch.cat([x_hft[:, :h], x_hft[:, h + 1:]], dim=1)
        h_hft = torch.cat([h_hft[:, :h], h_hft[:, h + 1:]], dim=1)
        senders = torch.cat([x_hft, h_hft], dim=-1)
        senders_mask = torch.ones(senders.size()[:2], dtype=senders.dtype, device=senders.device)
        senders = self.spatial_GCN(senders)
        m_hhth = messages_passing(receiver, senders, senders_mask,
                                                granularity=self.message_granularity,
                                                message_fn=self.humans_to_human_message_mlp)
        att_weights = entity_wise_attention(receiver, senders, senders_mask,
                                                attention_style=self.attention_style,
                                                attention_fn=self.humans_to_human_message_att_mlp)
        
        att_weights = torch.unsqueeze(att_weights, dim=-1)
        m_hhth = torch.sum(att_weights * m_hhth, dim=1)
        return m_hhth

    def h2h_SegFeatures_fusion(self, hx_hs, h, batch_size, dtype, device, direction: str, hh_dists=None):
  
        hidden_size = self.human_segment_rnn_fcell.hidden_size
        grab_state_fn = end_state_handler if direction == 'forward' else initial_state_handler
        receiver = grab_state_fn(hx_hs[h], batch_size, hidden_size, dtype, device, detach=False)
        senders = [grab_state_fn(hx_hsm, batch_size, hidden_size, dtype, device, detach=False)
                   for hx_hsm in hx_hs[:h] + hx_hs[h + 1:]]
        senders = torch.stack(senders, dim=1)
        senders = self.spatial_seg_GCN(senders)
        senders_mask = torch.ones(senders.size()[:2], dtype=senders.dtype, device=senders.device)
        
        mg_hhth = messages_passing(receiver, senders, senders_mask,
                                                    granularity=self.message_granularity,
                                                    message_fn=self.humans_to_human_segment_message_mlp)
            
        att_weights = entity_wise_attention(receiver, senders, senders_mask,
                                                attention_style=self.attention_style,
                                                attention_fn=self.humans_to_human_segment_message_att_mlp)
                
        att_weights = torch.unsqueeze(att_weights, dim=-1)
        mg_hhth = torch.sum(att_weights * mg_hhth, dim=1)
        return mg_hhth

    def h2o_features_fusion(self, x_hft, x_oftk, h_hft, h_oftk, oh_dists=None):
    
        receiver = torch.cat([x_oftk, h_oftk], dim=-1)
        senders = torch.cat([x_hft, h_hft], dim=-1)
        senders_mask = torch.ones(senders.size()[:2], dtype=senders.dtype, device=senders.device)       
        senders = self.spatial_GCN(senders)
        
        hok_message = messages_passing(receiver, senders, senders_mask,
                                                        granularity=self.message_granularity,
                                                        message_fn=self.human_to_object_message_mlp)
        att_weights = entity_wise_attention(receiver, senders, senders_mask,
                                                attention_style=self.attention_style,
                                                attention_fn=self.humans_to_object_message_att_mlp)
        att_weights = torch.unsqueeze(att_weights, dim=-1)
        hok_message = torch.sum(att_weights * hok_message, dim=1)
        return hok_message

    def h2o_SegFeatures_fusion(self, hx_hok, hx_hs, batch_size, dtype, device, direction: str,
                                          oh_dists=None):
        
        hidden_size = self.human_segment_rnn_fcell.hidden_size
        grab_state_fn = end_state_handler if direction == 'forward' else initial_state_handler
        receiver = grab_state_fn(hx_hok, batch_size, hidden_size, dtype, device, detach=False)
        senders = [grab_state_fn(hx_hsh, batch_size, hidden_size, dtype, device, detach=False) for hx_hsh in hx_hs]
        senders = torch.stack(senders, dim=1)
        senders = self.spatial_seg_GCN(senders)
        senders_mask = torch.ones(senders.size()[:2], dtype=senders.dtype, device=senders.device)
        
        mg_hot = messages_passing(receiver, senders, senders_mask,
                                                granularity=self.message_granularity,
                                                message_fn=self.human_to_object_segment_message_mlp)
            
        att_weights = entity_wise_attention(receiver, senders, senders_mask,
                                                attention_style=self.attention_style,
                                                attention_fn=self.humans_to_object_segment_message_att_mlp)
                
        att_weights = torch.unsqueeze(att_weights, dim=-1)
        mg_hot = torch.sum(att_weights * mg_hot, dim=1)
        return mg_hot

    def o2h_features_fusion(self, x_hfth, x_oft, h_hfth, h_oft, objects_mask, ho_dists=None):
        
        receiver = torch.cat([x_hfth, h_hfth], dim=-1)
        senders = torch.cat([x_oft, h_oft], dim=-1)
        senders_mask = objects_mask
        weights = None
        senders = self.spatial_GCN(senders)
        m_oht = messages_passing(receiver, senders, senders_mask,
                                                granularity=self.message_granularity,
                                                message_fn=self.objects_to_human_message_mlp)
            
        att_weights = entity_wise_attention(receiver, senders, senders_mask,
                                                attention_style=self.attention_style,
                                                attention_fn=self.objects_to_human_message_att_mlp)
                
        weights = att_weights
        att_weights = torch.unsqueeze(att_weights, dim=-1)
        m_oht = torch.sum(att_weights * m_oht, dim=1)
        return m_oht, weights

    def o2h_SegFeatures_fusion(self, hx_hsh, hx_os, objects_mask, direction: str, ho_dists=None):
        
        hidden_size = self.human_segment_rnn_fcell.hidden_size
        batch_size, dtype, device = objects_mask.size(0), objects_mask.dtype, objects_mask.device
        grab_state_fn = end_state_handler if direction == 'forward' else initial_state_handler
        receiver = grab_state_fn(hx_hsh, batch_size, hidden_size, dtype, device, detach=False)
        senders = [grab_state_fn(hx_osk, batch_size, hidden_size, dtype, device, detach=False) for hx_osk in hx_os]
        senders = torch.stack(senders, dim=1)
        senders = self.spatial_seg_GCN(senders)
        senders_mask = objects_mask
        weights = None
        
        mg_oht = messages_passing(receiver, senders, senders_mask,
                                                granularity=self.message_granularity,
                                                message_fn=self.objects_to_human_segment_message_mlp)
            
        att_weights = entity_wise_attention(receiver, senders, senders_mask,
                                                attention_style=self.attention_style,
                                                attention_fn=self.objects_to_human_segment_message_att_mlp)
                
        weights = att_weights
        att_weights = torch.unsqueeze(att_weights, dim=-1)
        mg_oht = torch.sum(att_weights * mg_oht, dim=1)
        return mg_oht, weights

    def o2o_features_fusion(self, x_oft, h_oft, k, objects_mask, oo_dists=None):
        
        x_oftk = x_oft[:, k]
        h_oftk = h_oft[:, k]
        receiver = torch.cat([x_oftk, h_oftk], dim=-1)
        x_oft = torch.cat([x_oft[:, :k], x_oft[:, k + 1:]], dim=1)
        h_oft = torch.cat([h_oft[:, :k], h_oft[:, k + 1:]], dim=1)
        senders = torch.cat([x_oft, h_oft], dim=-1)
        senders_mask = torch.cat([objects_mask[:, :k], objects_mask[:, k + 1:]], dim=1)
        senders = self.spatial_GCN(senders)
        m_ootk = messages_passing(receiver, senders, senders_mask,
                                                granularity=self.message_granularity,
                                                message_fn=self.objects_to_object_message_mlp)
            
        att_weights = entity_wise_attention(receiver, senders, senders_mask,
                                                attention_style=self.attention_style,
                                                attention_fn=self.objects_to_object_message_att_mlp)
                
        att_weights = torch.unsqueeze(att_weights, dim=-1)
        m_ootk = torch.sum(att_weights * m_ootk, dim=1)
        return m_ootk

    def o2o_SegFeatures_fusion(self, hx_os, objects_mask, k, direction: str, oo_dists=None):
        
        hidden_size = self.human_segment_rnn_fcell.hidden_size
        batch_size, dtype, device = objects_mask.size(0), objects_mask.dtype, objects_mask.device
        grab_state_fn = end_state_handler if direction == 'forward' else initial_state_handler
        receiver = grab_state_fn(hx_os[k], batch_size, hidden_size, dtype, device, detach=False)
        senders = [grab_state_fn(hx_osm, batch_size, hidden_size, dtype, device, detach=False)
                   for hx_osm in hx_os[:k] + hx_os[k + 1:]]
        senders = torch.stack(senders, dim=1)
        senders = self.spatial_seg_GCN(senders)
        senders_mask = torch.cat([objects_mask[:, :k], objects_mask[:, k + 1:]], dim=1)
        
        mg_oot = messages_passing(receiver, senders, senders_mask,
                                                granularity=self.message_granularity,
                                                message_fn=self.objects_to_object_segment_message_mlp)
            
        att_weights = entity_wise_attention(receiver, senders, senders_mask,
                                                attention_style=self.attention_style,
                                                attention_fn=self.objects_to_object_segment_message_att_mlp)
                
        att_weights = torch.unsqueeze(att_weights, dim=-1)
        mg_oot = torch.sum(att_weights * mg_oot, dim=1)
        return mg_oot

    def hum_segment_modelling(self, x_hfth, h_hfth, m_hhth, m_ohth, x_tt):
        
        update_human_segment_input = cat_valid_tensors([x_hfth, h_hfth, m_hhth, m_ohth, x_tt], dim=-1)
        u_hsts = self.update_human_segment_mlp(update_human_segment_input)
        u_hst, u_hsts = discrete_estimator(u_hsts, strategy=self.discrete_optimization_strategy,
                                           threshold=self.update_segment_threshold)
        return u_hst, u_hsts

    def obj_segment_modelling(self, x_oftk, h_oftk, m_hotk, m_ootk, u_hst, u_hsts, x_tt):
        
        update_object_segment_input = cat_valid_tensors([x_oftk, h_oftk, m_hotk, m_ootk, x_tt], dim=-1)
        u_ostks = self.update_object_segment_mlp(update_object_segment_input)
        u_ostk, u_ostks = discrete_estimator(u_ostks, strategy=self.discrete_optimization_strategy,
                                                threshold=self.update_segment_threshold)
        return u_ostk, u_ostks

    def biGRU_message_passing(self, x_st, u_st, hx_s, entity: str, direction: str):
        
        batch_size, hidden_size = x_st.size(0), self.human_segment_rnn_fcell.hidden_size
        dtype, device = x_st.dtype, x_st.device
        if direction == 'forward':
            h_st = end_state_handler(hx_s, batch_size, hidden_size, dtype, device, detach=False)
            if entity == 'human':
                h_st = u_st * self.human_segment_rnn_fcell(x_st, h_st) + (1.0 - u_st) * h_st
            else:
                h_st = u_st * self.object_segment_rnn_fcell(x_st, h_st) + (1.0 - u_st) * h_st
        else:
            h_st = initial_state_handler(hx_s, batch_size, hidden_size, dtype, device, detach=False)
            if entity == 'human':
                h_st = u_st * self.human_segment_rnn_bcell(x_st, h_st) + (1.0 - u_st) * h_st
            else:
                h_st = u_st * self.object_segment_rnn_bcell(x_st, h_st) + (1.0 - u_st) * h_st
        return h_st


def hidden_transformation(hx_s, ux_s):
   
    batch_size = hx_s.size(0)
    hx_s = [list(torch.unbind(hx_m)) for hx_m in torch.unbind(hx_s)]
    for m in range(batch_size):
        u_sm = ux_s[m]
        end_frames = [-1] + torch.nonzero(u_sm, as_tuple=True)[0].tolist()
        for start_frame, end_frame in zip(end_frames[:-1], end_frames[1:]):
            for t in range(start_frame + 1, end_frame):
                hx_s[m][t] = hx_s[m][end_frame]
    hx_s = torch.stack([torch.stack(hx_m, dim=0) for hx_m in hx_s], dim=0)
    return hx_s


def select_model(model_name: str):
    model_name_to_class_definition = {
        'GeoVisGNN': GeoVisGNN,
    }
    return model_name_to_class_definition[model_name]


def end_state_handler(hx, batch_size, hidden_size, dtype, device, detach=False):
    try:
        hx_t = hx[-1]
    except IndexError:
        hx_t = torch.zeros([batch_size, hidden_size], dtype=dtype, device=device)
    else:
        if detach:
            hx_t = hx_t.detach()
    return hx_t


def initial_state_handler(hx, batch_size, hidden_size, dtype, device, detach=False):
    try:
        hx_t = hx[0]
    except IndexError:
        hx_t = torch.zeros([batch_size, hidden_size], dtype=dtype, device=device)
    else:
        if detach:
            hx_t = hx_t.detach()
    return hx_t


def discrete_estimator(x, strategy: str = 'straight-through', threshold: float = 0.5):
    z, x = straight_through_gumbel_sigmoid(x, threshold=threshold)
    return z, x

def load_model_weights(model_dir: str, checkpoint_type: str):
    
    if checkpoint_type == 'tar':
        checkpoint_file = os.path.join(model_dir, os.path.basename(model_dir) + '.tar')
        checkpoint = torch.load(checkpoint_file)
        state_dict = checkpoint['model_state_dict']
        return state_dict
    else:
        checkpoint_path = model_dir + '/checkpoints/'
        all_checkpoints = [file for file in os.listdir(checkpoint_path) if file.endswith('.pth')]
        all_checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)), reverse=True)
        latest_checkpoint = all_checkpoints[0]
        checkpoint_file = os.path.join(checkpoint_path, latest_checkpoint)
        
        checkpoint = torch.load(checkpoint_file)
        return checkpoint

def messages_passing(receiver, senders, senders_mask, granularity, message_fn):
  
    mx_sr = []
    num_senders = senders.size(1)
    for s in range(num_senders):
        sender = senders[:, s]
        m_sr = message_fn(sender) * senders_mask[:, s:s + 1]
        mx_sr.append(m_sr)
    mx_sr = torch.stack(mx_sr, dim=1)
    return mx_sr


def entity_wise_attention(query, keys, keys_mask, attention_style, attention_fn=None):

    att_weights = []
    num_senders = keys.size(1)    
    for s in range(num_senders):
        key = keys[:, s]
        att_weight = torch.sum(query * key, dim=-1, keepdim=True)
        att_weight = att_weight / math.sqrt(key.size(-1))
        att_weights.append(att_weight)
    att_weights = torch.cat(att_weights, dim=-1)
    neg_inf_values = torch.full_like(att_weights, fill_value=float('-inf'))
    att_weights = torch.where(keys_mask.bool(), att_weights, neg_inf_values)
    att_weights = torch.nn.functional.softmax(att_weights, dim=1)
    att_weights = torch.where(torch.isnan(att_weights), torch.zeros_like(att_weights), att_weights)
    return att_weights