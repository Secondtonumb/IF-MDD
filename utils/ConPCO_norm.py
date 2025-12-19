# -*- coding: utf-8 -*-
# @Author  : Bi-Cheng Yan
# @Affiliation  : National Taiwan Normal University
# @Email   : bicheng@ntnu.edu.tw
# @File    : conPCO_norm.py

import torch
import torch.nn.functional as F
import random
import numpy as np

def euclidean_dist(x, y):
		"""
		Args:
			x: pytorch Variable, with shape [m, d]
			y: pytorch Variable, with shape [n, d]
		Returns:
			dist: pytorch Variable, with shape [m, n]
		"""
		m, n = x.size(0), y.size(0)
		xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
		yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
		dist = xx + yy
		dist.addmm_(1, -2, x, y.t())
		dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
		return dist

def up_triu(x):
		# return a flattened view of up triangular elements of a square matrix
		n, m = x.shape
		assert n == m
		_tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
		return x[_tmp]

class ContrastivePhonemicOrdinalRegularizer(torch.nn.Module):
	def __init__(self, lambda_d_phn, lambda_t_phn, lambda_clap_t2a, margin, ignore_index=-1):
		super().__init__()
		self.ignore_index = ignore_index
		self.lambda_d_phn = lambda_d_phn
		self.lambda_t_phn = lambda_t_phn
		self.lambda_clap_t2a = lambda_clap_t2a
		self.margin = margin
		
		print('margin:',self.margin)
		print('lambda_d_phn:',self.lambda_d_phn)
		print('lambda_t_phn:',self.lambda_t_phn)

	def forward(self, features, features_text, gt, phn_id):
		"""
		Features: a certain layer's features
		gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
		mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
		"""
		# import pdb; pdb.set_trace()
		f_n, f_l, f_c = features.size()
		_gt = gt.view(-1)

		#_norm : excluding padding 
		_mask_norm = _gt > 0
		_mask_norm = _mask_norm.to(torch.bool)
		_gt_norm = _gt[_mask_norm]

		#_high : score:2 only
		_mask_high = _gt == 2
		_mask_high = _mask_high.to(torch.bool)
		_gt_high = _gt[_mask_high]

		_phn_id_norm = phn_id.view(-1)[_mask_norm]
		u_value_phn_norm, u_index_phn_norm, u_counts_phn_norm = torch.unique(_phn_id_norm, return_inverse=True, return_counts=True)

		_phn_id_high = phn_id.view(-1)[_mask_high]
		u_value_phn_high, u_index_phn_high, u_counts_phn_high = torch.unique(_phn_id_high, return_inverse=True, return_counts=True)

		# skip some phone categories which do not have the 2.0 score 
		skip_phns = torch.tensor([i not in u_value_phn_high for i in u_value_phn_norm]).to(_gt.device)
		if sum(skip_phns) != 0:
			# from non-padding data mask some phone which do not have 2.0 score
			_mask_phn = torch.tensor([i not in skip_phns for i in _phn_id_norm]).to(_gt.device)
			_mask = _mask_phn
			_gt = _gt_norm[_mask]
			_features = features.reshape(-1, f_c)
			_features = _features[_mask_norm, :][_mask_phn, :]
			_features_text = features_text.reshape(-1, f_c)
			_features_text = _features_text[_mask_norm, :][_mask_phn, :]
			_phn_id = phn_id.view(-1)[_mask_norm][_mask_phn]
		else:
			_mask = _mask_norm
			_gt = _gt[_mask_norm] 
			_features = features.reshape(-1, f_c)
			_features = _features[_mask,:]
			_features_text = features_text.reshape(-1, f_c)
			_features_text = _features_text[_mask_norm,:]
			_phn_id = phn_id.view(-1)[_mask_norm]

		u_value_phn, u_index_phn, u_counts_phn = torch.unique(_phn_id, return_inverse=True, return_counts=True)

		# calculate a center for each phn
		center_f_phn = torch.zeros([len(u_value_phn), f_c]).to(_features.device)
		center_f_phn.index_add_(0, u_index_phn, _features)
		u_counts_phn = u_counts_phn.unsqueeze(1)
		center_f_phn = center_f_phn / u_counts_phn
		center_f_phn = F.normalize(center_f_phn, dim=1)

		# calculate a center for each phn (text_feats)
		center_f_phn_text = torch.zeros([len(u_value_phn), f_c]).to(_features.device)
		center_f_phn_text.index_add_(0, u_index_phn, _features_text)
		center_f_phn_text = center_f_phn_text / u_counts_phn
		center_f_phn_text = F.normalize(center_f_phn_text, dim=1)

		# calculate contrastive loss for audio and text
		# performs clap and anchor sets phn_audio_feats (A-T)
		cos_matrix_phn = torch.matmul(center_f_phn, center_f_phn_text.transpose(0,1)) # (37, 37)
		cos_matrix_phn = F.log_softmax(cos_matrix_phn, dim=1)
		cos_diag_path_phn = torch.diagonal(cos_matrix_phn, 0)
		loss_phn_clap_audio = cos_diag_path_phn.mean() * -1

		# performs clap and anchor sets phn_text_feats
		cos_matrix_text = cos_matrix_phn.transpose(0,1)
		cos_diag_path_text = torch.diagonal(cos_matrix_text, 0)
		loss_phn_clap_text = cos_diag_path_text.mean() * -1

		loss_center_clap = self.lambda_clap_t2a * loss_phn_clap_audio + (1-self.lambda_clap_t2a) * loss_phn_clap_text
		
		# calculate dist between phn-centers
		p_phn = F.normalize(center_f_phn, dim=1)
		_distance_phn = euclidean_dist(p_phn, p_phn)
		_distance_phn = up_triu(_distance_phn)

		# calculate diverse term form phn
		u_value_phn = u_value_phn.unsqueeze(1)
		# assume a margin is 1
		_distance_phn = _distance_phn * 1
		## L_d, diverse term, push away the distence between score-centers
		_entropy_phn = torch.mean(_distance_phn)
		_features = F.normalize(_features, dim=1)

		# calculate tightness term from phn
		# find phn-scnter for each features in the batch
		_features_center_phn = p_phn[u_index_phn, :]
		_features_phn = _features - _features_center_phn
		_features_phn = _features_phn.pow(2)
		_tightness_phn = torch.sum(_features_phn, dim=1)
		_mask = _tightness_phn > 0

		# come close to center while considering ordinal 
		# 3.0 set to be highest score
		high_score = torch.ones(_gt[_mask].size()).to(_gt.device) * 2.0
		ordinal_weight = (high_score - _gt[_mask]) + self.margin
		_tightness_phn = _tightness_phn[_mask] 
		_tightness_phn = torch.sqrt(_tightness_phn) * ordinal_weight
		_tightness_phn = torch.mean(_tightness_phn)

		loss_oe = (self.lambda_t_phn* _tightness_phn) - (self.lambda_d_phn * _entropy_phn)

		return loss_oe, loss_center_clap
