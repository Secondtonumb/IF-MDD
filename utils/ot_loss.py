# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
from torch.autograd import Function
from numba import cuda, float32, int32
import numpy as np

batched_bucketize = torch.vmap(torch.bucketize)

def jensen_shannon_divergence(p, q):
    """Compute the Jensen-Shannon divergence between two probability distributions.

    Parameters
    ----------
    p : tensor (n, d)
        The first probability distribution (logits or softmaxed).
    q : tensor (n, d)
        The second probability distribution (usually target).

    Returns
    -------
    tensor (n,)
        The Jensen-Shannon divergence between the two distributions (per example).
    """
    q = q + (1-q)*1e-12 - q*1e-12
    m = 0.5 * (p + q)
    m = torch.log(m)
    p = torch.log(p)
    q = torch.log(q)
    return 0.5 * (F.kl_div(p, m, log_target=True, reduction='none') + F.kl_div(q, m, log_target=True, reduction= 'none'))

def ot1d_transport(p_weights, q_weights):
    """Compute the 1D optimal transport plan between two discrete measures.

    Parameters
    ----------
    p_weights : tensor (n,)
        Weights of the first measure (must sum to 1).
    q_weights : tensor (m,)
        Weights of the second measure (must sum to 1).

    Returns
    -------
    tensor (n, m)
        The optimal transport matrix (gamma) between the two measures.
    """
    i = 0
    j = 0
    p_nbins = len(p_weights)
    q_nbins = len(q_weights)
    w_i = p_weights[i].clone()
    w_j = q_weights[j].clone()
    gamma = torch.zeros(p_nbins, q_nbins).to('cuda')
    while True:
        if w_i < w_j :
            gamma[i,j] = w_i
            i += 1
            if i == p_nbins:
                break
            w_j -= w_i
            w_i = p_weights[i].clone()
        else:
            gamma[i,j] = w_j
            j +=1 
            if j == q_nbins:
                break
            w_i -= w_j
            w_j = q_weights[j].clone()
    return gamma

def ot1d_batch_transport(p_weights_batch, q_weights_batch):
    """Compute 1D optimal transport plans for a batch of discrete measures.

    Parameters
    ----------
    p_weights_batch : tensor (b, n)
        Weights of the first measures in the batch.
    q_weights_batch : tensor (b, m)
        Weights of the second measures in the batch.

    Returns
    -------
    tensor (b, n, m)
        The batch of optimal transport matrices (gamma).
    """
    max_p_nbins = p_weights_batch.shape[1]
    max_q_nbins = q_weights_batch.shape[1]
    batch_size = p_weights_batch.shape[0]
    gamma_batch = torch.zeros(batch_size, max_p_nbins, max_q_nbins).to('cuda')

    for b in range(batch_size):
        p_weights = p_weights_batch[b]
        q_weights = q_weights_batch[b]
        i = 0
        j = 0
        p_nbins = len(p_weights)
        q_nbins = len(q_weights)
        w_i = p_weights[i].clone()
        w_j = q_weights[j].clone()
        while True:
            if w_i < w_j :
                gamma_batch[b,i,j] = w_i
                i += 1
                if i == p_nbins:
                    break
                w_j -= w_i
                w_i = p_weights[i].clone()
            else:
                gamma_batch[b,i,j] = w_j
                j +=1 
                if j == q_nbins:
                    break
                w_i -= w_j
                w_j = q_weights[j].clone()
    return gamma_batch

def ottc_loss_reference_implementation(p_weights, q_weights, pfeatures, qfeatures):
    '''Compute OTTC (Optimal Transport with Cross-Entropy cost) for a pair of measures.
    https://arxiv.org/pdf/2502.01588v1; equation (11)

    Parameters
    ----------
    p_weights : tensor (n,)
        Weights of the first measure.
    q_weights : tensor (m,)
        Weights of the second measure.
    pfeatures : tensor (n, d)
        Features (logits) for the first measure.
    qfeatures : tensor (m, d)
        Features (targets, typically one-hot) for the second measure.

    Returns
    -------
    tensor
        OTTC loss between the two measures.
    '''

    i = 0
    j = 0
    p_nbins = len(p_weights)
    q_nbins = len(q_weights)
    w_i = p_weights[i].clone()
    w_j = q_weights[j].clone()
    p_transport_indices = []
    q_transport_indices = []
    transport_w = []  # Initialize transport_w as an empty list
    while True:
        if w_i < w_j :
            transport_w.append(w_i.unsqueeze(0))  # Add new element to transport_w
            p_transport_indices.append(i)
            q_transport_indices.append(j)
            i += 1
            if i == p_nbins:
                break
            w_j -= w_i
            w_i = p_weights[i].clone()
        else:
            transport_w.append(w_j.unsqueeze(0))  # Add new element to transport_w
            p_transport_indices.append(i)
            q_transport_indices.append(j)
            j +=1 
            if j == q_nbins:
                break
            w_i -= w_j
            w_j = q_weights[j].clone()
    # Convert transport_w to a tensor and move it to GPU
    transport_w = torch.cat(transport_w).to('cuda')
    cross_list = F.cross_entropy(pfeatures[p_transport_indices,:] , qfeatures[q_transport_indices,:].float(), reduction= 'none')
    return (transport_w * cross_list).sum()

def batched_ottc_loss_reference_implementation(p_weights, q_weights, pfeatures, qfeatures):
    '''Compute OTTC (Optimal Transport with Cross-Entropy cost) for a batch of measures.
    https://arxiv.org/pdf/2502.01588v1; equation (11)

    Parameters
    ----------
    p_weights : tensor (b, n)
        Weights of the first measures in the batch.
    q_weights : tensor (b, m)
        Weights of the second measures in the batch.
    pfeatures : tensor (b, n, d)
        Features (logits) of the first measures.
    qfeatures : tensor (b, m, d)
        Features (targets) of the second measures.

    Returns
    -------
    tensor
        Sum of OTTC losses over the batch.
    '''
    
    batch_losses = []
    batch_size = p_weights.size(0)

    for b in range(batch_size):
        loss = ottc_loss_reference_implementation(
            p_weights[b],
            q_weights[b],
            pfeatures[b],
            qfeatures[b]
        )
        batch_losses.append(loss)

    return torch.stack(batch_losses).sum()
    
def batched_ottc_loss_bucketized(x, y, a, b, amask = None, bmask = None, euclidian = False, jsd = False, entropy = 0.0, loss_options = ''):
    """Compute OTTC loss for batched data using bucketization.
    
    https://arxiv.org/pdf/2502.01588v1; equation (11)
    used in all experiments of the paper.

    Implementation inspired from : https://github.com/gnies/1d-optimal-transport

    This version uses cumulative distribution functions (CDFs) and `torch.vmap` to compute the transport plan
    efficiently across batches.

    Parameters
    ----------
    x : tensor (b, n, d)
        Logits or features of the input sequences.
    y : tensor (b, m, d)
        One-hot or soft labels of the target sequences.
    a : tensor (b, n)
        Probability simplex (weights) for x.
    b : tensor (b, m)
        Probability simplex (weights) for y.
    amask : tensor (b, n), optional
        Mask for x indicating valid positions (1) vs padded (0).
    bmask : tensor (b, m), optional
        Mask for y indicating valid positions (1) vs padded (0).
    euclidian : bool, optional
        Whether to use Euclidean cost instead of cross-entropy.
    jsd : bool, optional
        Whether to use Jensen-Shannon divergence instead of cross-entropy.
    entropy : float, optional
        Entropy regularization weight.
    loss_options : str, optional
        Additional weighting options (e.g., 'inverse_input_renorm').

    Returns
    -------
    tuple
        ot_cost : float
            Total optimal transport cost.
        index_qx_ : tensor
            Indices of matched source bins.
        index_qy_ : tensor
            Indices of matched target bins.
        h_ : tensor
            Transport plan weights.
    """

        
    if amask is None:
        amask = torch.ones(a.shape).to('cuda')
    if bmask is None:
        bmask = torch.ones(b.shape).to('cuda')   
    
    x = x.reshape([-1,x.shape[-1]])
    y = y.reshape([-1,y.shape[-1]])
       
    ca = torch.cumsum(a, dim=1)
    cb = torch.cumsum(b, dim=1)
    cba = torch.sort(torch.cat([ca, cb], dim=1))[0]
    
    xbin_nb = amask.sum(axis =1)[:,None] 
    index_qx = batched_bucketize(cba, ca, right=False)    
    index_qx_ = torch.clamp(index_qx, max = xbin_nb.int()  -1)
    # keep_index_qx_mask  = ((xbin_nb - 1 - index_qx) >= 0).int()
    # index_qx = index_qx * keep_index_qx_mask 
    # index_qx_ = index_qx + (1 - keep_index_qx_mask) * index_qx.max(axis = 1)[0][:,None]
    index_qx = index_qx_ + (torch.arange(0,ca.shape[0])*(ca.shape[1]))[:,None].to('cuda')
    index_qx = index_qx.flatten()

    ybin_nb = bmask.sum(axis =1)[:,None]
    index_qy = batched_bucketize(cba, cb, right=False)  
    index_qy_ = torch.clamp(index_qy, max = ybin_nb.int()   -1)
    # keep_index_qy_mask = ((ybin_nb - 1 - index_qy) >= 0).int()
    # index_qy = index_qy * keep_index_qy_mask 
    # index_qy_ = index_qy + (1 - keep_index_qy_mask) * index_qy.max(axis = 1)[0][:,None]
    index_qy = index_qy_ + (torch.arange(0,cb.shape[0])*(cb.shape[1]))[:,None].to('cuda')
    index_qy = index_qy.flatten()    
    
    h_ = torch.diff(torch.cat([torch.tensor([0]*cba.shape[0]).to('cuda').reshape(-1,1), cba], dim=1)) # 1-d transport    
    w = h_.flatten()
    w_mask = w > 0

    index_qx = index_qx[w_mask]
    index_qy = index_qy[w_mask]
    w = w[w_mask]
    
    if jsd == True:
        cross_list_ = jensen_shannon_divergence(F.softmax(x[index_qx,:],dim = 1), y[index_qy,:].float()).sum(axis = 1)
        return (w * cross_list_).sum() , index_qx_, index_qy_ , h_
     
    if euclidian == True :
        cross_list_ = torch.sum((F.softmax(x[index_qx,:],dim = 1) -  y[index_qy,:].float())**2, axis = 1)
        return (w * cross_list_).sum() , index_qx_, index_qy_ , h_

    if entropy > 0:
        return (w * cross_list_).sum() - entropy * torch.sum(a * torch.log(a + 1e-12)), index_qx_, index_qy_ , h_        

    cross_list_ = F.cross_entropy(x[index_qx,:] , y[index_qy,:].float(), reduction= 'none')
    return (w * cross_list_).sum(),  index_qx_, index_qy_ , h_ 

def batched_ottc_loss_concatenated(x, y, a, b, euclidian = False, jsd = False, entropy = 0.0, loss_options = ''):
    """Compute OTTC loss for batched data using concatenated distributions.

    This method is faster but introduces very small numerical coupling between batches,
    which is negligible in practice and does not affect learning.

    Implementation inspired from : https://github.com/gnies/1d-optimal-transport

    Parameters
    ----------
    x : tensor (b, n, d)
        Logits or features of the input sequences.
    y : tensor (b, m, d)
        One-hot or soft labels of the target sequences.
    a : tensor (b, n)
        Probability simplex (weights) for x.
    b : tensor (b, m)
        Probability simplex (weights) for y.
    euclidian : bool, optional
        Whether to use Euclidean cost instead of cross-entropy.
    jsd : bool, optional
        Whether to use Jensen-Shannon divergence instead of cross-entropy.
    entropy : float, optional
        Entropy regularization weight.
    loss_options : str, optional
        Additional loss control options.

    Returns
    -------
    tuple
        ot_cost : float
            Total optimal transport cost.
        index_qx : tensor
            Indices of matched source bins.
        index_qy : tensor
            Indices of matched target bins.
        h : tensor
            Transport plan weights.
    """
    
    x = x.reshape([-1,x.shape[-1]])
    y = y.reshape([-1,y.shape[-1]])

    ca = torch.cumsum(a, dim=1)
    cb = torch.cumsum(b, dim=1)
    
    # ca = ca + torch.arange(0,ca.shape[0]*1,1)[:,None].to('cuda')
    ca.add_(torch.arange(0,ca.shape[0]*1,1, device = 'cuda')[:,None])
    ca = ca .reshape([-1,1 ]).squeeze()
    # cb = cb + torch.arange(0,cb.shape[0]*1,1)[:,None].to('cuda')
    cb.add_(torch.arange(0,cb.shape[0]*1,1, device = 'cuda')[:,None])
    cb = cb .reshape([-1,1 ]).squeeze()
    
    # points on which we need to evaluate the quantile functions
    cba = torch.sort(torch.cat([ca, cb], dim=0))[0]

    # construction of first quantile function index
    index_qx = torch.bucketize(cba, ca, right=False)    
    index_qx = torch.clamp(index_qx, max=x.shape[0] - 1)                                                        
                                                         
    # construction of second quantile function index
    index_qy = torch.bucketize(cba, cb, right=False)   
    index_qy = torch.clamp(index_qy, max=y.shape[0] - 1)
                                                         

    h = torch.diff(F.pad(cba, (1, 0)))  # 1-d transport w
    w = h[h >= 0]
     
    if jsd == True:
        cross_list_ = jensen_shannon_divergence(F.softmax(x[index_qx,:],dim = 1), y[index_qy,:].float()).sum(axis = 1)
        return (w * cross_list_).sum()
    
    if euclidian == True and not soft_before_euclidian:
        cross_list_ = torch.sum((x[index_qx,:] -  y[index_qy,:].float())**2, axis = 1)
        return (w * cross_list_).sum()
    
    if euclidian == True:
        cross_list_ = torch.sum((F.softmax(x[index_qx,:],dim = 1) -  y[index_qy,:].float())**2, axis = 1)
        return (w * cross_list_).sum()
    
    cross_list_ = F.cross_entropy(x[index_qx,:] , y[index_qy,:].float(), reduction= 'none')    
    return (w * cross_list_).sum(), index_qx, index_qy, h

if __name__ == "__main__":
    import time
    import numpy as np

    # Generate dummy inputs
    batch_size = 10
    seq_len_logits = 100
    seq_len_labels = 70
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len_logits, vocab_size).to('cuda')
    ohlabels = torch.randn(batch_size, seq_len_labels, vocab_size).to('cuda')

    weights_logits = torch.randn(batch_size, seq_len_logits).to('cuda')
    logits_mask = torch.ones(batch_size, seq_len_logits).to('cuda')
    for i in range(batch_size):
        length = torch.randint(10, seq_len_logits, (1,))
        logits_mask[i, length:] = 0
    weights_logits = torch.softmax(weights_logits.masked_fill(logits_mask == 0, -torch.inf), dim=-1)

    weights_labels = torch.randn(batch_size, seq_len_labels).to('cuda')
    labels_mask = torch.ones(batch_size, seq_len_labels).to('cuda')
    for i in range(batch_size):
        length = torch.randint(10, seq_len_labels, (1,))
        labels_mask[i, length:] = 0
    weights_labels = torch.softmax(weights_labels.masked_fill(labels_mask == 0, -torch.inf), dim=-1)

    # Compute losses
    loss_ref = batched_ottc_loss_reference_implementation(weights_logits, weights_labels, logits, ohlabels)
    loss_bucketized, *_ = batched_ottc_loss_bucketized(logits, ohlabels, weights_logits, weights_labels, logits_mask, labels_mask)
    loss_concatenated, *_ = batched_ottc_loss_concatenated(logits, ohlabels, weights_logits, weights_labels)
    # loss_concatenated_h100, *_ = batched_ottc_loss_concatenated_h100(logits, ohlabels, weights_logits, weights_labels)

    print("\n=== OTTC Loss Comparison ===")
    print(f"[Reference]      Loss: {loss_ref.sum().item():.6f}")
    print(f"[Bucketized]     Loss: {loss_bucketized.item():.6f}")
    print(f"[Concatenated]   Loss: {loss_concatenated.item():.6f}")
    # print(f"[Concatenated H100] Loss: {loss_concatenated_h100.item():.6f}")

    # Benchmark function
    def benchmark(func, *args):
        torch.cuda.synchronize()
        start = time.time()
        func(*args)
        torch.cuda.synchronize()
        return time.time() - start

    print("\n=== Timing ===")
    time_ref = benchmark(batched_ottc_loss_reference_implementation, weights_logits, weights_labels, logits, ohlabels)
    time_bucket = benchmark(batched_ottc_loss_bucketized, logits, ohlabels, weights_logits, weights_labels, logits_mask, labels_mask)
    time_concat = benchmark(batched_ottc_loss_concatenated, logits, ohlabels, weights_logits, weights_labels)
    # time_concat_h100 = benchmark(batched_ottc_loss_concatenated_h100, logits, ohlabels, weights_logits, weights_labels)

    print(f"[Reference]      Time: {time_ref:.6f} s")
    print(f"[Bucketized]     Time: {time_bucket:.6f} s")
    print(f"[Concatenated]   Time: {time_concat:.6f} s")
    # print(f"[Concatenated H100] Time: {time_concat_h100:.6f} s")

    print("\nSpeedup (concat / ref):    {:.2f}x".format(time_ref / time_concat))
    print("Speedup (bucket / ref):    {:.2f}x".format(time_ref / time_bucket))
    # print("Speedup (concat_h100 / ref): {:.2f}x".format(time_ref / time_concat_h100))
