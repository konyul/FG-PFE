import torch
import torch.nn as nn
import copy
from pcdet.utils.spconv_utils import spconv

# Codes are from https://github.com/raja-kumar/folding-batchnorm/blob/master/bn_folder.py

def bn_folding_model(model):
    new_model = copy.deepcopy(model)
    module_names = list(new_model._modules)
    for k, name in enumerate(module_names):
        if new_model._modules[name] is not None:
            if len(list(new_model._modules[name]._modules)) > 0:
                new_model._modules[name] = bn_folding_model(new_model._modules[name])
                
            else:
                if isinstance(new_model._modules[name], nn.BatchNorm2d) or isinstance(new_model._modules[name], nn.BatchNorm1d):
                    if isinstance(new_model._modules[module_names[k-1]], nn.Conv2d) or isinstance(new_model._modules[module_names[k-1]], nn.Linear) or isinstance(new_model._modules[module_names[k-1]], nn.Conv1d) or isinstance(new_model._modules[module_names[k-1]], spconv.SparseConv2d) or isinstance(new_model._modules[module_names[k-1]], spconv.SubMConv2d): # or isinstance(new_model._modules[module_names[k-1]], nn.ConvTranspose2d) :
                        # Folded BN
                        folded_conv = fold_conv_bn_eval(new_model._modules[module_names[k-1]], new_model._modules[name])

                        # Replace old weight values
                        new_model._modules.pop(name) # Remove the BN layer
                        new_model._modules[module_names[k-1]] = folded_conv # Replace the Convolutional Layer by the folded version
    return new_model

def bn_folding(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    
    w_fold = conv_w * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
    b_fold = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    
    return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)

def bn_folding_linear(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    
    w_fold = conv_w * (bn_w * bn_var_rsqrt).view(-1, 1)
    b_fold = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)

def fold_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)
    if isinstance(conv, nn.Linear) or isinstance(conv, nn.Conv1d):
        fused_conv.weight, fused_conv.bias = bn_folding_linear(fused_conv.weight, fused_conv.bias,
                                bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    elif isinstance(conv, nn.Conv2d) or isinstance(conv, nn.ConvTranspose2d) or isinstance(conv, spconv.SubMConv2d) or isinstance(conv, spconv.SparseConv2d):
        fused_conv.weight, fused_conv.bias = bn_folding(fused_conv.weight, fused_conv.bias,
                                bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return fused_conv