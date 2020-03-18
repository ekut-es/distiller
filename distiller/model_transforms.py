#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
#from torchvision.ops.misc import FrozenBatchNorm2d
from collections import OrderedDict
import distiller
import distiller.modules
from distiller.quantization.sim_bn_fold import SimulatedFoldedBatchNorm
import logging
msglogger = logging.getLogger()


__all__ = ["fold_batch_norms"]

class Identity(nn.Module):
    def forward(self, x):
        return x

def fold_batch_norms(model, dummy_input=None, adjacency_map=None, inference=True):
    """Scans the model for convolution / linear modules followed by batch-normalization. For each such valid pair,
    folds the parameters of the batch normalization module into the parameters of the parameter module, and replaces
    the batch normalization module with an identity operation.

    To infer the order of modules it is required to perform a forward pass on the model. Hence the need to pass the
    expected input shape.

    Args:
        model (nn.Module): Model instance on which the transformation is performed
        dummy_input (torch.Tensor or tuple): Dummy input to the model. Required if summary_graph is None
        adjacency_map (OrderedDict): Pre-computed adjacency map, via SummaryGraph.adjacency_map(). Must be based
          on the passed model, otherwise results are unexpected. If None, then the adjacency map will be created
          internally using the passed dummy_input.
        inference (bool): an indicator on whether or not the modules are in inference mode.
            This will hard-fuse all BatchNorms into the param-layers.
    """
    def fold_bn(param_module, bn_module):
        try:
            folded_module = SimulatedFoldedBatchNorm(param_module, bn_module)
        except ValueError:
            msglogger.warning("Can't fold, {} does not track running stats".format(bn_module.distiller_name))
            return None
        if inference:
            folded_module.freeze()
            return folded_module.param_module
        return folded_module


    foldables = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    batchnorms = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    
      
    if isinstance(model, nn.Sequential):
        for pos, submodule in enumerate(model):
            if pos > 0:
                last_submodule = model[pos-1]
                if (type(last_submodule) in foldables) and (type(submodule) in batchnorms):
                    folded_bn = fold_bn(last_submodule, submodule)
                    model[pos-1] = folded_bn
                    model[pos] = Identity()
        
    for module_name in model._modules:
        fold_batch_norms(model._modules[module_name], dummy_input, adjacency_map, inference)
    
    return model
