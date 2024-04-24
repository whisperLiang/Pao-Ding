import torch
import torchvision
from torchvision.models import *
from torch_split import utils, ops
import typing
import torch.nn as nn
import warnings
from collections import deque

class Node(object):
    """ Node of DepGraph
    """
    def __init__(self, module: nn.Module, grad_fn=None, name: str = None, inshape=None, outshape=None):
        # For Computational Graph (Tracing)
        self.inputs = []  # input nodes
        self.outputs = [] # output nodes
        self.module = module # reference to torch.nn.Module
        self.grad_fn = grad_fn # grad_fn of nn.module output

        self._name = name # node name
        self.type = ops.module2type(module) # node type (enum), op.OPTYPE
        self.module_class = module.__class__ # class type of the module
        self.indegree = 0 # indegree of the node to judge whether its inputs are all ready
        self.outdegree = 0 # outdegree of the node to judge whether it is necessary to save the result
        self.inshape = inshape # input shape
        self.outshape = outshape # output shape
        self.outresult = None # output result

        # For Dependency Graph
        self.dependencies = []  # Adjacency List
        self.enable_index_mapping = True # whether to enable index mapping
        self.pruning_dim = -1 # the dimension to be pruned

    @property
    def name(self):
        if self._name is None:
            return str(self.module)
        else:
            fmt = self._name
            if self.type != ops.OPTYPE.PARAMETER:
                fmt += " ({})".format(str(self.module))
            return fmt

    def add_input(self, node):
        self.inputs.append(node)


    def add_output(self, node):
        self.outputs.append(node)


    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<Node: ({})>".format(self.name)

    def details(self):
        fmt = "-" * 32 + "\n"
        fmt += "<Node: ({})>\n".format(self.name)
        fmt += " " * 4 + "IN:\n"
        for in_node in self.inputs:
            fmt += " " * 8 + "{}\n".format(in_node)
        fmt += " " * 4 + "OUT:\n"
        for out_node in self.outputs:
            fmt += " " * 8 + "{}\n".format(out_node)
        fmt += " " * 4 + "DEP:\n"
        for dep in self.dependencies:
            fmt += " " * 8 + "{}\n".format(dep)
        fmt += "\tEnable_index_mapping={}, pruning_dim={}\n".format(
            self.enable_index_mapping, self.pruning_dim)
        fmt += "-" * 32 + "\n"
        return fmt
    
    def forward(self, x):

        # for debug
        # if 'concatop_22' in self.name.lower():
        #     print("test")
        # elif 'concatop_20' in self.name.lower():
        #     print("test")
        
        try:  
            # 如果module实现了forward方法
            return self.module.forward(x)
        except:  
            #  module没有实现forward方法   
            return self.custom_forward(x)
        
    def custom_forward(self, x):
        # Custom forward logic using grad_fn property
        if "avgpool" in self.grad_fn.name().lower():
            # Get saved kernel_size, stride, and padding from grad_fn
            kernel_size = self.grad_fn._saved_kernel_size
            stride = self.grad_fn._saved_stride
            padding = self.grad_fn._saved_padding
            # Use nn.functional.avg_pool2d instead of avgpool
            return nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
        elif "maxpool" in self.grad_fn.name().lower():
            kernel_size = self.grad_fn._saved_kernel_size
            stride = self.grad_fn._saved_stride
            padding = self.grad_fn._saved_padding
            dilation = self.grad_fn._saved_dilation
            ceil_mode = self.grad_fn._saved_ceil_mode
            return nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)
        elif "mean" in self.grad_fn.name().lower():
            if self.grad_fn._saved_keepdim:
                return nn.functional.adaptive_avg_pool2d(x, (1, 1))
                
            else:
                return torch.mean(x, self.grad_fn._saved_dim)
        elif "transpose" in self.grad_fn.name().lower():
            dim0 = self.grad_fn._saved_dim0
            dim1 = self.grad_fn._saved_dim1
            return torch.transpose(x, dim0, dim1).contiguous()
        else:
            print(f"{self.name} has no forward method. Please check it correct.")


class DependencyGraph(object):
    def __init__(self):
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
        self.CUSTOMIZED_LAYERS = [nn.Dropout, nn.Dropout1d, nn.Dropout2d, torchvision.ops.StochasticDepth, GeneralizedRCNNTransform, DefaultBoxGenerator] # user-customized layers

        self.IGNORED_LAYERS = []
        self._op_id = 0 # operatior id, will be increased by 1 for each new operator

        # Pruning History
        # self._pruning_history = []

    def build_dependency(
        self,
        model: torch.nn.Module,
        example_inputs: typing.Union[torch.Tensor, typing.Sequence, typing.Dict],
        forward_fn: typing.Callable[[torch.nn.Module, typing.Union[torch.Tensor, typing.Sequence]], torch.Tensor] = None,
        output_transform: typing.Callable = None,
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None,
        CUSTOMIZED_LAYERS: typing.List[torch.nn.Module] = None,
        IGNORED_LAYERS: typing.List[torch.nn.Module] = None,
        verbose: bool = True,
    ) -> "DependencyGraph":
        """Build a dependency graph through tracing.
        Args:
            model (class): the model to be pruned.
            example_inputs (torch.Tensor or List): dummy inputs for tracing.
            forward_fn (Callable): a function to forward the model with example_inputs, which should return a reduced scalr tensor for backpropagation.
            output_transform (Callable): a function to transform network outputs.
            unwrapped_parameters (typing.Dict[nn.Parameter, int]): unwrapped nn.parameters that do not belong to standard nn.Module.
            CUSTOMIZED_LAYERS (typing.List[torch.nn.Module]): customized layers for a specific layer type or a specific layer instance.
            IGNORED_LAYERS (typing.List[torch.nn.Module]): ignored layers for a specific layer type or a specific layer instance.
            verbose (bool): verbose mode.
        """

        self.verbose = verbose
        self.model = model

        self.IGNORED_LAYERS += IGNORED_LAYERS if IGNORED_LAYERS is not None else self.IGNORED_LAYERS
        for m in self.IGNORED_LAYERS:
            for layer in m.modules():
                if layer not in self.IGNORED_LAYERS:
                    self.IGNORED_LAYERS.append(layer)

        self.CUSTOMIZED_LAYERS += CUSTOMIZED_LAYERS if CUSTOMIZED_LAYERS is not None else self.CUSTOMIZED_LAYERS
        self._module2name = {module: name for (name, module) in model.named_modules()} # nn.Module => module name

        self.head_node = None
        self.tail_node = None
     
        # Ignore all sub-modules of customized layers as they will be handled by the customized layers
        for layer_type_or_instance in self.CUSTOMIZED_LAYERS:            
            for m in self.model.modules():
                # a layer instance or a layer type
                if (m==layer_type_or_instance) or (not isinstance(layer_type_or_instance, torch.nn.Module) and isinstance(m, layer_type_or_instance)):
                    for sub_module in m.modules(): 
                        if sub_module != m:
                            self.IGNORED_LAYERS.append(sub_module)

        # Detect unwrapped nn.parameters
        self._param_to_name, self.unwrapped_parameters = self._detect_unwrapped_parameters(unwrapped_parameters)

        # Detect torch.no_grad()
        # 依赖关系图依赖于自动grad进行跟踪。请确保您的代码中没有torch.no_grad（）
        assert torch.is_grad_enabled(), "Dependency graph relies on autograd for tracing. Please make sure there is no torch.no_grad() in your code."
        
        # Build computational graph through tracing. 
        self.module2node = self._trace(
            model, example_inputs, forward_fn, output_transform=output_transform
        )

        return self
    
    def _trace(self, model, example_inputs, forward_fn, output_transform):
        """ Tracing the model as a graph
        """
        model.eval()
        gradfn2module = {}
        visited = {}
        gradfn2inshape = {}
        gradfn2outshape = {}
        self.module2result = {}
        self._2d_4d = True # only for pytorch<=1.8
        def _record_grad_fn(module, inputs, outputs):
            if module not in visited:
                visited[module] = 1
            else:
                visited[module] += 1
            
            if isinstance(module, nn.Linear) and len(outputs.shape)==3:
                self._2d_4d=False

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(outputs, torch.nn.utils.rnn.PackedSequence):
                outputs = outputs.data
            if isinstance(outputs, torch.Tensor):
                self.module2result[module] = outputs.clone().detach()

            if len(module._modules) == 0:
                gradfn2module[outputs.grad_fn] = module
                try:
                    gradfn2inshape[outputs.grad_fn] = inputs[0].shape
                except:
                    gradfn2inshape[outputs.grad_fn] = inputs[0][0].shape
                gradfn2outshape[outputs.grad_fn] = outputs.shape

        # Register hooks for layerable modules
        hooks = [
            m.register_forward_hook(_record_grad_fn)
            for m in model.modules()
            # if (isinstance(m, registered_types) and m not in self.IGNORED_LAYERS)
            # if m not in self.IGNORED_LAYERS and len(m._modules)==0
            # if m not in self.IGNORED_LAYERS and len(m._modules)==0 and not isinstance(m, torch.nn.modules.dropout.Dropout)# only register leaf modules
            if not isinstance(m, tuple(self.CUSTOMIZED_LAYERS))# only register leaf modules
        ]

        # Feed forward to record gradient functions of prunable modules
        if forward_fn is not None:
            out = forward_fn(model, example_inputs)
        elif isinstance(example_inputs, dict):
            out = model(**example_inputs)
        else:
            try:
                out = model(*example_inputs)
            except:
                out = model(example_inputs)
        for hook in hooks:
            hook.remove()

        # for recursive models or layers
        reused = [m for (m, count) in visited.items() if count > 1]

        # Graph tracing
        if output_transform is not None:
            out = output_transform(out)
        module2node = {} # create a mapping from nn.Module to tp.dependency.Node

        visited = set()
        for o in utils.flatten_as_list(out):
            self._trace_computational_graph(
                module2node, o.grad_fn, gradfn2module, gradfn2inshape, gradfn2outshape, reused, visited=visited)

        # TODO: Improving ViT pruning
        # This is a corner case for pruning ViT,
        # where concatination of pos_emb and cls_emv is not applied on the feature dim.
        # Notably, this is not a good practice and will be fixed in the future version
        if len(self.unwrapped_parameters) > 0:
            for node in module2node.values():
                if node.type in (ops.OPTYPE.CONCAT, ops.OPTYPE.SPLIT):
                    stack = [node]
                    visited = set()
                    while len(stack) > 0:
                        n = stack.pop(-1)
                        visited.add(n)
                        if n.type == ops.OPTYPE.PARAMETER and len(n.module.shape) == 3:
                            node.enable_index_mapping = False
                            break
                        else:
                            for ni in n.inputs:
                                if ni not in visited:
                                    stack.append(ni)
        return module2node

    def _trace_computational_graph(self, module2node, grad_fn_root, gradfn2module, gradfn2inshape, gradfn2outshape, reused, visited=set()):

        def create_node_if_not_exists(grad_fn):
            module = gradfn2module.get(grad_fn, None)
            inshape = gradfn2inshape.get(grad_fn, None)
            outshape = gradfn2outshape.get(grad_fn, None)

            # if module is not None and module in module2node and module not in reused:
            #     return module2node[module]

            # 1. link grad_fns and modules
            if module is None:  # a new module
                if not hasattr(grad_fn, "name"):
                    # we treat all unknwon modules as element-wise operations by default,
                    # which does not modify the #dimension/#channel of features.
                    # If you have some customized layers, please register it with DependencyGraph.register_customized_layer
                    module = ops._ElementWiseOp(self._op_id ,"Unknown")
                    self._op_id+=1
                    if self.verbose:
                        warnings.warn(
                            "[Warning] Unknown operation {} encountered, which will be handled as an element-wise op".format(
                                str(grad_fn))
                        )
                elif "catbackward" in grad_fn.name().lower():
                    module = ops._ConcatOp(self._op_id)
                    self._op_id+=1
                elif "split" in grad_fn.name().lower():
                    module = ops._SplitOp(self._op_id)
                    self._op_id+=1
                elif "unbind" in grad_fn.name().lower():
                    module = ops._UnbindOp(self._op_id)
                    self._op_id+=1
                elif 'reshape' in grad_fn.name().lower():
                    module = ops._ReshapeOp(self._op_id)
                    self._op_id+=1
                elif "view" in grad_fn.name().lower():
                    module = ops._ViewOp(self._op_id)
                    self._op_id+=1
                elif "addbackward" in grad_fn.name().lower():
                    module = ops._AddOp(self._op_id)
                    self._op_id+=1
                elif "mulbackward" in grad_fn.name().lower():
                    module = ops._MulOp(self._op_id)
                    self._op_id+=1
                elif "relubackward" in grad_fn.name().lower():
                    module = nn.ReLU(inplace=True)
                else:
                    # treate other ops as element-wise ones, like Add, Sub, Div, Mul.
                    module = ops._ElementWiseOp(self._op_id, grad_fn.name())
                    self._op_id+=1
                gradfn2module[grad_fn] = module

            # 2. link modules and nodes

            if grad_fn not in module2node:
                node = Node(
                    module=module,
                    grad_fn=grad_fn,
                    name=self._module2name.get(module, None),
                    inshape=inshape,
                    outshape=outshape
                )
                if (
                    type(module) in self.CUSTOMIZED_LAYERS
                ):  # mark it as a customized layer
                    node.type = ops.OPTYPE.CUSTOMIZED
                module2node[grad_fn] = node
            else:
                node = module2node[grad_fn]
            return node

        # non-recursive construction of computational graph
        processing_stack = [grad_fn_root]
        # 使用堆栈保存梯度函数，并不断弹出梯度函数，直到堆栈为空
        while len(processing_stack) > 0:
            grad_fn = processing_stack.pop(-1)
            if grad_fn in visited:
                continue
            node = create_node_if_not_exists(grad_fn=grad_fn)
            if grad_fn == grad_fn_root:
                self.tail_node = node
            if hasattr(grad_fn, "next_functions"):
                if grad_fn.next_functions[0][0] is None:
                    self.head_node = node
                for f in grad_fn.next_functions:
                    if f[0] is not None:
                        if (
                            hasattr(f[0], "name")
                            and "accumulategrad" in f[0].name().lower()
                        ):  # a leaf variable.
                            is_unwrapped_param = False
                            for (j, (p, dim)) in enumerate(self.unwrapped_parameters):
                                if f[0].variable is p:
                                    is_unwrapped_param = True
                                    gradfn2module[f[0]] = p
                                    self._module2name[p] = "UnwrappedParameter_{} ({})".format(j, p.shape)
                            if not is_unwrapped_param:
                                continue
                        input_node = create_node_if_not_exists(f[0])
                        node.add_input(input_node)
                        if node.inshape is None:
                            node.inshape = input_node.outshape
                        node.indegree += 1
                        input_node.outdegree += 1
                        input_node.add_output(node)
                        if input_node.outshape is None:
                            input_node.outshape = node.inshape
                        processing_stack.append(f[0])


            visited.add(grad_fn)

        
        for (param, dim) in self.unwrapped_parameters:
            module2node[param].pruning_dim = dim
        return module2node

    def _detect_unwrapped_parameters(self, unwrapped_parameters):
        # Detect wrapped nn.Parameters
        wrapped_parameters = []
        # prunable_module_types = self.REGISTERED_LAYERS
        for m in self.model.modules():
            op_type = ops.module2type(m)
            if op_type!=ops.OPTYPE.ELEMENTWISE or m.__class__ in self.CUSTOMIZED_LAYERS or m in self.CUSTOMIZED_LAYERS:
                wrapped_parameters.extend(list(m.parameters()))
       
        # Detect unwrapped nn.Parameters
        unwrapped_detected = []
        _param_to_name = {}
        for name, p in self.model.named_parameters():
            is_wrapped = False
            for p_wrapped in wrapped_parameters:
                if p is p_wrapped:
                    is_wrapped = True
                    break
            if not is_wrapped:
                unwrapped_detected.append(p)
                _param_to_name[p] = name
        if unwrapped_parameters is None:
            unwrapped_parameters = []
        unwrapped_detected = list( set(unwrapped_detected) - set([p for (p, _) in unwrapped_parameters]) )
        if len(unwrapped_detected)>0 and self.verbose:
            warning_str = "Unwrapped parameters detected: {}.\n Torch-Pruning will prune the last non-singleton dimension of these parameters. If you wish to change this behavior, please provide an unwrapped_parameters argument.".format([_param_to_name[p] for p in unwrapped_detected])
            warnings.warn(warning_str)

        return _param_to_name, unwrapped_parameters


# 2.foward the model with layer by layer
def forward_ll(dpg, x, ignored_layers=[]):
    # Step 1: create node for ignored layers
    if len(ignored_layers) > 0:
        ignored_layers_node = Node(module=ignored_layers[0])
        indegree = 0
        for m in ignored_layers:
            if len(m._modules) == 0:
                indegree += 1
        ignored_layers_node.indegree = indegree
    # Step 2: Initialize a queue with nodes having in-degree 0
    queue = deque([dpg.head_node])

    # Step 3: Perform topological sorting
    layer_topo = []
    while queue:
        
        # node = queue.popleft()
        # for correct sequence
        node = queue.pop()

        # Step 4: Update the in-degree of neighbors and enqueue if in-degree becomes 0
        for o_node in node.outputs:
            if o_node.module in ignored_layers:
                node.outputs.remove(o_node)
                node.outputs.append(ignored_layers_node)
                ignored_layers_node.inputs.append(node)
                ignored_layers_node.indegree -= 1
                if ignored_layers_node.indegree == 0:
                    queue.append(ignored_layers_node)
            else:
                o_node.indegree -= 1
                for i_node in o_node.inputs:
                    if i_node.indegree > 0 or i_node in queue:
                        break
                    # 如果遍历到最后一个输入节点，且所有输入节点入度为0，将其加入到队列中
                    if i_node == o_node.inputs[-1] and o_node not in queue:
                        o_node.indegree = 0
                        queue.append(o_node)


        # Step 5: forward the layer by layer
        if len(layer_topo) > 0 and layer_topo[-1] not in node.inputs:
            x = node.inputs[0].outresult
        
        # 将输入按序传入到节点中
        x_ = []
        for i_node in node.inputs:
            if i_node.outdegree > 0:
                if i_node.outresult is not None:
                    x_.append(i_node.outresult)
            else:
                x_.append(x)
        x = node.forward(x_ if len(x_) > 1 else x)
    
        # 将节点加入到layer_topo中
        layer_topo.append(node)
        # 判断当前节点的输出是否是下一个队列节点的输入
        if len(queue) > 0 and queue[-1] in node.outputs:
            node.outdegree -= 1
        # 出度大于0保存当前节点的输出结果
        if node.outdegree > 0:
            node.outresult = x
            # node.outresult = x.clone().detach()
        
        # 将当前节点的输入节点的出度减1，如果出度为0，将其输出结果置空
        for i_node in node.inputs:
            if i_node.outdegree > 0 and i_node != layer_topo[-2]:
                i_node.outdegree -= 1
                if i_node.outdegree == 0:
                    i_node.outresult = None
                

    # Step 6: Check if the graph is a DAG (no cycles)
    if layer_topo[-1] == dpg.tail_node:
        print("There exists a sorting in the DAG graph!")

    return x, layer_topo

# 3. draw computational graph
def draw_computational_graph(dpg, save_as, title='Computational Graph', figsize=(16, 16), dpi=200, cmap=None):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    n_nodes = len(dpg.module2node)
    module2idx = {m: i for (i, m) in enumerate(dpg.module2node.values())}
    G = np.zeros((n_nodes, n_nodes))
    fill_value = 1
    for module, node in dpg.module2node.items():
        # for input_node in node.inputs:
        #     G[module2idx[input_node], module2idx[node]] = fill_value
        #     G[module2idx[node], module2idx[input_node]] = fill_value
        for out_node in node.outputs:
            G[module2idx[out_node], module2idx[node]] = fill_value
            G[module2idx[node], module2idx[out_node]] = fill_value
        # pruner = dpg.get_pruner_of_module(module)
    fig, ax = plt.subplots(figsize=(figsize))
    ax.imshow(G, cmap=cmap if cmap is not None else plt.get_cmap('Blues'))
    plt.hlines(y=np.arange(0, n_nodes)+0.5, xmin=np.full(n_nodes, 0)-0.5, xmax=np.full(n_nodes, n_nodes)-0.5, color="#444444", linewidth=0.1)
    plt.vlines(x=np.arange(0, n_nodes)+0.5, ymin=np.full(n_nodes, 0)-0.5, ymax=np.full(n_nodes, n_nodes)-0.5, color="#444444", linewidth=0.1)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_as, dpi=dpi)
    return fig, ax

# 4. topo sort the graph
def topo_sorting(dpg, ignored_layers=[]):

    if len(ignored_layers) > 0:
        ignored_layers_node = Node(module=ignored_layers[0])
        indegree = 0
        for m in ignored_layers:
            if len(m._modules) == 0:
                indegree += 1
        ignored_layers_node.indegree = indegree
    # Step 2: Initialize a queue with nodes having in-degree 0
    queue = deque([dpg.head_node])

    # Step 3: Perform topological sorting
    layer_topo = []
    while queue:
        node = queue.popleft()

        layer_topo.append(node)
        if node == ignored_layers[0]:
            continue

        # Step 5: Update the in-degree of neighbors and enqueue if in-degree becomes 0
        for o_node in node.outputs:
            if o_node.module in ignored_layers:
                node.outputs.remove(o_node)
                node.outputs.append(ignored_layers_node)
                ignored_layers_node.inputs.append(node)
                ignored_layers_node.indegree -= 1
                if ignored_layers_node.indegree == 0:
                    queue.append(ignored_layers_node)
            else:
                o_node.indegree -= 1
                for i_node in o_node.inputs:
                    if i_node.indegree > 0:
                        break
                    # 如果遍历到最后一个输入节点，且所有输入节点入度为0，将其加入到队列中
                    if i_node == o_node.inputs[-1] and o_node not in queue:
                        o_node.indegree = 0
                        queue.append(o_node)
                

    # Step 6: Check if the graph is a DAG (no cycles)
    if layer_topo[-1] == dpg.tail_node or layer_topo[-1] == ignored_layers[0]:
        print("There exists a sorting in the DAG graph!")

    return layer_topo

            
