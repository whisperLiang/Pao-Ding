import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

# torchvision==0.13.1

###########################################
# Splitable Models
############################################

# from torchvision.models.vision_transformer import (
#     vit_b_16,
#     vit_b_32,
#     vit_l_16,
#     vit_l_32,
#     vit_h_14,
# )

from torchvision.models.alexnet import alexnet
from torchvision.models.convnext import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
)
from torchvision.models.densenet import (
    densenet121,
    densenet169,
    densenet201,
    densenet161,
)
from torchvision.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
)
from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models.regnet import (
    regnet_y_400mf,
    regnet_y_800mf,
    regnet_y_1_6gf,
    regnet_y_3_2gf,
    regnet_y_8gf,
    regnet_y_16gf,
    regnet_y_32gf,
    regnet_y_128gf,
)
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.vgg import (
    vgg11,
    vgg13,
    vgg16,
    vgg19,
    vgg11_bn,
    vgg13_bn,
    vgg16_bn,
    vgg19_bn,
)



###########################################
# Failue cases in this script
############################################
# from torchvision.models.swin_transformer import swin_t, swin_s, swin_b # TODO: support Swin ops 
from torchvision.models.shufflenetv2 import ( # TODO: support channel shuffling
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)


if __name__ == "__main__":

    entries = globals().copy()

    import torch
    import model_split as ms

    def my_split(model, example_inputs, model_name):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.cpu().eval()
        ignored_layers = []
        for p in model.parameters():
            p.requires_grad_(True)

        dpg = ms.DependencyGraph().build_dependency(model, example_inputs, IGNORED_LAYERS=ignored_layers)

        #########################################
        print("==============Before split=================")
        print("Model Name: {}".format(model_name))
        out_before = model(example_inputs)
        # print(f"before split: {out_before}")

        print("==============After split=================")

        #########################################
        # Testing 
        #########################################
        out_after, layer_topo = ms.forward_ll(dpg, example_inputs, ignored_blocks=ignored_layers)
        if torch.allclose(out_before, out_after):
            print(f"{model_name} splits success!")
            fig, ax = ms.draw_computational_graph(layer_topo, save_as=f'computational_graph/{model_name}_computational_graph.png', title='Computational Dependency Graph', figsize=(16, 16), cmap=None)
            return True
        else:
            print(f"{model_name} splits failed!")
            return False

    successful = []
    unsuccessful = []
    for model_name, entry in entries.items():
        if 'swin' in model_name.lower() or 'shufflenet' in model_name.lower() or 'raft' in model_name.lower(): # stuck or 'raft' in model_name.lower() 
            unsuccessful.append(model_name)
            continue

        if not callable(entry):
            continue
        if "inception" in model_name:
            example_inputs = torch.randn(1, 3, 299, 299)
        else:
            example_inputs = torch.randn(1, 3, 224, 224)

        if "googlenet" in model_name or "inception" in model_name:
            model = entry(aux_logits=False)
        else:
            model = entry()

        success_flag = my_split(
            model, example_inputs=example_inputs, model_name=model_name
        )
        if success_flag:
            successful.append(model_name)
        else:
            unsuccessful.append(model_name)
        print("Successful split: %d Models\n"%(len(successful)), successful)
        print("")
        print("Unsuccessful split: %d Models\n"%(len(unsuccessful)), unsuccessful)
        sys.stdout.flush()

print("Finished!")

print("Successful split: %d Models\n"%(len(successful)), successful)
print("")
print("Unsuccessful split: %d Models\n"%(len(unsuccessful)), unsuccessful)