import torch
from torchvision import models

def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_model(output_shape, device, seed_fn=set_seeds):
    if seed_fn:
        seed_fn()

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = True

    num_ftrs = model.classifier[1].in_features

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=num_ftrs, out_features=512, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=512, out_features=128, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=output_shape, bias=True)
    ).to(device)

    return model
