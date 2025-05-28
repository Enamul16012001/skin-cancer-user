import os
import numpy as np
import torch
from torch import nn
from timeit import default_timer as timer
from torchvision import datasets, transforms

from utils import (
    create_dataloaders,
    set_seeds,
    CustomCallback,
    train_step,
    val_step,
    test_step,
    train,
    train_transforms,
    test_transforms
)

from model import create_model

NUM_WORKERS = os.cpu_count()

image_size = (224, 224)
height, width = image_size
batch_size = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = './Dataset/train'
test_dir = './Dataset/test'

train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    train_transform=train_transforms,
                                                                    test_transform=test_transforms,
                                                                    batch_size=batch_size)

output_shape = 7

model = create_model(output_shape=output_shape, device=device)


initial_lr = 0.0001
weight_decay = 0.0001
factor = 0.04
c_check = False
callback = CustomCallback(initial_lr=initial_lr, factor=factor, c_check=c_check)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

start_time = timer()
results = train(model=model,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=8,
                       device=device,
                       callback=callback)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

results = test_step(model, test_dataloader, loss_fn, device)


