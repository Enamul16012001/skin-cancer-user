import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score
)
from sklearn.preprocessing import label_binarize


def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def balance_dataset(subset):
    # Access the dataset and targets from the Subset object
    dataset = subset.dataset
    subset_indices = subset.indices
    subset_targets = [dataset.targets[i] for i in subset_indices]

    # Count the instances per class
    class_counts = Counter(subset_targets)
    print(class_counts)

    balanced_indices = []
    for class_index, count in class_counts.items():
        # Determine the required number of images based on the given rules
        if count >= 1000:
            required_images = 3000
        elif 500 < count < 1000:
            required_images = 2500
        else:
            required_images = 2000

        # Collect indices of the current class
        class_indices = [subset_indices[i] for i, label in enumerate(subset_targets) if label == class_index]

        # Balance the class to the required number of images
        if count < required_images:
            additional_indices = np.random.choice(class_indices, required_images - count, replace=True)
            class_indices.extend(additional_indices)
        elif count > required_images:
            class_indices = np.random.choice(class_indices, required_images, replace=False).tolist()

        balanced_indices.extend(class_indices)

    return Subset(dataset, balanced_indices)


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int=os.cpu_count()
):

  full_train_data = datasets.ImageFolder(train_dir, transform=train_transform)
  test_data = datasets.ImageFolder(test_dir, transform=test_transform)

  train_indices, val_indices = train_test_split(list(range(len(full_train_data))), test_size=0.1, random_state=42)
  imbalanced_train_data = Subset(full_train_data, train_indices)
  val_data = Subset(full_train_data, val_indices)

  # Balance the training data
  train_data = balance_dataset(imbalanced_train_data)

  # Get class names
  class_names = full_train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, val_dataloader, test_dataloader, class_names


class CustomCallback:
    def __init__(self, initial_lr: float, factor: float = 0.04, c_check: bool = False):
        self.initial_lr = initial_lr
        self.factor = factor
        self.c_check = c_check
        self.best_loss = float('inf')
        self.best_weights = None
        self.lr = initial_lr

    def on_epoch_end(self, model: torch.nn.Module, val_loss: float, optimizer: torch.optim.Optimizer):
        if self.c_check:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_weights = model.state_dict()
            else:
                print("Validation loss increased. Resetting model weights and reducing learning rate.")
                model.load_state_dict(self.best_weights)
                self.lr *= self.factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr
                print(f"New learning rate: {self.lr}")
        return self.lr

    def on_train_end(self, model: torch.nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
        return model


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

  model.train()

  train_loss, train_acc = 0, 0

  batch_loop = tqdm(dataloader, desc="Training", leave=False)
  for batch, (X, y) in enumerate(batch_loop):
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item()

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def val_step(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            device: torch.device) -> Tuple[float, float]:

  # Put model in eval mode

  model.eval()

  # Setup test loss and test accuracy values
  val_loss, val_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          val_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(val_pred_logits, y)
          val_loss += loss.item()

          # Calculate and accumulate accuracy
          val_pred_labels = val_pred_logits.argmax(dim=1)
          val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch
  val_loss = val_loss / len(dataloader)
  val_acc = val_acc / len(dataloader)
  return val_loss, val_acc

def test_step(model, test_dataloader, loss_fn, device):
    model.eval()
    
    test_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            test_pred_labels = test_pred_logits.argmax(dim=1)
            total_correct += (test_pred_labels == y).sum().item()
            total_samples += len(test_pred_labels)
            
            all_preds.extend(test_pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            probs = torch.softmax(test_pred_logits, dim=1)
            all_probs.extend(probs.cpu().numpy())

    test_loss = test_loss / len(test_dataloader)
    test_acc = total_correct / total_samples

    overall_accuracy = accuracy_score(all_labels, all_preds)
    macro_accuracy = recall_score(all_labels, all_preds, average="macro")
    weighted_accuracy = recall_score(all_labels, all_preds, average="weighted")

    num_classes = len(np.unique(all_labels))
    if num_classes > 2:
        all_labels_bin = label_binarize(all_labels, classes=np.arange(num_classes))
        auc = roc_auc_score(all_labels_bin, np.array(all_probs), average="macro", multi_class="ovr")
    else:
        auc = roc_auc_score(all_labels, np.array(all_probs)[:,1])  # assumes positive class is at index 1

    sensitivity = recall_score(all_labels, all_preds, average="macro")
    specificity = precision_score(all_labels, all_preds, average="macro")  # Note: not true specificity

    conf_matrix = confusion_matrix(all_labels, all_preds)
    conf_matrix_display = ConfusionMatrixDisplay(conf_matrix)
    classification_rep = classification_report(all_labels, all_preds)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Overall Accuracy (sklearn): {overall_accuracy:.4f}")
    print(f"Macro Accuracy: {macro_accuracy:.4f}")
    print(f"Weighted Accuracy: {weighted_accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity (Precision Approx.): {specificity:.4f}")
    print("\nClassification Report:\n", classification_rep)

    conf_matrix_display.plot()
    
    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "overall_accuracy": overall_accuracy,
        "macro_accuracy": macro_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": conf_matrix,
        "classification_report": classification_rep
    }



def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          callback: CustomCallback) -> Dict[str, List]:

  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": []
  }
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs), desc="Epochs"):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)

      val_loss, val_acc = val_step(model=model,
          dataloader=val_dataloader,
          loss_fn=loss_fn,
          device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_acc: {val_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["val_loss"].append(val_loss)
      results["val_acc"].append(val_acc)

      # Step the scheduler
      scheduler.step()

      #new_lr = callback.on_epoch_end(model, val_loss, optimizer)

  model = callback.on_train_end(model)

  # Return the filled results at the end of the epochs
  return results


image_size = (224, 224)

# Define test transforms
test_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define train transforms
train_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-15, 15)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])