import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # Ensure WeightedRandomSampler is included
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B2_Weights
import os
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
train_dir = './Skin cancer ISIC The International Skin Imaging Collaboration/train'
test_dir = './Skin cancer ISIC The International Skin Imaging Collaboration/test'
classes = ["melanoma", "nevus", "basal cell carcinoma", "actinic keratosis", "benign keratosis", "vascular lesion"]

# Verify dataset paths
def verify_dataset_paths(classes, train_dir):
    for class_name in classes:
        class_path = os.path.join(train_dir, class_name)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"Error: Directory '{class_path}' not found. Please check dataset structure.")

verify_dataset_paths(classes, train_dir)

# Data Augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class SkinCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            raise
        if self.transform:
            image = self.transform(image)
        return image, label

# Load dataset
def load_dataset(train_dir, classes, test_size=0.2, random_state=42):
    all_image_paths, all_labels = [], []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                all_image_paths.append(os.path.join(class_dir, img_name))
                all_labels.append(label)
    return train_test_split(all_image_paths, all_labels, test_size=test_size, stratify=all_labels, random_state=random_state)

train_paths, test_paths, train_labels, test_labels = load_dataset(train_dir, classes)

# Create datasets and dataloaders
train_dataset = SkinCancerDataset(train_paths, train_labels, transform=transform_train)
test_dataset = SkinCancerDataset(test_paths, test_labels, transform=transform_test)

# Handle class imbalance
class_counts = Counter(train_labels)
weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

batch_size = 32  # Reduced batch size to prevent CUDA out of memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_B2_Weights.DEFAULT
model = models.efficientnet_b2(weights=weights)

# Fine-tune some layers
for param in model.features[-3:].parameters():
    param.requires_grad = True

model.classifier[1] = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, len(classes)),
    nn.Dropout(0.3)  # Reduced dropout rate
)
model = model.to(device)

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Higher initial learning rate
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

# Training loop with early stopping
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, patience=15):
    best_test_acc = 0
    epochs_no_improve = 0
    train_accs, train_losses, test_accs, test_losses = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate on test set
        test_loss, test_acc, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Step the scheduler
        scheduler.step(test_loss)

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}")

        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_skin_cancer_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                logging.info("Early stopping triggered")
                break

    plot_metrics(train_accs, test_accs, train_losses, test_losses)
    generate_confusion_matrix(all_labels, all_preds)
    plot_roc_auc(all_labels, all_probs)

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    return test_loss, test_acc, all_preds, all_labels, all_probs

def plot_metrics(train_accs, test_accs, train_losses, test_losses):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Test Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Test Loss')
    plt.show()

def generate_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

def plot_roc_auc(all_labels, all_probs):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Binarize the output
    from sklearn.preprocessing import label_binarize
    y_true = label_binarize(all_labels, classes=list(range(len(classes))))
    n_classes = y_true.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        if np.sum(y_true[:, i]) == 0:
            logging.warning(f"No samples for class {classes[i]}. Skipping ROC/AUC calculation.")
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Print ROC values
    for i in range(n_classes):
        if i in roc_auc:
            print(f"Class {classes[i]} ROC Values:")
            print(f"False Positive Rate (FPR): {fpr[i]}")
            print(f"True Positive Rate (TPR): {tpr[i]}")
            print(f"AUC: {roc_auc[i]:.2f}\n")

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, color in zip(range(n_classes), colors):
        if i in roc_auc:
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Function to augment test data
def augment_test_data(test_dir, classes, num_augmented_images=100):
    augmented_dir = os.path.join(test_dir, "augmented")
    os.makedirs(augmented_dir, exist_ok=True)

    transform_augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        augmented_class_dir = os.path.join(augmented_dir, class_name)
        os.makedirs(augmented_class_dir, exist_ok=True)

        images = os.listdir(class_dir)
        for i in range(num_augmented_images):
            img_name = images[i % len(images)]  # Cycle through existing images
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                augmented_img = transform_augment(img)
                save_path = os.path.join(augmented_class_dir, f"aug_{i}_{img_name}")
                # Convert tensor back to PIL image for saving
                augmented_img_pil = transforms.ToPILImage()(augmented_img)
                augmented_img_pil.save(save_path)
            except Exception as e:
                logging.error(f"Error augmenting image {img_path}: {e}")

    logging.info(f"Augmented {num_augmented_images} images per class in the test folder.")

if __name__ == '__main__':
    # Augment test data
    augment_test_data(test_dir, classes, num_augmented_images=100)

    # Reload dataset after augmentation
    train_paths, test_paths, train_labels, test_labels = load_dataset(train_dir, classes)
    test_dataset = SkinCancerDataset(test_paths, test_labels, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50)
