# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
1. Develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron.  
2. Train the model on a dataset containing images of various defected and non-defected capacitors to improve defect detection accuracy.  
3. Optimize and evaluate the model to ensure reliable classification for capacitor quality assessment in manufacturing.


## DESIGN STEPS
### STEP 1:
Collect and preprocess the dataset containing images of defected and non-defected capacitors.

### STEP 2:
Split the dataset into training, validation, and test sets.

### STEP 3:
Load the pretrained VGG19 model with weights from ImageNet.

### STEP 4:
Remove the original fully connected (FC) layers and replace the last layer with a single neuron (1 output) with a Sigmoid activation function for binary classification.

### STEP 5:
Train the model using binary cross-entropy loss function and Adam optimizer.

### STEP 6:
Evaluate the model with test data loader and intepret the evaluation metrics such as confusion matrix and classification report.

## PROGRAM
Include your code here
```python

# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False

# Modify the final fully connected layer to match the dataset classes

num_classes = len(train_dataset.classes)

in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)

# Include the Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the model
## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))


```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="786" height="273" alt="image" src="https://github.com/user-attachments/assets/fb999ae7-ed4b-4968-a133-79c781eb25c1" />

<img width="876" height="659" alt="image" src="https://github.com/user-attachments/assets/f3b7646b-f73f-49aa-9e54-7f9d414a6184" />

### Confusion Matrix
<img width="829" height="679" alt="image" src="https://github.com/user-attachments/assets/d6892e8c-b6e4-49b3-b6c0-b98182c3519e" />

### Classification Report

<img width="952" height="688" alt="image" src="https://github.com/user-attachments/assets/c823fe80-e82e-463d-b2e4-6236e8d53033" />

### New Sample Prediction

<img width="832" height="526" alt="image" src="https://github.com/user-attachments/assets/f79d6fc7-1cb2-4eea-ac29-0ce7c2589d48" />
<img width="722" height="508" alt="image" src="https://github.com/user-attachments/assets/3bcf4036-0e3f-4d7f-8a64-1b22756062d1" />


## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors
