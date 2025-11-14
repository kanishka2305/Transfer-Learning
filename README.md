# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop a deep learning model for image classification using transfer learning. Utilize the pre-trained VGG19 model as the feature extractor, fine-tune it, and adapt it to classify images into specific categories.

## DESIGN STEPS
### STEP 1:Import Libraries and Load Dataset
Import the necessary libraries.
Load the dataset.
Split the dataset into training and testing sets

### STEP 2:Initialize Model, Loss Function, and Optimizer
Define the model architecture.
Use CrossEntropyLoss for multi-class classification.
Choose the Adam optimizer for efficient training.

### STEP 3:Train the Model
Train the model using the training dataset.
Optimize the model parameters to minimize the loss.

### Step 4: Evaluate the Model
Test the model using the testing dataset.
Measure performance using appropriate evaluation metrics.

### Step 5: Make Predictions on New Data
Use the trained model to predict outcomes for new inputs.

## PROGRAM

### Load Pretrained Model and Modify for Transfer Learning
```py
model = models.vgg19(pretrained=True)
```

### Modify the final fully connected layer to match the dataset classes
```py
model.classifier[6] = nn.Linear(4096, num_classes)
```
### Include the Loss function and optimizer
```py
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
# Train the model
```py
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
            # Convert labels to one-hot encoding
            labels = nn.functional.one_hot(labels, num_classes=num_classes).float().to(device) # Change this line
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # Convert labels to one-hot encoding
                labels = nn.functional.one_hot(labels, num_classes=num_classes).float().to(device) # Change this line
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Kanishka V S")
    print("Register Number:212222230061")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="662" height="643" alt="image" src="https://github.com/user-attachments/assets/de053374-6558-44ec-b74d-919e098d6619" />

### Confusion Matrix
<img width="602" height="501" alt="image" src="https://github.com/user-attachments/assets/85f3910b-154a-4781-b4a3-35bd24e78523" />

### Classification Report

<img width="440" height="177" alt="image" src="https://github.com/user-attachments/assets/d4d109a5-238e-4803-8614-3a16a587cfda" />

### New Sample Prediction
<img width="356" height="340" alt="image" src="https://github.com/user-attachments/assets/d8872775-40e0-401b-b471-c88dd8cfef02" />


## RESULT

Thus the experiment was executed Successfully.
