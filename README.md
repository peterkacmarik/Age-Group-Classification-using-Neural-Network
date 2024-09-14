# Age Group Multi-Class Classification using Neural Network

This project involves building and training a neural network to classify age groups based on customer data. The project utilizes PyTorch for model building and training, and includes data preprocessing, model evaluation, and prediction functionalities.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Data](#data)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Usage](#usage)
- [License](#license)

## Project Overview
This project aims to classify customer age groups into predefined categories: 
- Youth (<25)
- Young Adults (25-34)
- Adults (35-64)
- Seniors (64+)

It includes:
- Data preprocessing and encoding
- Neural network architecture definition
- Training and evaluation of the model
- Making predictions on new data

## Requirements
- Python 3.x
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

You can install the required packages using pip:

```bash
pip install torch pandas numpy matplotlib scikit-learn
```
## Data
The dataset used in this project contains the following columns:

Customer_Age,
Customer_Gender,
Country,
State,
Product_Category,
Order_Quantity,
Profit,
Revenue.
## Model
### Neural Network Architecture
The neural network model AgeGroupNN is defined with the following layers:

Input layer,
Two hidden layers with batch normalization and dropout,
Output layer with 4 units corresponding to the age groups.
```bash
class AgeGroupNN(nn.Module):
    def __init__(self, in_features=8, hl1=32, hl2=16, out_features=4):
        super(AgeGroupNN, self).__init__()
        self.fc3 = nn.Linear(in_features, hl1)
        self.bn3 = nn.BatchNorm1d(hl1)
        self.fc4 = nn.Linear(hl1, hl2)
        self.bn4 = nn.BatchNorm1d(hl2)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(hl2, out_features)

    def forward(self, x):
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x
```
## Training
### Training Process
The training process involves:

Training the model for a specified number of epochs,
Implementing early stopping to prevent overfitting,
Tracking training and validation losses and accuracies.
```bash
def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    criterion, 
    optimizer, 
    num_epochs = 100,
    patience: int = 10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Training code here
```
## Evaluation
### Evaluation Metrics
The model's performance is evaluated using loss and accuracy metrics for both training and validation datasets. Results are visualized using matplotlib.
```bash
def plot_results(train_losses, train_accuracies, val_losses, val_accuracies):
    # Plotting code here
```
## Prediction
### Predict Function
The predict_age_group function allows you to make predictions on new data.
```bash
def predict_age_group(model, input_data, scaler, ordinal_encoder):
    # Prediction code here
```
### Generating Sample Data
You can generate random sample data for testing the prediction function.
```bash
def generate_input_data():
    # Data generation code here
```
## Usage
1. Generate Input Data:
```bash
input_data = generate_input_data()
```
2. Make Predictions:
```bash
predictions = predict_age_group(model, input_data, scaler, ordinal_encoder)
```
3. Visualize Results:
```bash
plot_results(train_losses, train_accuracies, val_losses, val_accuracies)
```
## License
This project is licensed under the MIT License - see the LICENSE file for details.
