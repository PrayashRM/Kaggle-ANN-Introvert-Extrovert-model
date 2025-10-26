<h1 align="center">🧠 Introvert vs Extrovert Classification</h1>

<p align="center">
  <b>Deep Learning | Personality Prediction | PyTorch | Neural Networks</b><br>
  Artificial Neural Network built <b>from scratch</b> using <b>PyTorch</b> for personality classification with <b>98.14% accuracy</b> 🎯
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Accuracy-98.14%25-brightgreen" alt="Accuracy">
  <img src="https://img.shields.io/badge/Optuna-Optimized-purple" alt="Optuna">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</p>

<hr>

## 🚀 Overview

A sophisticated Artificial Neural Network (ANN) built from scratch using **PyTorch** to classify personality types as either **Introvert** or **Extrovert** based on behavioral features. This project was developed for a **Kaggle competition** and demonstrates advanced techniques including hyperparameter optimization with **Optuna**, custom data preprocessing, and strategic model architecture design.

<hr>

## 🎯 Key Features

- ✨ **Custom ANN Architecture** - 2-layer neural network designed specifically for personality classification
- 🔧 **Hyperparameter Optimization** - Systematic tuning using Optuna (1000+ trials)
- 📊 **Smart Data Preprocessing** - Class-wise mean imputation for missing values
- 🎯 **High Accuracy** - Achieved 98.14% accuracy on test set
- ⏸️ **Early Stopping Implementation** - Prevents overfitting with patience-based monitoring
- 💾 **Model Checkpointing** - Automatic saving of best performing models
- 📈 **Comprehensive Evaluation** - F1 Score: 0.9725
- 🔥 **GPU Acceleration** - CUDA support for faster training

<hr>

## 📊 Dataset & Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 98.14% |
| **F1 Score (Weighted)** | 0.9725 |
| **Validation Accuracy** | 97.25% |
| **Training Samples** | 18,524 |
| **Test Samples** | 6,175 |
| **Features** | 7 behavioral indicators |

### Class Distribution:
- **Extroverts:** 13,699 samples (73.9%)
- **Introverts:** 4,825 samples (26.1%)

<hr>

## 🧩 Features Used

The model analyzes **7 carefully selected behavioral features** to predict personality type:

| Feature | Description | Range |
|---------|-------------|-------|
| **Time Spent Alone** | Hours per day spent alone | 0-11 |
| **Stage Fear** | Presence of public speaking anxiety | Binary (Yes/No) |
| **Social Event Attendance** | Frequency of attending social gatherings | 0-10 |
| **Going Outside** | Days per week spent outdoors | 0-7 |
| **Drained After Socializing** | Feeling exhausted after social interaction | Binary (Yes/No) |
| **Friends Circle Size** | Number of close friends | 0-15 |
| **Post Frequency** | Social media posting activity | 0-10 |

<hr>

## 🏗️ Model Architecture

Optimized 2-layer neural network with batch normalization and dropout regularization:

```python
class NNArch(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1
        self.fc1 = nn.Linear(7, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.114)

        # Layer 2
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.233)

        # Output Layer
        self.out = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)

        x = self.out(x)
        return x
```

### Architecture Flow:
- **Input Layer:** 7 behavioral features
- **Hidden Layer 1:** 64 neurons + BatchNorm + ReLU + Dropout(11.4%)
- **Hidden Layer 2:** 128 neurons + BatchNorm + ReLU + Dropout(23.3%)
- **Output Layer:** 2 classes (Introvert/Extrovert)
- **Total Parameters:** ~11,000

<hr>

## 🔧 Data Preprocessing Pipeline

### 1. Missing Value Handling
**Smart Class-wise Imputation:**

```python
# For training set: class-wise means
for feature in features:
    mean_intro = train[train['Personality'] == 1][feature].mean()
    mean_extro = train[train['Personality'] == 0][feature].mean()
    
    train.loc[(train['Personality'] == 1) & (train[feature].isnull()), feature] = mean_intro
    train.loc[(train['Personality'] == 0) & (train[feature].isnull()), feature] = mean_extro

# For test set: average of both class means
    avg_mean = (mean_intro + mean_extro) / 2
    test[feature] = test[feature].fillna(avg_mean)
```

### 2. Feature Encoding

| Feature | Original | Encoded |
|---------|----------|---------|
| Stage Fear | No / Yes | 0 / 1 |
| Drained After Socializing | No / Yes | 0 / 1 |
| Personality | Extrovert / Introvert | 0 / 1 |

### 3. Min-Max Normalization

```python
for feature in ['Time_spent_Alone', 'Social_event_attendance', 
                'Going_outside', 'Friends_circle_size', 'Post_frequency']:
    min_val = train[feature].min()
    max_val = train[feature].max()
    train[feature] = (train[feature] - min_val) / (max_val - min_val)
    test[feature] = (test[feature] - min_val) / (max_val - min_val)
```

<hr>

## ⚙️ Training Configuration

### Optimal Hyperparameters (Found via Optuna):

| Parameter | Value |
|-----------|-------|
| **Learning Rate** | 0.00133 |
| **Weight Decay** | 1e-05 |
| **Batch Size** | 128 |
| **Optimizer** | Adam |
| **Loss Function** | CrossEntropyLoss |
| **Activation** | ReLU |
| **Max Epochs** | 6000 |
| **Early Stopping Patience** | 7 epochs |

### Data Split:
- **Training Set:** 70% (~12,966 samples)
- **Validation Set:** 21% (~3,890 samples)
- **Test Set:** 9% (~1,668 samples)

<hr>

## 🎨 Hyperparameter Optimization with Optuna

Conducted **1000+ trials** to find optimal configuration:

### Search Space:

```python
trial.suggest_int('num_of_layers', 1, 3)
trial.suggest_categorical('hidden1', [64, 128, 256])
trial.suggest_categorical('hidden2', [64, 128, 256])
trial.suggest_categorical('hidden3', [16, 32, 64])
trial.suggest_float('dropout1', 0.1, 0.5)
trial.suggest_float('dropout2', 0.1, 0.5)
trial.suggest_float('dropout3', 0.1, 0.5)
trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
trial.suggest_categorical('weight_decay', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
trial.suggest_categorical('batch_size', [16, 32, 64, 128])
trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
trial.suggest_categorical('activation', ['relu', 'leakyrelu', 'elu'])
```

### Best Trial Results:
- 🏆 **Accuracy:** 98.14%
- 🎯 **Architecture:** 2 layers (7 → 64 → 128 → 2)
- ⚡ **Optimizer:** Adam
- 🔥 **Activation:** ReLU

<hr>

## 🛠️ Technologies & Dependencies

### Core Libraries:

```python
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna

# Google Colab specific
from google.colab import drive, files
```

### Required Packages:
🔥 PyTorch • 📊 Pandas • 🧮 NumPy • 📈 Scikit-learn • 🎨 Matplotlib • 🔧 Optuna

<hr>

## 📦 Installation & Setup

### 1. Install Dependencies

```bash
pip install torch pandas numpy scikit-learn matplotlib optuna
```

### 2. For Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device available:", device)
```

### 3. Clone Repository

```bash
git clone https://github.com/yourusername/introvert-extrovert-classifier.git
cd introvert-extrovert-classifier
```

<hr>

## 🚀 Usage

### 1. Training the Model

```python
# Initialize model
model = NNArch()
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.00133, weight_decay=1e-05)

# Train with early stopping
# See notebook for complete training loop
```

### 2. Load Pre-trained Model

```python
# Load best saved model
model = NNArch()
model.load_state_dict(torch.load('best_model_optuna.pth'))
model.to(device)
model.eval()
```

### 3. Make Predictions

```python
# Prepare input data (normalized and encoded)
x_tensor = torch.tensor(input_data, dtype=torch.float32)
x_tensor = x_tensor.to(device)

with torch.no_grad():
    predictions = model(x_tensor)
    predicted_class = torch.argmax(predictions, dim=1)
    
    # 0 = Extrovert, 1 = Introvert
    personality = 'Extrovert' if predicted_class == 0 else 'Introvert'
    print(f"Predicted Personality: {personality}")
```

<hr>

## 📁 Project Structure

```
Introvert-Extrovert-Classification/
│
├── Introvert_Extrovert_prediction.ipynb   # Complete implementation
├── README.md                               # Documentation
│
├── data/
│   ├── train.csv                          # Training dataset
│   └── test.csv                           # Test dataset
│
├── Saved_Models/
│   ├── Project08_Introvert_Extrovert_bestmodel_optuna.pth    # Best model
│   └── Project08_Introvert_Extrovert_currentmodel.pth        # Latest checkpoint
│
├── Introvert_Extrovert_Project08_study.db   # Optuna study database
├── submission.csv                            # Kaggle submission
└── requirements.txt                          # Dependencies
```

<hr>

## 🔑 Key Implementation Details

### Custom Dataset Class

```python
class customdataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.Y = torch.tensor(Y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
```

### Early Stopping Mechanism

```python
best_val_loss = float('inf')
patience = 7
wait = 0

for epoch in range(epochs):
    # Training and validation...
    
    if avg_validation_loss < best_val_loss:
        best_val_loss = avg_validation_loss
        wait = 0
        torch.save(model.state_dict(), current_model_path)
    else:
        wait += 1
        if wait >= patience:
            print(f"⛔ Early stopping at epoch {epoch + 1}")
            break
```

### Device Configuration

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device available:", device)

# Move model and data to GPU
model.to(device)
x_batch = x_batch.to(device).float()
y_batch = y_batch.to(device).long()
```

<hr>

## 📈 Training Progress

### Training Metrics:
- **Epochs Trained:** 13 (out of max 6000)
- **Early Stopping Triggered:** Epoch 13
- **Final Training Loss:** ~0.088
- **Final Validation Loss:** ~0.111
- **Test Loss:** 0.1145
- **Test Accuracy:** 98.14%

The model converged quickly thanks to optimal hyperparameters found through Optuna optimization.

<hr>

## 🎯 Model Performance Analysis

### Strengths:
- ✅ Exceptional accuracy (98.14%) on unseen test data
- ✅ High F1 score (0.9725) indicating balanced precision and recall
- ✅ Fast convergence with early stopping (13 epochs)
- ✅ Handles class imbalance effectively
- ✅ Lightweight architecture (~11K parameters)
- ✅ Low overfitting risk with batch normalization and dropout
- ✅ Robust preprocessing pipeline

### Technical Highlights:
- 🎯 Class-wise mean imputation improved model generalization
- 📊 Min-max normalization ensured feature scale consistency
- 🔄 Batch normalization stabilized training
- ⚡ Adam optimizer with optimal learning rate
- 🏆 Systematic hyperparameter tuning with Optuna

<hr>

## 🔮 Future Enhancements

- 🚀 **Web Deployment:** Flask/Streamlit app with interactive questionnaire
- 📱 **Mobile App:** Convert to ONNX/TFLite format for mobile deployment
- 🎨 **Feature Engineering:** Explore interaction features and polynomial terms
- 🏗️ **Advanced Architectures:** Experiment with attention mechanisms
- 📊 **Ensemble Methods:** Combine multiple models for higher accuracy
- 🔄 **Cross-validation:** K-fold CV for more robust evaluation
- 💡 **Explainability:** SHAP values to understand feature importance
- 🌐 **Multi-class Extension:** Classify into MBTI personality types

<hr>

## 📝 Notebook Features

The complete Jupyter notebook includes:

1. 📥 **Data Loading:** CSV import and initial exploration
2. 🔍 **Data Analysis:** Statistics and distribution analysis
3. 🧹 **Data Cleaning:** Missing value handling and duplicate removal
4. 🎨 **Preprocessing:** Encoding, normalization, and splitting
5. 🏗️ **Model Definition:** Custom ANN architecture
6. 🎓 **Training Loop:** With early stopping and checkpointing
7. 📊 **Evaluation:** Comprehensive metrics (accuracy, F1 score)
8. 📈 **Visualization:** Loss curves over epochs
9. 🔧 **Hyperparameter Tuning:** Optuna optimization (1000+ trials)
10. 💾 **Model Persistence:** Save and load functionality
11. 🧪 **Kaggle Submission:** Test set prediction and CSV export

<hr>

## ⚠️ Important Notes

- 🔧 **GPU Recommended:** CPU training will be significantly slower
- 💾 **Google Drive:** Models are saved in Google Drive when using Colab
- 📏 **Input Requirements:** Features must be preprocessed (normalized & encoded)
- 🎯 **Binary Classification:** Model outputs Introvert (1) or Extrovert (0)
- 🔄 **Preprocessing Consistency:** New data needs same transformations as training
- 📊 **Class Imbalance:** Model trained on imbalanced data (73.9% Extroverts)

<hr>

## 🐛 Troubleshooting

### Common Issues:

**Issue:** CUDA out of memory
```python
# Solution: Reduce batch size
batch_size = 64  # Instead of 128
```

**Issue:** Model not loading
```python
# Solution: Check device consistency
model = NNArch()
model.load_state_dict(torch.load('model.pth', map_location=device))
```

**Issue:** Poor predictions on new data
```python
# Solution: Ensure proper preprocessing
# 1. Encode categorical features (Stage_fear, Drained_after_socializing)
# 2. Normalize continuous features using training set min/max
# 3. Handle missing values with appropriate strategy
```

**Issue:** Optuna study not loading
```python
# Solution: Check database path and permissions
study = optuna.create_study(
    direction='maximize',
    study_name='Introvert_Extrovert_Project08_study',
    storage='sqlite:///path/to/study.db',
    load_if_exists=True
)
```

<hr>

## 📚 Learning Resources

- 📖 [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- 📖 [Optuna Documentation](https://optuna.readthedocs.io/)
- 📖 [Scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- 📖 [Kaggle Learn](https://www.kaggle.com/learn)

<hr>

## 👨‍💻 Author

<p align="center">
  <b>Prayash Ranjan Mohanty</b><br>
  B.Tech in Computer Science (AI & ML)<br>
  Kalinga Institute of Industrial Technology, Bhubaneswar<br>
  📧 <a href="mailto:prayashranjanmohanty11@gmail.com">prayashranjanmohanty11@gmail.com</a>
</p>

<p align="center">
  <a href="https://github.com/prayashmohanty">
    <img src="https://img.shields.io/badge/GitHub-PrayashRanjanMohanty-black?logo=github" alt="GitHub">
  </a>
</p>

<hr>

## 🙏 Acknowledgments

- 📚 **Kaggle:** For hosting the competition and providing the dataset
- 🔥 **PyTorch Team:** For the exceptional deep learning framework
- 🔧 **Optuna Team:** For the powerful hyperparameter optimization framework
- 🎓 **KIIT University:** For academic guidance and resources
- 💡 **Open Source Community:** For inspiration and learning resources

<hr>

<p align="center">
  <b>⭐ If you found this project helpful, please consider giving it a star! ⭐</b><br>
  <i>Made with ❤️ using PyTorch and Optuna</i>
</p>
