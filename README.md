<!DOCTYPE html>
<html lang="en">
<body>

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

<h2>🚀 Overview</h2>

<p>
A sophisticated Artificial Neural Network (ANN) built from scratch using <b>PyTorch</b> to classify personality types as either <b>Introvert</b> or <b>Extrovert</b> based on behavioral features. This project was developed for a <b>Kaggle competition</b> and demonstrates advanced techniques including hyperparameter optimization with <b>Optuna</b>, custom data preprocessing, and strategic model architecture design.
</p>

<hr>

<h2>🎯 Key Features</h2>

<ul>
  <li>✨ <b>Custom ANN Architecture</b> - 2-layer neural network designed specifically for personality classification</li>
  <li>🔧 <b>Hyperparameter Optimization</b> - Systematic tuning using Optuna (1000+ trials)</li>
  <li>📊 <b>Smart Data Preprocessing</b> - Class-wise mean imputation for missing values</li>
  <li>🎯 <b>High Accuracy</b> - Achieved 98.14% accuracy on test set</li>
  <li>⏸️ <b>Early Stopping Implementation</b> - Prevents overfitting with patience-based monitoring</li>
  <li>💾 <b>Model Checkpointing</b> - Automatic saving of best performing models</li>
  <li>📈 <b>Comprehensive Evaluation</b> - F1 Score: 0.9725</li>
  <li>🔥 <b>GPU Acceleration</b> - CUDA support for faster training</li>
</ul>

<hr>

<h2>📊 Dataset & Performance</h2>

<table align="center">
  <tr>
    <th>Metric</th>
    <th>Value</th>
  </tr>
  <tr>
    <td><b>Test Accuracy</b></td>
    <td>98.14%</td>
  </tr>
  <tr>
    <td><b>F1 Score (Weighted)</b></td>
    <td>0.9725</td>
  </tr>
  <tr>
    <td><b>Validation Accuracy</b></td>
    <td>97.25%</td>
  </tr>
  <tr>
    <td><b>Training Samples</b></td>
    <td>18,524</td>
  </tr>
  <tr>
    <td><b>Test Samples</b></td>
    <td>6,175</td>
  </tr>
  <tr>
    <td><b>Features</b></td>
    <td>7 behavioral indicators</td>
  </tr>
</table>

<h3>Class Distribution:</h3>
<ul>
  <li><b>Extroverts:</b> 13,699 samples (73.9%)</li>
  <li><b>Introverts:</b> 4,825 samples (26.1%)</li>
</ul>

<hr>

<h2>🧩 Features Used</h2>

<p>The model analyzes <b>7 carefully selected behavioral features</b> to predict personality type:</p>

<table align="center">
  <tr>
    <th>Feature</th>
    <th>Description</th>
    <th>Range</th>
  </tr>
  <tr>
    <td><b>Time Spent Alone</b></td>
    <td>Hours per day spent alone</td>
    <td>0-11</td>
  </tr>
  <tr>
    <td><b>Stage Fear</b></td>
    <td>Presence of public speaking anxiety</td>
    <td>Binary (Yes/No)</td>
  </tr>
  <tr>
    <td><b>Social Event Attendance</b></td>
    <td>Frequency of attending social gatherings</td>
    <td>0-10</td>
  </tr>
  <tr>
    <td><b>Going Outside</b></td>
    <td>Days per week spent outdoors</td>
    <td>0-7</td>
  </tr>
  <tr>
    <td><b>Drained After Socializing</b></td>
    <td>Feeling exhausted after social interaction</td>
    <td>Binary (Yes/No)</td>
  </tr>
  <tr>
    <td><b>Friends Circle Size</b></td>
    <td>Number of close friends</td>
    <td>0-15</td>
  </tr>
  <tr>
    <td><b>Post Frequency</b></td>
    <td>Social media posting activity</td>
    <td>0-10</td>
  </tr>
</table>

<hr>

<h2>🏗️ Model Architecture</h2>

<p>Optimized 2-layer neural network with batch normalization and dropout regularization:</p>

<pre><code>class NNArch(nn.Module):
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
</code></pre>

<h3>Architecture Flow:</h3>
<ul>
  <li><b>Input Layer:</b> 7 behavioral features</li>
  <li><b>Hidden Layer 1:</b> 64 neurons + BatchNorm + ReLU + Dropout(11.4%)</li>
  <li><b>Hidden Layer 2:</b> 128 neurons + BatchNorm + ReLU + Dropout(23.3%)</li>
  <li><b>Output Layer:</b> 2 classes (Introvert/Extrovert)</li>
  <li><b>Total Parameters:</b> ~11,000</li>
</ul>

<hr>

<h2>🔧 Data Preprocessing Pipeline</h2>

<h3>1. Missing Value Handling</h3>
<p><b>Smart Class-wise Imputation:</b></p>
<pre><code># For training set: class-wise means
for feature in features:
    mean_intro = train[train['Personality'] == 1][feature].mean()
    mean_extro = train[train['Personality'] == 0][feature].mean()
    
    train.loc[(train['Personality'] == 1) & (train[feature].isnull()), feature] = mean_intro
    train.loc[(train['Personality'] == 0) & (train[feature].isnull()), feature] = mean_extro

# For test set: average of both class means
    avg_mean = (mean_intro + mean_extro) / 2
    test[feature] = test[feature].fillna(avg_mean)
</code></pre>

<h3>2. Feature Encoding</h3>
<table align="center">
  <tr>
    <th>Feature</th>
    <th>Original</th>
    <th>Encoded</th>
  </tr>
  <tr>
    <td>Stage Fear</td>
    <td>No / Yes</td>
    <td>0 / 1</td>
  </tr>
  <tr>
    <td>Drained After Socializing</td>
    <td>No / Yes</td>
    <td>0 / 1</td>
  </tr>
  <tr>
    <td>Personality</td>
    <td>Extrovert / Introvert</td>
    <td>0 / 1</td>
  </tr>
</table>

<h3>3. Min-Max Normalization</h3>
<pre><code>for feature in ['Time_spent_Alone', 'Social_event_attendance', 
                'Going_outside', 'Friends_circle_size', 'Post_frequency']:
    min_val = train[feature].min()
    max_val = train[feature].max()
    train[feature] = (train[feature] - min_val) / (max_val - min_val)
    test[feature] = (test[feature] - min_val) / (max_val - min_val)
</code></pre>

<hr>

<h2>⚙️ Training Configuration</h2>

<h3>Optimal Hyperparameters (Found via Optuna):</h3>
<table align="center">
  <tr>
    <th>Parameter</th>
    <th>Value</th>
  </tr>
  <tr>
    <td><b>Learning Rate</b></td>
    <td>0.00133</td>
  </tr>
  <tr>
    <td><b>Weight Decay</b></td>
    <td>1e-05</td>
  </tr>
  <tr>
    <td><b>Batch Size</b></td>
    <td>128</td>
  </tr>
  <tr>
    <td><b>Optimizer</b></td>
    <td>Adam</td>
  </tr>
  <tr>
    <td><b>Loss Function</b></td>
    <td>CrossEntropyLoss</td>
  </tr>
  <tr>
    <td><b>Activation</b></td>
    <td>ReLU</td>
  </tr>
  <tr>
    <td><b>Max Epochs</b></td>
    <td>6000</td>
  </tr>
  <tr>
    <td><b>Early Stopping Patience</b></td>
    <td>7 epochs</td>
  </tr>
</table>

<h3>Data Split:</h3>
<ul>
  <li><b>Training Set:</b> 70% (~12,966 samples)</li>
  <li><b>Validation Set:</b> 21% (~3,890 samples)</li>
  <li><b>Test Set:</b> 9% (~1,668 samples)</li>
</ul>

<hr>

<h2>🎨 Hyperparameter Optimization with Optuna</h2>

<p>Conducted <b>1000+ trials</b> to find optimal configuration:</p>

<h3>Search Space:</h3>
<pre><code>trial.suggest_int('num_of_layers', 1, 3)
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
</code></pre>

<h3>Best Trial Results:</h3>
<ul>
  <li>🏆 <b>Accuracy:</b> 98.14%</li>
  <li>🎯 <b>Architecture:</b> 2 layers (7 → 64 → 128 → 2)</li>
  <li>⚡ <b>Optimizer:</b> Adam</li>
  <li>🔥 <b>Activation:</b> ReLU</li>
</ul>

<hr>

<h2>🛠️ Technologies & Dependencies</h2>

<h3>Core Libraries:</h3>
<pre><code>import torch
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
</code></pre>

<h3>Required Packages:</h3>
<p align="center">
  🔥 PyTorch • 📊 Pandas • 🧮 NumPy • 📈 Scikit-learn • 🎨 Matplotlib • 🔧 Optuna
</p>

<hr>

<h2>📦 Installation & Setup</h2>

<h3>1. Install Dependencies</h3>
<pre><code>pip install torch pandas numpy scikit-learn matplotlib optuna
</code></pre>

<h3>2. For Google Colab</h3>
<pre><code># Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device available:", device)
</code></pre>

<h3>3. Clone Repository</h3>
<pre><code>git clone https://github.com/yourusername/introvert-extrovert-classifier.git
cd introvert-extrovert-classifier
</code></pre>

<hr>

<h2>🚀 Usage</h2>

<h3>1. Training the Model</h3>
<pre><code># Initialize model
model = NNArch()
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.00133, weight_decay=1e-05)

# Train with early stopping
# See notebook for complete training loop
</code></pre>

<h3>2. Load Pre-trained Model</h3>
<pre><code># Load best saved model
model = NNArch()
model.load_state_dict(torch.load('best_model_optuna.pth'))
model.to(device)
model.eval()
</code></pre>

<h3>3. Make Predictions</h3>
<pre><code># Prepare input data (normalized and encoded)
x_tensor = torch.tensor(input_data, dtype=torch.float32)
x_tensor = x_tensor.to(device)

with torch.no_grad():
    predictions = model(x_tensor)
    predicted_class = torch.argmax(predictions, dim=1)
    
    # 0 = Extrovert, 1 = Introvert
    personality = 'Extrovert' if predicted_class == 0 else 'Introvert'
    print(f"Predicted Personality: {personality}")
</code></pre>

<hr>

<h2>📁 Project Structure</h2>

<pre><code>Introvert-Extrovert-Classification/
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
</code></pre>

<hr>

<h2>🔑 Key Implementation Details</h2>

<h3>Custom Dataset Class</h3>
<pre><code>class customdataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.Y = torch.tensor(Y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
</code></pre>

<h3>Early Stopping Mechanism</h3>
<pre><code>best_val_loss = float('inf')
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
</code></pre>

<h3>Device Configuration</h3>
<pre><code>device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device available:", device)

# Move model and data to GPU
model.to(device)
x_batch = x_batch.to(device).float()
y_batch = y_batch.to(device).long()
</code></pre>

<hr>

<h2>📈 Training Progress</h2>

<h3>Training Metrics:</h3>
<ul>
  <li><b>Epochs Trained:</b> 13 (out of max 6000)</li>
  <li><b>Early Stopping Triggered:</b> Epoch 13</li>
  <li><b>Final Training Loss:</b> ~0.088</li>
  <li><b>Final Validation Loss:</b> ~0.111</li>
  <li><b>Test Loss:</b> 0.1145</li>
  <li><b>Test Accuracy:</b> 98.14%</li>
</ul>

<p>The model converged quickly thanks to optimal hyperparameters found through Optuna optimization.</p>

<hr>

<h2>🎯 Model Performance Analysis</h2>

<h3>Strengths:</h3>
<ul>
  <li>✅ Exceptional accuracy (98.14%) on unseen test data</li>
  <li>✅ High F1 score (0.9725) indicating balanced precision and recall</li>
  <li>✅ Fast convergence with early stopping (13 epochs)</li>
  <li>✅ Handles class imbalance effectively</li>
  <li>✅ Lightweight architecture (~11K parameters)</li>
  <li>✅ Low overfitting risk with batch normalization and dropout</li>
  <li>✅ Robust preprocessing pipeline</li>
</ul>

<h3>Technical Highlights:</h3>
<ul>
  <li>🎯 Class-wise mean imputation improved model generalization</li>
  <li>📊 Min-max normalization ensured feature scale consistency</li>
  <li>🔄 Batch normalization stabilized training</li>
  <li>⚡ Adam optimizer with optimal learning rate</li>
  <li>🏆 Systematic hyperparameter tuning with Optuna</li>
</ul>

<hr>

<h2>🔮 Future Enhancements</h2>

<ul>
  <li>🚀 <b>Web Deployment:</b> Flask/Streamlit app with interactive questionnaire</li>
  <li>📱 <b>Mobile App:</b> Convert to ONNX/TFLite format for mobile deployment</li>
  <li>🎨 <b>Feature Engineering:</b> Explore interaction features and polynomial terms</li>
  <li>🏗️ <b>Advanced Architectures:</b> Experiment with attention mechanisms</li>
  <li>📊 <b>Ensemble Methods:</b> Combine multiple models for higher accuracy</li>
  <li>🔄 <b>Cross-validation:</b> K-fold CV for more robust evaluation</li>
  <li>💡 <b>Explainability:</b> SHAP values to understand feature importance</li>
  <li>🌐 <b>Multi-class Extension:</b> Classify into MBTI personality types</li>
</ul>

<hr>

<h2>📝 Notebook Features</h2>

<p>The complete Jupyter notebook includes:</p>

<ol>
  <li>📥 <b>Data Loading:</b> CSV import and initial exploration</li>
  <li>🔍 <b>Data Analysis:</b> Statistics and distribution analysis</li>
  <li>🧹 <b>Data Cleaning:</b> Missing value handling and duplicate removal</li>
  <li>🎨 <b>Preprocessing:</b> Encoding, normalization, and splitting</li>
  <li>🏗️ <b>Model Definition:</b> Custom ANN architecture</li>
  <li>🎓 <b>Training Loop:</b> With early stopping and checkpointing</li>
  <li>📊 <b>Evaluation:</b> Comprehensive metrics (accuracy, F1 score)</li>
  <li>📈 <b>Visualization:</b> Loss curves over epochs</li>
  <li>🔧 <b>Hyperparameter Tuning:</b> Optuna optimization (1000+ trials)</li>
  <li>💾 <b>Model Persistence:</b> Save and load functionality</li>
  <li>🧪 <b>Kaggle Submission:</b> Test set prediction and CSV export</li>
</ol>

<hr>

<h2>⚠️ Important Notes</h2>

<ul>
  <li>🔧 <b>GPU Recommended:</b> CPU training will be significantly slower</li>
  <li>💾 <b>Google Drive:</b> Models are saved in Google Drive when using Colab</li>
  <li>📏 <b>Input Requirements:</b> Features must be preprocessed (normalized & encoded)</li>
  <li>🎯 <b>Binary Classification:</b> Model outputs Introvert (1) or Extrovert (0)</li>
  <li>🔄 <b>Preprocessing Consistency:</b> New data needs same transformations as training</li>
  <li>📊 <b>Class Imbalance:</b> Model trained on imbalanced data (73.9% Extroverts)</li>
</ul>

<hr>

<h2>🐛 Troubleshooting</h2>

<h3>Common Issues:</h3>

<p><b>Issue:</b> CUDA out of memory</p>
<pre><code># Solution: Reduce batch size
batch_size = 64  # Instead of 128
</code></pre>

<p><b>Issue:</b> Model not loading</p>
<pre><code># Solution: Check device consistency
model = NNArch()
model.load_state_dict(torch.load('model.pth', map_location=device))
</code></pre>

<p><b>Issue:</b> Poor predictions on new data</p>
<pre><code># Solution: Ensure proper preprocessing
# 1. Encode categorical features (Stage_fear, Drained_after_socializing)
# 2. Normalize continuous features using training set min/max
# 3. Handle missing values with appropriate strategy
</code></pre>

<p><b>Issue:</b> Optuna study not loading</p>
<pre><code># Solution: Check database path and permissions
study = optuna.create_study(
    direction='maximize',
    study_name='Introvert_Extrovert_Project08_study',
    storage='sqlite:///path/to/study.db',
    load_if_exists=True
)
</code></pre>

<hr>

<h2>📚 Learning Resources</h2>

<ul>
  <li>📖 <a href="https://pytorch.org/tutorials/">PyTorch Official Tutorials</a></li>
  <li>📖 <a href="https://optuna.readthedocs.io/">Optuna Documentation</a></li>
  <li>📖 <a href="https://scikit-learn.org/stable/modules/preprocessing.html">Scikit-learn Preprocessing Guide</a></li>
  <li>📖 <a href="https://www.kaggle.com/learn">Kaggle Learn</a></li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>

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

<h2>🙏 Acknowledgments</h2>

<ul>
  <li>📚 <b>Kaggle:</b> For hosting the competition and providing the dataset</li>
  <li>🔥 <b>PyTorch Team:</b> For the exceptional deep learning framework</li>
  <li>🔧 <b>Optuna Team:</b> For the powerful hyperparameter optimization framework</li>
  <li>🎓 <b>KIIT University:</b> For academic guidance and resources</li>
  <li>💡 <b>Open Source Community:</b> For inspiration and learning resources</li>
</ul>

<hr>

<p align="center">
  <b>⭐ If you found this project helpful, please consider giving it a star! ⭐</b><br>
  <i>Made with ❤️ using PyTorch and Optuna</i>
</p>

</body>
</html>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
        }

        h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .content {
            padding: 40px;
        }

        h2 {
            color: #667eea;
            font-size: 2em;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        h3 {
            color: #764ba2;
            font-size: 1.5em;
            margin: 20px 0 10px 0;
        }

        .section {
            margin-bottom: 40px;
        }

        .badge {
            display: inline-block;
            padding: 8px 16px;
            margin: 5px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .stat-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }

        .stat-label {
            color: #666;
            font-size: 1.1em;
        }

        .feature-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .feature-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }

        code {
            background: #f4f4f4;
            padding: 3px 8px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            color: #d63384;
        }

        .code-block {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
        }

        .architecture-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
        }

        .layer {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid white;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        tr:hover {
            background: #f5f5f5;
        }

        .highlight-box {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }

        .success-box {
            background: #d4edda;
            border-left: 5px solid #28a745;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }

        ul {
            margin-left: 20px;
        }

        li {
            margin: 8px 0;
        }

        .author-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Introvert vs Extrovert Classification</h1>
            <p class="subtitle">Deep Learning Personality Prediction using Artificial Neural Networks</p>
            <div style="margin-top: 20px;">
                <span class="badge">PyTorch</span>
                <span class="badge">Neural Networks</span>
                <span class="badge">Kaggle Competition</span>
                <span class="badge">Optuna Optimization</span>
            </div>
        </header>

        <div class="content">
            <div class="section">
                <h2>📊 Project Overview</h2>
                <p>This project implements a sophisticated Artificial Neural Network (ANN) to classify personality types as either <strong>Introvert</strong> or <strong>Extrovert</strong> based on behavioral features. Developed for a Kaggle competition, this model achieves impressive accuracy through careful feature engineering, hyperparameter tuning, and architectural optimization.</p>
            </div>

            <div class="section">
                <h2>🎯 Key Results</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Test Accuracy</div>
                        <div class="stat-number">98.14%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">F1 Score</div>
                        <div class="stat-number">0.9725</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Training Samples</div>
                        <div class="stat-number">18,524</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Test Samples</div>
                        <div class="stat-number">6,175</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>📋 Dataset Features</h2>
                <p>The model uses 7 carefully selected behavioral features:</p>
                <div class="feature-list">
                    <div class="feature-item"><strong>Time Spent Alone</strong><br>Hours per day (0-11)</div>
                    <div class="feature-item"><strong>Stage Fear</strong><br>Binary (Yes/No)</div>
                    <div class="feature-item"><strong>Social Event Attendance</strong><br>Frequency (0-10)</div>
                    <div class="feature-item"><strong>Going Outside</strong><br>Days per week (0-7)</div>
                    <div class="feature-item"><strong>Drained After Socializing</strong><br>Binary (Yes/No)</div>
                    <div class="feature-item"><strong>Friends Circle Size</strong><br>Number (0-15)</div>
                    <div class="feature-item"><strong>Post Frequency</strong><br>Social media activity (0-10)</div>
                </div>
            </div>

            <div class="section">
                <h2>🏗️ Model Architecture</h2>
                <div class="architecture-box">
                    <h3 style="color: white;">Optimized Neural Network Structure</h3>
                    <div class="layer">
                        <strong>Input Layer:</strong> 7 features
                    </div>
                    <div class="layer">
                        <strong>Hidden Layer 1:</strong> 64 neurons<br>
                        • Batch Normalization<br>
                        • ReLU Activation<br>
                        • Dropout (11.4%)
                    </div>
                    <div class="layer">
                        <strong>Hidden Layer 2:</strong> 128 neurons<br>
                        • Batch Normalization<br>
                        • ReLU Activation<br>
                        • Dropout (23.3%)
                    </div>
                    <div class="layer">
                        <strong>Output Layer:</strong> 2 classes (Introvert/Extrovert)<br>
                        • CrossEntropyLoss
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>🔧 Data Preprocessing Pipeline</h2>
                
                <h3>1. Missing Value Handling</h3>
                <div class="highlight-box">
                    <strong>Smart Imputation Strategy:</strong> Missing values were filled using class-wise means. For training data, introverts and extroverts had separate mean calculations, while test data used averaged means from both classes.
                </div>

                <h3>2. Feature Encoding</h3>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Original Values</th>
                        <th>Encoded Values</th>
                    </tr>
                    <tr>
                        <td>Stage Fear</td>
                        <td>No / Yes</td>
                        <td>0 / 1</td>
                    </tr>
                    <tr>
                        <td>Drained After Socializing</td>
                        <td>No / Yes</td>
                        <td>0 / 1</td>
                    </tr>
                    <tr>
                        <td>Personality</td>
                        <td>Extrovert / Introvert</td>
                        <td>0 / 1</td>
                    </tr>
                </table>

                <h3>3. Feature Normalization</h3>
                <p>Continuous features were normalized using Min-Max scaling:</p>
                <div class="code-block">
normalized_value = (value - min) / (max - min)</div>
                <p><strong>Normalized Features:</strong> Time Spent Alone, Social Event Attendance, Going Outside, Friends Circle Size, Post Frequency</p>
            </div>

            <div class="section">
                <h2>⚙️ Training Configuration</h2>
                
                <h3>Optimal Hyperparameters</h3>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Learning Rate</td>
                        <td>0.00133</td>
                    </tr>
                    <tr>
                        <td>Weight Decay</td>
                        <td>1e-05</td>
                    </tr>
                    <tr>
                        <td>Batch Size</td>
                        <td>128</td>
                    </tr>
                    <tr>
                        <td>Optimizer</td>
                        <td>Adam</td>
                    </tr>
                    <tr>
                        <td>Epochs</td>
                        <td>6000 (with early stopping)</td>
                    </tr>
                    <tr>
                        <td>Early Stopping Patience</td>
                        <td>7 epochs</td>
                    </tr>
                </table>

                <h3>Data Split Strategy</h3>
                <ul>
                    <li><strong>Training Set:</strong> 70% (12,966 samples)</li>
                    <li><strong>Validation Set:</strong> 21% (3,890 samples)</li>
                    <li><strong>Test Set:</strong> 9% (1,668 samples)</li>
                </ul>
            </div>

            <div class="section">
                <h2>🎨 Hyperparameter Optimization</h2>
                <p>The model underwent extensive hyperparameter tuning using <strong>Optuna</strong>, an automatic hyperparameter optimization framework:</p>
                
                <h3>Search Space</h3>
                <ul>
                    <li><strong>Number of Layers:</strong> 1-3</li>
                    <li><strong>Hidden Units:</strong> [16, 32, 64, 128, 256]</li>
                    <li><strong>Dropout Rates:</strong> 0.1 - 0.5</li>
                    <li><strong>Learning Rate:</strong> 1e-4 to 1e-1 (log scale)</li>
                    <li><strong>Weight Decay:</strong> [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]</li>
                    <li><strong>Batch Size:</strong> [16, 32, 64, 128]</li>
                    <li><strong>Optimizers:</strong> Adam, SGD, RMSprop</li>
                    <li><strong>Activations:</strong> ReLU, LeakyReLU, ELU</li>
                </ul>

                <div class="success-box">
                    <strong>🏆 Best Configuration Found:</strong> After 1000+ trials, the optimal configuration achieved 98.14% accuracy with a 2-layer architecture using Adam optimizer and ReLU activation.
                </div>
            </div>

            <div class="section">
                <h2>📈 Training Process</h2>
                <p>The training process implemented several advanced techniques:</p>
                
                <h3>Key Features</h3>
                <ul>
                    <li><strong>Batch Normalization:</strong> Stabilizes and accelerates training</li>
                    <li><strong>Dropout Regularization:</strong> Prevents overfitting</li>
                    <li><strong>Early Stopping:</strong> Monitors validation loss and stops when no improvement</li>
                    <li><strong>GPU Acceleration:</strong> CUDA-enabled training for faster iterations</li>
                    <li><strong>Data Augmentation:</strong> Shuffled batches with stratified sampling</li>
                </ul>

                <h3>Loss Function</h3>
                <p>CrossEntropyLoss was used for multi-class classification, combining LogSoftmax and NLLLoss in one operation.</p>
            </div>

            <div class="section">
                <h2>💻 Implementation Details</h2>
                
                <h3>Technologies Used</h3>
                <div class="feature-list">
                    <div class="feature-item">PyTorch 2.x</div>
                    <div class="feature-item">Pandas</div>
                    <div class="feature-item">NumPy</div>
                    <div class="feature-item">Scikit-learn</div>
                    <div class="feature-item">Optuna</div>
                    <div class="feature-item">Matplotlib</div>
                </div>

                <h3>Code Structure</h3>
                <div class="code-block">
project/
├── Introvert_Extrovert_prediction.ipynb
├── train.csv
├── test.csv
├── submission.csv
└── saved_models/
    ├── current_model.pth
    └── best_model_optuna.pth</div>
            </div>

            <div class="section">
                <h2>🚀 How to Use</h2>
                
                <h3>1. Load the Trained Model</h3>
                <div class="code-block">
import torch
from model import NNArch

model = NNArch()
model.load_state_dict(torch.load('best_model_optuna.pth'))
model.eval()</div>

                <h3>2. Prepare Input Data</h3>
                <div class="code-block">
# Ensure data is normalized and encoded
x_tensor = torch.tensor(input_data, dtype=torch.float32)
x_tensor = x_tensor.to(device)</div>

                <h3>3. Make Predictions</h3>
                <div class="code-block">
with torch.no_grad():
    predictions = model(x_tensor)
    predicted_class = torch.argmax(predictions, dim=1)</div>
            </div>

            <div class="section">
                <h2>📊 Model Performance Analysis</h2>
                
                <h3>Class Distribution</h3>
                <p>The dataset shows class imbalance:</p>
                <ul>
                    <li><strong>Extroverts:</strong> 13,699 samples (73.9%)</li>
                    <li><strong>Introverts:</strong> 4,825 samples (26.1%)</li>
                </ul>

                <div class="highlight-box">
                    Despite class imbalance, the model achieves excellent performance on both classes through weighted loss functions and stratified sampling during training.
                </div>

                <h3>Validation Metrics</h3>
                <ul>
                    <li><strong>Validation Loss:</strong> 0.1113</li>
                    <li><strong>Validation Accuracy:</strong> 97.25%</li>
                    <li><strong>Test Accuracy:</strong> 98.14%</li>
                    <li><strong>F1 Score (Weighted):</strong> 0.9725</li>
                </ul>
            </div>

            <div class="section">
                <h2>🔍 Key Insights</h2>
                <ul>
                    <li>The model successfully learns behavioral patterns distinguishing introverts from extroverts</li>
                    <li>Time spent alone and social event attendance are highly predictive features</li>
                    <li>Batch normalization significantly improved training stability</li>
                    <li>Early stopping prevented overfitting while maintaining high accuracy</li>
                    <li>The 2-layer architecture proved optimal for this dataset size</li>
                </ul>
            </div>

            <div class="section">
                <h2>🎓 Learning Outcomes</h2>
                <ul>
                    <li>Implemented custom PyTorch Dataset and DataLoader classes</li>
                    <li>Applied advanced preprocessing techniques for missing data</li>
                    <li>Utilized Optuna for systematic hyperparameter optimization</li>
                    <li>Implemented early stopping and model checkpointing</li>
                    <li>Achieved production-ready model performance (>98% accuracy)</li>
                </ul>
            </div>

            <div class="section">
                <h2>🔮 Future Improvements</h2>
                <ul>
                    <li>Implement ensemble methods combining multiple models</li>
                    <li>Explore feature engineering for additional behavioral patterns</li>
                    <li>Apply SMOTE or other techniques for better class balance</li>
                    <li>Experiment with attention mechanisms</li>
                    <li>Deploy as a web application for real-time predictions</li>
                </ul>
            </div>
        </div>

        <div class="author-section">
            <h2 style="color: white;">👨‍💻 Project Information</h2>
            <p>Developed for Kaggle Competition</p>
            <p>Framework: PyTorch | Language: Python 3.11</p>
            <p>Environment: Google Colab with GPU (T4)</p>
        </div>
    </div>
</body>
</html>
