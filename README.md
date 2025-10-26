<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Introvert vs Extrovert Classification</title>
    <style>
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
