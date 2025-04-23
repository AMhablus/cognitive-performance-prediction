# 🧠 Cognitive Performance Prediction using Neural Networks

This project predicts human cognitive performance based on various demographic, physiological, and lifestyle factors. Three models are built and compared:

- A Feedforward Neural Network (FNN) implemented **from scratch using NumPy**
- An FNN built using **TensorFlow**
- A **Linear Regression** baseline model

The goal is to evaluate how well neural networks can model complex relationships in human cognitive function and to demonstrate the inner workings of deep learning through a scratch-built model.

---

## 📊 Dataset Information

- **Name:** Human Cognitive Performance Dataset
- **Source:** [Provide source if public or mention it was custom/generated]
- **Target variable:** `Cognitive_Score`
- **Features:**
  - Demographics: Age, Gender
  - Health & Lifestyle: Diet Type, Exercise Frequency, Sleep Quality
  - Physiological: Heart Rate, Blood Pressure

Preprocessing steps included:
- One-hot encoding of categorical variables
- Robust scaling for numerical features
- Outlier removal using IQR

---

## 🚀 How to Run the Code

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/cognitive-performance-prediction.git
cd cognitive-performance-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook or script
You can use a Jupyter Notebook or run the `.py` script:

```bash
jupyter notebook cognitive_prediction.ipynb
```
or
```bash
python cognitive_prediction.py
```

Make sure the dataset CSV file is placed in the correct path as defined in the code.

---

## 📊 Results

| Model              | MAE      | RMSE   | R²       |
|--------------------|---------|---------|----------|
| Linear Regression  | 0.6279  | 1.5274  |  0.9956  |
| TensorFlow FNN     | 4.8032  | 6.1603  |  0.9277  |
| Scratch FNN (NumPy)| 4.5905  | 5.9452  |  0.9326  |


---

## 📸 Visualizations

- Correlation Heatmap
- Feature Distributions
- Comparison Bar Charts (MAE, RMSE, R²)

![example-plot](plots/performance_comparison.png)

---

## 🚀 Future Work

- Add support for mini-batch and Adam optimizer in the scratch FNN
- Perform hyperparameter tuning
- Try deeper or wider networks
- Extend to classification task (e.g., cognitive category)

---



## 📁 Folder Structure

```
├── cognitive_prediction.ipynb / .py   # Main notebook or script
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Files to exclude from git
├── data/                            # Folder for dataset CSV
└── plots/                           # Folder for saved visualizations
```

---


Thanks for checking out my machine learning project! 🚀

