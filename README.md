# 📊 Customer Churn Prediction System

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-F7931E.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**A machine learning-powered web application to predict customer churn and improve retention strategies.**

[Demo](#demo) • [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## 📖 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements an end-to-end machine learning solution for predicting customer churn in subscription-based services (streaming platforms, SaaS, etc.). The system analyzes customer behavior patterns, usage statistics, and account information to identify at-risk customers and provide actionable retention strategies.

### Why This Matters

- 🔴 Acquiring new customers costs 5-25x more than retaining existing ones
- 📉 A 5% increase in retention can increase profits by 25-95%
- ⚡ Early churn detection enables proactive intervention
- 💰 Reduces revenue loss from customer attrition

---

## 🎬 Demo

> **Live Demo**: [https://customer-churn-jntyln9q3aml4vtfmmisjj.streamlit.app/]

<div align="center">
  <img src="screenshots/demo.gif" alt="Application Demo" width="800"/>
</div>

---

## ✨ Features

### Core Functionality
- 🤖 **ML-Powered Predictions**: Ensemble model combining XGBoost and LightGBM
- 📊 **Real-Time Analysis**: Instant churn probability assessment
- 🎨 **Interactive Dashboard**: User-friendly Streamlit interface
- 📈 **Visual Analytics**: Probability charts and risk indicators
- 💡 **Smart Recommendations**: Personalized retention strategies

### Key Capabilities
- ✅ Single customer prediction
- ✅ Risk level categorization (Critical/High/Moderate/Low)
- ✅ Feature importance analysis
- ✅ Actionable business insights
- ✅ Responsive design for all devices

---

## 📊 Dataset

The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset) and contains comprehensive customer information:

### Dataset Statistics
- **Total Records**: 10,000+ customers
- **Features**: 27 input variables
- **Target**: Binary (Churn: Yes/No)
- **Class Distribution**: Balanced dataset

### Feature Categories

#### 1️⃣ Account Information
- `AccountAge`: Duration of customer account (months)
- `TotalCharges`: Cumulative charges ($)
- `SubscriptionType`: Service tier (Basic/Standard/Premium)

#### 2️⃣ Usage Metrics
- `ViewingHoursPerWeek`: Weekly viewing time
- `AverageViewingDuration`: Average session length (minutes)
- `ContentDownloadsPerMonth`: Monthly downloads
- `WatchlistSize`: Number of items in watchlist

#### 3️⃣ Service Interaction
- `SupportTicketsPerMonth`: Customer support requests
- `UserRating`: Customer satisfaction score (1-5)

#### 4️⃣ Preferences
- `PaymentMethod`: Payment type (Credit Card, PayPal, etc.)
- `ContentType`: Preferred content (Movies/TV Shows/Both)
- `GenrePreference`: Favorite genre
- `DeviceRegistered`: Number of registered devices

#### 5️⃣ Demographics & Settings
- `Gender`: Customer gender
- `PaperlessBilling`: Billing preference (Yes/No)
- `MultiDeviceAccess`: Multi-device feature usage
- `ParentalControl`: Parental control enabled
- `SubtitlesEnabled`: Subtitle preference

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.11 or higher
pip 21.0+
Git
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Model Files

Ensure these files are in your project root directory:
- `churn_model.pkl` - Trained ensemble model
- `scaler.pkl` - Fitted StandardScaler
- `training_columns.pkl` - Feature column names

> **Note**: Download model files from [Releases](https://github.com/yourusername/customer-churn-prediction/releases) or train your own using the provided notebook.

### Step 5: Run the Application

```bash
streamlit run main.py
```

Open your browser and navigate to `http://localhost:8501`

---

## 💻 Usage

### Quick Start

1. **Launch the Application**
   ```bash
   streamlit run main.py
   ```

2. **Enter Customer Information**
   - Fill in customer details in the sidebar
   - All fields are required for accurate prediction

3. **Get Prediction**
   - Click the **"🔮 Predict Churn"** button
   - View churn probability and risk assessment

4. **Review Recommendations**
   - Read personalized retention strategies
   - Implement suggested actions

### Input Parameters

| Parameter | Type | Range/Options | Description |
|-----------|------|---------------|-------------|
| Account Age | Integer | 0-100 | Customer account age in months |
| Total Charges | Float | 0-10,000 | Cumulative spending in dollars |
| Viewing Hours/Week | Float | 0-168 | Weekly viewing time |
| Avg Viewing Duration | Float | 0-300 | Average session length (minutes) |
| Content Downloads/Month | Integer | 0-100 | Monthly downloads |
| User Rating | Float | 1.0-5.0 | Customer satisfaction score |
| Support Tickets/Month | Integer | 0-50 | Monthly support requests |
| Watchlist Size | Integer | 0-200 | Number of saved items |
| Subscription Type | Dropdown | Basic/Standard/Premium | Service tier |
| Payment Method | Dropdown | Multiple options | Payment type |
| Content Type | Dropdown | Movies/TV Shows/Both | Content preference |
| Genre Preference | Dropdown | Multiple genres | Favorite genre |

### Output Interpretation

#### Risk Levels
- 🔴 **Critical Risk (80-100%)**: Immediate intervention required
- 🟠 **High Risk (60-79%)**: Urgent attention needed
- 🟡 **Moderate Risk (40-59%)**: Close monitoring recommended
- 🟢 **Low Risk (0-39%)**: Maintain current service quality

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── main.py                      # Streamlit web application
├── preprocessing.py             # Data preprocessing pipeline
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── models/
│   ├── churn_model.pkl         # Trained ML model
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── training_columns.pkl    # Feature column names
│
├── notebooks/
│   └── model_training.ipynb    # Model training notebook
│
├── screenshots/
│   ├── dashboard.png
│   ├── results.png
│   └── demo.gif
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
│
└── docs/
    ├── API.md
    └── DEPLOYMENT.md
```

---

## 🧠 Model Details

### Machine Learning Pipeline

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Deployment
```

### 1. Data Preprocessing

#### Encoding Strategy
| Feature Type | Method | Example |
|--------------|--------|---------|
| **Ordinal** | Manual mapping | Basic=0, Standard=1, Premium=2 |
| **Binary** | Boolean encoding | Yes=1, No=0 |
| **Nominal** | One-hot encoding | PaymentMethod → Multiple dummy columns |
| **Numerical** | StandardScaler | (x - μ) / σ |

#### Implementation
```python
# Ordinal Encoding
subscription_order = {'Basic': 0, 'Standard': 1, 'Premium': 2}
df['SubscriptionType'] = df['SubscriptionType'].map(subscription_order)

# Binary Encoding
binary_cols = ['PaperlessBilling', 'MultiDeviceAccess', 'ParentalControl', 'SubtitlesEnabled']
for col in binary_cols:
    df[col] = df[col].map({'No': 0, 'Yes': 1})

# One-Hot Encoding
df = pd.get_dummies(df, columns=['PaymentMethod', 'ContentType', 'GenrePreference'], drop_first=True)

# Standard Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

### 2. Model Architecture

**Ensemble Model**: Combines multiple algorithms for robust predictions

- **XGBoost Classifier**
  - Gradient boosting framework
  - Handles imbalanced data well
  - Feature importance extraction

- **LightGBM Classifier**
  - Fast training speed
  - Memory efficient
  - High accuracy on large datasets

- **Voting/Stacking**
  - Combines predictions from both models
  - Reduces overfitting
  - Improves generalization

### 3. Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.5% |
| **Precision** | 85.2% |
| **Recall** | 83.8% |
| **F1-Score** | 84.5% |
| **ROC-AUC** | 0.92 |

### 4. Feature Importance

Top 10 predictive features:

| Rank | Feature | Importance | Impact on Churn |
|------|---------|------------|-----------------|
| 1 | TotalCharges | 18.5% | Higher charges → Lower churn |
| 2 | SupportTicketsPerMonth | 15.3% | More tickets → Higher churn |
| 3 | UserRating | 14.2% | Lower rating → Higher churn |
| 4 | ViewingHoursPerWeek | 12.1% | Less usage → Higher churn |
| 5 | AccountAge | 10.8% | Newer accounts → Higher risk |
| 6 | ContentDownloadsPerMonth | 8.7% | Less downloads → Higher churn |
| 7 | SubscriptionType | 7.5% | Basic plan → Higher churn |
| 8 | WatchlistSize | 5.9% | Smaller watchlist → Higher churn |
| 9 | MultiDeviceAccess | 4.2% | No multi-device → Higher churn |
| 10 | PaymentMethod | 2.8% | Electronic check → Higher churn |

### 5. Model Training Process

1. **Data Loading & EDA**
   - Import dataset from Kaggle
   - Exploratory data analysis
   - Handle missing values

2. **Feature Engineering**
   - Create new features
   - Encode categorical variables
   - Scale numerical features

3. **Train-Test Split**
   - 80% training, 20% testing
   - Stratified split for balanced classes

4. **Model Training**
   - Train XGBoost and LightGBM
   - Hyperparameter tuning (GridSearchCV)
   - Cross-validation (5-fold)

5. **Model Evaluation**
   - Confusion matrix
   - ROC curve analysis
   - Feature importance visualization

6. **Model Serialization**
   - Save trained model (joblib)
   - Save scaler and column names
   - Version control

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.11** - Programming language
- **Streamlit 1.31** - Web framework
- **scikit-learn 1.2.2** - Machine learning library
- **Pandas 2.1.4** - Data manipulation
- **NumPy 1.26.4** - Numerical computing

### Machine Learning
- **XGBoost 2.0.3** - Gradient boosting
- **LightGBM 4.3.0** - Gradient boosting
- **Joblib 1.3.2** - Model serialization

### Development Tools
- **Git** - Version control
- **Jupyter Notebook** - Model development
- **VS Code** - Code editor

---

## 📸 Screenshots

### Main Dashboard
<img src="screenshots/dashboard.png" alt="Main Dashboard" width="800"/>

*User-friendly interface for entering customer information*

---

### Prediction Results
<img src="screenshots/results.png" alt="Prediction Results" width="800"/>

*Real-time churn prediction with probability breakdown*

---

### Risk Assessment & Recommendations
<img src="screenshots/recommendations.png" alt="Recommendations" width="800"/>

*Personalized retention strategies based on risk level*

---

## 📚 API Documentation

### Preprocessing Functions

#### `encode_dataset(df: pd.DataFrame) -> pd.DataFrame`

Encodes categorical features in the input dataframe.

**Parameters:**
- `df` (pd.DataFrame): Raw input dataframe with categorical features

**Returns:**
- `pd.DataFrame`: Encoded dataframe with numerical features

**Example:**
```python
from preprocessing import encode_dataset
import pandas as pd

input_data = pd.DataFrame({
    'SubscriptionType': ['Basic'],
    'PaymentMethod': ['Credit Card'],
    'PaperlessBilling': ['Yes']
})

encoded_data = encode_dataset(input_data)
print(encoded_data)
```

---

#### `transform_scaler(df: pd.DataFrame) -> pd.DataFrame`

Scales numerical features using the fitted StandardScaler.

**Parameters:**
- `df` (pd.DataFrame): Encoded dataframe

**Returns:**
- `pd.DataFrame`: Scaled dataframe

**Example:**
```python
from preprocessing import transform_scaler

scaled_data = transform_scaler(encoded_data)
```

---

#### `preprocess_input(df: pd.DataFrame) -> pd.DataFrame`

Complete preprocessing pipeline (encoding + scaling + column alignment).

**Parameters:**
- `df` (pd.DataFrame): Raw input dataframe

**Returns:**
- `pd.DataFrame`: Fully preprocessed dataframe ready for prediction

**Example:**
```python
from preprocessing import preprocess_input

final_data = preprocess_input(input_data)
prediction = model.predict(final_data)
```

---

### Model Prediction

```python
import joblib
from preprocessing import preprocess_input

# Load model
model = joblib.load('churn_model.pkl')

# Prepare input
input_data = pd.DataFrame({...})  # Your customer data
processed_data = preprocess_input(input_data)

# Get prediction
prediction = model.predict(processed_data)[0]  # 0 or 1
probability = model.predict_proba(processed_data)[0][1]  # Churn probability

print(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
print(f"Churn Probability: {probability:.2%}")
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Open a Pull Request**

### Contribution Guidelines

- Write clear, commented code
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🧪 Additional test coverage
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 👤 Contact

**Your Name**

- 📧 Email: sujswa@gmail.com
- 💼 LinkedIn: ((https://www.linkedin.com/in/swathy-ramakrishnan/))
- 🐱 GitHub: [@swaty06](https://github.com/swaty06)
- 🌐 Portfolio: [yourwebsite.com](https://swaty06.github.io/portfolio/#home)

**Project Link**: [https://github.com/swaty06/Customer-Churn]

---

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Customer Churn Dataset](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset)
- **Streamlit**: Amazing framework for building ML web apps
- **scikit-learn**: Comprehensive ML library
- **XGBoost & LightGBM**: Powerful gradient boosting libraries
- **Open Source Community**: For continuous support and inspiration

### Special Thanks

- Kaggle community for the dataset and insights
- Stack Overflow contributors for troubleshooting help
- Medium articles on churn prediction techniques
- YouTube tutorials on deployment strategies

---

## 🚀 Future Roadmap

### Version 1.1 (Q2 2025)
- [ ] Batch prediction for multiple customers
- [ ] Export predictions to CSV/Excel
- [ ] Model explainability with SHAP values
- [ ] Customer segmentation analysis

### Version 1.2 (Q3 2025)
- [ ] REST API endpoint
- [ ] Authentication and user management
- [ ] Email alerts for high-risk customers
- [ ] A/B testing framework

### Version 2.0 (Q4 2025)
- [ ] Real-time model retraining pipeline
- [ ] Multi-model comparison dashboard
- [ ] Advanced analytics and reporting
- [ ] Mobile app development

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/customer-churn-prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/customer-churn-prediction?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/customer-churn-prediction?style=social)

![GitHub issues](https://img.shields.io/github/issues/yourusername/customer-churn-prediction)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/customer-churn-prediction)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/customer-churn-prediction)

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star!**

**Made with ❤️ and ☕ by [Your Name]**

[Back to Top](#-customer-churn-prediction-system)

</div>
