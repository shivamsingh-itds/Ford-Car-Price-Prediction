# Ford Car Price Prediction Using Machine Learning

This project predicts the prices of Ford used cars using machine learning regression models.  
It follows a clean, modular ML pipeline with proper data ingestion, preprocessing, model training, and evaluation.

The goal of this project is to compare traditional and tree-based regression models on a real-world, non-normal dataset.

---

## 📂 Dataset

- **Source:** Kaggle – Ford Car Price Prediction Dataset  
- **Target Variable:** `price`  
- **Features:** model, year, mileage, fuel type, transmission, engine size, etc.

The dataset contains real used-car listings and exhibits non-linear relationships and skewed distributions.

---

## 🧠 Models Used

The following regression models are implemented and compared:

- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor

Tree-based models are used to handle non-linearity and non-normal feature distributions effectively.

---

---

## ⚙️ Workflow

1. **Data Ingestion**
   - Load raw CSV data from `data/raw/`

2. **Preprocessing**
   - Train-test split  
   - Categorical encoding  
   - Feature scaling  
   - ColumnTransformer pipeline  

3. **Model Training**
   - Pipeline-based training for each model  
   - Same preprocessing applied consistently  

4. **Evaluation**
   - R² Score  
   - RMSE  
   - MAE  

---

## 🚀 How to Run the Project

1. Clone the repository
```
git clone https://github.com/shivamsingh-itds/Ford-Car-Price-Prediction.git
cd Ford-Car-Price-Prediction
```

2. Create and activate virtual environment
```
python -m venv ML
ML\Scripts\activate
```

Linux / macOS
```
python -m venv ML
source ML/bin/activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Run training pipeline
```
python -m src.train
```

---

## 📊 Evaluation Metrics

Each model is evaluated using:

R² Score – goodness of fit

RMSE – root mean squared error

MAE – mean absolute error

This allows fair comparison between baseline and advanced models.

---

## ✅ Key Highlights

Modular and reusable ML pipeline

CI/CD-safe project structure

No hardcoded paths

Supports multiple models with the same preprocessing

Clean Git history and repo hygiene

---

## 🔮 Future Improvements

Hyperparameter tuning

Model persistence (joblib)

Feature importance visualization

Experiment tracking

Deployment as a web application

--- 

## 👤 Author

**Shivam Singh**
Aspiring Data Scientist | Machine Learning Enthusiast

🔗 GitHub: [https://github.com/shivamsingh-itds]
🔗 LinkedIn: [www.linkedin.com/in/shivamsinghds]

---

⭐ If you find this project helpful, feel free to star the repository!
