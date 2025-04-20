# 🔥 Calories Burned Prediction using XGBoost

This project predicts the number of calories burned during physical activity using physiological and workout-related features. The model is trained on a dataset of 15,000 entries and achieves high accuracy using the XGBoost regression algorithm.

---

## 📁 Dataset Overview

The dataset is a combination of two CSV files:
- `calories.csv`: Contains `User_ID` and `Calories` burned.
- `exercise.csv`: Contains demographic and exercise-related metrics such as age, gender, height, weight, workout duration, heart rate, and body temperature.

After merging, the final dataset has the following features:

| Feature       | Description                                      |
|---------------|--------------------------------------------------|
| Gender        | Male or Female                                   |
| Age           | Age of the person                                |
| Height        | Height in centimeters                            |
| Weight        | Weight in kilograms                              |
| Duration      | Workout duration in minutes                      |
| Heart_Rate    | Heart rate during exercise                       |
| Body_Temp     | Body temperature during workout                  |
| Calories      | 🔥 Target variable — total calories burned       |

---

## 🔍 Exploratory Data Analysis (EDA)

- No missing values found in the dataset.
- Outliers detected using the IQR method and removed to improve model performance.
- Distribution plots and boxplots were used to understand variable ranges and skewness.
- Strong positive correlations found between `Calories` and features like:
  - `Duration`
  - `Heart_Rate`
  - `Body_Temp`

### 📊 Sample Visualizations:
- Histograms for numeric features
- Boxplots for outlier detection
- Correlation heatmap to analyze feature relationships

---

## 🧪 Preprocessing & Feature Engineering

- `Gender` encoded as binary: `male` → 0, `female` → 1
- `User_ID` dropped (not relevant for prediction)
- Data split into `X` (features) and `y` (target: Calories)
- Outliers removed using IQR method
- No feature scaling applied — XGBoost is tree-based and handles scale internally

---

## ⚙️ Model: XGBoost Regressor

# 🔧 Hyperparameter Tuning

Performed using `GridSearchCV` with the following parameter grid:

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}
```

---

## 🧠 Model Evaluation

| Metric   | Score   |
|----------|---------|
| MAE      | 1.10    |
| RMSE     | 1.68    |
| R² Score | **0.9993** ✅ |

---

## ✅ Benchmark Comparison

| Model               | R² Score   |
|---------------------|------------|
| Linear Regression   | 0.9664     |
| Random Forest       | 0.9865     |
| XGBoost (Tuned)     | **0.9993** ✅|

 ![Model Comparison](https://github.com/RishabhYadav1202/Calorie-Predictions-/blob/master/Images/Model%20comparison.png?raw=true)

---

## 📈 Visualization

- 📊 **Actual vs. Predicted Calories** (Scatter Plot)
  
 ![Actual vs Predicted](https://github.com/RishabhYadav1202/Calorie-Predictions-/blob/master/Images/actual%20vs%20prdicted.png?raw=true)
  
- 🔍 **XGBoost Feature Importance Plot**

  
 ![Feature Importance](https://github.com/RishabhYadav1202/Calorie-Predictions-/blob/master/Images/feature%20importance.png?raw=true)

 
---

## 📌 Key Takeaways

- Physiological features like **Heart Rate**, **Body Temperature**, and **Duration** are the most influential in calorie burn.
- **XGBoost with tuned hyperparameters** delivers exceptional performance on this dataset.
- Thorough **data cleaning and outlier handling** significantly improved model accuracy.
- Achieved a **near-perfect prediction** with R² ≈ **0.9993** on unseen test data.

---

## ⚠ Overfitting & Generalization

Despite the high R² score:

- ✅ **Overfitting was carefully addressed** by using:
  - 80/20 **train-test split**
  - `GridSearchCV` for robust internal cross-validation
- ✅ **No data leakage**: Target variable `Calories` was only present in the output (`y`), not the input features.

---

## 💡 Future Improvements

- 🔍 Integrate **SHAP values** for deeper model interpretability  
- 🖥 Build a **Streamlit or Flask app** to allow users to enter their own data  
- 💾 Save/load models using `joblib` or `pickle` for easy deployment  
- 🚀 Deploy the model via **REST API** or **web interface**





