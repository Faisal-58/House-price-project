**Melbourne House Price Prediction ( Regression Problem)**

An end-to-end Machine Learning project to predict housing prices in Melbourne, Australia. Built using XGBoost, FastAPI (for serving predictions), and Streamlit (for interactive frontend).

 **Important Aspects**
 1. Cleaning and Preprocessing of a real-world Melbourne Housing Dataset.
 2. Performing Exploratory Data Analysis (EDA) for better understanding of the data and viewing the insights.
 3. Regression Techniques such as Linear Regression, XgBoost and Random Forest are used for training of the models.
 4. The target (which is the price in this case) is log-transformed to minimize the skewness and improve the performance of the model.
 5. The model is evaluated using the evaluation metrics such as (R2 score, Mean Absolute Error and Mean Squared Error)
 6. FastAPI is used for backend serving predictions.
 7. Streamlit serves the purpose for UI and user can input using it.
 8. The coding is done in VS code having a modular structure creating virtual environment and also requirements.txt
 9. The project is uploaded by using git.



## Model Performance (with log-transformed prices) 
**Evaluation Metrics**

| Model            | R² Score | MAE     | MSE     |
|------------------|----------|---------|---------|
| Linear Regression| 0.7822   | 0.1834  | 0.0595  |
| Random Forest    | 0.8722   | 0.1380  | 0.0349  |
| XGBoost          | **0.8765**   | **0.1386**  | **0.0337**  |

The results confirmed the fact that XGBoost outperformes the other models. Also Random Forest did a good job with a R2 score of 0.8722. This confirms the fact the tree models mostle perform better compared to linear models for countering non-linear data.

## 🛠️ Tech Stack

- Python 3.11
- [pandas](w), [scikit-learn](w), [xgboost](w)
- [FastAPI](w)
- [Streamlit](w)
- [joblib](w)
- VS Code + Git

---

## 📁 Project Structure

