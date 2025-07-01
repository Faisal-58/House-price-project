from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import os


def train_and_evaluate(X_train,X_test,y_train, y_test, preprocessor):
    
    models = {
        'Linear Regression' : LinearRegression(),
        'Random Forest'  : RandomForestRegressor(),
        'XGBoost' : XGBRegressor()
    }

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        r2 = float(r2_score(y_test, pipeline.predict(X_test)))
        mae  = float(mean_absolute_error(y_test, pipeline.predict(X_test)))
        mse = float(mean_squared_error(y_test, pipeline.predict(X_test)))
        

        print(f"{name} Mean Absolute Error: {mae}")
        print(f"{name} Mean Sqaured Error : {mse}")

        print(f"{name} R2 score : {r2}")
        
        if name == 'XGBoost':
            joblib.dump(pipeline, "models/xgb_pipeline.pkl")
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, pipeline.predict(X_test), alpha=0.3, color='blue')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel("Actual Prices")
            plt.ylabel("Predicted Prices")
            plt.title("XGBoost Predictions vs Actual Prices")
            plt.tight_layout()
            os.makedirs("results", exist_ok=True)
            plt.savefig("results/xgb_predictions.png")
            plt.show()

            


        

