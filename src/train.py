# src/train.py

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from src.data_ingestion import load_data
from src.preprocessing import preprocess
from src.evaluate import evaluate

def main():
    df = load_data()

    X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror"
        )
    }

    results = {}
    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results[name] = evaluate(y_test, y_pred)

    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")


if __name__ == "__main__":
    main()
