import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run():
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train models.
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    mlflow.log_params({"n_estimators": 100, "max_depth":6, "max_features":3, "random_state": 42})
    mlflow.sklearn.log_model(rf, "model")

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    print(predictions)

# def train(X_train, y_train, hyperparameters):
    
#     n_estimators = hyperparameters["n_estimators"]
#     max_depth = hyperparameters["max_depth"]
#     max_features = hyperparameters["max_features"]
    
#     rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
#     rf.fit(X_train, y_train) 
    
#     return rf

# def predict(model, X_test):
#     predictions = model.predict(X_test)
#     return predictions
    