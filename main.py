
from fastapi import FastAPI
import joblib
from src.predict import Passenger, predict_survival

app = FastAPI(title="Titanic Survival Prediction API", 
              description="API for predicting passenger survival on the Titanic",
              version="1.0.0")

# Load the trained model from file when the app starts
model = joblib.load("models/titanic_pipeline.joblib")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(passenger: Passenger):
    """
    Predict passenger survival on the Titanic
    
    - **Pclass**: Passenger class (1, 2, or 3)
    - **Sex**: "male" or "female"
    - **Age**: Age of the passenger
    - **SibSp**: Number of siblings/spouses aboard
    - **Parch**: Number of parents/children aboard
    - **Fare**: Ticket fare
    - **Embarked**: Port of embarkation ("S", "C", or "Q")
    """
    try:
        prediction = predict_survival(passenger)
        survival_status = "Survived" if prediction == 1 else "Did not survive"
        return {
            "prediction": prediction,
            "survival_status": survival_status,
            "passenger_data": passenger.dict()
        }
    except Exception as e:
        return {"error": str(e)}
