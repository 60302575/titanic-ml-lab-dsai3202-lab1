import os
import pandas as pd
from pydantic import BaseModel
import joblib

# Load joblib model from default path or MODEL_PATH env variable
def load_model():
    path = os.getenv("MODEL_PATH", "models/titanic_pipeline.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Place your trained joblib model there or set MODEL_PATH."
        )
    return joblib.load(path)

model = load_model()
    
# Define the expected input schema for a passenger
class Passenger(BaseModel):
    Pclass: int       # Passenger class (1, 2, or 3)
    Sex: str          # "male" or "female"
    Age: float        # Age of the passenger
    SibSp: int        # Number of siblings/spouses aboard
    Parch: int        # Number of parents/children aboard
    Fare: float       # Ticket fare
    Embarked: str     # Port of embarkation ("S" for Southampton, "C" for Cherbourg, "Q" for Queenstown)

def predict_survival(passenger: Passenger) -> int:
    df = pd.DataFrame([passenger.dict()])
    pred = model.predict(df)[0]
    return int(pred)
