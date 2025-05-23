from fastapi import FastAPI, HTTPException
import uvicorn
from schema import UserRequest, UserResponse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Union, Optional
import logfire
from dotenv import load_dotenv
import os


# load the environment variable
load_dotenv()
logfire_key = os.getenv(key= "LOGFIRE_TOKEN")
logfire.configure(token=logfire_key, 
                  service_name= "Mobile Phone Project")

app = FastAPI(
    title= "Mobile Phone Prediction",
    version= "version 1.0.0"
)

# instrument the application for monitoring.
logfire.instrument_fastapi(app)

column_names = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi']

@app.get("/")
async def root():
    return {
        "message": "we are live!"
    }

@app.post("/predict/", response_model= UserResponse)
def predict(payload: UserRequest) -> UserResponse:
    pred : Optional[Union[int, str]] = None
    try:
        input_values = payload.predictors
        # create a payload data
        input_values =  {k: [v] for k,v in zip(column_names, input_values)}
        input_values = pd.DataFrame(input_values)
        logfire.info(f"payload converted to dataframe successfully!")
        model = joblib.load(filename= "/workspaces/phone_prediction_pipeline/src/model.pkl")
        pred = model.predict(input_values)
        output_style = ['low cost', 'medium cost', 'high cost', 'very high cost']
        pred: str = output_style[pred[0]]
        logfire.info(f"Prediction Successful!")
        
    except Exception as e:
        logfire.error(f"Encountered error {e} while making prediction.")
        raise HTTPException(status_code=500, detail= f"Error: {e}")
    
    return UserResponse(
        phone_class= pred
    )



if __name__ == "__main__":
    uvicorn.run("main:app", host = "localhost", reload=True)

