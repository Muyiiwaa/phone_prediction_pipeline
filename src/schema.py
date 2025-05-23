from pydantic import BaseModel,Field
from typing import List, Literal


class UserRequest(BaseModel):
    predictors: List = Field(..., examples=[
        list(range(1, 21))
    ])

class UserResponse(BaseModel):
    phone_class : Literal["low cost", 
                          "medium cost", "high cost",
                          "very high cost"] = Field(..., examples=["high cost"])
