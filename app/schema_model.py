from pydantic import BaseModel
from enum import Enum
# creates subclasses for some type of selection before using them in the main base model
class Online_security(str, Enum):
    Yes ='Yes'
    No = 'No'
    No_internet = 'No internet Service'

class Input(BaseModel):
    age : int 
    tenure : int 
    monthly_charges : float
    total_charges : float
    gender_Male : str
    online_security : Online_security
    # tenure_group
    # avg_monthly_charges
    # high_value_customer
    # service_count
    # contract_One year
    # contract_Two year
    # internet_service_Fiber optic
    # internet_service_No
    # online_security_No internet service
    # online_security_Yes
    # tech_support_No internet service
    # tech_support_Yes
    # streaming_tv_No internet service
    # streaming_tv_Yes
    # payment_method_Credit card
    # payment_method_Electronic check
    # payment_method_Mailed check
    # paperless_billing_Yes
