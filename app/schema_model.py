from pydantic import BaseModel, Field
from enum import Enum

# creates subclasses for some type of selection before using them in the main base model
class Online_security(str, Enum):
    Yes ='Yes'
    No = 'No'
    No_internet = 'No internet Service'

class Gender(str, Enum):
    male = 'Male'
    female = 'Female'

class Contract(str, Enum):
    Month_to_month  = 'Month-to-month'    
    One_year  =  'One year'  
    Two_year =' Two year'

class Internet(str, Enum):
    Fiber_optic = 'Fiber optic' 
    DSL  = 'DSL'          
    No   = 'No' 

class Paperless_Billing(str, Enum):
    Yes = 'Yes'
    No = 'No'

class Payment_method(str, Enum):
    Electronic_check = 'Electronic check'
    Bank_transfer = 'Bank transfer'
    Credit_card = 'Credit card'
    Mailed_check = 'Mailed check'

class Churn_Input(BaseModel):
    age : int = Field(..., description='Age of User', example= 45) 
    tenure : int = Field(..., description='Tenure', example= 45)
    monthly_charges : float = Field(..., description='Monthly charges', example= 45)
    total_charges : float = Field(..., description='Total Charges', example= 45)
    gender : Gender = Field(..., description='Gender')
    online_security : Online_security = Field(..., description='Do the user have online security')
    contract: Contract  = Field(..., description='Contract Type')
    internet_service: Internet  = Field(..., description='What type of Internet service is being used')
    tech_support : Online_security  = Field(..., description='Do the user have tech support')
    streaming_tv : Online_security  = Field(..., description='Do the TV Stream')
    payment_method : Payment_method  = Field(..., description='Whats the payment method')
    paperless_billing : Paperless_Billing  = Field(..., description='Paperless billing?')



