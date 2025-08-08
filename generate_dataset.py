import pandas as pd
import numpy as np
from pathlib import Path

def generate_customer_churn_dataset(n_samples=5000):
    """Generate a realistic customer churn dataset"""
    
    np.random.seed(42)
    
    # Customer demographics
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
    ages = np.random.normal(45, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.51, 0.49])
    
    # Service details
    tenures = np.random.exponential(24, n_samples).astype(int)
    tenures = np.clip(tenures, 1, 72)
    
    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                n_samples, p=[0.55, 0.25, 0.20])
    
    internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                       n_samples, p=[0.35, 0.45, 0.20])
    
    # Monthly charges
    monthly_charges = []
    for service in internet_services:
        if service == 'No':
            charge = np.random.uniform(20, 40)
        elif service == 'DSL':
            charge = np.random.uniform(35, 65)
        else:
            charge = np.random.uniform(65, 120)
        monthly_charges.append(round(charge, 2))
    
    monthly_charges = np.array(monthly_charges)
    total_charges = monthly_charges * tenures * np.random.uniform(0.85, 1.15, n_samples)
    total_charges = np.round(total_charges, 2)
    
    # Additional services
    def generate_service(internet_service, base_prob=0.3):
        if internet_service == 'No':
            return 'No internet service'
        else:
            return np.random.choice(['Yes', 'No'], p=[base_prob, 1-base_prob])
    
    online_security = [generate_service(service, 0.35) for service in internet_services]
    tech_support = [generate_service(service, 0.30) for service in internet_services]
    streaming_tv = [generate_service(service, 0.40) for service in internet_services]
    
    payment_methods = np.random.choice(['Electronic check', 'Mailed check', 
                                      'Bank transfer', 'Credit card'], 
                                     n_samples, p=[0.35, 0.15, 0.25, 0.25])
    
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35])
    
    # Generate churn
    churn_probability = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.15
        
        if contracts[i] == 'Month-to-month':
            prob += 0.25
        elif contracts[i] == 'Two year':
            prob -= 0.10
        
        if tenures[i] < 6:
            prob += 0.20
        elif tenures[i] > 36:
            prob -= 0.15
        
        if monthly_charges[i] > 80:
            prob += 0.15
        
        if payment_methods[i] == 'Electronic check':
            prob += 0.10
        
        services_count = sum([1 for service in [online_security[i], tech_support[i], streaming_tv[i]]
                             if service == 'Yes'])
        prob -= services_count * 0.05
        
        churn_probability[i] = max(0.05, min(0.80, prob))
    
    churn = np.random.binomial(1, churn_probability, n_samples)
    churn_labels = ['Yes' if c == 1 else 'No' for c in churn]
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'tenure': tenures,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contracts,
        'internet_service': internet_services,
        'online_security': online_security,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'payment_method': payment_methods,
        'paperless_billing': paperless_billing,
        'churn': churn_labels
    })
    
    return df

def save_dataset():
    """Generate and save the dataset"""
    # Create data directory if it doesn't exist
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    df = generate_customer_churn_dataset(5000)
    
    # Save to CSV
    df.to_csv("data/raw/customer_churn.csv", index=False)
    print(f"Dataset saved with {len(df)} samples")
    print(f"Churn rate: {(df['churn'] == 'Yes').mean():.2%}")

if __name__ == "__main__":
    save_dataset()