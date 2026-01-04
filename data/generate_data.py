# package imports 
import pandas as pd
import numpy as np

# create reandom seed for the data gen
np.random.seed(42)

# generate the data itself
n_samples = 1000

data = {
    'customer_id': range(1, n_samples + 1),
    'tenure': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples),
    'total_charges': np.random.uniform(100, 8000, n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'tech_support': np.random.choice(['Yes', 'No'], n_samples),
}

# now create target (the churn) with some logic
df = pd.DataFrame(data)

# write the churn logic
churn_probability = (
    (df['tenure'] < 12) * 0.3 +
    (df['contract_type'] == 'Month-to-month') * 0.25 +
    (df['monthly_charges'] > 80) * 0.2 +
    (df['tech_support'] == 'No') * 0.15
)
# add as a col to the df
df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)

# save as csv and print some logs
df.to_csv('data/churn_data.csv', index=False)

print(f'Generated {n_samples} samples')
print(f'Churn rate: {df['churn'].mean():.2%}') # the :.2% is a format specifier --> rounds to 2 decimal places and displays as percentage


