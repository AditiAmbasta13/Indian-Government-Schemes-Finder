# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
schemes_df = pd.read_excel('government_schemes.xlsx', sheet_name=0)

# Encode categorical features
le_gender = LabelEncoder()
le_age_group = LabelEncoder()
le_income_category = LabelEncoder()
le_primary_benefit = LabelEncoder()

schemes_df['Gender'] = le_gender.fit_transform(schemes_df['Gender'])
schemes_df['Age_Group'] = le_age_group.fit_transform(schemes_df['Age_Group'])
schemes_df['Income_Category'] = le_income_category.fit_transform(schemes_df['Income_Category'])
schemes_df['Primary_Benefit'] = le_primary_benefit.fit_transform(schemes_df['Primary_Benefit'])

# Select features and target
X = schemes_df[['Gender', 'Age_Group', 'Income_Category']]
y = schemes_df['Primary_Benefit']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and label encoders
joblib.dump(model, 'scheme_recommendation_model.joblib')
joblib.dump(le_gender, 'le_gender.joblib')
joblib.dump(le_age_group, 'le_age_group.joblib')
joblib.dump(le_income_category, 'le_income_category.joblib')
joblib.dump(le_primary_benefit, 'le_primary_benefit.joblib')
