import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------
# Title
# ------------------------------
st.image("PragyanAI_Transperent.png")
st.title("🚕 PragyanAI Taxi Fare Prediction App (End-to-End ML)")

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
    df = pd.read_csv(url)
    df = df.convert_dtypes()
    st.write(df.head())   # instead of st.dataframe(df)
    return df

df = load_data()

st.subheader("PragyanAI Dataset Preview")
#st.dataframe(df.head())
#st.write(df.head()) 
# ------------------------------
# Step 2: Data Preprocessing
# ------------------------------
df = df[['distance', 'fare']].dropna()
# Force numeric conversion for key columns
df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
df['fare'] = pd.to_numeric(df['fare'], errors='coerce')
X = df[['distance']]
y = df['fare']

# ------------------------------
# Step 3: Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Step 4: Train Model
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Step 5: Model Evaluation
# ------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📈 Model Performance")
st.write(f"R² Score: {r2:.2f}")
st.write(f"RMSE: {rmse:.2f}")

# ------------------------------
# Step 6: User Input (Step-by-Step UI)
# ------------------------------
st.subheader("🧮 Enter Trip Details")

distance = st.number_input(
    "Step 1: Enter Distance (km)",
    min_value=0.0,
    value=5.0
)

# Optional extension inputs (future scaling)
passengers = st.number_input(
    "Step 2: Number of Passengers",
    min_value=1,
    value=1
)

hour = st.number_input(
    "Step 3: Hour of Day (0-23)",
    min_value=0,
    max_value=23,
    value=12
)

# ------------------------------
# Step 7: Prediction
# ------------------------------
if st.button("🚀 Predict Fare"):
    
    input_data = np.array([[distance]])
    
    prediction = model.predict(input_data)
    
    st.success(f"💰 Estimated Fare: ${prediction[0]:.2f}")

# ------------------------------
# Step 8: Visualization
# ------------------------------
st.subheader("📉 Distance vs Fare")

# Create clean copy
chart_df = df[['distance', 'fare']].copy()

# Convert EVERYTHING safely to numeric
chart_df['distance'] = pd.to_numeric(chart_df['distance'], errors='coerce')
chart_df['fare'] = pd.to_numeric(chart_df['fare'], errors='coerce')

# Drop invalid rows
chart_df = chart_df.dropna()

# Reset index (important!)
chart_df = chart_df.reset_index(drop=True)

# Plot using matplotlib (NO Arrow dependency)
fig, ax = plt.subplots()

ax.scatter(chart_df['distance'], chart_df['fare'])
ax.set_xlabel("Distance")
ax.set_ylabel("Fare")
ax.set_title("Distance vs Fare")

st.pyplot(fig)

# ------------------------------
# Step 9: Insights
# ------------------------------
st.subheader("💡 Insights")

st.write("✔ Fare increases with distance")
st.write("✔ Linear model used for simplicity")
st.write("✔ Can improve using more features (time, location, traffic)")
