import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

st.title("🚗 Used-Car Price Estimator")

st.info("This is an App built for a project in our Computer Science class. We use Machine Learning and Multiple Regression Analysis of used car price data in order to estimate the price of any used car.")

with st.expander("Data"):
  st.subheader("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/kaiflury/KF_streamlit_CS_Project_Test/refs/heads/master/cleaned_car_data_final.csv")
  df

  st.write("**X**")
  X = df.drop("price", axis=1)
  X

  st.write("**Y**")
  Y = df.price
  Y

with st.expander("Data Visualization"):
  st.scatter_chart(data=df, x="milage", y="price", color="brand")


# 1. DATEN LADEN
print("Lade bereinigte Daten...")
df = pd.read_csv('merged_car_data.csv')

# 2. DATEN-VISUALISIERUNG (Anforderung 3)
# Ein kleiner Check: Wie verteilen sich die Preise? 
# Das hilft euch beim Verständnis eures Datensatzes.
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Verteilung der Gebrauchtwagenpreise')
plt.xlabel('Preis ($)')
plt.ylabel('Anzahl Autos')
plt.savefig('price_distribution.png') # Speichert die Grafik für euren Report
print("Visualisierung gespeichert unter 'price_distribution.png'")

# 3. FEATURE SELECTION (Eingabevariablen festlegen)
# Wir konzentrieren uns auf die wichtigsten Faktoren für den Preis
features = ['brand', 'model', 'year', 'milage', 'transmission', 'fuel_type']
X = df[features].copy()
y = df['price']

# 4. ENCODING (Text zu Zahlen)
# Computer verstehen kein "BMW", also geben wir jeder Marke eine Nummer.
# Wir speichern die Encoder, damit wir die User-Eingaben in der App später übersetzen können.
label_encoders = {}

for column in ['brand', 'model', 'transmission', 'fuel_type']:
    le = LabelEncoder()
    # Wir stellen sicher, dass alle Werte als String behandelt werden
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

# 5. TRAINING & TEST SPLIT
# 80% der Daten zum Lernen, 20% zum Testen der Genauigkeit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. DAS MODELL TRAINIEREN (Anforderung 5)
print("Training startet... (das kann einen Moment dauern)")
# Der RandomForestRegressor ist "Wald voller Entscheidungsbäume" – sehr robust!
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. EVALUIERUNG
# Wie gut ist das Modell?
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"Modell-Performance:")
print(f"Durchschnittlicher Fehler (MAE): ${mae:.2f}")
print(f"Bestimmtheitsmaß (R²): {r2:.2f} (1.0 wäre perfekt)")
print("-" * 30)

# 8. EXPORT FÜR DIE STREAMLIT-APP
# Wir "frieren" das Modell und die Übersetzer ein
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("Dateien 'car_price_model.pkl' und 'label_encoders.pkl' wurden erstellt!")