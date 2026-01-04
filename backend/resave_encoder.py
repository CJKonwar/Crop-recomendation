import joblib
from sklearn.preprocessing import LabelEncoder

# same crop labels you used in training
crop_labels = [
    "rice", "wheat", "maize", "chickpea", "kidneybeans",
    "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil",
    "pomegranate", "banana", "mango", "grapes", "watermelon",
    "muskmelon", "apple", "orange", "papaya", "coconut",
    "cotton", "jute", "coffee"
]

encoder = LabelEncoder()
encoder.fit(crop_labels)

joblib.dump(encoder, "label_encoder.pkl")
print("✅ Label encoder re-saved successfully with scikit-learn 1.7.1")