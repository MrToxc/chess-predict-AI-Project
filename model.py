import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
import numpy as np

# Nacteni datasetu

data = pd.read_csv("data/features.csv")
print(f"Nacteno {len(data)} her, {len(data.columns)} sloupcu")
data.head()

# Kodovani vystupu - result je kategoricky: "1-0", "0-1", "1/2-1/2"

le = LabelEncoder()
data['result_encoded'] = le.fit_transform(data['result'])
print("Tridy:", le.classes_)

# Nacteni vstupnich atributu z hlavicky CSV (vse krome 'result' a 'result_encoded')

input_features = [col for col in data.columns if col not in ['result', 'result_encoded']]
target_feature = 'result_encoded'

print(f"Pocet atributu: {len(input_features)}")

# Rozdeleni dat na trenovaci a testovaci

X_train, X_test, y_train, y_test = train_test_split(
    data[input_features], data[target_feature],
    test_size=0.1, random_state=1
)

print(f"Trenovaci data: {len(X_train)} | Testovaci data: {len(X_test)}")

# Standardizace dat

scaler = StandardScaler()
scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# =========================================================================
# POZNÁMKA K IMPLEMENTACI NEURONOVÉ SÍTĚ:
# V ukázkách z hodin se používá knihovna TensorFlow (Keras). Konkrétně:
# from tensorflow.keras.models import Sequential
# 
# Jelikož ale na mém počítači běží nejnovější verze Pythonu (3.14) a 
# TensorFlow pro tuto verzi ještě nevydalo kompatibilní build (nelze jej
# přes pip vůbec nainstalovat), vyřešil jsem neuronovou síť pomocí modulu
# MLPClassifier z knihovny scikit-learn.
#
# Jde o naprosto stejnou architekturu (Multi-Layer Perceptron).
# Parametr hidden_layer_sizes=(128, 64, 32) vytvoří tři skryté vrstvy.
# Výstupní vrstva o 3 neuronech (pro klasifikaci) a funkce softmax se 
# u tohoto modulu nastavují automaticky podle počtu tříd v y_train.
# =========================================================================

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), # 3 skryte vrstvy - ekvivalent k add(Dense(...))
    activation='relu',                # Aktivacni funkce pro skryte vrstvy
    solver='adam',                    # Optimizer 
    alpha=0.0005,
    batch_size=64,                    # Velikost davky
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,                     # Stejne jako epochs=500
    early_stopping=True,              # Zabrani pretrenovani pouzitim validation_data
    n_iter_no_change=30,
    validation_fraction=0.1,          # Vycleneni validacnich dat z trenovacich
    random_state=1,
    verbose=True,                     # Vypisuje prubeh uceni
)

# Trenovani

model.fit(X_train_std, y_train)

# Predikce a vyhodnoceni

y_pred = model.predict(X_test_std)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"\nMAE:      {mae:.4f}")
print(f"MSE:      {mse:.4f}")
print(f"Accuracy: {acc:.4f}")

print("\nClassification Report (Precision, Recall, F1 pro klasifikaci):")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Pravdepodobnosti vysledku (jako u sazkovych kancelari)

y_proba = model.predict_proba(X_test_std)

class_labels = list(le.classes_)
white_win_idx = class_labels.index('1-0')
draw_idx = class_labels.index('1/2-1/2')
black_win_idx = class_labels.index('0-1')

print(f"\n{'#':<4} {'White Win':>10} {'Draw':>10} {'Black Win':>10} | {'Actual':<10}")
print("-" * 55)

rng = np.random.RandomState(42)
sample_idx = rng.choice(len(y_test), size=20, replace=False)

for i in sample_idx:
    proba = y_proba[i]
    actual = le.inverse_transform([y_test.iloc[i]])[0]

    p_white = proba[white_win_idx] * 100
    p_draw = proba[draw_idx] * 100
    p_black = proba[black_win_idx] * 100

    print(f"{i:<4} {p_white:>9.1f}% {p_draw:>9.1f}% {p_black:>9.1f}% | {actual:<10}")
