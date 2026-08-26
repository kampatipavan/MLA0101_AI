# ============================================================
# HEART DISEASE RISK PREDICTION
# USING DECISION TREE CLASSIFIER
# ============================================================

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# 1. CREATE DEMO DATASET
# ============================================================

# Synthetic educational dataset
# target:
# 0 = No Heart Disease
# 1 = Heart Disease

data = {
    "age": [
        25, 32, 45, 51, 62, 58, 40, 67, 55, 48,
        29, 35, 60, 52, 44, 70, 63, 38, 57, 49,
        31, 46, 59, 65, 42, 54, 68, 36, 50, 61,
        27, 43, 56, 64, 47, 53, 69, 34, 41, 66
    ],

    "sex": [
        0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
        0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
        0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
        0, 1, 0, 1, 0, 1, 1, 0, 1, 1
    ],

    "cp": [
        0, 0, 1, 2, 3, 2, 1, 3, 2, 1,
        0, 1, 3, 2, 1, 3, 2, 0, 3, 2,
        0, 1, 2, 3, 1, 2, 3, 0, 2, 3,
        0, 1, 2, 3, 1, 2, 3, 0, 1, 2
    ],

    "trestbps": [
        110, 118, 125, 145, 160, 150, 120, 170, 155, 135,
        115, 130, 165, 140, 128, 175, 158, 112, 168, 142,
        108, 132, 150, 162, 122, 148, 172, 116, 138, 155,
        112, 126, 153, 166, 124, 145, 178, 118, 130, 160
    ],

    "chol": [
        180, 195, 210, 245, 280, 260, 190, 290, 275, 225,
        175, 215, 300, 250, 205, 310, 285, 185, 295, 240,
        170, 220, 265, 305, 200, 255, 315, 188, 230, 270,
        178, 212, 258, 298, 198, 248, 320, 190, 218, 278
    ],

    "fbs": [
        0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 0, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 1,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 1
    ],

    "restecg": [
        0, 0, 0, 1, 1, 1, 0, 2, 1, 0,
        0, 1, 2, 1, 0, 2, 1, 0, 2, 1,
        0, 0, 1, 2, 0, 1, 2, 0, 1, 2,
        0, 1, 1, 2, 0, 1, 2, 0, 1, 2
    ],

    "thalach": [
        180, 175, 165, 145, 125, 135, 172, 118, 130, 155,
        185, 160, 120, 140, 168, 110, 122, 178, 115, 145,
        188, 158, 132, 108, 170, 138, 105, 182, 150, 125,
        190, 165, 128, 112, 172, 142, 100, 180, 160, 120
    ],

    "exang": [
        0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 0, 1, 1,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 1,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 1
    ],

    "oldpeak": [
        0.0, 0.1, 0.2, 1.2, 2.5, 2.0, 0.3, 3.0, 2.2, 0.8,
        0.0, 0.5, 2.8, 1.5, 0.2, 3.2, 2.6, 0.0, 3.0, 1.4,
        0.0, 0.4, 2.0, 3.1, 0.1, 1.8, 3.5, 0.0, 0.7, 2.4,
        0.0, 0.3, 2.1, 3.0, 0.2, 1.6, 3.6, 0.0, 0.5, 2.7
    ],

    "slope": [
        2, 2, 2, 1, 0, 1, 2, 0, 1, 2,
        2, 2, 0, 1, 2, 0, 1, 2, 0, 1,
        2, 2, 1, 0, 2, 1, 0, 2, 2, 1,
        2, 2, 1, 0, 2, 1, 0, 2, 2, 1
    ],

    "ca": [
        0, 0, 0, 1, 2, 1, 0, 3, 2, 0,
        0, 0, 3, 1, 0, 3, 2, 0, 3, 1,
        0, 0, 2, 3, 0, 2, 3, 0, 1, 2,
        0, 0, 2, 3, 0, 2, 3, 0, 1, 2
    ],

    "thal": [
        2, 2, 2, 3, 3, 3, 2, 3, 3, 2,
        2, 2, 3, 3, 2, 3, 3, 2, 3, 3,
        2, 2, 3, 3, 2, 3, 3, 2, 2, 3,
        2, 2, 3, 3, 2, 3, 3, 2, 2, 3
    ],

    "target": [
        0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 0, 1, 1,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 1,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 1
    ]
}


df = pd.DataFrame(data)


# ============================================================
# 2. DISPLAY DATASET
# ============================================================

print("\n" + "=" * 60)
print("HEART DISEASE RISK PREDICTION")
print("DECISION TREE CLASSIFIER")
print("=" * 60)

print("\nDemo Dataset:")
print(df.head(10))

print("\nTotal Records:", len(df))


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. CREATE DECISION TREE
# ============================================================

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 7. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Disease", "Disease"]
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)


# ============================================================
# 12. GUI
# ============================================================

root = tk.Tk()

root.title("Heart Disease Risk Prediction")
root.geometry("700x850")

root.configure(bg="#f4f6f8")


# ============================================================
# TITLE
# ============================================================

title = tk.Label(
    root,
    text="HEART DISEASE RISK PREDICTION",
    font=("Arial", 22, "bold"),
    bg="#f4f6f8"
)

title.pack(pady=(20, 5))


subtitle = tk.Label(
    root,
    text="Decision Tree Machine Learning System",
    font=("Arial", 12),
    bg="#f4f6f8"
)

subtitle.pack(pady=(0, 15))


# ============================================================
# INPUT FRAME
# ============================================================

input_frame = tk.Frame(
    root,
    bg="white",
    bd=2,
    relief="groove"
)

input_frame.pack(
    padx=30,
    pady=10,
    fill="x"
)


entries = {}


# ============================================================
# INPUT INFORMATION
# ============================================================

fields = [
    ("Age", "age"),
    ("Sex (1=Male, 0=Female)", "sex"),
    ("Chest Pain Type (0-3)", "cp"),
    ("Resting Blood Pressure", "trestbps"),
    ("Cholesterol", "chol"),
    ("Fasting Blood Sugar (1/0)", "fbs"),
    ("Rest ECG (0-2)", "restecg"),
    ("Maximum Heart Rate", "thalach"),
    ("Exercise Angina (1/0)", "exang"),
    ("Oldpeak", "oldpeak"),
    ("Slope (0-2)", "slope"),
    ("CA (0-4)", "ca"),
    ("Thal (0-3)", "thal")
]


for row, (label_text, column_name) in enumerate(fields):

    label = tk.Label(
        input_frame,
        text=label_text,
        font=("Arial", 10),
        bg="white"
    )

    label.grid(
        row=row,
        column=0,
        padx=15,
        pady=5,
        sticky="w"
    )

    entry = tk.Entry(
        input_frame,
        width=25,
        font=("Arial", 10)
    )

    entry.grid(
        row=row,
        column=1,
        padx=15,
        pady=5
    )

    entries[column_name] = entry


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease():

    try:

        values = []

        for column in X.columns:

            value = float(entries[column].get())

            values.append(value)


        # Convert input into DataFrame
        input_data = pd.DataFrame(
            [values],
            columns=X.columns
        )


        # Prediction
        prediction = model.predict(input_data)[0]


        # Probability
        probabilities = model.predict_proba(input_data)[0]


        disease_probability = probabilities[1] * 100
        no_disease_probability = probabilities[0] * 100


        # ====================================================
        # RESULT
        # ====================================================

        if prediction == 1:

            result = (
                "⚠ HEART DISEASE RISK DETECTED\n\n"
                f"Disease Probability: "
                f"{disease_probability:.2f}%\n\n"
                "Please consult a qualified healthcare professional."
            )

        else:

            result = (
                "✓ LOW HEART DISEASE RISK\n\n"
                f"No Disease Probability: "
                f"{no_disease_probability:.2f}%"
            )


        result_label.config(
            text=result
        )


    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numerical values in all fields."
        )


    except Exception as e:

        messagebox.showerror(
            "Error",
            str(e)
        )


# ============================================================
# LOAD SAMPLE LOW-RISK INPUT
# ============================================================

def load_low_risk():

    sample = {
        "age": 32,
        "sex": 0,
        "cp": 0,
        "trestbps": 118,
        "chol": 195,
        "fbs": 0,
        "restecg": 0,
        "thalach": 175,
        "exang": 0,
        "oldpeak": 0.1,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }

    for column in X.columns:

        entries[column].delete(0, tk.END)

        entries[column].insert(
            0,
            str(sample[column])
        )


# ============================================================
# LOAD SAMPLE HIGH-RISK INPUT
# ============================================================

def load_high_risk():

    sample = {
        "age": 62,
        "sex": 1,
        "cp": 3,
        "trestbps": 160,
        "chol": 280,
        "fbs": 1,
        "restecg": 1,
        "thalach": 125,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 0,
        "ca": 2,
        "thal": 3
    }

    for column in X.columns:

        entries[column].delete(0, tk.END)

        entries[column].insert(
            0,
            str(sample[column])
        )


# ============================================================
# BUTTON FRAME
# ============================================================

button_frame = tk.Frame(
    root,
    bg="#f4f6f8"
)

button_frame.pack(pady=10)


# ============================================================
# SAMPLE BUTTONS
# ============================================================

low_button = tk.Button(
    button_frame,
    text="Load Low-Risk Sample",
    command=load_low_risk,
    font=("Arial", 10, "bold"),
    padx=10,
    pady=8
)

low_button.grid(
    row=0,
    column=0,
    padx=5
)


high_button = tk.Button(
    button_frame,
    text="Load High-Risk Sample",
    command=load_high_risk,
    font=("Arial", 10, "bold"),
    padx=10,
    pady=8
)

high_button.grid(
    row=0,
    column=1,
    padx=5
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = tk.Button(
    root,
    text="🔍  PREDICT DISEASE RISK",
    command=predict_disease,
    font=("Arial", 14, "bold"),
    padx=30,
    pady=12
)

predict_button.pack(pady=10)


# ============================================================
# RESULT
# ============================================================

result_label = tk.Label(
    root,
    text="Enter patient information\nand click Predict",
    font=("Arial", 14, "bold"),
    bg="white",
    bd=2,
    relief="groove",
    width=55,
    height=5,
    justify="center"
)

result_label.pack(
    padx=30,
    pady=10
)


# ============================================================
# MODEL ACCURACY
# ============================================================

accuracy_label = tk.Label(
    root,
    text=f"Decision Tree Accuracy: {accuracy * 100:.2f}%",
    font=("Arial", 12, "bold"),
    bg="#f4f6f8"
)

accuracy_label.pack(pady=5)


# ============================================================
# DISCLAIMER
# ============================================================

disclaimer = tk.Label(
    root,
    text=(
        "Educational demonstration only.\n"
        "This system is NOT a medical diagnosis."
    ),
    font=("Arial", 9),
    bg="#f4f6f8"
)

disclaimer.pack(pady=10)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()