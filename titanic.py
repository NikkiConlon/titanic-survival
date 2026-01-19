
# Titanic Survival Analysis & Prediction

# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Training Data
train_data = pd.read_csv("train.csv")

# Add readable survival status
train_data["Survival Status"] = train_data["Survived"].map({0: "Did Not Survive", 1: "Survived"})

# Create a clean survival list
survival_list = train_data[[
    "PassengerId",
    "Name",
    "Sex",
    "Age",
    "Pclass",
    "Survival Status"
]]

# Separate survivors and non-survivors and save
survivors = survival_list[survival_list["Survival Status"] == "Survived"]
non_survivors = survival_list[survival_list["Survival Status"] == "Did Not Survive"]

survivors.to_csv("survivors.csv", index=False)
non_survivors.to_csv("non_survivors.csv", index=False)

# Exploratory Data Analysis (Charts)

# Chart 1: Overall Survival Count
train_data["Survival Status"].value_counts().plot(kind="bar")
plt.title("Titanic Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")
plt.show()

# Chart 2: Survival Rate by Gender
gender_survival = train_data.groupby("Sex")["Survived"].mean() * 100
gender_survival.plot(kind="bar")
plt.title("Survival Rate by Gender (%)")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.show()

# Chart 3: Survival Rate by Passenger Class
class_survival = train_data.groupby("Pclass")["Survived"].mean() * 100
class_survival.plot(kind="bar")
plt.title("Survival Rate by Passenger Class (%)")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.show()

# Chart 4: Age Distribution of Survivors
survivors_age = train_data[train_data["Survived"] == 1]["Age"].dropna()
plt.hist(survivors_age, bins=20)
plt.title("Age Distribution of Survivors")
plt.xlabel("Age")
plt.ylabel("Number of Survivors")
plt.show()

# Chart 5: Age Distribution of Non-Survivors
non_survivors_age = train_data[train_data["Survived"] == 0]["Age"].dropna()
plt.hist(non_survivors_age, bins=20)
plt.title("Age Distribution of Non-Survivors")
plt.xlabel("Age")
plt.ylabel("Number of Non-Survivors")
plt.show()

# Machine Learning: Survival Prediction

# Prepare features
ml_data = train_data[["Sex", "Age", "Pclass", "Survived"]].copy()
ml_data["Age"].fillna(ml_data["Age"].median(), inplace=True)
ml_data["Sex"] = ml_data["Sex"].map({"male": 0, "female": 1})

X = ml_data[["Sex", "Age", "Pclass"]]
y = ml_data["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test split
y_pred = model.predict(X_test)

# Evaluate model
print("\n=== MACHINE LEARNING RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict Survival on Test Data

test_data = pd.read_csv("test.csv")
test_passenger_ids = test_data["PassengerId"]
test_features = test_data[["Sex", "Age", "Pclass"]].copy()

# Same preprocessing as training data
test_features["Age"].fillna(ml_data["Age"].median(), inplace=True)
test_features["Sex"] = test_features["Sex"].map({"male": 0, "female": 1})

# Predict and save
test_predictions = model.predict(test_features)
submission = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Survived": test_predictions
})
submission.to_csv("ml_submission.csv", index=False)
print("\nTest predictions saved as ml_submission.csv")

# Confusion Matrix Visualization

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.xticks([0, 1], ["Did Not Survive", "Survived"])
plt.yticks([0, 1], ["Did Not Survive", "Survived"])

# Add numbers inside cells
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()
