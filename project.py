from pathlib import Path


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# This CSV contains 90 SIMULATED records created for a school project.
# They are not real records from an actual business.
data = pd.read_csv(Path(__file__).resolve().parent / "data.csv")


print("\n==============================")
print(" BUSINESS DATASET")
print("==============================")
print(data)
print("\nDataset rows:", len(data))
print("Dataset columns:", len(data.columns))
print("Data source: Simulated sample data for demonstration")


correlation = data.corr(numeric_only=True)


print("\n==============================")
print(" CORRELATION MATRIX")
print("==============================")
print(correlation)


def insight(corr):
    if corr > 0.7:
        return "Strong positive relationship"
    elif corr > 0.3:
        return "Moderate positive relationship"
    elif corr < -0.7:
        return "Strong negative relationship"
    elif corr < -0.3:
        return "Moderate negative relationship"
    else:
        return "Weak or no linear relationship"


features = ["Ad_Spend", "Price", "Economy", "Competition"]
X = data[features]
y = data["Sales"]


# Keep a portion of the data unseen during training for evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)


# Evaluate predictions on the unseen test set.
test_predictions = model.predict(X_test)
coefficients = model.named_steps["linearregression"].coef_
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)


print("\n==============================")
print(" MODEL EVALUATION")
print("==============================")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("MAE =", round(mae, 3), "sales units")
print("R² =", round(r2, 3))
print("MAE is the average absolute prediction error; lower is better.")
print("R² measures how well the model explains sales variation on the test set.")


print("\n==============================")
print(" CORRELATION-BASED BUSINESS INSIGHTS")
print("==============================")
for col in features:
    corr_value = correlation.loc[col, "Sales"]
    print(f"{col} -> {insight(corr_value)} (corr = {corr_value:.2f})")


print("\n==============================")
print(" MODEL COEFFICIENTS")
print("==============================")
for feature, coefficient in zip(features, coefficients):
    print(f"{feature}: {coefficient:.3f}")
print("A coefficient estimates the change in predicted Sales for a one-unit")
print("increase in that feature, while the other features are held constant.")
print("These are associations, not proof of cause and effect.")


def get_value(feature):
    low, high = data[feature].min(), data[feature].max()
    while True:
        text = input(f"{feature} (between {low:.1f} and {high:.1f}): ")
        try:
            value = float(text)
        except ValueError:
            print("Please enter a number.")
            continue
        if low <= value <= high:
            return value
        print("Out of range, the model is only reliable inside the data's range.")


print("\n==============================")
print(" SALES PREDICTION")
print("==============================")


choice = input("Press Enter to use the example scenario, or type 'c' for custom input: ")


if choice.strip().lower() == "c":
    values = [get_value(col) for col in features]
else:
    values = [3500, 8, 70, 5]   # example scenario


future_input = pd.DataFrame([values], columns=features)
predicted_sales = model.predict(future_input)


for col in features:
    print(f"{col} = {future_input.loc[0, col]}")
print("Predicted Sales =", round(predicted_sales[0], 2))
print("This is an estimate from simulated data, not a dependable real-world forecast.")


print("\n==============================")
print(" LIMITATIONS")
print("==============================")
print("- The dataset is simulated and does not represent a real business.")
print("- Only 90 records are available, so results may change with other data.")
print("- Correlation does not prove that a factor causes sales to change.")
print("- Related input features can make individual effects hard to separate.")
print("- Test metrics are estimates based on this particular train/test split.")


# Scatter plots: each input feature compared with Sales.
for col in features:
    plt.figure()
    plt.scatter(data[col], data["Sales"])
    plt.xlabel(col)
    plt.ylabel("Sales")
    plt.title(f"{col} vs Sales")
    plt.tight_layout()
    plt.show()


# Correlation heatmap.
plt.figure()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Coefficient Bar Graph


plt.figure(figsize=(8, 5))
plt.bar(features, coefficients)


plt.axhline(y=0, color="black", linewidth=0.8)
plt.xlabel("Business Factors")
plt.ylabel("Regression Coefficient")
plt.title("Regression Coefficients of Business Factors")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()