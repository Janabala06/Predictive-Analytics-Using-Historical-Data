import pandas as pd
from sklearn.linear_model import LinearRegression

# load dataset
data = pd.read_csv("monthly_sales_data.csv")
data["Month"] = pd.to_datetime(data["Month"])

# numeric time feature
data["Month_Num"] = data["Month"].dt.year * 12 + data["Month"].dt.month

# features and target
X = data[["Month_Num", "Marketing_Spend", "Discount_Percent", "Customer_Count"]]
y = data["Sales"]

# train model on full data
model = LinearRegression()
model.fit(X, y)

# create future months (next 6 months)
future_months = pd.date_range(start=data["Month"].max(), periods=7, freq="ME")[1:]

future_data = pd.DataFrame({
    "Month": future_months,
    "Month_Num": future_months.year * 12 + future_months.month,
    "Marketing_Spend": [7000, 7200, 7400, 7600, 7800, 8000],
    "Discount_Percent": [30, 30, 35, 35, 40, 40],
    "Customer_Count": [430, 450, 470, 490, 510, 530]
})

# predict future sales
future_data["Predicted_Sales"] = model.predict(
    future_data[["Month_Num", "Marketing_Spend", "Discount_Percent", "Customer_Count"]]
)

# save predictions
future_data.to_csv("sales_predictions.csv", index=False)

print(future_data)