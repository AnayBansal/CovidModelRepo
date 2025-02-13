import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import ExponentialSmoothing
from darts import TimeSeries
from darts.metrics import mape, rmse, mae

#Generate Synthetic COVID-19 Case Data
dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
cases = np.cumsum(np.random.randint(50, 300, size=100))

# Create DataFrame
df = pd.DataFrame({"Date": dates, "Cases": cases})

# Convert to Darts TimeSeries format
series = TimeSeries.from_dataframe(df, time_col="Date", value_cols="Cases")

# Train an Exponential Smoothing Model
train, test = series.split_after(0.8)  # Use 80% for training, 20% for testing
model = ExponentialSmoothing()
model.fit(train)

# Make Forecast for the Next 20 Days
forecast = model.predict(len(test))

#  Evaluate the Model
mape_score = mape(test, forecast)  # Mean Absolute Percentage Error

# Print Evaluation Metrics
print(f"MAPE: {mape_score:.2f}%")


# Plot the Results
plt.figure(figsize=(10, 5))
train.plot(label="Training Data")
test.plot(label="Actual Cases")
forecast.plot(label="Forecasted Cases", linestyle="dashed")
plt.title("COVID-19 Case Forecast")
plt.legend()
plt.show()