import pandas as pd
from prophet import Prophet

df = pd.read_csv("D:/Codes/Python/Sources/data/example_wp_log_peyton_manning.csv")

# print(df.head())
# print("Hello world")

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
print(future.tail())
