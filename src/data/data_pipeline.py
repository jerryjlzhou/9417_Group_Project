import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

PROCESSED_PATH = '../../dataset/processed/AirQualityUCI_processed.csv'

# Load dataset
file_path = "../../dataset/raw/AirQualityUCI.csv"

df = pd.read_csv(file_path, sep=';', decimal=',')

# Drop trailing rows & columns
df = df.iloc[:,:-2]
df.dropna(how='all', inplace=True)

# Replace sentinel value with NaN
df.replace(-200, np.nan, inplace=True)

# Combine date and time
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
df = df.drop(columns=['Date', 'Time', 'NMHC(GT)']) # also drop the unusable NMHC col
df = df[['DateTime'] + [col for col in df.columns if col != 'DateTime']]

# Create derived features (hour, weekday, month)
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.dayofweek # 0=Monday, 6=Sunday
df['Month'] = df['DateTime'].dt.month

# Linear interpolation to handle missing values
df.interpolate(method='linear', inplace=True)

# Normalise continuous functions (minmax)
cols_to_scale = [col for col in df.columns if col not in ['DateTime', 'Hour', 'Day', 'Month']]
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


df.to_csv(PROCESSED_PATH, index=False)

print(f"Processed DataFrame successfully created and stored.")
print(f"CSV file saved to: {PROCESSED_PATH}")
print("\nFirst 5 rows of the processed DataFrame:")
print(df.head())