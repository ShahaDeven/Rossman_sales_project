# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# %%
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
store_df = pd.read_csv('store.csv')

# %%
train_df

# %%
test_df

# %%
store_df

# %%
merged_df = train_df.merge(store_df, how='left', on='Store')
merged_test_df = test_df.merge(store_df, how='left', on='Store')

# %%
merged_test_df

# %%
merged_df

# %%
merged_df.info()

# %%
merged_df.isnull().sum()

# %%
merged_df.describe()

# %%
merged_df.Date = pd.to_datetime(merged_df.Date)
merged_test_df.Date = pd.to_datetime(merged_test_df.Date)

# %%
sns.histplot(merged_df.Sales)

# %%
plt.figure(figsize=(10, 6))
plt.bar(merged_df.groupby('DayOfWeek').Sales.mean().index, merged_df.groupby('DayOfWeek').Sales.mean().values)
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.show()

# %%
merged_df['Open'].value_counts()

# %%
plt.hist(merged_df[merged_df['Open'] == True]['Sales'])  
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.title('Histogram of Sales for Open Entries')
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.boxplot(merged_df.select_dtypes(include='number').values, labels=merged_df.select_dtypes(include='number').columns)
plt.xticks(rotation=45) 
plt.xlabel('Columns')
plt.ylabel('Values')
plt.title('Boxplot for Each Column')
plt.show()

# %%
Q1 = merged_df['Sales'].quantile(0.25)  
Q3 = merged_df['Sales'].quantile(0.75)  
IQR = Q3 - Q1  

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

merged_df = merged_df[(merged_df['Sales'] >= lower_bound) & (merged_df['Sales'] <= upper_bound)]

# %%
plt.figure(figsize=(12, 6))
plt.boxplot(merged_df.select_dtypes(include='number').values, labels=merged_df.select_dtypes(include='number').columns)
plt.xticks(rotation=45) 
plt.xlabel('Columns')
plt.ylabel('Values')
plt.title('Boxplot for Each Column')
plt.show()

# %%
Q1 = merged_df['Customers'].quantile(0.25)  
Q3 = merged_df['Customers'].quantile(0.75)  
IQR = Q3 - Q1  

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

merged_df = merged_df[(merged_df['Customers'] >= lower_bound) & (merged_df['Customers'] <= upper_bound)]

# %%
plt.figure(figsize=(12, 6))
plt.boxplot(merged_df.select_dtypes(include='number').values, labels=merged_df.select_dtypes(include='number').columns)
plt.xticks(rotation=45) 
plt.xlabel('Columns')
plt.ylabel('Values')
plt.title('Boxplot for Each Column')
plt.show()

# %%
plt.hist(merged_df[merged_df['Open'] == True]['Sales'])  
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.title('Histogram of Sales for Open Entries')
plt.show()

# %%
merged_df['Date']

# %%
def split_date(df):
  df['Date'] = pd.to_datetime(df['Date'])
  df['Year'] = df['Date'].dt.year
  df['Month'] = df['Date'].dt.month
  df['Day'] = df['Date'].dt.day
  df['WeekOfYear'] = df.Date.dt.isocalendar().week

# %%
split_date(merged_df)
split_date(merged_test_df)

# %%
merged_df

# %%
month_map = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

def convert_promo_interval(promo_str):
    if pd.isna(promo_str):  
        return ""
    return ",".join(str(month_map[m]) for m in promo_str.split(","))

merged_df['PromoInterval_converted'] = merged_df['PromoInterval'].apply(convert_promo_interval)

# %%
merged_df

# %%
def check_promo_month(row):
    if pd.isna(row['PromoInterval_converted']) or row['PromoInterval_converted'] == "":
        return 0
    
    # Convert the comma-separated string to a list of integers
    promo_months = list(map(int, row['PromoInterval_converted'].split(',')))
    
    # Check if the Month value is in the list
    return 1 if row['Month'] in promo_months else 0

# Apply the function row-wise
merged_df['IsPromoMonth'] = merged_df.apply(check_promo_month, axis=1)


# %%
new_df = merged_df.copy()

# %%
new_df

# %%
new_df = new_df.drop(['PromoInterval', 'PromoInterval_converted','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear'], axis=1)
new_df

# %%
new_df.isnull().sum()

# %%
encoder = LabelEncoder()
new_df['StoreType'] = encoder.fit_transform(new_df['StoreType'])

# %%
new_df['Assortment'] = encoder.fit_transform(new_df['Assortment'])

# %%
new_df

# %%
new_df['StateHoliday'] = new_df['StateHoliday'].astype(str)
new_df['StateHoliday'] = encoder.fit_transform(new_df['StateHoliday'])

# %%
new_df

# %%
new_df = new_df.sort_values(by=["Date"])

# %%
new_df

# %%
features = new_df.drop(['Sales', 'Date'], axis=1).columns
target = 'Sales' 

# %%
scaler = MinMaxScaler()
new_df[features] = scaler.fit_transform(new_df[features])

# %%
def create_sequences(data, target_col, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i : i + seq_length][features].values)
        y.append(data.iloc[i + seq_length - 1][target_col])  
    return np.array(X), np.array(y)

X, y = create_sequences(new_df, target)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Input Shape:", X_train.shape) 

# %%
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    Dense(units=16, activation="relu"),
    Dense(units=1)  # Output layer (predicting sales)
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mse",  # Mean Squared Error
              metrics=["mae"])

# Model Summary
model.summary()

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=50,  
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# %%
import tensorflow as tf
print(tf.__version__)  # Check TensorFlow version
print("GPU available:", tf.config.list_physical_devices('GPU'))

# %%



