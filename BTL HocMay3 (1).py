import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data_path = r"C:\Users\Admin\Downloads\kddcup.data_10_percent.gz"
cols = """duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,
logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,
num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,
srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,
dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,
dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate"""

# Split and clean up column names
columns = [c.strip() for c in cols.split(',')]
columns.append('target')

# Load the dataset
df = pd.read_csv(data_path, names=columns, compression='gzip')

# Display the first few rows of the dataset
print(df.head())

# Kiểm tra giá trị unique của cột 'target' trước khi ánh xạ
print("Các giá trị unique của cột 'target' trước khi ánh xạ:")
print(df['target'].unique())

# Map attack types to broader categories
attack_mapping = {
    'normal.': 'normal', 'back.': 'dos', 'buffer_overflow.': 'u2r', 'ftp_write.': 'r2l',
    'guess_passwd.': 'r2l', 'imap.': 'r2l', 'ipsweep.': 'probe', 'land.': 'dos',
    'loadmodule.': 'u2r', 'multihop.': 'r2l', 'neptune.': 'dos', 'nmap.': 'probe',
    'perl.': 'u2r', 'phf.': 'r2l', 'pod.': 'dos', 'portsweep.': 'probe', 'rootkit.': 'u2r',
    'satan.': 'probe', 'smurf.': 'dos', 'spy.': 'r2l', 'teardrop.': 'dos', 'warezclient.': 'r2l',
    'warezmaster.': 'r2l'
}

# Map target to attack_type
df['attack_type'] = df['target'].map(attack_mapping)

# Kiểm tra các giá trị NaN sau khi ánh xạ
print(f"Số lượng giá trị NaN trong cột 'attack_type' sau khi ánh xạ: {df['attack_type'].isna().sum()}")

# Hiển thị các giá trị không khớp sau khi ánh xạ
if df['attack_type'].isna().sum() > 0:
    print("Các giá trị 'target' không ánh xạ được:")
    print(df[df['attack_type'].isna()]['target'].unique())

# Loại bỏ các hàng có giá trị NaN trong 'attack_type'
df.dropna(subset=['attack_type'], inplace=True)

print(f"Số lượng giá trị NaN sau khi xử lý: {df['attack_type'].isna().sum()}")

# Drop categorical columns that are not useful or hard to encode
df.drop(['service'], axis=1, inplace=True)

# Kiểm tra lại kích thước DataFrame sau khi xử lý
print(f"Kích thước DataFrame sau khi xử lý: {df.shape}")

# Map 'flag' feature
flag_map = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10}
df['flag'] = df['flag'].map(flag_map)

# Protocol Type mapping
protocol_map = {'tcp': 0, 'udp': 1, 'icmp': 2}
df['protocol_type'] = df['protocol_type'].map(protocol_map)

# Kiểm tra lại kích thước DataFrame sau khi xử lý
print(f"Kích thước DataFrame sau khi xử lý: {df.shape}")
print(df.head())

# Define features and target
X = df.drop(['attack_type', 'target'], axis=1)
y = df['attack_type']

# Kiểm tra kích thước của X và y trước khi chia tách dữ liệu
print(f"Shape of X before scaling: {X.shape}")
print(f"Shape of y: {y.shape}")

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Naive Bayes
nb = GaussianNB()
start_time = time.time()
nb.fit(X_train, y_train)
end_time = time.time()

# Predictions
y_train_pred = nb.predict(X_train)
y_test_pred = nb.predict(X_test)

# Evaluation
print(f"Naive Bayes Training Time: {end_time - start_time:.4f}s")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred, zero_division=1))

# Decision Tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=4)
start_time = time.time()
dt.fit(X_train, y_train)
end_time = time.time()

# Predictions
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

# Evaluation
print(f"Decision Tree Training Time: {end_time - start_time:.4f}s")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred, zero_division=1))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
rf.fit(X_train, y_train)
end_time = time.time()

# Predictions
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Evaluation
print(f"Random Forest Training Time: {end_time - start_time:.4f}s")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred, zero_division=1))

# Bar chart comparing model accuracies
model_names = ['Naive Bayes', 'Decision Tree', 'Random Forest']
train_accuracies = [accuracy_score(y_train, nb.predict(X_train)),
                    accuracy_score(y_train, dt.predict(X_train)),
                    accuracy_score(y_train, rf.predict(X_train))]

test_accuracies = [accuracy_score(y_test, nb.predict(X_test)),
                   accuracy_score(y_test, dt.predict(X_test)),
                   accuracy_score(y_test, rf.predict(X_test))]

plt.figure(figsize=(10, 5))
plt.bar(model_names, train_accuracies, color='blue', alpha=0.6, label='Train Accuracy')
plt.bar(model_names, test_accuracies, color='orange', alpha=0.6, label='Test Accuracy')
plt.title('Model Performance Comparison')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
