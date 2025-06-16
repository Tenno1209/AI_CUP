import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

# ======== 路徑設定 =========
train_info_path = 'C:\\Users\\user\\Desktop\\eric_python\\AI_CUP_2025\\train_info.csv'
test_info_path = 'C:\\Users\\user\\Desktop\\eric_python\\AI_CUP_2025\\test_info.csv'
train_data_dir = 'C:\\Users\\user\\Desktop\\eric_python\\AI_CUP_2025\\train_data'
test_data_dir = 'C:\\Users\\user\\Desktop\\eric_python\\AI_CUP_2025\\test_data'
output_csv_path = 'C:\\Users\\user\\Desktop\\eric_python\\AI_CUP_2025\\predictions_summary.csv'
# os.makedirs('39_Test_Dataset', exist_ok=True)

# ======== 特徵提取函數 =========
# === 批次 GPU 特徵擷取函數 ===
def batch_extract_features(file_paths, batch_size=1024):
    features_list = []

    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i+batch_size]
        batch_data = []

        for path in batch_paths:
            try:
                data = np.loadtxt(path)
                if data.ndim == 1:
                    data = data.reshape(-1, 6)
                batch_data.append(data[:, :6])
            except Exception as e:
                print(f"⚠️ 錯誤檔案: {path}, 錯誤: {e}")
                batch_data.append(np.zeros((1, 6)))

        max_len = max(arr.shape[0] for arr in batch_data)
        padded_batch = []

        for arr in batch_data:
            pad_len = max_len - arr.shape[0]
            if pad_len > 0:
                arr = np.pad(arr, ((0, pad_len), (0, 0)), 'constant')
            padded_batch.append(arr)

        batch_tensor = torch.tensor(padded_batch, dtype=torch.float32).cuda()

        means = batch_tensor.mean(dim=1)
        stds = batch_tensor.std(dim=1)
        maxs = batch_tensor.max(dim=1).values
        mins = batch_tensor.min(dim=1).values

        batch_features = torch.cat([means, stds, maxs, mins], dim=1).cpu().numpy()
        features_list.append(batch_features)

    return np.vstack(features_list)

# === 載入訓練資料 ===
train_info = pd.read_csv(train_info_path)
train_paths, y_train = [], []

for _, row in train_info.iterrows():
    uid = row['unique_id']
    path = os.path.join(train_data_dir, f"{uid}.txt")
    if os.path.exists(path):
        train_paths.append(path)
        y_train.append([
            row['gender'] - 1,
            row['hold racket handed'] - 1,
            row['play years'],
            row['level'] - 2
        ])

X_train = batch_extract_features(train_paths, batch_size=128)
y_train = np.array(y_train)

# === 資料標準化 + 模型訓練 ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = MultiOutputClassifier(DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42))
model.fit(X_train, y_train)

# === 載入測試資料並預測 ===
test_info = pd.read_csv(test_info_path)
test_paths, test_ids = [], []

for _, row in test_info.iterrows():
    uid = row['unique_id']
    path = os.path.join(test_data_dir, f"{uid}.txt")
    if os.path.exists(path):
        test_paths.append(path)
        test_ids.append(uid)

X_test = batch_extract_features(test_paths, batch_size=128)
X_test = scaler.transform(X_test)
y_probs = model.predict_proba(X_test)

# === 組裝 submission 結果 ===
submission = pd.DataFrame()
submission["unique_id"] = test_ids
submission["gender"] = [p[1] for p in y_probs[0]]
submission["hold racket handed"] = [p[1] for p in y_probs[1]]
submission["play years_0"] = [p[0] for p in y_probs[2]]
submission["play years_1"] = [p[1] for p in y_probs[2]]
submission["play years_2"] = [p[2] for p in y_probs[2]]
submission["level_2"] = [p[0] for p in y_probs[3]]
submission["level_3"] = [p[1] for p in y_probs[3]]
submission["level_4"] = [p[2] for p in y_probs[3]]
submission["level_5"] = [p[3] for p in y_probs[3]]

submission = submission.round(4)
submission.to_csv(output_csv_path, index=False)
print(f"✅ 已儲存預測檔案至 {output_csv_path}")
