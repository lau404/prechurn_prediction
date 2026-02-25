import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import argparse
import subprocess
import os
import gc
import tarfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend
import json
import pymysql


# 1. 接收日期参数
parser = argparse.ArgumentParser()
parser.add_argument("--run_date", required=True, help="调度运行日期，例如 2025-03-02")
args = parser.parse_args()
run_date = args.run_date

# cos上拉取预测数据集
cmd = f"~/data/myenv/bin/coscmd -c /data/configcenter/data/data_bigdata/cos_config_prechurn_v2.ini download -f th_prechurn/predict_data_{run_date}.tar.gz /home/user_00/data/temperedheroes/TH_prechurn_data_compressed/predict_data_{run_date}.tar.gz"
ret = os.system(cmd)
print(f"下载命令执行返回码: {ret}")
if ret != 0:
    print("文件下载失败，请检查coscmd命令或配置")
    exit(1)

# 文件路径和目标目录
tar_gz_path = f"/home/user_00/data/temperedheroes/TH_prechurn_data_compressed/predict_data_{run_date}.tar.gz"
extract_dir = "/home/user_00/data/temperedheroes/TH_prechurn_data_extracted"

print(f"Extracting: {tar_gz_path} → {extract_dir}/")

# 解压
with tarfile.open(tar_gz_path, "r:gz") as tar:
    tar.extractall(path=extract_dir)

print("Done.")

# 设置CSV文件所在的文件夹路径
folder_path = f'/home/user_00/data/temperedheroes/TH_prechurn_data_extracted/th_prechurn_{run_date}_predict.csv'
predict_df = pd.read_csv(folder_path)

# 只转换数值型列，避免炸内核
# for col in train_df.select_dtypes(include=[np.number]).columns:
#     if col != '#account_id':
#         train_df[col] = train_df[col].astype('float32')

for col in predict_df.select_dtypes(include=[np.number]).columns:
    if col != '#account_id':
        predict_df[col] = predict_df[col].astype('float32')

# print("训练数据集信息：")
# train_df.info()

print("预测数据集信息：")
predict_df.info()


# 清洗训练数据
# df_train = train_df.fillna(0)  
# df_train = train_df.drop(['log_date','#account_id', 'register_time'], axis=1) 
# 拆分群体，分别建模
# df_1 = df_train[(df_train['register_day'] <= 6) & (df_train['register_day'] >= 3)]
# df_2 = df_train[(df_train['register_day'] <= 29) & (df_train['register_day'] >= 7)]
# df_3 = df_train[(df_train['register_day'] >= 30) & (df_train['pay_amt'] > 0)]
# 清洗用于预测的数据
data_real = predict_df.fillna(0)  
# 拆分群体，分别建模
df_r_1 = data_real[(data_real['register_day'] <= 6) & (data_real['register_day'] >= 3)]
df_r_2 = data_real[(data_real['register_day'] <= 29) & (data_real['register_day'] >= 7)]
df_r_3 = data_real[(data_real['register_day'] >= 30)]


# 从 MySQL 读取模型配置并建模
def load_config_and_predict(group_label,df_r):
    conn = pymysql.connect(
    host='172.21.102.40',
    user='data',
    port=3306,       
    password='USzBmz5BKUcrUJ#L9uBU',
    db='dw_etl_data_statistics',
    charset='utf8mb4')
    
    cursor = conn.cursor()
    sql = "SELECT config_json FROM t_prechurn_model_configs WHERE group_label = %s ORDER BY created_at DESC LIMIT 1"
    cursor.execute(sql, (group_label,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result:
        config_json = result[0]
        config=json.loads(config_json)
    else:
        raise ValueError(f"此群体无模型配置: {group_label}")

    # 重建 Scaler 和 模型
    scaler_config=config["scaler"]
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_config["mean"])
    scaler.scale_ = np.array(scaler_config["scale"])
    scaler.feature_names_in_ = np.array(scaler_config["feature_names"])
    feature_names = config["scaler"]["feature_names"]

    # cos拉取模型json文件
    # cmd_download = f"/usr/local/service/anaconda3/envs/datacenter/bin/coscmd -c /data/configcenter/data/data_bigdata/cos_config_prechurn.ini download -f th_prechurn/{run_date}_xgb_model_{group_label}.bin /data/logs/TH_prechurn_feature_data/{run_date}_xgb_model_{group_label}.bin"
    # os.system(cmd_download)
    model_path=f'/home/user_00/data/temperedheroes/TH_prechurn_feature_data/{run_date}_xgb_model_{group_label}.json'
    model = XGBClassifier()
    model.load_model(model_path)

    # 进行预测
    X_r = df_r[feature_names]
    X_r_s = scaler.transform(X_r)

    y_proba_real = model.predict_proba(X_r_s)[:, 1]
    threshold = config["positive_metrics"]["threshold"]
    y_pred_real = (y_proba_real >= threshold).astype(int)

    df_p = df_r.drop(['is_churn'], axis=1)
    df_p['predicted']=y_pred_real

    del model
    gc.collect()
    
    return df_p


group_labels = ["群体1：生命周期3-6", "群体2：生命周期7-29", "群体3：生命周期30天及以上"]
real_dfs = [df_r_1, df_r_2, df_r_3]
df_all = pd.concat([load_config_and_predict(group_label,df_r) for group_label, df_r in zip(group_labels, real_dfs)],axis=0).reset_index(drop=True)


# 检查 log_date 列的唯一值
unique_dates = df_all['log_date'].unique()
df_all['log_date'] = df_all['log_date'].astype(str) + ' 15:15:00'

if len(unique_dates) == 1:
    # 获取唯一的 log_date 并格式化为文件名
    log_date = unique_dates[0]
    formatted_date = log_date.replace('-', '')  # 格式化为 YYYYMMDD
    csv_file_name = f'predict_result_{formatted_date}.csv'
    
    # 指定要保存的 CSV 文件路径
    csv_file_path = f'/home/user_00/data/temperedheroes/TH_prechurn_predict_result/{csv_file_name}'
    
    # 将 DataFrame 保存为 CSV 文件
    df_all.to_csv(csv_file_path, index=False)
    print(f"数据已成功写入 {csv_file_path}")
else:
    print(f"log_date 列有 {len(unique_dates)} 个唯一值: {unique_dates}")

# 预测结果文件路径
csv_path = f"/home/user_00/data/temperedheroes/TH_prechurn_predict_result/predict_result_{formatted_date}.csv"

# 预测结果压缩包路径
tar_gz_path = f"/home/user_00/data/temperedheroes/TH_prechurn_predict_result/predict_result_{formatted_date}.tar.gz"

# 创建预测结果压缩文件
with tarfile.open(tar_gz_path, "w:gz") as tar:
    # 第二个参数 arcname 是压缩包中文件的名字（不带路径）
    tar.add(csv_path, arcname=os.path.basename(csv_path))

print(f"✅ 压缩完成：{tar_gz_path}")

cmd_upload = f"~/data/myenv/bin/coscmd -c /data/configcenter/data/data_bigdata/cos_config_prechurn_v2.ini  upload {tar_gz_path} /th_prechurn/"
print(cmd_upload)
os.system(cmd_upload)
