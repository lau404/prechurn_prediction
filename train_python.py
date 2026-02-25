import pandas as pd
import numpy as np
import xgboost as xgb
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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# 1. 接收日期参数
parser = argparse.ArgumentParser()
parser.add_argument("--run_date", required=True, help="调度运行日期，例如 20250302")
args = parser.parse_args()
predict_date = args.run_date
print(predict_date)

# mkdir /data1/temperedheroes/TH_prechurn_data_compressed
# mkdir /data1/temperedheroes/TH_prechurn_data_extracted
cmd = f"~/data/myenv/bin/coscmd -c /data/configcenter/data/data_bigdata/cos_config_prechurn_v2.ini download -f th_prechurn/prechurn_data_{predict_date}.tar.gz /home/user_00/data/temperedheroes/TH_prechurn_data_compressed/prechurn_data_{predict_date}.tar.gz"
ret = os.system(cmd)
print(f"下载命令执行返回码: {ret}")
if ret != 0:
    print("文件下载失败，请检查coscmd命令或配置")
    exit(1)


# 所有压缩包的路径列表
tar_files = [f'/home/user_00/data/temperedheroes/TH_prechurn_data_compressed/prechurn_data_{predict_date}.tar.gz']

for tar_gz_file in tar_files:
    print(tar_gz_file)
    # 提取文件名作为目录名（去掉 .tar.gz）
    extract_dir = os.path.splitext(os.path.splitext(tar_gz_file)[0])[0]
    
    print(f"Extracting {tar_gz_file} to {extract_dir}/ ...")
    
    # 打开并解压
    with tarfile.open(tar_gz_file, 'r:gz') as tar:
        tar.extractall(path=f'/home/user_00/data/temperedheroes/TH_prechurn_data_extracted')


# 设置CSV文件所在的文件夹路径
# folder_path = f'/data/TH_prechurn_predict_data/prechurn_data_{run_date}'
folder_path = f'/home/user_00/data/temperedheroes/TH_prechurn_data_extracted'
# 获取所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith(f'_{predict_date}_train.csv')]
print(csv_files)
# 分别获取 train 和 predict 文件
train_files = [f for f in csv_files if f.endswith(f'_{predict_date}_train.csv')]
# predict_files = [f for f in csv_files if f.endswith('_predict.csv')]
# 合并所有 train 文件
train_df_list = []
for file in train_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    train_df_list.append(df)

train_df = pd.concat(train_df_list, ignore_index=True)
# train_df = pd.read_csv(folder_path)
# 读取唯一的 predict 文件
# if len(predict_files) != 1:
#     raise ValueError(f"预期有且仅有一个 *_predict.csv 文件，但实际找到 {len(predict_files)} 个")
# predict_file_path = os.path.join(folder_path, predict_files[0])
# predict_df = pd.read_csv(predict_file_path)

# 只转换数值型列，避免炸内核
for col in train_df.select_dtypes(include=[np.number]).columns:
    if col != '#account_id':
        train_df[col] = train_df[col].astype('float32')

# # for col in predict_df.select_dtypes(include=[np.number]).columns:
# #     if col != '#account_id':
# #         predict_df[col] = predict_df[col].astype('float32')

print("训练数据集信息：")
#train_df.info()
# # print("预测数据集信息：")
# # predict_df.info()



# 清洗训练数据
df_train = train_df.fillna(0,inplace=True)
df_train = train_df.drop(['log_date','#account_id', 'register_time'], axis=1)
df_train[np.isinf(df_train)] = 9999999
df_train.info()
#has_nan_or_inf = (
 #   df_train.isna().any().any() 
    #or
    #np.isinf(df_train.select_dtypes(include=[np.number])).any().any()
#)
#print("是否存在 NaN 或 Inf:", has_nan_or_inf)

# 拆分群体，分别建模
df_1 = df_train[(df_train['register_day'] <= 6) & (df_train['register_day'] >= 3)]
df_2 = df_train[(df_train['register_day'] <= 29) & (df_train['register_day'] >= 7)]
df_3 = df_train[(df_train['register_day'] >= 30)]
# 清洗用于预测的数据
# data_real = predict_df.fillna(0)  
# 拆分群体，分别建模
# df_r_1 = data_real[(data_real['register_day'] <= 6) & (data_real['register_day'] >= 3)]
# df_r_2 = data_real[(data_real['register_day'] <= 29) & (data_real['register_day'] >= 7)]
# df_r_3 = data_real[(data_real['register_day'] >= 30) & (data_real['pay_amt'] > 0)]



# 封装为函数
def model_params(df_t, group_label):
    X = df_t.drop(['is_churn'], axis=1)
    y = df_t['is_churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    feature_names = X.columns.tolist()  # 举例：排除目标列
    # 释放内存
    del X, y
    gc.collect()
    
    # 1.1特征标准归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 1. 限制OpenMP/BLAS线程，避免段错误
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

    # 1.2定义参数网格
    param_grid = {
        "max_depth": [3, 5, 7],          # 树的最大深度
        "learning_rate": [0.01, 0.1],     # 学习率
        "n_estimators": [100, 200],       # 树的数量
        "subsample": [0.7, 0.8],          # 每棵树随机采样的样本比例
        "colsample_bytree": [0.7, 0.8],    # 每棵树随机采样的特征比例
    }

    # 1.3初始化模型
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method = "gpu_hist",n_jobs=1)

    # 1.4网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",                # 使用AUC作为评估指标
        cv=5,                             # 5折交叉验证
        n_jobs=4,                         # 使用全部CPU核心
        verbose=0
    )

    with parallel_backend("threading", n_jobs=4):
    	# 1.5执行搜索
    	grid_search.fit(X_train, y_train)

    # 1.6输出最优参数
    print("Best Params:", grid_search.best_params_)
    print("Best AUC:", grid_search.best_score_)

    # 1.7使用最佳参数
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight= 3 / 1,  # 调整类别权重
        n_estimators=grid_search.best_params_['n_estimators'],
        max_depth=grid_search.best_params_['max_depth'],
        learning_rate=grid_search.best_params_['learning_rate'],
        subsample=grid_search.best_params_['subsample'],
        colsample_bytree=grid_search.best_params_['colsample_bytree'],
        early_stopping_rounds=10,
        use_label_encoder=False
    )
    # 训练并验证早停
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=10)
    # 释放内存
    del X_train, y_train
    gc.collect()

    # 1.8预测并评估在测试集上的表现
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # 流失概率
    # 释放内存
    del X_test
    gc.collect()
    
    # 评估
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))


    # 1.9计算PR曲线并可视化
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    # 可视化
   # plt.plot(thresholds, precision[:-1], label="Precision")
   # plt.plot(thresholds, recall[:-1], label="Recall")
   # plt.xlabel("Threshold")
   # plt.legend()
   # plt.show()

    # 1.10使用精准率0.7对应的阈值进行预测
    # 找出第一个 precision >= 0.7 的位置
    for p, t in zip(precision, thresholds):
        if p >= 0.7:
            selected_threshold = t
            break
    else:
        selected_threshold = 0.9  # 如果 precision 始终达不到 0.7，阈值定位0.9
    print(f"选定的阈值（precision >= 0.7）: {selected_threshold:.4f}")
    # 使用该阈值进行分类
    y_pred = (y_proba >= selected_threshold).astype(int)
    # 打印评估结果
    print(classification_report(y_test, y_pred))

    # 1.11将scaler、模型参数、模型预测正类的结果保存为json
    report = classification_report(y_test, y_pred, output_dict=True)
    model_path = f"//home/user_00/data/temperedheroes/TH_prechurn_feature_data/{predict_date}_xgb_model_{group_label}.json"
    # cmd_upload = f"/home/user_00/.local/bin/coscmd -c /data/configcenter/data/data_bigdata/cos_config_prechurn.ini  upload {model_path} /th_prechurn/"
    # print(cmd_upload)
    model.save_model(model_path)
    # os.system(cmd_upload)
    print(f"-----------{group_label}保存成功！------------------------")
    model_config = {
    "scaler": {
        "mean": np.round(scaler.mean_, 4).tolist(),
        "scale": np.round(scaler.scale_, 4).tolist(),
        "feature_names": feature_names
            },
    "model": {
        "model_path": model_path
            },
    "positive_metrics":{
        "threshold": round(float(selected_threshold), 4),
        "recall": round(report["1.0"]["recall"], 4),
        "precision": round(report["1.0"]["precision"], 4),
        "f1_score": round(report["1.0"]["f1-score"], 4),
        "sample_size": len(y_test)
            }
                    }
    return model_config

    # 2.1在真实数据上预测
    # X_r = df_r.drop(['log_date','#account_id', 'register_time','is_churn'], axis=1) 
    #X_r = df_r.drop(['is_churn'], axis=1)
    # X_r_s = scaler.transform(X_r)  

    # 2.2获取流失概率（正类）
    # y_proba_real = model.predict_proba(X_r_s)[:, 1]

    # 2.3使用上述阈值
    # threshold = selected_threshold
    # y_pred_real = (y_proba_real >= threshold).astype(int)

    # 2.4将预测结果与特征拼接
    # df_p = df_r.drop(['is_churn'], axis=1)
    # df_p['predicted']=y_pred_real

    # del model
    # gc.collect()


# 写入 MySQL
def write_model_config_to_mysql(model_config, group_label, predict_date):
    conn = pymysql.connect(
        host='172.21.102.40',      # 主机
        user='data',      # 用户名
        port=3306,            #端口
        password='USzBmz5BKUcrUJ#L9uBU',  # 密码
        db='dw_etl_data_statistics',          # 数据库名
        charset='utf8mb4'
    )
    cursor = conn.cursor()

    config_str = json.dumps(model_config)
    sql = "INSERT INTO t_prechurn_model_configs (group_label, config_json, predict_date) VALUES (%s, %s, %s)"

    cursor.execute(sql, (group_label, config_str, predict_date))
    conn.commit()

    cursor.close()
    conn.close()


train_dfs = [
    (df_1, "群体1：生命周期3-6"), (df_2, "群体2：生命周期7-29"), (df_3, "群体3：生命周期30天及以上")]

# 逐个处理并写入数据库
for df_t, group_label in train_dfs:
    config = model_params(df_t, group_label)
    write_model_config_to_mysql(config, group_label, predict_date)

print("✅ 所有模型配置已成功写入数据库")
