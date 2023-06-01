#Dataset：https://www.kaggle.com/datasets/abhinav89/telecom-customer：w


#lightgbm(python)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import lightgbm as lgb
from lightgbm import Dataset
from sklearn import metrics
from sklearn.metrics import accuracy_score # モデル評価用(正答率)
from sklearn.metrics import log_loss # モデル評価用(logloss)
from sklearn.metrics import roc_auc_score # モデル評価用(auc)
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
file_url = "./Telecom_customer churn.csv"
df2 = pd.read_csv(file_url)
X=df2[['mou_Mean','totmrc_Mean','ovrmou_Mean','ovrrev_Mean','peak_vce_Mean','mou_peav_Mean','avgrev','avg3mou','avg3qty','avg3rev','dualband']]
Y=df2['churn']
ce_oe = ce.OrdinalEncoder(cols='dualband',handle_unknown='impute')
#文字を序数に変換
df_all_2 = ce_oe.fit_transform(X)
#値を1の始まりから0の始まりにする
df_all_2['dualband'] = df_all_2['dualband'] - 1
X=df_all_2
X = preprocessing.minmax_scale(X)
Y = preprocessing.minmax_scale(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=0)
# 学習に使用するデータを設定
lgb_train = Dataset(x_train, y_train)
lgb_eval = Dataset(x_test, y_test, reference=lgb_train)
# LightGBM parameters
params = {
'task': 'train',
'boosting_type': 'gbdt',
'objective': 'binary', # 目的 : 2クラス分類
'metric': {'binary_error'}, # 評価指標 : 誤り率(= 1-正答率)
}
# モデルの学習
model = lgb.train(params,
train_set=lgb_train, # トレーニングデータの指定
valid_sets=lgb_eval, # 検証データの指定
)
# テストデータの予測 (クラス1の予測確率(クラス1である確率)を返す)
y_pred_prob = model.predict(x_test)
# テストデータの予測 (予測クラス(0 or 1)を返す)
y_pred = np.where(y_pred_prob < 0.5, 0, 1) # 0.5より小さい場合0 ,そうでない場合1を返す
# 真値と予測値の表示
df_pred = pd.DataFrame({'target':y_test,'target_pred':y_pred})
display(df_pred)
# 真値と予測確率の表示
df_pred_prob = pd.DataFrame({'target':y_test, 'target0_prob':1-y_pred_prob, 'target1_prob':y_pred_prob})
display(df_pred_prob)
# モデル評価
# acc : 正答率
acc = accuracy_score(y_test,y_pred)
print('Acc :', acc)
# logloss
logloss =  log_loss(y_test,y_pred_prob) # 引数 : log_loss(正解クラス,[クラス0の予測確率,クラス1の予測確率])
print('logloss :', logloss)
# AUC
auc = roc_auc_score(y_test,y_pred_prob) # 引数 : roc_auc_score(正解クラス, クラス1の予測確率)
print('AUC :', auc)

# ROC曲線の描画
# cf : https://tips-memo.com/python-roc
from sklearn import metrics
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.show()