## 基本情報  
・問題：telecom customer(https://www.kaggle.com/datasets/abhinav89/telecom-customer)  
・データ：Telecom_customer churn.csv(上記Kaggleからダウンロード)  
・環境：kaggle notebook  
|ライブラリ|バージョン|
----|----
 | lightgbm | 3.3.2 |
 | pandas | 1.5.3 |
 | python | 3.5.2 | 
 | scikit-learn | 1.2.2 |
 
 ## 概要  
2年ほど前になりますが、telecom customerというKaggleの問題について、lightgbmを用いて解きました。  
データセットは100 個の変数と約 10 万件のレコードで構成されており、通信業界の属性を説明するさまざまな変数と、通信業界の顧客と取引する際に重要と考えられるさまざまな要素が含まれています。ここでのターゲット変数は、顧客が解約するかどうかを説明します。このデータセットを使用すると、利用可能なさまざまな変数に応じて、顧客の解約を予測することができます。  
lightgbm.pyでは、データ(Telecom_customer churn.csv)を0と1に正規化したあと、教師データとテストデータに分けて、lightgbmで学習を行いました。その後、正答率を算出し、結果を描画しています。実装の詳細はlightgbm.pyの各行にコメントで記載しています。  
結果は下記の画像の通りです。テストデータに対して正しく予想できたのは約60%でした。今回はlightgbmで素直に学習を行いましたが、ハイパーパラメータや学習方法の調整、他モデルを試すことにより、精度が上がるように改善していきたいです。

<img width="284" alt="result" src="https://github.com/precedence-inc-adc/Telecom-customer_lightgbm/assets/135094689/28206081-a3f7-4825-87a4-46d185b7cef5">
