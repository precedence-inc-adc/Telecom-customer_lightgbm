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
 
 ## 詳細  
telecom customerというKaggleの問題について、lightgbmを用いて解きました。  
データセットは100 個の変数と約 10 万件のレコードで構成されており、通信業界の属性を説明するさまざまな変数と、通信業界の顧客と取引する際に重要と考えられるさまざまな要素が含まれています。ここでのターゲット変数は、顧客が解約するかどうかを説明する解約です。 このデータセットを使用すると、利用可能なさまざまな変数に応じて、解約する顧客または解約しない顧客を予測することができます。  
今回は、
