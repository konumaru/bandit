# Multi Arm Bandit 

- バンディットモデルのクラスをモジュール化

## TODO 
### デモデータの形式を変える
- バンディット問題として扱いやすいようにする
- 普通のバンディットの場合、`Feature Vector` を必要としないのですべて同じものを付与する（or 無視する）
- Arm ID もはじめから 0 ~ N ではなく、文字列とかにする
  - 間に、何かしらの選択肢を 0 ~ N に置き換える処理を明記したい
  - クラスメソッドに入れてもいいくらい

```例
誰が、何を引き、どんな結果だったか。というデータ構成にしたい。

- デモデータの構成を変える
  - Feature Data
    - Feature_ID
    - Feature Vector
  - Trail Data
    - Feature ID
    - Arm ID
    - Reward
```

### 変数の規則
- trial_data: (arm_id, reward)
  - 本当は、(featrue_id, arm_id, reward) にしたい


## Overview
### Demo Data

### Pull Arm & Update Rewards

### Select Arm
