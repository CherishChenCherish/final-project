# BTC Market Regime Classification - 每一步详细执行指南

## 项目结构
```
final/
├── data/
│   ├── raw/           # 原始下载数据
│   └── processed/     # 处理后的特征+标签
├── notebooks/         # Jupyter notebook（主要工作区）
├── src/
│   ├── data_collection.py      # 数据采集
│   ├── feature_engineering.py  # 特征工程 + 标签构建
│   ├── models.py               # 模型训练 + 评估
│   ├── temporal_cnn.py         # PyTorch 时序CNN
│   └── visualization.py       # 可视化
├── models/            # 保存结果JSON
├── figures/           # 保存图表
└── requirements.txt   # 依赖包
```

---

## Week 1: 数据采集 + EDA

### Step 1.1: 环境搭建
```bash
cd ~/Desktop/final
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 1.2: 获取价格数据 (Yahoo Finance)
```python
# 在 Jupyter notebook 或直接运行
python src/data_collection.py
```
**做什么：**
- 用 `yfinance` 下载 BTC-USD 2015-01-01 到 2026-03-05 的日线 OHLCV
- 保存到 `data/raw/btc_price.csv`
- 预期：约 4,100 行

**检查点：** 打开 CSV 确认日期范围完整，没有大段缺失

### Step 1.3: 获取链上数据 (CoinMetrics)
**做什么：**
- 通过 CoinMetrics Community API (免费) 获取以下指标：
  - `CapMVRVCur` — MVRV 比率 ⭐ 最重要，用来构建标签
  - `CapRealUSD` — 实现市值
  - `NVTAdj` — 网络价值/交易量比率
  - `AdrActCnt` — 活跃地址数
  - `HashRate` — 算力
  - `RevUSD` — 矿工收入（用于算 Puell Multiple）
  - `SplyCntCDD90d` — 币天销毁
- 保存到 `data/raw/btc_onchain.csv`

**如果 API 挂了：** 去 https://coinmetrics.io/community-network-data/ 手动下载 CSV

**检查点：** 确认 MVRV 有 2015 年以来的数据，某些指标可能 2017/2018 才开始有

### Step 1.4: EDA (探索性数据分析)
**在 notebook 里做：**

1. **画 BTC 价格走势图（对数坐标）**
   - 标注四次减半日期：2012-11-28, 2016-07-09, 2020-05-11, 2024-04-20

2. **画 MVRV 历史走势**
   - 标注 MVRV = 1.0 和 MVRV = 3.5 的水平线
   - 观察这两条线是否合理地分开了吸筹/扩张/派发区间

3. **各指标的基本统计量**
   - `df.describe()` 看分布
   - 检查缺失值：`df.isnull().sum()`

4. **检查数据质量**
   - 有没有明显的异常值？
   - 各指标的起始日期是否一致？

---

## Week 2: 标签构建 + 特征工程

### Step 2.1: 构建 Regime 标签
```python
python src/feature_engineering.py
```

**三组阈值（用于敏感性分析）：**

| 变体 | Accumulation (吸筹) | Distribution (派发) |
|------|---------------------|---------------------|
| base | MVRV < 1.0 | MVRV > 3.5 |
| conservative | MVRV < 0.8 | MVRV > 3.8 |
| aggressive | MVRV < 1.2 | MVRV > 3.0 |

中间区域 = Expansion (扩张)

**检查点：**
- 画三组标签的分布柱状图
- 预期 Distribution 最少（约 15%），这是 class imbalance
- 在价格图上叠加 regime 颜色，目视确认是否合理

### Step 2.2: 计算技术指标（价格衍生特征）
- 30日均线 (SMA_30)
- 价格/均线比 (Price_SMA_Ratio)
- RSI-14
- MACD (12, 26, 9) + Signal + Histogram
- 30日滚动波动率
- 30日前瞻收益率（回归子任务用）

### Step 2.3: 处理链上特征
- Puell Multiple = 矿工日收入 / 365日均值
- MVRV Z-Score
- 对活跃地址、算力做 log 变换
- 计算 7日、30日变化率

### Step 2.4: 定义 Walk-Forward 分割
四个减半周期：
| 周期 | 时间范围 | 用途（示例） |
|------|----------|-------------|
| Cycle 1 | 2013-01 ~ 2016-07 | Train |
| Cycle 2 | 2016-07 ~ 2020-05 | Train/Test |
| Cycle 3 | 2020-05 ~ 2024-04 | Train/Test |
| Cycle 4 | 2024-04 ~ 2026-03 | Test only |

Walk-forward 产生 3 个 fold：
- Fold 1: train=C1, test=C2
- Fold 2: train=C1+C2, test=C3
- Fold 3: train=C1+C2+C3, test=C4

**注意：** Fold 1 训练数据很少（~300-500天），模型表现可能差，这是正常的，要在报告里诚实说明

---

## Week 3: 基线模型 (Ridge + Linear SVM)

### Step 3.1: 准备特征矩阵
```python
# 特征集 A: 只有价格特征
features_A = ['SMA_30', 'Price_SMA_Ratio', 'RSI_14', 'MACD', 'MACD_Signal',
              'MACD_Hist', 'Volatility_30d', 'Returns']

# 特征集 B: 价格 + 链上特征
features_B = features_A + ['CapMVRVCur', 'CapRealUSD', 'NVTAdj', 'AdrActCnt',
                           'HashRate', 'Puell_Multiple', 'MVRV_Z', ...]
```

### Step 3.2: 训练 Ridge Regression (分类版)
```python
from sklearn.linear_model import RidgeClassifier
```
- 对每个 fold：StandardScaler fit on train, transform test
- 记录每个 fold 的 macro-F1
- 分别跑 Feature Set A 和 B

### Step 3.3: 训练 Linear SVM
```python
from sklearn.svm import LinearSVC
```
- 同上流程
- 调参：C = [0.01, 0.1, 1, 10]（在训练集内部用时序 CV 选）
- **成功标准：Linear SVM macro-F1 ≥ 0.45**

### Step 3.4: 记录结果
| Model | Feature Set | Fold 1 F1 | Fold 2 F1 | Fold 3 F1 | Avg |
|-------|-------------|-----------|-----------|-----------|-----|
| Ridge | A (price) | ? | ? | ? | ? |
| Ridge | B (price+chain) | ? | ? | ? | ? |
| SVM | A (price) | ? | ? | ? | ? |
| SVM | B (price+chain) | ? | ? | ? | ? |

**计算 F1 delta = B - A，这就是回答研究问题的关键数据**

---

## Week 4: 非线性模型 (Random Forest + XGBoost)

### Step 4.1: 训练 Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
# class_weight='balanced' 处理类别不均衡
```
- n_estimators=300, max_depth=10
- 同样跑 A 和 B 特征集

### Step 4.2: 训练 XGBoost
```python
from xgboost import XGBClassifier
```
- n_estimators=300, max_depth=6, learning_rate=0.05
- subsample=0.8, colsample_bytree=0.8

### Step 4.3: SHAP 特征重要性分析 ⭐
```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```
**这是 proposal 里承诺的关键交付物**——看哪些链上特征对分类贡献最大

### Step 4.4: MVRV 消融实验
- 特征集 C = B 去掉所有含 MVRV 的特征
- 如果 C 的 F1 仍然比 A 高 → 其他链上指标独立有预测力
- 如果 C ≈ A → 只有 MVRV 有用（但 MVRV 又参与了标签构建，有循环论证风险）

### Step 4.5: 开始写报告（并行）
从这周开始写 Introduction 和 Method 部分，不要等到最后一周

---

## Week 5: Temporal CNN (PyTorch)

### Step 5.1: 准备序列数据
- 滑动窗口：取过去 60 天的特征作为输入
- 输入 shape: (batch, n_features, 60)
- 输出：3 个类别的概率

### Step 5.2: 模型架构
```
Conv1d(n_features, 64, kernel=5) → BN → ReLU
Conv1d(64, 128, kernel=3) → BN → ReLU
Conv1d(128, 64, kernel=3) → BN → ReLU
AdaptiveAvgPool1d(1) → Dropout(0.3) → Linear(64, 3)
```

### Step 5.3: 训练
- Loss: CrossEntropyLoss with class weights（处理不均衡）
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau
- 50 epochs, batch_size=32
- **成功标准：最好的模型 macro-F1 ≥ 0.58**

### Step 5.4: 如果 CNN 效果差的 Fallback
proposal 里写了：如果 CNN 不行就退回到 MLP
```python
# 简单 MLP 替代方案
nn.Sequential(
    nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 3)
)
```

---

## Week 6: 报告 + 收尾

### Step 6.1: 30天收益率回归子任务
- 用同样的特征和 walk-forward 设置
- 评估 MAE 和方向准确率（预测涨跌是否正确）
- 方向准确率 > 55% = 弱正面结果

### Step 6.2: 敏感性分析
- 用三组阈值 (base, conservative, aggressive) 重新跑最好的模型
- 三组结论一致 → 结果可信
- 结论随阈值翻转 → 标注为可能是人为产物

### Step 6.3: 生成所有图表
```python
python src/visualization.py
```
需要的图：
1. BTC 价格 + regime 背景色
2. MVRV 分布 by regime
3. 特征相关性热力图
4. 类别分布柱状图
5. 模型 F1 对比图（A vs B）
6. SHAP 特征重要性
7. 每个 fold 的混淆矩阵
8. 敏感性分析结果

### Step 6.4: 写报告
报告结构：
1. Introduction — 问题和动机
2. Related Work — 三篇引用文献
3. Data — 数据源、时间范围、特征列表
4. Method — 标签构建、特征工程、模型、walk-forward
5. Results — A vs B 对比、SHAP、消融、敏感性
6. Discussion — 发现了什么、局限性、如果是 null result 怎么解释
7. Conclusion

### Step 6.5: 打包提交
- 最终 PDF 报告
- `requirements.txt`
- 代码 zip 或 Colab notebook
- README 说明如何复现

---

## 关键提醒

1. **诚实报告**：如果链上数据没用（B ≈ A），这也是有价值的结论，不要造假
2. **不要数据泄露**：绝对不能用普通 K-fold，必须用 walk-forward
3. **MVRV 循环论证**：一定要做消融实验，这是 proposal 里承诺的
4. **所有阈值预注册**：proposal 里定的数字不能看到结果后改
5. **每个 fold 单独报告**：不要只报平均值，要让读者看到 variance
