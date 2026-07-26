# 案例:一条通达信商业指标的因子化拆解 / Case Study: Tearing Down a Retail TDX Indicator

> **EN abstract** — A commercial TDX (通达信) stock-picking indicator, sold on the story of
> "detecting institutional accumulation", was decomposed into six continuous factors and tested
> cross-sectionally on ~5,000 A-share names over 600 trading days. Three findings: (1) the
> indicator's own narrative direction is **inverted** in the data — its "golden cross" is negative
> alpha; (2) four of the six variables are 0.8–0.95 correlated with each other — one formula does
> not give you six factors; (3) the full boolean AND-chain **loses to a mechanical random baseline**
> once you bucket by market regime, and drop-one ablation shows the momentum-confirmation
> conditions are what destroy the reversal alpha the accumulation condition actually carries.

**说明**:本文只公开方法与结论,不公开原指标的公式与源码(第三方商业指标)。
数据为本机自建的后复权研究库,不随本仓库分发,因此本文是**研究记录**,不是可复现工件;
可复现的是方法——[`factorkit`](../factorkit) 里的函数就是这套流程的实现,
[examples/](../examples) 用合成数据把同样的流程完整跑了一遍。

---

## 一、原始物是什么

一条在散户圈流传的通达信选股指标,卖点是「识别庄家吸筹」。读源码先读出三样东西:

| 成分 | 内容 | 处置 |
|---|---|---|
| 防盗版机关 | 时间锁(到某年后失效)、恒真条件、未定义引用 | 删 |
| 画图语句 | STICKLINE / COLOR / 显示用中间量 | 删,非逻辑 |
| 真正的逻辑 | 一条 11 个条件的 AND 链 + 若干连续中间变量 | 留,拆开分别测 |

拆完得到两条可检验的路:

- **路线 1**:抽出 6 个连续中间变量,当独立因子走 IC 流程;
- **路线 2**:整条 AND 链当布尔信号,走事件研究 + 机械基线。

---

## 二、路线 1:六个连续变量的截面实测

样本:全市场约 5,000 只股票 × 600 个交易日(后复权、含退市股、PIT 股票池)。

| 因子(按功能命名) | IC(1日) | t | IC(5日) | ICIR(5日) | top10% 日换手 | 判定 |
|---|---|---|---|---|---|---|
| 吸筹(新低卖压衰竭) | +0.027 | +4.6 | +0.046 | **0.34** | 0.17 | **唯一入池候选**:显著 + 新鲜 + 慢衰减 |
| 散户线(55 日反向位置) | +0.029 | +3.9 | +0.043 | 0.27 | 0.29 | 有效,但与「庄散差」相关 −0.95,只能留一个 |
| 长期位置(30 日) | −0.028 | −3.9 | −0.037 | −0.24 | 0.40 | 与池内已有因子同公式,无增量 |
| 庄家线(J 值 EMA) | −0.024 | −3.4 | −0.035 | −0.23 | 0.14 | 与「庄散差」0.95 共线 |
| 庄散差 | −0.029 | −3.8 | −0.042 | −0.27 | 0.19 | **方向与指标叙事相反** |
| 量比 | −0.026 | −7.2 | −0.018 | — | 0.74 | 1~2 日超短反转,换手过高 |

三条结论:

1. **指标的多头叙事在截面上是反的。** 「庄家线上穿散户线 = 金叉 = 买点」这个卖点,
   实测方向为负。方向只能以数据为准,不能以名字和故事为准。
2. **一条公式拆出来的变量大概率共线**:6 个里有 4 个互相 |corr| > 0.8,
   入池前必须先内部去重,否则等于把同一笔钱数了四遍。
3. **量比的衰减曲线**(t+1 −0.026 → t+3 −0.005)是「快因子必须快执行」的标准形状:
   配上 0.74 的日换手,成本一算就判死。

补充:该样本段(2022–2024 上半年)是弱市,全部因子都呈**反转**方向——
「位置高/动量强/放量」皆负,「贴底/吸筹」皆正。
这意味着**这批结论带着强烈的 regime 负载**,牛市段是盲区,不能声称普适。
这条限制我没法在训练段内解除,只能明写出来带着走。

---

## 三、路线 2:整条 AND 链的事件研究

口径:同信号日、同股票池、**随机等量抽票**、同出场规则的机械基线(称作 C4 基线),
报「净抬升 = 信号均值 − 基线均值」,不报裸胜率。

结果:**判死**。

- 全持有期跑输随机基线(5 日净抬升 −0.26pt);
- **弱市桶 10 日裸收益 +0.94%,看起来像金矿;但同一桶的随机基线是 +1.38%**
  ——弱市反弹段随机买更赚,这个信号的选股贡献是负的。

**这是「裸胜率好看 → 分桶机械基线一扣就死」的标准案例。**
事件研究不配分桶机械基线,等于没做。

### 逐条件消融(drop-one)

把 AND 链里每个条件轮流删掉,看净抬升怎么变:

| 条件 | 删掉后 | 角色 |
|---|---|---|
| 吸筹持续 | 恶化 0.29pt | **唯一承重柱** |
| 阳线实体 | 改善 ~0.28pt | **毒药** |
| 温和放量 | 改善 ~0.28pt | **毒药** |

结论很直白:**动量确认类条件(要阳线、要放量)把吸筹条件本来带的反转 alpha 毁掉了。**
指标作者把两族方向相反的逻辑 AND 在了一起,还以为条件越多越保险。

另外做了截断不变性探针(见 [03](03-backtest-integrity.md)),结果阴性——
没有未来函数,上面的负结论不是 bug 造成的。

---

## 四、这个案例教给我的四件事

1. **先拆再测。** 大 AND 链整体测 IC 得不到任何可行动的信息;拆开之后才知道
   六个变量里只有一个值得留,而且那一个的价值还被其他条件抵消掉了。
2. **基线比结果重要。** 没有 C4 基线,「弱市 10 日 +0.94%」会被我当成发现。
   一条信号的价值 = 它比随机好多少,不是它赚了多少。
3. **叙事是最不可靠的证据。** 卖方/散户指标的故事讲得越圆,越要先测方向。
4. **负结果也是产出。** 这个案例最终的交付物是「这条指标里只有一个变量值得留,
   而且它被同一条公式里的其他条件毁掉了」——这个结论帮我省下的算力和仓位,
   比再挖十个因子更值钱。

---

## 五、方法在本仓库中的对应实现

| 本案例的步骤 | 仓库里的实现 |
|---|---|
| 连续变量 IC / ICIR / t / 单调性 | `factorkit.rank_ic` / `ic_summary` / `quantile_stats` |
| 共线性识别与去重 | `factorkit.correlation_with_pool` / `residualize` |
| 换手与成本判死 | `factorkit.top_turnover` / `annual_cost_drag` |
| 快慢因子判别 | `factorkit.decay_curve` |
| 事件研究 + C4 机械基线 | `factorkit.event_study` |
| 分市场状态桶 | `factorkit.bucketed_event_study` |
| 逐条件消融 | `factorkit.drop_one_ablation` |
| 未来函数排查 | `factorkit.truncation_invariance` |

`examples/run_event_study.py` 在合成数据上重演了同一个故事:
一条「突破追涨」的 4 条件 AND 链在每个市场状态桶里都跑输随机基线,
而消融表把「阳线实体 > 3%」这个动量确认条件标成了毒药。
