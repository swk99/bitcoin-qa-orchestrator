# 🐨 RADE: Retrieval-Augmented Diagnostic Engine

> **Cost-Aware Anomaly and Fraud Pattern Detection in Bitcoin Core Node Operations**
> via Multi-Scale Belief Estimation and KDE Adaptive Thresholds
>
> 🌱 *Work in Progress* | 🏫 Goldsmiths, University of London | 🐝 Seonwoo Kim

---

## 🧸 what even is this

ok so Bitcoin nodes need to be monitored constantly but monitoring costs money
and you can't just escalate everything forever 😅

this project treats Bitcoin node diagnosis as a **POMDP** — at each step, an agent
decides: `probe` 🔦, `skip` 🙈, or `escalate` 🆘

the twist? the anomaly thresholds from 2019 literally classify **99.8% of 2026 data as anomalous**:

```
fixed threshold (2019):  pending_tx > 15,000
real network (2026):     pending_tx ≈ 80,000+   😭
```

so RADE learns the real distribution from live Bitcoin data and adapts automatically.
no fake synthetic environments. no hardcoded thresholds. just real blocks. 🌱

---

## 🌸 what's new here

| thing | what it does |
|---|---|
| 🪐 **Multi-scale Belief** | 3 time windows (10min / 1hr / 24hr) — anomalies happen at different speeds |
| 🫧 **KDE Adaptive Threshold** | learns the ACTUAL 2026 distribution instead of using 2019 vibes |
| 🍓 **Block-Event Collection** | triggers on real block arrival, not every 10 min — genuine Poisson samples |
| 🚨 **Fraud Detection** | wash trading + mempool flooding from live fee data |
| 🚫🤖 **No LLM** | fully data-driven, no API costs, runs offline |

---

## 🍋 the non-stationarity story (this IS the paper)

watch what happens as we collect more real blocks:

```
n=11   → P95 pending: 47,891   anomaly rate: 100%  😱
n=28   → P95 pending: 55,798   anomaly rate: 100%  😬
n=41   → P95 pending: 97,656   anomaly rate: 100%  📈
n=144+ → P95 pending: ???      anomaly rate: ~5%?  🌱
```

the KDE is *learning* the real 2026 distribution!
fixed thresholds could never 💅

also: inter-block time excluding outage events = **676s** (theory: 600s ✅)
and we caught two real network outage events (12,535s and 10,536s) as genuine anomalies 🔥

---

## 🗂️ project structure

```
bitcoin-qa-orchestrator/
│
├── 📡 Data Collection
│   ├── btc_block_collector.py      ← ⭐ canonical collector (block-event driven)
│   ├── btc_live_runner.py          ← legacy time-driven (monitoring only)
│   ├── btc_historical_collector.py ← bulk historical (auxiliary only)
│   ├── btc_live_db.py              ← PostgreSQL schema
│   └── btc_calibrate.py            ← MLE fitting + AIC/BIC comparison
│
├── 🧠 Core Algorithm
│   ├── rade_belief.py              ← multi-scale belief + KDE threshold + fraud belief
│   └── rade_train_live.py          ← GAE-A2C training + baseline policies
│
├── 🗄️ Persistence
│   ├── db.py                       ← experiment logging
│   └── memory.py                   ← ChromaDB episodic memory
│
└── 🛠️ Utilities
    ├── check_status.py             ← quick DB status check
    └── delete_historical.py        ← cleanup script
```

**data source hierarchy:**
```
block_event           ← calibration + training (USE THIS)
live                  ← legacy time-driven (monitoring reference only)
historical_reconstructed ← heuristic reconstruction (auxiliary only)
```

---

## 🐣 setup

### you need

- Python 3.11+
- PostgreSQL 15+
- mempool.space API (free, no key needed 🥳)

### install

```bash
git clone https://github.com/seonwoojh/bitcoin-qa-orchestrator.git
cd bitcoin-qa-orchestrator

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### database

```bash
createdb btcqa

python -c "from btc_live_db import LiveSnapshotDB; LiveSnapshotDB.from_env().init_schema(); print('done 🐨')"
```

---

## 🍀 step 1: start block-event collection

⭐ **this is the canonical data source for RADE**

```bash
# triggers a snapshot on every new block confirmation (~10 min avg)
# stores as source='block_event' with exact block timestamps
python btc_block_collector.py
```

you'll see something like:

```
╔══════════════════════════════════════════════════════╗
║   RADE — Block-Event Driven Collector                ║
║   source = "block_event"  (calibration-grade data)  ║
╚══════════════════════════════════════════════════════╝

     Block     Mempool     Pending   Inter(s)     Fee  Flags
──────────────────────────────────────────────────────────────
 948,826      38.18MB      77,793       641s    3.0    ⚠
 948,827      36.39MB      71,573       674s    2.0    ⚠
 948,828      34.56MB      65,387        57s    1.0  🚨⚠   ← fraud detected!
```

every 10 blocks it prints a distribution comparison:

```
Inter-block time (n=40)
  exponential  [MLE]:          AIC=650.5
  gamma        [method-of-moments]: AIC=634.8
  lognormal    [MLE]:          AIC=627.5  ← Best by AIC
```

> 🍋 **why block-event and not time-driven polling?**
> polling every T seconds creates a "staircase artefact" — inter-block time becomes
> multiples of T instead of genuine Poisson samples. block-event collection
> records the EXACT timestamp difference between confirmed blocks.
> this is a methodological contribution of the paper 🔥

### check status

```bash
python check_status.py
```

```
block_event   : 41개   ← use for calibration
live          : 242개  ← legacy reference
historical    : 1,376개
전체           : 1,659개
```

---

## 📊 step 2: calibrate MDP parameters

```bash
# use block_event source only (canonical)
python btc_calibrate.py --days 1 --source block_event --plot --output params_block.json
```

example output:

```
=======================================================
  RADE MDP Calibration Results
=======================================================
  Samples   : 41  (source=block_event)

  Mempool size (MB)
    Distribution : LogNormal(mu=3.611, sigma=0.042)
    P95          : 39.3 MB  ← adaptive threshold

  Inter-block time (s)
    Mean (clean) : 676.4s  (theoretical: 600s ✅)
    P95          : 3531 s

  Pending tx
    P95          : 97,656  ← vs fixed threshold 15,000 (6.5× higher!!)

  Anomaly rate : 100.0%  (KDE will fix this as data accumulates)
=======================================================
```

> 🥹 **the convergence story (this is literally the whole paper)**
> ```
> n=11  → P95 pending: 47,891   100% anomaly
> n=28  → P95 pending: 55,798   100% anomaly
> n=41  → P95 pending: 97,656   100% anomaly  ← you are here
> n=144 → P95 pending: ???      ~18%? anomaly
> n=500 → P95 pending: ???      ~5%?  anomaly 🌱
> ```
> the KDE adaptive threshold is learning the real distribution!!

---

## 🧠 step 3: train RADE

```bash
# DB replay mode: uses block_event snapshots from PostgreSQL
# no API calls during training → no rate limiting
python rade_train_live.py \
  --steps 2000 \
  --ablation-type mabse_full \
  --use-live \
  --mu-mempool 3.611 \
  --sigma-mempool 0.042 \
  --lambda-inter 0.000823

# quick synthetic test (no DB needed)
python rade_train_live.py --steps 200 --ablation-type mabse_full --no-live --no-db
```

> 🐨 **why DB replay and not live API?**
> training makes thousands of requests per run — mempool.space will 429 you 😭
> DB replay decouples training from real-time API availability
> and makes experiments reproducible

---

## 🔬 step 4: ablation study

```bash
# AC baseline (no belief, no KDE)  ← Joseph et al. 2020
python rade_train_live.py --steps 2000 --ablation-type ac_base --use-live

# RADE-SS (single-scale belief)    ← Blier & Ollivier 2022
python rade_train_live.py --steps 2000 --ablation-type mabse_no_multiscale --use-live

# RADE-FT (fixed threshold)        ← Hundman et al. 2018
python rade_train_live.py --steps 2000 --ablation-type mabse_no_kde --use-live

# RADE full ⭐
python rade_train_live.py --steps 2000 --ablation-type mabse_full --use-live
```

plus three rule-based baselines built into `rade_train_live.py`:
- `FixedThreshold` — 2019 legacy thresholds
- `RollingP95` — sliding window quantile
- `EWMA-Z` — exponential weighted z-score

| variant | 3-scale belief | KDE | dim | basis |
|---|:---:|:---:|:---:|---|
| FixedThreshold | — | — | — | 2019 legacy |
| RollingP95 | — | partial | — | Hundman et al. |
| EWMA-Z | — | — | — | standard monitoring |
| AC base | ✗ | ✗ | 6D | Joseph et al. |
| RADE-SS | ⚡ single | ✗ | 7D | Blier & Ollivier |
| RADE-FT | ✅ | ✗ | 9D | Hundman et al. |
| **RADE (full)** 🐨 | ✅ | ✅ | **10D** | proposed |

---

## 📐 algorithm overview

### multi-scale belief state

$$\hat{b}^\tau_t = \frac{\sum_{k} \text{sim}(\mathbf{s}_t, \mathbf{s}_k) \cdot w_k^\tau \cdot \mathbf{1}[z_k=1]}{\sum_k w_k^\tau}$$

with exponential time-decay:

$$w_k^\tau = \frac{\exp(-\text{age}_k / \tau)}{\sum_j \exp(-\text{age}_j / \tau)}$$

three scales:
- 🟢 **short** ($\tau_s = 10$ min): block-level congestion
- 🟡 **mid** ($\tau_m = 60$ min): fee pressure patterns
- 🔴 **long** ($\tau_l = 1440$ min): regime-level context (halving, inscriptions...)

### KDE adaptive threshold

analytic CDF via error function (100× faster than numerical integration):

$$\hat{F}_h(x) = \frac{1}{N}\sum_{i=1}^N \frac{1}{2}\left[1 + \text{erf}\left(\frac{x - m_i}{h\sqrt{2}}\right)\right]$$

belief-gated false-alarm rate (monotone by proof):

$$\alpha_t = \alpha_0 \cdot \exp(-\lambda_\alpha \cdot \bar{b}_t)$$

high belief → lower threshold → more sensitive detection 🎯

### six-term reward

$$r_t = \underbrace{\alpha U}_\text{utility} - \underbrace{\lambda c(a)}_\text{cost} - \underbrace{\beta \bar{b}_t \mathbf{1}[\text{skip}]}_\text{RAG risk} - \underbrace{\delta \hat{b}^\text{fr}_t \mathbf{1}[\text{skip}]}_\text{fraud risk} - \underbrace{\gamma_{FN} \mathbf{1}[\text{miss}]}_\text{FN} - \underbrace{\gamma_{FP} \hat{b}^\text{fr}_t \mathbf{1}[\text{esc, low fraud}]}_\text{false alarm}$$

---

## 🫧 full data flow

```
mempool.space API 🌐
        │ (new block detected)
        ▼
btc_block_collector.py     ← block-event driven, exact timestamps
        │  source='block_event'
        ▼
PostgreSQL (live_snapshots)
        │
        ├──► btc_calibrate.py  ← LogNormal + Exponential MLE + AIC/BIC
        │           │
        │           ▼
        │      params_block.json
        │
        ▼
rade_belief.py             ← multi-scale belief + KDE + fraud belief
        │
        ▼
rade_train_live.py         ← GAE-A2C + DB replay (no API calls)
        │
        ▼
db.py                      ← experiment results 🗄️
```

---

## ⚙️ hyperparameters

| param | value | why |
|---|---|---|
| $\gamma$ | 0.97 | long-horizon discount |
| $\lambda_\text{GAE}$ | 0.95 | bias-variance balance |
| $\tau_s / \tau_m / \tau_l$ | 10 / 60 / 1440 min | block / hour / day |
| $K$ neighbours | 6 | top-K retrieval per scale |
| KDE bandwidth | Silverman's rule | minimises MISE |
| $\alpha_0$ | 0.05 | 5% base false alarm rate |
| $\lambda_\alpha$ | 2.0 | belief-gate sensitivity |

---

## 🌻 expected convergence

```
blocks collected  │  P95 pending  │  anomaly rate
──────────────────┼───────────────┼──────────────
41  (now 🐨)      │  97,656       │  100%   😱
144 (~1 day)      │  ???          │  ~18%?  📉
500 (~3.5 days)   │  ???          │  ~5%    🌱✅
```

---

## 🌈 reproduce paper results

```bash
# 1. collect block-event data (keep running!)
python btc_block_collector.py

# 2. calibrate
python btc_calibrate.py --days 7 --source block_event --plot --output params_7d.json

# 3. run all ablations (4 variants × 6 lambdas × 5 seeds = 120 experiments)
for ABLATION in ac_base mabse_no_multiscale mabse_no_kde mabse_full; do
  for LAMBDA in 0.05 0.10 0.20 0.30 0.40 0.50; do
    for SEED in 41 42 43 44 45; do
      python rade_train_live.py \
        --steps 2000 \
        --ablation-type $ABLATION \
        --lambda-cost $LAMBDA \
        --seed $SEED \
        --use-live
    done
  done
done
```

---

## 📚 key references

- Krishnamurthy (2016) — POMDP belief state foundations
- Joseph et al. (2020) — AC for anomaly detection (AC base baseline)
- Schulman et al. (2015) — GAE
- Blier & Ollivier (2022) — Retrieval-Augmented RL (RADE-SS baseline)
- Hundman et al. (2018) — Non-parametric dynamic thresholds (RADE-FT baseline)
- Lindstrom et al. (2020) — Functional KDE for time series
- iADCPS (2025) — Dynamic thresholds for cyber-physical systems
- Cong et al. (2023) — Crypto wash trading
- Neudecker et al. (2016) — Bitcoin P2P network analysis

---

## 🌷 citation

```bibtex
@misc{kim2025rade,
  author = {Kim, Seonwoo},
  title  = {{RADE}: Retrieval-Augmented Diagnostic Engine for
            Cost-Aware Anomaly and Fraud Pattern Detection
            in Bitcoin Core Node Operations},
  year   = {2025},
  note   = {Work in Progress}
}
```

---

## 🐨 about

**Seonwoo Kim** | MSc Computing (Data Science) | Goldsmiths, University of London

📮 skim008@gold.ac.uk | 🔗 [ORCID](https://orcid.org/0009-0005-9599-0514)

---

*built with way too much coffee and real Bitcoin blocks 🫠🐨*
