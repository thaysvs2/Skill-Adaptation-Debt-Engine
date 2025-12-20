<p align="center">
  <h1 align="center">Skill Adaptation Debt Engine</h1>
  <p align="center">
    A pressure-first lens on labor markets: measure <b>adaptation debt</b> (skill churn + novelty + breadth) instead of predicting outcomes.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-2ea44f" />
</p>

---
 
## What this project is

Most labor-market tools try to predict outcomes (salary, growth, “hot jobs”). This repo does something different:

**It treats each job role as a stability problem** and computes how much “skill adaptation debt” a role accumulates when:

- demanded skills churn month-to-month (instability / volatility),
- demanded skills are rare (hard to hire/train),
- and the role requires many distinct skills simultaneously (cognitive + coordination load).

The output is not “will you lose your job.”  
It’s **where stability is being borrowed**, where teams will feel **training friction**, and which roles have **high learning debt** if the market moves.

---

## Dataset & Copyright

This project uses the **“Future Jobs and Skills Demand 2025”** dataset from Kaggle:

* Dataset page: [https://www.kaggle.com/datasets/ahsanneural/future-jobs-and-skills-demand-2025](https://www.kaggle.com/datasets/ahsanneural/future-jobs-and-skills-demand-2025)

**Attribution**

* Full credit belongs to the dataset owner/uploader on Kaggle.
* This repository does **not** claim ownership of the dataset.

**License / Usage Notes**

* The dataset is subject to Kaggle’s dataset terms and the license shown on the dataset page.
* Please review the license **before** reusing, redistributing, or publishing derived versions of the raw data.
* This repo is intended for **educational and analytical** purposes; if you plan to include the raw dataset in repository/releases, make sure the dataset license explicitly allows redistribution.

**How this project uses the data**

* The app reads the dataset to compute **skill rarity**, **skill churn**, and **role/industry “adaptation debt”** metrics.
* The outputs (charts/tables) are derived analytics and do not replace the original dataset source.

---

## Core idea: Adaptation Debt (pressure-first metric)

Adaptation debt is a **weighted pressure index**, not a prediction.

### 1) Churn Pressure (instability over time)
A role can look fine today but still be unstable if its required skills **change frequently**.  
We measure churn as a proxy for “how quickly the definition of competence is moving.”

**High churn pressure** means:
- onboarding becomes harder,
- internal training becomes stale faster,
- hiring accuracy drops (because yesterday’s skill list no longer fits).

### 2) Novelty Pressure (rare + churning = learning friction)
A skill can be rare but stable (fine).  
A skill can be common but volatile (manageable).  
The hardest regime is **rare + volatile**, because it produces **planning failure**: you can’t reliably staff or train ahead of the curve.

Novelty pressure is designed to highlight:
- “the skill doesn’t exist widely yet”
- while also “the target keeps moving”

### 3) Breadth Pressure (role complexity)
Breadth is the **number of distinct skills** expected in the role.

High breadth pressure usually signals:
- a “Swiss-army role” (too much scope),
- a coordination burden (“you must integrate many systems”),
- or a maturity mismatch (companies stuffing multiple jobs into one listing).

### Final metric (what sliders control)
The sidebar “Debt weights (advanced)” controls the mixture:

- **Churn weight** → how much volatility dominates debt
- **Novelty weight** → how much rare+volatile dominates debt
- **Breadth weight** → how much complexity dominates debt

> You’re not changing “truth”, you’re changing the *lens* (what kind of instability you care about).

---

## App pages: what each view answers

### Role Report
**Question:** “For this specific role, what is the debt profile and which skills cause it?”

You get:
- adaptation debt score (final index),
- churn / novelty / breadth sub-pressures,
- top skill drivers (the “why” list).

### Industry Report
**Question:** “Inside an industry, which roles concentrate adaptation debt and why?”

You get:
- a ranked table of roles within one industry,
- pressure components per role,
- a quick “how to interpret” guide.

### Skill Explorer
**Question:** “Which skills are globally rare, churning, or novelty-heavy?”

You get:
- global counts/shares (how common),
- rarity,
- churn index,
- novelty index,
- and a search field for quick lookup.

### Scenario Simulator
**Question:** “If market pressure increases (more churn / more novelty / more breadth), how sensitive is this role?”

You get:
- baseline vs scenario debt,
- a delta explanation (“what this means”),
- and a simple visual comparison to communicate fragility vs robustness.

### Dataset Explorer
**Question:** “What’s actually inside the processed dataset? Can I inspect slices and export?”

You get:
- dataset metrics (rows / roles / industries / unique skills),
- filters,
- a preview table for transparency.

---

## How to run

### 1) Install
```bash
pip install -r requirements.txt
````

### 2) Launch Streamlit

```bash
streamlit run app/app.py
```

### 3) (Optional) If you have a CLI in this repo

```bash
python -m src.cli --help
```

---

## Plotly config (recommended)

Streamlit is deprecating Plotly keyword args and wants a `config=` dict.

Use something like this once and reuse it everywhere:

```python
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "responsive": True,
    "scrollZoom": True,
    "displaylogo": False,
}
```

Then render like:

```python
st.plotly_chart(fig, config=PLOTLY_CONFIG, width="stretch")
```

---

## Project structure

```text
Skill-Adaptation-Debt-Engine/
├─ app/
│  └─ app.py                 # Streamlit UI (all pages / plots)
├─ src/
│  ├─ config.py              # paths, constants, defaults (optional)
│  ├─ data.py                # load dataset, caching helpers
│  ├─ metrics.py             # churn/rarity/novelty/breadth + debt score
│  ├─ plots.py               # plotly chart builders (optional but clean)
│  └─ cli.py                 # CLI entry points (optional)
├─ data/
│  ├─ raw/                   # raw CSV from Kaggle (optional)
│  └─ processed/             # processed/scored outputs (optional)
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

# Screenshots & walkthrough

---

## 1) Dataset Explorer (transparency + slicing)

<img width="1331" height="653" alt="Screenshot 2025-12-20 at 17-45-07 Skill Adaptation Debt Engine" src="https://github.com/user-attachments/assets/8c5f14ac-89d7-4d84-b678-deb3d687abb0" />

**What you’re seeing**

* The left sidebar sets the *lens* (weights) and the *view mode*.
* The main panel is a transparency layer: it tells you what the app actually computed and stored.

**What the top counters mean**

* **Rows**: total job postings loaded into the view (after parsing/cleaning).
* **Roles**: distinct job titles normalized into role groups (depends on dataset).
* **Industries**: distinct categories (AI, Blockchain, Quantum Computing).
* **Unique skills**: distinct skills extracted after tokenization + cleanup.

**Why this view matters**

* Pressure metrics can feel abstract. This page prevents “black-box syndrome.”
* It’s where you sanity-check parsing (“Did ‘PyTorch’ become ‘Pytorch’ twice?”).
* It’s where you export slices (role-filtered, industry-filtered) for reports.

---

## 2) Industry Report (where debt concentrates)

<img width="1309" height="511" alt="Screenshot 2025-12-20 at 17-42-35 Skill Adaptation Debt Engine" src="https://github.com/user-attachments/assets/80204c5d-94a3-4632-88b8-70f3e950b7de" />

**What you’re seeing**

* A single industry is selected (example shows **AI**).
* Roles inside that industry are ranked by **debt_score**.

**How to read the columns**

* **debt_score**: final weighted pressure index (the “headline”).
* **churn_pressure**: volatility / month-to-month movement in demanded skills.
* **novelty_pressure**: rare + churning combined (learning friction).
* **breadth_pressure**: how many distinct skills the role stacks.
* **unique_skills**: raw count of extracted distinct skills (useful for debugging breadth).

**How to interpret rankings**

* Top roles aren’t necessarily “best paid” or “most important.”
* They’re the roles where organizations will feel:

  * hiring friction,
  * training lag,
  * skill mismatch,
  * and role definition drift.

---

## 3) Industry Report chart (top debt roles)

<img width="1025" height="356" alt="Screenshot 2025-12-20 at 17-42-51 Skill Adaptation Debt Engine" src="https://github.com/user-attachments/assets/ebd2b16f-0cb0-4b03-931c-27ced9253a38" />

**What you’re seeing**

* A quick visual of the **top adaptation-debt roles** in the selected industry.
* This chart is designed for *communication*: a manager can understand it in seconds.

**What “high debt” typically means in practice**

* Job descriptions are unstable or overloaded.
* Teams must constantly retool.
* Onboarding time increases.
* The role’s “definition of done” shifts frequently.

**Use case**

* Put this plot into an internal slide deck to justify:

  * upskilling budget,
  * narrowing scope (“split this role into two”),
  * improving documentation and internal tooling.

---

## 4) Skill Pressure Map (rarity × churn)

<img width="1005" height="475" alt="Screenshot 2025-12-20 at 17-43-45 Skill Adaptation Debt Engine" src="https://github.com/user-attachments/assets/d4608e23-1e43-49a6-8cfc-67e5e7a82c0d" />

**What you’re seeing**

* Each point is a skill.
* **x-axis = rarity** (how uncommon the skill is)
* **y-axis = churn_index** (how volatile demand is)
* Bubble size often encodes volume or impact (depends on implementation)

**How to read quadrants**

* **Top-right (rare + churning)** → worst zone (high novelty pressure)
* **Bottom-right (rare + stable)** → specialized but plan-able
* **Top-left (common + churning)** → market volatility, but easier staffing
* **Bottom-left (common + stable)** → low pressure baseline skills

**Why this is powerful**
This chart turns “skills” into a **map of risk**:

* it shows which capabilities are becoming unstable,
* and which unstable ones are also scarce (planning failure zone).

---

## 5) Scenario Simulator (baseline vs scenario)

<img width="1110" height="396" alt="Screenshot 2025-12-20 at 17-44-27 Skill Adaptation Debt Engine" src="https://github.com/user-attachments/assets/7802722a-dae4-4c3a-83c2-ec4bed031344" />

**What you’re seeing**

* A role is selected (example: AI Engineer).
* You scale pressure intensity:

  * churn intensity (market volatility),
  * novelty intensity (hard-to-find skills),
  * breadth intensity (role complexity).

**What the delta means**

* **Small delta** → the role is robust; extra volatility doesn’t change debt much.
* **Large delta** → the role is fragile; stability depends on calm conditions.
* This is **not** “probability of job loss.”
  It’s “how quickly the role becomes hard to staff and train.”

**How to use this**

* Stress-test strategic plans:

  * “If the market shifts faster next quarter, which roles break first?”
* Compare roles:

  * “Which roles remain stable even when churn rises?”

---

## 6) Skill Explorer (global table)

<img width="1092" height="574" alt="Screenshot 2025-12-20 at 17-43-18 Skill Adaptation Debt Engine" src="https://github.com/user-attachments/assets/c14bf4a8-ba69-4c8a-bae8-487eb1e28078" />

**What you’re seeing**
A searchable table that surfaces global skill properties:

* **global_count / global_share**: prevalence in the dataset
* **rarity**: inverse-ish measure of prevalence (rarer → higher)
* **churn_index**: volatility of demand over time
* **novelty_index**: combined proxy (rare + volatile)

**Why this exists**
If a role has high debt, you need to know *which skills* are responsible:

* the Skill Explorer lets you validate whether the model is “blaming” reasonable things
* and it helps you find leverage:

  * replace one novelty-heavy skill with an equivalent stable one,
  * or invest in building internal capability for a rare skill.

---

## 7) Scenario Simulator page (full context)

![Screenshot 2025-12-20 at 17-44-14 Skill Adaptation Debt Engine](https://github.com/user-attachments/assets/805e334e-6a07-49a8-a1bb-898db78d7ab2)

**What you’re seeing**
This is the full narrative around scenario simulation:

* a clear statement that this is **pressure-based**, not predictive
* definitions of what “churn/novelty/breadth” represent
* the resulting baseline and scenario debt values

**Why the text matters**
Most dashboards fail because users treat outputs as prophecy.
This UI tries to prevent that:

* it frames results as **counterfactual stress tests**
* and keeps interpretation anchored to operational reality.

---

## 8) Role Report (debt profile + drivers)

<img width="1310" height="602" alt="Screenshot 2025-12-20 at 17-38-53 Skill Adaptation Debt Engine" src="https://github.com/user-attachments/assets/079f9d8d-60e3-4f47-b881-ca19e3f4a357" />

**What you’re seeing**

* A role is selected.
* The top row shows:

  * adaptation debt (final score),
  * churn pressure,
  * novelty pressure,
  * breadth pressure.
* Then you see top skill drivers (skills most associated with the role’s debt).

**How to use this in real decisions**

* If churn dominates → reduce volatility exposure:

  * standardize internal tech stack,
  * invest in tooling/documentation,
  * reduce dependency on fast-moving libraries.
* If novelty dominates → reduce planning failure:

  * build internal capability,
  * partner with training pipelines,
  * simplify the skill graph.
* If breadth dominates → fix scope:

  * split the role,
  * create specialist tracks,
  * or restructure responsibilities.

---

## 9) Monthly churn lines (volatility evidence)

<img width="987" height="489" alt="Screenshot 2025-12-20 at 17-41-36 Skill Adaptation Debt Engine" src="https://github.com/user-attachments/assets/87d6ae9e-c805-454d-99fa-f169e1733ff7" />

**What you’re seeing**

* Time series of demand counts for top skills (by role or globally).
* This is “evidence layer” for churn.

**How to interpret**

* Smooth, stable lines → low churn pressure
* Spikes / dips / crossing patterns → high churn pressure
* A skill that rises fast may be “hot,” but it also creates training lag.

**Why it’s included**
Churn is easy to claim and hard to trust.
A chart like this makes churn *visible* instead of abstract.

---

## 10) Global skill space (rarity × churn overview)

![photo_2025-12-20_17-45-30](https://github.com/user-attachments/assets/505493b3-f137-48f0-907f-1d7a457b80c7)

**What you’re seeing**
A global map of the entire skill ecosystem:

* it shows clusters of stable/common skills vs risky skills
* and helps explain why certain roles are high debt (their skills live in the risky regions)

**How to use it**

* As a “macro” companion to Role Report:

  * Role Report tells you *which skills*
  * This map tells you *what kind of world those skills live in*

---

## What this repo is NOT

To keep interpretation honest:

* Not a “future salary predictor”
* Not a “job replacement probability model”
* Not a guarantee of hiring difficulty

It’s a structured way to answer:

> “Where is skill stability being borrowed from the future?”
