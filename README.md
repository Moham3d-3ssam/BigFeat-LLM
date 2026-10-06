# BigFeat: Scalable and Interpretable Automated Feature Engineering Framework

## What is BigFeat?
BigFeat is a scalable and interpretable automated feature engineering framework designed to enhance the quality of input features to maximize predictive performance based on a user-defined metric. It supports both **classification** and **regression** tasks, employing a dynamic feature generation and selection mechanism to construct expressive, interpretable features that improve prediction performance.

This repository adds one optional layer on top of the original framework: instead of manually picking BigFeat's hyperparameters (iterations, batch size, feature count, correlation threshold, tree depth, number of trees, and which operators to use) and its initial feature-importance guess, you can hand your dataset's column description to an LLM (ChatGPT, Gemini, DeepSeek, Claude, or your own) and have it produce that entire configuration for you — or, if you'd rather not call an API from code at all, preview the exact prompt, paste it into a chat UI yourself, and feed its JSON reply back in from a file. Using BigFeat the original, manual way still works exactly as before — this is purely additive.

## Folder structure

```
BigFeat-LLM/
├── README.md
└── setup/
    ├── __init__.py
    ├── bigfeat_base.py          # the BigFeat class itself
    ├── local_utils.py           # operator implementations (unary/binary/group_by)
    ├── llm_runner.py            # the LLM-driven auto-initialization layer
    ├── llm_config_template.py   # copy -> llm_config.py and fill in your own key(s)
    └── llm_config.py            # your personal config (not committed with real keys)
```

Your notebook/script sits next to `BigFeat-LLM/`, not inside it:

```
project/
├── BigFeat-LLM/        (everything above)
└── your_notebook.ipynb
```

## Setup and Installation

```bash
pip install pandas numpy scikit-learn lightgbm requests
```

If you plan to use `provider="gemini"`, also install Google's current SDK
(the older `google-generativeai` package reached end-of-life on
2025-11-30 and is no longer maintained):
```bash
pip install google-genai
```

Fill in the API key for the ONE provider you intend to use in
`setup/llm_config.py` — you do not need to fill in all four:
```python
# BigFeat-LLM/setup/llm_config.py
LLM_CONFIG = {
    "claude": {
        "api_key": "sk-ant-xxxxxxxxxxxxxxxx",
        "model": "claude-sonnet-4-5",
    },
}
```

## Usage from your notebook

Since `setup/` lives inside `BigFeat-LLM/`, add that folder to `sys.path`
once at the top of your notebook:

```python
import sys
sys.path.append("BigFeat-LLM")

import pandas as pd
from setup.llm_runner import preview_prompt, prepare_bigfeat_from_llm, run_bigfeat_with_llm, profile_dataframe
```

### Option A — the original, manual way

Still works exactly as it always has. You pick every value yourself.

```python
import numpy as np
from sklearn.model_selection import train_test_split
import setup.local_utils as local_utils
from setup.bigfeat_base import BigFeat

df = pd.read_csv("data.csv")
X = df.drop(columns="label")
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

bf = BigFeat(task_type="classification")

# n_trees, binary_operators and unary_operators are plain attributes -
# set them directly, or leave them alone to use the original defaults:
bf.n_trees = 200
bf.binary_operators = [np.multiply, np.add, np.subtract, local_utils.group_by]
bf.unary_operators = [np.abs, np.square, local_utils.unary_cube, local_utils.unary_sqrtabs, local_utils.original_feat]

X_train_bf = bf.fit(
    X_train.to_numpy(), y_train.to_numpy(),
    iterations=10, gen_size=10, N=20, alpha=10, eta=0.75, height=3,
    estimator="avg", random_state=42
)
X_test_bf = bf.transform(X_test.to_numpy())
```

### Option B — LLM-driven auto-initialization

**Step 1 — see exactly what would be sent, before deciding anything else.**
No API key, no model name, no JSON file needed yet:

```python
df = pd.read_csv("data.csv")
train_df, test_df = df.iloc[:int(len(df)*0.8)], df.iloc[int(len(df)*0.8):]

preview = preview_prompt(train_df, target_column="label")
print(preview["final_prompt"])   # system prompt + the dataset profile it built, together
```

**Step 2 — now get the actual configuration, one of two ways:**

**B.1 — call an LLM live:**
```python
bf = prepare_bigfeat_from_llm(
    df=train_df, target_column="label", provider="claude",
    extra_user_prompt="Recall on the minority class matters most here."  # optional
)
```

**B.2 — or, if you pasted `preview["final_prompt"]` into a chat UI yourself**
and saved its JSON reply to a file (e.g. `my_llm_reply.json`), skip the
live call entirely — no `provider`, `api_key`, or `model` needed:
```python
bf = prepare_bigfeat_from_llm(
    df=train_df, target_column="label",
    llm_response_path="my_llm_reply.json"
)
```

`provider` and `llm_response_path` are mutually exclusive — pass exactly
one. Both paths leave you with the same fully-configured `bf`.

**Step 3 — inspect what was used (works the same regardless of which path you took):**
```python
print(bf.system_prompt)   # the system prompt
print(bf.user_prompt)     # the dataset profile + your notes, as sent/intended
print(bf.final_prompt)    # both together
print(bf.response)        # the raw reply text (from the API, or read from your file)
print(bf.llm_config)      # that reply, parsed into a dict
print(bf.n_trees, bf.binary_operators, bf.unary_operators)
print(bf.fit_kwargs)      # ready-to-use fit() keyword arguments
```

**Step 4 — fit it:**
```python
X_train_bf = bf.fit(bf.X, bf.y, **bf.fit_kwargs)

_, _, X_test, y_test, _, _ = profile_dataframe(test_df, target_column="label")
X_test_bf = bf.transform(X_test)
```

**Or, the one-call version that also fits automatically** (same `provider`
XOR `llm_response_path` rule applies):
```python
bf = run_bigfeat_with_llm(df=train_df, target_column="label", provider="claude")
X_train_new_features = bf.X_transformed
```

### Reading back the result: which columns are new, and what are they built from?

BigFeat always places the newly generated features first and the original
input columns last in whatever `.fit()` / `.transform()` returns. Two
helpers in `llm_runner.py` make this easy to inspect, whether you got `bf`
from the LLM flow above or built it manually (Option A):

```python
from setup.llm_runner import describe_generated_features, split_original_vs_new

X_train_bf = bf.fit(bf.X, bf.y, **bf.fit_kwargs)

# One readable line per new feature: which original columns + operators
# went into building it (not the exact nested formula, just the ingredients)
for line in describe_generated_features(bf):
    print(line)

# Split the combined output back into (new_features, original_features)
new_only, original_only = split_original_vs_new(bf, X_train_bf)
print("New features shape:", new_only.shape)
```

### Supported providers, and adding your own

`provider="openai"`, `"gemini"`, `"deepseek"`, and `"claude"` are built in,
each with automatic wait-and-retry if you hit a rate limit (HTTP 429).
You only need to fill in the API key for the one you actually use in
`llm_config.py`. To add a fifth provider, add its key/model to
`llm_config.py` and a matching `"call"` function to the `PROVIDERS` dict
at the top of `llm_runner.py`.

## Key Parameters for `BigFeat.fit`
- `gen_size` (K): number of candidate features generated per iteration, as a multiplier of the input feature count.
- `random_state`: seed for reproducibility.
- `iterations`: number of feature generation/selection rounds.
- `estimator`: method for feature importance (`'avg'` uses RandomForest and LightGBM).
- `feat_imps`: whether to compute feature importance from a trained model to guide generation.
- `initial_importances`: optional externally supplied initial feature-importance vector (this is what the LLM layer fills in); overrides whatever `feat_imps` would otherwise compute, as the seed for the very first generation round only.
- `split_feats`: strategy for combining split-path information (`'comb'` or `'splits'`).
- `check_corr`: whether to check and remove highly correlated features.
- `N`: number of features retained at each selection step (independent of the original input feature count).
- `alpha`: number of stability-selection bootstrap models.
- `eta`: Pearson correlation threshold above which a feature is dropped as redundant.
- `height`: maximum depth of generated computation trees.
- `selection`: feature selection method (`'stability'` or `'fAnova'`).
- `combine_res`: whether to combine results across iterations.

## Cite Us
If you use BigFeat in your research, please cite the following paper:

```bib
@inproceedings{eldeeb2022bigfeat,
  title={BigFeat: Scalable and Interpretable Automated Feature Engineering Framework},
  author={Eldeeb, Hassan and Mohamed Essam},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={515--524},
  year={2022},
  organization={IEEE}
}
```