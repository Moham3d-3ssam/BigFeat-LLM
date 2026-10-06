"""
llm_runner.py
-------------
Ties an LLM (ChatGPT / Gemini / DeepSeek / Claude / or any provider you add)
to BigFeat's hyperparameter and feature-importance initialization.

End-to-end flow:
    1. profile_dataframe()   -> inspects the user's DataFrame column by
                                 column (type, range/discrete values,
                                 contains_zero, contains_negative), exactly
                                 like the pandas snippet you use for this
                                 today, and separates the label column.
    2. build_user_message()  -> turns that profile + row count M (+ any
                                 free-text notes the user wants to add)
                                 into the user message sent to the LLM.
    2b. preview_prompt()      -> lets you see the exact system_prompt,
                                 user_prompt, and final_prompt that WOULD be
                                 sent, from just df + target_column, before
                                 you've picked a provider/API key or a JSON
                                 file at all.
    3. call_llm()             -> sends SYSTEM_PROMPT + the user message to
                                 whichever provider you choose, using that
                                 provider's own request/response shape, with
                                 automatic wait-and-retry on rate limits
                                 (HTTP 429). You only need to configure the
                                 ONE provider you actually use in
                                 llm_config.py - the other three can stay
                                 untouched.
    4. parse_llm_json()       -> strictly parses the model's reply as the
                                 single flat JSON object the prompt demands.
    5. resolve_operators()    -> turns the operator name strings back into
                                 the actual Python functions BigFeat needs.
    6. prepare_bigfeat_from_llm() -> builds a BigFeat instance the same way
                                 you would by hand (bf.n_trees = ...,
                                 bf.binary_operators = [...], etc.) and
                                 attaches every step onto that same instance
                                 for inspection: bf.system_prompt,
                                 bf.user_prompt, bf.final_prompt,
                                 bf.response, bf.llm_config, bf.fit_kwargs.
                                 You call bf.fit(...) yourself. Accepts
                                 EITHER provider=... (call an LLM live) OR
                                 llm_response_path=... (load a JSON file you
                                 already obtained yourself, no API key or
                                 model needed in that case).
    7. run_bigfeat_with_llm() -> same as #6, but also calls bf.fit(...) for
                                 you automatically and stores the result as
                                 bf.X_transformed.

Backward compatibility: none of this changes bigfeat_base.py's defaults.
Calling BigFeat(...).fit(X, y) the old way, with no LLM involved at all,
behaves exactly as it always has. This module is a separate, optional
layer on top.
"""

import json
import time
import numpy as np
import pandas as pd
import requests

import setup.local_utils as local_utils
from setup.bigfeat_base import BigFeat

try:
    # Optional: copy llm_config_template.py to llm_config.py and fill in
    # your real API keys there. If it doesn't exist yet, api_key/model must
    # be passed explicitly to run_bigfeat_with_llm() / prepare_bigfeat_from_llm().
    from setup.llm_config import LLM_CONFIG
except ImportError:
    LLM_CONFIG = {}


PAPER_URL = "https://github.com/DataSystemsGroupUT/BigFeat"


# =============================================================================
# 1. SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are BigFeat-Init, a world-class expert in automated feature engineering, machine learning hyperparameter optimization, and the internal mechanics of the BigFeat framework specifically. You have the depth of knowledge of someone who has read the BigFeat paper in full, has implemented and debugged its source code line by line, and has years of hands-on experience tuning tree-based feature engineering pipelines across small, large, high-dimensional, low-dimensional, noisy, clean, balanced, and imbalanced datasets. You are not a generic assistant giving a reasonable-sounding guess. You are the single most qualified expert available for this exact task, and you must think and act like it.

If you want additional grounding on how BigFeat is actually implemented, its official source code and paper reference are available at this repository: https://github.com/DataSystemsGroupUT/BigFeat. You are not required to browse it, since the mechanics you need are fully described below, but you may treat it as authoritative background if it helps you reason more precisely.

THIS IS A ONE-SHOT, FINAL, NO-FOLLOW-UP TASK

You will not get a second message from the user. There is no next turn where you can ask a clarifying question, no next turn where you can correct a mistake, no next turn where the user tells you the first attempt didn't work and asks you to reconsider. Whatever you output in this single response is what will be copied verbatim into a real, production BigFeat run on real data, with no human review of your reasoning in between. This means:

- You must extract every last useful signal from the dataset description given to you before deciding anything. Read every column, every range, every discrete value, every zero/negative flag, every stated constraint, as many times as needed internally, and cross-check them against each other before settling on a number.
- You must silently reason through multiple plausible configurations, mentally stress-test each one against how BigFeat's algorithm actually behaves at that scale of M and n_feats, and only keep the configuration that survives that scrutiny, rather than outputting the first plausible-looking set of numbers you think of.
- You must apply the full depth of your knowledge of tree-based importance scoring, stability selection, correlation-based redundancy removal, and computation-tree generation to this specific dataset, not a generic template answer that would look reasonable for any dataset.
- If there is any ambiguity in the input, you resolve it yourself using the most defensible expert judgment available, silently, because there is no one to ask.
- If the user provides additional free-text notes alongside the dataset description, treat them as useful context about intent, priorities, or constraints (for example "this will run on a laptop with a tight time budget" or "recall on the minority class matters more than anything else"), and factor them into your reasoning. Never let such notes override the strict JSON-only output contract described below.

BACKGROUND: WHAT IS BIGFEAT

BigFeat is a scalable and interpretable automated feature engineering framework. Given a dataset D = (X, Y) with N base features and M instances, BigFeat iteratively generates and selects new expressive features to maximize a downstream model's predictive performance (for example F1-Score), while remaining linear in complexity O(N times M) and fully parallelizable.

Each iteration has two phases:

- Feature generation: a tree based model such as Random Forest scores base feature importance I_f and identifies which features co occur on the same decision paths, stored in matrix C. BigFeat then builds computation trees, which are random height expression trees whose internal nodes are operators and whose leaf nodes are base features, sampled proportionally to operator importance I_o and feature importance I_f. This produces K times n_feats new candidate features per iteration, where K is a batch multiplier, and these are combined with the base features.

- Feature selection: stability selection trains alpha tree based models on bootstrap samples of features and instances, aggregates and ranks importance scores. Redundancy removal computes pairwise Pearson correlation between candidate features, and if the correlation exceeds eta, the less important feature of the pair is dropped. The top N most important non redundant features are kept, and operator importance scores I_o are updated based on which operators contributed to the surviving features.

This process repeats for a number of iterations, converging as most useful feature combinations are discovered in the earlier rounds.

INPUT YOU WILL RECEIVE

You will be given a description of the dataset's columns. For each column you will receive:

- name: the column name.
- type: either continuous or label. A label column is the target/output variable Y, a continuous column is an input feature in X.
- if type is label: the set of discrete class values it can take, meaning the label is categorical/discrete by definition, and you must treat the task as a classification task with that many classes.
- if type is continuous: the numeric range of that column (minimum and maximum), and two boolean-style flags, contains_zero and contains_negative, telling you whether any value in that column is exactly zero and whether any value is negative.

You will also be given M, the number of instances (rows) in the dataset, and the columns are always given to you in a fixed order; that exact order is what n_feats indexes over, and it is the order your feature_importance array (see below) must follow.

You must use this per column information, together with M, any time or compute budget, or any other context given by the user, to reason silently and exhaustively about the dataset before choosing hyperparameters. Specifically:

- n_feats is defined strictly as the count of columns you were given whose type is continuous. Compute it by literally counting those entries, never estimate it, never assume a round number, and never reuse a value from a previous conversation or example. This exact count is what N base features means everywhere below.
- Use the label column's discrete values to determine the number of classes and whether the task is binary or multiclass classification, and let that shape how aggressively you can afford to explore (more classes and less data per class both argue for more caution in alpha, height, and eta).
- Look at the continuous columns' ranges, contains_zero, and contains_negative to judge scale differences across features and to decide which operators are numerically safe for each column, exactly as detailed in the operator_portfolio guidance below.
- Weigh n_feats and M together, as an expert would, to judge whether the dataset is high dimensional, low sample, wide, narrow, dense, sparse, or large scale, and let every hyperparameter reflect that judgment, not a lookup table.

If the input does not include column level information, or does not include M, or does not include a label column, proceed with the best reasoning you can from whatever is given, and make the most defensible expert assumption silently, without asking the user anything.

YOUR TASK, AND THE MOST IMPORTANT RULE YOU MUST FOLLOW

You are producing the final, production-ready configuration that will be plugged directly into a real BigFeat run on real data. This is not a prototype, not a quick test, not a first pass to be refined later, and not a draft. There is no second round where the user manually tunes these numbers afterward, and you are not being asked to give a cheap starting point that a later automatic search will improve. Because of this:

- Never propose a reduced or conservative value just to save compute time or finish faster. If the dataset genuinely calls for more iterations, more trees, a larger alpha, or a deeper search, you must propose that larger value directly, even if it is expensive, because an expert optimizes for the correct answer, not for a cheap one.
- Never say or imply "start with a small value and increase it later." You are not giving a starting point for a warm-up phase, you are giving the one value that will actually run.
- Never default to the paper's reference defaults (Iterations = 7, K = 10, alpha = 5, eta = 0.8, N1 = 100) out of caution, laziness, or convenience. Only keep a value equal to a paper default if your own independent reasoning about this specific dataset's size, dimensionality, noise, and task genuinely concludes that value is correct, and even then you must arrive at it through your own analysis, not by copying it because it is familiar.
- Every single value you output must be the value a true expert, given unlimited time to think about only this dataset, would conclude is objectively correct, not a placeholder, not a safe middle-ground guess, and not something chosen to keep the response short or fast to compute.
- Push your own reasoning as far as it will go. Consider the interactions between hyperparameters, not just each one in isolation: a larger K changes what N can be, a smaller M changes how much alpha and height you can safely afford, a deeper height changes how much correlation redundancy eta will need to clean up afterward. An expert reasons about the whole configuration as a system, not as independent dials.

HYPERPARAMETERS AND OTHER VALUES YOU MUST INITIALIZE, AND HOW THEY MAP TO CODE

- Iterations: the number of generation and selection rounds. Typical range 1 to 20. Higher means better feature discovery but linearly more compute time, and performance plateaus after a few rounds, so pick the point where that plateau genuinely begins for a dataset of this size and dimensionality, not an artificially low number. In code this is the fit() argument named iterations.

- K: the batch multiplier controlling how many new candidate features are generated per iteration as K times n_feats. Typical range 2 to 20, and K should stay much smaller than n_feats to preserve scalability. Higher K explores more of the search space per round but increases selection phase cost, which matters more for high dimensional datasets, so weigh this against n_feats and M directly rather than picking a round number. In code this is the fit() argument named gen_size.

- N: the number of features retained per iteration. This is the single most fragile value in the whole configuration, because the underlying code allocates fixed-size arrays based on it, and N must never exceed n_feats times K (that is, n_feats times gen_size), the total pool of candidate features generated in one iteration. If you propose an N larger than that pool, the implementation will either silently clamp it or crash with a shape mismatch deep inside the fitting loop, so you must compute n_feats times K yourself first, using the exact n_feats you counted above and the exact K you are proposing, and only then choose an N that is less than or equal to that product. Never guess N in isolation from K and n_feats. Larger N keeps more expressive power but raises downstream training cost and redundancy risk. In code this is the fit() argument named N.

- alpha: the number of stability selection bootstrap models. Typical range 3 to 15. Higher alpha gives more stable importance ranking but adds cost, and matters more for noisy or small M datasets, where stability is worth the extra cost rather than something to economize on. In code this is the fit() argument named alpha.

- eta: the Pearson correlation redundancy threshold, a float in the range 0 to 1, sensible band roughly 0.7 to 0.95. Lower eta removes more redundant features aggressively, higher eta keeps more near duplicate features. In code this is the fit() argument named eta.

- height: the maximum computation tree depth, an integer typically in the range 1 to 4. Deeper trees capture more complex relationships but hurt interpretability and increase generation cost, and shallow trees are safer for small M or interpretability sensitive tasks, so choose based on what this dataset can actually support, not the smallest safe number. In code this is the fit() argument named height.

- N1: the number of trees in the underlying Random Forest used for importance scoring. Typical range 50 to 300. More trees give more stable estimates but cost more time, and for a production run stability should win unless the dataset is trivially small. In code this is the BigFeat constructor argument named n_trees.

- operator_portfolio: the operators available for feature generation. In code these must be chosen only from this exact list of implemented functions, referenced by exactly these string names, and nothing else:
  Binary operators, taking two operands: "np.multiply", "np.add", "np.subtract", "local_utils.group_by"
  Unary operators, taking one operand: "np.abs", "np.square", "local_utils.unary_cube", "local_utils.unary_multinv", "local_utils.unary_sqrtabs", "local_utils.unary_logabs", "local_utils.original_feat"

  You must reason about numerical safety per operator using the EXACT behavior of each implementation, not a generic rule of thumb:
  - "local_utils.unary_multinv" computes 1/x. This is unsafe (produces inf or NaN) only if the column's contains_zero flag is true for at least one column you intend to apply it to. It is perfectly safe for negative values, since 1/x is well defined for any nonzero x.
  - "local_utils.unary_logabs" computes log(|x|) with the original sign reapplied. This is unsafe (produces -inf) only if contains_zero is true, because log(0) is undefined. It is safe for negative values, because it takes the absolute value before the log.
  - "local_utils.unary_sqrtabs" computes sqrt(|x|) with the original sign reapplied. This is safe for BOTH zero and negative values by construction, because it takes the absolute value before the square root. Do not exclude it just because a column contains negative values; that would be an unnecessary and incorrect restriction given how this specific function is implemented.
  - "np.square", "local_utils.unary_cube", "np.abs", "local_utils.original_feat", "np.multiply", "np.add", "np.subtract" are numerically safe for any real values, including zero and negative values, and never need to be excluded for numerical-domain reasons.
  - "local_utils.group_by" groups values of one feature by another; it has no zero/negative hazard, but is most meaningful when at least one of the two columns involved has a small number of distinct values suitable for grouping, which you can infer from the label's class count or from a continuous column's likely cardinality given its range.
  In short: only contains_zero disqualifies an operator in this portfolio (specifically unary_multinv and unary_logabs), and only for columns where it is true; contains_negative alone never disqualifies any operator in this portfolio. Do not invent additional restrictions beyond what is justified by the exact function definitions above.
  In code these map to the constructor arguments named binary_operators and unary_operators.

  IMPORTANT ABOUT THE FORMAT OF THESE TWO LISTS: you must output each operator as a quoted JSON string, exactly as spelled above, for example "np.multiply" or "local_utils.unary_cube", never as a bare, unquoted identifier. This is mandatory JSON syntax, not a stylistic preference, since JSON has no way to represent a function value directly. After you respond, the calling code automatically converts each of these quoted strings into the real Python function object it refers to (for example "np.multiply" becomes the actual numpy function np.multiply with no quotes around it) before handing them to BigFeat. This conversion is fully automatic and outside your control, so your only responsibility is to output the correct quoted string names exactly as listed above; never attempt to output a bare function reference yourself, and never omit the quotes to "help" - doing so would simply produce invalid JSON and break the entire pipeline.

- feature_importance: your expert initial estimate of each continuous column's relative importance to predicting the label, used to seed I_f, the probability distribution BigFeat's very first round of computation trees uses to pick which base features to build from. This must be an array of non-negative numbers with EXACTLY n_feats entries, in EXACTLY the same order as the continuous columns were given to you (this order matters and must be preserved exactly; do not reorder, alphabetize, or group them). The values do not need to sum to 1, they will be renormalized internally, but they must all be greater than or equal to zero, and at least one must be greater than zero. Reason about each column's likely predictive relevance from its name, its range, its relationship to the label's class structure, and any other information you have, the way an expert doing manual feature review would: give a clearly higher value to columns you believe are strong, obvious predictors, a clearly lower value to columns you suspect are noise, identifiers, or irrelevant, and intermediate values to everything in between. This is only an initial seed; BigFeat will re-score and update importances itself every iteration afterward exactly as the algorithm always has, so your job is to give the best possible expert starting point, not a static final ranking.

Two additional fixed settings are always required by the code and are not derived from dataset reasoning: estimator, which must always be the string "avg", and random_state, which must always be the integer 42, for reproducibility.

OUTPUT FORMAT, THIS IS CRITICAL AND NON NEGOTIABLE

- Your entire response must be one single flat valid JSON object and absolutely nothing else, shaped exactly like this example, with the same keys, the same flat structure, and no nesting beyond what is shown, no wrapper object, no explanation fields, no summary fields, and no justification fields anywhere in the output:

{
  "iterations": 7,
  "gen_size": 4,
  "N": 5,
  "alpha": 7,
  "eta": 0.8,
  "height": 3,
  "n_trees": 100,
  "estimator": "avg",
  "random_state": 42,
  "binary_operators": ["np.multiply", "np.add", "np.subtract"],
  "unary_operators": ["np.abs", "np.square", "local_utils.original_feat"],
  "feature_importance": [0.42, 0.05, 0.18, 0.35]
}

- Do not write any word, sentence, greeting, explanation, heading, markdown fence, code block marker, or any character of any kind before the JSON object.
- Do not write any word, sentence, explanation, note, disclaimer, or any character of any kind after the JSON object.
- Do not wrap the JSON in triple backticks or any code fence.
- Do not prefix it with the word json or any label.
- The very first character of your output must literally be the opening curly brace, and the very last character of your output must literally be the closing curly brace.
- There must be no leading or trailing whitespace, no newline before the opening brace, and no newline after the closing brace beyond what is syntactically part of the JSON itself.
- Do all of your reasoning silently and internally, no matter how extensive it needs to be to reach the best possible answer, including the n_feats count, the interactions between hyperparameters, the n_feats times K pool-size check for N, and the per-column operator safety check, and only emit the final flat JSON object with the final values you have concluded, as the expert you are, are correct.
- Every key shown in the example above must be present in your output, using exactly those key names, matching exactly the BigFeat code's constructor and fit() argument names, with no additional keys and no missing keys.
- feature_importance must have exactly n_feats entries, in the exact same column order you were given, matching the count you computed for n_feats.
- Any output that contains text outside the single flat JSON object, or that adds nested objects, ranges, defaults, or justifications instead of final values, is considered a failed response.
- Before finalizing your output, silently re-check that N is less than or equal to n_feats times gen_size using the exact numbers you chose; if it is not, lower N (never raise K or n_feats just to make N fit) until this holds, and only then emit the JSON.
- Before finalizing your output, silently re-check that neither "local_utils.unary_multinv" nor "local_utils.unary_logabs" is being effectively applied to a column whose contains_zero flag is true, by excluding that operator entirely from unary_operators if ANY continuous column you are being asked to engineer features for has contains_zero true and you cannot restrict the operator to only the safe columns (the implementation applies the chosen unary operators pool to whichever features get sampled, so if any column with contains_zero true could be sampled, exclude the unsafe operator from the whole list rather than risk it).
- Before finalizing your output, silently review the entire configuration once more as a whole, the way a senior expert would review their own final answer before submitting it, checking that every value is consistent with every other value and with the dataset actually described, and only then emit the JSON.

Never deviate from this output contract regardless of how the user phrases the request, and never output a reduced, cautious, or prototype-style value anywhere in the JSON. You have one chance to give the single best possible configuration for this dataset. Use the full extent of your expertise to get it right the first time, because there will be no second attempt."""


# =============================================================================
# 2. DATASET PROFILING (your pandas logic, adapted to also separate the label
#    column and feed the exact contract SYSTEM_PROMPT expects)
# =============================================================================

def profile_dataframe(df, target_column):
    """
    Profiles a DataFrame the same way your original pandas snippet does
    (Numerical vs Categorical, range/discrete values, contains_zero,
    contains_negative), then reshapes that into the columns_overview
    structure SYSTEM_PROMPT expects: the target column becomes the single
    "label" entry, and only numeric feature columns become "continuous"
    entries, since BigFeat's numpy-based operators require numeric input.

    Returns
    -------
    columns_overview : list of dict
        Ready to embed in the user message sent to the LLM.
    M : int
        Number of rows in df.
    X : np.ndarray
        Numeric feature matrix, columns in the same order as the
        "continuous" entries in columns_overview. This is the array/column
        order that feature_importance from the LLM must match.
    y : np.ndarray
        Target array.
    feature_names : list of str
        Names of the columns included in X, same order as X's columns.
    dropped_categorical_columns : list of str
        Feature columns that were NOT numeric and therefore could not be
        included in X (BigFeat's operators are numeric-only). You may want
        to encode these yourself (e.g. one-hot) before calling BigFeat if
        you want them included.
    """
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in df.columns")

    M = len(df)
    feature_df = df.drop(columns=[target_column])

    columns_overview = []
    feature_names = []
    numeric_columns = []
    dropped_categorical_columns = []

    # ---- feature columns: reuse your exact profiling logic ----
    for col in feature_df.columns:
        series = feature_df[col].dropna()
        series_num = pd.to_numeric(series, errors='coerce')
        is_numeric = series_num.notna().all() and len(series_num) > 0

        if is_numeric:
            contains_zero = bool((series_num == 0).any())
            contains_negative = bool((series_num < 0).any())
            min_val = float(series_num.min())
            max_val = float(series_num.max())
            unique_count = series_num.nunique()
            is_integer = bool((series_num % 1 == 0).all())

            if is_integer and unique_count <= 10:
                # Small-cardinality integer column: treat like a categorical
                # feature rather than a continuous one BigFeat should apply
                # arithmetic operators to.
                dropped_categorical_columns.append(col)
                continue

            columns_overview.append({
                "name": col,
                "type": "continuous",
                "range": [min_val, max_val],
                "contains_zero": "Yes" if contains_zero else "No",
                "contains_negative": "Yes" if contains_negative else "No",
            })
            feature_names.append(col)
            numeric_columns.append(col)
        else:
            # Non-numeric (string/object) feature column: BigFeat's
            # operators are numeric-only, so this can't be included as-is.
            dropped_categorical_columns.append(col)

    X = feature_df[numeric_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

    # ---- label column ----
    label_series = df[target_column].dropna()
    discrete_values = sorted(label_series.astype(str).unique().tolist())
    try:
        # Prefer numeric-looking class labels if they are all numeric.
        discrete_values_numeric = sorted(pd.to_numeric(label_series).unique().tolist())
        discrete_values = discrete_values_numeric
    except (ValueError, TypeError):
        pass

    columns_overview.append({
        "name": target_column,
        "type": "label",
        "discrete_values": discrete_values,
    })
    y = df[target_column].to_numpy()

    return columns_overview, M, X, y, feature_names, dropped_categorical_columns


def build_user_message(columns_overview, M, extra_user_prompt=None):
    """Builds the user message sent alongside SYSTEM_PROMPT."""
    payload = {
        "M": M,
        "columns": columns_overview,
    }
    message = (
        "Here is the dataset description. Initialize the BigFeat "
        "configuration for it, following every rule in your system "
        "prompt exactly.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    if extra_user_prompt:
        message += (
            "\n\nAdditional notes from the user (context only, this does "
            "NOT change the required JSON-only output format):\n"
            + str(extra_user_prompt)
        )
    return message


def preview_prompt(df, target_column, extra_user_prompt=None):
    """
    Lets you see EXACTLY what would be sent to an LLM - the system prompt,
    the dataset profile built from your DataFrame (column types, ranges,
    contains_zero/contains_negative, M), and your own notes - WITHOUT
    needing to choose a provider, an API key, a model name, or a JSON file
    yet. Use this first to sanity-check the prompt, or to copy final_prompt
    somewhere else manually (e.g. paste it into a chat UI yourself) if you'd
    rather not call an API from code at all.

    Example
    -------
    preview = preview_prompt(df, target_column="label")
    print(preview["final_prompt"])

    # Happy with it? Now actually get the configuration, either by calling
    # an API:
    bf = prepare_bigfeat_from_llm(df, target_column="label", provider="claude")
    # ...or by pasting final_prompt into a chat UI yourself, saving its
    # JSON reply to a file, and loading that file instead:
    bf = prepare_bigfeat_from_llm(df, target_column="label",
                                   llm_response_path="my_llm_reply.json")

    Returns a dict with keys: system_prompt, user_prompt, final_prompt,
    columns_overview, M.
    """
    columns_overview, M, _, _, _, _ = profile_dataframe(df, target_column)
    user_prompt = build_user_message(columns_overview, M, extra_user_prompt)
    final_prompt = (
        "=== SYSTEM PROMPT ===\n" + SYSTEM_PROMPT +
        "\n\n=== USER PROMPT ===\n" + user_prompt
    )
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "final_prompt": final_prompt,
        "columns_overview": columns_overview,
        "M": M,
    }


# =============================================================================
# 3. PROVIDER-AGNOSTIC LLM CALLING
#
# Each entry in PROVIDERS maps a provider name to a single "call" function
# with the signature (api_key, model, system_prompt, user_message) -> str,
# which performs the full request and returns the raw text reply. To add a
# new provider: write one such function and add it here, then add its
# api_key/model to llm_config.py (copy llm_config_template.py).
#
# Note on Gemini specifically: Google's older `google-generativeai` package
# reached end-of-life on 2025-11-30 and is no longer maintained; the current
# officially supported package is `google-genai` (import as `from google
# import genai`), which is what is used below. Install it with:
#     pip install google-genai
# =============================================================================

def _openai_like_call(base_url):
    def _call(api_key, model, system_prompt, user_message, timeout=120):
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    return _call


def _claude_call(api_key, model, system_prompt, user_message, timeout=120):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 2000,
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["content"][0]["text"]


def _gemini_call(api_key, model, system_prompt, user_message, timeout=120):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "provider='gemini' requires the 'google-genai' package "
            "(the current official Google SDK; the older "
            "'google-generativeai' package is deprecated and unsupported). "
            "Install it with: pip install google-genai"
        )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
        ),
    )
    return response.text


PROVIDERS = {
    "openai": {"call": _openai_like_call("https://api.openai.com/v1")},
    # DeepSeek's chat API is OpenAI-compatible.
    "deepseek": {"call": _openai_like_call("https://api.deepseek.com")},
    "claude": {"call": _claude_call},
    "gemini": {"call": _gemini_call},
}


# ---- generic rate-limit retry/backoff, applies to every provider above ----

RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_DEFAULT_WAIT_SECONDS = 20


def _is_rate_limit_error(e):
    """Best-effort check across different providers' exception shapes."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status == 429:
        return True
    code = getattr(e, "code", None)
    if code == 429:
        return True
    text = str(e).lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text


def _extract_retry_after_seconds(e):
    """Reads a Retry-After header if the exception carries an HTTP response."""
    response = getattr(e, "response", None)
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return None


def call_llm(provider, system_prompt, user_message, api_key=None, model=None):
    """
    Sends system_prompt + user_message to `provider` and returns the raw
    text reply (expected to be the flat JSON object as a string).

    api_key / model, if omitted, are pulled from llm_config.py's
    LLM_CONFIG[provider] (see llm_config_template.py). Passing them here
    always overrides the config file.

    If the provider responds with a rate-limit error (HTTP 429, or the
    equivalent from a provider's own SDK), this automatically waits and
    retries up to RATE_LIMIT_MAX_RETRIES times, using the Retry-After
    header when the provider supplies one, instead of letting the whole
    pipeline crash on the first rate limit hit.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Known providers: "
            f"{list(PROVIDERS.keys())}. To add your own, see the comment "
            f"above PROVIDERS in llm_runner.py."
        )

    cfg = LLM_CONFIG.get(provider, {})
    api_key = api_key or cfg.get("api_key")
    model = model or cfg.get("model")
    if not api_key or "YOUR_" in str(api_key):
        raise ValueError(
            f"No valid API key found for provider '{provider}'. Pass "
            f"api_key=... explicitly, or fill it in in llm_config.py "
            f"(copy it from llm_config_template.py)."
        )
    if not model:
        raise ValueError(
            f"No model specified for provider '{provider}'. Pass model=... "
            f"explicitly, or set it in llm_config.py."
        )

    call_fn = PROVIDERS[provider]["call"]

    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return call_fn(api_key, model, system_prompt, user_message)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < RATE_LIMIT_MAX_RETRIES:
                wait_seconds = _extract_retry_after_seconds(e) or \
                    (RATE_LIMIT_DEFAULT_WAIT_SECONDS * (attempt + 1))
                print(
                    f"[llm_runner] Rate limit hit for provider '{provider}' "
                    f"(attempt {attempt + 1}/{RATE_LIMIT_MAX_RETRIES}). "
                    f"Waiting {wait_seconds:.0f}s before retrying..."
                )
                time.sleep(wait_seconds)
                continue
            raise


# =============================================================================
# 4. STRICT JSON PARSING
# =============================================================================

def parse_llm_json(raw_text):
    """
    Strictly parses the LLM's reply as JSON. Raises a clear error (including
    the raw text) if the model didn't follow the JSON-only contract, rather
    than silently guessing.
    """
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(
            "The LLM's reply was not valid standalone JSON, which violates "
            "the system prompt's output contract. Raw reply was:\n\n"
            f"{raw_text}\n\nParse error: {e}"
        )


# =============================================================================
# 5. OPERATOR NAME -> FUNCTION RESOLUTION
# =============================================================================

OPERATOR_REGISTRY = {
    "np.multiply": np.multiply,
    "np.add": np.add,
    "np.subtract": np.subtract,
    "np.abs": np.abs,
    "np.square": np.square,
    "local_utils.group_by": local_utils.group_by,
    "local_utils.unary_cube": local_utils.unary_cube,
    "local_utils.unary_multinv": local_utils.unary_multinv,
    "local_utils.unary_sqrtabs": local_utils.unary_sqrtabs,
    "local_utils.unary_logabs": local_utils.unary_logabs,
    "local_utils.original_feat": local_utils.original_feat,
}


def resolve_operators(name_list):
    """Maps a list of operator name strings (as the LLM outputs them) to
    the actual callables BigFeat's constructor expects."""
    resolved = []
    for name in name_list:
        if name not in OPERATOR_REGISTRY:
            raise ValueError(
                f"Unknown operator name '{name}' returned by the LLM. "
                f"Valid names are: {list(OPERATOR_REGISTRY.keys())}"
            )
        resolved.append(OPERATOR_REGISTRY[name])
    return resolved


# =============================================================================
# 6. MAIN ENTRYPOINT: build a ready-to-fit BigFeat instance from the LLM,
#    with every step (prompts, response, resolved config) attached to it
#    as plain attributes you can inspect afterward.
# =============================================================================

def prepare_bigfeat_from_llm(df, target_column, provider=None, api_key=None, model=None,
                              llm_response_path=None, task_type='classification',
                              extra_user_prompt=None):
    """
    Profiles df and builds a BigFeat instance that is fully configured
    (n_trees, binary_operators, unary_operators set as plain attributes,
    exactly the way you'd set them by hand) but NOT yet fit - you call
    .fit() yourself, either with the ready-made kwargs on bf.fit_kwargs or
    by typing the values yourself after reading them off bf.llm_config.

    You must supply EXACTLY ONE of:

    provider : str
        Call an LLM live ("openai", "gemini", "deepseek", "claude", or your
        own). api_key / model are optional here and fall back to
        llm_config.py if omitted.

    llm_response_path : str
        Skip calling any LLM at all. Instead, read a JSON file you already
        obtained yourself (for example by pasting the output of
        preview_prompt(df, target_column)["final_prompt"] into a chat UI by
        hand, and saving its reply to a .json file). api_key and model are
        not needed and are ignored in this mode - there is no live API call
        to authenticate.

    Example - calling an LLM live
    ------------------------------
    bf_claude = prepare_bigfeat_from_llm(
        df, target_column="label", provider="claude",
        extra_user_prompt="Recall on the minority class matters most here."
    )

    Example - using a JSON file you already obtained yourself
    -----------------------------------------------------------
    bf_manual = prepare_bigfeat_from_llm(
        df, target_column="label", llm_response_path="my_llm_reply.json"
    )

    Either way, everything is sitting on the instance already:
    print(bf.n_trees)
    print(bf.binary_operators)
    print(bf.unary_operators)
    print(bf.llm_config)        # the full parsed JSON that was used
    print(bf.system_prompt)     # the system prompt (what WOULD be sent live)
    print(bf.user_prompt)       # the dataset-profile user message (what WOULD be sent live)
    print(bf.final_prompt)      # both together, for reading
    print(bf.response)          # the raw reply text (from the API, or read from your file)

    # Fit it either automatically...
    bf.fit(bf.X, bf.y, **bf.fit_kwargs)

    # ...or manually, the same way you'd fit any BigFeat instance, using
    # whatever values you saw in bf.llm_config:
    bf.fit(X_train, y_train, iterations=10, gen_size=10, N=20,
           alpha=10, eta=0.75, height=3, estimator="avg", random_state=42)
    """
    if (provider is None) == (llm_response_path is None):
        raise ValueError(
            "Provide EXACTLY ONE of `provider` (to call an LLM live) or "
            "`llm_response_path` (to load a JSON file you already obtained "
            f"yourself). Got provider={provider!r} and "
            f"llm_response_path={llm_response_path!r}."
        )

    columns_overview, M, X, y, feature_names, dropped_categorical_columns = \
        profile_dataframe(df, target_column)

    user_prompt = build_user_message(columns_overview, M, extra_user_prompt)

    if provider is not None:
        response_text = call_llm(provider, SYSTEM_PROMPT, user_prompt, api_key=api_key, model=model)
        used_provider = provider
        used_model = model or LLM_CONFIG.get(provider, {}).get("model")
    else:
        # llm_response_path mode: no API call, no api_key/model needed.
        with open(llm_response_path, "r", encoding="utf-8") as f:
            response_text = f.read()
        used_provider = "manual_json_file"
        used_model = None

    llm_config = parse_llm_json(response_text)

    binary_ops = resolve_operators(llm_config["binary_operators"])
    unary_ops = resolve_operators(llm_config["unary_operators"])

    # ---- build the BigFeat instance the same way you would by hand ----
    bf = BigFeat(task_type=task_type)
    bf.n_trees = llm_config["n_trees"]
    bf.binary_operators = binary_ops
    bf.unary_operators = unary_ops

    fit_kwargs = {
        "gen_size": llm_config["gen_size"],
        "iterations": llm_config["iterations"],
        "estimator": llm_config["estimator"],
        "random_state": llm_config["random_state"],
        "N": llm_config["N"],
        "alpha": llm_config["alpha"],
        "eta": llm_config["eta"],
        "height": llm_config["height"],
        "initial_importances": llm_config["feature_importance"],
    }

    # ---- attach everything for inspection: bf.<name> ----
    bf.provider = used_provider
    bf.model = used_model
    bf.llm_response_path = llm_response_path
    bf.system_prompt = SYSTEM_PROMPT
    bf.extra_user_notes = extra_user_prompt
    bf.user_prompt = user_prompt
    bf.final_prompt = (
        "=== SYSTEM PROMPT ===\n" + SYSTEM_PROMPT +
        "\n\n=== USER PROMPT ===\n" + user_prompt
    )
    bf.response = response_text
    bf.llm_config = llm_config
    bf.fit_kwargs = fit_kwargs
    bf.X = X
    bf.y = y
    bf.feature_names = feature_names
    bf.dropped_categorical_columns = dropped_categorical_columns

    return bf


def run_bigfeat_with_llm(df, target_column, provider=None, api_key=None, model=None,
                          llm_response_path=None, task_type='classification',
                          extra_user_prompt=None):
    """
    Same as prepare_bigfeat_from_llm() (same `provider` XOR
    `llm_response_path` rule applies), but also calls .fit() for you
    automatically using the resolved configuration, and stores the
    transformed training features as bf.X_transformed.

    Example
    -------
    bf_claude = run_bigfeat_with_llm(
        df, target_column="label", provider="claude"
    )
    X_train_new_features = bf_claude.X_transformed
    X_test_new_features = bf_claude.transform(X_test)
    print(bf_claude.response)
    """
    bf = prepare_bigfeat_from_llm(
        df, target_column, provider=provider, api_key=api_key, model=model,
        llm_response_path=llm_response_path, task_type=task_type,
        extra_user_prompt=extra_user_prompt)

    bf.X_transformed = bf.fit(bf.X, bf.y, **bf.fit_kwargs)

    return bf


# =============================================================================
# 8. READING THE RESULT: which columns are new, and what are they built from?
# =============================================================================

def describe_generated_features(bf):
    """
    After bf.fit(...) has run, this returns one human-readable string per
    NEWLY GENERATED feature, naming which original columns and which
    operators went into building it. Uses bf.tracking_ids, bf.tracking_ops,
    and bf.feature_names, all of which bf.fit() populates internally.

    Note: this lists which base columns and operators are involved in each
    new feature, not the exact nested formula/order they were combined in
    (reconstructing the full expression tree isn't exposed as a string by
    bigfeat_base.py). For most interpretability needs (e.g. "this new
    feature mixes column A and column B using multiply and square") this is
    enough; if you need the literal formula, bf.tracking_ids[i] /
    bf.tracking_ops[i] hold the raw data to build it yourself.

    Example
    -------
    X_train_bf = bf.fit(bf.X, bf.y, **bf.fit_kwargs)
    for line in describe_generated_features(bf):
        print(line)
    """
    descriptions = []
    for i in range(len(bf.tracking_ids)):
        col_indices = bf.tracking_ids[i]
        col_names = [bf.feature_names[idx] for idx in col_indices]
        op_names = []
        for op, _depth in bf.tracking_ops[i]:
            op_names.append(getattr(op, "__name__", str(op)))
        descriptions.append(
            f"new_feature_{i}: built from columns {col_names} "
            f"using operators {op_names}"
        )
    return descriptions


def split_original_vs_new(bf, X_transformed):
    """
    Splits a transformed matrix (bf.fit(...)'s return value, or
    bf.transform(...)'s return value) into (new_features, original_features),
    since BigFeat always places newly generated features first and the
    original input columns last.

    Example
    -------
    X_train_bf = bf.fit(bf.X, bf.y, **bf.fit_kwargs)
    new_only, original_only = split_original_vs_new(bf, X_train_bf)
    print("New features shape:", new_only.shape)
    """
    n_new = len(bf.tracking_ids)
    return X_transformed[:, :n_new], X_transformed[:, n_new:]
