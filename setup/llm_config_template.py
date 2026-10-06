"""
llm_config_template.py
-----------------------
Copy this file to `llm_config.py` and fill in your own API key(s).
`llm_runner.py` imports from `llm_config.py` if it exists, and otherwise
you can pass api_key / model explicitly to run_bigfeat_with_llm() /
prepare_bigfeat_from_llm(), which always takes priority over anything in
this file.

You do NOT need to fill in all four providers below. Fill in only the ONE
you actually intend to use (for example just "claude", or just "openai")
and pass that same name as provider="..." when calling the functions in
llm_runner.py. The other three entries can be left as placeholders, or
deleted entirely - they are only ever read if you actually pass that
provider's name.

The four built-in providers are "openai" (ChatGPT), "gemini", "deepseek",
and "claude". You are not limited to these four: add a new key here with
your own "api_key" and "model", and then add a matching "call" function
for it in PROVIDERS inside llm_runner.py (see the comment there for the
exact shape a new provider needs).

If you plan to use provider="gemini", you also need to install Google's
current official SDK (the older `google-generativeai` package reached
end-of-life on 2025-11-30 and no longer works for new integrations):
    pip install google-genai
"""

LLM_CONFIG = {
    "openai": {
        "api_key": "YOUR_OPENAI_API_KEY_HERE",
        "model": "gpt-4o",  # check https://platform.openai.com/docs/models for the current best model name
    },
    "gemini": {
        "api_key": "YOUR_GEMINI_API_KEY_HERE",
        # Check https://ai.google.dev/gemini-api/docs/models for whatever
        # the current model name is - these get renamed/replaced often.
        "model": "YOUR_GEMINI_MODEL_NAME_HERE",
    },
    "deepseek": {
        "api_key": "YOUR_DEEPSEEK_API_KEY_HERE",
        "model": "deepseek-chat",  # check https://api-docs.deepseek.com/quick_start/pricing for the current name
    },
    "claude": {
        "api_key": "YOUR_ANTHROPIC_API_KEY_HERE",
        "model": "claude-sonnet-4-5",  # check https://docs.claude.com/en/docs/about-claude/models for the current name
    },

    # Example of how to add a fifth provider of your own, as long as you
    # also add a matching entry to PROVIDERS in llm_runner.py:
    #
    # "my_custom_llm": {
    #     "api_key": "YOUR_KEY_HERE",
    #     "model": "my-model-name",
    # },
}
