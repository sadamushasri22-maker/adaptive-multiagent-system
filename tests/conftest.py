import os

# Set dummy key for test collection so unit tests with mock LLM calls can run
if not os.environ.get("API_KEY") or os.environ.get("API_KEY", "").startswith("your_"):
    os.environ["API_KEY"] = "sk-dummy-test-key"
