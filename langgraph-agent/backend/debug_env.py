from dotenv import load_dotenv
import os

# Mimic main.py loading
load_dotenv(override=True)

pk = os.getenv("LANGFUSE_PUBLIC_KEY")
sk = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_HOST")

print("--- DEBUG ENV ---")
print(f"Public Key: '{pk}'")
print(f"Secret Key: '{sk}'")
print(f"Host:       '{host}'")
print("-----------------")

if pk == "pk-lf-dcdaebf9-1716-4574-ad6e-362fd6576259":
    print("SUCCESS: Public Key matches expected value.")
else:
    print("FAILURE: Public Key does NOT match.")

if sk == "sk-lf-0a177413-c9e2-4fe0-8e9e-48dee1b550e4":
    print("SUCCESS: Secret Key matches expected value.")
else:
    print("FAILURE: Secret Key does NOT match.")
