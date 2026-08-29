import os
from dotenv import load_dotenv

load_dotenv()

app_name = os.getenv("APP_NAME", "Default App")
environment = os.getenv("ENVIRONMENT", "development")
debug_value = os.getenv("DEBUG", "false")

debug = debug_value.lower() == "true"

print("Application:", app_name)
print("Environment:", environment)
print("Debug mode:", debug)