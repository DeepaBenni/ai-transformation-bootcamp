import os
from dotenv import load_dotenv

load_dotenv()

app_name = os.getenv("APP_NAME", "Default App")
environment = os.getenv("ENVIRONMENT", "development")
debug = os.getenv("DEBUG", "false")

print("App:", app_name)
print("Environment:", environment)
print("Debug:", debug)