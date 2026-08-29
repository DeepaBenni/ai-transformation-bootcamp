import os
from dotenv import load_dotenv

load_dotenv()

app_name = os.getenv("APP_NAME")
environment = os.getenv("ENVIRONMENT")

if app_name is None:
    print("APP_NAME is missing")
else:
    print("APP_NAME is configured")

if environment is None:
    print("ENVIRONMENT is missing")
else:
    print("ENVIRONMENT is configured")