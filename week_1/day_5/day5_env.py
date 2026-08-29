import os
from dotenv import load_dotenv

load_dotenv()

app_name = os.getenv("APP_NAME")
environment = os.getenv("ENVIRONMENT")

print("App:", app_name)
print("Environment:", environment)