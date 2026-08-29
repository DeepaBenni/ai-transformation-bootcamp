import requests

url = "https://jsonplaceholder.typicode.com/todos"

limit = 5
skip = 0

all_data = []

while True:
    response = requests.get(
        url,
        params={
            "_limit": limit,
            "_start": skip
        },
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        break

    all_data.extend(data)

    skip += limit

    if len(all_data) >= 15:
        break

print("Total records:", len(all_data))

# 11. SERVICENOW API PARAMETERS
params = {
    "sysparm_limit": 10,
    "sysparm_fields": "number,priority,short_description"
}

print(params)