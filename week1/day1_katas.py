duration = "2h 30m"

parts = duration.split(" ")
"2h 30m".split(" ")

print(parts)
# KATA 1
# Usage of split function

duration = "2h 30m"

parts = duration.split(" ")

print(parts)

hours = parts[0][:-1]
minutes = parts[1][:-1]

print(hours)
print(minutes)

total = int(hours) * 60 + int(minutes)

print(total)
# Kata 2 - set

numbers = [3, 1, 3, 2, 1]

unique = set(numbers)

print(unique)
# KATA 3
# Invert dictionary keys and values

data = {"a": 1, "b": 2}

result = {}

for key, value in data.items():
    result[value] = key

print(result)
# Kata 4 - nested dictionary

data = {"user": {"name": "Deepa", "city": "Belagavi"}}

print(data["user"]["name"])
print(data["user"]["city"])
# Kata 5 - grouping

records = [
    {"name": "A", "team": "X"},
    {"name": "B", "team": "Y"},
    {"name": "C", "team": "X"}
]

groups = {}

for record in records:
    team = record["team"]
    groups.setdefault(team, []).append(record)

print(groups)
# Kata 6 - try/except

attempts = 0

for i in range(3):
    try:
        attempts += 1
        print("Attempt", attempts)
        break
    except ValueError:
        print("Failed")
# Kata 7 - dictionary counting

text = "python is easy python"

words = text.split()

counts = {}

for word in words:
    counts[word] = counts.get(word, 0) + 1

print(counts)
# Kata 8 - count items

numbers = [1, 2, 2, 3, 2, 1]

counts = {}

for number in numbers:
    counts[number] = counts.get(number, 0) + 1

print(counts)
print(max(counts, key=counts.get))
# Kata 8 - count items

numbers = [1, 2, 2, 3, 2, 1]

counts = {}

for number in numbers:
    counts[number] = counts.get(number, 0) + 1

print(counts)
print(max(counts, key=counts.get))
# Kata 10 - set comprehension

tickets = [
    {"priority": "P1"},
    {"priority": "P2"},
    {"priority": "P1"},
    {"priority": "P3"}
]

priorities = {ticket["priority"] for ticket in tickets}

print(priorities)
# Kata 11 - filtering

tickets = [
    {"priority": 1},
    {"priority": 3},
    {"priority": 2}
]

high = []

for ticket in tickets:
    if ticket["priority"] >= 2:
        high.append(ticket)

print(high)
# Kata 12 - strings

text = "  SERVER   IS   DOWN  "

clean = " ".join(text.strip().lower().split())

print(clean)
# Kata 13 - find text

text = "Application failed with ERROR-500"

start = text.find("ERROR-")

print(text[start:start + 9])
# Kata 14 - average

numbers = [10, 20, 30]

average = sum(numbers) / len(numbers)

print(average)
# Kata 15 - exception handling

value = "42"

try:
    number = int(value)
except ValueError:
    number = None

print(number)
# Kata 16 - longest word

text = "Python makes automation easy"

words = text.split()

longest = max(words, key=len)

print(longest)
# Kata 17 - count vowels

text = "Python is easy"

vowels = "aeiou"

count = 0

for char in text.lower():
    if char in vowels:
        count += 1

print(count)
# Kata 18 - sorting

numbers = [10, 50, 20, 40, 30]

numbers.sort(reverse=True)

print(numbers[:3])
# Kata 19 - list comprehension

numbers = [1, 2, 3, 4, 5, 6]

even = [number for number in numbers if number % 2 == 0]

print(even)
# Kata 20 - reverse string

text = "Python"

reversed_text = text[::-1]

print(reversed_text)        
