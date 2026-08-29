def parse_duration(duration):
    parts = duration.split(" ")
    hours = parts[0][:-1]
    minutes = parts[1][:-1]
    return int(hours) * 60 + int(minutes)