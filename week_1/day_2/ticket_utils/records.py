def group_by_team(records):
    groups = {}

    for record in records:
        team = record["team"]
        groups.setdefault(team, []).append(record)

    return groups