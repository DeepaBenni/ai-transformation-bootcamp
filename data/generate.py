import pandas as pd
import numpy as np
from datetime import datetime, timedelta
n = 50000

incident_ids = [f"INC{i:05d}" for i in range(1, n + 1)]

print(incident_ids[:5])
print(len(incident_ids))
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)

dates = pd.date_range(start_date, end_date, periods=n)

print(dates[:5])
print(dates[-1])
priorities = np.random.choice(["P1", "P2", "P3"], size=n)

print(priorities[:10])
print(len(priorities))
categories = np.random.choice(
    ["Network", "Hardware", "Software", "Database"],
    size=n
)

print(categories[:10])
print(len(categories))
groups = {
    "Network": "Network Team",
    "Hardware": "Hardware Team",
    "Software": "App Team",
    "Database": "DB Team"
}

assignment_groups = [groups[category] for category in categories]

print(assignment_groups[:10])
print(len(assignment_groups))
resolution_minutes = np.random.randint(30, 1000, size=n)

print(resolution_minutes[:10])
print(len(resolution_minutes))
reassignment_count = np.random.randint(0, 4, size=n)

print(reassignment_count[:10])
print(len(reassignment_count))
sla_targets = {"P1": 240, "P2": 480, "P3": 1440}

sla_target = [sla_targets[priority] for priority in priorities]

print(sla_target[:10])
print(len(sla_target))
sla_breached = [
    resolution > target
    for resolution, target in zip(resolution_minutes, sla_target)
]

print(sla_breached[:10])
print(len(sla_breached))

df = pd.DataFrame({
    "incident_id": incident_ids,
    "opened_at": dates,
    "priority": priorities,
    "category": categories,
    "assignment_group": assignment_groups,
    "resolution_minutes": resolution_minutes,
    "reassignment_count": reassignment_count,
    "sla_target": sla_target,
    "sla_breached": sla_breached
})

print(df.columns)
print(df.shape)