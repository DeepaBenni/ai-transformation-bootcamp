"""Score a handful of synthetic (never-seen) tickets through the Day 7 pipeline to
sanity-check behavior across the risk spectrum, including two out-of-vocabulary
edge cases the OneHotEncoder / group-rate lookup must handle gracefully.
"""
import datetime as dt
import pickle

import pandas as pd

from features import build_features, load_group_rates, FEATURES

model = pickle.load(open('model.pkl', 'rb'))
threshold = float(pickle.load(open('threshold.pkl', 'rb')))
group_rates = load_group_rates('group_rates.pkl')

tickets = [
    dict(
        label='Low risk: safest group, business hours, low priority',
        opened_at=dt.datetime(2026, 8, 11, 10, 15),  # Tue
        priority='P4', assignment_group='Network Operations', category='Network',
        subcategory='LAN', cmdb_ci='Core-Switch', contact_type='Self-service',
        requesting_department='IT', short_description='Wifi slow in building 3',
        description='User reports intermittent wifi drops in building 3, floor 2. Observed error TIMEOUT-108.',
    ),
    dict(
        label='High risk: riskiest group, weekend overnight, top priority',
        opened_at=dt.datetime(2026, 8, 15, 2, 40),  # Sat
        priority='Sev1', assignment_group='Mainframe Operations', category='Data Warehouse',
        subcategory='Batch', cmdb_ci='Generic-App', contact_type='Monitoring',
        requesting_department='Finance', short_description='Batch job failed overnight',
        description='Nightly settlement batch job failed on the mainframe. Observed error ABEND-0C7.',
    ),
    dict(
        label='Mixed: risky group, but business hours',
        opened_at=dt.datetime(2026, 8, 12, 11, 0),  # Wed
        priority='P3', assignment_group='Mainframe Operations', category='Data Warehouse',
        subcategory='Batch', cmdb_ci='Generic-App', contact_type='Phone',
        requesting_department='Finance', short_description='Report generation delayed',
        description='Scheduled finance report generation running slower than usual. Observed error QUEUE-DELAY.',
    ),
    dict(
        label='Mixed: safe group, but weekend overnight',
        opened_at=dt.datetime(2026, 8, 16, 3, 20),  # Sun
        priority='P2', assignment_group='Network Operations', category='Network',
        subcategory='WAN', cmdb_ci='Firewall-01', contact_type='Monitoring',
        requesting_department='Operations', short_description='WAN link flapping',
        description='Primary WAN link between sites is flapping intermittently. Observed error LINK-DOWN.',
    ),
    dict(
        label='Boundary: exactly 08:00 Monday (business-hours edge)',
        opened_at=dt.datetime(2026, 8, 10, 8, 0),  # Mon
        priority='P4', assignment_group='End User Computing', category='Hardware',
        subcategory='Laptop', cmdb_ci='Business-Service', contact_type='Walk-up',
        requesting_department='Sales', short_description='Laptop wont boot',
        description='Laptop fails to power on after firmware update. Observed error BOOT-FAIL.',
    ),
    dict(
        label='Out-of-vocabulary: brand-new group/category/CI never seen in training',
        opened_at=dt.datetime(2026, 8, 13, 14, 0),  # Thu
        priority='P2', assignment_group='AI Platform Ops', category='Quantum Computing',
        subcategory='Qubit Calibration', cmdb_ci='Quantum-Node-07', contact_type='Chat',
        requesting_department='R&D', short_description='Qubit calibration drifting',
        description='Calibration drift detected on quantum node during overnight run. Observed error DRIFT-042.',
    ),
    dict(
        label='Month-end + high priority + risky group',
        opened_at=dt.datetime(2026, 8, 30, 15, 0),  # Sun, month-end window
        priority='Sev1', assignment_group='Platform Engineering', category='Data Warehouse',
        subcategory='ETL', cmdb_ci='Generic-App', contact_type='Monitoring',
        requesting_department='Finance', short_description='Month-end ETL pipeline stalled',
        description='Month-end close ETL pipeline stalled with no progress for 40 minutes. Observed error STALL-01.',
    ),
]

raw = pd.DataFrame(tickets)
feat = build_features(raw, group_rates)
proba = model.predict_proba(feat[FEATURES])[:, 1]

report = pd.DataFrame({
    'scenario': raw['label'],
    'group': raw['assignment_group'],
    'priority': raw['priority'],
    'business_hours': feat['is_business_hours'],
    'group_hist_rate': feat['group_breach_rate_encoded'].round(4),
    'breach_prob': proba.round(4),
    'decision': ['ESCALATE' if p >= threshold else 'low risk' for p in proba],
})
pd.set_option('display.max_colwidth', 55)
print(f'threshold = {threshold}\n')
print(report.to_string(index=False))
