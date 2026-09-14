"""Shared feature logic for the SLA-breach model — imported by both the training
notebook (day_7.ipynb) and the serving app (app.py) so the two can never drift apart.
"""
import pickle

import pandas as pd

JUNK = ['issue', 'urgent', 'problem', 'pls fix',
        'help!!!', 'urgent pls', 'not working', "doesn't work"]

CAT = ['assignment_group', 'category', 'subcategory', 'cmdb_ci', 'contact_type',
       'requesting_department', 'client_scheme', 'day_of_week']
NUM = ['hour', 'is_business_hours', 'is_weekend', 'is_month_end', 'month',
       'priority_level', 'desc_length', 'short_desc_words', 'is_junk',
       'group_breach_rate_encoded']
FEATURES = CAT + NUM


def load_group_rates(path='group_rates.pkl'):
    """The frozen train-only assignment-group breach-rate map from Section 6."""
    return pickle.load(open(path, 'rb'))


def build_features(df, group_rates):
    """Add every derived column the model expects. Takes a raw ticket frame plus
    the frozen group_rates dict from load_group_rates()."""
    df = df.copy()
    o = pd.to_datetime(df['opened_at'])

    df['hour'] = o.dt.hour
    df['is_business_hours'] = ((o.dt.weekday < 5) & o.dt.hour.between(8, 17)).astype(int)
    df['is_weekend'] = (o.dt.weekday >= 5).astype(int)
    df['is_month_end'] = (o.dt.day >= o.dt.days_in_month - 2).astype(int)
    df['month'] = o.dt.month
    df['day_of_week'] = o.dt.dayofweek.astype(str)
    df['priority_level'] = df['priority'].str[-1].astype(int)
    df['client_scheme'] = df['priority'].str[0]
    df['desc_length'] = df['description'].str.len()
    df['short_desc_words'] = df['short_description'].str.split().str.len()
    df['is_junk'] = df['short_description'].str.strip().str.lower().isin(JUNK).astype(int)

    stats, global_rate = group_rates['stats'], group_rates['global_rate']
    df['group_breach_rate_encoded'] = df['assignment_group'].map(stats).fillna(global_rate)

    return df
