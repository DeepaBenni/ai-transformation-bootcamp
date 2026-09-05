"""Streamlit app: scores a new ticket at intake using the pipeline trained in
day_7.ipynb. Run with:  streamlit run app.py
"""
import datetime as dt
import pickle

import pandas as pd
import streamlit as st

from features import build_features, load_group_rates, FEATURES

model = pickle.load(open('model.pkl', 'rb'))
threshold = float(pickle.load(open('threshold.pkl', 'rb')))
group_rates = load_group_rates('group_rates.pkl')

prep = model.named_steps['prep']
options = dict(zip(prep.transformers_[0][2], prep.named_transformers_['oh'].categories_))

st.set_page_config(page_title='SLA Breach Risk', page_icon='\U0001F6A8')
st.title('SLA Breach Risk')
st.caption(
    f'Scores a ticket at intake. Alert threshold {threshold:.2f} '
    f'(tuned on validation: a missed breach costs 12.5x a false alarm).'
)

c1, c2 = st.columns(2)
with c1:
    d = st.date_input('Date opened', dt.date(2026, 7, 15))
    group = st.selectbox('Assignment group', options['assignment_group'])
    cat = st.selectbox('Category', options['category'])
    sub = st.selectbox('Subcategory', options['subcategory'])
    contact = st.selectbox('Contact type', options['contact_type'])
with c2:
    t = st.time_input('Time opened', dt.time(22, 30))
    priority = st.selectbox('Priority', ['P1', 'P2', 'P3', 'P4', 'Sev1', 'Sev2', 'Sev3', 'Sev4'])
    ci = st.selectbox('Configuration item', options['cmdb_ci'])
    dept = st.selectbox('Requesting department', options['requesting_department'])

short_desc = st.text_input('Short description', 'VPN not connecting for remote users')
desc = st.text_area('Description', 'User reports VPN issue. Observed error AUTH-401.')

if st.button('Predict', type='primary'):
    raw = pd.DataFrame([{
        'opened_at': dt.datetime.combine(d, t), 'priority': priority,
        'assignment_group': group, 'category': cat, 'subcategory': sub,
        'cmdb_ci': ci, 'contact_type': contact, 'requesting_department': dept,
        'short_description': short_desc, 'description': desc,
    }])
    row = build_features(raw, group_rates)[FEATURES]

    prob = model.predict_proba(row)[0, 1]
    st.metric('Breach probability', f'{prob:.1%}', f'{prob - threshold:+.1%} vs threshold')
    if prob >= threshold:
        st.error(f'ESCALATE — {prob:.1%} breach risk (threshold {threshold:.0%})')
    else:
        st.success(f'Low risk — {prob:.1%} (threshold {threshold:.0%})')

    with st.expander('Features sent to the model'):
        st.dataframe(row.T.rename(columns={0: 'value'}))
