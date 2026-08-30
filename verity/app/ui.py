"""Streamlit demo UI for Verity.

Run with:
    streamlit run verity/app/ui.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from verity.app.service import VerityDemoService
from verity.governance import DEMO_PRINCIPALS


@st.cache_resource
def _service() -> VerityDemoService:
    return VerityDemoService()


service = _service()

st.set_page_config(page_title="Verity", layout="wide")
st.title("Verity")
st.caption("Governed KPI intelligence-to-action demo")

scenario_options = {f"{s['id']} - {s['label']}": s["id"] for s in service.scenarios()}
left, right = st.columns([1, 1])
scenario_id = left.selectbox("Scenario", list(scenario_options), index=0)
principal_key = right.selectbox("User", ["analyst", "cfo", "west_manager", "east_manager"], index=0)
persona = right.selectbox("Persona view", ["analyst", "cfo", "ops"], index=0)

bundle = service.run_scenario(
    scenario_options[scenario_id],
    principal=DEMO_PRINCIPALS[principal_key],
    persona=persona,
)
payload = bundle.as_dict()

metrics = payload["assessment"]
cols = st.columns(5)
cols[0].metric("Movement", f"{metrics['change_pct']:+.1f}%")
cols[1].metric("Severity", metrics["severity"])
cols[2].metric("Confidence", f"{payload['evidence_pack']['confidence']:.0%}")
cols[3].metric("Route", payload["route"]["name"])
cols[4].metric("Cost saved", f"{payload['cost']['savings_pct']:.0f}%")

st.subheader("Investigation")
st.write(payload["narrative"]["summary"])

findings = pd.DataFrame(payload["attribution"]["findings"])
st.dataframe(findings, use_container_width=True, hide_index=True)

st.subheader("Evidence Pack")
evidence_rows = []
for item in payload["evidence_pack"]["evidence"]:
    evidence_rows.append(
        {
            "id": item["id"],
            "source": item["source"],
            "type": item["type"],
            "score": item["score"],
            "reliability": item["reliability"],
            "title": item["title"],
        }
    )
st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)

if payload["evidence_pack"]["contradictions"]:
    st.warning("Contradictory evidence detected; the system abstains from a root-cause claim.")
    st.json(payload["evidence_pack"]["contradictions"])

if payload["decision"]:
    st.subheader("Decision War Room")
    st.write(payload["decision"]["memo"])
    st.dataframe(pd.DataFrame(payload["decision"]["positions"]), use_container_width=True, hide_index=True)
    st.json(payload["decision"]["action_payload"])

tabs = st.tabs(["Lineage", "Evaluation", "Audit", "Risk Radar"])
with tabs[0]:
    graph = service.lineage(metrics["kpi"]).as_dict()
    st.dataframe(pd.DataFrame(graph["nodes"]), use_container_width=True, hide_index=True)
    st.dataframe(pd.DataFrame(graph["edges"]), use_container_width=True, hide_index=True)
    if graph["conflicts"]:
        st.warning("KPI definition conflict surfaced from the semantic contract.")
        st.json(graph["conflicts"])

with tabs[1]:
    report = service.evaluation()
    st.json(report.as_dict())
    st.dataframe(pd.DataFrame([row.__dict__ for row in report.scenarios]), use_container_width=True, hide_index=True)

with tabs[2]:
    if st.button("Exercise West manager denied East read"):
        service.exercise_denial()
    st.dataframe(pd.DataFrame(service.audit_view()), use_container_width=True, hide_index=True)

with tabs[3]:
    st.dataframe(pd.DataFrame(service.risk_radar()), use_container_width=True, hide_index=True)
