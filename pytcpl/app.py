import numpy as np
import plotly.graph_objects as go
import streamlit as st  # data web app development

from fit_models import get_fit_model
from pipeline import get_mc5_data
from tcpl_hit import get_nested_mc4
from tcpl_load_data import tcpl_load_data


# Load data initially or when id changes, and cache the result
@st.cache_data
def fetch_data(id):
    global spid_row, mc4, nested_mc4
    spid_row = 0
    dat = tcpl_load_data(lvl=3, fld='aeid', val=id)
    grouped = dat.groupby(['aeid', 'spid'])
    mc4 = grouped.agg(conc=('logc', lambda x: list(10 ** x))).reset_index()
    nested_mc4 = get_nested_mc4(get_mc5_data(id), True)
    return mc4, nested_mc4


def update(aeid):
    mc4, nested_mc4 = fetch_data(aeid)
    spid_row = st.session_state.spid_row
    fitparams = nested_mc4.iloc[spid_row]["params"]
    conc = np.array(mc4["conc"].iloc[spid_row])
    fig = go.Figure()
    fit_models = list(fitparams.keys())
    for model in fit_models:
        params = fitparams[model]
        pars = np.array(list(params["pars"].values()))
        modl = np.array(get_fit_model(model)(pars, conc))
        x = np.linspace(np.min(conc), np.max(conc), 100)
        y = np.array(get_fit_model(model)(pars, x))
        fig.add_trace(go.Scatter(x=conc, y=modl, mode='markers', name=model, showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=model))
    return fig


def load_next_row(aeid):
    if "spid_row" not in st.session_state:
        reset_spid_row()

    _, nested_mc4 = fetch_data(aeid)
    st.session_state.spid_row = (st.session_state.spid_row + 1) % nested_mc4.shape[0]

def reset_spid_row():
    st.session_state["spid_row"] = 0


st.set_page_config(
    page_title="Viz pytcpl",
    page_icon="✅",
    # layout="wide",
)


def main():
    st.title("Curve-fit Visualization")
    aeid = int(st.number_input(label="Enter assay id (aeid)", value=5, on_change=reset_spid_row))
    st.button("Next sample", on_click=load_next_row(aeid))
    st.plotly_chart(update(aeid))


if __name__ == '__main__':
    main()
