import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fit_models import get_fit_model
from pipeline_helper import get_mc5_data, load_config
from tcpl_hit import get_nested_mc4
from tcpl_load_data import tcpl_load_data


def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power(np.linspace(start, stop, num=num), power)

# Load data initially or when id changes, and cache the result
@st.cache_data
def fetch_data(id):
    check_reset()
    dat = tcpl_load_data(lvl=3, fld='aeid', ids=id)
    grouped = dat.groupby(['aeid', 'spid'])
    mc4 = grouped.agg(concentration_unlogged=('logc', lambda x: list(10 ** x)), response=('resp', list)).reset_index()
    nested_mc4 = get_nested_mc4(get_mc5_data(id), parallelize=True, n_jobs=-1)
    return mc4, nested_mc4


def update(aeid):
    mc4, nested_mc4 = fetch_data(aeid)
    check_reset()
    spid_row = st.session_state.spid_row
    print(spid_row)
    fitparams = nested_mc4.iloc[spid_row]["params"]
    conc = mc4['concentration_unlogged'].iloc[spid_row]
    resp = np.array(mc4['response'].iloc[spid_row])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.log10(conc), y=resp, mode='markers', marker=dict(color="black", symbol="circle-open", size = 10), name="response"))
    fit_models = list(fitparams.keys())
    # add_verticla_lines(conc, fig)

    pars_dict = {}
    for m, model in enumerate(fit_models):
        params = fitparams[model]
        pars = list(params["pars"].values())
        pars_dict[model] = list(pars)
        pars = np.array(pars)
        modl = np.array(get_fit_model(model)(pars, np.array(conc)))
        x = powspace(np.min(conc), np.max(conc), 10, 1000)
        y = np.array(get_fit_model(model)(pars, x))
        color = px.colors.qualitative.Light24[m]
        fig.add_trace(go.Scatter(x=np.log10(conc), y=modl, legendgroup=model, marker=dict(color=color, symbol="x"), mode='markers', name=model, showlegend=False))
        fig.add_trace(go.Scatter(x=np.log10(x), y=y, legendgroup=model, marker=dict(color=color), mode='lines', name=model))

    height = 500
    fig.update_layout(height=int(height))
    return fig, pars_dict


def add_verticla_lines(conc, fig):
    uconc = np.unique(conc)
    for i in range(uconc.size):
        fig.add_vline(x=float(np.log10(uconc[i])), line_dash="dash", line_width=1, opacity=.5,)


def load_new_sample(aeid, direction):
    check_reset()
    _, nested_mc4 = fetch_data(aeid)
    if direction == "next":
        new_spid_row = st.session_state.spid_row + 1
    else:
        new_spid_row = st.session_state.spid_row - 1
    st.session_state.spid_row = new_spid_row % nested_mc4.shape[0]


def check_reset():
    if "spid_row" not in st.session_state:
        reset_spid_row()


def reset_spid_row():
    st.session_state["spid_row"] = 0


st.set_page_config(
    page_title="Viz pytcpl",
    page_icon="✅",
    # layout="wide",
)


def main():
    config = load_config()["pytcpl"]
    st.title("Curve-fit Visualization")
    aeid = int(st.number_input(label="Enter assay id (aeid)", value=config['aeid'], on_change=reset_spid_row))
    col1, col2 = st.columns(2)
    with col1:
        st.button("Previous sample", on_click=load_new_sample, args=(aeid, "previous",))
    with col2:
        st.button("Next sample", on_click=load_new_sample, args=(aeid, "next",))
    fig, pars_dict = update(aeid)
    st.plotly_chart(fig)
    st.json(pars_dict)


if __name__ == '__main__':
    main()
