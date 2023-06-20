import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from fit_models import get_fit_model
from pipeline_helper import get_mc5_data, load_config
from query_db import tcpl_query
from tcpl_hit import get_nested_mc4
from tcpl_load_data import tcpl_load_data
from acy import acy

# Run command `streamlit run pytcpl/app.py`

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
    mc4 = mc4.head(config["test"]) if  config["test"] else mc4
    nested_mc4 = get_nested_mc4(get_mc5_data(id), parallelize=True, n_jobs=-1)

    # hitcall data
    df = tcpl_load_data(lvl=6, fld='aeid', ids=id)
    d = {str(m5id): group for m5id, group in df.groupby("m4id")}

    return mc4, nested_mc4, d


def update():
    mc4, nested_mc4, hit_data = fetch_data(st.session_state.aeid)  # needs input for unique caching
    check_reset()

    spid = set_spid(mc4, nested_mc4, st.session_state.trigger)
    if spid is None:
        return

    conc, d, fitparams, resp = get_row_data(hit_data, mc4, nested_mc4)

    fig = go.Figure()
   
    add_title(fig, d)

    fig.add_trace(
        go.Scatter(x=np.log10(conc), y=resp, mode='markers', marker=dict(color="black", symbol="x-thin-open", size=10),
                   name="response"))
    
    cutoff = round(d['coff'], 2)
    fig.add_hline(y=cutoff, line_dash="dash", opacity=.5, annotation_position="top left",
                  annotation_text=f"efficacy cutoff: {cutoff}")

    return fig, add_curves(conc, fig, fitparams, d)


def add_title(fig, d):
    try:
        casn, chnm, dsstox_substance_id = get_chem_info(st.session_state.spid)
    except:
        return
    fig.update_layout(height=600)
    # fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    qstring = f"SELECT normalized_data_type FROM assay_component_endpoint WHERE aeid = {st.session_state.aeid};"
    normalized_data_type = tcpl_query(qstring).iloc[0]["normalized_data_type"]
    fig.update_layout(
        title=f"SPID: {st.session_state.spid}       CASN: {casn}       DSSTOX_SID: {dsstox_substance_id}<br>Chemical: {chnm}<br>Best fit model: {d['modl']}<br>Hit-call: {round(d['hitcall'], 7)}",
        margin=dict(t=150),
        xaxis_title="Concentration (uM)",
        yaxis_title=str(normalized_data_type),
    )


def add_curves(conc, fig, fitparams, d):
    fit_models = list(fitparams.keys())
    pars_dict = {}
    for m, model in enumerate(fit_models):
        params = fitparams[model]
        pars = list(params["pars"].values())
        pars_dict[model] = list(pars)
        pars = np.array(pars)
        modl = np.array(get_fit_model(model)(pars, np.array(conc)))
        x = powspace(np.min(conc) / 3, np.max(conc) * 1.5, 100, 200)
        y = np.array(get_fit_model(model)(pars, x))
        color = px.colors.qualitative.Light24[m]
        # fig.add_trace(go.Scatter(x=np.log10(conc), y=modl, legendgroup=model, marker=dict(color=color, symbol="x"),
        #                          mode='markers', name=model, showlegend=False))
        fig.add_trace(
            go.Scatter(x=np.log10(x), y=y, opacity=.7, legendgroup=model, marker=dict(color=color), mode='lines', name=model))
        if model == d['modl']:
            
            top = round(d['top'], 2)
            
            fig.add_hline(y=top, line_dash="dash", opacity=.5, annotation_position="top left",
                          annotation_text=f"curve-top best fit: {top}")
            
            fig.add_vline(x=np.log10(acy(top, params["pars"], model)), line_dash="dash", opacity=.5, annotation_position="bottom",
                          annotation_text=f"actop")
            
            fig.add_vline(x=round(np.log10(d['ac10']), 2), line_dash="dash", opacity=.5, annotation_position="bottom",
                          annotation_text=f"ac10")
            
            fig.add_vline(x=round(np.log10(d['ac50']), 2), line_dash="dash", opacity=.5, annotation_position="bottom",
                          annotation_text=f"ac50")
            
            # fig.add_vline(x=round(np.log10(d['ac95']), 2), line_dash="dash", opacity=.5, annotation_position="bottom",
            #               annotation_text=f"ac95")
    return pars_dict


def get_row_data(hit_data, mc4, nested_mc4):
    spid_row = st.session_state.spid_row
    df = hit_data[str(nested_mc4.iloc[spid_row]["m4id"])]
    d = {}
    d["modl"] = df["modl"].iloc[0]
    d["coff"] = df["coff"].iloc[0]
    d.update(pd.Series(df.hit_val.values, index=df.hit_param).to_dict())
    fitparams = nested_mc4.iloc[spid_row]["params"]
    conc = mc4['concentration_unlogged'].iloc[spid_row]
    resp = np.array(mc4['response'].iloc[spid_row])
    return conc, d, fitparams, resp


def get_chem_info(spid):
    chid = tcpl_query(query=f"SELECT chid FROM sample WHERE spid = '{str(spid)}';").iloc[0]['chid']
    chem = tcpl_query(query=f"SELECT * FROM chemical WHERE chid = {str(chid)};").iloc[0]
    casn = chem['casn']
    chnm = chem['chnm']
    dsstox_substance_id = chem['dsstox_substance_id']
    return casn, chnm, dsstox_substance_id



def set_spid(mc4, nested_mc4, trigger):
    spid = st.session_state.spid
    if spid in mc4['spid'].values and trigger == "spid":
        st.session_state.spid_row = mc4[mc4['spid'] == spid].index[0]
    elif trigger == "new_sample":
        dir = st.session_state.direction
        if dir == "next":
            new_spid_row = st.session_state.spid_row + 1
        elif dir == "previous":
            new_spid_row = st.session_state.spid_row - 1
        else:
            new_spid_row = st.session_state.spid_row

        st.session_state.spid_row = new_spid_row % nested_mc4.shape[0]
        spid = mc4.iloc[st.session_state.spid_row]["spid"]
        st.session_state.spid = spid
    else:
        return None
    return spid


def load_new_sample(direction):
    st.session_state.direction = direction
    st.session_state.trigger = "new_sample"


def check_reset():
    if "spid_row" not in st.session_state:
        print("reset to start sample")
        reset_spid_row()
    if "direction" not in st.session_state:
        st.session_state.direction = "stay"
    if "trigger" not in st.session_state:
        st.session_state.trigger = "new_sample"
    if "spid" not in st.session_state:
        st.session_state.spid = ""


def reset_spid_row():
    st.session_state["spid_row"] = 0


def filter_spid():
   st.session_state.trigger = "spid"
    
    
st.set_page_config(page_title="Viz pytcpl", # page_icon="✅", layout="wide",
)

config = load_config()["pytcpl"]
st.title("Curve-fit Visualization")
st.session_state.aeid = int(st.number_input(label="Enter assay id (aeid)", value=config['aeid'], on_change=reset_spid_row))
st.session_state.spid = st.text_input(label="Filter sample id (spid)", on_change=filter_spid)
col1, col2  = st.columns(2)
with col1:
    st.button("Previous sample", on_click=load_new_sample, args=("previous",))
with col2:
    st.button("Next sample", on_click=load_new_sample, args=("next",))

try:
    fig, pars_dict = update()
    st.plotly_chart(fig)
    st.json(pars_dict)
except Exception as e:
    print(e)
    st.write("No data found")


