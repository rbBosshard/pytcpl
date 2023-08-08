import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fit_models import get_model, powspace
from pipeline_helper import load_config, get_assay_info, print_
from query_db import query_db

# Run command `streamlit run pytcpl/app.py`
# Ensure: test = 0 in config.yaml
# Suppress the specific warning
pd.options.mode.chained_assignment = None  # default='warn'


# Load data initially or when id changes, and cache the result
@st.cache_data
def fetch_data(aeid):  # aeid parameter used to handle correct caching
    print_(f"Fetch data from DB with assay ID {aeid}..")
    qstring = f"SELECT * FROM output WHERE aeid = {st.session_state.aeid};"
    df = query_db(query=qstring)
    print_(f"{df.shape[0]} series loaded")
    qstring = f"SELECT * FROM cutoffs WHERE aeid = {st.session_state.aeid};"
    cutoff = query_db(query=qstring)['cutoff'].iloc[0]
    return df, cutoff


def get_series(df):
    st.session_state.length = df.shape[0]
    series = df.iloc[st.session_state.id]
    series['conc'] = json.loads(series['conc'])
    series['resp'] = json.loads(series['resp'])
    series['fit_params'] = json.loads(series['fit_params'])
    return series.to_dict()


def get_chem_info(spid):
    try:
        chid = query_db(query=f"SELECT chid FROM sample WHERE spid = '{str(spid)}';").iloc[0]['chid']
    except:
        try:
            chid = query_db(query=f"SELECT chid FROM chemical WHERE chnm = '{str(spid)}';").iloc[0]['chid']
        except:
            try:
                chid = query_db(query=f"SELECT chid FROM chemical WHERE chnm LIKE '%{str(spid)}%';").iloc[0]['chid']
            except Exception as e:
                print(f"Error on spid {spid}: {e}")
                return None, None, None

    chem = query_db(query=f"SELECT * FROM chemical WHERE chid = {str(chid)};").iloc[0]
    casn = chem['casn']
    chnm = chem['chnm']
    dsstox_substance_id = chem['dsstox_substance_id']
    return casn, chnm, dsstox_substance_id


def init_figure(series, cutoff):
    fig = go.Figure()
    casn, chnm, dsstox_substance_id = get_chem_info(series['spid'])
    fig.update_layout(height=600, width=1400)
    # fig.update_layout(hovermode="x unified")  # uncomment to enable unified hover
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    assay_infos = get_assay_info(st.session_state.aeid)
    normalized_data_type = assay_infos["normalized_data_type"]
    assay_component_endpoint_name = assay_infos["assay_component_endpoint_name"]
    assay_component_endpoint_desc = assay_infos["assay_component_endpoint_desc"]

    with st.expander("Details"):
        st.write(f"spid: {series['spid']}")
        st.write(f"Chemical: {chnm if chnm else 'N/A'}")
        link = f"[{dsstox_substance_id}](https://comptox.epa.gov/dashboard/chemical/details/{dsstox_substance_id})" if dsstox_substance_id else "N/A"
        st.write(f"DSSTOX Substance ID: {link}")
        st.write(f"CASN: {casn if casn else 'N/A'}")
        st.write(f"Assay Endpoint: {assay_component_endpoint_name}")
        st.write(f"{assay_component_endpoint_desc}")

    fig.update_layout(title=(f"Assay Endpoint: <i>{assay_component_endpoint_name}</i>"
               f"<br>Chemical: <i>{chnm if chnm else 'N/A'}</i><br>"
               f"Best Model Fit: <i>{series['best_aic_model']}</i>, hitcall: <i>{round(series['hitcall'], 7)}</i>"),
                margin=dict(t=150), xaxis_title="log10(Concentration) μM", yaxis_title=str(normalized_data_type))

    fig.add_hline(y=cutoff, line_color="LightSkyBlue")
    fig.add_hrect(y0=-cutoff, y1=cutoff, fillcolor='LightSkyBlue', opacity=0.4, layer='below', line=dict(width=0), 
                  annotation_text="efficacy cutoff", annotation_position="top left")
    fig.update_layout(legend=dict(groupclick="toggleitem"))

    return fig


def add_curves(series, fig):
    conc, resp, fit_params = np.array(series['conc']), series['resp'], series['fit_params']
    
    fig.add_trace(go.Scatter(x=np.log10(conc), y=resp, mode='markers', legendgroup="Response", legendgrouptitle_text="Repsonse", 
                   marker=dict(color="black", symbol="x-thin-open", size=10), name="response", showlegend=False))
    
    if fit_params is None or series['hitcall'] <= 0:
        return
    
    ac50 = series["ac50"]
    potencies = ["acc", "ac50", "actop"]
    efficacies = ["top"]
    iterator = fit_params.items()
    pars_dict = {}
    for index, (fit_model, fit_params) in enumerate(iterator):
        params = fit_params["pars"]
        pars_dict[fit_model] = params
        er = params.pop('er')
        aic = round(fit_params["aic"], 2)
        min_val, max_val = np.min(conc), np.max(conc)
        min_val = min_val if ac50 is None else min(min_val, ac50)
        x = powspace(min_val, max_val, 10, 200)
        y = np.array(get_model(fit_model)(x, **params))
        color = px.colors.qualitative.Bold[index]
        best = fit_model == series['best_aic_model']
        best_fit = f"(BEST FIT)" if best else ""
        dash = 'solid' if best else "dash"
        name = f"{fit_model} {best_fit} {aic}"
        width = 3 if best else 2
        legendgroup = fit_model if best else None
        opacity = 1 if best else .8

        fig.add_trace(go.Scatter(x=np.log10(x), y=y, opacity=opacity, legendgroup=legendgroup, marker=dict(color=color), mode='lines',
                                    name=name, line=dict(width=width, dash=dash)))

        # Todo: Makes it slow!!
        for p in potencies:
            if p in series:
                fig.add_vline(x=np.log10(series[p]), line_color=color, line_width=2, annotation_position="bottom left", annotation_text=p,
                            layer="below", opacity=0.5)

        for eff in efficacies:
            if eff in series:
                fig.add_hline(y=series[eff], line_color=color, line_width=2, annotation_position="bottom left", annotation_text=eff,
                      layer="below", opacity=0.5)

    return pars_dict



def check_reset():
    if "id" not in st.session_state:
        reset_id()
    if "aeid" not in st.session_state:
        st.session_state.aeid = 0
    if "spid" not in st.session_state:
        st.session_state.spid = ""
    if "sort_by" not in st.session_state:
        st.session_state.sort_by = "hitcall"
    if "asc" not in st.session_state:
        st.session_state.asc = False
    if "length" not in st.session_state:
        st.session_state.length = 0
    if "trigger" not in st.session_state:
        st.session_state.trigger = ""


def reset_id():
    st.session_state.id = 0

def next_id():
    st.session_state.trigger = "next"
    length = st.session_state.length
    if length:
        st.session_state.id = (st.session_state.id + 1) % length

def prev_id():
    st.session_state.trigger = "prev"
    length = st.session_state.length
    if length:
        st.session_state.id = (st.session_state.id - 1) % length

def trigger(trigger):
    st.session_state.trigger = trigger

def filter_spid():
    st.session_state.trigger = "spid"
    df, cutoff = fetch_data(st.session_state.aeid)  # requires id parameter for unique caching
    res = df[df['spid'] == st.session_state.spid]
    if res.empty:
        print(f"Input string '{st.session_state.spid}' not found")
    else:
        st.session_state.id = res.index[0]

def update():
    if "df" not in st.session_state:
        df, cutoff = fetch_data(st.session_state.aeid)  # requires id parameter for unique caching
        st.session_state.df = df
        st.session_state.cutoff = cutoff
    else:
        df = st.session_state.df
        cutoff = st.session_state.cutoff
    if st.session_state.trigger == "hitcall_slider":
        interval = st.session_state.hitcall_slider
        df = df.query(f'{interval[0]} <= hitcall <= {interval[1]}').reset_index(drop=True)
        st.session_state.df = df
        reset_id()
    st.session_state.length = len(df)
    st.write(f"Number of series: {st.session_state.length}")
    if st.session_state.length == 0:
        st.session_state.id = 0
        return None, None
        
    st.session_state.length = len(df)
    if st.session_state.trigger == "sort_by" or st.session_state.trigger == "hitcall_slider":
        df = df.sort_values(by=st.session_state.sort_by, ascending=st.session_state.asc).reset_index(drop=True)
        reset_id()
    series = get_series(df)
    fig = init_figure(series, cutoff)
    pars_dict = add_curves(series, fig)
    return fig, pars_dict


config, _ = load_config()
check_reset()
st.set_page_config(page_title="Curve surfer", page_icon="☣️", layout="wide")
st.title("Curve surfer")
st.session_state.aeid = int(st.number_input(label="Enter assay id (aeid)", value=int(config['aeid'])))
st.session_state.spid = st.text_input(label="Filter sample id (spid)", on_change=filter_spid)
if st.button("Process"):
    filter_spid()
st.session_state.hitcall_slider = st.slider('Select the range of hitcalls', 0.0, 1.0, (0.0, 1.0), on_change=trigger, args=("hitcall_slider",))
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button("Previous", on_click=prev_id)
with col2:
    st.button("Next", on_click=next_id)
with col3:
    st.session_state.sort_by = st.selectbox("Sort By", ("hitcall", "ac50", "actop"), on_change=trigger, args=("sort_by",))
with col4:
    st.session_state.asc = st.selectbox("Ascending", (False, True))

try:
    fig, pars_dict = update()
    if fig is None:
        raise Exception("No data found")
    st.plotly_chart(fig)
    st.json(pars_dict, expanded=False)
except Exception as e:
    print(e)
    st.write(f"Error: {e}")
