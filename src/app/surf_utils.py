import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go, express as px
from st_files_connection import FilesConnection

from src.utils.fit_models import get_model
from src.utils.models.get_inverse import pow_space
from src.utils.pipeline_helper import print_, get_assay_info, get_cutoff, set_config, get_chemical
from src.utils.query_db import query_db
from src.utils.constants import OUTPUT_DIR_PATH

# Add the parent folder to sys.path
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_folder_path)


CONFIG = {}
SUFFIX = ''
ERR_MSG = f"No data found for this query"


@st.cache_data
def load_data(aeid):  # aeid parameter used to handle correct caching
    df = get_output_data()
    cutoff_df = get_cutoff()
    return df, cutoff_df


def get_output_data():
    tbl = 'output'
    path = os.path.join(OUTPUT_DIR_PATH, f"{CONFIG['aeid']}{SUFFIX}")
    print_(f"Fetch data with assay ID {CONFIG['aeid']}..")
    if not os.path.exists(path):
        if CONFIG['enable_reading_db']:
            qstring = f"SELECT * FROM {tbl} WHERE aeid = {st.session_state.aeid};"
            df = query_db(query=qstring)
        else:
            conn = st.experimental_connection('s3', type=FilesConnection)
            data_source = os.path.join(CONFIG['bucket'], tbl, f"{CONFIG['aeid']}{SUFFIX}")
            df = conn.read(data_source, input_format="parquet", ttl=600)
    else:
        df = pd.read_parquet(path)
    length = df.shape[0]
    if length == 0:
        st.error(f"No data found for AEID {CONFIG['aeid']}", icon="🚨")
    print_(f"{length} series loaded")
    return df


def set_config_app(config):
    global CONFIG, SUFFIX
    CONFIG = config
    set_config(config)
    SUFFIX = f".{config['data_file_format']}.gzip"


def get_series():
    series = st.session_state.df.iloc[st.session_state.id]
    series['conc'] = json.loads(series['conc'])
    series['resp'] = json.loads(series['resp'])
    series['fit_params'] = json.loads(series['fit_params'])
    return series.to_dict()


def get_chem_info(spid):
    chem = get_chemical([spid]).iloc[0]
    casn = chem['casn']
    chnm = chem['chnm']
    dsstox_substance_id = chem['dsstox_substance_id']
    return casn, chnm, dsstox_substance_id


def init_figure(series, cutoff_df):
    cutoff = cutoff_df['cutoff'].iloc[0]
    bmad = cutoff_df['bmad'].iloc[0]
    onesd = cutoff_df['onesd'].iloc[0]
    st.session_state.series.update({"cutoff": cutoff, "bmad": bmad, "onesd": onesd})
    fig = go.Figure()
    casn, chnm, dsstox_substance_id = get_chem_info(series['spid'])
    # fig.update_layout(hovermode="x unified")  # uncomment to enable unified hover
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    assay_infos = get_assay_info(st.session_state.aeid)
    normalized_data_type = assay_infos["normalized_data_type"]
    assay_component_endpoint_name = assay_infos["assay_component_endpoint_name"]
    title = "AEID: " + str(st.session_state.aeid) + " | " + assay_component_endpoint_name
    hitcall = f"{series['hitcall']:.2f}"  # f"{series['hitcall']:.0%}"
    subtitle = f"<br><sup>Hitcall: {hitcall} | {str(series['spid'])} | {chnm} | {str(dsstox_substance_id)} | {str(casn)}</sup>"
    fig.update_layout(title=title + subtitle, title_font=dict(size=26))
    fig.update_layout(margin=dict(t=150), xaxis_title="log10(Concentration) μM", yaxis_title=str(normalized_data_type))
    fig.add_hrect(y0=-cutoff, y1=cutoff, fillcolor='black', opacity=0.1, layer='below', line=dict(width=0),
                  annotation_text="efficacy cutoff", annotation_position="top left")
    fig.update_layout(legend=dict(groupclick="toggleitem"))
    return fig


def add_curves(series, fig):
    conc, resp, fit_params = np.array(series['conc']), series['resp'], series['fit_params']

    fig.add_trace(
        go.Scatter(x=np.log10(conc), y=resp, mode='markers', legendgroup="Response", legendgrouptitle_text="Repsonse",
                   marker=dict(color="gray", symbol="x-thin-open", size=10), name="Repsonse", showlegend=True))

    potencies = [potency for potency in ["acc", "ac50", "actop", "ac1sd", "bmd"] if potency in series]
    efficacies = [efficacy for efficacy in ["top", "cutoff", "bmad", "onesd"] if efficacy in series]

    potency_values = [series[potency] for potency in potencies]
    efficacy_values = [series[efficacy] for efficacy in efficacies]

    sorted_pairs = sorted(zip(potency_values, potencies))
    potency_values, potencies = zip(*sorted_pairs)

    sorted_pairs = sorted(zip(efficacy_values, efficacies))
    efficacy_values, efficacies = zip(*sorted_pairs)

    pars_dict = {}
    min_val, max_val = np.min(conc), np.max(conc)
    min_potencies = np.min(potency_values) if len(potency_values) > 0 else 10000000

    min_val = min(min_val, min_potencies)
    x = pow_space(min_val, max_val, 10, 200)
    best_model = series['best_aic_model']
    color_best = "gray"
    if fit_params is not None and series['hitcall'] > 0:
        iterator = fit_params.items()
        for index, (fit_model, fit_parameters) in enumerate(iterator):
            params = fit_parameters["pars"].copy()
            pars_dict[fit_model] = params
            er = params.pop('er')
            aic = round(fit_parameters["aic"], 2)
            y = np.array(get_model(fit_model)('fun')(x, **params))
            color = px.colors.qualitative.Bold[index]
            is_best = fit_model == best_model
            color_best = color if is_best else color_best
            is_best_tag = f"(BEST)" if is_best else ""
            dash = 'solid' if is_best else "dash"
            name = f"{fit_model} {is_best_tag}"
            width = 4 if is_best else 2
            opacity = 0.9

            fig.add_trace(
                go.Scatter(x=np.log10(x), y=y, opacity=opacity, legendgroup="Fit models", legendgrouptitle_text="Fit models",
                           marker=dict(color=color), mode='lines', visible="legendonly" if fit_model == 'cnst' else True,
                           name=name, line=dict(width=width, dash=dash)))

    for i, efficacy in enumerate(efficacies):
        eff = efficacy_values[i]
        if eff is not None and not np.isnan(eff):
            fig.add_trace(go.Scatter(
                name=f"{efficacy} = {eff:0.2f}",
                x=[min(np.log10(x)), max(np.log10(x))],
                y=[eff, eff],
                mode='lines',
                opacity=0.8,
                line=dict(color='gray', width=2, dash='dashdot'),
                legendgroup="Efficacy",
                legendgrouptitle_text="Efficacy",
                visible="legendonly"
            ))

    for i, potency in enumerate(potencies):
            pot = potency_values[i]
            if pot is not None and not np.isnan(pot):
                pot_log = np.log10(series[potency])
                params = fit_params[best_model]['pars'].copy()
                params.pop('er')
                pot_y = get_model(best_model)('fun')(np.array([pot]), **params)[0]
                fig.add_trace(go.Scatter(
                    name=f"{potency} = {pot:0.1f}",
                    x=[pot_log, pot_log],
                    y=[0, pot_y],
                    mode='lines',
                    opacity=0.8,
                    line=dict(color=color_best, width=3, dash='dot'),
                    legendgroup="Potency estimates",
                    legendgrouptitle_text="Potency estimates",
                ))


    return pars_dict


def check_reset():
    if "id" not in st.session_state:
        reset_id()
    if "aeid" not in st.session_state:
        st.session_state.aeid = 0
    if "last_aeid" not in st.session_state:
        st.session_state.last_aeid = -1
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
    if "df" not in st.session_state:
        st.session_state.df = None
    if "series" not in st.session_state:
        st.session_state.series = None


def reset_id():
    st.session_state.id = 0


def trigger(trigger):
    st.session_state.trigger = trigger


def filter_spid():
    st.session_state.trigger = "spid"
    st.session_state.hitcall_slider = (0.0, 1.0)


def sort():
    st.session_state.df = st.session_state.df.sort_values(
        by=st.session_state.sort_by, ascending=st.session_state.asc, na_position='last').reset_index(drop=True)
    reset_id()


def update():
    trigger = st.session_state.trigger
    fresh_load = "df" not in st.session_state or st.session_state.last_aeid != st.session_state.aeid
    refresh_load = fresh_load or trigger in ["spid", "hitcall_slider"]
    if refresh_load:
        st.session_state.df, st.session_state.cutoff = load_data(
            st.session_state.aeid)  # requires id parameter for unique caching
        st.session_state.last_aeid = st.session_state.aeid
        if fresh_load:
            sort()

    if trigger == "hitcall_slider":
        interval = st.session_state.hitcall_slider
        df = st.session_state.df.query(f'{interval[0]} <= hitcall <= {interval[1]}').reset_index(drop=True)
        st.session_state.df = pd.concat([df, st.session_state.df.query("hitcall.isna()")]) if interval[0] == 0.0 else df
        reset_id()

    length = len(st.session_state.df)
    st.session_state.length = length

    if trigger in ["sort_by", "asc", "hitcall_slider"]:
        sort()

    if trigger == "next":
        st.session_state.id += 1
    if trigger == "prev":
        st.session_state.id -= 1
        length = st.session_state.length

    if length == 0:
        raise Exception(ERR_MSG)

    st.session_state.id = st.session_state.id % st.session_state.length

    if trigger == "spid" and st.session_state.spid != "":
        res = st.session_state.df[st.session_state.df['spid'] == st.session_state.spid]
        if res.empty:
            st.error(f"Input string {st.session_state.spid}' not found", icon="🚨")
        else:
            st.session_state.id = res.index[0]
    else:
        st.session_state.spid = st.session_state.df.iloc[st.session_state.id]['spid']

    st.session_state.series = get_series()
    if st.session_state.series is None:
        raise Exception(ERR_MSG)

    fig = init_figure(st.session_state.series, st.session_state.cutoff)
    pars_dict = add_curves(st.session_state.series, fig)
    return fig, pars_dict


def get_assay_and_sample_info():
    assay_infos = get_assay_info(st.session_state.aeid)
    assay_component_endpoint_name = assay_infos["assay_component_endpoint_name"]
    assay_component_endpoint_desc = assay_infos["assay_component_endpoint_desc"]
    # st.subheader(f"AEID: {st.session_state.aeid} | {assay_component_endpoint_name}")
    casn, chnm, dsstox_substance_id = get_chem_info(st.session_state.spid)
    dsstox_substance_id = f"https://comptox.epa.gov/dashboard/chemical/details/{dsstox_substance_id}" if dsstox_substance_id else "N/A"
    df = pd.DataFrame(
        {
            "Hitcall": [f"{st.session_state.series['hitcall']:.2f}"],
            "SPID": [st.session_state.spid],
            "Chemical": [chnm],
            "DSSTOXSID": [dsstox_substance_id],
            "CASRN": [casn],
        }
    )
    with st.expander("Sample info :petri_dish:", expanded=True):
        st.dataframe(df, column_config={
            "Hitcall": st.column_config.ProgressColumn(
                "Hitcall",
                help="Bioactivity score :biohazard_sign: ",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Chemical": st.column_config.TextColumn(label="Chemical", help="Chemical Name :test_tube:"),
            "SPID": st.column_config.TextColumn(label="SPID", help="Sample :id:"),
            "DSSTOXSID": st.column_config.LinkColumn("DSSTOXSID",
                                                     help="Distributed Structure-Searchable Toxicity Substance ID. Link to dashboard :bar_chart:"),
            "CASRN": st.column_config.TextColumn(label="CASRN",
                                                 help="CAS Registry Number :information_source:"),
        },
                     hide_index=True,
                     use_container_width=True,
                     )
    return assay_component_endpoint_desc
