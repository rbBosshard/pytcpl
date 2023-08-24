import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go, express as px

from src.pipeline.pipeline_helper import get_assay_info, init_config, get_chemical, merge_all_outputs, get_output_data, get_cutoff
from src.pipeline.pipeline_constants import METADATA_SUBSET_DIR_PATH
from src.pipeline.models.helper import pow_space
from src.pipeline.models.models import get_model

CONFIG = {}
SUFFIX = ''
ERR_MSG = f"No data found for this query"


@st.cache_data
def load_assay_endpoint(aeid):  # aeid parameter used to handle correct caching
    """
    Load data for a specific AEID.

    This function loads data for a given assay endpoint ID (AEID). It retrieves the data from an output table and a cutoff
    table. The data is either fetched from a local file or a remote source based on configuration settings. The loaded data
    is returned as two dataframes.

    Args:
        aeid (int): Assay endpoint ID for which to load data.

    Returns:
        tuple: A tuple containing two dataframes, one for the output data and the other for the cutoff data.

    """

    df = get_output_data(aeid).reset_index(drop=True)
    cutoff_df = get_cutoff(aeid).reset_index(drop=True)

    return df, cutoff_df


@st.cache_data
def merge_all_outputs_wrapper(cache_id):
    return merge_all_outputs()


def set_config_app(config):
    """
    Set the configuration for the Streamlit app.

    This function sets the global configuration variable CONFIG using the provided configuration dictionary.

    Args:
        config (dict): Configuration settings for the app.

    """
    global CONFIG
    CONFIG = config
    init_config(config)


def get_series():
    """
    Get the series data for the currently selected index.

    This function retrieves the data for the currently selected index (id) from the session_state dataframe. The data is
    processed and returned as a dictionary.

    Returns:
        dict: Dictionary containing the series data.

    """
    series = st.session_state.df.iloc[st.session_state.id]
    series['conc'] = json.loads(series['conc'])
    series['resp'] = json.loads(series['resp'])
    series['fit_params'] = json.loads(series['fit_params'])
    return series.to_dict()


def get_chem_info(spid):
    """
    Get chemical information for a given SPID.

    This function retrieves chemical information such as CAS number, chemical name, and DSSTox substance ID for a
    specified Sample ID (SPID).

    Args:
        spid (str): Sample ID for which to retrieve chemical information.

    Returns:
        tuple: A tuple containing CAS number, chemical name, and DSSTox substance ID.

    """
    
    chem = get_chemical([spid]).iloc[0]
    casn = chem['casn']
    chnm = chem['chnm']
    dsstox_substance_id = chem['dsstox_substance_id']
    return casn, chnm, dsstox_substance_id


def init_figure(series, cutoff_df):
    """
    Initialize the plot figure for visualizing data.

    This function initializes a Plotly figure object for visualizing the data. It sets up various plot attributes such as
    title, labels, and annotations. It also adds the efficacy cutoff line to the plot.

    Args:
        series (dict): Dictionary containing series data.
        cutoff_df (pd.DataFrame): DataFrame containing cutoff data.

    Returns:
        go.Figure: Initialized Plotly figure object.

    """
    cutoff = cutoff_df['cutoff'].iloc[0]
    bmad = cutoff_df['bmad'].iloc[0]
    onesd = cutoff_df['onesd'].iloc[0]
    st.session_state.series.update({"cutoff": cutoff, "bmad": bmad, "onesd": onesd})
    fig = go.Figure()
    try:
        casn, chnm, dsstox_substance_id = get_chem_info(series['spid'])
    except IndexError:
        casn, chnm, dsstox_substance_id = None, None, None
        st.warning(f"No chemical information found for SPID {series['spid']}")
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
    """
    Add curves to the plot based on series data.

    This function adds response and fitted curves to the plot based on the provided series data. It also adds potency and
    efficacy markers to the plot.

    Args:
        series (dict): Dictionary containing series data.
        fig (go.Figure): Plotly figure object to which curves are added.

    Returns:
        dict: Dictionary containing fitted curve parameters.

    """
    conc, resp, fit_params = np.array(series['conc']), series['resp'], series['fit_params']

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
                go.Scatter(x=np.log10(x), y=y, opacity=opacity, legendgroup="Fit models",
                           legendgrouptitle_text="Fit models",
                           marker=dict(color=color), mode='lines',
                           visible=True,
                           name=name, line=dict(width=width, dash=dash)))

    for i, efficacy in enumerate(efficacies):
        eff = efficacy_values[i]
        if eff is not None and not np.isnan(eff):
            fig.add_trace(go.Scatter(
                name=f"{efficacy} = {eff:.1e}",
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
                name=f"{potency} = {pot:.1e}",
                x=[pot_log, pot_log],
                y=[0, pot_y],
                mode='lines',
                opacity=0.8,
                line=dict(color=color_best, width=3, dash='dot'),
                legendgroup="Potency estimates",
                legendgrouptitle_text="Potency estimates",
            ))

    fig.add_trace(go.Scatter(x=np.log10(conc), y=resp, mode='markers', legendgroup="Response", legendgrouptitle_text="Repsonse",
                             marker=dict(color="blue", symbol="circle-open-dot", size=20), name="Repsonse", showlegend=True))
    return pars_dict


def check_reset():
    """
    Check and set session state variables.

    This function checks and sets various session state variables used by the app. It ensures that necessary variables are
    initialized or reset as needed.

    """
    if "id" not in st.session_state:
        reset_id()
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
        st.session_state.trigger = "assay_info"
    if "assay_endpoint_info" not in st.session_state:
        st.session_state.assay_endpoint_info = None
    if "assay_info" not in st.session_state:
        st.session_state.assay_info = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info{CONFIG['file_format']}"))
    if "assay_info_distinct_values" not in st.session_state:
        with open(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info_distinct_values.json"), "r") as json_file:
            st.session_state.assay_info_distinct_values = json.load(json_file)
    if "aeids" not in st.session_state:
        st.session_state.aeids_all = pd.read_parquet(
            os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_sorted{CONFIG['file_format']}"))['aeid'].reset_index(drop=True)
        st.session_state.aeids = st.session_state.aeids_all.copy()
    if "aeid" not in st.session_state:
        st.session_state.aeid = st.session_state.aeids.iloc[-1]
    if "aeid_index" not in st.session_state:
        st.session_state.aeid_index = 0
    if "num_assay_endpoints_filtered" not in st.session_state:
        st.session_state.num_assay_endpoints_filtered = len(st.session_state.aeids)
    if "assay_info_column" not in st.session_state:
        st.session_state.assay_info_column = "aeid"
    if "assay_info_selected_fields" not in st.session_state:
        st.session_state.assay_info_selected_fields = [st.session_state.aeid]
    if "df" not in st.session_state:
        st.session_state.df, st.session_state.cutoff = load_assay_endpoint(st.session_state.aeid) 
    if "series" not in st.session_state:
        st.session_state.series = get_series()


def reset_id():
    """
    Reset the index (id) of the current series.

    This function resets the index (id) of the currently selected series to the initial value.

    """
    st.session_state.id = 0


def trigger(trigger):
    """
    Set the trigger in session state.

    This function sets the trigger value in the session state, which is used to control app behavior based on user
    interactions.

    Args:
        trigger (str): The trigger value to set.

    """
    st.session_state.trigger = trigger
    if trigger == "spid":
        st.session_state.hitcall_slider = (0.0, 1.0)



def sort():
    """
    Sort the data based on user-selected criteria.

    This function sorts the data in the session state dataframe based on user-selected sorting criteria and ascending/descending
    order.

    """
    st.session_state.df = st.session_state.df.sort_values(
        by=st.session_state.sort_by, ascending=st.session_state.asc, na_position='last').reset_index(drop=True)
    reset_id()


def update():
    """
    Update the app state and visualization.

    This function updates the app's state and visualization based on user interactions and triggers. It handles loading data,
    filtering, sorting, and updating the Plotly figure.

    Returns:
        tuple: A tuple containing the updated Plotly figure and a dictionary of fitted curve parameters.

    """
    trigger = st.session_state.trigger

    if trigger == "next_assay_endpoint":
        st.session_state.aeid_index += 1
    if trigger == "prev_assay_endpoint":
        st.session_state.aeid_index -= 1

    if trigger == "assay_info":
        if st.session_state.assay_info_selected_fields == []:
            st.session_state.aeids = st.session_state.aeids_all.copy()
        else:
            st.session_state.aeids = st.session_state.assay_info[
                st.session_state.assay_info[st.session_state.assay_info_column].isin(
                st.session_state.assay_info_selected_fields)]['aeid']
        st.session_state.aeid_index = 0

    st.session_state.aeid_index = st.session_state.aeid_index % len(st.session_state.aeids)
    st.session_state.aeid = st.session_state.aeids.iloc[st.session_state.aeid_index]


    fresh_load = "df" not in st.session_state or st.session_state.last_aeid != st.session_state.aeid
    refresh_load = fresh_load or trigger in ["spid", "hitcall_slider"]
    if refresh_load:
        st.session_state.df, st.session_state.cutoff = load_assay_endpoint(st.session_state.aeid)  
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

    if trigger == "next_compound":
        st.session_state.id += 1
    if trigger == "prev_compound":
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

    st.session_state.spid = st.session_state.df.iloc[st.session_state.id]['spid']

    st.session_state.series = get_series()
    if st.session_state.series is None:
        raise Exception(ERR_MSG)

    fig = init_figure(st.session_state.series, st.session_state.cutoff)
    pars_dict = add_curves(st.session_state.series, fig)
    return fig, pars_dict


def get_assay_and_sample_info():
    """
    Get assay and sample information for display.

    This function retrieves assay and sample information for the currently selected AEID and SPID. It formats the information
    and prepares it for display in an expander.

    Returns:
        str: Formatted assay and sample information.

    """
    assay_infos = get_assay_info(st.session_state.aeid)
    assay_component_endpoint_name = assay_infos["assay_component_endpoint_name"]
    assay_component_endpoint_desc = assay_infos["assay_component_endpoint_desc"]
    # st.subheader(f"AEID: {st.session_state.aeid} | {assay_component_endpoint_name}")
    try:
        casn, chnm, dsstox_substance_id = get_chem_info(st.session_state.spid)
    except IndexError:
        casn, chnm, dsstox_substance_id = None, None, None
        st.warning(f"No chemical information found for SPID {st.session_state.spid}")

    dsstox_substance_id = f"https://comptox.epa.gov/dashboard/chemical/details/{dsstox_substance_id}" if dsstox_substance_id else "N/A"
    casn = f"https://commonchemistry.cas.org/detail?cas_rn={casn}" if casn else "N/A"
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
            "CASRN": st.column_config.LinkColumn(label="CASRN",
                                                 help="CAS Registry Number :information_source:"),
        },
                     hide_index=True,
                     use_container_width=True,
                     )
    return assay_component_endpoint_desc


subset_assay_info_columns = ["aeid",
                             "assay_component_endpoint_name",
                             "biological_process_target",
                             "intended_target_type",
                             "intended_target_type_sub",
                             "intended_target_family",
                             "intended_target_family_sub",
                             "ToxicityEndpoint",
                             "MechanisticTarget"
                             ]
