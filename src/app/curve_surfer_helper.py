import json
import re

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go, express as px

from src.pipeline.models.helper import pow_space
from src.pipeline.models.models import get_model
from src.pipeline.pipeline_helper import init_aeid, init_config, get_chemical, get_output_data, \
    get_cutoff, get_output_compound

CONFIG = {}
SUFFIX = ''
ERR_MSG = f"No data found for this query"


@st.cache_data
def load_assay_endpoint(aeid):  # id used to facilitate caching
    """
    Load assay endpoint data for a given aeid.
    """
    df = get_output_data(aeid).reset_index(drop=True)
    return df


@st.cache_data
def load_cutoff(aeid):  # id used to facilitate caching
    """
    Load cutoff data for a given aeid.
    """
    cutoff_df = get_cutoff().reset_index(drop=True)
    return cutoff_df


@st.cache_data
def load_output_compound(dss_tox_substance_id):  # id used to facilitate caching
    """
    Load output compound data for a given dsstox_substance_id.
    """
    return get_output_compound(dss_tox_substance_id).reset_index(drop=True)


def set_config_app(config):
    global CONFIG
    CONFIG = config
    init_config(config)


def get_series():
    """
    Get series from dataframe based on current df_index.
    """
    series = st.session_state.df.iloc[st.session_state.df_index]
    series['cautionary_flags'] = json.loads(series['cautionary_flags'])
    series['conc'] = json.loads(series['conc'])
    series['resp'] = json.loads(series['resp'])
    series['fit_params'] = json.loads(series['fit_params'])
    st.session_state.series = series.to_dict()

    if st.session_state.series is None:  # Todo: check if this is needed
        raise Exception(ERR_MSG)
    
    return st.session_state.series


def get_chem_info(spid):
    """
    Get chemical information for a given spid.
    """
    chem = get_chemical([spid]).iloc[0]
    casn = chem['casn']
    chnm = chem['chnm']
    dsstox_substance_id = chem['dsstox_substance_id']
    return casn, chnm, dsstox_substance_id


def init_figure():
    """
    Initialize figure.
    """
    series = st.session_state.series
    cutoff_df = st.session_state.cutoff_df
    cutoff = cutoff_df.at[0, 'cutoff']
    series.update({"cutoff": cutoff, "bmad": cutoff_df.at[0, 'bmad'],"onesd": cutoff_df.at[0, 'onesd']})
    fig = go.Figure()
    margin = 0
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Update Layout", value=True, key="ShowTitle", help="Show title and subtitle"):
            margin = 100
            try:
                casn, chnm, dsstox_substance_id = get_chem_info(series['spid'])
            except IndexError:
                casn, chnm, dsstox_substance_id = None, None, None
                st.warning(f"No chemical information found for SPID {series['spid']}")
            title = st.session_state.assay_info["assay_component_endpoint_name"] + " | " +  "aeid: " + str(st.session_state.aeid)
            hitcall = f"{series['hitcall']:.2f}"  # f"{series['hitcall']:.0%}"
            subtitle = f"<br><sup>{chnm} | Hitcall: {hitcall} | {str(dsstox_substance_id)} | {str(casn)}</sup>"
            fig.update_layout(title=title + subtitle, title_font=dict(size=26))
    
    signal_direction = st.session_state.assay_info['signal_direction']
    y0 = -cutoff if signal_direction != "gain" else 0
    y1 = cutoff if signal_direction != "loss" else 0
       
    fig.update_xaxes(showspikes=True, tickfont=dict(size=22, color="black"))  # Adjust 'spikesize' as needed
    fig.update_yaxes(showspikes=True, tickfont=dict(size=22, color="black"))  # Adjust 'spikesize' as needed

    fig.update_layout(
        xaxis_title="log10(Concentration) μM",
        yaxis_title=f"{st.session_state.assay_info['normalized_data_type']}",
        xaxis_title_font=dict(size=25, color="black"),
        yaxis_title_font=dict(size=25, color="black"),
    )
    fig.add_hrect(y0=y0, y1=y1, fillcolor='blue', opacity=0.15, layer='below',
                  annotation_text="efficacy cutoff", annotation_position="top left", annotation=dict(font=dict(size=20)))
    fig.update_layout(legend=dict(groupclick="toggleitem", font=dict(size=15, color='black')), font=dict(color="black", size=15), margin=dict(t=margin))
    fig.update_layout(xaxis=dict(showgrid=True))
    fig.update_layout(yaxis=dict(showgrid=True))

    # fig.update_layout(hovermode="x unified")  # uncomment to enable unified hover

    return fig, col2


def add_curves(fig, col2):
    """
    Add curves and annotations to figure.
    """
    height = 720
    series = st.session_state.series
    conc, resp, fit_params = np.array(series['conc']), series['resp'], series['fit_params']
    potency_candidates, efficacy_candidates =["acc", "actop", 'ac50', "cytotox_acc"], ["top", "cutoff"]
    with col2:
        verbose_checkbox = st.checkbox("Verbose", value=False, key="verbose", help="Show all potency and efficacy estimates")
        if verbose_checkbox:
            height = 840
            potency_candidates, efficacy_candidates = ["ac1sd", "bmd", "ac95", "ac50", "acc", "actop", "cytotox_acc"], ["cutoff", "bmad", "top"] # , "onesd" 

    potencies = [potency for potency in potency_candidates if potency in series and series[potency] is not None] # "ac1sd", bmd", "ac95", "ac50", 
    efficacies = [efficacy for efficacy in efficacy_candidates if efficacy in series and series[efficacy] is not None] # , "bmad", "onesd"

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
    best_model = get_correct_model_name(series['best_aic_model'])
    if fit_params is not None:# and series['hitcall'] > 0:
        iterator = fit_params.items()
        for index, (fit_model, fit_parameters) in enumerate(iterator):
            params = fit_parameters["pars"].copy()
            pars_dict[fit_model] = params
            er = params.pop('er')
            aic = round(fit_parameters["aic"], 2)
            fit_model = get_correct_model_name(fit_model)
            y = np.array(get_model(fit_model)('fun')(x, **params))
            color = px.colors.qualitative.Plotly[index]
            is_best = fit_model == best_model
            is_best_tag = f"(BEST)" if is_best else ""
            dash = 'solid' if is_best else "dash"
            name = f"{fit_model} {is_best_tag}"
            width = 5 if is_best else 4
            opacity = 0.9

            fig.add_trace(
                go.Scatter(x=np.log10(x), y=y, opacity=opacity, legendgroup="Fit models",
                           legendgrouptitle_text="Fit models",
                           marker=dict(color=color), mode='lines',
                           visible=True,
                           name=name, line=dict(width=width, dash=dash)))

            if is_best:
                for i, efficacy in enumerate(efficacies):
                    eff = efficacy_values[i]
                    if eff is not None and not np.isnan(eff):
                        visible = True if efficacy in ['top', 'bmad'] else "legendonly"
                        x_min = min(np.log10(x)) 
                        fig.add_trace(go.Scatter(
                            name=f"{efficacy} = {eff:.1e}",
                            x=[x_min, max(np.log10(x))],
                            y=[eff, eff],
                            mode='lines',
                            opacity=1,
                            line=dict(color=color, width=4, dash="dot"),
                            legendgroup="Efficacy",
                            legendgrouptitle_text="Efficacy",
                            visible=visible
                        ))
                        if efficacy != "cutoff":
                            annotation = go.layout.Annotation(
                                x=x_min, y=eff,
                                text=f"{efficacy}",
                                showarrow=True,
                                ax=-20, ay=0,
                                arrowwidth=0.1,
                                arrowhead=0,
                                font=dict(size=20),
                                visible=True # if visible != "legendonly" else False  # Set annotation visibility based on trace visibility
                            )
                            fig.add_annotation(annotation)

                for i, potency in enumerate(potencies):
                    pot = potency_values[i]

                    if pot is not None and not np.isnan(pot) and pot < 1000:
                        pot_log = np.log10(series[potency])

                        params = fit_params[get_original_name(best_model)]['pars'].copy()
                        params.pop('er')
                        pot_y = get_model(best_model)('fun')(np.array([pot]), **params)[0]
                        visible = True if potency in ['acc', 'ac50', 'actop'] else "legendonly"

                        fig.add_trace(go.Scatter(
                            name=f"{potency} = {pot:.1e}",
                            x=[pot_log, pot_log],
                            y=[0, pot_y],
                            mode='lines',
                            opacity=0.8,
                            line=dict(color=color, width=4),
                            legendgroup="Potency estimates",
                            legendgrouptitle_text="Potency estimates",
                            visible=visible
                        ))
                        if potency in ['acc', 'ac50', 'actop']:
                            annotation = go.layout.Annotation(
                                x=pot_log, y=0,
                                text=f"{potency}",
                                showarrow=True,
                                arrowhead=0,
                                # arrow show direct upwards
                                ax=0, ay=25,
                                font=dict(size=22),
                                visible=True # if visible != "legendonly" else False  # Set annotation visibility based on trace visibility
                            )
                            fig.add_annotation(annotation)

    
    fig.add_trace(go.Scatter(x=np.log10(conc), y=resp, mode='markers', legendgroup="Observed responses", legendgrouptitle_text="Observed responses",
                             marker=dict(symbol="x", size=16, color="royalblue"), name="Repsonses", showlegend=True))

    unique_conc = np.unique(conc)
    # get max response (i.e. max median response for multi-valued responses) and corresponding conc
    resp = np.array(resp)
    median_resp = np.array([np.median(np.array(resp)[conc == c]) for c in unique_conc])
    fig.add_trace(
        go.Scatter(x=np.log10(unique_conc), y=median_resp, mode='markers', legendgroup="Observed responses",
                   marker=dict(color="red", symbol="star", size=16), name="Median responses", showlegend=True, visible="legendonly"))

    return pars_dict, height

def get_correct_model_name(fit_model):
    if fit_model == 'sigmoid':
        fit_model = 'gnls2'
    return fit_model

def get_original_name(fit_model):
    if fit_model == 'gnls2':
        fit_model = 'sigmoid'
    return fit_model


def reset_df_index():
    st.session_state.df_index = 0


def set_trigger(trigger):
    """
    Set trigger in session state.
    """
    st.session_state.trigger = trigger
    if trigger == 'filter_assay_endpoints':
        on_filter_assay_endpoints(trigger)
        update_aeid()
        refresh_data(trigger)
        update_df_length()
        update_df_index()
        # Update current specific
        update_spid()

        # Get corresponding series from dataframe
        get_series()
    if trigger.startswith("select_compound"):
        st.session_state.hitcall_slider = (0.0, 1.0)


def sort():
    """
    Sort dataframe based on current sort_by and asc values.
    """
    st.session_state.df = st.session_state.df.sort_values(
        by=st.session_state.sort_by, ascending=st.session_state.asc, na_position='last').reset_index(drop=True)
    reset_df_index()


def update_spid():
    """
    Update spid in session state.
    """
    st.session_state.compound_id = st.session_state.df.iloc[st.session_state.df_index]['spid']


def on_select_compound(trigger, slider):
    """
    Select compound based on current compound_id.
    """
    if trigger == "select_compound" or (trigger == "select_compound_focus_changed" and st.session_state.focus_on_compound):
        res = st.session_state.df[st.session_state.df['dsstox_substance_id'] == st.session_state.compound_id]
        if res.empty:
            st.error(f"Input string {st.session_state.compound_id} not found", icon="🚨")
            return
        st.session_state.df_index = res.index[0]

    if trigger.startswith("select_compound"):
        slider.empty()
        st.session_state.slider_iteration += 1
        slider.slider("Select hitcall range", 0.0, 1.0, (0.0, 1.0), key=st.session_state.slider_iteration)




def update_df_index():
    st.session_state.df_index = st.session_state.df_index % st.session_state.df_length


def on_iterate_compounds(trigger):
    """
    Iterate through compounds.
    """
    if trigger == "next_compound":
        st.session_state.df_index += 1
    if trigger == "prev_compound":
        st.session_state.df_index -= 1


def on_iterate_assay_endpoint(trigger):
    """
    Iterate through assay endpoints.
    """
    if st.session_state.focus_on_compound:
        if trigger == "next_assay_endpoint":
            st.session_state.df_index += 1
        if trigger == "prev_assay_endpoint":
            st.session_state.df_index -= 1

    if trigger == "next_assay_endpoint":
        st.session_state.aeid_index += 1
    if trigger == "prev_assay_endpoint":
        st.session_state.aeid_index -= 1


def check_sort(trigger):
    """
    Check how dataframe needs to be sorted.
    """
    if trigger in ["sort_by", "asc", "hitcall_slider"]:
        sort()


def update_df_length():
    """
    Update length of dataframe.
    """
    st.session_state.df_length =  len(st.session_state.df)
    if st.session_state.df_length == 0:
        raise Exception(ERR_MSG)


def on_hitcall_slider(trigger):
    """
    Filter dataframe based on hitcall slider.
    """
    if trigger == "hitcall_slider":
        interval = st.session_state.hitcall_slider
        if st.session_state.sort_by == "hitcall":
            sort_slider = "hitcall"
        else:
            sort_slider = "hitcall_c"

        df = st.session_state.df.query(f'{interval[0]} <= {sort_slider} <= {interval[1]}').reset_index(drop=True)
        st.session_state.df = pd.concat([df, st.session_state.df.query(f"{sort_slider}.isna()")]) if interval[0] == 0.0 else df
        reset_df_index()


def refresh_data(trigger):
    """
    Refresh data based on current trigger.
    """
    fresh_load = st.session_state.last_aeid != st.session_state.aeid or st.session_state.focus_on_compound_submitted
    refresh_load = fresh_load or trigger in ["compound", "hitcall_slider"]
    if refresh_load:
            init_aeid(st.session_state.aeid)
            if (trigger == 'hitcall_slider' or st.session_state.focus_on_compound_submitted) and st.session_state.focus_on_compound:
                output_compound = load_output_compound(st.session_state.series['dsstox_substance_id'])
                st.session_state.df = pd.merge(output_compound, st.session_state.aeids, on='aeid', how='inner')
                on_hitcall_slider("hitcall_slider")
            else:
                st.session_state.df = load_assay_endpoint(st.session_state.aeid)  

            st.session_state.cutoff_df = load_cutoff(st.session_state.aeid)
            st.session_state.last_aeid = st.session_state.aeid
            if fresh_load:
                sort()


def update_aeid():
    """
    Update aeid in session state.
    """
    if st.session_state.focus_on_compound:
        if len(st.session_state.df) == 0:
            raise Exception(ERR_MSG)
        st.session_state.aeid_index = st.session_state.aeid_index % len(st.session_state.df)
        merged = pd.merge(st.session_state.df, st.session_state.aeids, on='aeid', how='inner')
        st.session_state.aeid = merged.iloc[st.session_state.aeid_index]['aeid']
    else:
        st.session_state.aeid_index = st.session_state.aeid_index % len(st.session_state.aeids)
        st.session_state.aeid = st.session_state.aeids.iloc[st.session_state.aeid_index]
    st.session_state.assay_info = st.session_state.assay_info_df[st.session_state.assay_info_df['aeid'] == st.session_state.aeid].iloc[0]


def on_filter_assay_endpoints(trigger):
    """
    Filter assay endpoints based on selected fields.
    """
    if trigger == "filter_assay_endpoints":
        if not st.session_state.assay_info_selected_fields:
            st.session_state.aeids = st.session_state.aeids_all.copy()
        else:
            st.session_state.aeids = st.session_state.assay_info_df[
                st.session_state.assay_info_df[st.session_state.assay_info_column].isin(
                st.session_state.assay_info_selected_fields)]['aeid']
        st.session_state.aeid_index = 0
        if st.session_state.focus_on_compound:
            st.session_state.df = pd.merge(st.session_state.df, st.session_state.aeids, on='aeid', how='inner')


def get_trigger():
    trigger = st.session_state.trigger
    return trigger


def get_compound_info():
    """
    Get compound information.
    """
    try:
        casn, chnm, dsstox_substance_id = get_chem_info(st.session_state.compound_id)
    except IndexError:
        casn, chnm, dsstox_substance_id = None, None, None

    dsstox_substance_id_link = f"https://comptox.epa.gov/dashboard/chemical/details/{dsstox_substance_id}"
    dsstox_substance_id_link = dsstox_substance_id_link if dsstox_substance_id is not None else "N/A"
    casn_link = f'https://commonchemistry.cas.org/detail?cas_rn={casn}'
    casn_link = casn_link if casn is not None else 'N/A'
    fitc = int(st.session_state.series['fitc'])
    cytotox_ref = st.session_state.series['cytotox_ref']
    cytotox_prob = st.session_state.series['cytotox_prob']
    cytotox_acc = st.session_state.series['cytotox_acc']
    compound_info_df = pd.DataFrame(
        {   
            "Chemical name": [chnm],
            "DSSTOXSID": [dsstox_substance_id],
            "CASRN": [casn],
            "SPID": [st.session_state.compound_id],
            "Hitcall": [f"{st.session_state.series['hitcall']:.2f}"],
            "Hitcall_cytotox_corrected": [f"{st.session_state.series['hitcall_c']:.2f}"],
            "Cytotox ref": [f"{cytotox_ref} assay based" if cytotox_ref is not None else "no target"],
            "Cytotox ref acc": [f"{(cytotox_acc):.1e}" if cytotox_acc is not None else "N/A"],
            "Cytotox ref prob": [f"{(cytotox_prob):.2f}" if cytotox_prob is not None else "N/A"],
            "Fitc": [f"{fitc}"],
            "Caution Flags": [f"{st.session_state.series['cautionary_flags']}"],
            "ICE Omit Flag": [f"{st.session_state.series['omit_flag']}"],
            "DSSTOXSID Link": [dsstox_substance_id_link],
            "CASRN Link": [casn_link],

        }
    )

    column_config = {
        "Hitcall": st.column_config.ProgressColumn(
            "Hitcall",
            help=f"Bioactivity score :biohazard_sign:",
            format="%.2f",
            min_value=0,
            max_value=1,
        ), 
        "Hitcall_cytotox_corrected": st.column_config.ProgressColumn(
            "Hitcall_cytotox_corrected",
            help=f"Bioactivity score :biohazard_sign:",
            format="%.2f",
            min_value=0,
            max_value=1,
        ), 
        "DSSTOXSID": st.column_config.TextColumn(label="DSSTOXSID", help="Distributed Structure-Searchable Toxicity Substance ID :bar_chart:"),
        "CASRN": st.column_config.TextColumn(label="CASRN", help="CAS Registry Number :information_source:"),
        "Chemical name": st.column_config.TextColumn(label="Chemical name", help="Chemical name :test_tube:"),
        "SPID": st.column_config.TextColumn(label="SPID", help="Sample :id:"),
        "DSSTOXSID Link": st.column_config.LinkColumn("DSSTOXSID Link", help="Link to comptox"),
        "CASRN Link": st.column_config.LinkColumn(label="CASRN Link", help="Link to commonchemistry"),
    }
  
    return compound_info_df, column_config


def get_assay_info(subset=False, transpose=False, replace=False):
    """
    Get assay information.
    """
    assay_info_df = st.session_state.assay_info[subset_assay_info_columns] if subset else st.session_state.assay_info
    assay_info_df = assay_info_df.to_frame()
    if transpose:
        assay_info_df = assay_info_df.T

    if replace:
        column_mapping = {col: col.replace("_", " ") for col in assay_info_df.columns}

        assay_info_df = assay_info_df.rename(columns=column_mapping)

        # Function to convert camel case to words with spaces
        def camel_case_to_words(name):
            conversion = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
            # Uppercase the first character
            return conversion[0].upper() + conversion[1:]
            
        # Create a mapping dictionary for column renaming
        column_mapping = {col: camel_case_to_words(col) for col in assay_info_df.columns}

        # Rename columns using the mapping dictionary
        assay_info_df = assay_info_df.rename(columns=column_mapping)
        assay_info_df = assay_info_df.rename(columns={"assay component endpoint name": "assay endpoint"})
        assay_info_df = assay_info_df.rename(columns={"Aeid": "aeid"})

    return assay_info_df


subset_assay_info_columns = ["assay_component_endpoint_name",
                             "aeid",
                             "assay_function_type",
                             "signal_direction",
                             "MechanisticTarget",
                             "ToxicityEndpoint",
                             "biological_process_target",
                             "intended_target_family",
                             "intended_target_family_sub",
                             "intended_target_type",
                             "intended_target_type_sub",
                             "burst_assay",
                             ]
