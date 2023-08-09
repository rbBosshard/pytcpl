import json

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go, express as px


from fit_models import powspace, get_model
from pipeline_helper import print_, get_assay_info
from query_db import query_db


@st.cache_data
def load_data(aeid):  # aeid parameter used to handle correct caching
    print_(f"Fetch data from DB with assay ID {aeid}..")
    qstring = f"SELECT * FROM output WHERE aeid = {st.session_state.aeid};"
    df = query_db(query=qstring)
    length = df.shape[0]
    if length == 0:
        st.error(f"No data found for AEID {aeid}", icon="🚨")
    print_(f"{length} series loaded")
    qstring = f"SELECT * FROM cutoffs WHERE aeid = {st.session_state.aeid};"
    cutoff = query_db(query=qstring)['cutoff'].iloc[0]
    return df, cutoff


def get_series():
    series = st.session_state.df.iloc[st.session_state.id]
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
    # fig.update_layout(hovermode="x unified")  # uncomment to enable unified hover
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    assay_infos = get_assay_info(st.session_state.aeid)
    normalized_data_type = assay_infos["normalized_data_type"]
    assay_component_endpoint_name = assay_infos["assay_component_endpoint_name"]
    title = "AEID: " + str(st.session_state.aeid) + " | " + assay_component_endpoint_name
    hitcall =  f"{series['hitcall']:.2f}"  # f"{series['hitcall']:.0%}"
    subtitle = f"<br><sup>Hitcall: {hitcall} | {str(series['spid'])} | {chnm} | {str(dsstox_substance_id)} | {str(casn)}</sup>" 
    fig.update_layout(title=title + subtitle)
    fig.update_layout(margin=dict(t=150), xaxis_title="log10(Concentration) μM", yaxis_title=str(normalized_data_type))
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
        y = np.array(get_model(fit_model)('fun')(x, **params))
        color = px.colors.qualitative.Bold[index]
        best = fit_model == series['best_aic_model']
        best_fit = f"(BEST)" if best else ""
        dash = 'solid' if best else "dash"
        name = f"{fit_model} {best_fit}"
        width = 3 if best else 2
        legendgroup = "Fit models"  #fit_model if best else None
        opacity = 1 if best else .8

        fig.add_trace(go.Scatter(x=np.log10(x), y=y, opacity=opacity, legendgroup=legendgroup, legendgrouptitle_text=legendgroup, marker=dict(color=color), mode='lines',
                                    name=name, line=dict(width=width, dash=dash)))

        if best:
            # for potency in potencies:
            #     if potency in series:
            #         pot = np.log10(series[potency])
            #         if pot is not None and not np.isnan(pot):
            #             fig.add_vline(x=pot, line_color=color, line_width=2, annotation_position="bottom left", annotation_text=potency,
            #                     layer="below", opacity=0.5)


            for efficacy in efficacies:
                if efficacy in series:
                    eff = series[efficacy]
                    if eff is not None and not np.isnan(eff):
                        fig.add_hline(y=eff, line_color=color, line_width=2, annotation_position="bottom left", annotation_text=efficacy,
                        layer="below", opacity=0.5)
            

            # # Add vertical marker as a trace
            for potency in potencies:
                if potency in series:
                    pot = series[potency]
                    pot_log = np.log10(series[potency])
                    if pot is not None and not np.isnan(pot):
                        pot_y = get_model(fit_model)('fun')(np.array([pot]), **params)[0]
                        fig.add_trace(go.Scatter(
                            name=f"{potency} = {pot:0.1f}",
                            x=[pot_log, pot_log],
                            y=[0, pot_y],  # Coordinates to cover the entire vertical range of the plot
                            mode='lines',
                            line=dict(color=color, width=2),
                            legendgroup="Potency estimates",
                            legendgrouptitle_text="Potency estimates",
                            # showlegend=False,  # Do not show the marker in the legend
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
        st.session_state.asc = True
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
    st.session_state.hitcall_slider =  (0.0, 1.0)


def sort():
    st.session_state.df = st.session_state.df.sort_values(
        by=st.session_state.sort_by, ascending=st.session_state.asc, na_position='last').reset_index(drop=True)
    reset_id()


def update():
    trigger = st.session_state.trigger
    fresh_load = "df" not in st.session_state or st.session_state.last_aeid != st.session_state.aeid
    refresh_load = fresh_load or trigger in ["spid", "hitcall_slider"]
    if refresh_load:
        st.session_state.df, st.session_state.cutoff = load_data(st.session_state.aeid)  # requires id parameter for unique caching
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
        st.error(f"Input string {st.session_state.spid}' not found", icon="🚨")
        return None, None
    
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
        st.error("No data not found", icon="🚨")

    fig = init_figure(st.session_state.series, st.session_state.cutoff)
    pars_dict = add_curves(st.session_state.series, fig)
    return fig, pars_dict


def get_assay_and_sample_info():
    assay_infos = get_assay_info(st.session_state.aeid)
    assay_component_endpoint_name = assay_infos["assay_component_endpoint_name"]
    assay_component_endpoint_desc = assay_infos["assay_component_endpoint_desc"]
    st.subheader(f"AEID: {st.session_state.aeid} | {assay_component_endpoint_name}")
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
        st.dataframe(df,
                     column_config={
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
                         "CASRN": st.column_config.TextColumn(label="CASN",
                                                              help="CAS Registry Number :information_source:"),
                     },
                     hide_index=True,
                     use_container_width=True,
                     )
    return assay_component_endpoint_desc
