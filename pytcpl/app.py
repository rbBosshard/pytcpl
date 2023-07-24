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
from pipeline_helper import starting, elapsed


# Run command `streamlit run pytcpl/app.py`
# Ensure: test = 0 in config.yaml


def powspace(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


# Load data initially or when id changes, and cache the result
@st.cache_data
def fetch_data(id):
    start = starting("Fetch data")
    check_reset()
    dat = tcpl_load_data(lvl=3, fld='aeid', ids=id)
    grouped = dat.groupby(['aeid', 'spid'])
    mc4 = grouped.agg(concentration_unlogged=('logc', lambda x: list(10 ** x)), response=('resp', list)).reset_index()
    mc4 = mc4.head(config["test"]) if config["test"] and config["aeid"] == id else mc4
    df = get_mc5_data(id)
    nested_mc4 = get_nested_mc4(df, config["fit_strategy"], parallelize=True, n_jobs=-1)  # heavy lifting
    print(elapsed(start))
    df = tcpl_load_data(lvl=6, fld='aeid', ids=id)  # hitcall data
    # d = {str(m5id): group for m5id, group in df.groupby("m4id")}
    return mc4, nested_mc4, df


def update():
    mc4, nesetd_mc4, hit_data = fetch_data(st.session_state.aeid)  # needs input for unique caching
    check_reset()
    hitcall_rows = hit_data[hit_data["hit_param"] == 'hitcall'].reset_index()

    if st.session_state.option == "hitcall desc":
        hitcall_rows_sorted = hitcall_rows.sort_values(by="hit_val", ascending=False)
    elif st.session_state.option == "hitcall asc":
        hitcall_rows_sorted = hitcall_rows.sort_values(by="hit_val", ascending=True)
    else:
        hitcall_rows_sorted = hitcall_rows

    m4ids = hitcall_rows_sorted["m4id"]
    m4ids = m4ids.values.tolist()

    st.session_state.m4id = m4ids[st.session_state.spid_row]

    spid = get_spid(mc4, nested_mc4, st.session_state.trigger)
    if spid is None:
        return

    conc, d, fitparams, resp = get_row_data(hit_data, mc4, nested_mc4)

    fig = go.Figure()

    add_title(fig, d)

    fig.add_trace(
        go.Scatter(x=np.log10(conc), y=resp, mode='markers', legendgroup="Response", legendgrouptitle_text="Repsonse",
                   marker=dict(color="black", symbol="x-thin-open", size=10),
                   name="response", showlegend=False))

    return fig, add_curves(conc, fig, fitparams, d)


def add_title(fig, d):
    try:
        casn, chnm, dsstox_substance_id = get_chem_info(st.session_state.spid)
    except Exception as e:
        print(e)
        return
    fig.update_layout(height=600, width=1400)
    # fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    qstring = f"SELECT * FROM assay_component_endpoint WHERE aeid = {st.session_state.aeid};"
    assay_component_endpoint = tcpl_query(qstring).iloc[0]
    normalized_data_type = assay_component_endpoint["normalized_data_type"]
    assay_component_endpoint_name = assay_component_endpoint["assay_component_endpoint_name"]
    assay_component_endpoint_desc = assay_component_endpoint["assay_component_endpoint_desc"]

    with st.expander("Details"):
        st.write(f"spid: {st.session_state.spid}")
        st.write(f"Chemical: {chnm if chnm else 'N/A'}")
        link = f"[{dsstox_substance_id}](https://comptox.epa.gov/dashboard/chemical/details/{dsstox_substance_id})" if dsstox_substance_id else "N/A"
        st.write(f"DSSTOX Substance ID: {link}")
        st.write(f"CASN: {casn if casn else 'N/A'}")
        st.write(f"Assay Endpoint: {assay_component_endpoint_name}")
        st.write(f"{assay_component_endpoint_desc}")

    fig.update_layout(
        title=f"Assay Endpoint: <i>{assay_component_endpoint_name}</i><br>Chemical: <i>{chnm if chnm else 'N/A'}</i><br>Best Model Fit: <i>{d['modl']}</i>, hitcall: <i>{round(d['hitcall'], 7)}</i>",
        margin=dict(t=150),
        xaxis_title="log10(Concentration) μM",
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

        min_val = np.min(conc)
        min_val = min_val if d['hitcall'] <= 0 else min(min_val, d['ac50'])

        x = powspace(min_val, np.max(conc), 100, 500)
        y = np.array(get_fit_model(model)(x, *pars[:-1]))
        color = px.colors.qualitative.Bold[m]
        if model != d['modl'] and model != "none":
            fig.add_trace(
                go.Scatter(x=np.log10(x), y=y, opacity=.7, marker=dict(color=color), mode='lines',
                           name=model, line=dict(width=2, dash='dash')))

        elif model == d['modl']:
            fig.add_trace(
                go.Scatter(x=np.log10(x), y=y, legendgroup=model, marker=dict(color=color), mode='lines',
                           name=f"{model} (BEST FIT)", line=dict(width=3)))

            if d['hitcall'] > 0.0:
                # potencies = ["bmd", "acc", "ac1sd", "ac10", "ac20", "ac50", "ac95"]
                potencies = ["acc", "ac50"]
                for p in potencies:
                    if p in d:
                        fig.add_vline(x=np.log10(d[p]), line_color=color, line_width=2,
                                      annotation_position="bottom left",
                                      annotation_text=f"{p}", layer="below")

                # efficacies = ["top", "bmr"]
                efficacies = ["top"]
                for e in efficacies:
                    if e in d:
                        fig.add_hline(y=d[e], line_color=color, line_width=2,
                                      annotation_position="bottom left",
                                      annotation_text=f"{e}", layer="below")

        else:  # model == "none"
            pass

    cutoff = d['coff']
    fig.add_hline(y=cutoff, line_color="LightSkyBlue")

    fig.add_hrect(
        y0=-cutoff,
        y1=cutoff,
        fillcolor='LightSkyBlue',
        opacity=0.4,
        layer='below',
        line=dict(width=0),
        annotation_text="efficacy cutoff", annotation_position="top left",
    )

    fig.update_layout(legend=dict(groupclick="toggleitem"))
    return pars_dict


def get_row_data(hit_data, mc4, nested_mc4):
    m4id = st.session_state.m4id
    df = hit_data.loc[hit_data["m4id"] == m4id]
    d = {}
    d["modl"] = df["modl"].iloc[0] # recompute?
    d["coff"] = df["coff"].iloc[0] # new table?
    d.update(pd.Series(df.hit_val.values, index=df.hit_param).to_dict())
    fitparams = nested_mc4.loc[nested_mc4["m4id"] == m4id]["params"].iloc[0]
    qstring = f"SELECT * FROM mc4_ WHERE m4id = {m4id};"
    mc4_row = tcpl_query(qstring)
    mc4_row_spid = mc4_row["spid"].iloc[0]
    conc = mc4[mc4["spid"] == mc4_row_spid]['concentration_unlogged'].iloc[0]
    resp = np.array(mc4[mc4["spid"] == mc4_row_spid]['response'].iloc[0])
    return conc, d, fitparams, resp


def get_chem_info(spid):
    try:
        chid = tcpl_query(query=f"SELECT chid FROM sample WHERE spid = '{str(spid)}';").iloc[0]['chid']
    except:
        try:
            chid = tcpl_query(query=f"SELECT chid FROM chemical WHERE chnm = '{str(spid)}';").iloc[0]['chid']
        except:
            try:
                chid = tcpl_query(query=f"SELECT chid FROM chemical WHERE chnm LIKE '%{str(spid)}%';").iloc[0]['chid']
            except Exception as e:
                print(f"Error on spid {spid}: {e}")
                return None, None, None

    chem = tcpl_query(query=f"SELECT * FROM chemical WHERE chid = {str(chid)};").iloc[0]
    casn = chem['casn']
    chnm = chem['chnm']
    dsstox_substance_id = chem['dsstox_substance_id']
    return casn, chnm, dsstox_substance_id


def get_spid(mc4, nested_mc4, trigger):
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
    check_reset()
    st.session_state.direction = direction
    st.session_state.trigger = "new_sample"


def check_reset():
    if "spid_row" not in st.session_state:
        reset_spid_row()
    if "direction" not in st.session_state:
        st.session_state.direction = "stay"
    if "trigger" not in st.session_state:
        st.session_state.trigger = "new_sample"
    if "spid" not in st.session_state:
        st.session_state.spid = ""
    if "m4id" not in st.session_state:
        st.session_state.m4id = ""
    if "option" not in st.session_state:
        st.session_state.option = "hitcall desc"


def reset_spid_row():
    print("reset to start sample")
    st.session_state.spid_row = 0


def filter_spid():
    st.session_state.trigger = "spid"


st.set_page_config(page_title="Curve surfer", page_icon="✅", layout="wide",
                   )

config = load_config()["pytcpl"]
# st.title("Curve surfer")
st.session_state.aeid = int(
    st.number_input(label="Enter assay id (aeid)", value=config['aeid'], on_change=reset_spid_row))
st.session_state.spid = st.text_input(label="Filter sample id (spid)", on_change=filter_spid)
col1, col2, col3 = st.columns(3)
with col1:
    st.button("Previous sample", on_click=load_new_sample, args=("previous",))
with col2:
    st.button("Next sample", on_click=load_new_sample, args=("next",))
with col3:
    st.session_state.option = st.selectbox(
        "Sort by",
        ("hitcall desc", "hitcall asc", "random"),
        on_change=reset_spid_row,
    )

try:
    fig, pars_dict = update()
    st.plotly_chart(fig)
    # st.json(pars_dict)
except Exception as e:
    print(e)
    st.write("No data found")
