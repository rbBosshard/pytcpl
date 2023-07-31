import json

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fit_models import get_fit_model
from pipeline_helper import get_assay_info, print_
from pipeline_helper import load_config
from query_db import query_db


# Run command `streamlit run pytcpl/app.py`
# Ensure: test = 0 in config.yaml


def powspace(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


# Load data initially or when id changes, and cache the result
@st.cache_data
def fetch_data(aeid):  # aeid parameter used to handle correct caching
    print_(f"Fetch data from DB with assay ID {aeid}...")
    check_reset()
    qstring = f"SELECT * FROM output WHERE aeid = {st.session_state.aeid};"
    dat = query_db(query=qstring)
    print_(f"Data fetched: {dat.shape[0]} rows.")
    return dat


def update():
    df = fetch_data(st.session_state.aeid)  # needs input for unique caching
    check_reset()

    if st.session_state.option == "hitcall desc":
        df = df.sort_values(by="hitcall", ascending=False)
    elif st.session_state.option == "hitcall asc":
        df = df.sort_values(by="hitcall", ascending=True)
    else:
        pass

    ids = df["id"]
    ids = ids.values.tolist()

    st.session_state.id = ids[st.session_state.spid_row]

    spid = get_spid(df, st.session_state.trigger)
    if spid is None:
        return

    row = df[df["id"] == st.session_state.id]

    row.loc[:, 'concentration_unlogged'] = row['concentration_unlogged'].apply(json.loads)
    row.loc[:, 'response'] = row['response'].apply(json.loads)
    row.loc[:, 'fitparams'] = row['fitparams'].apply(json.loads)
    row = row.to_dict()
    # Remove all nested dictionaries using dictionary comprehension
    row = {key: value for key, nested_dict in row.items() for value in nested_dict.values()}

    fig = go.Figure()

    add_title(fig, row)

    fig.add_trace(
        go.Scatter(x=np.log10(row["concentration_unlogged"]), y=row["response"], mode='markers', legendgroup="Response",
                   legendgrouptitle_text="Repsonse",
                   marker=dict(color="black", symbol="x-thin-open", size=10),
                   name="response", showlegend=False))

    return fig, add_curves(fig, row)


def add_title(fig, row):
    try:
        casn, chnm, dsstox_substance_id = get_chem_info(st.session_state.spid)
    except Exception as e:
        print(e)
        return
    fig.update_layout(height=600, width=1400)
    # fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    assay_infos = get_assay_info(st.session_state.aeid)
    normalized_data_type = assay_infos["normalized_data_type"]
    assay_component_endpoint_name = assay_infos["assay_component_endpoint_name"]
    assay_component_endpoint_desc = assay_infos["assay_component_endpoint_desc"]

    with st.expander("Details"):
        st.write(f"spid: {st.session_state.spid}")
        st.write(f"Chemical: {chnm if chnm else 'N/A'}")
        link = f"[{dsstox_substance_id}](https://comptox.epa.gov/dashboard/chemical/details/{dsstox_substance_id})" if dsstox_substance_id else "N/A"
        st.write(f"DSSTOX Substance ID: {link}")
        st.write(f"CASN: {casn if casn else 'N/A'}")
        st.write(f"Assay Endpoint: {assay_component_endpoint_name}")
        st.write(f"{assay_component_endpoint_desc}")

    fig.update_layout(
        title=(f"Assay Endpoint: <i>{assay_component_endpoint_name}</i>"
               f"<br>Chemical: <i>{chnm if chnm else 'N/A'}</i><br>"
               f"Best Model Fit: <i>{row['fit_model']}</i>, hitcall: <i>{round(row['hitcall'], 7)}</i>"),
        margin=dict(t=150),
        xaxis_title="log10(Concentration) μM",
        yaxis_title=str(normalized_data_type),
    )


def add_curves(fig, row):
    fitparams = row['fitparams']
    conc = row['concentration_unlogged']
    fit_models = list(fitparams.keys())
    pars_dict = {}
    for m, model in enumerate(fit_models):
        params = fitparams[model]["pars"]
        params = {param: value for param, value in params.items() if param in get_fit_model(model).__code__.co_varnames}

        min_val = np.min(conc)
        # min_val = min_val if row['hitcall'] >= 0 else min(min_val, row['ac50'])

        x = powspace(min_val, np.max(conc), 100, 500)
        y = np.array(get_fit_model(model)(x, **params))
        color = px.colors.qualitative.Bold[m]
        aic = {round(fitparams[model]['aic'], 2)}
        rmse = {round(fitparams[model]['rmse'], 2)}
        try:
            if model != row['fit_model'] and model != "none":
                fig.add_trace(
                    go.Scatter(x=np.log10(x), y=y, opacity=.7, marker=dict(color=color), mode='lines',
                               name=f"{model} {aic} {rmse}", line=dict(width=2, dash='dash')))

            elif model == row['fit_model']:
                fig.add_trace(
                    go.Scatter(x=np.log10(x), y=y, legendgroup=model, marker=dict(color=color), mode='lines',
                               name=f"{model} (BEST FIT) {aic} {rmse}", line=dict(width=3)))

                if row['hitcall'] > 0.0:
                    # potencies = ["bmd", "acc", "ac1sd", "ac10", "ac20", "ac50", "ac95"]
                    potencies = ["acc", "ac50"]
                    for p in potencies:
                        if p in row:
                            fig.add_vline(x=np.log10(row[p]), line_color=color, line_width=2,
                                          annotation_position="bottom left",
                                          annotation_text=f"{p}", layer="below")

                    # efficacies = ["top", "bmr"]
                    efficacies = ["top"]
                    for e in efficacies:
                        if e in row:
                            fig.add_hline(y=row[e], line_color=color, line_width=2,
                                          annotation_position="bottom left",
                                          annotation_text=f"{e}", layer="below")

        except Exception as e:
            print(f"{e}")
            pass
        else:  # model == "none"
            pass

    cutoff = row['cutoff']
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


def get_spid(df, trigger):
    spid = st.session_state.spid
    if spid in df['spid'].values and trigger == "spid":
        st.session_state.spid_row = df[df['spid'] == spid].index[0]
    elif trigger == "new_sample":
        dir = st.session_state.direction
        if dir == "next":
            new_spid_row = st.session_state.spid_row + 1
        elif dir == "previous":
            new_spid_row = st.session_state.spid_row - 1
        else:
            new_spid_row = st.session_state.spid_row

        st.session_state.spid_row = new_spid_row % df.shape[0]
        spid = df.iloc[st.session_state.spid_row]["spid"]
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
    if "id" not in st.session_state:
        st.session_state.id = ""
    if "option" not in st.session_state:
        st.session_state.option = "hitcall desc"


def reset_spid_row():
    st.session_state.spid_row = 0


def filter_spid():
    st.session_state.trigger = "spid"


st.set_page_config(page_title="Curve surfer", page_icon="✅", layout="wide",
                   )

config, _ = load_config()
# st.title("Curve surfer")
st.session_state.aeid = int(
    st.number_input(label="Enter assay id (aeid)", value=int(config['aeid']), on_change=reset_spid_row))
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
