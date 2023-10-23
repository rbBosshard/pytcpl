from src.app.curve_surfer_helper import CONFIG, get_series, load_assay_endpoint, load_cutoff, reset_df_index, sort
from src.pipeline.pipeline_constants import METADATA_SUBSET_DIR_PATH


import pandas as pd
import streamlit as st


import json
import os

from src.pipeline.pipeline_helper import init_aeid


def check_reset(config):
    """
    Check if the session state needs to be reset.
    """
    if "df_index" not in st.session_state:
        reset_df_index()
    if "last_aeid" not in st.session_state:
        st.session_state.last_aeid = -1
    if "sort_by" not in st.session_state:
        st.session_state.sort_by = "hitcall"
    if "asc" not in st.session_state:
        st.session_state.asc = False
    if "df_length" not in st.session_state:
        st.session_state.df_length = 0
    if "trigger" not in st.session_state:
        st.session_state.trigger = "assay_info"
    if "assay_endpoint_info" not in st.session_state:
        st.session_state.assay_endpoint_info = None
    if "assay_info_df" not in st.session_state:
        st.session_state.assay_info_df = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info{config['file_format']}"))
    if "assay_info_distinct_values" not in st.session_state:
        with open(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info_distinct_values.json"), "r") as json_file:
            st.session_state.assay_info_distinct_values = json.load(json_file)
    if "aeids" not in st.session_state:
        st.session_state.aeids_all = pd.read_parquet(
            os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_sorted{config['file_format']}"))['aeid'].reset_index(drop=True)
        st.session_state.aeids = st.session_state.aeids_all.copy()
    if "aeid" not in st.session_state:
        st.session_state.aeid = st.session_state.aeids.iloc[0]
    if "assay_info" not in st.session_state:
        st.session_state.assay_info = st.session_state.assay_info_df.iloc[0]
    if "aeid_index" not in st.session_state:
        st.session_state.aeid_index = 0
    if "num_assay_endpoints_filtered" not in st.session_state:
        st.session_state.num_assay_endpoints_filtered = len(st.session_state.aeids)
    if "assay_info_column" not in st.session_state:
        st.session_state.assay_info_column = "aeid"
    if "assay_info_selected_fields" not in st.session_state:
        st.session_state.assay_info_selected_fields = []
    if "df" not in st.session_state:
        init_aeid(st.session_state.aeid)
        st.session_state.df = load_assay_endpoint(st.session_state.aeid)
        st.session_state.cutoff_df = load_cutoff(st.session_state.aeid)
        sort()
    if "series" not in st.session_state:
        st.session_state.series = get_series()
    if "compound_id" not in st.session_state:
        st.session_state.compound_id = ''
    if "focus_on_compound" not in st.session_state:
        st.session_state.focus_on_compound = False
    if "focus_on_compound_submitted" not in st.session_state:
        st.session_state.focus_on_compound_submitted = False
    if "init" not in st.session_state:
        st.session_state.init = True
    if "hitcall_slider" not in st.session_state:
        st.session_state.hitcall_slider = (0.0, 1.0)
    if "slider_iteration" not in st.session_state:
        st.session_state.slider_iteration = 0
    if "selectbox_iteration" not in st.session_state:
        st.session_state.selectbox_iteration = -1
    if "compound_ids" not in st.session_state:
        C = []
