import os
import sys
import traceback

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
pd.options.mode.chained_assignment = None  # default='warn'

from src.app.curve_surfer_helper import check_reset, set_trigger, get_assay_and_sample_info, \
    set_config_app, subset_assay_info_columns, get_series, load_output_compound
from src.app.curve_surfer_core import update
from src.pipeline.pipeline_helper import load_config


def main():
    """
    The main function for the PyTCPL Curve Surfer application.

    This function initializes the Streamlit app layout and configuration settings. It sets up user interface elements,
    handles user inputs and interactions, triggers updates based on user actions, and displays visualizations and
    information.

    Note:
    The function utilizes various Streamlit components and session_state attributes for UI and interaction management.

    """
    title = "PyTCPL Curve Surfer"
    st.set_page_config(page_title=title, page_icon="☣️", layout='wide')
    config, _ = load_config()
    set_config_app(config)

    with st.spinner('Wait for it...'):
        check_reset()

    with st.sidebar:
        st.header(title + "🏄")

        st.divider()

        st.subheader("Assay endpoint settings")

        st.session_state.assay_info_column = st.selectbox("Filter on:", subset_assay_info_columns, on_change=set_trigger, args=("assay_info_column",))
        st.session_state.assay_info_selected_fields = [st.session_state.assay_info_distinct_values[st.session_state.assay_info_column][0]]
        with st.form("Filter assay endpoints"):
            st.session_state.assay_info_selected_fields = st.multiselect("Multiselect fields:",  
                st.session_state.assay_info_distinct_values[st.session_state.assay_info_column], 
                default=st.session_state.assay_info_selected_fields, placeholder="ALL")
            submitted = st.form_submit_button("Submit!", on_click=set_trigger, args=("filter_assay_endpoints",))
            placeholder_assay_info = st.empty()
            placeholder_assay_info.write(f"{len(st.session_state.aeids)} assay endpoints in filter")

        st.button(":arrow_up_small: Next assay", on_click=set_trigger, args=("next_assay_endpoint",))
        st.button(":arrow_down_small: Previous assay", on_click=set_trigger, args=("prev_assay_endpoint",))

        st.divider()

        st.subheader("Compounds settings")
        with st.form("Input assay endpoint ID (SPID)"):
            spids = st.session_state.df["spid"]
            dtxsids = st.session_state.df["dsstox_substance_id"]
            ids = pd.concat([dtxsids, spids])
            id = st.selectbox(label="Input DTXSID or SPID", options=ids)
            focus_on_compound = st.checkbox(label=f"Focus on this compound", value=st.session_state.focus_on_compound)
            st.session_state.focus_on_compound_submitted = st.form_submit_button("Submit!", on_click=set_trigger, args=("select_compound",))
            if st.session_state.focus_on_compound_submitted:
                st.session_state.focus_on_compound = focus_on_compound
                st.session_state.compound_id = id

        if not st.session_state.focus_on_compound:
            st.button(":arrow_up_small: Next compound", on_click=set_trigger, args=("next_compound",))
            st.button(":arrow_down_small: Previous compound", on_click=set_trigger, args=("prev_compound",))
            st.session_state.sort_by = st.selectbox("Sort By", ["hitcall", "ac50", "actop"], on_change=set_trigger,
                                                    args=("sort_by",))
            st.session_state.asc = st.selectbox("Ascending", (False, True), on_change=set_trigger, args=("asc",))
            with st.form("Select hitcall range"):
                st.session_state.hitcall_slider = st.slider("Select hitcall range", 0.0, 1.0, (0.0, 1.0))
                submitted = st.form_submit_button("Submit!", on_click=set_trigger, args=("hitcall_slider",))
                placeholder_hitcall_slider = st.empty()
                placeholder_hitcall_slider.write(f"{st.session_state.df_length} series in filter")        
    

    compound_info_container = st.empty()
    fig, pars_dict = update()
    placeholder_assay_info.write(f"{len(st.session_state.aeids)} assay endpoints in filter")

    if not st.session_state.focus_on_compound:
        placeholder_hitcall_slider.write(f"{st.session_state.df_length} series in filter")

    compound_info, assay_component_endpoint_desc = get_assay_and_sample_info()
    compound_info_container.write(compound_info, unsafe_allow_html=True)
   


    height = 720
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, height=height)

    
    # Todo: Provide curve fit model functions
    with st.expander("Curve fit parameters"):
        st.json(pars_dict)
    with st.expander("Assay component endpoint description "):
        st.write(f"{assay_component_endpoint_desc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        st.error(e, icon="🚨")
        st.error(traceback.print_exc(), icon="🚨")
