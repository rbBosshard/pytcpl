import os
import sys
import traceback

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
pd.options.mode.chained_assignment = None  # default='warn'

from src.app.curve_surfer_helper import check_reset, trigger, filter_spid, update, get_assay_and_sample_info, set_config_app
from src.pipeline.pipeline_helper import load_config, init_aeid


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

    check_reset()
    with st.sidebar:
        st.header(title + "🏄")
        aeid_value = 762
        init_aeid(7)
        st.session_state.aeid = int(st.number_input(label="Input assay endpoint ID (AEID)", value=aeid_value))
        col1, col2 = st.columns(2)
        with col1:
            st.button(":arrow_left: Previous", on_click=trigger, args=("prev",))
        with col2:
            st.button("Next :arrow_right:", on_click=trigger, args=("next",))
        st.session_state.sort_by = st.selectbox("Sort By", ["hitcall", "ac50", "actop"], on_change=trigger,
                                                args=("sort_by",))
        st.session_state.asc = st.selectbox("Ascending", (False, True), on_change=trigger, args=("asc",))
        with st.form("Select hitcall range"):
            st.session_state.hitcall_slider = st.slider("Select hitcall range", 0.0, 1.0, (0.0, 1.0))
            submitted = st.form_submit_button("Submit", on_click=trigger, args=("hitcall_slider",))
            placeholder_hitcall_slider = st.empty()
        with st.form("Input assay endpoint ID (SPID)"):
            st.session_state.spid = st.text_input(label="Input sample ID (SPID)")
            submitted = st.form_submit_button("Submit", on_click=filter_spid)

    fig, pars_dict = update()

    placeholder_hitcall_slider.write(f"{st.session_state.length} series in filter")

    height = 710
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, height=height)

    assay_component_endpoint_desc = get_assay_and_sample_info()
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
