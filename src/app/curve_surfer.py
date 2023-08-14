import os
import sys

import pandas as pd
import streamlit as st

# Run command `streamlit run pytcpl/curve_surfer.py`
# Ensure: enable_data_subsetting: 0 raw config/config.yaml
# Add the src dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
pd.options.mode.chained_assignment = None  # default='warn'

from surf_utils import check_reset, trigger, filter_spid, update, get_assay_and_sample_info, set_config_app
from src.utils.pipeline_helper import load_config


def main():
    title = "PyTCPL Curve Surfer"
    st.set_page_config(page_title=title, page_icon="☣️", layout='wide')
    config, _ = load_config()
    set_config_app(config)
    
    check_reset()
    with st.sidebar:
        st.header(title + "🏄")
        st.session_state.aeid = int(st.number_input(label="Input assay endpoint ID (AEID)", value=int(config['aeid'])))
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
    
    st.markdown(
    """
    <style>
    .stApp {
        margin-top: -70px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    height = 710
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, height=height)

    assay_component_endpoint_desc = get_assay_and_sample_info()
    # Todo: Provide curve fit model functions
    with st.expander("Curve fit parameters"):
        st.json(pars_dict)
    with st.expander("Assay component endpoint description "):
        st.write(f"{assay_component_endpoint_desc}")


def test():
    import streamlit as st
    from st_files_connection import FilesConnection

    # Create connection object and retrieve file contents.
    # Specify input format is a csv and to cache the result for 600 seconds.
    conn = st.experimental_connection('s3', type=FilesConnection)
    df = conn.read("datapytcpl/5.parquet.gzip", input_format="parquet", ttl=600)

    # Print results.
    for row in df.itertuples():
        st.write(f"{row.Owner} has a :{row.Pet}:")


if __name__ == "__main__":
    try:
        main()
        test()
    except Exception as e:
        print(e)
        st.error(e, icon="🚨")
