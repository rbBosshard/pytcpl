import pandas as pd
import streamlit as st

from app_helper import check_reset, trigger, filter_spid, update, get_assay_and_sample_info
from pipeline_helper import load_config

# Run command `streamlit run pytcpl/app.py`
# Ensure: test = 0 in config.yaml
pd.options.mode.chained_assignment = None  # default='warn'

title = "PyTCPL Curve Surfer"
def main():
    config, _ = load_config()
    st.set_page_config(page_title=title, page_icon="☣️", layout='wide')
    check_reset()
    with st.sidebar:
        st.header(title + "🏄")
        st.session_state.aeid = int(st.number_input(label="Input assay endpoint ID (AEID)", value=int(config['aeid'])))
        col1, col2 = st.columns(2)
        with col1:
            st.button(":arrow_left: Previous", on_click=trigger, args=("prev",))
        with col2:
            st.button("Next :arrow_right:", on_click=trigger, args=("next",))
        st.session_state.sort_by = st.selectbox("Sort By", ["hitcall", "ac50", "actop"], on_change=trigger, args=("sort_by",))
        st.session_state.asc = st.selectbox("Ascending", (True, False), on_change=trigger, args=("asc",))
        with st.form("Select hitcall range"):
            st.session_state.hitcall_slider = st.slider("Select hitcall range", 0.0, 1.0, (0.0, 1.0))
            submitted = st.form_submit_button("Submit", on_click=trigger, args=("hitcall_slider",))
            placeholder_hitcall_slider = st.empty()
        with st.form("Input assay endpoint ID (SPID)"):
            st.session_state.spid = st.text_input(label="Input sample ID (SPID)")
            submitted = st.form_submit_button("Submit", on_click=filter_spid)

    fig, pars_dict = update()
    if fig is None:
        st.error('No data found', icon="🚨")

    placeholder_hitcall_slider.write(f"{st.session_state.length} series in filter")
    assay_component_endpoint_desc = get_assay_and_sample_info()

    height = 600
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, height=height)
    st.caption(f"ac50: {st.session_state.series['ac50']:.2f} | actop: {st.session_state.series['actop']:.2f} | acc: {st.session_state.series['acc']:.2f} ")
    
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