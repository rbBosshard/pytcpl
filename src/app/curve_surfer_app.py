import os
import sys
import traceback

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
pd.options.mode.chained_assignment = None  # default='warn'

from src.app.curve_surfer_helper import set_trigger, get_compound_info, \
    set_config_app, subset_assay_info_columns, get_assay_info
from src.app.curve_surfer_core import update
from src.app.curve_surfer_session_state import check_reset
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
    title = "PYTCPL CURVE SURFER"
    st.set_page_config(page_title=title, page_icon="☣️", layout='wide')
    config, _ = load_config()
    set_config_app(config)

    with st.spinner('Wait for it...'):
        check_reset(config)

    with st.sidebar:
        st.title(title + "🏄")

        st.header("Assay endpoint settings")

        st.session_state.assay_info_column = st.selectbox("Filter on:", subset_assay_info_columns, on_change=set_trigger, args=("assay_info_column",))
        # st.session_state.assay_info_selected_fields = [st.session_state.assay_info_distinct_values[st.session_state.assay_info_column][0]]
        with st.expander("Filter assay endpoints", expanded=True):
            with st.form("Filter assay endpoints"):
                assay_info_selected_fields = st.multiselect("Multiselect fields:",  
                    st.session_state.assay_info_distinct_values[st.session_state.assay_info_column], 
                    placeholder="ALL", label_visibility="collapsed")
                submitted = st.form_submit_button("Submit!")
                if submitted:
                    st.session_state.assay_info_selected_fields = assay_info_selected_fields
                    set_trigger("filter_assay_endpoints")
                placeholder_assay_info = st.empty()
                placeholder_assay_info.write(f"{len(st.session_state.aeids)} assay endpoints in filter")

        # st.write("Assay endpoint")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Previous", on_click=set_trigger, args=("prev_assay_endpoint",), use_container_width=True, key="prev_assay_endpoint")
        with col2:
            st.button("Next", on_click=set_trigger, args=("next_assay_endpoint",), use_container_width=True, key="next_assay_endpoint")

        st.header("Compounds settings")
       
        with st.expander("Select specific DTXSID", expanded=True):
            with st.form("Select specific DTXSID"):
                dtxsids = st.session_state.df["dsstox_substance_id"]
                dtxsids = dtxsids.drop_duplicates().dropna().tolist()
                st.session_state.compound_ids = dtxsids
                compound_selectbox = st.empty()
                id = compound_selectbox.selectbox(label="Select specific DTXSID", options=dtxsids, label_visibility="collapsed")
                focus_on_compound = st.checkbox(label=f"Focus on only this compound")
                st.session_state.focus_on_compound_submitted = st.form_submit_button("Submit!", on_click=set_trigger, args=("select_compound",))
                if st.session_state.focus_on_compound_submitted:
                    if focus_on_compound != st.session_state.focus_on_compound:
                        set_trigger("select_compound_focus_changed")
                    st.session_state.focus_on_compound = focus_on_compound
                    st.session_state.compound_id = id
                

        if not st.session_state.focus_on_compound:
            col1, col2 = st.columns(2)
            with col1:
                st.button("Previous", on_click=set_trigger, args=("prev_compound",), use_container_width=True)
            with col2:
                st.button("Next", on_click=set_trigger, args=("next_compound",), use_container_width=True)

        with st.expander("Sort", expanded=False):
            st.session_state.sort_by = st.selectbox("Sort By", ["hitcall", "hitcall_c", "acc", "ac50", "actop", "cytotox_acc", "cytotox_prob"], on_change=set_trigger, args=("sort_by",))
            st.session_state.asc = st.selectbox("Ascending", (False, True), on_change=set_trigger, args=("asc",))
        with st.expander("Select hitcall range", expanded=False):
            with st.form("Select hitcall range"):
                slider = st.empty()
                st.session_state.hitcall_slider = slider.slider("Select hitcall range", 0.0, 1.0, (0.0, 1.0),
                                                                 key=st.session_state.slider_iteration, label_visibility="collapsed")
                submitted = st.form_submit_button("Submit!", on_click=set_trigger, args=("hitcall_slider",))
                placeholder_hitcall_slider = st.empty()


    # with st.expander("Assay endpoint infos", expanded=False):
    #     assay_info_container = st.empty()
    # with st.expander("Compound infos", expanded=False):
    #     compound_info_container = st.empty()
    assay_info_container = st.empty()
    compound_info_container = st.empty()

    fig, pars_dict, height = update(slider)

    placeholder_assay_info.write(f"{len(st.session_state.aeids)} assay endpoints in filter")
    placeholder_hitcall_slider.write(f"{st.session_state.df_length} series in filter")

    compund_info_df, column_config = get_compound_info()

    assay_info_container.dataframe(get_assay_info(subset=True, transpose=True, replace=True), hide_index=True, use_container_width=True)
    compound_info_container.dataframe(compund_info_df, column_config=column_config, hide_index=True, use_container_width=True)

    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, height=height)




    with st.expander("Curve fit parameters"):
        st.json(pars_dict)
    with st.expander("Assay endpoint infos extensive"):
        st.dataframe(get_assay_info(transpose=True), use_container_width=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback_info = traceback.format_exc()
        print(e)
        st.error(e, icon="🚨")
        st.error(traceback_info)
