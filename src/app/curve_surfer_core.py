import streamlit as st

from src.app.curve_surfer_helper import get_trigger, on_filter_assay_endpoints, on_iterate_assay_endpoint, update_aeid, \
    refresh_data, on_hitcall_slider, update_df_length, check_sort, on_iterate_compounds, update_df_index, \
    on_select_compound, update_spid, get_series, init_figure, add_curves


def update(slider):
    """
    Update the app state and visualization.

    This function updates the app's state and visualization based on user interactions and triggers. It handles loading data,
    filtering, sorting, and updating the Plotly figure.

    Returns:
        tuple: A tuple containing the updated Plotly figure and a dictionary of fitted curve parameters.

    """
    # Get trigger
    trigger = get_trigger()

    # Filter assay endpoints
    on_filter_assay_endpoints(trigger)

    # Iterate assay endpoint
    on_iterate_assay_endpoint(trigger)

    # Update assay endpoint id
    update_aeid()

    # Refresh data
    if not st.session_state.focus_on_compound or trigger == 'hitcall_slider':
        refresh_data(trigger)

    # Filter compounds on hitcall range
    on_hitcall_slider(trigger)

    # Update dataframe length
    update_df_length()

    # Sort dataframe
    check_sort(trigger)

    # Iterate compound
    on_iterate_compounds(trigger)

    # Update dataframe index
    update_df_index()

    # Select current compound
    on_select_compound(trigger, slider)

    # Update current specific
    update_spid()

    # Get corresponding series from dataframe
    get_series()

    # Refresh data when focused on compound
    if st.session_state.focus_on_compound_submitted or (trigger == 'hitcall_slider' and st.session_state.focus_on_compound):
        refresh_data(trigger)

    fig, col2 = init_figure()
    pars_dict, height = add_curves(fig, col2)
    return fig, pars_dict, height
