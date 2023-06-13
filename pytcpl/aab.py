import numpy as np
import plotly.graph_objects as go

# Create sample data
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Calculate the maximum and minimum marker values for each trace
max_y1 = np.max(y1)
min_y1 = np.min(y1)
max_y2 = np.max(y2)
min_y2 = np.min(y2)

# Create a Plotly figure
fig = go.Figure()

# Add scatter traces with legendgroup attribute
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines+markers', name='Trace 1', legendgroup='group1'))
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name='Trace 2', legendgroup='group2'))

# Add custom JavaScript callback for legend item click
fig.update_layout(
    legend=dict(itemclick="toggle"),
    updatemenus=[dict(active=-1, buttons=list([
        dict(label="Show All",
             method="update",
             args=[{"visible": [True, True]},
                   {"title": "All Traces"}]),
        dict(label="Trace 1",
             method="update",
             args=[{"visible": [True, False]},
                   {"title": "Trace 1", "annotations": [
                       {"text": f"Max: {max_y1:.2f}", "x": 0.5, "y": max_y1, "showarrow": False},
                       {"text": f"Min: {min_y1:.2f}", "x": 0.5, "y": min_y1, "showarrow": False}
                   ]}]),
        dict(label="Trace 2",
             method="update",
             args=[{"visible": [False, True]},
                   {"title": "Trace 2", "annotations": [
                       {"text": f"Max: {max_y2:.2f}", "x": 0.5, "y": max_y2, "showarrow": False},
                       {"text": f"Min: {min_y2:.2f}", "x": 0.5, "y": min_y2, "showarrow": False}
                   ]}])
    ]))],
)

# Show the plot
fig.show()
