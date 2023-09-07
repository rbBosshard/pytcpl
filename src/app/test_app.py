import pandas as pd
import streamlit as st

# Create a sample DataFrame with links
data = {'Name': ['John', 'Jane', 'Mike'],
        'Email': ['john@example.com', 'jane@example.com', 'mike@example.com']}
df = pd.DataFrame(data)

# Add a new column with HTML links
df['Profile'] = df['Name'].apply(lambda name: f'<a href="https://example.com/profile/{name}">Profile</a>')

# Display the DataFrame with links
st.table(df, unsafe_allow_html=True)