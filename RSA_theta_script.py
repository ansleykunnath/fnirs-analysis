import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind
import numpy as np
from statsmodels.stats.multitest import multipletests

# Load the data
df = pd.read_csv('df_cha.csv')

# Remove all rows with 'hbr'
df = df[df['Chroma'] != 'hbr']

# Separate data into 'trained' and 'control' groups
#trained_df_1 = df[(df['group'] == 'trained') & (df['day'] == 1) & (df['Condition'].isin(['A', 'V', 'AV']))]

trained_df_1 = df[(df['group'] == 'trained') & (df['day'] == 1)]
control_df_1 = df[(df['group'] == 'control') & (df['day'] == 1)]
trained_df_3 = df[(df['group'] == 'trained') & (df['day'] == 3)]
control_df_3 = df[(df['group'] == 'control') & (df['day'] == 3)]
trained_df = df[(df['group'] == 'trained')]
control_df = df[(df['group'] == 'control')]


# Function to create RDMs for each channel
def create_rdms_for_each_channel(group_df):
    rdms = {}
    channels = group_df['ch_name'].unique()

    for channel in channels:
        # Filter the DataFrame for the current channel
        channel_df = group_df[group_df['ch_name'] == channel]
        # Create condition vectors (participants as rows, conditions as columns)
        condition_vectors = channel_df.pivot_table(index='subject', columns=['day','Condition'], values='theta', aggfunc='mean').fillna(0)
#        condition_vectors = channel_df.pivot_table(index='subject', columns=['Condition'], values='theta', aggfunc='mean').fillna(0)
        # Calculate RDM
        rdm = pd.DataFrame(squareform(pdist(condition_vectors.T, 'euclidean')), index=condition_vectors.columns, columns=condition_vectors.columns)
        rdms[channel] = rdm

    return rdms

# Create RDMs for each channel for 'trained' and 'control' groups
trained_rdms_pre = create_rdms_for_each_channel(trained_df_1)
control_rdms_pre = create_rdms_for_each_channel(control_df_1)
trained_rdms_post = create_rdms_for_each_channel(trained_df_3)
control_rdms_post = create_rdms_for_each_channel(control_df_3)
trained_rdms = create_rdms_for_each_channel(trained_df)
control_rdms = create_rdms_for_each_channel(control_df)

# # Function to get the lower triangle of the matrix
def get_lower_triangle(matrix):
    return matrix.values[np.tril_indices_from(matrix, k=-1)]

# # Identify common channels between 'trained' and 'control' groups
common_channels = set(trained_rdms_pre.keys()).intersection(control_rdms_pre.keys())

# # Initialize lists for storing p-values and channel names
p_values = []
channel_names = []

# # Perform t-tests for common channels
for channel in common_channels:
    trained_lower_triangle = get_lower_triangle(trained_rdms[channel])
    control_lower_triangle = get_lower_triangle(control_rdms[channel])
    _, p_value = ttest_ind(trained_lower_triangle, control_lower_triangle, nan_policy='omit')
    p_values.append(p_value)
    channel_names.append(channel)

# # Apply FDR correction
reject, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# # Collect significant channels
significant_channels = [(channel, p_val) for channel, p_val, is_reject in zip(channel_names, p_values_corrected, reject) if is_reject]

# # Convert significant channels to a DataFrame
significant_channels_df = pd.DataFrame(significant_channels, columns=['Channel', 'Corrected P-Value'])

# # Save the significant channels to a CSV file
significant_channels_csv_path = './significant_channels_fdr_corrected.csv'
significant_channels_df.to_csv(significant_channels_csv_path, index=False)

# Output the path to the saved CSV
print(f"Significant channels saved to: {significant_channels_csv_path}")

# Save RDMs to a Python (.py) file
def save_rdms_to_py_file(trained_rdms_pre, trained_rdms_post, control_rdms_pre, control_rdms_post, trained_rdms, control_rdms, filepath):
    with open(filepath, 'w') as file:
        file.write("trained_rdms_pre = {\n")
        for channel, rdm in trained_rdms_pre.items():
            file.write(f"    '{channel}': {rdm.to_dict('list')},\n")
        file.write("}\n\n")
        
        file.write("control_rdms_pre = {\n")
        for channel, rdm in control_rdms_pre.items():
            file.write(f"    '{channel}': {rdm.to_dict('list')},\n")
        file.write("}\n\n")

        file.write("trained_rdms_post = {\n")
        for channel, rdm in trained_rdms_post.items():
            file.write(f"    '{channel}': {rdm.to_dict('list')},\n")
        file.write("}\n\n")

        file.write("control_rdms_post = {\n")
        for channel, rdm in control_rdms_post.items():
            file.write(f"    '{channel}': {rdm.to_dict('list')},\n")
        file.write("}\n\n")

        file.write("trained_rdms = {\n")
        for channel, rdm in trained_rdms.items():
            file.write(f"    '{channel}': {rdm.to_dict('list')},\n")
        file.write("}\n\n")

        file.write("control_rdms = {\n")
        for channel, rdm in control_rdms.items():
            file.write(f"    '{channel}': {rdm.to_dict('list')},\n")
        file.write("}\n")

# Specify the path for the .py file to save RDMs
py_file_path = './group_rdms.py'

# Save the RDMs to a Python file
save_rdms_to_py_file(trained_rdms_pre, trained_rdms_post, control_rdms_pre, control_rdms_post, trained_rdms, control_rdms, py_file_path)

# Output the path to the saved Python file
print(f"RDMs saved to: {py_file_path}")
