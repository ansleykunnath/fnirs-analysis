import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from statsmodels.stats.multitest import multipletests
from warnings import simplefilter
from warnings import filterwarnings
filterwarnings('ignore')

# Load CSV files
fncl_conn_df = pd.read_csv("../../fncl_conn_analysis.csv")  # Functional connectivity changes
behavior_diff_df = pd.read_csv("../../behavior_diff_data_only.csv")  # Word recognition changes

# Merge datasets on subject I
merged_df = fncl_conn_df.merge(behavior_diff_df, on="subject")

# Ensure 'group' and 'subject' are categorical
merged_df["group"] = merged_df["group"].astype("category")
merged_df["subject"] = merged_df["subject"].astype("category")

# Define behavioral measures (outcomes)
behavior_vars = ['AO_WR', 'AV_WR', 'VO_WR']

# Define functional connectivity measures (predictors)
fncl_conn_cols = [col for col in fncl_conn_df.columns if col != "subject"]

# Rename columns to remove spaces and hyphens
merged_df = merged_df.rename(columns=lambda x: x.replace(" ", "_").replace("-", "_"))
fncl_conn_cols = [col.replace(" ", "_").replace("-", "_") for col in fncl_conn_cols]
behavior_vars = [col.replace(" ", "_").replace("-", "_") for col in behavior_vars]

# Run mixed effects models
interaction_results = []

for behavior in behavior_vars:
    for conn in fncl_conn_cols:
        try:
            formula = f"{conn} ~ -1 + {behavior}:group"
            model = smf.mixedlm(formula, merged_df, groups=merged_df["subject"])
            result = model.fit(method='powell')

            # Extract p-values and parameters
            model_results = pd.DataFrame({"Estimate": result.params, "P-Value": result.pvalues})

            # Remove '[constant]' terms if present
            model_results = model_results[~model_results.index.str.contains(r'\[constant\]', regex=True)]

            # Store results
            for index, row in model_results.iterrows():
                interaction_results.append([behavior, conn, index, row["Estimate"], row["P-Value"]])

        except Exception as e:
            print(f"Error for {behavior} and {conn}: {e}")
            interaction_results.append([behavior, conn, None, None, None])

# Convert results to DataFrame
interaction_results_df = pd.DataFrame(
    interaction_results,
    columns=['Behavioral Measure', 'Functional Connectivity Pair', 'Interaction Term', 'Estimate', 'P-Value']
)

# Compute Z-Scores BEFORE FDR filtering
interaction_results_df["Z-Score"] = interaction_results_df.groupby("Behavioral Measure")["Estimate"].transform(
    lambda x: zscore(x, nan_policy='omit') if x.notna().sum() > 1 else np.nan
)
# Apply FDR correction separately for each behavioral outcome
separate_fdr_results = []

for behavior in behavior_vars:
    subset_df = interaction_results_df[interaction_results_df["Behavioral Measure"] == behavior].dropna(subset=["P-Value"])
    
    if not subset_df.empty:
        p_values = subset_df["P-Value"].astype(float)
        rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        # Store FDR-corrected values
        subset_df["FDR-Corrected P-Value"] = pvals_corrected
        subset_df["Significant (FDR < 0.05)"] = rejected
        separate_fdr_results.append(subset_df)

# Combine results
if separate_fdr_results:
    final_fdr_df = pd.concat(separate_fdr_results)

    # Keep only significant results
    significant_results = final_fdr_df[final_fdr_df["Significant (FDR < 0.05)"] == True]

    print("Significant results found after FDR correction.")
#    print(significant_results)
else:
    print("No significant interactions found after FDR correction.")
#    print(separate_fdr_results)

##### PLOT 1: B-estimates

import matplotlib.pyplot as plt
import seaborn as sns

# Function to categorize connectivity pairs into hemispheres
def categorize_hemisphere(pair):
    left_regions = ["lSTG", "lMTG", "lAG", "lADC", "lVC"]
    right_regions = ["rSTG", "rMTG", "rAG", "rADC", "rVC"]

    parts = pair.split("_to_")
    if parts[0] in left_regions and parts[1] in left_regions:
        return "Left Hemisphere"
    elif parts[0] in right_regions and parts[1] in right_regions:
        return "Right Hemisphere"
    elif parts[0] in left_regions and parts[1] in right_regions:
        return "Left to Right"
    elif parts[0] in right_regions and parts[1] in left_regions:
        return "Right to Left"
    else:
        return "Other"
label_map = {
    "AO_WR": "Δ Auditory WR",
    "AV_WR": "Δ Audiovisual WR",
    "VO_WR": "Δ Visual WR"
}

# Apply hemisphere categorization
significant_results["Hemisphere"] = significant_results["Functional Connectivity Pair"].apply(categorize_hemisphere)
significant_results["Behavioral Measure"] = significant_results["Behavioral Measure"].replace(label_map)

# Reorder categories to ensure proper layout
hemisphere_order = ["Left Hemisphere", "Right Hemisphere", "Left to Right", "Right to Left"]
significant_results = significant_results[significant_results["Hemisphere"].isin(hemisphere_order)]

heatmap_data = significant_results.pivot("Functional Connectivity Pair", "Behavioral Measure", "Estimate")

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0)
plt.title("Significant Estimates of AV Connectivity Changes")
plt.xlabel("")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.show()


#### PLOT 2: z-scores

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Function to categorize connectivity pairs into hemispheres
def categorize_hemisphere(pair):
    left_regions = ["lSTG", "lMTG", "lAG", "lADC", "lVC"]
    right_regions = ["rSTG", "rMTG", "rAG", "rADC", "rVC"]

    parts = pair.split("_to_")
    if parts[0] in left_regions and parts[1] in left_regions:
        return "Left Hemisphere"
    elif parts[0] in right_regions and parts[1] in right_regions:
        return "Right Hemisphere"
    elif parts[0] in left_regions and parts[1] in right_regions:
        return "Left to Right"
    elif parts[0] in right_regions and parts[1] in left_regions:
        return "Right to Left"
    else:
        return "Other"

# Apply hemisphere categorization
significant_results["Hemisphere"] = significant_results["Functional Connectivity Pair"].apply(categorize_hemisphere)

# Reorder categories to ensure proper layout
hemisphere_order = ["Left Hemisphere", "Right Hemisphere", "Left to Right", "Right to Left"]
significant_results = significant_results[significant_results["Hemisphere"].isin(hemisphere_order)]

# Rename x-axis labels for clarity
label_map = {
    "AO_WR": "Δ Auditory WR",
    "AV_WR": "Δ Audiovisual WR",
    "VO_WR": "Δ Visual WR"
}
significant_results["Behavioral Measure"] = significant_results["Behavioral Measure"].replace(label_map)

# Create pivot table for heatmap
heatmap_data = significant_results.pivot("Functional Connectivity Pair", "Behavioral Measure", "Z-Score")

# Plot heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={'label': 'Z-Score'})
plt.title("Z-Scored Significant Estimates (AV Condition)")
plt.xlabel("")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.show()