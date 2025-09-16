import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')

# Load CSV files
fncl_conn_df = pd.read_csv("../../fncl_conn_analysis.csv")  # Functional connectivity changes
behavior_diff_df = pd.read_csv("../../behavior_diff_data_only.csv")  # Word recognition changes

# Merge datasets on subject ID
merged_df = fncl_conn_df.merge(behavior_diff_df, on="subject")

# Ensure 'group' and 'subject' are categorical
merged_df["group"] = merged_df["group"].astype("category")
merged_df["subject"] = merged_df["subject"].astype("category")

# Define functional connectivity measures (predictors)
fncl_conn_cols = [col for col in fncl_conn_df.columns if col != "subject"]

# Rename columns to remove spaces and hyphens
merged_df = merged_df.rename(columns=lambda x: x.replace(" ", "_").replace("-", "_"))
fncl_conn_cols = [col.replace(" ", "_").replace("-", "_") for col in fncl_conn_cols]

# Run mixed effects models for group-level analysis
group_comparison_results = []

for conn in fncl_conn_cols:
    try:
        # Construct formula with group as a predictor
        formula = f"{conn} ~ group"
        
        # Fit the model at the group level
        model = smf.ols(formula, data=merged_df).fit()  # OLS regression (instead of mixed effects)
        
        # Extract coefficients and p-values
        model_results = pd.DataFrame({"Estimate": model.params, "P-Value": model.pvalues})        
        model_results = model_results[model_results.index == "group[T.trained]"]
        print(model_results)

        # Store results (only for group effect)
        for index, row in model_results.iterrows():
            group_comparison_results.append([conn, index, row["Estimate"], row["P-Value"]])

    except Exception as e:
        print(f"Error for {conn}: {e}")
        group_comparison_results.append([conn, None, None, None])

# Convert results to DataFrame
group_comparison_df = pd.DataFrame(
    group_comparison_results,
    columns=['Functional Connectivity Pair', 'Group Effect', 'Estimate', 'P-Value']
)

# Step 1: Apply FDR correction to the p-values for the group effect
valid_pvals_df = group_comparison_df.dropna(subset=["P-Value"]).copy()

# Apply FDR correction only to valid rows
if not valid_pvals_df.empty:
    p_values = valid_pvals_df["P-Value"].astype(float)
    
    if len(p_values) > 0:  # Ensure there are p-values to correct
        rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        # Assign FDR-corrected p-values back to the dataframe
        valid_pvals_df["FDR-Corrected P-Value"] = pvals_corrected
        valid_pvals_df["Significant (FDR < 0.05)"] = rejected

        # Merge back into the original dataframe
        group_comparison_df = group_comparison_df.merge(
            valid_pvals_df[["Functional Connectivity Pair", "FDR-Corrected P-Value", "Significant (FDR < 0.05)"]],
            on="Functional Connectivity Pair",
            how="left"
        )

        # Keep only significant results (FDR < 0.05)
        significant_group_results = group_comparison_df[group_comparison_df["Significant (FDR < 0.05)"] == True]

        # Display the significant results
        significant_group_results.to_csv("aggregated_group_comparisons.csv", index=False)
        print("Aggregated results saved to 'aggregated_group_comparisons.csv'.")

else:
    print("\nNo valid p-values found for FDR correction.")

# Plot heatmap

# Check if there are aggregated results to plot
if not significant_group_results.empty:
    # Pivot the data to make it suitable for a heatmap (Functional Connectivity Pair vs Group Effect)
    heatmap_data = significant_group_results.pivot("Functional Connectivity Pair", "Group Effect", "Estimate")

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={'label': 'Group-Level Estimate'})
    plt.title("Aggregated Group-Level Significant Functional Connectivity Differences (FDR < 0.05)")
    plt.xlabel("Group Effect")
    plt.ylabel("Functional Connectivity Pair")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

    # Save the heatmap figure
    plt.savefig("group_level_heatmap.png")
else:
    print("\nNo aggregated results to plot.")


# Plot boxplots 

# Step 1: Sort the results by absolute value of the Estimate to get the top 10 largest differences
top_10_significant = significant_group_results.assign(
    abs_estimate=significant_group_results["Estimate"].abs()
).sort_values(by="abs_estimate", ascending=False).head(10)

# Step 2: Create the subplots with a shared y-axis
fig, axes = plt.subplots(2, 5, figsize=(12, 8), sharey=True)  # Share y-axis across all subplots

# Define custom colors for the groups
control_color = "darkgray"
trained_color = "lightblue"

for i, conn in enumerate(top_10_significant["Functional Connectivity Pair"].unique()):
    ax = axes[i//5, i%5]  # Determine the correct subplot position
    sns.boxplot(data=merged_df, x="group", y=conn, color="white", boxprops=dict(facecolor="lightgray", edgecolor="black"), ax=ax)
    sns.stripplot(data=merged_df, x="group", y=conn, color="black", alpha=0.6, ax=ax)

    # Adjust labels and title
    ax.set_title(f"{conn}")
    ax.set_ylabel("Δ Functional Connectivity Estimate")
    ax.set_xlabel("")

    # Set consistent y-axis limits
    ax.set_ylim(0.1, 0.6)

plt.tight_layout()  # Adjust subplots to fit within the figure area
plt.show()
