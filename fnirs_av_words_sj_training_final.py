# Import packages
import gc
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as op
import time
import pandas as pd
import glob
import csv
import mne
from mne.preprocessing.nirs import tddr
from nilearn.glm.first_level import make_first_level_design_matrix  
from mne_nirs.channels import get_long_channels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from scipy import signal
from scipy.stats import ttest_rel, zscore
import mne_nirs

# Open plots in new window
mne.viz.set_browser_backend('matplotlib')

# String to append to output file names
output_suffix = "final"

# Create the subject to group mapping dictionary
subject_group_mapping = pd.read_csv('../../subject_group_mapping.csv')
subject_group_mapping = subject_group_mapping.dropna(subset=['Subject'])  # Drop rows where 'Subject' is NA
subject_group_mapping['Subject'] = subject_group_mapping['Subject'].astype(int)  # Convert 'Subject' to integer
subject_to_group = dict(zip(subject_group_mapping['Subject'], subject_group_mapping['Group']))
subjects = subject_group_mapping['Subject'].astype(str).tolist()

sfreq = 4.807692
conditions = ('A', 'V', 'AV', 'W')
groups = ('trained', 'control')
days = ('1', '3')
runs = (1, 2)

exp_name = 'av'
duration = 1.8
design = 'event'
plot_subject = '205'
plot_day = 1
plot_run = 1
filt_kwargs = dict(l_freq=0.01, h_freq=0.2) 
n_jobs = 4  # for GLM

# SET FOLDER LOCATIONS
raw_path = '../../data'
proc_path = '../../processed'
results_path = '../../results'
subjects_dir = '../../subjects'
behavior_results_path = '../../fnirs-behavior-results'
behavior_file = '../../behavior_diff_data.csv'

os.makedirs(proc_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)
os.makedirs(subjects_dir, exist_ok=True)
# mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # Only need to run once

use = None
all_sci = list()
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

colors = dict(
    A='#4477AA',  # blue
    AV='#CCBB44',  # yellow
    V='#EE7733',  # orange
    W='#AA3377',  # purple
)

# Prep making bad channels report
bad_channels_filename = op.join(results_path, f'bad_channels_report_{output_suffix}.csv')
with open(bad_channels_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Subject', 'Day', 'Run', 'Percent Bad'])

def normalize_channel_names(channels_set):
    return {name.split()[0] for name in channels_set}

def add_bad_channel_entry(subject, day, run, percentage_bad):
    with open(bad_channels_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([subject, day, run, f'{percentage_bad:.2f}%'])

def preprocess_fnirs_data(raw_intensity, proc_path, base):
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity, verbose='error')

    # Identify bad channels
    peaks = np.ptp(raw_od.get_data('fnirs'), axis=-1)
    flat_names = [raw_od.ch_names[f].split(' ')[0] for f in np.where(peaks < 0.001)[0]]
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    sci_mask = (sci < 0.25)
    got = np.where(sci_mask)[0]
    percentage_bad = (len(got) / len(raw_od.ch_names)) * 100
    print(f'    Run {base}')

    # Assign bads
    assert raw_od.info['bads'] == []
    bads = set(raw_od.ch_names[pick] for pick in got)
    bads = bads | set(ch_name for ch_name in raw_od.ch_names if ch_name.split(' ')[0] in flat_names)
    bads = sorted(bads)

    # Further preprocessing
    raw_tddr = tddr(raw_od)
    raw_tddr_bp = raw_tddr.copy().filter(**filt_kwargs)
    raw_tddr_bp.info['bads'] = bads
    picks = mne.pick_types(raw_tddr_bp.info, fnirs=True)
    peaks = np.ptp(raw_tddr_bp.get_data(picks), axis=-1)
    assert (peaks > 1e-5).all()
    raw_tddr_bp.info['bads'] = [] 
    
    raw_h = mne.preprocessing.nirs.beer_lambert_law(raw_tddr_bp, 6.)
    raw_h = get_long_channels(raw_h)

    # Normalize and verify bad channels
    h_bads = [ch_name for ch_name in raw_h.ch_names if ch_name.split(' ')[0] in set(bad.split(' ')[0] for bad in bads)]
    set_bads = set(bads)
    set_h_bads = set(h_bads)
    normalized_bads = normalize_channel_names(set_bads)
    normalized_h_bads = normalize_channel_names(set_h_bads)
    assert normalized_bads == normalized_h_bads
    raw_h.info['bads'] = h_bads
    raw_h.info._check_consistency()

    # Further verification
    picks = mne.pick_types(raw_h.info, fnirs=True)
    peaks = np.ptp(raw_h.get_data(picks), axis=-1)
    assert (peaks > 1e-9).all()

    # Interpolate bad channels
    raw_h_interp = raw_h.copy().interpolate_bads(reset_bads=True, method=dict(fnirs='nearest'))
    raw_h_interp.save(op.join(proc_path, f'{subject}_{day}_{run:03d}_long_hbo_{output_suffix}_raw.fif'), overwrite=True)
    assert len(raw_h.ch_names) == len(raw_h_interp.ch_names)

    return raw_h_interp, percentage_bad, bads

# Sanity check for subjects
subjects_check = {int(subject) for subject in subjects}
subject_to_group_check = set(subject_to_group.keys())
if subjects_check == subject_to_group_check:
    print("N=" + str(len(subjects)))
    del subjects_check, subject_to_group_check
else:
    print("Error loading subject info") 

##############################################################################
# Remove subjects with over 30% bad channels on average across days and runs

subjects_to_remove = ['202', '203', '204', '206', '214', '221', '223', '226', '233']

# Initialize counters for each group
removed_trained = 0
removed_control = 0

remaining_trained = 0
remaining_control = 0

# Count and remove the subjects
for subject in subjects_to_remove:
    subject_int = int(subject)  # Convert to integer for dictionary key comparison
    if subject_int in subject_to_group:
        # Increment the appropriate counter based on the group of the subject
        if subject_to_group[subject_int] == "trained":
            removed_trained += 1
        elif subject_to_group[subject_int] == "control":
            removed_control += 1
        # Remove the subject from the dictionary
        subject_to_group.pop(subject_int, None)

# Update the subjects list after counting the removed subjects
subjects = [subject for subject in subjects if subject not in subjects_to_remove]

for group in subject_to_group.values():
    if group == "trained":
        remaining_trained += 1
    elif group == "control":
        remaining_control += 1

# Output the results
print(f'Removed {removed_trained} trained subjects.')
print(f'Removed {removed_control} control subjects.')
print(f'Remaining trained subjects: {remaining_trained}')
print(f'Remaining control subjects: {remaining_control}')

###############################################################################
# Load participant data

subjects = ['205'] #testing

for subject in subjects:
    for day in days:
        for run in runs:
            group = subject_to_group.get(int(subject), "unknown")
            root1 = f'Day{day}'
            root2 = f'{subject}_{day}'
            root3 = f'*-*-*_{run:03d}'
            fname_base = op.join(raw_path, root1, root2, root3)
            fname = glob.glob(fname_base)
            base = f'{subject}_{day}_{run:03d}'
            base_pr = base.ljust(20)
            raw_intensity = mne.io.read_raw_nirx(fname[0])
            raw_h_long, percentage_bad_long, bads_long = preprocess_fnirs_data(raw_intensity, proc_path, base + '_long')
            add_bad_channel_entry(subject, day, run, percentage_bad_long)
            del raw_intensity, raw_h_long, percentage_bad_long, bads_long
            gc.collect()  #

fname2 = op.join(proc_path, f'{plot_subject}_{plot_day}_{plot_run:03d}_long_hbo_{output_suffix}_raw.fif')
use = mne.io.read_raw_fif(fname2, preload=True)
events, _ = mne.events_from_annotations(use)
ch_names = [ch_name.rstrip(' hbo') for ch_name in use.ch_names]
info = use.info

###############################################################################

# Plot the design matrix and some raw traces
def make_design(raw_h_long, design, subject=None, run=None, day=None, group=None):
    annotations_to_remove = raw_h_long.annotations.description == '255.0'
    raw_h_long.annotations.delete(annotations_to_remove)
    events, _ = mne.events_from_annotations(raw_h_long)
    rows_to_remove = events[:, -1] == 1
    events = events[~rows_to_remove]
    
    # Mis-codings
    if len(events) == 101:
        events = events[1:]
        
    n_times = len(raw_h_long.times)
    stim = np.zeros((n_times, 4))
    events[:, 2] -= 1
    assert len(events) == 100, len(events)
    want = [0] + [25] * 4
    count = np.bincount(events[:, 2])
    assert np.array_equal(count, want), count
    assert events.shape == (100, 3), events.shape
    if design == 'block':
        events = events[0::5]
        duration = 20.
        assert np.array_equal(np.bincount(events[:, 2]), [0] + [5] * 4)
    else:
        # assert design == 'event'
        assert len(events) == 100
        duration = 1.8
        assert events.shape == (100, 3)
        events_r = events[:, 2].reshape(20, 5)
        assert (events_r == events_r[:, :1]).all()
        del events_r
        
    idx = (events[:, [0, 2]] - [0, 1]).T
    assert np.in1d(idx[1], np.arange(len(conditions))).all()
    stim[tuple(idx)] = 1
    
    n_block = int(np.ceil(duration * sfreq))
    stim = signal.fftconvolve(stim, np.ones((n_block, 1)), axes=0)[:n_times]
    dm_events = pd.DataFrame({
        'trial_type': [conditions[ii] for ii in idx[1]],
        'onset': idx[0] / raw_h_long.info['sfreq'],
        'duration': n_block / raw_h_long.info['sfreq']})
    dm = make_first_level_design_matrix(
        raw_h_long.times, dm_events, hrf_model='glover',
        drift_model='polynomial', drift_order=0)
        
    return stim, dm, events

fig, axes = plt.subplots(2, 1, figsize=(6., 3), constrained_layout=True)
# Design
ax = axes[0]
raw_h = use
stim, dm, _ = make_design(raw_h, design)

##### NEW
# Specify the figure size and limits per chromophore
fig, axes = plt.subplots(nrows=1, ncols=len(all_evokeds), figsize=(17, 5))
lims = dict(hbo=[-5, 12], hbr=[-5, 12])

for pick, color in zip(["hbo", "hbr"], ["r", "b"]):
    for idx, evoked in enumerate(all_evokeds):
        plot_compare_evokeds(
            {evoked: all_evokeds[evoked]},
            combine="mean",
            picks=pick,
            axes=axes[idx],
            show=False,
            colors=[color],
            legend=False,
            ylim=lims,
            ci=0.95,
            show_sensors=idx == 2,
        )
        axes[idx].set_title(f"{evoked}")
axes[0].legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])


# Scatter plot of values
sns.catplot(
    x="Condition",
    y="Value",
    hue="ID",
    data=df.query("Chroma == 'hbo'"),
    errorbar=None,
    palette="muted",
    height=4,
    s=10,
)

from nilearn.plotting import plot_design_matrix
fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
fig = plot_design_matrix(dm, ax=ax1)
##### NEW

for ci, condition in enumerate(conditions):
    color = colors[condition]
    ax.fill_between(
        raw_h.times, stim[:, ci], 0, edgecolor='none', facecolor='k',
        alpha=0.5)
    model = dm[conditions[ci]].to_numpy()
    ax.plot(raw_h.times, model, ls='-', lw=1, color=color)
    x = raw_h.times[np.where(model > 0)[0][0]]
    ax.text(
        x + 10, 1.1, condition, color=color, fontweight='bold', ha='center')
ax.set(ylabel='Modeled\noxyHb', xlabel='', xlim=raw_h.times[[0, -1]])

# HbO/HbR
ax = axes[1]
picks = [pi for pi, ch_name in enumerate(raw_h.ch_names)
         if 'S7_D19' in ch_name]
colors = dict(hbo='r', hbr='b')
ylim = np.array([-2, 2])
for pi, pick in enumerate(picks):
    color = colors[raw_h.ch_names[pick][-3:]]
    data = raw_h.get_data(pick)[0] * 1e6
    val = np.ptp(data)
    assert val > 0.01
    ax.plot(raw_h.times, data, color=color, lw=1.)
ax.set(ylim=ylim, xlabel='Time (s)', ylabel='μM',
       xlim=raw_h.times[[0, -1]])
del raw_h
for ax in axes:
    for key in ('top', 'right'):
        ax.spines[key].set_visible(False)
fig.savefig(op.join(results_path, f'figure_1_{output_suffix}.png'))


###############################################################################
# Run GLM analysis and epoching

subj_cha_list = []
for subject in subjects:
    group = subject_to_group.get(int(subject), "unknown")
    for day in days:
        for run in runs:
            fname_long = op.join(proc_path, f'{subject}_{day}_{run:03d}_long_hbo_{output_suffix}_raw.fif')
            raw_h_long = mne.io.read_raw_fif(fname_long)
            _, dm, _ = make_design(raw_h_long, design, subject, run, day, group)
            glm_est = mne_nirs.statistics.run_glm(
                raw_h_long, dm, noise_model='ols', n_jobs=n_jobs)
            cha = glm_est.to_dataframe()
            cha['subject'] = subject
            cha['run'] = run
            cha['day'] = day
            cha['group'] = group
            subj_cha_list.append(cha)
            del raw_h_long, dm, glm_est, cha
            gc.collect()  #
        print(f'***Finished processing subject {subject} day {day}.')

df_cha = pd.concat(subj_cha_list, ignore_index=True)
df_cha.reset_index(drop=True, inplace=True)

##### NEW
glm_est.scatter()
glm_est.plot_topo()
glm_est.copy().surface_projection(condition="A", view="dorsal", chroma="hbo")

##### NEW

###############################################################################

# Block averages
event_id = {condition: ci for ci, condition in enumerate(conditions, 1)}
evokeds = {condition: dict() for condition in conditions}
for day in days:
    for subject in subjects:
        fname = op.join(proc_path, f'{subject}_{day}_{exp_name}_{output_suffix}-ave.fif')
        tmin, tmax = -2, 38
        baseline = (None, 0)
        t0 = time.time()
        print(f'Creating block average for {subject} day {day}... ', end='')
        raws = list()
        events = list()
        for run in runs:
            fname2 = op.join(proc_path, f'{subject}_{day}_{run:03d}_long_hbo_{output_suffix}_raw.fif')
            raw_h = mne.io.read_raw_fif(fname2)
            events.append(make_design(raw_h, None, 'block', subject, run)[2])
            raws.append(raw_h)
        bads = sorted(set(sum((r.info['bads'] for r in raws), [])))
        for r in raws:
            r.info['bads'] = bads
        raw_h, events = mne.concatenate_raws(raws, events_list=events)
        epochs = mne.Epochs(raw_h, events, event_id, tmin=tmin, tmax=tmax,
                            baseline=baseline)
        this_ev = [epochs[condition].average() for condition in conditions]
        assert all(ev.nave > 0 for ev in this_ev)
        mne.write_evokeds(fname, this_ev, overwrite=True)
        print(f'{time.time() - t0:0.1f} sec')
        for condition in conditions:
            evokeds[condition][subject] = mne.read_evokeds(fname, condition)
        print(f'Done for {group} {subject} day {day} run {run:03d}... ', end='')
        del raws, events, raw_h, epochs, this_ev
        gc.collect()  #

# Mark bad channels
bad = dict()
bb = dict()

for day in days:
    for subject in subjects:
        for run in runs:
            fname2 = op.join(proc_path, f'{subject}_{day}_{run:03d}_long_hbo_{output_suffix}_raw.fif')
            this_info = mne.io.read_info(fname2)
            bad_channels = [idx - 1 for idx in sorted(
                this_info['ch_names'].index(bad) + 1 for bad in this_info['bads'])]
            valid_indices = np.arange(len(use.ch_names))
            bb = [b for b in bad_channels if b in valid_indices]
            bad[(subject, run, day)] = bb
        assert np.in1d(bad[(subject, run, day)], np.arange(len(use.ch_names))).all()

bad_combo = dict()
for day in days:
    for (subject, run, day), bb in bad.items():
        bad_combo[subject] = sorted(set(bad_combo.get(subject, [])) | set(bb))
bad = bad_combo

start = len(df_cha)
n_drop = 0
for day in days:
    for (subject, run, day), bb in bad.items():
        if not len(bb):
            continue
        drop_names = [use.ch_names[b] for b in bb]
        is_subject = (df_cha['subject'] == subject)
        is_day = (df_cha['day'] == day)
        drop = df_cha.index[
            is_subject &
            is_day &
            np.in1d(df_cha['ch_name'], drop_names)]
        n_drop += len(drop)
        if len(drop):
            print(f'Dropping {len(drop)} for {subject} day {day}')
            df_cha.drop(drop, inplace=True)
end = len(df_cha)
assert n_drop == start - end, (n_drop, start - end)

# Combine runs by averaging
sorts = ['subject', 'ch_name', 'Chroma', 'Condition', 'group', 'day', 'run']
df_cha.sort_values(sorts, inplace=True)
theta = np.array(df_cha['theta']).reshape(-1, len(runs)).mean(-1)
df_cha.drop(
    [col for col in df_cha.columns if col not in sorts[:-1]], axis='columns',
    inplace=True)
df_cha.reset_index(drop=True, inplace=True)
df_cha = df_cha[::len(runs)]
df_cha.reset_index(drop=True, inplace=True)
df_cha['theta'] = theta


###############################################################################
# CALCULATE HbDIFF

# Load the data
df_cha_nolabels = df_cha.copy()
df_cha_nolabels['ch_name'] = df_cha_nolabels['ch_name'].str[:-4]

# Separate HbO and HbR
df_hbo = df_cha_nolabels[df_cha_nolabels['Chroma'].str.endswith('hbo')].set_index(['subject', 'Condition', 'group', 'day', 'ch_name']).sort_index()
df_hbr = df_cha_nolabels[df_cha_nolabels['Chroma'].str.endswith('hbr')].set_index(['subject', 'Condition', 'group', 'day', 'ch_name']).sort_index()

# Compute the difference
df_cha_diff_list = []
for ch_name in df_hbo.index.get_level_values('ch_name').unique():
    # Get aligned indices
    df_hbo_ch = df_hbo.loc[(slice(None), slice(None), slice(None), slice(None), ch_name), :].sort_index()
    df_hbr_ch = df_hbr.loc[(slice(None), slice(None), slice(None), slice(None), ch_name), :].sort_index()
    
    # Ensure df_hbo_ch and df_hbr_ch have the same length
    common_index = df_hbo_ch.index.intersection(df_hbr_ch.index)
    df_hbo_ch = df_hbo_ch.loc[common_index]
    df_hbr_ch = df_hbr_ch.loc[common_index]
    
    # Calculate the difference
    df_diff = df_hbo_ch[['theta']].sub(df_hbr_ch[['theta']])
    
    # Align df_cha_ch with df_diff
    df_cha_ch = df_hbo_ch.reset_index()
    df_cha_ch['theta'] = df_diff.values
    df_cha_ch['Chroma'] = 'hbdiff'
    df_cha_ch['ch_name'] = df_cha_ch['ch_name'] + ' hbdiff'
    
    if not df_cha_ch.empty:
        df_cha_diff_list.append(df_cha_ch)

df_cha_diff_concat = pd.concat(df_cha_diff_list, ignore_index=True)

# Concatenate original df_cha with df_cha_diff_concat
df_final = pd.concat([df_cha, df_cha_diff_concat], ignore_index=True)
df_final.to_csv(op.join(results_path, f'df_combined_final_cha_{output_suffix}.csv'), index=False)

###############################################################################
# Analyze changes!

# Load the datasets
behavior_df = pd.read_csv(behavior_file)
behavior_df['subject'] = behavior_df['subject'].astype(str)

theta_df = df_final.copy()
theta_df['subject'] = theta_df['subject'].astype(str)
theta_df_filtered = theta_df[theta_df['day'] == '1']
theta_df_filtered['ch_name'] = theta_df_filtered['ch_name'].str.split(' ').str[0]

# Get the unique conditions
conditions = ['A', 'AV', 'V']
response_vars = ['AO_WR', 'AV_WR', 'TBW']
chromas = ['hbo', 'hbr', 'hbdiff']

def perform_analysis(group, output_suffix):
    # Initialize a list to store significant models
    significant_models = []
    
    # Initialize a dictionary to store p-values and model data by condition, response variable, and chroma
    all_p_values = {condition: {response_var: {chroma: [] for chroma in chromas} for response_var in response_vars} for condition in conditions}
    all_model_data = {condition: {response_var: {chroma: [] for chroma in chromas} for response_var in response_vars} for condition in conditions}

    # Track maximum R-squared value
    max_r_squared = 0
    max_r_squared_model = None

    for chroma in chromas:
        # Filter theta_df for the specific chroma
        theta_dataset = theta_df_filtered[theta_df_filtered['Chroma'] == chroma]
        theta_dataset['ch_name'] = theta_dataset['ch_name'].str.split(' ').str[0]

        # Collect all p-values and model data for each response variable and condition
        for condition in conditions:
            for response_var in response_vars:
                # Filter the dataset for the current condition and day 1
                theta_df_day1 = theta_dataset[theta_dataset['Condition'] == condition]
                
                # Pivot the theta_df to have channel names as columns
                theta_pivot = theta_df_day1.pivot_table(index=['subject', 'group', 'Condition'], columns='ch_name', values='theta').reset_index()
                theta_pivot['subject'] = theta_pivot['subject'].astype(str)  # Ensure subject is string

                # Merge the datasets based on 'subject' and 'group'
                merged_df = pd.merge(theta_pivot, behavior_df[['subject', 'group', 'TBW', 'AO_WR', 'AV_WR', 'VO_WR', 'age', 'AO_WR_1', 'AV_WR_1', 'VO_WR_1', 'TBW_1']], on=['subject', 'group'])
                channels = theta_df_day1['ch_name'].unique()  # list of all channel names
                
                for channel in channels:
                    df = merged_df[[channel, response_var, 'group', 'TBW_1']].dropna()  # drop rows with missing values
                    model = smf.ols(f"{response_var} ~ {channel}", df[df['group'] == group]).fit()
                    r_sq = model.rsquared
                    p_value_channel = model.pvalues[channel]  # p-value for the channel
                    all_p_values[condition][response_var][chroma].append(p_value_channel)
                    all_model_data[condition][response_var][chroma].append((condition, channel, response_var, model, r_sq, p_value_channel, df, chroma))

                    # Check if this model has the highest R-squared value
                    if r_sq > max_r_squared:
                        max_r_squared = r_sq
                        max_r_squared_model = {
                            'Condition': condition,
                            'Channel': channel,
                            'Response Variable': response_var,
                            'R-squared': r_sq,
                            'Model Summary': model.summary().as_text(),
                            'Chroma': chroma
                        }
                del model, r_sq, p_value_channel, df
                gc.collect()  #

    # Apply FDR correction for each condition, response variable, and chroma
    for response_var in response_vars:
        for chroma in chromas:
            for condition in conditions:
                p_values = all_p_values[condition][response_var][chroma]
                model_data = all_model_data[condition][response_var][chroma]
                
                if p_values:
                    # Apply FDR correction
                    rejected, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
                    
                    # Find the minimum value in p_values_corrected
                    max_r_squared_in_group = max([data[4] for data in model_data]) if model_data else 0
                    print(f"The maximum R-squared value for condition {condition}, response variable {response_var}, and chroma {chroma} is: {max_r_squared_in_group}")
                    
                    # Filter results based on FDR corrected p-values
                    for (condition, channel, response_var, model, r_sq, p_value, df, chroma), p_val_corr, reject in zip(model_data, p_values_corrected, rejected):
                        if p_value < 0.05:
                            significant_models.append({
                                'Condition': condition,
                                'Channel': channel,
                                'Response Variable': response_var,
                                'R-squared': r_sq,
                                'P-value': p_value,
                                'P-value Corrected': p_val_corr,
                                'Model Summary': model.summary().as_text(),
                                'Chroma': chroma
                            })
                        if p_val_corr < 0.05:
                            plt.figure(figsize=(8, 6))

                            # Plot trained data
                            trained_df = df[df['group'] == 'trained']
                            if not trained_df.empty:
                                trained_model = smf.ols(f"{response_var} ~ {channel}", trained_df).fit()
                                sns.scatterplot(x=trained_df[channel], y=trained_df[response_var], label='Trained', color='#92b6f0', s=100)
                                sns.lineplot(x=trained_df[channel], y=trained_model.predict(trained_df), color='#92b6f0', linewidth=2)

                            # Plot control data
                            control_df = df[df['group'] == 'control']
                            if not control_df.empty:
                                control_model = smf.ols(f"{response_var} ~ {channel}", control_df).fit()
                                sns.scatterplot(x=control_df[channel], y=control_df[response_var], label='Control', color='gray', s=100)
                                sns.lineplot(x=control_df[channel], y=control_model.predict(control_df), color='gray', linewidth=2)
                            
                            xlabel = ('[HbO] on Day 1' if chroma == 'hbo' else
                                      '[HbR] on Day 1' if chroma == 'hbr' else
                                      '[HbDiff] on Day 1' if chroma == 'hbdiff' else
                                      f'{chroma.upper()} on Day 1')
                            ylabel = ('Change in Auditory Word Recognition' if response_var == 'AO_WR' else
                                      'Change in Audiovisual Word Recognition' if response_var == 'AV_WR' else
                                      'Change in Visual Word Recognition' if response_var == 'VO_WR' else
                                      f'Change in {response_var}')
                            plt.legend(loc='upper right') 
                            plt.xlabel(xlabel, fontsize=16)
                            plt.ylabel(ylabel, fontsize=16)
                            plt.title(f'{ylabel} vs.\nCortical Response to {condition} Speech ({channel})', fontsize=16)
                            plt.savefig(op.join(results_path, f'{condition}_{channel}_{response_var}_plot_{chroma}_{output_suffix}.png'))
                            plt.close()

    # Convert the list of significant models to a DataFrame and save as CSV
    if significant_models:
        significant_models_df = pd.DataFrame(significant_models).sort_values(by='R-squared', ascending=False)
        significant_models_df.to_csv(op.join(results_path, f'{group}_fnirs-behavior-models_{output_suffix}.csv'), index=False)
        # Print the maximum R-squared value and its corresponding model details
        print(f"Maximum R-squared value: {max_r_squared}")
        print(f"Model details:\nCondition: {max_r_squared_model['Condition']}, Channel: {max_r_squared_model['Channel']}, Response Variable: {max_r_squared_model['Response Variable']}")
        print(max_r_squared_model['Model Summary'])
    else:
        print("No significant models found.")
    return max_r_squared, max_r_squared_model

# Perform analysis for the trained group
max_r_squared_trained, max_r_squared_model_trained = perform_analysis('trained', 'trained')

# Perform analysis for the control group
max_r_squared_control, max_r_squared_model_control = perform_analysis('control', 'control')

# Perform analysis for both groups combined
max_r_squared_combined, max_r_squared_model_combined = perform_analysis('combined', 'combined')


###############################################################################
# RUN T-TEST AND PLOT THE CHANGES OVER TIME

# Load the final combined dataframe
df_final = pd.read_csv(op.join(results_path, f'df_combined_final_cha_{output_suffix}.csv'))
conditions = ['A', 'AV', 'V']
groups = ['trained', 'control']
chromas = ['hbo', 'hbr', 'hbdiff']

df_final['ch_name'] = df_final['ch_name'].str.split(' ').str[0]
fname = op.join(proc_path, f'205_1_001_long_hbo_final_raw.fif')
use = mne.io.read_raw_fif(fname, preload=True)
use.load_data()
new_ch_names = {}
seen_names = set()
for ch_name in use.info['ch_names']:
    new_name = ch_name.split(' ')[0]
    if new_name not in seen_names:
        new_ch_names[ch_name] = new_name
        seen_names.add(new_name)

use.rename_channels(new_ch_names)
use = use.pick_channels(list(new_ch_names.values()))

# Perform analysis for each group and Chroma
for group in groups:
    for chroma in chromas:
        # Prepare figure for composite plots
        fig, axes = plt.subplots(1, len(conditions), figsize=(15, 5))
        for idx, condition in enumerate(conditions):
            # Filter data for day 1 and day 3 for the specific group and Chroma
            df_day1 = df_final.query(f"group == '{group}' and Chroma == '{chroma}' and day == 1").copy()
            df_day3 = df_final.query(f"group == '{group}' and Chroma == '{chroma}' and day == 3").copy()

            # Ensure ch_name and Condition columns are of the same data type
            df_day1['ch_name'] = df_day1['ch_name'].astype(str)
            df_day1['Condition'] = df_day1['Condition'].astype(str)
            df_day3['ch_name'] = df_day3['ch_name'].astype(str)
            df_day3['Condition'] = df_day3['Condition'].astype(str)

            # Set index and sort
            df_day1 = df_day1.set_index(['subject', 'group', 'ch_name', 'Condition', 'Chroma']).sort_index()
            df_day3 = df_day3.set_index(['subject', 'group', 'ch_name', 'Condition', 'Chroma']).sort_index()

            # Merge dataframes to align day 1 and day 3 data
            df_merged = df_day1[['theta']].rename(columns={'theta': 'theta_day1'}).merge(
                df_day3[['theta']].rename(columns={'theta': 'theta_day3'}),
                left_index=True, right_index=True)

            # Calculate the difference and z-score
            df_merged['theta_diff'] = df_merged['theta_day3'] - df_merged['theta_day1']
            df_merged['z'] = zscore(df_merged['theta_diff'])

            # Perform paired t-test for each channel and condition across subjects
            t_stats = []
            p_values = []
            ch_names = []
            condition_list = []

            for (ch_name, cond), group_df in df_merged.groupby(['ch_name', 'Condition']):
                t_stat, p_value = ttest_rel(group_df['theta_day1'], group_df['theta_day3'])
                t_stats.append(t_stat)
                p_values.append(p_value)
                ch_names.append(ch_name)
                condition_list.append(cond)

            # Create a results DataFrame
            results_df = pd.DataFrame({
                'ch_name': ch_names,
                'Condition': condition_list,
                't_stat': t_stats,
                'p_value': p_values
            })

            # Combine with z-score data
            z_scores = df_merged.groupby(['ch_name', 'Condition'])['z'].mean().reset_index()
            
            # Ensure consistent data types before merging
            z_scores['ch_name'] = z_scores['ch_name'].astype(str)
            z_scores['Condition'] = z_scores['Condition'].astype(str)
            results_df['ch_name'] = results_df['ch_name'].astype(str)
            results_df['Condition'] = results_df['Condition'].astype(str)

            results_df = results_df.merge(z_scores, on=['ch_name', 'Condition'])

            # Correct for multiple comparisons
            print(f'Correcting for {len(results_df["p_value"])} comparisons using FDR')
            _, results_df['P_fdr'] = mne.stats.fdr_correction(results_df['p_value'], method='indep')
            results_df['SIG'] = results_df['P_fdr'] < 0.05
            
            # Print significant results
            significant_results = results_df.loc[results_df.SIG == True]
            print(significant_results)

            # Prepare data for brain plots
            ch_of_interest = use.pick_channels([ch_name for ch_name in use.info['ch_names']])
            info_of_interest = ch_of_interest.info

            zs = {}
            condition_data = results_df[(results_df['Condition'] == condition)]
                        
            zs[condition] = np.array([
                condition_data.loc[(condition_data['ch_name'] == ch_name), 'z'].values[0]
                if not condition_data.loc[(condition_data['ch_name'] == ch_name), 'z'].empty and condition_data.loc[(condition_data['ch_name'] == ch_name), 'p_value'].values[0] < 0.05
                else 0
                for ch_name in condition_data['ch_name']
            ])
            
            # Create an EvokedArray for each condition
            evoked = mne.EvokedArray(zs[condition][:, np.newaxis], info_of_interest)
            picks = np.arange(len(info_of_interest['ch_names']))

            stc = mne.stc_near_sensors(
                evoked, trans='fsaverage', subject='fsaverage', mode='weighted',
                distance=0.02, project=True, picks=picks, subjects_dir=subjects_dir)

            # Plot the brain and capture the image in-memory
            brain = stc.plot(hemi='both', views=['lat', 'frontal', 'lat'],
                             cortex='low_contrast', time_viewer=False, show_traces=False,
                             surface='pial', smoothing_steps=0, size=(1200, 400),
                             clim=dict(kind='value', pos_lims=[0, 0.75, 1.5]),
                             colormap='RdBu_r', view_layout='horizontal',
                             colorbar=(0, 1), time_label='', background='w',
                             brain_kwargs=dict(units='m'),
                             add_data_kwargs=dict(colorbar_kwargs=dict(
                                 title_font_size=16, label_font_size=12, n_labels=5,
                                 title='z score')), subjects_dir=subjects_dir)
            brain.show_view('lat', hemi='lh', row=0, col=0)
            brain.show_view(azimuth=270, elevation=90, row=0, col=1)
            brain.show_view('lat', hemi='rh', row=0, col=2)

            # Capture the plot as an image in memory
            screenshot = brain.screenshot(time_viewer=False)
            brain.close()

            # Display the image in the composite figure
            ax = axes[idx]
            ax.imshow(screenshot)
            ax.axis('off')
            ax.set_title(f'{group.capitalize()} - Condition {condition} ({chroma})', fontsize=18)

            del df_day1, df_day3, df_merged, t_stats, p_values, ch_names, condition_list, results_df, z_scores
            gc.collect()  #

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(op.join(results_path, f'{group}_{chroma}_composite_brain_plots_insig.png'))
        plt.show()
        plt.close(fig)

        del fig, axes, ch_of_interest, info_of_interest, evoked, stc, brain, screenshot
        gc.collect()  #



###############################################################################
# Plot sensor locations


df_r2 = pd.read_csv(op.join(results_path, 'trained_fnirs-behavior-models_trained.csv'))
#df_r2 = pd.read_csv(op.join(results_path, 'control_fnirs-behavior-models_control.csv'))

df_filtered = df_r2[(df_r2['P-value Corrected'] < 0.05)]
ch_names = df_filtered['Channel'].values 
info = use.copy().pick_types(fnirs='hbo', exclude=())
info_picked = info.pick_channels(ch_names)

fig = mne.viz.plot_sensors(info_picked.info, kind='topomap', show_names=True, pointsize=100, linewidth=0]
