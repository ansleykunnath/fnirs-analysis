import PyQt5
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from collections import defaultdict
from mne.viz import plot_compare_evokeds
import os
from os import path as op
import time
import warnings
from scipy import signal, stats
import pooch
import pandas as pd
with warnings.catch_warnings(record=True):
    warnings.simplefilter('ignore', FutureWarning)
    from nilearn.glm.first_level import \
        make_first_level_design_matrix, compute_regressor  # noqa
import statsmodels.formula.api as smf
import mne_nirs.preprocessing
import mne_nirs.statistics
import mne_nirs.utils
import mne_nirs.statistics
import mne
from mne.preprocessing.nirs import tddr
import glob


warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

subjects = ('201 202 203 204 205 206 207 208 209 212 213 214 215 216 217 218 219 221').split()
# Mapping of subjects to groups
subject_to_group = {
    201: "trained",
    202: "control",
    203: "trained",
    204: "control",
    205: "control",
    206: "control",
    207: "trained",
    208: "control",
    209: "control",
    212: "trained",
    213: "trained",
    214: "trained",
    215: "control",
    216: "trained",
    217: "control",
    218: "control",
    219: "trained",
    221: "trained",
}

sfreq = 4.807692
conditions = ('A', 'V', 'AV', 'W')
groups = ('trained','control')
days = ('1', '3')
runs = (1, 2)

condition_colors = dict(  # https://personal.sron.nl/~pault/data/colourschemes.pdf
    A='#4477AA',  # sblue
    AV='#CCBB44',  # yellow
    V='#EE7733',  # orange
    W='#AA3377',  # purple
)
exp_name = 'av'
duration = 1.8
design = 'event'
plot_subject = '201'
plot_run = 1
beh_title, beh_idx = 'AV', 0
filt_kwargs = dict(
    l_freq=0.02, l_trans_bandwidth=0.02,
    h_freq=0.2, h_trans_bandwidth=0.02)
run_h = True  # regenerate HbO/HbR
n_jobs = 4  # for GLM

raw_path = '../../data'
proc_path = '../../processed'
results_path = '../../results'
subjects_dir = '../../subjects'
os.makedirs(results_path, exist_ok=True)
os.makedirs(proc_path, exist_ok=True)
os.makedirs(subjects_dir, exist_ok=True)
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)
use = None
all_sci = list()
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

###############################################################################
# Load participant data

for subject in subjects[0 if run_h else subjects.index(plot_subject):]:
    for day in days:
        for run in runs:
            group = subject_to_group.get(int(subject), "unknown")
            root1 = f'Day{day}'
            root2 = f'{subject}_{day}'
            root3 = f'*-*-*_{run:03d}'
            fname_base = op.join(raw_path, root1, root2, root3)
            fname = glob.glob(fname_base)
            print(fname)
            base = f'{subject}_{day}_{run:03d}'
            base_pr = base.ljust(20)
            if not run_h:
                if subject != plot_subject or run != plot_run:
                    continue
            raw_intensity = mne.io.read_raw_nirx(fname[0])
            raw_od = mne.preprocessing.nirs.optical_density(
                raw_intensity, verbose='error')
            # good/bad channels
            peaks = np.ptp(raw_od.get_data('fnirs'), axis=-1)
            flat_names = [
                raw_od.ch_names[f].split(' ')[0]
                for f in np.where(peaks < 0.001)[0]]
            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
            all_sci.extend(sci)
            sci_mask = (sci < 0.25)
            got = np.where(sci_mask)[0]
            print(f'    Run {base_pr}: {len(got)}/{len(raw_od.ch_names)} bad')
            # assign bads
            assert raw_od.info['bads'] == []
            bads = set(raw_od.ch_names[pick] for pick in got)
            bads = bads | set(ch_name for ch_name in raw_od.ch_names
                            if ch_name.split(' ')[0] in flat_names)
            bads = sorted(bads)
            raw_tddr = tddr(raw_od)
            raw_tddr_bp = raw_tddr.copy().filter(**filt_kwargs)
            raw_tddr_bp.info['bads'] = bads
            picks = mne.pick_types(raw_tddr_bp.info, fnirs=True)
            peaks = np.ptp(raw_tddr_bp.get_data(picks), axis=-1)
            assert (peaks > 1e-5).all()
            raw_tddr_bp.info['bads'] = []
            raw_h = mne.preprocessing.nirs.beer_lambert_law(raw_tddr_bp, 6.)
            # wait until now to assign bads so that we can choose later whether
            # we want the MATLAB bads or the Python ones
            h_bads = [
                ch_name for ch_name in raw_h.ch_names
                if ch_name.split(' ')[0] in set(bad.split(' ')[0] for bad in bads)]
            assert len(bads) == len(h_bads)
            raw_h.info['bads'] = h_bads
            raw_h.info._check_consistency()
            picks = mne.pick_types(raw_h.info, fnirs=True)
            peaks = np.ptp(raw_h.get_data(picks), axis=-1)
            assert (peaks > 1e-9).all()  # TODO: Maybe too small
            raw_h.save(op.join(proc_path, f'{base}_hbo_raw.fif'), 
                       overwrite=True)   
            if subject == plot_subject and run == plot_run:
                use = None
                assert use is None
                use = dict(intensity=raw_intensity,
                        od=raw_od,
                        tddr=raw_tddr,
                        h=raw_h,
                        run=run,
                        day=day,
                        group=group)
            del raw_intensity, raw_od, raw_tddr, raw_tddr_bp, raw_h

assert isinstance(use, dict)
ch_names = [ch_name.rstrip(' hbo') for ch_name in use['h'].ch_names[::2]]
info = use['h'].info

###############################################################################
# Channel example figure

sfreq = 4.807692  # all analysis at this rate

for subject in subjects:
    for day in days:
        for run in runs:
            fname = op.join(proc_path, f'{subject}_{day}_{run:03d}_hbo_raw.fif')
            raw_h = mne.io.read_raw_fif(fname)
            events, _ = mne.events_from_annotations(raw_h)
            print(len(events))

def _make_design(raw_h, design, subject=None, run=None, day=None, group=None):
    annotations_to_remove = raw_h.annotations.description == '255.0'
    raw_h.annotations.delete(annotations_to_remove)
    events, _ = mne.events_from_annotations(raw_h)
    rows_to_remove = events[:, -1] == 1
    events = events[~rows_to_remove]
    # mis-codings
    if len(events)==101:
        events = events[1:]
    n_times = len(raw_h.times)
    stim = np.zeros((n_times, 4))
    events[:, 2] -= 1
    assert len(events) == 100, len(events)
    want = [0] + [25] * 4
    count = np.bincount(events[:, 2])
    assert np.array_equal(count, want), count
    assert events.shape == (100, 3), events.shape
    mne.viz.plot_events(events)
    if design == 'block':
        events = events[0::5]
        duration = 20.
        assert np.array_equal(np.bincount(events[:, 2]), [0] + [5] * 4)
    else:
        assert design == 'event'
        assert len(events) == 100
        duration = 1.8
        assert events.shape == (100, 3)
        events_r = events[:, 2].reshape(20, 5)
        assert (events_r == events_r[:, :1]).all()
        del events_r
    idx = (events[:, [0, 2]] - [0, 1]).T
    assert np.in1d(idx[1], np.arange(len(conditions))).all()
    stim[tuple(idx)] = 1
    assert raw_h.info['sfreq'] == sfreq  # necessary for below logic to work
    n_block = int(np.ceil(duration * sfreq))
    stim = signal.fftconvolve(stim, np.ones((n_block, 1)), axes=0)[:n_times]
    dm_events = pd.DataFrame({
        'trial_type': [conditions[ii] for ii in idx[1]],
        'onset': idx[0] / raw_h.info['sfreq'],
        'duration': n_block / raw_h.info['sfreq']})
    dm = make_first_level_design_matrix(
        raw_h.times, dm_events, hrf_model='glover',
        drift_model='polynomial', drift_order=0)
    return stim, dm, events


###############################################################################
# Plot the design matrix and some raw traces

fig, axes = plt.subplots(2, 1, figsize=(6., 3), constrained_layout=True)
# Design
ax = axes[0]
raw_h = use['h']
stim, dm, _ = _make_design(raw_h, design)
for ci, condition in enumerate(conditions):
    color = condition_colors[condition]
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
         if 'S1_D2' in ch_name]
assert len(picks) == 2
fnirs_colors = dict(hbo='r', hbr='b')
ylim = np.array([-0.5, 0.5])
for pi, pick in enumerate(picks):
    color = fnirs_colors[raw_h.ch_names[pick][-3:]]
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
for ext in ('png', 'svg'):
    fig.savefig(
        op.join(
            results_path, f'figure_1_{exp_name}.{ext}'))


###############################################################################
# Run GLM analysis and epoching

sfreq = 4.807692050933838

df_cha = pd.DataFrame()
for day in days:
    for subject in subjects:
        fname = op.join(proc_path, f'{subject}_{day}_{exp_name}.h5')
        if not op.isfile(fname):
            group = subject_to_group.get(int(subject), "unknown")
            fname = op.join(proc_path, f'{subject}_{day}_{exp_name}.h5')
            subj_cha = pd.DataFrame()
            t0 = time.time()
            for run in runs:
                print(f'Running GLM for {group} {subject} day {day} run {run:03d}... ', end='')
                fname2 = op.join(proc_path, f'{subject}_{day}_{run:03d}_hbo_raw.fif')
                raw_h = mne.io.read_raw_fif(fname2)
                _, dm, _ = _make_design(raw_h, design, subject, run, day, group)
                glm_est = mne_nirs.statistics.run_glm(
                    raw_h, dm, noise_model='ols', n_jobs=n_jobs)
                cha = glm_est.to_dataframe()
                cha['subject'] = subject
                cha['run'] = run
                cha['day'] = day
                cha['group'] = group
                cha['good'] = ~np.in1d(cha['ch_name'], bads)
                subj_cha = pd.concat([subj_cha, cha], ignore_index=True)
                del raw_h
            subj_cha.to_hdf(fname, 'subj_cha', mode='w')
            print(f'{time.time() - t0:0.1f} sec')
        df_cha = pd.concat([df_cha, pd.read_hdf(fname)], ignore_index=True)
df_cha.reset_index(drop=True, inplace=True)

# block averages
event_id = {condition: ci for ci, condition in enumerate(conditions, 1)}
evokeds = {condition: dict() for condition in conditions}
for day in days:
    for subject in subjects:
        fname = op.join(
            proc_path, f'{subject}_{day}_{exp_name}-ave.fif')
        if not op.isfile(fname):
            tmin, tmax = -2, 38
            baseline = (None, 0)
            t0 = time.time()
            print(f'Creating block average for {subject} ... ', end='')
            raws = list()
            events = list()
            for run in runs:
                fname2 = op.join(proc_path, f'{subject}_{day}_{run:03d}_hbo_raw.fif')
                raw_h = mne.io.read_raw_fif(fname2)
                events.append(_make_design(raw_h, 'block', subject, run)[2])
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

# Exclude bad channels
bad = dict()
for day in days:
    for subject in subjects:
        for run in runs:
            fname2 = op.join(proc_path, f'{subject}_{day}_{run:03d}_hbo_raw.fif')
            this_info = mne.io.read_info(fname2)
            bad_channels = [idx - 1 for idx in sorted(
                this_info['ch_names'].index(bad) + 1 for bad in this_info['bads'])]
            valid_indices = np.arange(len(use['h'].ch_names))
            bb = [b for b in bad_channels if b in valid_indices]
            bad[(subject, run, day)] = bb
#        assert np.in1d(bad[(subject, run, day)], np.arange(len(use['h'].ch_names))).all()  # noqa: E501

# make life easier by combining across runs
bad_combo = dict()
for day in days:
    for (subject, run, day), bb in bad.items():
        bad_combo[subject] = sorted(set(bad_combo.get(subject, [])) | set(bb))
bad = bad_combo
assert set(bad) == set(subjects)
start = len(df_cha)
n_drop = 0
for day in days:
    for (subject, run, day), bb in bad.items():
        if not len(bb):
            continue
        drop_names = [use['h'].ch_names[b] for b in bb]
        is_subject = (df_cha['subject'] == subject)
        is_day = (df_cha['day'] == day)
        assert len(is_subject) == len(df_cha)
        is_day = (df_cha['day'] == day)
        drop = df_cha.index[
            is_subject &
            is_day &
            np.in1d(df_cha['ch_name'], drop_names)]
        n_drop += len(drop)
        if len(drop):
            print(f'Dropping {len(drop)} for {subject} day {day}')  # {run}')
            df_cha.drop(drop, inplace=True)
end = len(df_cha)
assert n_drop == start - end, (n_drop, start - end)

# combine runs by averaging estimates
sorts = ['subject', 'ch_name', 'Chroma', 'Condition', 'group', 'day', 'run']
df_cha.sort_values(
    sorts, inplace=True)
assert (np.array(df_cha['run']).reshape(-1, 2) == runs).all()
theta = np.array(df_cha['theta']).reshape(-1, len(runs)).mean(-1)
df_cha.drop(
    [col for col in df_cha.columns if col not in sorts[:-1]], axis='columns',
    inplace=True)
df_cha.reset_index(drop=True, inplace=True)
df_cha = df_cha[::len(runs)]
df_cha.reset_index(drop=True, inplace=True)
df_cha['theta'] = theta
df_cha.to_csv(op.join(results_path, 'df_cha.csv'), index=False)
df_cha.to_csv(op.join('df_cha.csv'), index=False)

# Mixed linear model
def _mixed_df(ch_summary):
    formula = "theta ~ -1 + ch_name:Condition" 
    ch_model = smf.mixedlm(  
        formula, ch_summary, groups=ch_summary["subject"]).fit(method='powell')
    ch_model_df = mne_nirs.statistics.statsmodels_to_results(ch_model)
    ch_model_df['P>|z|'] = ch_model.pvalues
    ch_model_df.drop([idx for idx in ch_model_df.index if '[constant]' in idx],
                    inplace=True)
    return ch_model_df

# Run group level model and convert to dataframe
use_subjects = [subj for subj in subjects]
ch_summary = df_cha.query("Chroma in ['hbo']").copy()
ch_summary_use = ch_summary.query(
    f"subject in {use_subjects}").copy()
ch_model_df = _mixed_df(ch_summary_use) 
ch_model_df.reset_index(inplace=True)

# Correct for multiple comparisons
print(f'Correcting for {len(ch_model_df["P>|z|"])} comparisons using FDR')
_, ch_model_df['P_fdr'] = mne.stats.fdr_correction(
    ch_model_df['P>|z|'], method='indep')
ch_model_df['SIG'] = ch_model_df['P_fdr'] < 0.05
ch_model_df.to_csv(op.join('ch_model_corrected.csv'), index=False)
ch_model_df.loc[ch_model_df.SIG == True]


###############################################################################
# Plot significant channels

sig_chs = dict()
zs = dict()
for condition in conditions:
    sig_df = ch_model_df[
        (ch_model_df['P_fdr'] < 0.05) &
        (ch_model_df['Condition'] == condition)]
    sig_chs[(condition)] = sorted(
        (use['h'].ch_names.index(row[1]['ch_name']), row[1]['P_fdr'])
        for row in sig_df.iterrows())
    ch_model_df[ch_model_df['Condition'] == condition]
    zs[condition] = np.array([
        ch_model_df.loc[(ch_model_df['Condition'] == condition) & 
                        (ch_model_df['ch_name'] == ch_name), 'z'].iloc[0]
        for ch_name in info['ch_names'][::2]], float)
    #assert zs[condition].shape == (42,)
    assert np.isfinite(zs[condition]).all()

def _plot_sig_chs(sigs, ax):
    if sigs and isinstance(sigs[0], tuple):
        sigs = [s[0] for s in sigs]
    ch_groups = [sigs, np.setdiff1d(np.arange(info['nchan']), sigs)]
    mne.viz.plot_sensors(
        info, 'topomap', 'hbo', title='', axes=ax,
        show_names=True, ch_groups=ch_groups)
    ax.collections[0].set(lw=0)
    c = ax.collections[0].get_facecolor()
    c[(c[:, :3] == (0.5, 0, 0)).all(-1)] = (0., 0., 0., 0.1)
    c[(c[:, :3] == (0, 0, 0.5)).all(-1)] = (0., 1., 0., 0.5)
    ax.collections[0].set_facecolor(c)
    ch_names = [info['ch_names'][idx] for idx in sigs]
    texts = list(ax.texts)
    got = []
    for text in list(texts):
        try:
            idx = ch_names.index(text.get_text())
        except ValueError:
            text.remove()
        else:
            got.append(idx)
            text.set_text(f'{sigs[idx] // 2 + 1}')
            text.set(fontsize='xx-small', zorder=5, ha='center')
    assert len(got) == len(sigs), (got, list(sigs))

def _plot_sigs(sig_chs, all_corrs=()):
    n_col = max(len(x) for x in sig_chs.values()) + 1
    n_row = len(conditions)
    figsize = (n_col * 1.0, n_row * 1.0)
    fig, axes = plt.subplots(
        n_row, n_col, figsize=figsize, constrained_layout=True, squeeze=False)
    h_colors = {0: 'r', 1: 'b'}
    xticks = [0, 10, 20, 30]
    ylim = [-0.2, 0.3]
    yticks = [-0.2, -0.1, 0, 0.1, 0.2, 0.3]
    xlim = [times[0], 35]
    ylim = np.array(ylim)
    yticks = np.array(yticks)
    for ci, condition in enumerate(conditions):
        ii = 0
        sigs = sig_chs[condition]
        if len(sigs) == 0:
            sigs = [(None, None)]
        for ii, (ch_idx, ch_p) in enumerate(sigs):
            ax = axes[ci, ii]
            if ch_idx is not None:
                for jj in range(2):  # HbO, HbR
                    color = h_colors[jj]
                    a = 1e6 * np.array(
                        [evokeds[condition][subject].data[ch_idx + jj]
                         for subject in use_subjects
                         if ch_idx + jj not in bad.get(subject, [])], float)
                    m = np.mean(a, axis=0)
                    lower, upper = stats.t.interval(
                        0.95, len(a) - 1, loc=m, scale=stats.sem(a, axis=0))
                    ax.fill_between(
                        times, lower, upper, facecolor=color,
                        edgecolor='none', lw=0, alpha=0.25, zorder=3,
                        clip_on=False)
                    ax.plot(times, m, color=color, lw=1, zorder=4,
                            clip_on=False)
                # Correlations
                this_df = ch_summary_use.query(
                    f'ch_name == {repr(use["h"].ch_names[ch_idx])} and '
                    f'Chroma == "hbo" and '
                    f'Condition == {repr(condition)}')
                #assert 8 <= len(this_df) <= len(subjects), len(this_df)
                a = np.array(this_df['theta'])
                cs = list()
                if len(cs):
                    cs = [''] + cs
                c = '\n'.join(cs)
                ax.text(times[-1], ylim[1],
                        f'ch{ch_idx // 2 + 1}\np={ch_p:0.5f}{c}',
                        ha='right', va='top', fontsize='x-small')
            ax.axvline(20, ls=':', color='0.5', zorder=2, lw=1)
            ax.axhline(0, ls='-', color='k', zorder=2, lw=0.5)
            ax.set(xticks=xticks, yticks=yticks)
            ax.set(xlim=xlim, ylim=ylim)
            for key in ('top', 'right'):
                ax.spines[key].set_visible(False)
            if ax.get_subplotspec().is_last_row():
                ax.set(xlabel='Time (sec)')
            else:
                ax.set_xticklabels([''] * len(xticks))
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(condition)
            else:
                ax.set_yticklabels([''] * len(yticks))
            for key in ('top', 'right'):
                ax.spines[key].set_visible(False)
        for ii in range(ii + 1, n_col - 1):
            fig.delaxes(axes[ci, ii])
        # montage
        ax = axes[ci, -1]
        if sigs[0][0] is None:
            fig.delaxes(ax)
        else:
            # plot montage
            _plot_sig_chs(sigs, ax)
    return fig

times = evokeds[conditions[0]][subjects[0]].times
info = evokeds[conditions[0]][subjects[0]].info
fig = _plot_sigs(sig_chs)
for ext in ('png', 'svg'):
    fig.savefig(op.join(results_path, f'stats_{exp_name}.{ext}'))

###############################################################################
# Source space projection

import pyvista
import pyvistaqt

info = use['h'].copy().pick_types(fnirs='hbo', exclude=()).info
info['bads'] = []
assert tuple(zs) == conditions

evoked = mne.EvokedArray(np.array(list(zs.values())).T, info)
picks = np.arange(len(evoked.ch_names))
for ch in evoked.info['chs']:
    assert ch['coord_frame'] == mne.io.constants.FIFF.FIFFV_COORD_HEAD
stc = mne.stc_near_sensors(
    evoked, trans='fsaverage', subject='fsaverage', mode='weighted',
    distance=0.02, project=True, picks=picks, subjects_dir=subjects_dir)
# Split channel indices by left lat, posterior, right lat:
num_map = {name: str(ii) for ii, name in enumerate(evoked.ch_names)}
evoked.copy().rename_channels(num_map).plot_sensors(show_names=True)
view_map = [np.arange(19), np.arange(19, 33), np.arange(33, 52)]
surf = mne.read_bem_surfaces(  # brain surface
    f'{subjects_dir}/fsaverage/bem/fsaverage-5120-5120-5120-bem.fif', s_id=1)

for ci, condition in enumerate(conditions):
    this_sig = [v[0] // 2 for v in sig_chs[condition]]
    #assert np.in1d(this_sig, np.arange(52)).all()
    pos = np.array([info['chs'][idx]['loc'][:3] for idx in this_sig])
    pos.shape = (-1, 3)  # can be empty
    trans = mne.transforms._get_trans('fsaverage', 'head', 'mri')[0]
    pos = mne.transforms.apply_trans(trans, pos)  # now in MRI coords
    pos = mne.surface._project_onto_surface(pos, surf, project_rrs=True)[2]
    # plot
    brain = stc.plot(hemi='both', views=['lat', 'frontal', 'lat'],
                    initial_time=evoked.times[ci], cortex='low_contrast',
                    time_viewer=False, show_traces=False,
                    surface='pial', smoothing_steps=0, size=(1200, 400),
                    clim=dict(kind='value', pos_lims=[0., 1.25, 2.5]),
                    colormap='RdBu_r', view_layout='horizontal',
                    colorbar=(0, 1), time_label='', background='w',
                    brain_kwargs=dict(units='m'),
                    add_data_kwargs=dict(colorbar_kwargs=dict(
                        title_font_size=24, label_font_size=24, n_labels=5,
                        title='z score')), subjects_dir=subjects_dir)
    brain.show_view('lat', hemi='lh', row=0, col=0)
    brain.show_view(azimuth=270, elevation=90, row=0, col=1)
    pl = brain.plotter
    used = np.zeros(len(this_sig))
    brain.show_view('lat', hemi='rh', row=0, col=2)
    plt.imsave(
        op.join(results_path, f'brain_{exp_name}_{condition}.png'), pl.image)

# fOLD specificity
fold_files = ['10-10.xls', '10-5.xls']
for fname in fold_files:
    if not op.isfile(fname):
        pooch.retrieve(f'https://github.com/nirx/fOLD-public/raw/master/Supplementary/{fname}', None, fname, path=os.getcwd())  # noqa
raw_spec = use['h'].copy()
raw_spec.pick_channels(raw_spec.ch_names[::2])
specs = mne_nirs.io.fold_channel_specificity(raw_spec, fold_files, 'Brodmann')
for si, spec in enumerate(specs, 1):
    spec['Channel'] = si
    spec['negspec'] = -spec['Specificity']
specs = pd.concat(specs, ignore_index=True)
specs.drop(['Source', 'Detector', 'Distance (mm)', 'brainSens',
            'X (mm)', 'Y (mm)', 'Z (mm)'], axis=1, inplace=True)
specs.sort_values(['Channel', 'negspec'], inplace=True)
specs.drop('negspec', axis=1, inplace=True)
specs.reset_index(inplace=True, drop=True)
specs.to_csv(op.join(results_path, 'specificity.csv'), index=False)
