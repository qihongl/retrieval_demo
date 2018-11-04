"""
save plots, for all conditions
- assume the results for all specified conditions are there
"""
import os
import numpy as np
import pandas as pd
from itertools import product
from util_plt import compute_cph, load_result_csv_fixm2, data2df
from util_plt import column_labels

import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

sns.set(style='white', context='talk', font_scale=.9,
        rc={"lines.linewidth": 2.5})
np.random.seed(0)

# constants
ALL_RULE_TYPES = ['null', 'demand', 'inhibit', 'demand+inhibit']
# simulation parameters
chk_siz_list = [4, 8, 16, 32, 64]
min_mat_list = [0, 1, 2, 4, 8, 16, 32, 64]
max_mis_list = [0, 1, 2, 4, 8, 16, 32, 64]
ncks = len(chk_siz_list)
nm1s = len(min_mat_list)
nm2s = len(max_mis_list)

"""
variation parameters
"""

# # set params
# penalty_strenth = 1
# ambiguity = .25
# noise = 0.01
# n_retrievals = 3
# mismatch_sensitivity = 1.0
# fixed_m2_val = 0

ambiguity_all = [.25]
noise_all = [0, 0.01]
n_retrievals_all = [1, 3]
mismatch_sensitivity_all = [0, 1.0]
penalty_strenth_all = [1, .5, 0]

for ambiguity, noise, n_retrievals, mismatch_sensitivity, penalty_strenth, fixed_m2_val in product(
    ambiguity_all, noise_all, n_retrievals_all, mismatch_sensitivity_all, penalty_strenth_all, max_mis_list,
):
    #     print(ambiguity, noise, n_retrievals, mismatch_sensitivity)

    """
    fixed parameters
    """

    # event parameters
    n_params = 64
    event_len = 512
    n_branches = 3
    event_len_trunc = 64
    ms_deday = 0.0

    # sample size
    n_sims = 500

    n_qs = event_len // event_len_trunc
    model_param_lists = [chk_siz_list, min_mat_list, max_mis_list]

    # log dir
    plt_root = '/tigress/qlu/logs/retrieval_dynamics/plots/tradeoff'
    log_root = '/tigress/qlu/logs/retrieval_dynamics/log/tradeoff'

    """
    load results
    """

    # only analyze data one max mismatch value
    n_sims_to_load = -1

    # set rule type
    DF = []
    # loop over rule types
    for rule_type in ALL_RULE_TYPES:
        print('rule = ', rule_type)
        df = load_result_csv_fixm2(
            fixed_m2_val,
            rule_type, n_params, n_branches, event_len, event_len_trunc,
            ambiguity, noise, n_retrievals, mismatch_sensitivity, ms_deday,
            n_sims, n_sims_to_load, model_param_lists,
            log_root, plt_root
        )
        DF.append(df)
        # print some info
        event_len_truncated = np.max(df['Unnamed: 0'])+1
        n_sims_done = int(np.max(df['sim_id']))
        print('num sims completed = ', n_sims_done)

    """
    compute CPH
    """

    AVG_CPH = []
    SEM_CPH = []
    # df = DF[-1]
    for df in DF:
        # compute CPH
        avg_cph, sem_cph = compute_cph(
            df, min_mat_list, chk_siz_list, event_len_trunc, n_qs, penalty_strenth
        )
        AVG_CPH.append(avg_cph)
        SEM_CPH.append(sem_cph)

    print('penalty_strenth: ', penalty_strenth)
    print('avg_cph shape: ', np.shape(avg_cph))

    """
    plot the data
    """

    list_plt = AVG_CPH
    # list_plt = SEM_CPH
    mask = np.zeros_like(avg_cph[:, :, 0, 0])
    mask[np.triu_indices_from(mask, k=4)] = True

    n_rows = n_cols = 2
    f, axes = plt.subplots(n_rows, n_cols, figsize=(11, 8))
    # make plots
    for i in range(n_rows * n_cols):
        i_col, i_row = np.unravel_index(i, (n_rows, n_cols))
        sns.heatmap(
            list_plt[i][:, :, 0, -1],
            vmin=0, vmax=1,
            mask=mask,
            square=True, cmap='viridis',
            annot=True, annot_kws={"size": 12}, fmt='.2f',
            xticklabels=min_mat_list,
            yticklabels=chk_siz_list,
            cbar=False,
            ax=axes[i_col, i_row])
        axes[i_col, i_row].set_title('\nRule: %s' % (ALL_RULE_TYPES[i]))

    for i in range(2):
        axes[i, 0].set_ylabel('Chunk size')
        axes[1, i].set_xlabel('Min match')

    f.tight_layout()

    info = """
    Penalized cumulative hits, time %d (%s)
    max-mismatch = %d, ambiguity = %.2f, noise = %.4f,
    penalty = %.2f,
    n_retrievals = %d, mismatch sensitivity = %.3f, mismatch sensitivity = %.3f
    n = %d
    """ % (
        event_len_truncated, rule_type,
        fixed_m2_val, ambiguity, noise,
        penalty_strenth,
        n_retrievals, mismatch_sensitivity, ms_deday,
        n_sims_done
    )
    print(info)

    """
    save plot
    """
    temp_plt_path = os.path.join(plt_root, 'temp')
    plt_format = '.png'
    fname = 'amb_%.2f_noise_%.3f_nRs_%d_mms_%d_m2v_%d_ps_%1.f' % (
        ambiguity, noise, n_retrievals, mismatch_sensitivity, fixed_m2_val, penalty_strenth
    ) + plt_format
    print(fname)

    f.savefig(
        os.path.join(temp_plt_path, fname),
        bbox_inches='tight',
        dpi=300
    )
