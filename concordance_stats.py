import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import seaborn as sns

from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.utils import resample


def nan_med_hard(v):
    if np.sum(np.isnan(v))>1:
        return None
    elif np.sum(np.isnan(v))==1:
        if len(np.unique(v[~np.isnan(v)]))==1:
            return np.unique(v[~np.isnan(v)])[0]
    else:
        return np.median(v)

def nan_med(v):
    if np.sum(np.isnan(v))>1:
        return None
    else:
        return np.nanmedian(v)

def kappa_lin(a, b):
    return cohen_kappa_score(a, b, weights='linear')

def kappa(a, b):
    return cohen_kappa_score(a, b)

def agg_rate(a, b):
    return np.mean(np.array(a) == np.array(b))

def disc_rate(a, b):
    return np.mean(np.array(a) != np.array(b))

def disc_2_rate(a, b):
    return np.mean(np.abs(np.array(b) - np.array(a)) >= 2)

def overcall_rate(a, b):
    """ Overcall := (b > a) """
    return np.mean((np.array(b) - np.array(a)) > 0)

def undercall_rate(a, b):
    """ Undercall := (b < a) """
    return np.mean((np.array(b) - np.array(a)) < 0)

def mean_diff(a, b):
    return np.mean(np.array(b) - np.array(a))

def disc_weighted_bias(a, b):
    """ 1 := b > a """
    na = np.sum(np.array(a) > np.array(b))
    nb = np.sum(np.array(a) < np.array(b))
    nc = np.sum(np.array(a) == np.array(b))

    return (2*nb/(na+nb)-1)*(nb+na)/(na+nb+nc)

def abs_disc_weighted_bias(a, b):
    """ 1 := b > a """
    return np.abs(disc_weighted_bias(a,b))

def clean_nans(l):
    a = pd.DataFrame(l)
    a = a.dropna(axis=1)
    return [np.array(i) for i in a.to_numpy()]

def stat_str(fe, ci, ndecimal=2):
    return f"{np.around(fe, ndecimal)} ({np.around(ci[0], ndecimal)}, {np.around(ci[1], ndecimal)})"

def ci_stats(fn, a, b, stratify=None, n_samples=None, name='AIM', do_abs=False):

    if stratify is not None:
        a, b, stratify = clean_nans([a, b, stratify])
    else:
        a, b = clean_nans([a, b])

    if n_samples is not None:
        N = n_samples
    else:
        N = len(a)

    def _local_abs(a):
        if do_abs:
            return np.abs(np.array(a))
        else:
            return a

    fe = _local_abs(fn(a, b))

    bs = []

    for ii in range(10000):
        a_bs, b_bs = resample(a, b, stratify=stratify, n_samples=n_samples)
        bs.append(_local_abs(fn(a_bs, b_bs)))

    ci = np.percentile(bs, [2.5, 97.5])

    stat_df = pd.DataFrame({
        name: stat_str(fe, ci),
        'N': N
    }, index=[0])

    tall_df = pd.DataFrame({
        'AIM v CON': [fe] + list(ci),
        'Stat': ['PE', 'LB95', 'UB95'],
        'N': N
    })

    return stat_df, tall_df

def legacy_stats(fn, aim: np.array, np1:np.array, np2:np.array, np3:np.array,
                 stratify=None, n_samples=None, do_abs=False):

    if stratify is not None:
        aim, np1, np2, np3, stratify = clean_nans([aim, np1, np2, np3, stratify])
    else:
        aim, np1, np2, np3 = clean_nans([aim, np1, np2, np3])

    con = np.median([np1, np2, np3], axis=0)

    if n_samples is not None:
        N = n_samples
    else:
        N = len(aim)

    def _local_abs(a):
        if do_abs:
            return np.abs(np.array(a))
        else:
            return a

    def fn_np(np1, np2, np3):
        s1 = _local_abs([fn(np1, np2), fn(np1, np3), fn(np2, np3)])
        return np.mean(s1), s1[0], s1[1], s1[2]

    def fn_aim(aim, con):
        return _local_abs(fn(aim, con))

    fe_aim = fn_aim(aim, con)
    fe_np, fe_np12, fe_np13, fe_np23 = fn_np(np1, np2, np3)

    bs_aim = []
    bs_np = []
    bs_np12 = []
    bs_np13 = []
    bs_np23 = []

    for ii in range(10000):
        aim_bs, np1_bs, np2_bs, np3_bs, con_bs = resample(aim, np1, np2, np3, con, stratify=stratify, n_samples=n_samples)
        bs_aim.append(fn_aim(aim_bs, con_bs))
        ibs_np, ibs_np12, ibs_np13, ibs_np23 = fn_np(np1_bs, np2_bs, np3_bs)
        bs_np.append(ibs_np)
        bs_np12.append(ibs_np12)
        bs_np13.append(ibs_np13)
        bs_np23.append(ibs_np23)


    ci_aim = np.percentile(bs_aim, [2.5, 97.5])
    ci_np = np.percentile(bs_np, [2.5, 97.5])
    ci_np12 = np.percentile(bs_np12, [2.5, 97.5])
    ci_np13 = np.percentile(bs_np13, [2.5, 97.5])
    ci_np23 = np.percentile(bs_np23, [2.5, 97.5])

    stat_str_df = pd.DataFrame({
        'AIM v CON': stat_str(fe_aim, ci_aim),
        'NP PW AVG': stat_str(fe_np, ci_np),
        'NP0-NP1': stat_str(fe_np12, ci_np12),
        'NP0-NP2': stat_str(fe_np13, ci_np13),
        'NP1-NP2': stat_str(fe_np23, ci_np23),
        'N': N
    }, index=[0])

    tall_stat_df = pd.DataFrame({
            'AIM v CON': [fe_aim]+list(ci_aim),
            'NP PW AVG': [fe_np]+list(ci_np),
            'NP0-NP1': [fe_np12]+list(ci_np12),
            'NP0-NP2': [fe_np13]+list(ci_np13),
            'NP1-NP2': [fe_np23]+list(ci_np23),
            'Stat':['PE','LB95','UB95'],
            'N': N
    })

    return stat_str_df, tall_stat_df

def pairwise_stats( fn, aim: np.array, np1:np.array, np2:np.array, np3:np.array,
                    stratify=None, n_samples=None, do_abs=False)->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes AIM + 3 NP score arrays and returns mean pairwise stats.
    Set CI power with n_samples if needed.
    """
    if stratify is not None:
        aim, np1, np2, np3, stratify = clean_nans([aim, np1, np2, np3, stratify])
    else:
        aim, np1, np2, np3 = clean_nans([aim, np1, np2, np3])

    if n_samples is not None:
        N = n_samples
    else:
        N = len(aim)

    def _local_abs(a):
        if do_abs:
            return np.abs(np.array(a))
        else:
            return a

    def fn_np(np1, np2, np3):
        s1 = _local_abs([fn(np1, np2), fn(np1, np3), fn(np2, np3)])
        return np.mean(s1), s1[0], s1[1], s1[2]

    def fn_aim(aim, np1, np2, np3):
        s1 = _local_abs([fn(aim, np1), fn(aim, np2), fn(aim, np3)])
        return np.mean(s1)

    fe_aim = fn_aim(aim, np1, np2, np3)
    fe_np, fe_np12, fe_np13, fe_np23 = fn_np(np1, np2, np3)

    bs_aim = []
    bs_np = []
    bs_np12 = []
    bs_np13 = []
    bs_np23 = []

    for ii in range(10000):
        aim_bs, np1_bs, np2_bs, np3_bs = resample(aim, np1, np2, np3, stratify=stratify, n_samples=n_samples)
        bs_aim.append(fn_aim(aim_bs, np1_bs, np2_bs, np3_bs))
        ibs_np, ibs_np12, ibs_np13, ibs_np23 = fn_np(np1_bs, np2_bs, np3_bs)
        bs_np.append(ibs_np)
        bs_np12.append(ibs_np12)
        bs_np13.append(ibs_np13)
        bs_np23.append(ibs_np23)


    ci_aim = np.percentile(bs_aim, [2.5, 97.5])
    ci_np = np.percentile(bs_np, [2.5, 97.5])
    ci_np12 = np.percentile(bs_np12, [2.5, 97.5])
    ci_np13 = np.percentile(bs_np13, [2.5, 97.5])
    ci_np23 = np.percentile(bs_np23, [2.5, 97.5])

    stat_str_df = pd.DataFrame({
        'AIM PW AVG': stat_str(fe_aim, ci_aim),
        'NP PW AVG': stat_str(fe_np, ci_np),
        'NP0-NP1': stat_str(fe_np12, ci_np12),
        'NP0-NP2': stat_str(fe_np13, ci_np13),
        'NP1-NP2': stat_str(fe_np23, ci_np23),
        'N': N
    }, index=[0])

    tall_stat_df = pd.DataFrame({
            'AIM PW AVG': [fe_aim]+list(ci_aim),
            'NP PW AVG': [fe_np]+list(ci_np),
            'NP0-NP1': [fe_np12]+list(ci_np12),
            'NP0-NP2': [fe_np13]+list(ci_np13),
            'NP1-NP2': [fe_np23]+list(ci_np23),
            'Stat':['PE','LB95','UB95'],
            'N': N
    })

    return stat_str_df, tall_stat_df

def loo_stats(fn, aim: np.array, np1: np.array, np2: np.array, np3: np.array, n_samples = None, stratify=None, do_abs=False):
    """
    Takes AIM + 3 NP score arrays and returns leave-one-out stats.
    Set CI power with n_samples if needed.
    """

    if stratify is not None:
        aim, np1, np2, np3, stratify = clean_nans([aim, np1, np2, np3, stratify])
    else:
        aim, np1, np2, np3 = clean_nans([aim, np1, np2, np3])

    if n_samples is not None:
        N = n_samples
    else:
        N = len(aim)

    def _local_abs(a):
        if do_abs:
            return np.abs(np.array(a))
        else:
            return a

    def fn_np(cp1, cp2, cp3):
        cks_pa = []
        cks_pa.append(np.mean([
            fn(cp1, np.ceil(np.median([cp2, cp3], axis=0))),
            fn(cp1, np.floor(np.median([cp2, cp3], axis=0)))
        ]))
        cks_pa.append(np.mean([
            fn(cp2, np.ceil(np.median([cp1, cp3], axis=0))),
            fn(cp2, np.floor(np.median([cp1, cp3], axis=0)))
        ]))
        cks_pa.append(np.mean([
            fn(cp3, np.ceil(np.median([cp2, cp1], axis=0))),
            fn(cp3, np.floor(np.median([cp2, cp1], axis=0)))
        ]))
        return np.mean(_local_abs(cks_pa))

    def fn_aim(aim, cp1, cp2, cp3):
        cks_pa = []
        cks_pa.append(np.mean([
            fn(aim, np.ceil(np.median([cp2, cp3], axis=0))),
            fn(aim, np.floor(np.median([cp2, cp3], axis=0)))
        ]))
        cks_pa.append(np.mean([
            fn(aim, np.ceil(np.median([cp1, cp3], axis=0))),
            fn(aim, np.floor(np.median([cp1, cp3], axis=0)))
        ]))
        cks_pa.append(np.mean([
            fn(aim, np.ceil(np.median([cp2, cp1], axis=0))),
            fn(aim, np.floor(np.median([cp2, cp1], axis=0)))
        ]))
        return np.mean(_local_abs(cks_pa))


    fe_aim = fn_aim(aim, np1, np2, np3)
    fe_np = fn_np(np1, np2, np3)

    bs_aim = []
    bs_np = []

    for ii in range(10000):
        aim_bs, np1_bs, np2_bs, np3_bs = resample(aim, np1, np2, np3, stratify=stratify, n_samples=n_samples)
        bs_aim.append(fn_aim(aim_bs, np1_bs, np2_bs, np3_bs))
        ibs_np = fn_np(np1_bs, np2_bs, np3_bs)
        bs_np.append(ibs_np)

    ci_aim = np.percentile(bs_aim, [2.5, 97.5])
    ci_np = np.percentile(bs_np, [2.5, 97.5])

    stat_str_df = pd.DataFrame({
        'AIM LOO AVG': stat_str(fe_aim, ci_aim),
        'NP LOO AVG': stat_str(fe_np, ci_np),
        'N': N
    }, index=[0])

    tall_stat_df = pd.DataFrame({
        'AIM LOO AVG': [fe_aim] + list(ci_aim),
        'NP LOO AVG': [fe_np] + list(ci_np),
        'Stat': ['PE', 'LB95', 'UB95'],
        'N': N
    })

    return stat_str_df, tall_stat_df

def grand_med_stats(fn,
                    aim: np.array,
                    np1: np.array,
                    np2: np.array,
                    np3: np.array,
                    stratify: np.array = None,
                    n_samples: int = None,
                    do_abs: bool = False):
    """
    Takes AIM + 3 NP score arrays and returns grand median stats.
    Set CI power with n_samples if needed.
    """
    if stratify is not None:
        aim, np1, np2, np3, stratify = clean_nans([aim, np1, np2, np3, stratify])
    else:
        aim, np1, np2, np3 = clean_nans([aim, np1, np2, np3])

    if n_samples is not None:
        N = n_samples
    else:
        N = len(aim)

    gm = np.median([aim, np1, np2, np3], axis=0)

    def _local_abs(a):
        if do_abs:
            return np.abs(np.array(a))
        else:
            return a

    def fn_np(cp1, cp2, cp3, gm):
        cks_pa = []
        cks_pa.append(np.mean([
            fn(cp1, np.ceil(gm)),
            fn(cp1, np.floor(gm))
        ]))
        cks_pa.append(np.mean([
            fn(cp2, np.ceil(gm)),
            fn(cp2, np.floor(gm))
        ]))
        cks_pa.append(np.mean([
            fn(cp3, np.ceil(gm)),
            fn(cp3, np.floor(gm))
        ]))
        cks_pa = _local_abs(cks_pa)
        return np.mean(cks_pa), cks_pa[0],cks_pa[1],cks_pa[2]


    def fn_aim(aim, gm):
        cks_pa = []
        cks_pa.append(np.mean([
            fn(aim, np.ceil(gm)),
            fn(aim, np.floor(gm))
        ]))
        cks_pa = _local_abs(cks_pa)
        return np.mean(cks_pa)


    fe_aim = fn_aim(aim, gm)
    fe_np, fe_np1, fe_np2, fe_np3 = fn_np(np1, np2, np3, gm)

    bs_aim = []
    bs_np = []
    bs_np1 = []
    bs_np2 = []
    bs_np3 = []

    for ii in range(10000):
        aim_bs, np1_bs, np2_bs, np3_bs, gm_bs = resample(aim, np1, np2, np3, gm, stratify=stratify, n_samples=n_samples)
        bs_aim.append(fn_aim(aim_bs, gm_bs))
        i_np,i_np1,i_np2,i_np3 =  fn_np(np1_bs, np2_bs, np3_bs, gm_bs)
        bs_np.append(i_np)
        bs_np1.append(i_np1)
        bs_np2.append(i_np2)
        bs_np3.append(i_np3)

    ci_aim = np.percentile(bs_aim, [2.5, 97.5])
    ci_np = np.percentile(bs_np, [2.5, 97.5])
    ci_np1 = np.percentile(bs_np1, [2.5, 97.5])
    ci_np2 = np.percentile(bs_np2, [2.5, 97.5])
    ci_np3 = np.percentile(bs_np3, [2.5, 97.5])

    stat_str_df = pd.DataFrame({
        'AIM': stat_str(fe_aim, ci_aim),
        'NP AVG': stat_str(fe_np, ci_np),
        'NP0': stat_str(fe_np1, ci_np1),
        'NP1': stat_str(fe_np2, ci_np2),
        'NP2': stat_str(fe_np3, ci_np3),
        'N': N
    }, index=[0])

    tall_stat_df = pd.DataFrame({
        'AIM': [fe_aim] + list(ci_aim),
        'NP AVG': [fe_np] + list(ci_np),
        'NP0': [fe_np1] + list(ci_np1),
        'NP1': [fe_np2] + list(ci_np2),
        'NP2': [fe_np3] + list(ci_np3),
        'Stat': ['PE', 'LB95', 'UB95'],
        'N': N
    })

    return stat_str_df, tall_stat_df

def grand_loo_stats(fn, aim: np.array, np1: np.array, np2: np.array, np3: np.array, n_samples = None, stratify=None, do_abs=False):
    """
    Takes AIM + 3 NP score arrays and returns leave-one-out stats.
    Set CI power with n_samples if needed.
    """

    if stratify is not None:
        aim, np1, np2, np3, stratify = clean_nans([aim, np1, np2, np3, stratify])
    else:
        aim, np1, np2, np3 = clean_nans([aim, np1, np2, np3])

    if n_samples is not None:
        N = n_samples
    else:
        N = len(aim)

    def _local_abs(a):
        if do_abs:
            return np.abs(np.array(a))
        else:
            return a

    def fn_np(cp1, cp2, cp3, aim):
        cks_pa = []

        cks_pa.append(fn(np.median([aim, cp2, cp3], axis=0), cp1))
        cks_pa.append(fn(np.median([cp1, aim, cp3], axis=0), cp2))
        cks_pa.append(fn(np.median([cp1, cp2, aim], axis=0), cp3))

        return np.mean(_local_abs(cks_pa))

    def fn_aim(aim, cp1, cp2, cp3):
        return _local_abs(fn(np.median([cp1, cp2, cp3], axis=0), aim))


    fe_aim = fn_aim(aim, np1, np2, np3)
    fe_np = fn_np(np1, np2, np3, aim)

    bs_aim = []
    bs_np = []

    for ii in range(10000):
        aim_bs, np1_bs, np2_bs, np3_bs = resample(aim, np1, np2, np3, stratify=stratify, n_samples=n_samples)
        bs_aim.append(fn_aim(aim_bs, np1_bs, np2_bs, np3_bs))
        ibs_np = fn_np(np1_bs, np2_bs, np3_bs, aim_bs)
        bs_np.append(ibs_np)

    ci_aim = np.percentile(bs_aim, [2.5, 97.5])
    ci_np = np.percentile(bs_np, [2.5, 97.5])

    stat_str_df = pd.DataFrame({
        'AIM GLOO AVG': stat_str(fe_aim, ci_aim),
        'NP GLOO AVG': stat_str(fe_np, ci_np),
        'N': N
    }, index=[0])

    tall_stat_df = pd.DataFrame({
        'AIM GLOO AVG': [fe_aim] + list(ci_aim),
        'NP GLOO AVG': [fe_np] + list(ci_np),
        'Stat': ['PE', 'LB95', 'UB95'],
        'N': N
    })

    return stat_str_df, tall_stat_df


@np.vectorize
def ct_annot(r: float, c: int) -> str:
    return f"{c}\n({np.around(r,decimals=2)})"

def plot_confusion_table(a: np.array, b: np.array,
                         name_a: str = 'A', name_b: str = 'B',
                         title: str = None,
                         norm: str = 'row',
                         max_score: int = None) -> Tuple[ plt.Axes, pd.DataFrame, pd.DataFrame ]:

    a = a.astype(int)
    b = b.astype(int)

    ac  = pd.Categorical(a, categories=np.arange(max_score+1))
    bc  = pd.Categorical(b, categories=np.arange(max_score+1))

    if norm == 'row':
        r = pd.DataFrame(confusion_matrix(ac, bc, normalize='true'))
    elif norm == 'all':
        r = pd.DataFrame(confusion_matrix(ac, bc, normalize='all'))
    r.columns.name = name_b
    r.index.name = name_a

    c = pd.crosstab(ac, bc, rownames=[name_a], colnames=[name_b], dropna=False)

    g = ct_annot(r, c)

    with sns.axes_style("white"):
        ax = sns.heatmap(r, cbar=False, square=True, cmap='Blues', linewidths=1, annot=g, fmt='')
        ax.set_title(title)
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            horizontalalignment='right'
        )
        for ii in range(1+np.min([np.max(r.columns),np.max(r.index)])):
            ax.add_patch(Rectangle((ii, ii), 1, 1, fill=False, edgecolor='gray', lw=1.5))

    return ax, c, r
