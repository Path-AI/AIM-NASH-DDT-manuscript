import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple

import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, median_test, chi2_contingency

from r_models import vanElterenTest

from textwrap import TextWrapper

def tw(txt,ncol):
    tw = TextWrapper(ncol)
    return '\n'.join(tw.wrap(txt))

def comb_two_strata(row,a,b):
    if row[b] is not None and row[a] is not None:
        return f'{row[a]}_{row[b]}'
    else:
        return None

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

def CMH_test(df, treat_col, outcome_col, strata_col, treat_order=None, outcome_order=None):
    """https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.StratifiedTable.html"""
    warnings.filterwarnings("ignore")
    strata_items = df[strata_col].unique()
    tables = []
    for strata in strata_items:
        ct = pd.crosstab(
            df[df[strata_col]==strata][treat_col],
            df[df[strata_col]==strata][outcome_col]
        )
        if treat_order is not None:
            ct = ct.reindex(treat_order, axis="rows")
        if outcome_order is not None:
            ct = ct.reindex(outcome_order, axis="columns")
        ct = ct.fillna( int(0) )
        tables.append(ct)
    st = sm.stats.StratifiedTable(tables)

    s_out = pd.Series({
        'OR':st.oddsratio_pooled,
        'OR (LB)': st.oddsratio_pooled_confint()[0],
        'OR (UB)': st.oddsratio_pooled_confint()[1],
        'Test Stat': st.test_null_odds().statistic,
        'Test P': st.test_null_odds().pvalue,
        'N Strata': len(strata_items),
        'N Samples': len(df.index)
    })

    return s_out 

def chi_sq_test(df, treat_col, outcome_col, treat_order=None, outcome_order=None):
    """https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.StratifiedTable.html"""
    warnings.filterwarnings("ignore")
    ct = pd.crosstab(
        df[treat_col],
        df[outcome_col]
    )
    if treat_order is not None:
        ct = ct.reindex(treat_order, axis="rows")
    if outcome_order is not None:
        ct = ct.reindex(outcome_order, axis="columns")
    ct = ct.fillna( int(0) )
    
    chi2, p, dof, ex = chi2_contingency(ct, correction=False)

    s_out = pd.Series({
        'Test Stat': chi2,
        'Test P': p,
        'N Samples': len(df.index)
    })

    return s_out 



class NASHEndpoint():
    """Supports Per-Subject data frames with different timepoints as suffixes."""
    
    tp_def = {
        'pre':'_BL',
        'post':'_W48'
    }
    
    cd_def = {
        'f':'GNN CRN_SCORE_TRICHROME',
        'i':'GNN LOBULAR_SCORE_HE',
        'b':'GNN BALLOONING_SCORE_HE',
        's':'GNN STEATOSIS_SCORE_HE'
    }
    
    def __init__(self, cd: dict = None, tp: dict = None ):
        
        if cd is None:
            self.cd = cd_def
        else:
            self.cd = cd
            
        if tp is None:
            self.tp = tp_def
        else:
            self.tp = tp
            
        self._combine_cd_tp()
        
    def _combine_cd_tp(self):
        self.ct = {}
        for k_tp,v_tp in self.tp.items():
            for k_cd, v_cd in self.cd.items():
                self.ct[f"{k_cd}_{k_tp}"] = f"{v_cd}{v_tp}"
    
    def fib_red(self, row):
        """Any fibrosis reduction."""
        fib_0 = row.loc[self.ct['f_pre']]
        fib_1 = row.loc[self.ct['f_post']]

        if not any(np.isnan([fib_0, fib_1])):
            return fib_1 < fib_0
        else:
            return None
    
    def nas_2pt_red(self, row):
        """(>= 2) pt NAS reduction"""
        nas_0 = row.loc[[self.ct['i_pre'],self.ct['b_pre'],self.ct['s_pre']]].sum()
        nas_1 = row.loc[[self.ct['i_post'],self.ct['b_post'],self.ct['s_post']]].sum()
    
        if not any(np.isnan([nas_0, nas_1])):
            return ((nas_1 - nas_0) <= (-2)) 
        else:
            return None
    
    def nas_2pt_red_1pt_bal_or_inf(self, row):
        """(>= 2) pt NAS reduction with at least 1pt reduction in Ballooning or Inflammation"""
        
        nas_0 = row.loc[[self.ct['i_pre'],self.ct['b_pre'],self.ct['s_pre']]].sum()
        nas_1 = row.loc[[self.ct['i_post'],self.ct['b_post'],self.ct['s_post']]].sum()
    
        bal_0 = row.loc[self.ct['b_pre']]
        bal_1 = row.loc[self.ct['b_post']]
        
        inf_0 = row.loc[self.ct['i_pre']]
        inf_1 = row.loc[self.ct['i_post']]
        
        if not any(np.isnan([nas_0, nas_1, bal_0, bal_1, inf_0, inf_1])):
            return ( ((nas_1 - nas_0) <= (-2)) & ( (bal_1 < bal_0) | (inf_1<inf_0) ) )
        else:
            return None
    
    def nash_res(self, row):
        """NASH resolution"""
        bal_1 = row.loc[self.ct['b_post']]
        inf_1 = row.loc[self.ct['i_post']]
        
        if not any(np.isnan([bal_1, inf_1])):
            return (bal_1 == 0) & (inf_1 <= 1)
        else:
            return None
        
    def nash_res_bl(self, row):
        """NASH resolution at Baseline"""

        bal_0 = row.loc[self.ct['b_pre']]
        inf_0 = row.loc[self.ct['i_pre']]
        
        if not any(np.isnan([bal_0, inf_0])):
            return (bal_0 == 0) & (inf_0 <= 1)
        else:
            return None
    
    def nash_f4_bl(self, row):
        """NASH resolution at Baseline"""

        fib_0 = row.loc[self.ct['f_pre']]
        
        if not any(np.isnan([fib_0])):
            return (fib_0 == 4)
        else:
            return None
        
    def nash_res_nas_2pt_red(self,row):
        """NASH resolution and NAS 2pt reduciton"""
        nash_res = self.nash_res(row)
        nas_2pt = self.nas_2pt_red(row)
        
        if nash_res is not None and nas_2pt is not None:
            return nash_res & nas_2pt
        else:
            return None
        
    def nash_res_nas_no_worse_fib_2pt_red_mod(self,row):
        """NASH resolution and NAS 2pt reduciton"""
        nash_res = self.nash_res_no_worse_fib(row)
        nas_2pt = self.nas_2pt_red_1pt_bal_or_inf(row)
        
        if nash_res is not None and nas_2pt is not None:
            return nash_res & nas_2pt
        else:
            return None
        
    def nash_res_no_worse_fib(self, row):
        
        """NASH resolution with no worsening of Fibrosis"""
        fib_0 = row.loc[self.ct['f_pre']]
        fib_1 = row.loc[self.ct['f_post']]
        
        bal_1 = row.loc[self.ct['b_post']]
        
        inf_1 = row.loc[self.ct['i_post']]
        
        if not any(np.isnan([fib_0, fib_1, bal_1, inf_1])):
            return (fib_1 <= fib_0) & (bal_1 == 0) & (inf_1 <= 1)
        else:
            return None
        
    def fib_red_no_worse_nash(self, row):
        
        """Fibrosis reduction no worse NASH"""
        fib_0 = row.loc[self.ct['f_pre']]
        fib_1 = row.loc[self.ct['f_post']]
        
        bal_0 = row.loc[self.ct['b_pre']]
        bal_1 = row.loc[self.ct['b_post']]
        
        inf_0 = row.loc[self.ct['i_pre']]
        inf_1 = row.loc[self.ct['i_post']]
        
        if not any(np.isnan([fib_0, fib_1, bal_0, bal_1, inf_0, inf_1])):
            return (fib_1 < fib_0) & (bal_1 <= bal_0) & (inf_1 <= inf_0)
        else:
            return None
        
    def fib_red_nash_res(self, row):
        
        """Fibrosis reduction with NASH resolution"""
        fib_0 = row.loc[self.ct['f_pre']]
        fib_1 = row.loc[self.ct['f_post']]
        
        bal_0 = row.loc[self.ct['b_pre']]
        bal_1 = row.loc[self.ct['b_post']]
        
        inf_0 = row.loc[self.ct['i_pre']]
        inf_1 = row.loc[self.ct['i_post']]
        
        if not any(np.isnan([fib_0, fib_1, bal_1, inf_1])):
            return (fib_1 < fib_0) & (bal_1 == 0) & (inf_1 <= 1)
        else:
            return None
        
    def fib_red_no_worse_nas(self, row):
        """Fibrosis reduction no worse NAS"""
        fib_0 = row.loc[self.ct['f_pre']]
        fib_1 = row.loc[self.ct['f_post']]
        
        nas_0 = row.loc[[self.ct['i_pre'],self.ct['b_pre'],self.ct['s_pre']]].sum()
        nas_1 = row.loc[[self.ct['i_post'],self.ct['b_post'],self.ct['s_post']]].sum()
        
        if not any(np.isnan([fib_0, fib_1, nas_0, nas_1])):
            return (fib_1 < fib_0) & (nas_1 <= nas_0)
        else:
            return None


class PrimaryEndpointStats:
    def __init__(self, df, tcol, t0, t1, strata:list = None):
        # add data
        
        # add legends, etc
        self.tcol = tcol
        self.t1 = t1
        self.t0 = t0
        
        self.tdf = df[df[self.tcol].isin([self.t0,self.t1])].copy()
        self.stat_list = []

    def _add_strata(self, strata:list):
        if strata is None:
            self.strata = None
            self.tdf['strata'] = 'No Strata'
        else:
            self.strata = strata
            self.tdf['strata'] = self.tdf.apply(self._combine_strata, axis=1)        

    def _combine_strata(self, row):
        slist = [str(row[c]) for c in self.strata]
        if not None in slist:
            return "_".join(slist)
        else:
            return None
    
    def _convert_endpoint(self,pep):
        # conver 0/1, T/F to Y/N
        self.tdf[pep] = self.tdf[pep].replace({
            True:'Y',False:'N', 1:'Y',0:'N',
        })
    
    def _stats_list_to_df(self):
        self.stats_df = pd.DataFrame(self.stat_list)
    
    def clear_stats(self):
        self.pep_list = []
    
    def get_stats(self, pep, strata:list = None):
        self._convert_endpoint(pep)
        self._add_strata(strata)
        
        tdf = self.tdf.copy()
        tdf = tdf[tdf[pep].isin(['N','Y'])]
        ct_out = pd.crosstab(tdf[self.tcol],tdf[pep],margins=True)
        cmh_out = CMH_test(tdf,self.tcol,pep,'strata',treat_order=[self.t0, self.t1],outcome_order=['N','Y'])
        chisq_out = chi_sq_test(tdf,self.tcol,pep,treat_order=[self.t0, self.t1],outcome_order=['N','Y'])
        
        pepd = {}
        pepd['End Point'] = pep
        pepd[f'N {self.t1}'] = ct_out['All'][self.t1]
        pepd[f'N {self.t0}'] = ct_out['All'][self.t0]
        pepd[f'N Respond {self.t1}'] = ct_out['Y'][self.t1]
        pepd[f'N Respond {self.t0}'] = ct_out['Y'][self.t0]
        pepd[f'Response Rate {self.t1}'] = ct_out['Y'][self.t1]/ct_out['All'][self.t1]
        pepd[f'Response Rate {self.t0}'] = ct_out['Y'][self.t0]/ct_out['All'][self.t0]
        pepd['OR'] = cmh_out['OR']
        pepd['OR (LB)'] = cmh_out['OR (LB)']
        pepd['OR (UB)'] = cmh_out['OR (UB)']
        pepd['CMH p'] = cmh_out['Test P']
        pepd['CHISQ p'] = chisq_out['Test P']
        
        
        self.stat_list.append(pepd)
        self._stats_list_to_df()
    
    def plot_response_rate(self, pep, title=None):
        plot_df = self.stats_df[self.stats_df['End Point']==pep].copy()
        rr_df = self.stats_df[[f'Response Rate {self.t0}',f'Response Rate {self.t1}']]
        labels = [self.t0, self.t1]      
        
        pp = rr_df.T
        ax = pp.plot.bar(color=['tab:red'],figsize=(6,4))

        try:
            stats_str =  f"\n OR: {np.around(plot_df['OR'].iloc[0],3)}({np.around(plot_df['OR (LB)'].iloc[0],3)},{np.around(plot_df['OR (UB)'].iloc[0],3)}) | CMH p: {plot_df['CMH p'].iloc[0]:.1e} | CHI2 p: {plot_df['CHISQ p'].iloc[0]:.1e}"
        except:
            stats_str =  f"\n OR: NA | CMH p: {plot_df['CMH p'].iloc[0]:.1e} | CHI2 p: {plot_df['CHISQ p'].iloc[0]:.1e}"

        if title is None:
            ax.set_title(f'{pep}'+stats_str)
        else:
            ax.set_title(title+stats_str)

        ax.set_xlabel('')
        ax.set_ylabel('Response Rate')
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0],ylim[1]+0.02])

        for i, v in enumerate(pp.iloc[:,0]):
            if i == 0:
                n_r = plot_df[f'N Respond {self.t0}'].iloc[0]
                n = plot_df[f'N {self.t0}'].iloc[0]
            else:
                n_r = plot_df[f'N Respond {self.t1}'].iloc[0]
                n = plot_df[f'N {self.t1}'].iloc[0]

            txt = f"{np.around(v,decimals=3)} ({n_r} / {n})"
            ax.text(i-.24, v+.005, txt)

        ax.set_xticklabels(labels, rotation=0, ha='center')
        ax.set_title(title,fontsize=12,loc='right')
        ax.set_ylabel(tw(ax.get_ylabel(),40))
        ax.set_xlabel(tw(ax.get_xlabel(),60))
        ax.get_legend().remove()
        plt.show()
        
        
class ContinuousEndpointStats:

    def __init__(self, df, tcol, t0, t1, strata: list = None):
        # add data

        # add legends, etc
        self.tcol = tcol
        self.t1 = t1
        self.t0 = t0

        self.tdf = df[df[self.tcol].isin([self.t0, self.t1])].copy()
        self.stat_list = []

    def _add_strata(self, strata: list):
        if strata is None:
            self.strata = None
            self.tdf['strata'] = 'No Strata'
        else:
            self.strata = strata
            self.tdf['strata'] = self.tdf.apply(self._combine_strata, axis=1)

    def _combine_strata(self, row):
        slist = [str(row[c]) for c in self.strata]
        if not None in slist:
            return "_".join(slist)
        else:
            return None

    def _stats_list_to_df(self):
        self.stats_df = pd.DataFrame(self.stat_list)

    def clear_stats(self):
        self.pep_list = []

    def get_stats(self, cep, strata: list = None, test: str = 'mw'):
        self._add_strata(strata)

        tdf = self.tdf.copy()
        pepd = {}

        if test == 'mw':
            # do man whitney test (without strata)
            trt = tdf[tdf[self.tcol] == self.t1][cep].dropna()
            plac = tdf[tdf[self.tcol] == self.t0][cep].dropna()
            _, p_mu = mannwhitneyu(trt, plac)

            pepd['End Point'] = cep
            pepd[f'N {self.t1}'] = len(trt)
            pepd[f'N {self.t0}'] = len(plac)
            pepd[f'Median {self.t1}'] = np.median(trt)
            pepd[f'Median {self.t0}'] = np.median(plac)
            pepd[f'Median Dif'] = np.median(trt) - np.median(plac)
            pepd['Test'] = 'MW'
            pepd['p'] = p_mu

        elif test == 'med':
            # do median test
            trt = tdf[tdf[self.tcol] == self.t1][cep].dropna()
            plac = tdf[tdf[self.tcol] == self.t0][cep].dropna()
            _, p_med,_,_ = median_test(trt, plac)
            
            pepd['End Point'] = cep
            pepd[f'N {self.t1}'] = len(trt)
            pepd[f'N {self.t0}'] = len(plac)
            pepd[f'Median {self.t1}'] = np.median(trt)
            pepd[f'Median {self.t0}'] = np.median(plac)
            pepd[f'Median Dif'] = np.median(trt) - np.median(plac)
            pepd['Test'] = 'MED'
            pepd['p'] = p_med
            
        elif test == 've':
            # do vanElteren test (with strata) (low power)
            vet = vanElterenTest()
            trt = tdf[tdf[self.tcol] == self.t1][cep].dropna()
            plac = tdf[tdf[self.tcol] == self.t0][cep].dropna()
            p = vet.test(tdf, cep, self.tcol, self.t0, self.t1, strata_col='strata')

            pepd['End Point'] = cep
            pepd[f'N {self.t1}'] = len(trt)
            pepd[f'N {self.t0}'] = len(plac)
            pepd[f'Median {self.t1}'] = np.median(trt)
            pepd[f'Median {self.t0}'] = np.median(plac)
            pepd[f'Median Dif'] = np.median(trt) - np.median(plac)
            pepd['Test'] = 'VE'
            pepd['p'] = p

        elif test == 'med_strata':
            # do stratified median test (supports strata through CMH test)
            print('not implimented')

        else:
            # raise value error
            print('test',test,'not recognized')

        self.stat_list.append(pepd)
        self._stats_list_to_df()

    def plot_continuous_endpoint(self, cep, strata: list = None, title: str = None, add_stats: bool = False):
        if strata is not None:
            self._add_strata(strata)
        
        tdf = self.tdf.copy()
        
        ax = sns.boxplot(data=tdf,
                         y=cep,
                         x = self.tcol,
                         saturation=0, width=.3, linewidth=1, fliersize=0,color='lightgray')
        
        if strata is not None:
            ax = sns.swarmplot(data=tdf, x=self.tcol, y=cep, hue='strata')
        else:
            ax = sns.swarmplot(data=tdf, x=self.tcol, y=cep)
        
        if add_stats:
            x0 = tdf[tdf[self.tcol]==self.t0][cep].dropna()
            x1 = tdf[tdf[self.tcol]==self.t1][cep].dropna()
            mw,p_mw = mannwhitneyu(x0, x1)
            mt,p_mt,_,_ = median_test(x0, x1)
            
            stats_str =  f"\n mw: ({np.around(mw,3)}, {p_mw:.1e}) med: ({np.around(mt,3)}, {p_mt:.1e})"
        else:
            stats_str = ''
        
        if title is None:
            ax.set_title(f'{cep}'+stats_str)
        else:
            ax.set_title(title+stats_str)
            
        ax.set_ylabel(tw(ax.get_ylabel(),40))
        ax.set_xlabel(tw(ax.get_xlabel(),60))
        
        ax.axhline(0, ls=':', lw=.5, c='k', zorder=0)
#         plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
    
    def plot_paired_slope_endpoint(self, cep, title: str = None, add_stats: bool = False):
        fig, ax = plt.subplots(figsize=(4, 3))
        
        tdf = self.tdf.copy()
        
        # Set up the x-axis values
        x1 = i - 0.2
        x2 = i + 0.2

        # Plot the lines connecting the dots
        for hi, di in zip(h, d):
            ax.plot([x1, x2], [hi, di], c='gray')

        # Plot the points
        ax.scatter(len(h)*[x1-0.01], h, c='k',
                   s=25, label='healthy')
        ax.scatter(len(d)*[x2+0.01], d, c='k',
                   s=25, label='disease')
        
        if add_stats:
            x0 = tdf[tdf[self.tcol]==self.t0][cep].dropna()
            x1 = tdf[tdf[self.tcol]==self.t1][cep].dropna()
            mw,p_mw = mannwhitneyu(x0, x1)
            mt,p_mt,_,_ = median_test(x0, x1)
            
            stats_str =  f"\n mw: ({np.around(mw,3)}, {p_mw:.1e}) med: ({np.around(mt,3)}, {p_mt:.1e})"
        else:
            stats_str = ''
        
        if title is None:
            ax.set_title(f'{cep}'+stats_str)
        else:
            ax.set_title(title+stats_str)
            
        ax.set_ylabel(tw(ax.get_ylabel(),40))
        ax.set_xlabel(tw(ax.get_xlabel(),60))
        
        # Set up list to track sites
        sites = []
        i = 1.0
        for site, subdf in df.groupby('site'):
            sites.append(site)
            # Get the values for healthy and disease patients
            h = subdf.query('label == "healthy"')['value'].values
            d = subdf.query('label == "disease"')['value'].values

         
            # Update x-axis
            i += 1

        # Fix the axes and labels
        ax.set_xticks([1, 2, 3])
        _ = ax.set_xticklabels(sites, fontsize='x-large')



    