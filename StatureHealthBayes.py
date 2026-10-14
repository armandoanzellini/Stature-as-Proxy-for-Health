# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:46:24 2026
Updated on Oct 14 2026

@author: Armando Anzellini
"""

import json
import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, floor
from scipy import stats
from itertools import combinations
from matplotlib.ticker import AutoMinorLocator

# Load variable from .env file that defines the path to the dataset xlsx file
with open("env.json", "r") as f:
    config = json.load(f)

# Fetch the path securely from the environment
direct = config['data_path']

file = r"\StatureData.xlsx"

mstats = pd.read_excel(direct + file, sheet_name='Male',   index_col='Site')
fstats = pd.read_excel(direct + file, sheet_name='Female', index_col='Site')

# Use site names for plots and results of Bayes ANOVA; only need to get from one sex
exclude = ['La Petaca', 'Diablo Wasi']
site_names = [site for site in list(mstats.index) if site not in exclude]

# make box-plots
# Calculate Q1 and Q3 first
def q1(mean, std):
    return mean - (0.6745*std)
    
def q3(mean, std):
    return mean + (0.6745*std)

# estimate minimum and maximum of Ancón as if it were a Student's t dist
def minimum(df):
    IQR = df['q3'] - df['q1']
    maxim = df['q3'] + (1.*IQR)
    minim = df['q1'] - (1.*IQR)
    return minim, maxim

# apply functions to df to get quartiles
mstats['q1'] = q1(mstats['Mean'], mstats['STD'])
mstats['q3'] = q3(mstats['Mean'], mstats['STD'])

fstats['q1'] = q1(fstats['Mean'], fstats['STD'])
fstats['q3'] = q3(fstats['Mean'], fstats['STD'])

# get dictionaries of values needed for boxplots
stat_dict = {'Mean' : 'med', 'Min': 'whislo', 'Max' : 'whishi'} # define rename dict

bmstats = mstats[['Mean', 'q1', 'q3', 'Min', 'Max']].rename(stat_dict, axis=1)
bfstats = fstats[['Mean', 'q1', 'q3', 'Min', 'Max']].rename(stat_dict, axis=1)

# remove La Petaca from Anatomical Stature Box Plots
bmstats = bmstats.loc[site_names]
bfstats = bfstats.loc[site_names]

# make them list of dicts
bml = [value for value in bmstats.to_dict(orient='index').values()]
bfl = [value for value in bfstats.to_dict(orient='index').values()]

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 15))
# male boxplots
ax1.bxp(bml, showfliers=False)
ax1.set_xticklabels(list(bmstats.index), fontsize=14)
ax1.set_ylabel('Anatomical Stature (cm)', fontsize=14)
ax1.set_title('Distribution of Male Stature', fontsize=14)
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# female boxplots
ax2.bxp(bfl, showfliers=False)
ax2.set_xticklabels(list(bfstats.index), fontsize=14 )
ax2.set_ylabel('Anatomical Stature (cm)', fontsize=14)
ax2.set_title('Distribution of Female Stature', fontsize=14)
ax2.yaxis.set_minor_locator(AutoMinorLocator())

# plt.savefig(direct + r"\CombinedStat_dist.png", dpi = 800, bbox_inches='tight')
plt.show()


# just defining linestyles for plotting later
linestyles = [
             'solid', 
             'dashed', 
             'dashdot',
             (5, (10, 3)),
             (0, (5, 1)),
             (0, (3, 10, 1, 10)),
             (0, (3, 1, 1, 1)),
             (0, (3, 5, 1, 5, 1, 5)),
             (0, (3, 10, 1, 10, 1, 10)),
             (0, (3, 1, 1, 1, 1, 1))
             ]


# Start statistical analysis
# Bayesian reanylisis from summary stats using a hierarchical model
# Hierarchical model of multigroup comparisons 2009 A FirstCourseinBayesian
# get standard error for each group

def bayes_ANOVA(df, label, savefig=False):
    df['SE'] = df['STD']/np.sqrt(df['N'])
    
    gm = sum(df['N']*df['Mean'])/df['N'].sum() # grand mean to use in Bayes ANOVA
    
    with pm.Model() as bayes_anova:
        # --- Priors ---
        # Grand mean (weakly informative; center on overall observed mean)
        grand_mean = pm.Normal("grand_mean", mu=gm, sigma=10)
    
        # Between-group SD (assume 0 and check if different from it but always  >0)
        beta          = pm.HalfNormal('beta', sigma=5)
        sigma_between = pm.HalfCauchy("sigma_between", beta=beta)
    
        # True latent mean for each group
        true_group_means = pm.Normal("true_group_means",
                                      mu=grand_mean,
                                      sigma=sigma_between,
                                      shape=len(df))
    
        # --- Likelihood ---
        obs = pm.StudentT("obs",
                        nu=df['N'] - 1,
                        mu=true_group_means,
                        sigma=df['SE'],          # known (estimated) SE
                        observed=df['Mean'])
    
    
        trace = pm.sample(4000, tune=2000, core=1, chains=4, target_accept=0.95, random_seed=42)
    
    # Inspect results by extracting data and plotting    
    summarydf = az.summary(trace, var_names=["true_group_means", "grand_mean", "sigma_between"]) 
    
    # rename summary df to have site names
    sites = list(df.index)
    summarydf = summarydf.rename(index=dict(zip(summarydf.index[:len(sites)], sites)))
    summarydf.to_excel(direct + rf"\ANOVAstats_{label}.xlsx")
    
    # Extract all posterior samples for group means and sigma_between
    post_means = trace.posterior["true_group_means"].values  # shape: (chains, draws, groups)
    sigmas = trace.posterior["sigma_between"].values.flatten()
    
    # Flatten chains and draws together for group means
    samples = post_means.reshape(-1, 5).T  # shape: (groups, total_draws)
    
    xmin = floor(np.min(samples))
    xmax = ceil(np.max(samples))
    
    # Plot the posterior traces of all vairables
    fig, ((ax0, ax2), (ax1, ax3)) = plt.subplots(2, 2, height_ratios=[4, 1.6], figsize=(14, 6))
    ax1.sharex(ax0)
    ax3.sharex(ax2)
    for i in range(len(samples)):
        # def const and distributions
        dens0 = stats.gaussian_kde(samples[i])
        hdi0 = az.hdi(samples[i], hdi_prob=0.94)
        
        #plot hist and line
        ax0.hist(samples[i], bins=40, density=True, color="steelblue", alpha=0.25, edgecolor='k')
        ax0.plot(x0 := sorted(samples[i]), dens0(x0), 
                color='k', alpha=0.8, linestyle=linestyles[i],
                label=sites[i])
        
        
        err = ax1.errorbar(x=np.mean(x0), y=site_names[i], 
                     xerr=[[np.mean(x0) - hdi0[0]], [hdi0[1] - np.mean(x0)]],
                     color='k',
                     marker='o',alpha=0.8)
        err[-1][0].set_linestyle(linestyles[i])
        ax1.spines[['top', 'right', 'left']].set_visible(False)
        ax1.set_xlabel('94% Highest Density Interval (HDI)')
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(left=False)
        ax1.set_yticklabels([])
        ax1.margins(0.2)
        
        ax0.set_title(f'Post. Distributions for Stature by Site - {label}')
        ax0.set_xlabel('Post. Estimated Mean')
        ax0.xaxis.set_minor_locator(AutoMinorLocator())
        ax0.spines[['top', 'right', 'left']].set_visible(False)
        ax0.set_yticklabels([])
        ax0.tick_params(left=False)
        
    ax0.legend(loc='upper center', bbox_to_anchor=(1, 1), fontsize='small', frameon=False)
    ax1.set_xlim(left=xmin, right=xmax)
    
    dens1 = stats.gaussian_kde(sigmas)
    hdi1 = az.hdi(sigmas, hdi_prob=0.94)
    
    #plot hist and line
    ax2.hist(sigmas, bins=40, density=True, color="steelblue", alpha=0.25, edgecolor='k')
    ax2.plot(x1 := sorted(sigmas), dens1(x1), color='black', alpha=0.8)
    ax2.axvline(0, linestyle=":", color='k')
    
    ax3.errorbar(x=np.mean(sigmas), y='Sigma', 
                 xerr=[[np.mean(sigmas) - hdi1[0]], [hdi1[1] - np.mean(sigmas)]],
                 color='k',
                 marker='o',alpha=0.8)
    ax3.axvline(0, linestyle=":", color='k')
    
    ax3.spines[['top', 'right', 'left']].set_visible(False)
    ax3.set_xlabel('94% Highest Density Interval (HDI)')
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.tick_params(left=False)
    ax3.set_yticklabels([])
    
    ax2.set_title(f'Between-Group Variance ($\\sigma^2$) - {label}')
    ax2.set_xlabel('$\\sigma^2$')
    ax2.spines[['top', 'right', 'left']].set_visible(False)
    ax2.set_yticklabels([])
    ax2.tick_params(left=False)
    
    peak = dens1(x1)[np.argmax(dens1(x1))] # get max bin size to place label
    ax2.annotate(f'  $\\sigma^2$ = {np.mean(sigmas):.2f}\n \
{'—'.join(np.round(az.hdi(sigmas, hdi_prob=0.94), 2).astype(str))}',
                    xy=(hdi1[1], 0.6*peak),
                    xycoords='data',
                    fontsize='small')
    
        
    # get extent of 90% of cauchy to limit plot
    beta = az.summary(trace, var_names=['beta'])['mean'].values
    xmin2 = -2.0
    xmax2 = np.mean(sigmas) + 6.314*beta # 90% of values in dist
    ax2.set_xlim(left=xmin2, right=floor(xmax2))
        
    if savefig:
        plt.savefig(direct + rf"\BetweenVarDist_{label}.png", dpi = 800, bbox_inches='tight')
    else:
        plt.show()
    
    
    # Make table of results
    post = trace.posterior["true_group_means"]
    anova_resdict = {}
    
    for i, j in combinations(post['true_group_means_dim_0'].values, 2):
        diff = (post.isel(true_group_means_dim_0=j).values.flatten() - 
                post.isel(true_group_means_dim_0=i).values.flatten())
        
        hdi = az.hdi(diff, hdi_prob=0.94)
        
        key = f'{sites[j]} vs {sites[i]}'
        
        anova_resdict[key] = {'Difference of Means' : round(diff.mean(), 3),
                              '94% HDI'         : f'{hdi[0]:.3f}-{hdi[1]:.3f}'}
        
        print(f"{site_names[j]} vs {site_names[i]}:")
        print(f"  Mean difference: {diff.mean():.3f}")
        print(f"  94% HDI:         [{hdi[0]:.3f}, {hdi[1]:.3f}]")
        
    
    resdf = pd.DataFrame.from_dict(anova_resdict, orient='index').sort_index()
    resdf.to_excel(direct + rf"\PairwiseComp_{label}.xlsx")


def BEST(df1, df2, label, savefig=False):
    se_1 = (std_1 := df1['STD']) / np.sqrt(n_1 := df1['N'])
    se_2 = (std_2 := df2['STD']) / np.sqrt(n_2 :=df2['N'])
    
    mu_m = np.average([df1['Mean'],df2['Mean']], weights=[n_1, n_2])
    mu_s = np.sqrt((std_1**2 + std_2**2)) * 2  # pooled, diffuse
    
    with pm.Model() as BEST_summary:
    
        # --- Priors ---
        mu_1 = pm.Normal("mu_1", mu=mu_m, sigma=mu_s)
        mu_2 = pm.Normal("mu_2", mu=mu_m, sigma=mu_s)
    
        # nu fixed at n-1 since we can't estimate it from summary stats
        nu_1 = n_1 - 1
        nu_2 = n_2 - 1
    
        # --- Likelihoods (StudentT on observed group means) ---
        obs_1 = pm.StudentT("obs_1", nu=nu_1, mu=mu_1, sigma=se_1, observed=df1['Mean'])
        obs_2 = pm.StudentT("obs_2", nu=nu_2, mu=mu_2, sigma=se_2, observed=df2['Mean'])
    
        # --- Deterministics ---
        diff_means  = pm.Deterministic("diff_means",  mu_1 - mu_2)
        effect_size = pm.Deterministic("effect_size",
                          diff_means / np.sqrt((std_1**2 + std_2**2) / 2))  # pooled SD fixed
    
        trace = pm.sample(2000, tune=2000, cores=1, chains=4, target_accept=0.95, random_seed=42)
    
    # The key posteriors Kruschke plots available for these data
    summarydf = az.summary(trace) 
    
    # rename summary df to have site names
    summarydf = summarydf.rename(index=dict(zip(summarydf.index[:2], ['Kuelap', 'La Petaca'])))
    summarydf.to_excel(direct + rf"\BESTstats_{label}.xlsx")
    
    samples1     = trace.posterior["mu_1"].values.flatten()
    samples2     = trace.posterior["mu_2"].values.flatten()
    samples_diff = trace.posterior["diff_means"].values.flatten()
    samples_es   = trace.posterior["effect_size"].values.flatten()
    
    samples = np.stack([samples1, samples2])
    
    xmin = floor(np.min(samples))
    xmax = ceil(np.max(samples))
    
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, height_ratios=[8, 1])
    for j,i in {'Kuelap' : 0, 'La Petaca' : 1}.items():
        # def const and distributions
        dens = stats.gaussian_kde(samples[i])
        hdi = az.hdi(samples[i], hdi_prob=0.94)
        
        #plot hist and line
        ax0.hist(samples[i], bins=40, density=True, color="steelblue", alpha=0.25, edgecolor='k')
        ax0.plot(x := sorted(samples[i]), dens(x), 
                color='k', alpha=0.8, linestyle=linestyles[i],
                label=j)
        
        
        err = ax1.errorbar(x=np.mean(x), y=i, 
                     xerr=[[np.mean(x) - hdi[0]],
                           [hdi[1]- np.mean(x)]],
                     color='k',
                     marker='o', alpha=0.8)
        err[-1][0].set_linestyle(linestyles[i])
        ax1.spines[['top', 'right', 'left']].set_visible(False)
        ax1.set_xlabel('94% Highest Density Interval (HDI)')
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(left=False)
        ax1.set_yticklabels([])
        ax1.margins(0.5)
        
        ax0.set_title(f'BEST Distributions - {label}')
        ax0.set_xlabel("Post. Estimated Mean")
        ax0.xaxis.set_minor_locator(AutoMinorLocator())
        ax0.spines[['top', 'right', 'left']].set_visible(False)
        ax0.set_yticklabels([])
        ax0.tick_params(left=False)
    
    ax0.legend(loc='upper center', bbox_to_anchor=(.9, 1), fontsize='medium', frameon=False)
    ax1.set_xlim(left=xmin, right=xmax)
    
    if savefig:
        plt.savefig(direct + rf"\BESTDist_{label}.png", dpi = 800, bbox_inches='tight')
    else:
        plt.show()
    
    
    
    sample_stats = np.stack([samples_diff, samples_es])
    
    # Plot the stats value distributions
    fig, (ax0, ax1) = plt.subplots(2,2, sharex='col', figsize=[10,5], height_ratios=[11, 1])
    for j,i in {'Difference between Group Means' : 0, 'Effect Size' : 1}.items():
        dens = stats.gaussian_kde(sample_stats[i])
        hdi = az.hdi(sample_stats[i], hdi_prob=0.94)
        
        #plot hist and line
        ax0[i].hist(sample_stats[i], bins=40, density=True, color="steelblue", alpha=0.25, edgecolor='k')
        ax0[i].plot(x := sorted(sample_stats[i]), dens(x), color='black', alpha=0.8)
        ax0[i].axvline(0, linestyle=":", color='k')
        ax0[i].set_title(j)
        
        ax1[i].errorbar(x=np.mean(sample_stats[i]), y='Sigma', 
                     xerr=[[np.mean(sample_stats[i]) - hdi[0]], 
                           [hdi[1] - np.mean(sample_stats[i])]],
                     color='k',
                     marker='o',alpha=0.8)
        ax1[i].axvline(0, linestyle=":", color='k')
        ax1[i].spines[['top', 'right', 'left']].set_visible(False)
        ax1[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax1[i].tick_params(left=False)
        ax1[i].set_yticklabels([])
        
        
        ax0[i].spines[['top', 'right', 'left']].set_visible(False)
        ax0[i].set_yticklabels([])
        ax0[i].tick_params(left=False)
        
        # annotate with data
        if j == 'Difference between Group Means':
            statlabl  = '$\\delta\\mu$'
            xlabl     = '$\\delta$'
            stat      = samples_diff
        elif j == 'Effect Size':
            statlabl = '$\\alpha$'
            xlabl    = '$\\alpha$'
            stat     = samples_es
        
        ax0[i].set_xlabel(xlabl)
        peak = dens(x)[np.argmax(dens(x))] # get max bin size to place label
        ax0[i].annotate(f'  {statlabl} = {stat.mean():.2f}\n \
{'—'.join(np.round(az.hdi(stat, hdi_prob=0.94), 2).astype(str))}',
                        xy=(hdi[1], 0.6*peak),
                        xycoords='data',
                        fontsize='small')
    
    
    # fig.suptitle(f'{label}')
    fig.supxlabel('94% Highest Density Interval (HDI)')
    
    if savefig:
        plt.savefig(direct + rf"\BEST_{label}.png", dpi = 800, bbox_inches='tight')
    else:
     plt.show() 

# --- DATA ---
# data for Bayes ANOVA
dfm = mstats.loc[site_names].sort_values('Mean')
dff = fstats.loc[site_names].sort_values('Mean')

# data for BEST for La Petaca v Kuelap (Krushke 2013)
kmdf = mstats.loc['Kuelap']
lpmdf = mstats.loc['La Petaca']

kfdf = fstats.loc['Kuelap']
lpfdf = fstats.loc['La Petaca']

savefigs = True

# --- Run ---
if __name__ == '__main__':
    bayes_ANOVA(dfm, 'Males',  savefig=savefigs)
    BEST(kmdf, lpmdf, 'Males', savefig=savefigs)
    
    bayes_ANOVA(dff, 'Females',  savefig=savefigs)
    BEST(kfdf, lpfdf, 'Females', savefig=savefigs)