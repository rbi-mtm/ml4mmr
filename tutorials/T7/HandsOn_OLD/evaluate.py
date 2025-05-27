import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import pandas as pd
from sklearn.metrics import r2_score
import scipy


def get_stats(targ_dict: dict, pred_dict: dict, 
              output_keys: list, grad_keys: list):
    test_stats = {}
    keys = [*output_keys, *grad_keys]
    
    ediff_keys = [['energy_1', 'energy_0'],
                  ['energy_2', 'energy_1']]
    
    for key in keys:
        try: 
            pred = torch.vstack(pred_dict[key]).numpy().flatten()  
        except:
            pred = torch.stack(pred_dict[key]).numpy().flatten()
            
        try:
            targ = torch.vstack(targ_dict[key]).numpy().flatten()
        except:
            targ = torch.stack(targ_dict[key]).numpy().flatten()
    
        pearson_r, p = scipy.stats.pearsonr(targ, pred)
        R2 = 1 - np.power((targ - pred), 2).sum()/np.power(targ - targ.mean(), 2).sum()
    
        mean = targ.mean()
        targ -= mean
        pred -= mean
        mae  = abs(pred-targ).mean()
        rmse = np.sqrt(np.power(pred-targ, 2).mean())
    
        test_stats[key] = {
                "MAE": np.float64(mae), 
                "RMSE": np.float64(rmse), 
                "pearson_r": np.float64(pearson_r),
                "R2": np.float64(R2)}
    
    for key1, key2 in ediff_keys:
        pred1 = torch.vstack(pred_dict[key1]).numpy().flatten()
        targ1 = torch.stack(targ_dict[key1]).numpy().flatten()
        pred2 = torch.vstack(pred_dict[key2]).numpy().flatten()
        targ2 = torch.stack(targ_dict[key2]).numpy().flatten()
        pred = pred1 - pred2
        targ = targ1 - targ2
        mae = abs(pred-targ).mean()
        rmse = np.sqrt(np.power(pred-targ, 2).mean())
        pearson_r, p = scipy.stats.pearsonr(targ, pred)
        R2 = 1 - np.power((targ - pred), 2).sum()/np.power(targ - targ.mean(), 2).sum()
    
        test_stats[f"delta_{key1}_{key2}"] = {
                "MAE": np.float64(mae), 
                "RMSE": np.float64(rmse), 
                "pearson_r": np.float64(pearson_r),
                "R2": np.float64(R2)}
    
    ##############################################################
    # Summary evaluations
    all_preds = []
    all_targs = []
    for key in output_keys:
        pred = torch.vstack(pred_dict[key]).numpy().flatten()
        targ = torch.stack(targ_dict[key]).numpy().flatten()
        all_preds.append(pred)
        all_targs.append(targ)
    targ = np.array(all_targs).flatten()
    pred = np.array(all_preds).flatten()
    
    pearson_r, p = scipy.stats.pearsonr(targ, pred)
    R2 = 1 - np.power((targ - pred), 2).sum()/np.power(targ - targ.mean(), 2).sum()  
    mean = targ.mean()
    targ -= mean
    pred -= mean
    mae  = abs(pred-targ).mean()
    rmse = np.sqrt(np.power(pred-targ, 2).mean())
    
    test_stats["energy"] = {
            "MAE": np.float64(mae), 
            "RMSE": np.float64(rmse), 
            "pearson_r": np.float64(pearson_r),
            "R2": np.float64(R2)}
    
    all_preds = []
    all_targs = []
    for key in grad_keys:
        pred = torch.vstack(pred_dict[key]).numpy().flatten()
        targ = torch.vstack(targ_dict[key]).numpy().flatten()
        all_preds.append(pred)
        all_targs.append(targ)
    targ = np.array(all_targs).flatten()
    pred = np.array(all_preds).flatten()
    
    pearson_r, p = scipy.stats.pearsonr(targ, pred)
    R2 = 1 - np.power((targ - pred), 2).sum()/np.power(targ - targ.mean(), 2).sum()  
    mean = targ.mean()
    targ -= mean
    pred -= mean
    mae  = abs(pred-targ).mean()
    rmse = np.sqrt(np.power(pred-targ, 2).mean())
    
    test_stats["energy_grad"] = {
            "MAE": np.float64(mae), 
            "RMSE": np.float64(rmse), 
            "pearson_r": np.float64(pearson_r),
            "R2": np.float64(R2)}
    
    all_preds = []
    all_targs = []
    for key1, key2 in ediff_keys:
        pred1 = torch.vstack(pred_dict[key1]).numpy().flatten()
        targ1 = torch.stack(targ_dict[key1]).numpy().flatten()
        pred2 = torch.vstack(pred_dict[key2]).numpy().flatten()
        targ2 = torch.stack(targ_dict[key2]).numpy().flatten()
        pred = pred1 - pred2
        targ = targ1 - targ2
        all_preds.append(pred)
        all_targs.append(targ)
    targ = np.array(all_targs).flatten()
    pred = np.array(all_preds).flatten()
    
    pearson_r, p = scipy.stats.pearsonr(targ, pred)
    R2 = 1 - np.power((targ - pred), 2).sum()/np.power(targ - targ.mean(), 2).sum()  
    mean = targ.mean()
    targ -= mean
    pred -= mean
    mae  = abs(pred-targ).mean()
    rmse = np.sqrt(np.power(pred-targ, 2).mean())
    
    test_stats["delta_energy"] = {
            "MAE": np.float64(mae), 
            "RMSE": np.float64(rmse), 
            "pearson_r": np.float64(pearson_r),
            "R2": np.float64(R2)}
    
    ##############################################################
    return test_stats

def make_scatterplot(name: str, 
                    targets: dict,
                    predictions: dict):

    figsize=(16, 14)
    fontsize      = 20
    labelsize     = 20
    titelsize     = labelsize * 1.2
    pad           = labelsize / 3
    tickwidth     = 3
    maj_tick_size = 6
    min_tick_size = 3
    dpi=100
    fig, axs = plt.subplots(3, 3, figsize=figsize, layout="constrained")

    units = {"energy": r"kcal/mol",
             "grad": r"kcal/mol Å"}

    min_max = {"energy" : np.array([1.e6, -1.e6]),
               "grad": np.array([1.e6, -1.e6]),
               "energy_diff": np.array([1.e6, -1.e6])}

    for state in range(3):
        for idx, key in enumerate(["energy", "grad"]):
            ax = axs[idx, state]
            if key=='energy':
                targ, pred = (torch.vstack(targets[f"{key}_{state}"]).numpy().flatten(), 
                                torch.vstack(predictions[f"{key}_{state}"]).numpy().flatten())
            else:
                targ, pred = (torch.vstack(targets[f"energy_{state}_grad"]).numpy().flatten(), 
                                torch.vstack(predictions[f"energy_{state}_grad"]).numpy().flatten())
            min_max[key][0] = min(min_max[key][0], targ.min(), pred.min())
            min_max[key][1] = max(min_max[key][1], targ.max(), pred.max())

            hb = ax.hexbin(targ, pred,
                           cmap='Blues',
                           gridsize=25,
                           mincnt=1,
                           bins="log",
                           edgecolors=None,
                           linewidths=(0.2,),
                           xscale="linear",
                           yscale="linear",
                           )
            pearson_r, p = scipy.stats.pearsonr(targ, pred)
            R2 = r2_score(targ, pred)
            mae = abs(pred-targ).mean()
            rmse = np.sqrt(np.power(pred-targ, 2).mean())
            ax.text(0.05, 0.9, 'MAE: %.2f' % (mae),
                   transform=ax.transAxes, fontsize=labelsize*0.9,
                   zorder=10)
            ax.text(0.05, 0.8, 'RMSE: %.2f' % (rmse),
                   transform=ax.transAxes, fontsize=labelsize*0.9,
                   zorder=10)
            ax.text(0.6, 0.2, r'$R^2$: %.2f' % (R2),
               transform=ax.transAxes, fontsize=labelsize*0.9,
               zorder=10)
            ax.text(0.6, 0.1, r'$\rho$: %.2f' % (pearson_r),
                   transform=ax.transAxes, fontsize=labelsize*0.9,
                   zorder=10)
            
    for state in range(2):
        key ="energy"
        ax = axs[2, state]
        targ, pred = (torch.vstack(targets[f"{key}_{state+1}"]).numpy().flatten() 
                      - torch.vstack(targets[f"{key}_{state}"]).numpy().flatten(), 
                      torch.vstack(predictions[f"{key}_{state+1}"]).numpy().flatten() 
                      - torch.vstack(predictions[f"{key}_{state}"]).numpy().flatten())
        min_max["energy_diff"][0] = min(min_max["energy_diff"][0], targ.min(), pred.min())
        min_max["energy_diff"][1] = max(min_max["energy_diff"][1], targ.max(), pred.max())

        hb = ax.hexbin(targ, pred,
                       cmap='Blues',
                       gridsize=25,
                       mincnt=1,
                       bins="log",
                       edgecolors=None,
                       linewidths=(0.2,),
                       xscale="linear",
                       yscale="linear",
                       )
        pearson_r, p = scipy.stats.pearsonr(targ, pred)
        R2 = r2_score(targ, pred)
        mae = abs(pred-targ).mean()
        rmse = np.sqrt(np.power(pred-targ, 2).mean())
        ax.text(0.05, 0.9, 'MAE: %.2f' % (mae),
               transform=ax.transAxes, fontsize=labelsize*0.9,
               zorder=10)
        ax.text(0.05, 0.8, 'RMSE: %.2f' % (rmse),
               transform=ax.transAxes, fontsize=labelsize*0.9,
               zorder=10)
        ax.text(0.6, 0.2, r'$R^2$: %.2f' % (R2),
           transform=ax.transAxes, fontsize=labelsize*0.9,
           zorder=10)
        ax.text(0.6, 0.1, r'$\rho$: %.2f' % (pearson_r),
               transform=ax.transAxes, fontsize=labelsize*0.9,
               zorder=10)
    


    for key, ax_row in zip(["energy", "grad"], axs[:-1]):
        cbar = fig.colorbar(hb, ax=ax_row.ravel().tolist(), location='right')
        cbar.set_label(label='Count', fontsize=labelsize)
        cbar.ax.tick_params(which='major', width=tickwidth, length=maj_tick_size, labelsize=fontsize)
        cbar.ax.tick_params(which='minor', width=tickwidth, length=min_tick_size, labelsize=fontsize)
        cbar.outline.set_linewidth(tickwidth)

        for ax in ax_row:
            ax.set_xlim(min_max[key]*1.1)
            ax.set_ylim(min_max[key]*1.1)
            ax.set_aspect('equal')
            ax.plot(min_max[key]*1.1,
                   min_max[key]*1.1,
                   color='#000000',
                   zorder=10,
                   linewidth=0.5)
            ax.spines['bottom'].set_linewidth(tickwidth)
            ax.spines['top'].set_linewidth(tickwidth)
            ax.spines['left'].set_linewidth(tickwidth)
            ax.spines['right'].set_linewidth(tickwidth)
            ax.tick_params(axis='x', length=maj_tick_size, width=tickwidth,
                           labelsize=labelsize, pad=pad,
                           direction='in')
            ax.tick_params(axis='y', length=0, width=tickwidth,
                   labelsize=0, pad=0,
                   direction='in')
            
    ax_row = axs[2]
    key = 'energy'
    cbar = fig.colorbar(hb, ax=ax_row.ravel().tolist(), location='right')
    cbar.set_label(label='Count', fontsize=labelsize)
    cbar.ax.tick_params(which='major', width=tickwidth, length=maj_tick_size, labelsize=fontsize)
    cbar.ax.tick_params(which='minor', width=tickwidth, length=min_tick_size, labelsize=fontsize)
    cbar.outline.set_linewidth(tickwidth)

    for ax in ax_row[:-1]:
        ax.set_xlim(min_max["energy_diff"]*1.1)
        ax.set_ylim(min_max["energy_diff"]*1.1)
        ax.set_aspect('equal')
        ax.plot(min_max["energy_diff"]*1.1,
               min_max["energy_diff"]*1.1,
               color='#000000',
               zorder=10,
               linewidth=0.5)
        ax.spines['bottom'].set_linewidth(tickwidth)
        ax.spines['top'].set_linewidth(tickwidth)
        ax.spines['left'].set_linewidth(tickwidth)
        ax.spines['right'].set_linewidth(tickwidth)
        ax.tick_params(axis='x', length=maj_tick_size, width=tickwidth,
                       labelsize=labelsize, pad=pad,
                       direction='in')
        ax.tick_params(axis='y', length=0, width=tickwidth,
               labelsize=0, pad=0,
               direction='in')

    for ax in axs[:,0]:
        ax.tick_params(axis='y', length=maj_tick_size, width=tickwidth,
                   labelsize=labelsize, pad=pad,
                   direction='in')

    # titles
    for idx, ax in enumerate(axs[0]):
        ax.set_title("$E_{S_%i}$" % idx, fontsize=titelsize, pad=2*pad)
    for idx, ax in enumerate(axs[1]):
        ax.set_title(r"$\partial E_{S_%i} / \partial \mathbf{R}$" % idx, fontsize=titelsize, pad=2*pad)
    for idx, ax in enumerate(axs[2][:-1]):
        ax.set_title("$E_{S_%i} - E_{S_%i}$" % (idx+1, idx), fontsize=titelsize, pad=2*pad)

    # axis labels
    for idx, ax in enumerate(axs[0]):
        ax.set_xlabel(f"Target [{units['energy']}]", fontsize=fontsize)
        if idx==0:
            ax.set_ylabel(f"Prediction [{units['energy']}]", fontsize=fontsize)
    for idx, ax in enumerate(axs[1]):
        ax.set_xlabel(f"Target [{units['grad']}]", fontsize=fontsize)
        if idx==0:
            ax.set_ylabel(f"Prediction [{units['grad']}]", fontsize=fontsize)
    for idx, ax in enumerate(axs[2][:-1]):
        ax.set_xlabel(f"Target [{units['energy']}]", fontsize=fontsize)
        if idx==0:
            ax.set_ylabel(f"Prediction [{units['energy']}]", fontsize=fontsize)

    ax = axs[-1, -1]
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.tick_params(axis='x', length=0, width=0,
                   labelsize=0, pad=0,
                   direction='in')
    ax.tick_params(axis='y', length=0, width=0,
           labelsize=0, pad=0,
           direction='in')
            
    plt.savefig(name, dpi=dpi)
    #plt.show()
    plt.close()