### import predictions from different scenarios, and observed data. Calculate the BS-prior from Sigurds paper.
# Find the optimal weights by optimization.

import numpy as np
from scipy.optimize import minimize
import pickle

def _bias_corr(pred,pred_hf):
    # Correction is an adjustment of the mean value, such that all levels have the same mean as the HF level
    tot_data = pred[0][0].keys()
    for c,elem in enumerate(pred):# one time index
        for dat in tot_data:
            ref_mean = elem[-1][dat] = pred_hf[c][dat].mean(axis=1) # calc HF mean
            if len(ref_mean):
                for level in range(len(elem)-1): # adjust the coarse means
                    coarse_mean =elem[level][dat].mean(axis=1)
                    if coarse_mean:
                        elem[level][dat] += (ref_mean - coarse_mean)
            pred[c][-1][dat] = pred_hf[c][dat]
    return pred

def _ml_bme(pred,data,inv_var,w):
    bme = []
    for p in pred:
        bme.append(_MC(p,inv_var,data))
    return np.sum([w[c]*el for c,el in enumerate(bme)])
def _MC(pred,inv_cd,data):
    r = pred - data[:, np.newaxis]
    mf = np.diag(-0.5*(r.T.dot(r*inv_cd[:,None])))
    idx = abs(mf).argmin()

    mf0 = mf[idx]

    return np.exp(mf0)

def find_weights(BME_S1,BME_S2,BME_S3,BME_S4):

    fun = lambda w, BME_S1,BME_S2,BME_S3,BME_S4: -np.mean(np.log([np.dot(w,np.array([s1,s2,s3,s4])) for (s1,s2,s3,s4) in
                                    zip(BME_S1,BME_S2,BME_S3,BME_S4)]))

    # Define the constraint: weights must sum to 1
    cons = ({'type': 'eq', 'fun': lambda w:  np.sum(w)-1})

    #Bounds
    bnds = ((0,None),(0,None),(0,None),(0,None))

    # Initial guess for weights
    w0 = np.array([0.25, 0.25, 0.25, 0.25])

    # Call the minimize function with the SLSQP method
    res = minimize(fun, w0,args=(BME_S1,BME_S2,BME_S3,BME_S4), method='SLSQP',bounds=bnds, constraints=cons)

    # Print the optimal weights
    #print("Optimal weights: ", res.x)
    return res.x

if __name__ == "__main__":
    # load predictions from different scenarios
    with open('../../SCENARIO_1/sim_results.p','rb') as f:
        pred_scenario_1 =pickle.load(f)
    with open('../../SCENARIO_1/hf_sim_results.p','rb') as f:
        pred_scenario_1_hf =pickle.load(f)
    with open('../../SCENARIO_2/sim_results.p','rb') as f:
        pred_scenario_2 =pickle.load(f)
    with open('../../SCENARIO_2/hf_sim_results.p','rb') as f:
        pred_scenario_2_hf =pickle.load(f)
    with open('../../SCENARIO_3/sim_results.p','rb') as f:
        pred_scenario_3 =pickle.load(f)
    with open('../../SCENARIO_3/hf_sim_results.p','rb') as f:
        pred_scenario_3_hf =pickle.load(f)
    with open('../../SCENARIO_4/sim_results.p','rb') as f:
        pred_scenario_4 =pickle.load(f)
    with open('../../SCENARIO_4/hf_sim_results.p','rb') as f:
        pred_scenario_4_hf =pickle.load(f)

    # Bias adjust the forecasts
    pred_scenario_1 = _bias_corr(pred_scenario_1, pred_scenario_1_hf)
    pred_scenario_2 = _bias_corr(pred_scenario_2, pred_scenario_2_hf)
    pred_scenario_3 = _bias_corr(pred_scenario_3, pred_scenario_3_hf)
    pred_scenario_4 = _bias_corr(pred_scenario_4, pred_scenario_4_hf)

    # Load data and variance
    obs = np.load('../../SCENARIO_1/obs_var.npz',allow_pickle=True)['obs']
    var = np.load('../../SCENARIO_1/obs_var.npz',allow_pickle=True)['var']

    list_data = obs[0].keys()
    # Calculate the ML BME

    tot_level = 7
    tot_assim_step = 247
    scale_variance = 1e5
    N_rep = 100

    tot_pred_s1 = []
    tot_pred_s2 = []
    tot_pred_s3 = []
    tot_pred_s4 = []

    for l in range(tot_level):
        tot_pred_s1.append(np.concatenate(tuple(pred_scenario_1[el][l][dat] for el in range(tot_assim_step) if pred_scenario_1[el][l]
                         is not None for dat in list_data if obs[el][dat] is not None and
                         len(pred_scenario_1[el][l][dat].shape)==2)))
        tot_pred_s2.append(np.concatenate(tuple(pred_scenario_2[el][l][dat] for el in range(tot_assim_step) if pred_scenario_2[el][l]
                                 is not None for dat in list_data if obs[el][dat] is not None and
                                 len(pred_scenario_2[el][l][dat].shape) == 2)))
        tot_pred_s3.append(np.concatenate(tuple(pred_scenario_3[el][l][dat] for el in range(tot_assim_step) if pred_scenario_3[el][l]
                                 is not None for dat in list_data if obs[el][dat] is not None and
                                 len(pred_scenario_3[el][l][dat].shape) == 2)))
        tot_pred_s4.append(np.concatenate(tuple(pred_scenario_4[el][l][dat] for el in range(tot_assim_step) if pred_scenario_4[el][l]
                                 is not None for dat in list_data if obs[el][dat] is not None and
                                 len(pred_scenario_4[el][l][dat].shape) == 2)))

    ne = [p.shape[1] for p in tot_pred_s1]
    w_n = [n/sum(ne) for n in ne]

    tot_obs = np.concatenate(tuple(
        obs[el][dat] for el in range(tot_assim_step) for dat in list_data if obs[el][dat] is not None
        and len(pred_scenario_4[el][0][dat].shape) == 2))
    tot_var = np.concatenate(tuple(
        var[el][dat] for el in range(tot_assim_step) for dat in list_data if obs[el][dat] is not None
        and len(pred_scenario_4[el][0][dat].shape) == 2))
    tot_std = np.sqrt(tot_var)
    tot_var *= scale_variance
    tot_inv = np.array([el**-1 for el in tot_var])

    BME_S1= []
    BME_S2= []
    BME_S3= []
    BME_S4= []
    for i in range(N_rep):
        eps = np.random.randn()
        BME_S1.append(_ml_bme(tot_pred_s1,tot_obs + tot_std*eps,tot_inv,w_n))
        eps = np.random.randn()
        BME_S2.append(_ml_bme(tot_pred_s2,tot_obs + tot_std*eps,tot_inv,w_n))
        eps = np.random.randn()
        BME_S3.append(_ml_bme(tot_pred_s3,tot_obs + tot_std*eps,tot_inv,w_n))
        eps = np.random.randn()
        BME_S4.append(_ml_bme(tot_pred_s4,tot_obs + tot_std*eps,tot_inv,w_n))

    weights_BS = find_weights(BME_S1,BME_S2,BME_S3,BME_S4)
    weights_BMA = np.mean([(s1/sum([s1,s2,s3,s4]),s2/sum([s1,s2,s3,s4]),s3/sum([s1,s2,s3,s4]),s4/sum([s1,s2,s3,s4]))
                   for (s1,s2,s3,s4) in zip(BME_S1,BME_S2,BME_S3,BME_S4)],axis=0)

    np.savez('weights.npz',**{'BS':weights_BS, 'BMA':weights_BMA})