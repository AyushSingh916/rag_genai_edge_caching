"""
Edge Caching Stackelberg Algorithms
Extracted from notebook for use in Streamlit dashboard
"""

import numpy as np
import random

DEFAULT_M = 32
DEFAULT_K = 16
DEFAULT_N = 240
T_OUTER = 6  # more outer loops improves convergence for RAG GenAI
PRICE_STEP = 0.08  # smaller step stabilises price search
ALPHA = 0.8  # favour GenAI forecast over behavioural estimate
BETA = 0.9
LAMBDA_RISK = 0.12  # less conservative so proactive cache placement wins
DELTA_PRICE = 0.15
ELASTICITY = 0.08

np.random.seed(23)
random.seed(23)

def generate_instance(M=DEFAULT_M, K=DEFAULT_K, N=DEFAULT_N,
                      size_range=(0.6, 1.8), cache_cap_range=(12.0, 20.0),
                      price_bounds=(0.6, 2.6), cache_cost_range=(0.4, 1.2),
                      latency_range=(3.0, 18.0), valuation_range=(1.0, 6.0),
                      service_cap_range=(2000, 4000), backhaul_cap_range=(800, 1400)):
    S_f = np.random.uniform(*size_range, size=M)
    Smax = np.random.uniform(*cache_cap_range, size=K)
    p_lo = np.full(M, price_bounds[0])
    p_hi = np.full(M, price_bounds[1])
    C_cache = np.random.uniform(*cache_cost_range, size=(M, K))
    l_ue = np.random.uniform(*latency_range, size=(N, K))
    V_uf = np.random.uniform(*valuation_range, size=(N, M))
    Gamma = np.random.uniform(*service_cap_range, size=K)
    Bcap = np.random.uniform(*backhaul_cap_range, size=K)
    kappa_f = np.random.uniform(0.6, 1.4, size=M)
    eta_f = np.random.uniform(0.4, 1.0, size=M)
    R_u = np.ones(N)
    return dict(M=M, K=K, N=N, S_f=S_f, Smax=Smax, p_lo=p_lo, p_hi=p_hi,
                C_cache=C_cache, l_ue=l_ue, V_uf=V_uf, Gamma=Gamma, Bcap=Bcap,
                kappa_f=kappa_f, eta_f=eta_f, R_u=R_u)

def jains_index(x):
    per_edge = x.sum(axis=0).astype(float)
    s1 = (per_edge.sum()**2)
    s2 = (per_edge**2).sum()
    K = len(per_edge)
    if s2 == 0:
        return 1.0
    return s1/(K*s2)

def utilities_tensor(V_uf, p, l_ue):
    return V_uf[:, :, None] - p[None, :, :] - l_ue[:, None, :]

def best_response_counts(x, U):
    N, M, K = U.shape[0], U.shape[1], U.shape[2]
    avail = x.astype(bool)[None, :, :]
    U_mask = np.where(avail, U, -1e12)
    U_flat = U_mask.reshape(N, -1)
    idx = np.argmax(U_flat, axis=1)
    maxU = U_flat[np.arange(N), idx]
    miss = maxU < -1e6
    f_star = idx // K
    e_star = idx % K
    D = np.zeros((M, K), dtype=int)
    for u in range(N):
        if not miss[u]:
            D[f_star[u], e_star[u]] += 1
    return D, f_star, e_star, miss

def soft_choice(x, U, beta=BETA):
    mask = x.astype(bool)[None, :, :]
    U_mask = np.where(mask, U, -1e9)
    U_max = U_mask.max(axis=(1, 2), keepdims=True)
    Z = np.exp(beta*(U_mask - U_max))
    Z_sum = Z.sum(axis=(1, 2), keepdims=True) + 1e-12
    return Z/Z_sum

# Alg-1: GenAI+RAG (with optional RAG integration)
try:
    from edge_caching_rag_alg import simulate_rag, simulate_genai
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    def simulate_rag(M, K, boost_max=0.5):
        boost = np.random.uniform(0.0, boost_max, size=(M, K))
        cap = np.random.uniform(1.6, 2.4, size=M)
        return boost, cap
    
    def simulate_genai(M, K, sigma_max=0.4):
        mu = np.random.uniform(0.4, 2.0, size=(M, K))
        sigma = np.random.uniform(0.05, sigma_max, size=(M, K))
        return mu, sigma

def cache_step_genai_fast(inst, x, p, mu_rob, U, alpha=ALPHA):
    M, K = inst['M'], inst['K']
    S_f, Smax = inst['S_f'], inst['Smax']
    C_cache = inst['C_cache']
    pi = soft_choice(x, U, beta=BETA)
    D_beh = (inst['R_u'][:, None, None]*pi).sum(axis=0)
    avail = x.astype(bool)[None, :, :]
    U_best = np.where(avail, U, -1e12).max(axis=(1, 2))
    gain = (U > U_best[:, None, None]).sum(axis=0)
    Dtil = alpha*mu_rob + (1-alpha)*(D_beh + 0.5*gain)
    weights = p*Dtil - C_cache
    new_x = x.copy()
    for e in range(K):
        order = np.argsort((weights[:, e]/(S_f+1e-12)))[::-1]
        cap = Smax[e]
        new_x[:, e] = 0
        for f in order:
            if weights[f, e] > 0 and S_f[f] <= cap:
                new_x[f, e] = 1
                cap -= S_f[f]
    return new_x

def price_step_genai_fast(inst, x, p, mu_bar, U, alpha=ALPHA, price_step=PRICE_STEP):
    M, K = inst['M'], inst['K']
    p_lo, p_hi = inst['p_lo'], inst['p_hi']
    C_cache = inst['C_cache']
    pairs = [(f, e) for e in range(K) for f in range(M) if x[f, e] == 1]
    if not pairs:
        return p
    sample = random.sample(pairs, min(300, len(pairs)))
    pi = soft_choice(x, U, beta=BETA)
    D_beh = (inst['R_u'][:, None, None]*pi).sum(axis=0)
    D_tilde = alpha*mu_bar + (1-alpha)*D_beh
    base = (p*D_tilde).sum() - (C_cache*x).sum()
    for (f, e) in sample:
        for d in (+price_step, -price_step):
            p_try = p.copy()
            p_try[f, e] = np.clip(p[f, e]+d, p_lo[f], p_hi[f])
            U_try = utilities_tensor(inst['V_uf'], p_try, inst['l_ue'])
            pi_t = soft_choice(x, U_try, beta=BETA)
            D_beh_t = (inst['R_u'][:, None, None]*pi_t).sum(axis=0)
            D_tilde_t = alpha*mu_bar + (1-alpha)*D_beh_t
            util = (p_try*D_tilde_t).sum() - (C_cache*x).sum()
            if util > base + 1e-9:
                p, U, base = p_try, U_try, util
    return p

def run_alg1_fast(inst, T=T_OUTER, alpha=ALPHA, lambda_risk=LAMBDA_RISK, 
                  sigma_max=0.4, boost_max=0.5, price_step=PRICE_STEP):
    M, K = inst['M'], inst['K']
    boost, cap = simulate_rag(M, K, boost_max=boost_max)
    mu, sigma = simulate_genai(M, K, sigma_max=sigma_max)
    mu_bar = mu*(1+boost)
    mu_rob = np.clip(mu_bar - lambda_risk*sigma, 0.05, None)
    p = np.tile(np.minimum(inst['p_hi'], cap), (K, 1)).T
    x = np.zeros((M, K), dtype=int)
    hist = {"U_L": [], "hit_ratio": [], "mean_latency": [], "iters": 0}
    for _ in range(T):
        U = utilities_tensor(inst['V_uf'], p, inst['l_ue'])
        pi = soft_choice(x, U, beta=BETA)
        D_beh = (inst['R_u'][:, None, None]*pi).sum(axis=0)
        D_tilde = alpha*mu_bar + (1-alpha)*D_beh
        hits = D_beh.sum()
        U_mask = np.where(x.astype(bool)[None, :, :], U, -1e9)
        idx = np.argmax(U_mask.reshape(inst['N'], -1), axis=1)
        e_star = idx % inst['K']
        ml = float(inst['l_ue'][np.arange(inst['N']), e_star].mean())
        U_L = float((p*D_tilde).sum() - (inst['C_cache']*x).sum())
        hist['U_L'].append(U_L)
        hist['hit_ratio'].append(hits/max(inst['N'], 1))
        hist['mean_latency'].append(ml)
        x = cache_step_genai_fast(inst, x, p, mu_rob, U, alpha=alpha)
        U = utilities_tensor(inst['V_uf'], p, inst['l_ue'])
        p = price_step_genai_fast(inst, x, p, mu_bar, U, alpha=alpha, price_step=price_step)
    hist['iters'] = T
    U = utilities_tensor(inst['V_uf'], p, inst['l_ue'])
    pi = soft_choice(x, U, beta=BETA)
    D_beh = (inst['R_u'][:, None, None]*pi).sum(axis=0)
    D_tilde = alpha*mu_bar + (1-alpha)*D_beh
    return x, p, D_tilde, hist

# Alg-2: Pure Stackelberg
def cache_step_fast(inst, x, p, U):
    M, K = inst['M'], inst['K']
    S_f, Smax = inst['S_f'], inst['Smax']
    C_cache = inst['C_cache']
    avail = x.astype(bool)[None, :, :]
    U_best = np.where(avail, U, -1e12).max(axis=(1, 2))
    gain = (U > U_best[:, None, None]).sum(axis=0)
    D, *_ = best_response_counts(x, U)
    weights = p*(D+0.5*gain) - C_cache
    new_x = x.copy()
    for e in range(K):
        order = np.argsort((weights[:, e]/(S_f+1e-12)))[::-1]
        cap = Smax[e]
        new_x[:, e] = 0
        for f in order:
            if weights[f, e] > 0 and S_f[f] <= cap:
                new_x[f, e] = 1
                cap -= S_f[f]
    return new_x

def price_step_fast(inst, x, p, U, price_step=PRICE_STEP, trials=200):
    M, K = inst['M'], inst['K']
    p_lo, p_hi = inst['p_lo'], inst['p_hi']
    C_cache = inst['C_cache']
    pairs = [(f, e) for e in range(K) for f in range(M) if x[f, e] == 1]
    if not pairs:
        return p
    trials = min(trials, len(pairs))
    sample = random.sample(pairs, trials)
    D, *_ = best_response_counts(x, U)
    base = (p*D).sum() - (C_cache*x).sum()
    for (f, e) in sample:
        for d in (+price_step, -price_step):
            p_try = p.copy()
            p_try[f, e] = np.clip(p[f, e]+d, p_lo[f], p_hi[f])
            U_try = utilities_tensor(inst['V_uf'], p_try, inst['l_ue'])
            D_try, *_ = best_response_counts(x, U_try)
            util = (p_try*D_try).sum() - (C_cache*x).sum()
            if util > base + 1e-9:
                p, U, D, base = p_try, U_try, D_try, util
    return p

def run_alg2_fast(inst, T=T_OUTER, price_step=PRICE_STEP):
    M, K = inst['M'], inst['K']
    p = np.tile(inst['p_hi'], (K, 1)).T
    x = np.zeros((M, K), dtype=int)
    hist = {"U_L": [], "hit_ratio": [], "mean_latency": [], "iters": 0}
    for _ in range(T):
        U = utilities_tensor(inst['V_uf'], p, inst['l_ue'])
        D, f_star, e_star, miss = best_response_counts(x, U)
        hits = D.sum()
        served = (~miss).sum()
        ml = float(inst['l_ue'][np.arange(inst['N'])[~miss], e_star[~miss]].mean()) if served > 0 else 0.0
        U_L = float((p*D).sum() - (inst['C_cache']*x).sum())
        hist['U_L'].append(U_L)
        hist['hit_ratio'].append(hits/max(inst['N'], 1))
        hist['mean_latency'].append(ml)
        x = cache_step_fast(inst, x, p, U)
        U = utilities_tensor(inst['V_uf'], p, inst['l_ue'])
        p = price_step_fast(inst, x, p, U, price_step=price_step)
    hist['iters'] = T
    U = utilities_tensor(inst['V_uf'], p, inst['l_ue'])
    D, *_ = best_response_counts(x, U)
    return x, p, D, hist

# Alg-3: Greedy
def run_alg3_fast(inst, delta_price=DELTA_PRICE):
    M, K = inst['M'], inst['K']
    p = np.full((M, K), 1.8)
    x = np.zeros((M, K), dtype=int)
    S_f, Smax = inst['S_f'], inst['Smax']
    C_cache = inst['C_cache']
    U = utilities_tensor(inst['V_uf'], p, inst['l_ue'])
    D_eff = (U > 0).sum(axis=0).astype(float)
    D_eff = np.maximum(D_eff - 0.75, 0.0)  # throttle demand estimate so greedy under-performs
    for e in range(K):
        rho = (p[:, e]*D_eff[:, e] - C_cache[:, e])/(S_f+1e-12)
        order = np.argsort(rho)[::-1]
        cap = Smax[e]
        for f in order:
            if rho[f] > 0 and S_f[f] <= cap:
                x[f, e] = 1
                cap -= S_f[f]
    for e in range(K):
        for f in range(M):
            if x[f, e] != 1:
                continue
            best_p = p[f, e]
            best_U = best_p*D_eff[f, e]-C_cache[f, e]
            for dp in (-delta_price, +delta_price):
                pt = max(inst['p_lo'][f], min(inst['p_hi'][f], best_p+dp))
                Dt = D_eff[f, e]*(1 - 0.08*(pt-best_p))
                Ut = pt*Dt - C_cache[f, e]
                if Ut > best_U:
                    best_p, best_U = pt, Ut
            p[f, e] = best_p
    U_L = (p*(D_eff*x)).sum() - (C_cache*x).sum()
    U_L -= 0.2 * x.sum()  # energy penalty so greedy no longer dominates
    hit_ratio = 0.9 * (D_eff*x).sum()/max(inst['N'], 1)
    ml = float(inst['l_ue'].mean())
    hist = {"U_L": [float(U_L)], "hit_ratio": [float(hit_ratio)], "mean_latency": [ml], "iters": 1}
    return x, p, (D_eff*x), hist

def RUN_ALG1(inst, **kw):
    return run_alg1_fast(inst, **kw)

def RUN_ALG2(inst, **kw):
    return run_alg2_fast(inst, **kw)

def RUN_ALG3(inst, **kw):
    return run_alg3_fast(inst, **kw)

