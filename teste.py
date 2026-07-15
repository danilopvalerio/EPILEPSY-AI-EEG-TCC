"""
Comparacao de pipeline completo para chb06 — v5
Correcoes:
  - OOM: aplica preproc apenas no trecho W_S necessario (ultimos 25min)
  - 437 feats: usa apenas os primeiros 17 canais (mapeamento NB3)
"""

import json, warnings, re, time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis as kurt
from scipy.signal import welch
from scipy.integrate import trapezoid

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
SIGNAL_DIR = Path('data/signals')
W_S        = 25 * 60          # janela experimental (25 min)
SEEDS      = [42, 43, 44, 45, 46]
N_FEATS    = 19
N_CH_USE   = 17               # usa so os primeiros 17 canais (igual ao NB3)

CH_GROUPS = {
    'temporal':  [0, 1, 2, 3],
    'frontal':   [4, 5, 6, 7, 8, 9, 10],
    'central':   [11, 12, 13, 14],
    'parietal':  [15, 16],
}

SEGMENTATIONS = [
    (30.0, 15.0, '30s/15s'),
    (16.0,  8.0, '16s/8s'),
    (10.0,  5.0, '10s/5s'),
]
PREPROC    = ['baseline', 'wsog', 'wsog_car']
FEAT_MODES = ['all_323', 'regional_76', 'mean_std_38']

# ── WSOG ──────────────────────────────────────────────────────────────────────
def wsog_denoise(signal_1d, wavelet='db4', level=4):
    coeffs    = pywt.wavedec(signal_1d, wavelet, level=level)
    sigma     = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_1d) + 1))
    out       = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft')
                                for c in coeffs[1:]]
    return pywt.waverec(out, wavelet)[:len(signal_1d)]

def apply_preproc(signal, sfreq, mode):
    """
    signal: (n_ch, n_samples) — ja filtrado pelo NB1
    Aplica preproc APENAS no trecho necessario (evita OOM em sinais longos).
    Trecho = ultimos W_S segundos + margem de 30s para segmentacao.
    """
    n_keep = int((W_S + 60) * sfreq)   # W_S + 60s de margem
    if signal.shape[1] > n_keep:
        sig = signal[:, -n_keep:].astype(np.float32).copy()
    else:
        sig = signal.astype(np.float32).copy()

    # Usa so os primeiros N_CH_USE canais
    sig = sig[:N_CH_USE]

    if mode in ('wsog', 'wsog_car'):
        sig64 = sig.astype(np.float64)
        for ch in range(sig64.shape[0]):
            sig64[ch] = wsog_denoise(sig64[ch])
        sig = sig64.astype(np.float32)

    if mode == 'wsog_car':
        sig -= sig.mean(axis=0, keepdims=True)

    return sig

# ── Features ──────────────────────────────────────────────────────────────────
def extract_features_window(window, sfreq):
    feats = []
    for ch in window:
        std    = np.std(ch)
        diff1  = np.diff(ch)
        std_d1 = np.std(diff1)
        mob    = std_d1 / (std + 1e-10)
        feats += [std, np.var(ch), float(np.sqrt(np.mean(ch**2))),
                  float(np.sum(np.abs(diff1))), float(mob),
                  float(skew(ch)), float(kurt(ch))]
        f, psd = welch(ch, fs=sfreq, nperseg=min(len(ch), int(sfreq * 4)))
        def bp(lo, hi):
            m = (f >= lo) & (f <= hi)
            return float(trapezoid(psd[m], f[m])) if m.any() else 0.0
        d,t,a,b,g = bp(.5,4),bp(4,8),bp(8,13),bp(13,30),bp(30,40)
        tot = d+t+a+b+g+1e-10
        pn  = psd/(psd.sum()+1e-10); pn = pn[pn>0]
        feats += [d,t,a,b,g, float(-np.sum(pn*np.log2(pn))), float(b/tot)]
        c = pywt.wavedec(ch, 'db4', level=4)
        feats += [float(np.sum(c[4]**2)), float(np.sum(c[3]**2)),
                  float(np.sum(c[2]**2)), float(np.sum(c[1]**2))]
        diff2  = np.diff(diff1)
        mob_d1 = np.std(diff2) / (std_d1 + 1e-10)
        feats.append(float(mob_d1 / (mob + 1e-10)))
    return np.array(feats, dtype=np.float32)

def segment_and_extract(signal, sfreq, win_s, step_s):
    win_n  = int(win_s  * sfreq)
    step_n = int(step_s * sfreq)
    feats, centers = [], []
    for start in range(0, signal.shape[1] - win_n + 1, step_n):
        feats.append(extract_features_window(signal[:, start:start+win_n], sfreq))
        centers.append((start + win_n/2) / sfreq)
    if not feats:
        return np.empty((0, signal.shape[0]*N_FEATS), dtype=np.float32), np.array([])
    return np.array(feats, dtype=np.float32), np.array(centers)

def aggregate_features(X, n_ch, mode):
    X3 = X.reshape(len(X), n_ch, N_FEATS)
    if mode == 'all_323':
        return X
    elif mode == 'mean_std_38':
        return np.hstack([X3.mean(axis=1), X3.std(axis=1)])
    elif mode == 'regional_76':
        parts = []
        for ch_idx in CH_GROUPS.values():
            valid = [i for i in ch_idx if i < n_ch]
            if valid:
                parts.append(X3[:, valid, :].mean(axis=1))
        return np.hstack(parts)
    return X

# ── LOSO ──────────────────────────────────────────────────────────────────────
def run_loso(Xp, Xi, cp, ci, contexts):
    results = []
    for fold_n, held_out in enumerate(contexts, 1):
        train_ctx = [c for c in contexts if c != held_out]
        tr_p = Xp[np.isin(cp, train_ctx)]
        tr_i = Xi[np.isin(ci, train_ctx)]
        te_p = Xp[cp == held_out]
        te_i = Xi[ci == held_out]
        if len(tr_p)==0 or len(tr_i)==0 or len(te_p)==0 or len(te_i)==0:
            print(f'      fold {fold_n}/ctx{held_out}: sem dados — pulado')
            continue

        X_test = np.vstack([te_p, te_i])
        y_test = np.concatenate([np.ones(len(te_p)), np.zeros(len(te_i))])
        n_min  = min(len(tr_p), len(tr_i))
        seed_aucs = []

        for seed in SEEDS:
            rng   = np.random.RandomState(seed)
            idx_p = rng.choice(len(tr_p), n_min, replace=False)
            idx_i = rng.choice(len(tr_i), n_min, replace=False)
            X_tr  = np.nan_to_num(np.vstack([tr_p[idx_p], tr_i[idx_i]]), nan=0.)
            y_tr  = np.concatenate([np.ones(n_min), np.zeros(n_min)])
            X_te  = np.nan_to_num(X_test, nan=0.)
            sc    = StandardScaler()
            Xtr   = np.clip(sc.fit_transform(X_tr), -50, 50)
            Xte   = np.clip(sc.transform(X_te),     -50, 50)
            try:
                clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                             class_weight='balanced',
                                             random_state=42, n_jobs=-1)
                clf.fit(Xtr, y_tr)
                ys = clf.predict_proba(Xte)[:, 1]
                if len(set(y_test)) > 1:
                    seed_aucs.append(roc_auc_score(y_test, ys))
            except: pass

        auc_fold = float(np.mean(seed_aucs)) if seed_aucs else float('nan')
        flag = ' <<< RUIM' if auc_fold < 0.45 else (' *** BOM' if auc_fold > 0.75 else '')
        print(f'      fold {fold_n}/ctx{int(held_out)}: '
              f'AUC={auc_fold:.3f}  '
              f'treino={len(tr_p)}pre+{len(tr_i)}inter  '
              f'teste={len(te_p)}pre+{len(te_i)}inter{flag}')

        if seed_aucs:
            results.append({'fold': int(held_out), 'auc': auc_fold,
                            'n_feats': X_test.shape[1],
                            'n_tr_p': len(tr_p), 'n_tr_i': len(tr_i)})
    return results

# ── Carrega sinais ────────────────────────────────────────────────────────────
print('='*65)
print('COMPARACAO DE PIPELINE — chb06')
print('='*65)
print(f'Carregando sinais de {SIGNAL_DIR}...')

sig_files = sorted(SIGNAL_DIR.glob('CHBMIT__chb06__ctx*.npz'))
if not sig_files:
    print(f'ERRO: nenhum arquivo em {SIGNAL_DIR}/CHBMIT__chb06__ctx*.npz')
    exit(1)

RAW_DATA = {}
for sf in sig_files:
    m = re.search(r'ctx(\d+)', sf.stem)
    if not m: continue
    ctx_id = int(m.group(1))
    d = np.load(sf, allow_pickle=True)
    RAW_DATA[ctx_id] = {
        'pre':   d['pre'].astype(np.float32),
        'inter': d['inter'].astype(np.float32),
        'sfreq': float(d['sfreq']),
    }

n_ch_raw = list(RAW_DATA.values())[0]['pre'].shape[0]
print(f'  {len(RAW_DATA)} contextos  |  {n_ch_raw} canais no .npz  '
      f'(usando primeiros {N_CH_USE})')
print(f'  W experimental = {W_S//60} min')
print(f'  Trecho carregado por contexto: ultimos {(W_S+60)//60:.0f} min (W + 1min margem)')
print()
print(f'  Preprocessamentos : {PREPROC}')
print(f'  Segmentacoes      : {[s[2] for s in SEGMENTATIONS]}')
print(f'  Estrategias feat  : {FEAT_MODES}')
total = len(PREPROC) * len(SEGMENTATIONS) * len(FEAT_MODES)
print(f'  Total combinacoes : {total}')
print()

# ── Loop principal ────────────────────────────────────────────────────────────
all_rows = []
combo_n  = 0
t_global = time.time()

for preproc, (win_s, step_s, seg_label), feat_mode in product(
        PREPROC, SEGMENTATIONS, FEAT_MODES):

    combo_n += 1
    t0 = time.time()
    n_feats_expected = N_CH_USE * N_FEATS  # 323 para all, menos para outros

    print(f'[{combo_n:02d}/{total}] preproc={preproc:<12} seg={seg_label:<8} feat={feat_mode}')

    Xp_list, Xi_list, cp_list, ci_list = [], [], [], []

    for ctx_id, data in sorted(RAW_DATA.items()):
        sfreq = data['sfreq']

        # Aplica preproc so no trecho necessario
        pre_sig   = apply_preproc(data['pre'],   sfreq, preproc)
        inter_sig = apply_preproc(data['inter'], sfreq, preproc)

        n_ch = pre_sig.shape[0]  # deve ser N_CH_USE

        Xp, tp = segment_and_extract(pre_sig,   sfreq, win_s, step_s)
        Xi, ti = segment_and_extract(inter_sig, sfreq, win_s, step_s)

        if len(Xp) == 0 or len(Xi) == 0:
            print(f'    ctx{ctx_id}: sem janelas — pulado')
            continue

        # Referencial negativo
        pre_dur_s   = pre_sig.shape[1]   / sfreq
        inter_dur_s = inter_sig.shape[1] / sfreq
        tp_neg = tp - pre_dur_s
        ti_neg = ti - inter_dur_s

        Xp_agg = aggregate_features(Xp, n_ch, feat_mode)
        Xi_agg = aggregate_features(Xi, n_ch, feat_mode)

        mask_p = tp_neg >= -W_S
        mask_i = ti_neg >= -W_S
        n_p = mask_p.sum(); n_i = mask_i.sum()

        print(f'    ctx{ctx_id}: {n_p} jan pre / {n_i} jan inter  '
              f'({Xp_agg.shape[1]} feats, {n_ch} canais)')

        Xp_list.append(Xp_agg[mask_p])
        Xi_list.append(Xi_agg[mask_i])
        cp_list.append(np.full(n_p, ctx_id, dtype=np.int32))
        ci_list.append(np.full(n_i, ctx_id, dtype=np.int32))

    if not Xp_list:
        print('    -> sem dados\n'); continue

    Xp_all = np.vstack(Xp_list); cp_all = np.concatenate(cp_list)
    Xi_all = np.vstack(Xi_list); ci_all = np.concatenate(ci_list)

    # Z-score por contexto
    for arr, ids in [(Xp_all, cp_all), (Xi_all, ci_all)]:
        for cid in np.unique(ids):
            mask = ids == cid
            if mask.sum() < 2: continue
            mu = arr[mask].mean(axis=0, keepdims=True)
            sd = arr[mask].std(axis=0,  keepdims=True) + 1e-10
            arr[mask] = (arr[mask] - mu) / sd

    contexts = sorted(set(cp_all) & set(ci_all))
    print(f'    -> LOSO ({len(contexts)} contextos, '
          f'{len(Xp_all)} jan pre / {len(Xi_all)} jan inter):')

    results = run_loso(Xp_all, Xi_all, cp_all, ci_all, contexts)

    if not results:
        print('    -> nenhum fold completado\n'); continue

    aucs    = [r['auc'] for r in results]
    mean_a  = float(np.mean(aucs))
    std_a   = float(np.std(aucs))
    elapsed = time.time() - t0

    print(f'    -> RESULTADO: AUC={mean_a:.3f} ± {std_a:.3f}  '
          f'(min={min(aucs):.3f} max={max(aucs):.3f})  [{elapsed:.0f}s]')
    print()

    for r in results:
        all_rows.append({'preproc': preproc, 'seg': seg_label,
                         'feat': feat_mode, **r})

# ── Resultados ────────────────────────────────────────────────────────────────
elapsed_total = time.time() - t_global
print('='*65)
print(f'CONCLUIDO em {elapsed_total/60:.1f} min')
print('='*65)

df      = pd.DataFrame(all_rows)
summary = (df.groupby(['preproc','seg','feat'])
             .agg(auc_media=('auc','mean'), auc_std=('auc','std'),
                  auc_min=('auc','min'),    auc_max=('auc','max'),
                  n_feats=('n_feats','first'), n_folds=('fold','count'))
             .reset_index()
             .sort_values('auc_media', ascending=False))

print('\nRANKING COMPLETO:')
print(f'  {"#":<4} {"preproc":<14} {"seg":<10} {"feat":<16} '
      f'{"n_f":<6} {"media":<8} {"std":<7} {"min":<7} {"max"}')
print('  ' + '-'*78)
for rank, (_, row) in enumerate(summary.iterrows(), 1):
    print(f'  {rank:<4} {row["preproc"]:<14} {row["seg"]:<10} {row["feat"]:<16} '
          f'{int(row["n_feats"]):<6} '
          f'{row["auc_media"]:.3f}    {row["auc_std"]:.3f}   '
          f'{row["auc_min"]:.3f}   {row["auc_max"]:.3f}')

print('\nMELHOR POR SEGMENTACAO:')
for seg in [s[2] for s in SEGMENTATIONS]:
    sub = summary[summary['seg']==seg]
    if sub.empty: continue
    b = sub.iloc[0]
    print(f'  {seg:<10} -> {b["preproc"]:<12} | {b["feat"]:<16} '
          f'AUC={b["auc_media"]:.3f} ± {b["auc_std"]:.3f}')

print('\nMELHOR POR PREPROCESSAMENTO:')
for pp in PREPROC:
    sub = summary[summary['preproc']==pp]
    if sub.empty: continue
    b = sub.iloc[0]
    print(f'  {pp:<12} -> {b["seg"]:<10} | {b["feat"]:<16} '
          f'AUC={b["auc_media"]:.3f} ± {b["auc_std"]:.3f}')

print('\nMELHOR POR ESTRATEGIA DE FEATURES:')
for fm in FEAT_MODES:
    sub = summary[summary['feat']==fm]
    if sub.empty: continue
    b = sub.iloc[0]
    print(f'  {fm:<16} -> {b["preproc"]:<12} | {b["seg"]:<10} '
          f'AUC={b["auc_media"]:.3f} ± {b["auc_std"]:.3f}')

out = Path('data/results/compare_pipeline_chb06.csv')
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
summary.to_csv(str(out).replace('.csv','_summary.csv'), index=False)
print(f'\nCSVs salvos em {out.parent}/')