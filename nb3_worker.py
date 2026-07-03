'''Módulo de trabalho do NB3 v5 — otimizado para processar R5 uma vez e
derivar R3/R2/R1/R0 por fatiamento de colunas, em vez de recalcular features
para cada nível separadamente. Reduz o número de tarefas pesadas de 5x para 1x
(para CHBMIT/Siena/Mendeley). SeizeIT2 mantém fluxo próprio (só R0, canais
diferentes). Importado pelo notebook e pelos processos filhos do
ProcessPoolExecutor — precisa ser um arquivo .py real para funcionar no Windows.
'''
import os, re, glob, gc
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import pywt

np.trapz = getattr(np, 'trapz', getattr(np, 'trapezoid', None))

# ── Diretórios ──
ROOT_DIR   = 'data'
SIGNAL_DIR = os.path.join(ROOT_DIR, 'signals')
L1_DIR     = os.path.join(ROOT_DIR, 'level1_signals')
FEAT_DIR   = os.path.join(ROOT_DIR, 'features')
LOG_DIR    = os.path.join(ROOT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# ── Pacientes selecionados ──
PATIENTS = {
    'CHBMIT'  : ['chb01','chb03','chb04','chb05','chb06','chb07',
                 'chb08','chb10','chb11','chb12','chb13','chb14'],
    'Siena'   : ['PN00','PN01','PN03','PN05','PN06','PN09',
                 'PN10','PN12','PN13','PN14','PN16','PN17'],
    'Mendeley': ['p10','p11','p12','p13','p14','p15'],
    'SeizeIT2': ['sub-001','sub-002','sub-004','sub-005',
                 'sub-007','sub-008','sub-011','sub-012',
                 'sub-035','sub-039','sub-047','sub-073'],
}

# ── Janelas pré-ictais ──
PREICTAL_WINDOWS_MIN = [10, 15, 30, 45]

# ── Níveis de canal ──
LEVELS = ['R5', 'R3', 'R2', 'R1', 'R0']
LEVEL_DS = {
    'R5': ['CHBMIT','Siena','Mendeley'],
    'R3': ['CHBMIT','Siena','Mendeley'],
    'R2': ['CHBMIT','Siena','Mendeley'],
    'R1': ['CHBMIT','Siena','Mendeley'],
    'R0': ['CHBMIT','Siena','Mendeley','SeizeIT2'],
}

# ── Composição de canais por nível e dataset ──
FRONTAL_CH  = {'CHBMIT':['FP1-F7','F7-T7','FP1-F3','FP2-F4','FP2-F8','F8-T8','FZ-CZ'],
               'Siena':['Fp1','F3','F7','Fz','Fp2','F4','F8'],
               'Mendeley':['Fp1','Fp2','F3','F4','F7','F8','Fz']}
TEMPORAL_CH = {'CHBMIT':['T7-P7','T7-FT9','FT10-T8','T8-P8-0'],
               'Siena':['T3','T4','T5','T6'],
               'Mendeley':['T3','T4','T5','T6']}
CENTRAL_CH  = {'CHBMIT':['F3-C3','F4-C4'],'Siena':['C3','C4'],'Mendeley':['C3','C4']}
PARIETAL_CH = {'CHBMIT':['P3-O1','P4-O2'],'Siena':['P3','P4'],'Mendeley':['P3','P4']}
OCCIPITAL_CH= {'CHBMIT':['P7-O1','P8-O2'],'Siena':['O1','O2'],'Mendeley':['O1','O2']}
R0_CORE_CH  = {'CHBMIT':['T7-FT9','FT10-T8'],'Siena':['T3','T4'],'Mendeley':['T3','T4']}

REGION_DS = ['CHBMIT','Siena','Mendeley']
LEVEL_CHANNELS = {'R5':{},'R3':{},'R2':{},'R1':{},'R0':{}}
for _ds in REGION_DS:
    LEVEL_CHANNELS['R5'][_ds] = FRONTAL_CH[_ds]+TEMPORAL_CH[_ds]+CENTRAL_CH[_ds]+PARIETAL_CH[_ds]+OCCIPITAL_CH[_ds]
    LEVEL_CHANNELS['R3'][_ds] = FRONTAL_CH[_ds]+TEMPORAL_CH[_ds]+CENTRAL_CH[_ds]
    LEVEL_CHANNELS['R2'][_ds] = FRONTAL_CH[_ds]+TEMPORAL_CH[_ds]
    LEVEL_CHANNELS['R1'][_ds] = TEMPORAL_CH[_ds]
    LEVEL_CHANNELS['R0'][_ds] = R0_CORE_CH[_ds]

CH_RENAME = {'SeizeIT2': {'bteleftsd':'t3','bterightsd':'t4','crosstopsd':'t3'}}
SEIZEIT2_R0_TARGETS = ['t3','t4']

# Validações
_EXPECTED = {'R5':17,'R3':13,'R2':11,'R1':4,'R0':2}
_ORDER    = ['R0','R1','R2','R3','R5']
for _lv, _exp in _EXPECTED.items():
    for _ds in REGION_DS:
        _c = LEVEL_CHANNELS[_lv][_ds]
        assert len(_c)==_exp, f'{_lv}/{_ds}: {len(_c)} != {_exp}'
        assert len(set(_c))==len(_c), f'{_lv}/{_ds}: duplicado'
for _ds in REGION_DS:
    for _i in range(len(_ORDER)-1):
        assert set(LEVEL_CHANNELS[_ORDER[_i]][_ds]) <= set(LEVEL_CHANNELS[_ORDER[_i+1]][_ds])
del _lv,_exp,_ds,_c,_i

# Cria diretórios de features
for lv in LEVELS:
    os.makedirs(os.path.join(FEAT_DIR, lv), exist_ok=True)

# ── Slices de colunas para derivar níveis a partir do R5 ──────────────────
# Calculados analiticamente: cada canal ocupa N_FEAT=19 colunas consecutivas
# no vetor de features do R5, na ordem de LEVEL_CHANNELS['R5'][ds].
# Verificado numericamente: todos os slices são contíguos nos 3 datasets.
#
# Offsets (iguais para CHBMIT, Siena e Mendeley por construção):
#   R5: [0:323]   (17 canais × 19)
#   R3: [0:247]   (13 canais × 19 — frontal+temporal+central são os primeiros 13)
#   R2: [0:209]   (11 canais × 19 — frontal+temporal são os primeiros 11)
#   R1: [133:209] (4 canais × 19  — temporal começa no canal 7, índice 7*19=133)
#   R0: varia por dataset:
#       CHBMIT:          [152:190]  (T7-FT9=canal8, FT10-T8=canal9 → 8*19=152)
#       Siena/Mendeley:  [133:171]  (T3=canal7, T4=canal8 → 7*19=133)
#
# Estes slices são pré-computados uma vez aqui e usados em process_file_r5.
N_FEAT = 19

def _compute_r5_slices(ds):
    '''Retorna dict {nível: slice} para fatiar o vetor R5 de um dataset.'''
    R5 = LEVEL_CHANNELS['R5'][ds]
    slices = {}
    for lv in ['R5','R3','R2','R1','R0']:
        ch_list = LEVEL_CHANNELS[lv][ds]
        idxs = [R5.index(ch) for ch in ch_list]
        col_start = idxs[0]  * N_FEAT
        col_end   = (idxs[-1]+1) * N_FEAT
        slices[lv] = slice(col_start, col_end)
    return slices

R5_SLICES = {ds: _compute_r5_slices(ds) for ds in REGION_DS}

# ── Parâmetros de janelamento ──
WIN_SEC    = 30
STEP_SEC   = 15
MIN_PURITY = 0.80
LBL        = dict(interictal=0, preictal=1, ictal=2, postictal=3, unknown=-1)

# ── Features ──
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,40)}
FEAT_NAMES = ['std','var','rms','line_len','mobility','skewness','kurtosis',
              'delta','theta','alpha','beta','gamma','sp_entropy','beta_rel',
              'dwt_d1','dwt_d2','dwt_d3','dwt_d4','complexity']


def normalize_ch(name, rename_map=None):
    s = str(name).lower()
    for sub in ['eeg','-ref','-le','-a1','-a2','ref']: s = s.replace(sub,' ')
    s = s.strip().replace(' ','')
    if '-' in s: s = s.split('-')[0]
    s = re.sub(r'[^a-z0-9]','',s)
    return rename_map.get(s,s) if rename_map else s

def normalize_ch_full(name):
    s = str(name).lower()
    for sub in ['eeg','-ref','-le','-a1','-a2','ref']: s = s.replace(sub,' ')
    s = s.strip().replace(' ','')
    return re.sub(r'[^a-z0-9-]','',s)

def channel_feats(sig, sfreq=256.0):
    if len(sig) < 2: return np.zeros(N_FEAT, dtype=np.float32)
    d1 = np.diff(sig); d2 = np.diff(d1) if len(d1)>1 else np.array([0.0])
    act = float(np.var(sig)); mob = float(np.sqrt(np.var(d1)/(act+1e-10)))
    mob2 = float(np.sqrt(np.var(d2)/(np.var(d1)+1e-10)))
    temporal = [float(np.std(sig)), act, float(np.sqrt(np.mean(sig**2))),
                float(np.sum(np.abs(d1))), mob, float(skew(sig)), float(kurtosis(sig))]
    nperseg = min(int(sfreq), max(int(sfreq//2), len(sig)//2))
    nperseg = max(min(nperseg, len(sig)), 2)
    f, psd = welch(sig, fs=sfreq, nperseg=nperseg); total = psd.sum()+1e-10
    bp = []
    for lo,hi in BANDS.values():
        idx = (f>=lo)&(f<=hi)
        bp.append(float(np.trapz(psd[idx],f[idx])) if idx.sum()>1 else 0.0)
    pn = psd/total; pn = pn[pn>0]; sp_ent = float(-np.sum(pn*np.log(pn)))
    b_idx = (f>=13)&(f<=30)
    beta_rel = float(np.trapz(psd[b_idx],f[b_idx])/total) if b_idx.sum()>1 else 0.0
    spectral = bp + [sp_ent, beta_rel]
    ml = min(4, pywt.dwt_max_level(len(sig),'db4')) if len(sig)>1 else 0
    if ml>=1:
        coeffs = pywt.wavedec(sig if sig.dtype==np.float64 else sig.astype(np.float64),'db4',level=ml)
        dwt = [float(np.sum(c**2)) for c in coeffs[1:5]]
        while len(dwt)<4: dwt.append(0.0)
    else:
        dwt = [0.0]*4
    complexity = mob2/(mob+1e-10)
    return np.array(temporal+spectral+dwt+[complexity], dtype=np.float32)

def build_fvec_r5(win_data, ch_names_raw, dataset):
    '''Calcula features apenas para R5 (todos os canais).
    Os outros níveis são derivados por fatiamento — não chamam esta função.'''
    ch_map = {normalize_ch_full(c):i for i,c in enumerate(ch_names_raw)}
    parts = []
    for tgt in LEVEL_CHANNELS['R5'][dataset]:
        i = ch_map.get(normalize_ch_full(tgt))
        parts.append(channel_feats(win_data[i]) if i is not None
                     else np.zeros(N_FEAT, dtype=np.float32))
    return np.concatenate(parts)   # shape (323,)

def build_fvec_seizeit2(win_data, ch_names_raw):
    '''SeizeIT2: só R0, 2 canais behind-the-ear.'''
    rename = CH_RENAME.get('SeizeIT2',{})
    ch_map = {normalize_ch(c,rename):i for i,c in enumerate(ch_names_raw)}
    parts = []
    for tgt in SEIZEIT2_R0_TARGETS:
        i = ch_map.get(tgt)
        parts.append(channel_feats(win_data[i]) if i is not None
                     else np.zeros(N_FEAT, dtype=np.float32))
    return np.concatenate(parts)   # shape (38,)


def _save_level(out_path, Xp, Xi, tp, ti, ch_list, dataset, patient, fkey,
                level, window_min, file_order, n_seizures,
                col_start=0, col_end=None):
    '''Salva um arquivo de features para um nível específico.

    Metadados salvos no .npz:
      X_pre / X_inter  — matrizes de features (n_janelas × n_features)
      t_pre / t_inter  — timestamps em segundos desde o início do arquivo
      channels         — nomes dos canais neste nível (ex: ['T7-FT9', 'FT10-T8'])
      feat_names       — nomes das 19 features por canal
      dataset, patient, fkey, level — identificação
      window_min       — janela pré-ictal em minutos (10/15/30/45)
      file_order       — índice cronológico do arquivo dentro do paciente
      n_seizures       — número de crises neste arquivo
      n_pre, n_inter, n_total — contagem de janelas
      win_sec, step_sec — tamanho e passo da janela em segundos
      n_feat_per_ch    — número de features por canal (19)
      sfreq            — frequência de amostragem em Hz
      col_start/col_end — offsets de coluna no vetor R5 (para auditoria/rastreabilidade)
    '''
    if col_end is None:
        col_end = Xp.shape[1] if len(Xp) else Xi.shape[1] if len(Xi) else 0
    np.savez_compressed(
        out_path,
        X_pre=Xp,   t_pre=tp,
        X_inter=Xi, t_inter=ti,
        channels=np.array(ch_list),
        feat_names=np.array(FEAT_NAMES),
        dataset=np.str_(dataset), patient=np.str_(patient),
        fkey=np.str_(fkey), level=np.str_(level),
        window_min=np.int64(window_min),
        file_order=np.int64(file_order),
        n_seizures=np.int64(n_seizures),
        n_pre=np.int64(len(Xp)), n_inter=np.int64(len(Xi)),
        n_total=np.int64(len(Xp)+len(Xi)),
        win_sec=np.float32(WIN_SEC), step_sec=np.float32(STEP_SEC),
        n_feat_per_ch=np.int32(N_FEAT),
        col_start=np.int32(col_start),
        col_end=np.int32(col_end),
    )


def process_file_r5(dataset, patient, fkey, window_min, file_order):
    '''Processa UMA gravação de CHBMIT/Siena/Mendeley para UMA janela pré-ictal.
    Calcula features R5 uma única vez e deriva R3/R2/R1/R0 por fatiamento.
    Salva 5 arquivos .npz (um por nível) em uma única passagem pelo sinal.

    Retorna lista de dicts de metadados (um por nível salvo com sucesso).
    '''
    results = []

    # Verifica quais níveis já estão prontos para essa gravação+janela
    out_paths = {
        lv: os.path.join(FEAT_DIR, lv, f'{dataset}__{patient}__{fkey}__w{window_min}.npz')
        for lv in LEVELS
    }
    pending_levels = [lv for lv in LEVELS if not os.path.exists(out_paths[lv])]

    if not pending_levels:
        # Todos os 5 níveis já existem — retorna metadados do cache
        for lv, out_path in out_paths.items():
            try:
                z = np.load(out_path, allow_pickle=True)
                results.append(dict(
                    dataset=dataset, patient=patient, fkey=fkey, level=lv,
                    window_min=window_min, file_order=file_order,
                    n_pre=int(z['n_pre']), n_inter=int(z['n_inter']),
                    n_total=int(z['n_total']), n_seizures=int(z['n_seizures']),
                    status='cached', path=out_path))
                z.close()
            except Exception:
                pass
        return results

    # ── Carrega rótulos (usa w{window_min}) ──
    l1_path = os.path.join(L1_DIR, f'{dataset}__{patient}__{fkey}_L1_w{window_min}.npz')
    if not os.path.exists(l1_path):
        for lv in pending_levels:
            results.append(dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                                window_min=window_min, file_order=file_order,
                                status='sem_l1', path=None))
        return results

    try:
        z1     = np.load(l1_path, allow_pickle=True)
        labels = z1['labels']
        sfreq  = float(z1['sfreq'])
        chs    = list(z1['ch_names'])
        n      = int(z1['n_samples'])
        z1.close(); del z1
    except Exception as e:
        for lv in pending_levels:
            results.append(dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                                window_min=window_min, file_order=file_order,
                                status=f'erro_l1:{e}', path=None))
        return results

    # ── Carrega sinal bruto com mmap (leitura sob demanda, não carrega tudo na RAM) ──
    sig_path = os.path.join(SIGNAL_DIR, f'{dataset}__{patient}__{fkey}_signal.npz')
    if not os.path.exists(sig_path):
        del labels
        for lv in pending_levels:
            results.append(dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                                window_min=window_min, file_order=file_order,
                                status='sem_signal', path=None))
        return results

    try:
        zs   = np.load(sig_path, allow_pickle=True, mmap_mode='r')
        data = zs['data']   # array mapeado — páginas carregadas sob demanda
    except Exception as e:
        del labels
        for lv in pending_levels:
            results.append(dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                                window_min=window_min, file_order=file_order,
                                status=f'erro_signal:{e}', path=None))
        return results

    # ── Conta crises ──
    ictal_arr = (labels == LBL['ictal']).astype(int)
    n_seizures = int(np.sum(np.diff(np.concatenate([[0], ictal_arr])) == 1))

    # ── Extração de janelas ── (percorre o sinal UMA VEZ, calcula R5, salva tudo)
    win_n  = int(WIN_SEC * sfreq)
    step_n = max(1, int(STEP_SEC * sfreq))
    slices = R5_SLICES[dataset]   # dict {lv: slice} pré-calculado

    # Acumuladores por nível
    X_pre_r5  = []; X_inter_r5  = []
    t_pre     = []; t_inter     = []

    for start in range(0, n - win_n + 1, step_n):
        wl    = labels[start:start+win_n]
        valid = wl[wl >= 0]
        if len(valid) == 0:
            continue
        vals, counts = np.unique(valid, return_counts=True)
        dom = vals[np.argmax(counts)]
        if dom not in (LBL['interictal'], LBL['preictal']):
            continue
        if counts.max() / win_n < MIN_PURITY:
            continue

        # Calcula features R5 (17 canais × 19 = 323 features)
        fv_r5 = build_fvec_r5(data[:, start:start+win_n], chs, dataset)
        t_sec = start / sfreq

        if dom == LBL['preictal']:
            X_pre_r5.append(fv_r5); t_pre.append(t_sec)
        else:
            X_inter_r5.append(fv_r5); t_inter.append(t_sec)

    del data; gc.collect()

    # Converte para arrays numpy
    Xp5 = np.array(X_pre_r5,   dtype=np.float32) if X_pre_r5   else np.empty((0,323),dtype=np.float32)
    Xi5 = np.array(X_inter_r5, dtype=np.float32) if X_inter_r5 else np.empty((0,323),dtype=np.float32)
    tp  = np.array(t_pre,   dtype=np.float64)
    ti  = np.array(t_inter, dtype=np.float64)

    del X_pre_r5, X_inter_r5, labels; gc.collect()

    n_total = len(Xp5) + len(Xi5)

    # ── Salva cada nível pendente por fatiamento ──
    for lv in pending_levels:
        sl = slices[lv]
        Xp_lv = Xp5[:, sl] if len(Xp5) else np.empty((0, sl.stop-sl.start), dtype=np.float32)
        Xi_lv = Xi5[:, sl] if len(Xi5) else np.empty((0, sl.stop-sl.start), dtype=np.float32)

        if n_total == 0:
            results.append(dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                                window_min=window_min, file_order=file_order,
                                n_pre=0, n_inter=0, n_total=0, n_seizures=n_seizures,
                                status='vazio', path=None))
            continue

        ch_list = LEVEL_CHANNELS[lv][dataset]
        _save_level(out_paths[lv], Xp_lv, Xi_lv, tp, ti,
                    ch_list, dataset, patient, fkey, lv,
                    window_min, file_order, n_seizures,
                    col_start=sl.start, col_end=sl.stop)

        results.append(dict(
            dataset=dataset, patient=patient, fkey=fkey, level=lv,
            window_min=window_min, file_order=file_order,
            n_pre=len(Xp_lv), n_inter=len(Xi_lv),
            n_total=len(Xp_lv)+len(Xi_lv),
            n_seizures=n_seizures, status='ok', path=out_paths[lv]))

    del Xp5, Xi5, tp, ti; gc.collect()
    return results


def process_file_seizeit2(patient, fkey, window_min, file_order):
    '''SeizeIT2: processa só R0, canais behind-the-ear (t3, t4).
    Fluxo separado porque os canais são completamente diferentes de R5.
    '''
    dataset = 'SeizeIT2'
    lv      = 'R0'
    out_path = os.path.join(FEAT_DIR, lv, f'{dataset}__{patient}__{fkey}__w{window_min}.npz')

    if os.path.exists(out_path):
        try:
            z = np.load(out_path, allow_pickle=True)
            r = dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                     window_min=window_min, file_order=file_order,
                     n_pre=int(z['n_pre']), n_inter=int(z['n_inter']),
                     n_total=int(z['n_total']), n_seizures=int(z['n_seizures']),
                     status='cached', path=out_path)
            z.close()
            return r
        except Exception:
            pass

    l1_path = os.path.join(L1_DIR, f'{dataset}__{patient}__{fkey}_L1_w{window_min}.npz')
    if not os.path.exists(l1_path):
        return dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                    window_min=window_min, file_order=file_order,
                    status='sem_l1', path=None)
    try:
        z1 = np.load(l1_path, allow_pickle=True)
        labels = z1['labels']; sfreq = float(z1['sfreq'])
        chs = list(z1['ch_names']); n = int(z1['n_samples'])
        z1.close(); del z1
    except Exception as e:
        return dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                    window_min=window_min, file_order=file_order,
                    status=f'erro_l1:{e}', path=None)

    sig_path = os.path.join(SIGNAL_DIR, f'{dataset}__{patient}__{fkey}_signal.npz')
    if not os.path.exists(sig_path):
        del labels
        return dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                    window_min=window_min, file_order=file_order,
                    status='sem_signal', path=None)
    try:
        zs = np.load(sig_path, allow_pickle=True, mmap_mode='r')
        data = zs['data']
    except Exception as e:
        del labels
        return dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                    window_min=window_min, file_order=file_order,
                    status=f'erro_signal:{e}', path=None)

    ictal_arr = (labels == LBL['ictal']).astype(int)
    n_seizures = int(np.sum(np.diff(np.concatenate([[0], ictal_arr])) == 1))
    win_n  = int(WIN_SEC * sfreq)
    step_n = max(1, int(STEP_SEC * sfreq))
    X_pre = []; X_inter = []; t_pre = []; t_inter = []

    for start in range(0, n - win_n + 1, step_n):
        wl    = labels[start:start+win_n]
        valid = wl[wl >= 0]
        if len(valid) == 0: continue
        vals, counts = np.unique(valid, return_counts=True)
        dom = vals[np.argmax(counts)]
        if dom not in (LBL['interictal'], LBL['preictal']): continue
        if counts.max() / win_n < MIN_PURITY: continue
        fv = build_fvec_seizeit2(data[:, start:start+win_n], chs)
        t  = start / sfreq
        if dom == LBL['preictal']: X_pre.append(fv); t_pre.append(t)
        else:                        X_inter.append(fv); t_inter.append(t)

    del data, labels; gc.collect()
    Xp = np.array(X_pre,   dtype=np.float32) if X_pre   else np.empty((0,38),dtype=np.float32)
    Xi = np.array(X_inter, dtype=np.float32) if X_inter else np.empty((0,38),dtype=np.float32)
    tp = np.array(t_pre,   dtype=np.float64)
    ti = np.array(t_inter, dtype=np.float64)

    if len(Xp)+len(Xi) == 0:
        return dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                    window_min=window_min, file_order=file_order,
                    n_pre=0, n_inter=0, n_total=0, n_seizures=n_seizures,
                    status='vazio', path=None)

    _save_level(out_path, Xp, Xi, tp, ti, SEIZEIT2_R0_TARGETS,
                dataset, patient, fkey, lv, window_min, file_order, n_seizures,
                col_start=0, col_end=38)
    return dict(dataset=dataset, patient=patient, fkey=fkey, level=lv,
                window_min=window_min, file_order=file_order,
                n_pre=len(Xp), n_inter=len(Xi), n_total=len(Xp)+len(Xi),
                n_seizures=n_seizures, status='ok', path=out_path)


def build_task_list_v5():
    '''Monta lista de tarefas.
    - CHBMIT/Siena/Mendeley: 1 tarefa por (fkey, window_min) — processa todos os 5 níveis
    - SeizeIT2: 1 tarefa por (fkey, window_min) — só R0

    Uma tarefa é "pendente" se QUALQUER nível esperado ainda não existe no disco.
    Isso garante retomada correta mesmo se apenas alguns níveis foram salvos.
    '''
    tasks_r5  = []   # (dataset, patient, fkey, window_min, file_order)
    tasks_s2  = []   # (patient,  fkey, window_min, file_order)

    for ds in ['CHBMIT','Siena','Mendeley']:
        for pat in PATIENTS[ds]:
            ref_files = sorted(glob.glob(os.path.join(L1_DIR, f'{ds}__{pat}__*_L1_w{PREICTAL_WINDOWS_MIN[0]}.npz')))
            if not ref_files:
                ref_files_any = sorted(glob.glob(os.path.join(L1_DIR, f'{ds}__{pat}__*_L1_w*.npz')))
                seen = set(); unique = []
                for fp in ref_files_any:
                    fk = os.path.basename(fp).replace(f'{ds}__{pat}__','').split('_L1_')[0]
                    if fk not in seen: seen.add(fk); unique.append(fp)
                ref_files = unique

            for file_order, ref_fp in enumerate(ref_files):
                bn   = os.path.basename(ref_fp)
                fkey = bn.replace(f'{ds}__{pat}__','').split('_L1_')[0]
                for w in PREICTAL_WINDOWS_MIN:
                    # Pendente se QUALQUER dos 5 níveis não existe
                    any_missing = any(
                        not os.path.exists(os.path.join(FEAT_DIR, lv, f'{ds}__{pat}__{fkey}__w{w}.npz'))
                        for lv in LEVELS
                    )
                    if any_missing:
                        tasks_r5.append((ds, pat, fkey, w, file_order))

    for pat in PATIENTS['SeizeIT2']:
        ref_files = sorted(glob.glob(os.path.join(L1_DIR, f'SeizeIT2__{pat}__*_L1_w{PREICTAL_WINDOWS_MIN[0]}.npz')))
        if not ref_files:
            ref_files_any = sorted(glob.glob(os.path.join(L1_DIR, f'SeizeIT2__{pat}__*_L1_w*.npz')))
            seen = set(); unique = []
            for fp in ref_files_any:
                fk = os.path.basename(fp).replace(f'SeizeIT2__{pat}__','').split('_L1_')[0]
                if fk not in seen: seen.add(fk); unique.append(fp)
            ref_files = unique

        for file_order, ref_fp in enumerate(ref_files):
            bn   = os.path.basename(ref_fp)
            fkey = bn.replace(f'SeizeIT2__{pat}__','').split('_L1_')[0]
            for w in PREICTAL_WINDOWS_MIN:
                out = os.path.join(FEAT_DIR, 'R0', f'SeizeIT2__{pat}__{fkey}__w{w}.npz')
                if not os.path.exists(out):
                    tasks_s2.append((pat, fkey, w, file_order))

    return tasks_r5, tasks_s2
