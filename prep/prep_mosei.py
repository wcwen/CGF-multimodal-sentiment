# -*- coding: utf-8 -*-
"""Build CMU-MOSEI aligned_50.pkl (MMSA schema) from official .csd computational sequences.

Conventions (validated to reproduce MMSA MOSI aligned_50.pkl exactly, cosine=1.0):
- text:  bert-base-uncased last hidden states, tokens padded/truncated to 50 (float32)
- text_bert: (3,50) = [input_ids, attention_mask, token_type_ids] (int64)
- audio/vision: raw COVAREP(74)/FACET-4.2(35) frames averaged within each aligned-word
  interval (absolute video time), word-level steps padded with zeros to 50 steps (float64)
- id: f"{video}$_{seg_idx}"; labels col0 sentiment in [-3,3]
Usage:
  python prep_mosei.py av      # stage 1: audio+visual alignment (resumable)
  python prep_mosei.py bert    # stage 2: BERT text (resumable, long)
  python prep_mosei.py pack    # stage 3: assemble final pkl
"""
import os, sys, pickle, time
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1')
import numpy as np
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
RAW = os.path.join(ROOT, 'mosei_raw')
WORK = os.path.join(ROOT, 'mosei_work')
os.makedirs(WORK, exist_ok=True)

MAX_W = 50  # word-level steps for audio/vision
MAX_T = 50  # token-level steps for text

# ---- official SDK folds ----
_folds = {}
exec(open(os.path.join(RAW, 'cmu_mosei_std_folds.py'), encoding='utf-8').read(), _folds)
SPLITS = {'train': _folds['standard_train_fold'],
          'valid': _folds['standard_valid_fold'],
          'test': _folds['standard_test_fold']}


def load_csd(name):
    f = h5py.File(os.path.join(RAW, name), 'r')
    d = f[list(f.keys())[0]]['data']
    return f, d


def seg_words(words_feat, words_int, s, e):
    """word strings and intervals whose midpoint falls inside segment [s,e]; drop 'sp'"""
    mid = (words_int[:, 0] + words_int[:, 1]) / 2.0
    idx = np.where((mid >= s) & (mid <= e))[0]
    ws = [words_feat[i][0].decode('utf-8', 'replace') for i in idx]
    wi = words_int[idx]
    keep = [j for j, w in enumerate(ws) if w != 'sp']
    return [ws[j] for j in keep], wi[keep]


def avg_within(frames, fint, bounds):
    """average frames overlapping each interval in bounds -> (n, D); zero if none"""
    n, D = bounds.shape[0], frames.shape[1]
    out = np.zeros((n, D), dtype=np.float64)
    for j in range(n):
        ws, we = bounds[j]
        m = (fint[:, 0] < we) & (fint[:, 1] > ws)
        if m.any():
            v = np.nanmean(frames[m], axis=0)
            out[j] = np.nan_to_num(v)
    return out


def build_index(videos, words_d, labels_d, cov_d, facet_d):
    rows = []
    for vid in videos:
        if vid not in labels_d:
            continue
        w_feat = words_d[vid]['features'][...]
        w_int = words_d[vid]['intervals'][...]
        lab = labels_d[vid]['features'][...]
        ints = labels_d[vid]['intervals'][...]
        for k in range(ints.shape[0]):
            s, e = float(ints[k, 0]), float(ints[k, 1])
            ws, wi = seg_words(w_feat, w_int, s, e)
            rows.append({'vid': vid, 'k': k, 's': s, 'e': e,
                         'label': float(lab[k, 0]), 'words': ws, 'wint': wi,
                         'has_cov': vid in cov_d, 'has_facet': vid in facet_d})
    return rows


def stage_av():
    print('loading labels/words ...', flush=True)
    _, words_d = load_csd('CMU_MOSEI_TimestampedWords.csd')
    _, labels_d = load_csd('CMU_MOSEI_Labels.csd')
    f_cov, cov_d = load_csd('CMU_MOSEI_COVAREP.csd')
    f_fa, facet_d = load_csd('CMU_MOSEI_VisualFacet42.csd')
    for split, videos in SPLITS.items():
        ck = os.path.join(WORK, f'av_{split}.npz')
        if os.path.exists(ck):
            print(split, 'done, skip'); continue
        t0 = time.time()
        rows = build_index(videos, words_d, labels_d, cov_d, facet_d)
        n = len(rows)
        A = np.zeros((n, MAX_W, 74), dtype=np.float64)
        V = np.zeros((n, MAX_W, 35), dtype=np.float64)
        ids, labels, raws = [], [], []
        cov_cache = fa_cache = None
        last_cov = last_fa = None
        for i, r in enumerate(rows):
            if r['vid'] != last_cov:
                cov_cache = cov_d[r['vid']]['features'][...]
                cov_int = cov_d[r['vid']]['intervals'][...]
                last_cov = r['vid']
            if r['vid'] != last_fa:
                fa_cache = facet_d[r['vid']]['features'][...]
                fa_int = facet_d[r['vid']]['intervals'][...]
                last_fa = r['vid']
            wi = r['wint'][:MAX_W]
            if wi.shape[0]:
                A[i, :wi.shape[0]] = avg_within(cov_cache, cov_int, wi)
                V[i, :wi.shape[0]] = avg_within(fa_cache, fa_int, wi)
            ids.append(f"{r['vid']}$_{r['k']}")
            labels.append(r['label'])
            raws.append(' '.join(r['words']))
            if (i + 1) % 1000 == 0:
                print(f'{split} {i+1}/{n} {time.time()-t0:.0f}s', flush=True)
        np.savez_compressed(ck, audio=A, vision=V, ids=ids, labels=labels, raws=raws)
        print(f'{split} finished n={n} {time.time()-t0:.0f}s', flush=True)
    f_cov.close(); f_fa.close()


def stage_bert():
    import torch
    from transformers import BertTokenizerFast, BertModel
    tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased'); model.eval()
    for split in SPLITS:
        ck = os.path.join(WORK, f'av_{split}.npz')
        out = os.path.join(WORK, f'bert_{split}.npy')
        done_f = os.path.join(WORK, f'bert_{split}.done')
        if os.path.exists(done_f):
            print(split, 'done, skip'); continue
        z = np.load(ck, allow_pickle=True)
        raws = z['raws']; n = len(raws)
        T = np.zeros((n, MAX_T, 768), dtype=np.float32)
        TB = np.zeros((n, 3, MAX_T), dtype=np.int64)
        start = 0
        if os.path.exists(out):
            T = np.load(out); start = int(np.load(os.path.join(WORK, f'bert_{split}.idx.npy')))
            print(split, 'resume from', start, flush=True)
        BS = 64
        with torch.no_grad():
            i = start
            while i < n:
                batch = [str(x) for x in raws[i:i+BS]]
                enc = tok(batch, padding='max_length', max_length=MAX_T,
                          truncation=True, return_tensors='pt')
                hs = model(**enc).last_hidden_state
                T[i:i+BS] = hs.numpy().astype(np.float32)
                TB[i:i+BS, 0] = enc['input_ids'].numpy()
                TB[i:i+BS, 1] = enc['attention_mask'].numpy()
                TB[i:i+BS, 2] = enc['token_type_ids'].numpy()
                i += BS
                if i % (BS*50) == 0 or i >= n:
                    np.save(out, T); np.save(os.path.join(WORK, f'bert_{split}.idx.npy'), np.array(i))
                    np.save(os.path.join(WORK, f'bert_{split}.tb.npy'), TB[:i])
                    print(f'{split} {i}/{n}', flush=True)
        np.save(done_f, np.array(1))
        print(split, 'BERT done', flush=True)


def stage_pack():
    out = {}
    for split in SPLITS:
        z = np.load(os.path.join(WORK, f'av_{split}.npz'), allow_pickle=True)
        T = np.load(os.path.join(WORK, f'bert_{split}.npy'))
        TB = np.load(os.path.join(WORK, f'bert_{split}.tb.npy'))
        labels = z['labels'].astype(np.float64)
        cls = np.where(labels > 0, 2, np.where(labels < 0, 0, 1)).astype(np.float64)
        ann = np.where(labels > 0, 'Positive', np.where(labels < 0, 'Negative', 'Neutral'))
        out[split] = {
            'raw_text': list(z['raws']),
            'audio': z['audio'],
            'vision': z['vision'],
            'id': list(z['ids']),
            'text': T,
            'text_bert': TB,
            'annotations': list(ann),
            'classification_labels': cls,
            'regression_labels': labels,
        }
        print(split, 'n=', len(labels), 'text', T.shape, 'audio', z['audio'].shape)
    dst = os.path.join(ROOT, 'sucai', 'CMU-MOSEI', 'mosei_aligned_50.pkl')
    with open(dst, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print('saved:', dst, round(os.path.getsize(dst)/1e9, 2), 'GB')


if __name__ == '__main__':
    {'av': stage_av, 'bert': stage_bert, 'pack': stage_pack}[sys.argv[1]]()
