"""Classificação LLDA em 54 classes: trie, múltiplos beams, lotes CUDA e retomada."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, jaccard_score, precision_recall_fscore_support
from topicos_predefinidos import CLASSES, CATALOG
from llda_decoding_restrito import LabelTrie, build_prompt, label_token_sequences, DATA_PATH, HERE

# Descritores repetidos recebem qualificadores para preservar as 54 saídas.
QUALIFIED = {
    'cmp-lg': 'Computation and Language (Legacy Category)',
    'cs.IT': 'Information Theory (Computer Science)',
    'math.IT': 'Information Theory (Mathematics)',
    'cs.NA': 'Numerical Analysis (Computer Science)',
    'math.NA': 'Numerical Analysis (Mathematics)',
    'math.ST': 'Statistics Theory (Mathematics)',
    'stat.TH': 'Statistics Theory (Statistics)',
}
LABELS = [QUALIFIED.get(c, CATALOG[c][0]) for c in CLASSES]
assert len(LABELS) == len(set(LABELS)) == 54


def full_dataset():
    """Recupera os 2.000 resumos, conferindo prefixo e classes sem usar rótulos no prompt."""
    df = pd.read_csv(DATA_PATH).reset_index(drop=True)
    if len(df) != 2000 or df.doc_id.duplicated().any():
        raise ValueError('Esperados 2.000 doc_ids únicos.')
    corpus = {}
    for path in sorted((HERE.parent / 'Castor-data/datasets/AAPD').glob('*.txt')):
        lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(lines) % 2:
            raise ValueError(f'Arquivo com número ímpar de linhas: {path}')
        for offset in range(0, len(lines), 2):
            text = ' '.join(lines[offset].split())
            codes = frozenset(lines[offset + 1].split())
            corpus.setdefault((text[:140], codes), []).append((text, f'{path.name}:artigo_{offset//2}'))
    texts, sources = [], []
    for row in df.itertuples():
        codes = frozenset(c.strip() for c in row.classes_reais_aapd.split(','))
        if not codes or codes - set(CLASSES):
            raise ValueError(f'Classes inválidas no doc_id {row.doc_id}')
        prefix = row.resumo_texto.removesuffix('...')
        matches = corpus.get((prefix[:140], codes), [])
        matches = [(text, source) for text, source in matches if text.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f'doc_id={row.doc_id}: {len(matches)} textos completos correspondentes.')
        text, source = matches[0]
        texts.append(text)
        sources.append(source)
    df['resumo_csv_abreviado'] = df.resumo_texto
    df['resumo_texto'] = texts
    df['fonte_texto'] = sources
    return df


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA não está acessível. Execute em um kernel com acesso à GPU NVIDIA. Não há fallback para CPU.')
    return {'name': torch.cuda.get_device_name(0), 'free_bytes': torch.cuda.mem_get_info()[0],
            'total_bytes': torch.cuda.mem_get_info()[1]}


def evaluate_results(df, outputs, top_k=3):
    expected = set(df.doc_id)
    if set(outputs.doc_id) != expected:
        raise ValueError('A avaliação exige resultados de todos os documentos selecionados.')
    label_index = {label: i for i, label in enumerate(LABELS)}
    code_index = {code: i for i, code in enumerate(CLASSES)}
    Y, predicted = np.zeros((len(df), 54), dtype=int), np.zeros((len(df), 54), dtype=int)
    top1, precisions, recalls = [], [], []
    grouped = {doc_id: rows.sort_values('rank') for doc_id, rows in outputs.groupby('doc_id')}
    for i, row in enumerate(df.itertuples()):
        Y[i, [code_index[c.strip()] for c in row.classes_reais_aapd.split(',')]] = 1
        ranked = grouped[row.doc_id]
        indices = [label_index[label] for label in ranked.head(top_k).topico]
        predicted[i, indices] = 1
        first = label_index[ranked.iloc[0].topico]
        top1.append(Y[i, first])
        hit = (Y[i] & predicted[i]).sum()
        precisions.append(hit / max(predicted[i].sum(), 1))
        recalls.append(hit / Y[i].sum())
    metrics = pd.DataFrame([{
        'documentos': len(df), 'classes': 54, 'validade_exata': outputs.topico.isin(LABELS).mean(),
        'acerto_top1': np.mean(top1), f'precisao_top{top_k}': np.mean(precisions),
        f'recall_top{top_k}': np.mean(recalls),
        'F1 micro': f1_score(Y, predicted, average='micro', zero_division=0),
        'F1 macro (54)': f1_score(Y, predicted, average='macro', zero_division=0),
        'Jaccard': jaccard_score(Y, predicted, average='samples', zero_division=0),
    }])
    precision, recall, f1, support = precision_recall_fscore_support(Y, predicted, average=None, zero_division=0)
    per_class = pd.DataFrame({'Classe': CLASSES, 'Nome': LABELS, 'Precisão': precision,
                              'Recall': recall, 'F1': f1, 'Suporte': support})
    return metrics, per_class


def run_gpu(output_dir=None, model_name='Qwen/Qwen2.5-0.5B-Instruct', batch_size=8,
            num_beams=4, top_k=3, cache_dir=None, local_files_only=False, limit=None):
    gpu = require_gpu()
    if batch_size < 1 or not 1 < num_beams <= 54 or not 1 <= top_k <= num_beams:
        raise ValueError('Confira batch_size, num_beams e top_k.')
    df = full_dataset()
    if limit is not None:
        if not 1 <= limit <= len(df):
            raise ValueError('limit deve estar entre 1 e 2000.')
        df = df.head(limit).copy()
    output = Path(output_dir) if output_dir else HERE / 'resultados/llda_trie_54_gpu'
    output.mkdir(parents=True, exist_ok=True)
    payload = df[['doc_id', 'resumo_texto', 'classes_reais_aapd']].to_json(orient='records')
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    config = {'model_name': model_name, 'num_beams': num_beams, 'top_k': top_k,
              'batch_size': batch_size, 'labels': LABELS, 'classes': CLASSES,
              'data_sha256': fingerprint, 'n_documents': len(df), 'seed': 42,
              'dtype': 'float16', 'length_penalty': 1.0, 'decoding': 'standard trie beam search'}
    config_path = output / 'configuracao.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if any(previous.get(key) != value for key, value in config.items()):
            raise ValueError('A pasta contém outra configuração/dataset. Use outra pasta para preservar a execução.')
    checkpoint = output / 'outputs.csv'
    outputs = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
    if len(outputs):
        if not set(outputs.doc_id) <= set(df.doc_id):
            raise ValueError('Checkpoint contém documentos de outro conjunto.')
        for _, rows in outputs.groupby('doc_id'):
            if sorted(rows['rank']) != list(range(1, num_beams + 1)) or not rows.topico.isin(LABELS).all():
                raise ValueError('Checkpoint incompleto ou com saídas inválidas.')
    done = set(outputs.doc_id) if len(outputs) else set()
    pending = df[~df.doc_id.isin(done)].copy()
    print(f'GPU: {gpu["name"]}; {len(df)} artigos; {len(done)} concluídos; {len(pending)} pendentes', flush=True)
    pd.DataFrame({'Classe': CLASSES, 'Nome para LLM': LABELS}).to_csv(output / 'catalogo.csv', index=False)
    df.to_csv(output / 'artigos_completos.csv', index=False)
    if len(pending):
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,
                                                 local_files_only=local_files_only)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir,
                                                    local_files_only=local_files_only,
                                                    dtype=torch.float16).to('cuda').eval()
        config.update({'gpu': gpu['name'], 'model_revision': getattr(model.config, '_commit_hash', None)})
        import importlib.metadata
        config['versions'] = {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'pandas']}
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        # O mesmo sufixo do chat_template define uma fronteira fixa de resposta.
        reference_prompt = build_prompt(tokenizer, '', LABELS)
        reference_ids, label_ids = label_token_sequences(tokenizer, reference_prompt, LABELS)
        probe_ids, probe_label_ids = label_token_sequences(
            tokenizer, build_prompt(tokenizer, 'Boundary check', LABELS), LABELS)
        if label_ids != probe_label_ids:
            raise ValueError('Os tokens dos nomes dependem do texto do prompt.')
        suffix_length = 0
        for left, right in zip(reversed(reference_ids), reversed(probe_ids)):
            if left != right:
                break
            suffix_length += 1
        if not suffix_length:
            raise ValueError('O chat_template não tem um sufixo estável de resposta.')
        trie = LabelTrie(label_ids, tokenizer.eos_token_id)
        budget = max(map(len, label_ids)) + 1
        pending['prompt'] = pending.resumo_texto.map(lambda text: build_prompt(tokenizer, text, LABELS))
        encoded = tokenizer(pending.prompt.tolist(), add_special_tokens=False)['input_ids']
        for ids in encoded:
            if ids[-suffix_length:] != reference_ids[-suffix_length:]:
                raise ValueError('Fronteira do template varia; a trie precisa ser construída por prompt.')
        pending['n_tokens'] = list(map(len, encoded))
        if pending.n_tokens.max() + budget > model.config.max_position_embeddings:
            raise ValueError('Resumo completo excede o contexto do modelo; não truncamos silenciosamente.')
        # Lotes de comprimento próximo reduzem o padding. Os IDs originais são mantidos.
        pending = pending.sort_values('n_tokens', kind='stable')
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(pending), batch_size):
            batch = pending.iloc[start:start + batch_size]
            inputs = tokenizer(batch.prompt.tolist(), add_special_tokens=False, padding=True,
                               return_tensors='pt').to('cuda')
            width = inputs.input_ids.shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, do_sample=False, num_beams=num_beams,
                    num_return_sequences=num_beams, max_new_tokens=budget,
                    prefix_allowed_tokens_fn=trie.callback(width),
                    early_stopping=True, length_penalty=1.0,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    forced_eos_token_id=None, renormalize_logits=False,
                    return_dict_in_generate=True, output_scores=True)
            answers = tokenizer.batch_decode(generated.sequences[:, width:], skip_special_tokens=True)
            if not all(answer in LABELS for answer in answers):
                raise AssertionError('Resposta fora do catálogo ou incompleta.')
            scores = generated.sequences_scores.cpu().tolist()
            rows = []
            for position, document in enumerate(batch.itertuples()):
                for beam in range(num_beams):
                    index = position * num_beams + beam
                    label = answers[index]
                    rows.append({'doc_id': document.doc_id, 'rank': beam + 1, 'topico': label,
                                 'classe': CLASSES[LABELS.index(label)], 'score_beam': scores[index],
                                 'valido': True})
            outputs = pd.concat([outputs, pd.DataFrame(rows)], ignore_index=True)
            # Substituição atômica: a retomada nunca lê metade de um lote.
            temporary = output / 'outputs.tmp.csv'
            outputs.to_csv(temporary, index=False)
            temporary.replace(checkpoint)
            completed = len(done) + min(start + batch_size, len(pending))
            if start == 0 or (start // batch_size + 1) % 10 == 0 or completed == len(df):
                elapsed = time.perf_counter() - started
                processed = min(start + batch_size, len(pending))
                eta = (len(pending) - processed) * elapsed / processed
                print(f'{completed}/{len(df)} artigos; {processed/elapsed:.2f} artigos/s; ETA {eta/60:.1f} min', flush=True)
            del inputs, generated
        runtime = {'gpu': gpu['name'], 'segundos_execucao': time.perf_counter() - started,
                   'artigos_nesta_execucao': len(pending),
                   'vram_pico_gib': torch.cuda.max_memory_allocated() / 1024**3}
        (output / 'execucao_gpu.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
        del model
        torch.cuda.empty_cache()
    metrics, per_class = evaluate_results(df, outputs, top_k)
    metrics.to_csv(output / 'metricas.csv', index=False)
    per_class.to_csv(output / 'metricas_por_classe.csv', index=False)
    print(metrics.round(4).to_string(index=False), flush=True)
    return df, outputs, metrics, per_class


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-beams', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--cache-dir')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run_gpu(**vars(args))
