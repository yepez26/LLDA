"""Scores exatos de sequências, múltiplos verbalizadores e correção contextual.

A inferência usa os logits do vocabulário COMPLETO, sem máscara de trie ou beams.
O cache do prompt é compartilhado; cada candidato é avaliado por teacher forcing.
"""
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp, softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import (label_ranking_average_precision_score, label_ranking_loss,
                             coverage_error, average_precision_score, f1_score, jaccard_score,
                             precision_recall_fscore_support)
from transformers import AutoTokenizer, AutoModelForCausalLM
from llda_trie_54 import full_dataset, require_gpu, HERE, CLASSES
from llda_decoding_restrito import build_prompt, label_token_sequences
from verbalizadores_54 import VERBALIZERS, EXPRESSIONS

REFERENCES = ['', 'N/A', 'Scientific abstract.']
METHODS = ['nome_unico', 'nome_unico_corrigido', 'multiplas_expressoes', 'multiplas_expressoes_corrigidas']


def exact_scores(model, tokenizer, prompt, expressions=EXPRESSIONS, candidate_batch=24, prepared=None):
    """Log P(expressão + EOS | prompt), incluindo cada token uma única vez.

    Cada ramificação recebe uma cópia do cache; não há contaminação entre candidatos.
    Retorna soma e comprimento, sem normalização por comprimento.
    """
    if prepared is None:
        prompt_ids, responses = label_token_sequences(tokenizer, prompt, expressions)
    else:
        responses, suffix = prepared
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if prompt_ids[-len(suffix):] != suffix:
            raise ValueError('A fronteira da resposta mudou; recompile os candidatos.')
    if len(prompt_ids) + max(map(len, responses)) + 1 > model.config.max_position_embeddings:
        raise ValueError('Contexto excedido; o resumo não será truncado.')
    device = model.device
    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], device=device)
        prefill = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                        logits_to_keep=1)
        first_logp = prefill.logits[0, -1].float().log_softmax(-1)
        cache = prefill.past_key_values
        result = []
        for start in range(0, len(responses), candidate_batch):
            batch = responses[start:start+candidate_batch]
            n, width = len(batch), max(map(len, batch))
            inputs = torch.full((n, width), tokenizer.pad_token_id, device=device, dtype=torch.long)
            targets = torch.full((n, width), -100, device=device, dtype=torch.long)
            mask = torch.zeros((n, width), device=device, dtype=torch.long)
            for i, response in enumerate(batch):
                inputs[i, :len(response)] = torch.tensor(response, device=device)
                targets[i, :len(response)] = torch.tensor(response[1:] + [tokenizer.eos_token_id], device=device)
                mask[i, :len(response)] = 1
            branch_cache = copy.deepcopy(cache)
            branch_cache.batch_repeat_interleave(n)
            attention = torch.cat([torch.ones((n, len(prompt_ids)), dtype=torch.long, device=device), mask], dim=1)
            logits = model(input_ids=inputs, attention_mask=attention, past_key_values=branch_cache,
                           use_cache=True).logits
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                      targets.reshape(-1), reduction='none', ignore_index=-100)
            scores = -loss.reshape(n, width).sum(1)
            first_tokens = torch.tensor([response[0] for response in batch], device=device)
            scores += first_logp[first_tokens]
            result.extend(scores.cpu().double().tolist())
            del logits, branch_cache
    return np.array(result), np.array([len(response)+1 for response in responses])


def aggregate(logp, reference_logp, corrected=False, multiple=False, strength=1.0):
    shaped = np.asarray(logp).reshape(-1, 54, 3)
    if corrected:
        shaped = shaped - strength * np.asarray(reference_logp).reshape(1, 54, 3)
    # Mesmo número de expressões e peso 1/3 para todas as classes.
    return logsumexp(shaped, axis=2) - np.log(3) if multiple else shaped[:, :, 0]


def metric_row(y, scores, k=3):
    ranking = np.argsort(-scores, axis=1, kind='stable')
    prediction = np.zeros_like(y)
    prediction[np.arange(len(y))[:, None], ranking[:, :k]] = 1
    return {
        'LRAP': label_ranking_average_precision_score(y, scores),
        'Ranking loss': label_ranking_loss(y, scores),
        'Coverage error': coverage_error(y, scores),
        'AP micro': average_precision_score(y, scores, average='micro'),
        'Acerto top1': y[np.arange(len(y)), ranking[:, 0]].mean(),
        'F1 micro top3': f1_score(y, prediction, average='micro', zero_division=0),
        'F1 macro top3': f1_score(y, prediction, average='macro', zero_division=0),
        'Jaccard top3': jaccard_score(y, prediction, average='samples', zero_division=0),
    }, prediction


def analyze(df, logp, reference_logp, output):
    Y = np.zeros((len(df), 54), dtype=int)
    for i, row in enumerate(df.classes_reais_aapd):
        Y[i, [CLASSES.index(c.strip()) for c in row.split(',')]] = 1
    validation, evaluation = train_test_split(np.arange(len(df)), test_size=0.8, random_state=42)
    # Média das probabilidades de três entradas neutras, em espaço log.
    prior = logsumexp(reference_logp, axis=0) - np.log(len(reference_logp))
    grid_rows, strengths = [], {}
    for multiple in [False, True]:
        for strength in [0, 0.25, 0.5, 0.75, 1.0]:
            scores = aggregate(logp[validation], prior, True, multiple, strength)
            value = label_ranking_average_precision_score(Y[validation], scores)
            grid_rows.append({'multiplas_expressoes': multiple, 'lambda': strength, 'LRAP validacao': value})
        rows = [r for r in grid_rows if r['multiplas_expressoes'] == multiple]
        strengths[multiple] = max(rows, key=lambda row: row['LRAP validacao'])['lambda']
    metric_rows, per_class_rows, distributions = [], [], {}
    for name, multiple, corrected in [(METHODS[0], False, False), (METHODS[1], False, True),
                                       (METHODS[2], True, False), (METHODS[3], True, True)]:
        strength = strengths[multiple] if corrected else 0
        scores = aggregate(logp, prior, corrected, multiple, strength)
        relative = softmax(scores, axis=1)
        distributions[name] = relative
        metrics, prediction = metric_row(Y[evaluation], scores[evaluation])
        metric_rows.append({'Metodo': name, 'lambda': strength, **metrics})
        precision, recall, f1, support = precision_recall_fscore_support(Y[evaluation], prediction, average=None, zero_division=0)
        per_class_rows.append(pd.DataFrame({'Metodo': name, 'Classe': CLASSES, 'Precisao': precision,
                                           'Recall': recall, 'F1': f1, 'Suporte avaliacao': support}))
        for prefix, matrix in [('scores', scores), ('distribuicao_relativa', relative)]:
            table = pd.DataFrame(matrix, columns=CLASSES)
            table.insert(0, 'doc_id', df.doc_id.to_numpy())
            table.to_csv(output / f'{prefix}_{name}.csv', index=False)
    metrics = pd.DataFrame(metric_rows)
    per_class = pd.concat(per_class_rows, ignore_index=True)
    metrics.to_csv(output / 'metricas_avaliacao.csv', index=False)
    per_class.to_csv(output / 'metricas_por_classe.csv', index=False)
    pd.DataFrame(grid_rows).to_csv(output / 'busca_lambda_validacao.csv', index=False)
    split = np.full(len(df), 'avaliacao', dtype=object)
    split[validation] = 'validacao'
    pd.DataFrame({'doc_id': df.doc_id, 'split': split}).to_csv(output / 'split.csv', index=False)
    pd.DataFrame({'Classe': CLASSES, 'Validacao': Y[validation].sum(0), 'Avaliacao': Y[evaluation].sum(0)}).to_csv(output / 'suporte.csv', index=False)
    return metrics, per_class, distributions


def run(output_dir=None, cache_dir=None, local_files_only=False, candidate_batch=24):
    gpu = require_gpu()
    output = Path(output_dir) if output_dir else HERE / 'resultados/llda_scores_exatos_54'
    output.mkdir(parents=True, exist_ok=True)
    df = full_dataset()
    configuration = {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'expressions': EXPRESSIONS,
                     'references': REFERENCES, 'candidate_batch': candidate_batch, 'classes': CLASSES,
                     'data_sha256': hashlib.sha256(df[['doc_id','resumo_texto']].to_json().encode()).hexdigest(),
                     'scoring': 'full vocabulary log P(expression + EOS), no length normalization',
                     'prompt': 'same complete 162-expression catalog for all ablations', 'seed': 42}
    config_path = output / 'configuracao.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if any(old.get(key) != value for key, value in configuration.items()):
            raise ValueError('Pasta contém outra configuração. Use uma nova pasta.')
    state_file = output / 'checkpoint.npz'
    if state_file.exists():
        with np.load(state_file) as state:
            logp, references = state['logp'].copy(), state['references'].copy()
            lengths = state['lengths'].copy()
    else:
        logp = np.full((len(df), len(EXPRESSIONS)), np.nan)
        references = np.full((len(REFERENCES), len(EXPRESSIONS)), np.nan)
        lengths = np.zeros(len(EXPRESSIONS), dtype=int)
    if logp.shape != (2000, 162) or references.shape != (3,162):
        raise ValueError('Checkpoint incompatível.')
    pending = np.flatnonzero(np.isnan(logp).any(1))
    if len(pending) or np.isnan(references).any():
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(configuration['model'], cache_dir=cache_dir, local_files_only=local_files_only)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(configuration['model'], cache_dir=cache_dir,
                                                    local_files_only=local_files_only, dtype=torch.float16).to('cuda').eval()
        revision = getattr(model.config, '_commit_hash', None)
        if config_path.exists() and old.get('model_revision') != revision:
            raise ValueError('Revisão do modelo mudou; não misture checkpoints.')
        import importlib.metadata
        configuration.update({'gpu': gpu['name'], 'model_revision': revision,
                              'versions': {p: importlib.metadata.version(p) for p in ['torch','transformers','numpy']}})
        config_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding='utf-8')
        base_ids, response_ids = label_token_sequences(tokenizer, build_prompt(tokenizer, '', EXPRESSIONS), EXPRESSIONS)
        probe_ids, probe_responses = label_token_sequences(tokenizer, build_prompt(tokenizer, 'Boundary check', EXPRESSIONS), EXPRESSIONS)
        if response_ids != probe_responses:
            raise ValueError('Tokenização depende do prompt.')
        size = 0
        for left, right in zip(reversed(base_ids), reversed(probe_ids)):
            if left != right:
                break
            size += 1
        if size == 0:
            raise ValueError('Template sem sufixo estável.')
        prepared = (response_ids, base_ids[-size:])
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for i, reference in enumerate(REFERENCES):
            if np.isnan(references[i]).any():
                references[i], lengths = exact_scores(model, tokenizer, build_prompt(tokenizer, reference, EXPRESSIONS), candidate_batch=candidate_batch, prepared=prepared)
        for count, i in enumerate(pending, 1):
            logp[i], current_lengths = exact_scores(model, tokenizer, build_prompt(tokenizer, df.iloc[i].resumo_texto, EXPRESSIONS), candidate_batch=candidate_batch, prepared=prepared)
            if not np.array_equal(lengths, current_lengths):
                raise ValueError('Tokenização dos candidatos varia entre prompts.')
            if count % 10 == 0 or count == len(pending):
                temporary = output / 'checkpoint.tmp.npz'
                np.savez_compressed(temporary, logp=logp, references=references, lengths=lengths)
                temporary.replace(state_file)
            if count == 1 or count % 50 == 0 or count == len(pending):
                elapsed = time.perf_counter() - started
                print(f'{2000-len(pending)+count}/2000; {count/elapsed:.2f} artigos/s; ETA {(len(pending)-count)*elapsed/count/60:.1f} min', flush=True)
        runtime = {'gpu': gpu['name'], 'segundos': time.perf_counter()-started,
                   'documentos_nesta_execucao': len(pending),
                   'vram_pico_gib': torch.cuda.max_memory_allocated()/1024**3}
        (output / 'execucao_gpu.json').write_text(json.dumps(runtime, indent=2))
        del model
        torch.cuda.empty_cache()
    assert np.isfinite(logp).all() and np.isfinite(references).all()
    catalog = pd.DataFrame([{'Classe': CLASSES[i], 'Variante': j, 'Expressao': expression,
                             'Tokens incluindo EOS': int(lengths[3*i+j])}
                            for i, variants in enumerate(VERBALIZERS) for j, expression in enumerate(variants)])
    catalog.to_csv(output / 'verbalizadores.csv', index=False)
    raw = pd.DataFrame(logp, columns=EXPRESSIONS)
    raw.insert(0, 'doc_id', df.doc_id.to_numpy())
    raw.to_csv(output / 'logp_brutos.csv', index=False)
    pd.DataFrame(references, index=REFERENCES, columns=EXPRESSIONS).to_csv(output / 'logp_referencias.csv')
    df[['doc_id','fonte_texto','classes_reais_aapd']].to_csv(output / 'documentos.csv', index=False)
    metrics, per_class, distributions = analyze(df, logp, references, output)
    print(metrics.round(4).to_string(index=False), flush=True)
    return df, metrics, per_class, distributions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--output-dir')
    parser.add_argument('--candidate-batch', type=int, default=24)
    run(**vars(parser.parse_args()))
