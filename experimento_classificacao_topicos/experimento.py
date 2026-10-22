"""Dados, avaliação multi-rótulo e execução dos dois pipelines."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (label_ranking_average_precision_score, label_ranking_loss,
                             coverage_error, f1_score, jaccard_score,
                             precision_recall_fscore_support)
from topicos_predefinidos import CLASSES, DESCRIPTIONS, SEED_TOPICS
from seeded_lda import SeededLDA
from guided_bertopic import GuidedBERTopicClassifier

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / 'experimento_aapd_comparativo/experimento_aapd_comparativo.csv'


def load_data(path=DATA_PATH, seed=42):
    df = pd.read_csv(path)
    required = ['doc_id', 'resumo_texto', 'classes_reais_aapd']
    if len(df) != 2000 or df[required].isna().any().any() or df.doc_id.duplicated().any():
        raise ValueError('Esperados 2.000 artigos únicos, sem campos ausentes.')
    if df.resumo_texto.str.strip().eq('').any():
        raise ValueError('Há resumos vazios.')
    Y = np.zeros((len(df), len(CLASSES)), dtype=int)
    index = {c: i for i, c in enumerate(CLASSES)}
    for i, row in enumerate(df.classes_reais_aapd):
        labels = [label.strip() for label in row.split(',')]
        if not labels or any(label not in index for label in labels):
            raise ValueError(f'Classes inválidas no documento {df.iloc[i].doc_id}')
        Y[i, [index[label] for label in labels]] = 1
    train, test = train_test_split(np.arange(len(df)), test_size=0.25, random_state=seed)
    train, validation = train_test_split(train, test_size=0.2, random_state=seed)
    return df, Y, {'treino': train, 'validacao': validation, 'teste': test}


def predict_top_k(scores, k):
    """Até k rótulos; scores nulos não geram atribuição arbitrária."""
    if k < 1 or k > scores.shape[1]:
        raise ValueError('k fora do intervalo de classes.')
    prediction = np.zeros_like(scores, dtype=int)
    indices = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    rows = np.arange(len(scores))[:, None]
    prediction[rows, indices] = (scores[rows, indices] > 0).astype(int)
    return prediction


def select_k(y_validation, scores):
    rows = [{'k': k, 'F1 micro validação': f1_score(y_validation, predict_top_k(scores, k),
                                                  average='micro', zero_division=0)}
            for k in range(1, 6)]
    table = pd.DataFrame(rows)
    return int(table.loc[table['F1 micro validação'].idxmax(), 'k']), table


def evaluate(y_true, scores, k):
    prediction = predict_top_k(scores, k)
    valid = scores.sum(axis=1) > 0
    hit = y_true[np.arange(len(y_true)), scores.argmax(axis=1)] * valid
    metrics = {
        'LRAP': label_ranking_average_precision_score(y_true, scores),
        'Ranking loss': label_ranking_loss(y_true, scores),
        'Coverage error': coverage_error(y_true, scores),
        'Acerto top-1': hit.mean(),
        'F1 micro': f1_score(y_true, prediction, average='micro', zero_division=0),
        'F1 macro (54)': f1_score(y_true, prediction, average='macro', zero_division=0),
        'Jaccard': jaccard_score(y_true, prediction, average='samples', zero_division=0),
        'Sem atribuição': (~valid).mean(), 'k validado': k,
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, average=None, zero_division=0)
    per_class = pd.DataFrame({'Classe': CLASSES, 'Precisão': precision, 'Recall': recall,
                              'F1': f1, 'Suporte teste': support})
    return metrics, per_class, prediction


def run(method='ambos', output=None, iterations=20, seed_strength=20.0,
        local_files_only=False, embedding_model='sentence-transformers/all-MiniLM-L6-v2',
        min_cluster_size=15):
    df, Y, split = load_data()
    documents = df.resumo_texto.to_numpy()
    output = Path(output) if output else HERE / 'resultados' / method
    output.mkdir(parents=True, exist_ok=True)
    models = {}
    if method in ('seeded', 'ambos'):
        models['Seeded LDA'] = SeededLDA(CLASSES, SEED_TOPICS, max_iter=iterations,
                                         seed_strength=seed_strength)
    if method in ('guided', 'ambos'):
        models['Guided BERTopic'] = GuidedBERTopicClassifier(
            CLASSES, DESCRIPTIONS, SEED_TOPICS, local_files_only=local_files_only,
            embedding_model=embedding_model, min_cluster_size=min_cluster_size)
    metrics_rows, class_rows = [], []
    for name, model in models.items():
        print(f'\n{name}: treino={len(split["treino"])}, validação={len(split["validacao"])}, teste={len(split["teste"])}', flush=True)
        model.fit(documents[split['treino']])
        validation_scores = model.transform(documents[split['validacao']])
        k, validation_table = select_k(Y[split['validacao']], validation_scores)
        test_scores = model.transform(documents[split['teste']])
        metrics, per_class, prediction = evaluate(Y[split['teste']], test_scores, k)
        if name == 'Seeded LDA':
            metrics['Sem termos no vocabulário'] = model.empty_documents_.mean()
        else:
            metrics['Outliers HDBSCAN'] = model.outliers_.mean()
        metrics_rows.append({'Método': name, **metrics})
        class_rows.append(per_class.assign(Método=name))
        slug = 'seeded_lda' if name == 'Seeded LDA' else 'guided_bertopic'
        validation_table.to_csv(output / f'{slug}_validacao_k.csv', index=False)
        scored = pd.DataFrame(test_scores, columns=CLASSES)
        scored.insert(0, 'doc_id', df.iloc[split['teste']].doc_id.to_numpy())
        scored.to_csv(output / f'{slug}_scores.csv', index=False)
        assignments = df.iloc[split['teste']][['doc_id', 'classes_reais_aapd']].copy()
        assignments['classes_atribuidas'] = [', '.join(np.array(CLASSES)[row.astype(bool)]) for row in prediction]
        assignments.to_csv(output / f'{slug}_atribuicoes.csv', index=False)
        if name == 'Seeded LDA':
            model.seed_coverage_.to_csv(output / 'seeded_lda_cobertura_sementes.csv', index=False)
            model.topic_words().to_csv(output / 'seeded_lda_palavras.csv', index=False)
        else:
            model.mapping_.to_csv(output / 'guided_bertopic_mapeamento.csv', index=False)
    summary = pd.DataFrame(metrics_rows).set_index('Método')
    per_class = pd.concat(class_rows, ignore_index=True)
    summary.to_csv(output / 'metricas.csv')
    per_class.to_csv(output / 'metricas_por_classe.csv', index=False)
    split_names = np.empty(len(df), dtype=object)
    for name, indices in split.items():
        split_names[indices] = name
    pd.DataFrame({'doc_id': df.doc_id, 'split': split_names}).to_csv(output / 'split.csv', index=False)
    pd.DataFrame({'Classe': CLASSES, **{name: Y[idx].sum(0) for name, idx in split.items()}}).to_csv(output / 'suporte_classes.csv', index=False)
    import importlib.metadata
    packages = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'bertopic', 'sentence-transformers', 'umap-learn', 'hdbscan']
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'não instalado'
    config = {'seed': 42, 'method': method, 'iterations': iterations, 'seed_strength': seed_strength,
              'embedding_model': embedding_model, 'local_files_only': local_files_only,
              'min_cluster_size': min_cluster_size,
              'data_path': str(DATA_PATH), 'classes': CLASSES,
              'descriptions': DESCRIPTIONS, 'seed_topics': SEED_TOPICS, 'versions': versions,
              'split': {'treino': 1200, 'validacao': 300, 'teste': 500}}
    (output / 'configuracao.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.round(4).to_string(), flush=True)
    return summary, per_class, models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metodo', choices=['seeded', 'guided', 'ambos'], default='ambos')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--seed-strength', type=float, default=20.0)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--min-cluster-size', type=int, default=15)
    parser.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()
    run(args.metodo, args.output, args.iterations, args.seed_strength,
        args.local_files_only, args.embedding_model, args.min_cluster_size)
