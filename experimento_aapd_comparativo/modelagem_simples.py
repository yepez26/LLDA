"""Ensaio descritivo de modelagem: tópicos descobertos não são classes predefinidas."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'experimento_classificacao_topicos'))
from llda_trie_54 import full_dataset, CLASSES


def class_spread(assignments, Y, k, coverage=0.8):
    """Menor número de tópicos dominantes que reúne a fração especificada da classe.

    Cada artigo contribui para todas as suas classes; nenhum rótulo é eleito primário.
    """
    if not 0 < coverage <= 1:
        raise ValueError('coverage deve estar em (0,1].')
    counts = np.zeros((k, Y.shape[1]), dtype=int)
    for topic in range(k):
        counts[topic] = Y[np.asarray(assignments) == topic].sum(0)
    support = counts.sum(0)
    required = np.zeros(Y.shape[1], dtype=int)
    for c, n in enumerate(support):
        if n:
            cumulative = np.sort(counts[:, c])[::-1].cumsum()
            required[c] = np.searchsorted(cumulative, int(np.ceil(coverage*n))) + 1
    return counts, required


def run(output_dir=None, iterations=35):
    output = Path(output_dir) if output_dir else HERE / 'resultados_modelagem_simples'
    output.mkdir(parents=True, exist_ok=True)
    df = full_dataset()
    if df.topico_llda.isna().any():
        raise ValueError('Título LLDA ausente.')
    Y = np.zeros((len(df), len(CLASSES)), dtype=int)
    for i, labels in enumerate(df.classes_reais_aapd):
        Y[i, [CLASSES.index(c.strip()) for c in labels.split(',')]] = 1
    inputs = {
        'LLDA + LDA': df.topico_llda.map(lambda value: ' '.join(str(value).strip('"\'`').split())).tolist(),
        'LDA direto': df.resumo_texto.str[:1500].tolist(),
    }
    configuration = {'seed':42, 'k_values':[10,54], 'iterations':iterations,
        'coverage':0.8, 'max_input_characters_direct':1500,
        'llda_titles':'reused from original CSV; original generation capped at 1500 characters',
        'evaluation':'descriptive, all 2000 documents; labels used only after topic fitting',
        'vectorizer':{'min_df':2,'max_df':0.95,'max_features':10000,'ngram_range':[1,2]},
        'data_sha256':hashlib.sha256(df[['doc_id','resumo_texto','topico_llda','classes_reais_aapd']].to_json().encode()).hexdigest()}
    vectors, matrices = {}, {}
    for method, texts in inputs.items():
        vectors[method] = CountVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS), min_df=2,
            max_df=0.95,max_features=10000,ngram_range=(1,2),token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
        matrices[method] = vectors[method].fit_transform(texts)
    common = np.logical_and.reduce([matrix.getnnz(axis=1)>0 for matrix in matrices.values()])
    configuration['diagnostic_documents'] = int(common.sum())
    configuration['empty_documents_policy'] = 'same intersection of nonempty vectorized documents for both diagnostics; LDA fits all 2000'
    summaries, spreads, topic_rows, dominant_rows = [], [], [], []
    with threadpool_limits(limits=1):
        for method, texts in inputs.items():
            vectorizer = vectors[method]
            dtm = matrices[method]
            words = vectorizer.get_feature_names_out()
            for k in [10,54]:
                print(f'{method}: K={k}; {len(df)} artigos; {dtm.shape[1]} termos',flush=True)
                model = LatentDirichletAllocation(n_components=k,random_state=42,max_iter=iterations,
                                                  learning_method='batch',n_jobs=2)
                theta = model.fit_transform(dtm)
                assignments = theta.argmax(1)
                counts, required = class_spread(assignments[common],Y[common],k)
                summaries.append({'Metodo':method,'K':k,'Documentos':len(df),
                    'Artigos comuns no diagnostico':int(common.sum()),
                    'Topicos usados':len(np.unique(assignments[common])),
                    'Classes espalhadas em 2+ topicos (80%)':int((required>1).sum()),
                    'Mediana de topicos para 80% da classe':float(np.median(required)),
                    'Documentos sem termos':int((dtm.getnnz(axis=1)==0).sum())})
                spreads.append(pd.DataFrame({'Metodo':method,'K':k,'Classe':CLASSES,
                    'Artigos da classe':Y[common].sum(0),'Topicos para 80%':required}))
                occupied = np.bincount(assignments[common],minlength=k)
                for t in range(k):
                    keywords = ', '.join(words[model.components_[t].argsort()[-8:][::-1]])
                    indices = np.argsort(-counts[t],kind='stable')[:3]
                    common_classes = '; '.join(f'{CLASSES[c]}: {counts[t,c]}/{occupied[t]}' for c in indices if counts[t,c]>0)
                    topic_rows.append({'Metodo':method,'K':k,'Topico':t+1,'Artigos':int(occupied[t]),
                                       'Palavras':keywords,'Classes frequentes (artigos/total)':common_classes})
                dominant_rows.append(pd.DataFrame({'Metodo':method,'K':k,'doc_id':df.doc_id,
                    'Topico dominante':assignments+1,'classes_reais_aapd':df.classes_reais_aapd,
                    'Incluido no diagnostico':common}))
                slug='hibrido' if method=='LLDA + LDA' else 'direto'
                pd.DataFrame(counts,index=np.arange(1,k+1),columns=CLASSES).rename_axis('Topico').to_csv(output/f'contagem_topico_classe_{slug}_k{k}.csv')
                np.savez_compressed(output/f'theta_{slug}_k{k}.npz',theta=theta,doc_ids=df.doc_id.to_numpy())
    summary=pd.DataFrame(summaries)
    spread=pd.concat(spreads,ignore_index=True)
    topics=pd.DataFrame(topic_rows)
    dominant=pd.concat(dominant_rows,ignore_index=True)
    for name, table in [('resumo',summary),('dispersao_por_classe',spread),('topicos',topics),('atribuicoes',dominant)]:
        table.to_csv(output/f'{name}.csv',index=False)
    pd.DataFrame({'doc_id':df.doc_id,'fonte_texto':df.fonte_texto,
                  'caracteres_resumo_completo':df.resumo_texto.str.len(),
                  'caracteres_entrada_direta':df.resumo_texto.str[:1500].str.len(),
                  'topico_llda':df.topico_llda}).to_csv(output/'entradas.csv',index=False)
    import importlib.metadata
    configuration['versions']={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn']}
    (output/'configuracao.json').write_text(json.dumps(configuration,ensure_ascii=False,indent=2))
    print(summary.to_string(index=False),flush=True)
    return summary,spread,topics,dominant


if __name__=='__main__':
    run()
