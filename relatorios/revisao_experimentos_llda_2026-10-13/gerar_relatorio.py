"""Reconstrói tabelas e gráficos a partir dos artefatos, sem repetir inferência."""
from pathlib import Path
import hashlib
import json
import html
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import plotly.express as px
from plotly.offline import get_plotlyjs
from markdown_it import MarkdownIt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = OUT / 'dados'
DATA.mkdir(exist_ok=True)
C = ROOT / 'experimento_classificacao_topicos/resultados'
A = ROOT / 'experimento_aapd_comparativo'
figures = {}

def table(df):
    # Sem dependência opcional tabulate.
    values = df.copy()
    for col in values.columns:
        values[col] = values[col].map(lambda x: f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x))
    return '| ' + ' | '.join(values.columns) + ' |\n| ' + ' | '.join(['---']*len(values.columns)) + ' |\n' + '\n'.join('| '+' | '.join(row)+' |' for row in values.to_numpy())

def save(name, df):
    df.to_csv(DATA / f'{name}.csv',index=False)
    return df

def figure(name, fig):
    fig.update_layout(template='plotly_white',font=dict(family='Arial',size=13),margin=dict(t=85,b=75))
    fig.write_json(DATA / f'{name}.plotly.json')
    figures[name] = fig

m = pd.read_csv(ROOT/'experimento_diversidade_transformers/metricas_diversidade.csv')
diversity = save('diversidade_por_penalidade',m.groupby('diversity_penalty').agg(
    documentos=('doc_id','size'), taxa_unicos=('unique_topic_rate','mean'), distinct_2=('distinct_2','mean'),
    diversidade_semantica=('semantic_diversity','mean'), diversidade_global=('global_semantic_diversity','first')).reset_index())
d = diversity.melt(id_vars=['diversity_penalty','documentos'],var_name='Métrica',value_name='Valor')
figure('diversidade',px.line(d[d['Métrica']!='diversidade_global'],x='diversity_penalty',y='Valor',color='Métrica',markers=True,
    title='Grade de diversidade: 1.836 documentos por penalidade<br><sup>Variedade das sugestões; coerência tópico–documento não foi medida</sup>',height=430))

k54a=pd.read_csv(A/'resultados_comparativo_54_classes/metricas_alinhamento.csv')
k54p=pd.read_csv(A/'resultados_comparativo_54_classes/metricas_sonda.csv')
k54=save('comparativo_k54',pd.concat([k54a.assign(Avaliacao='Alinhamento direto'),k54p.assign(Avaliacao='Sonda supervisionada')],ignore_index=True))
simple=save('modelagem_simplificada',pd.read_csv(A/'resultados_modelagem_simples/resumo.csv'))
figure('k54',px.bar(simple,x='Metodo',y='Mediana de topicos para 80% da classe',color='Metodo',facet_col='K',
    text='Mediana de topicos para 80% da classe',
    title='Quantos tópicos reúnem 80% dos artigos de uma classe?<br><sup>Mediana entre 54 classes; 1.986 artigos comuns; dispersão não é erro de modelagem</sup>',height=450))

baselines=save('baselines_classificacao',pd.read_csv(C/'ambos/metricas.csv'))
b=baselines[['Método','LRAP','F1 micro','F1 macro (54)']].melt(id_vars='Método',var_name='Métrica',value_name='Valor')
figure('baselines',px.bar(b,x='Método',y='Valor',color='Método',facet_col='Métrica',
    title='Classificação orientada: Seeded LDA e Guided BERTopic<br><sup>500 artigos de teste; ambos receberam fragmentos e escolheram k na validação</sup>',height=430))

exact=save('ablacoes_scores_exatos',pd.read_csv(C/'llda_scores_exatos_54/metricas_avaliacao.csv'))
labels={'nome_unico':'Nome único','nome_unico_corrigido':'Nome único + correção','multiplas_expressoes':'Três expressões','multiplas_expressoes_corrigidas':'Três expressões + correção'}
e=exact[['Metodo','LRAP','Acerto top1','F1 macro top3']].copy();e.Metodo=e.Metodo.map(labels)
e=e.melt(id_vars='Metodo',var_name='Métrica',value_name='Valor')
figure('ablacoes',px.bar(e,x='Metodo',y='Valor',color='Metodo',facet_col='Métrica',
    title='Ablações comparáveis: mesma LLM, prompt e 1.600 artigos<br><sup>Lambda escolhido em outros 400 artigos; avaliação exploratória</sup>',height=500))

per_class=pd.read_csv(C/'llda_scores_exatos_54/metricas_por_classe.csv')
p=per_class.pivot(index='Classe',columns='Metodo',values='F1')
p['delta_F1']=p.multiplas_expressoes_corrigidas-p.nome_unico
delta=save('ganhos_perdas_por_classe',p.reset_index())
figure('classes',px.bar(delta.sort_values('delta_F1'),x='Classe',y='delta_F1',color='delta_F1',color_continuous_scale='RdBu',
    title='A melhora média não é uniforme entre as 54 classes<br><sup>Δ F1: três expressões corrigidas menos nome único; mesmos 1.600 artigos</sup>',height=490))

trie_runtime=json.loads((C/'llda_trie_54_gpu/execucao_gpu.json').read_text())
exact_runtime=json.loads((C/'llda_scores_exatos_54/execucao_gpu.json').read_text())
runtime=save('custos_gpu',pd.DataFrame([
    {'Método':'Trie: 4 beams, 54 nomes','Segundos':trie_runtime['segundos_execucao'],'VRAM alocada GiB':trie_runtime['vram_pico_gib'],'Quantidade de hipóteses/scores':8000},
    {'Método':'Score exato: 162 expressões','Segundos':exact_runtime['segundos'],'VRAM alocada GiB':exact_runtime['vram_pico_gib'],'Quantidade de hipóteses/scores':324000},
]))
figure('custos',px.bar(runtime,x='Método',y='Segundos',color='Método',hover_data=['VRAM alocada GiB','Quantidade de hipóteses/scores'],
    title='Viabilidade computacional observada na RTX 5080<br><sup>Cargas de trabalho e prompts diferentes; não é um benchmark controlado de velocidade</sup>',height=430))

with np.load(C/'llda_scores_exatos_54/checkpoint.npz') as checkpoint:
    covered_mass=np.exp(logsumexp(checkpoint['logp'],axis=1))
assert np.isfinite(covered_mass).all() and (covered_mass<=1.0001).all()
ids=pd.read_csv(C/'llda_scores_exatos_54/documentos.csv').doc_id
mass=save('massa_dos_nomes_exatos',pd.DataFrame({'doc_id':ids,'massa_162_nomes_EOS':covered_mass}))
figure('massa',px.histogram(mass,x='massa_162_nomes_EOS',nbins=35,
    title='Diagnóstico adicional: probabilidade dos 162 nomes exatos + EOS<br><sup>2.000 artigos; massa externa pode incluir paráfrases e formatações válidas semanticamente</sup>',height=430))

inventory=[]
for folder in sorted(ROOT.glob('experimento*')):
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ['.py','.ipynb','.csv','.json','.npz','.md']:
            continue
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        entry={'Arquivo':str(path.relative_to(ROOT)), 'Bytes':path.stat().st_size,'SHA256':digest.hexdigest()}
        if path.suffix=='.csv':
            frame=pd.read_csv(path);entry['Linhas']=len(frame);entry['Colunas']=len(frame.columns)
            if 'doc_id' in frame:entry['Documentos']=frame.doc_id.nunique()
        if path.suffix=='.ipynb':
            notebook=json.loads(path.read_text());entry['Celulas_codigo']=sum(c['cell_type']=='code' for c in notebook['cells'])
            entry['Celulas_com_saidas']=sum(bool(c.get('outputs')) for c in notebook['cells'])
        inventory.append(entry)
for path in [ROOT/'dataset_loader.py',ROOT/'README.md',ROOT/'requirements.txt',*sorted((ROOT/'Castor-data/datasets/AAPD').glob('*.txt'))]:
    inventory.append({'Arquivo':str(path.relative_to(ROOT)), 'Bytes':path.stat().st_size,
                      'SHA256':hashlib.sha256(path.read_bytes()).hexdigest()})
save('inventario_e_proveniencia',pd.DataFrame(inventory))
bootstrap=json.loads((A/'resultados_comparativo_54_classes/configuracao.json').read_text())
save('bootstrap_k54_alinhamento',pd.DataFrame([{'Diferenca LRAP':bootstrap['diferenca_lrap'],
    'IC95 inferior':bootstrap['ic95'][0], 'IC95 superior':bootstrap['ic95'][1]}]))

md=(OUT/'sintese_experimentos.md').read_text()
replacements={
    '{{TABELA_DIVERSIDADE}}':table(diversity),
    '{{TABELA_K54}}':table(k54[['Método','Avaliacao','LRAP']]),
    '{{TABELA_MODELAGEM_SIMPLES}}':table(simple[['Metodo','K','Topicos usados','Classes espalhadas em 2+ topicos (80%)','Mediana de topicos para 80% da classe']]),
    '{{TABELA_BASELINES}}':table(baselines[['Método','LRAP','F1 micro','F1 macro (54)','k validado']]),
    '{{TABELA_EXATOS}}':table(exact[['Metodo','lambda','LRAP','Acerto top1','F1 micro top3','F1 macro top3']]),
    '{{TABELA_CUSTOS}}':table(runtime),
}
# Mantém o template separado para regenerar tabelas sem duplicação.
template=OUT/'sintese_experimentos.template.md'
if template.exists():md=template.read_text()
else:template.write_text(md)
for token,value in replacements.items():md=md.replace(token,value)
(OUT/'sintese_experimentos.md').write_text(md)
renderer=MarkdownIt('commonmark',{'html':True}).enable('table')
body=renderer.render(md)
for name,fig in figures.items():
    marker=f'<!-- FIG:{name} -->'
    body=body.replace(marker,f'<div class="chart" id="{name}">{fig.to_html(full_html=False,include_plotlyjs=False)}</div>')
style='''body{font:17px/1.65 system-ui,sans-serif;color:#182637;background:#f3f6fa;margin:0}main{max-width:1120px;margin:32px auto;padding:42px;background:white;border-radius:12px}h1{line-height:1.2}h2{margin-top:42px;border-bottom:1px solid #dce4ed;padding-bottom:8px}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto;margin:20px 0}td,th{padding:9px 12px;border:1px solid #dde4ec;text-align:left}th{background:#edf3fa}a{color:#135ab5}code{font-size:.88em}blockquote{border-left:4px solid #3c77b9;padding-left:18px;color:#33475d}.chart{margin:30px 0}pre{overflow:auto;background:#f5f7fa;padding:16px}@media print{body{background:white}main{margin:0;padding:0;max-width:none}.chart{break-inside:avoid}table{font-size:11px}a{color:inherit}}'''
page=f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Revisão dos experimentos LLDA</title><style>{style}</style><script>{get_plotlyjs()}</script></head><body><main>{body}</main></body></html>'
(OUT/'sintese_experimentos.html').write_text(page)
print(f'Relatório gerado: {len(figures)} gráficos, {len(inventory)} arquivos inventariados.')
