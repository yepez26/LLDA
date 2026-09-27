# -*- coding: utf-8 -*-
"""
dataset_loader.py

Módulo unificado para carregamento e pré-processamento de datasets
utilizados nos experimentos de modelagem de tópicos (LLDA / BERTopic).

Suporta:
1. '20newsgroups': Dataset clássico do scikit-learn (20 categorias).
2. 'aapd' (ArXiv Academic Paper Dataset): Abstracts científicos com classificação
   multi-label (54 categorias do arXiv em Ciência da Computação / Estatística),
   conforme distribuído no Castor-data (git.uwaterloo.ca/jimmylin/Castor-data.git).
"""

import os
import re
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

try:
    from sklearn.datasets import fetch_20newsgroups
except ImportError:
    fetch_20newsgroups = None


# 54 Categorias oficiais do arXiv presentes no AAPD (Computer Science, Math, Stats, Physics)
AAPD_ARXIV_CATEGORIES = [
    'cmp-lg', 'cond-mat.dis-nn', 'cond-mat.stat-mech', 'cs.AI', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL',
    'cs.CR', 'cs.CV', 'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.FL', 'cs.GT',
    'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO', 'cs.MA', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE',
    'cs.NI', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC', 'cs.SE', 'cs.SI', 'cs.SY', 'math.CO', 'math.IT',
    'math.LO', 'math.NA', 'math.NT', 'math.OC', 'math.PR', 'math.ST', 'nlin.AO', 'physics.data-an',
    'physics.soc-ph', 'q-bio.NC', 'q-bio.QM', 'quant-ph', 'stat.AP', 'stat.ME', 'stat.ML', 'stat.TH'
]


@dataclass
class DatasetContainer:
    """Contêiner padrão que encapsula dados de qualquer dataset carregado."""
    name: str
    docs: List[str]
    target_names: List[str]
    targets: List[int]  # Rótulo principal (inteiro) de cada documento
    targets_multilabel: Optional[List[List[int]]] = None  # Lista de índices ativos por doc (multi-label)
    target_labels_text: Optional[List[str]] = None  # Nome textual da classe principal
    metadata: Dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """Limpa espaços extras e caracteres estranhos mantendo a integridade textual."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_20newsgroups_dataset(
    subset: str = "all",
    remove: Tuple[str, ...] = ("headers", "footers", "quotes")
) -> DatasetContainer:
    """Carrega o dataset 20 Newsgroups via scikit-learn."""
    if fetch_20newsgroups is None:
        raise ImportError("scikit-learn não está instalado. Execute: pip install scikit-learn")

    print(f"[DatasetLoader] Carregando 20 Newsgroups (subset='{subset}')...")
    data = fetch_20newsgroups(subset=subset, remove=remove)
    
    docs = [clean_text(d) for d in data["data"]]
    targets = list(data["target"])
    target_names = list(data.target_names)
    target_labels_text = [target_names[t] for t in targets]

    return DatasetContainer(
        name="20newsgroups",
        docs=docs,
        target_names=target_names,
        targets=targets,
        targets_multilabel=[[t] for t in targets],
        target_labels_text=target_labels_text,
        metadata={"subset": subset, "remove": remove}
    )


def find_aapd_dir(custom_path: Optional[str] = None) -> str:
    """Procura automaticamente o diretório do dataset AAPD em caminhos comuns."""
    candidates = []
    if custom_path:
        candidates.append(custom_path)

    # Caminhos comuns relativos ao projeto
    candidates.extend([
        "Castor-data/datasets/AAPD",
        "../Castor-data/datasets/AAPD",
        "datasets/AAPD",
        "data/AAPD",
        "AAPD",
        "datasets/aapd",
        "data/aapd",
        "aapd"
    ])

    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path):
            files = os.listdir(path)
            if any(f.endswith(".tsv") or f.endswith(".json") or f.endswith(".txt") for f in files):
                return path

    return custom_path or "Castor-data/datasets/AAPD"


def parse_aapd_tsv_line(line: str) -> Optional[Tuple[List[int], str]]:
    """
    Interpreta uma linha do arquivo TSV do AAPD.
    Formato esperado: <label_binary_or_indices>\t<abstract_text>
    Exemplo: '0010000000...0\tIn this paper, we propose...'
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split("\t", 1)
    if len(parts) != 2:
        # Tenta separar por espaço se não houver tabulação
        parts = line.split(" ", 1)
        if len(parts) != 2:
            return None

    label_str, text = parts[0].strip(), parts[1].strip()

    # Formato binário: string de 0s e 1s (ex: '01000...')
    if all(c in "01 " for c in label_str) and len(label_str.replace(" ", "")) >= 10:
        cleaned_label = label_str.replace(" ", "")
        active_indices = [idx for idx, bit in enumerate(cleaned_label) if bit == "1"]
    else:
        # Formato de índices separados por vírgula ou espaço
        try:
            tokens = re.split(r"[,;\s]+", label_str)
            active_indices = [int(tok) for tok in tokens if tok.isdigit()]
        except Exception:
            active_indices = [0]

    if not active_indices:
        active_indices = [0]

    return active_indices, clean_text(text)


def load_aapd_dataset(
    dataset_path: Optional[str] = None,
    subset: str = "all"  # 'all', 'train', 'dev'/'val', 'test'
) -> DatasetContainer:
    """
    Carrega e formata o dataset AAPD (ArXiv Academic Paper Dataset).

    Args:
        dataset_path: Caminho da pasta que contém os arquivos TSV do AAPD
                      (ex: 'Castor-data/datasets/AAPD').
        subset: 'all' para todos os splits combinados, ou 'train', 'dev'/'val', 'test'.

    Returns:
        DatasetContainer pronto para os modelos de tópicos.
    """
    aapd_dir = find_aapd_dir(dataset_path)

    if not os.path.exists(aapd_dir):
        raise FileNotFoundError(
            f"Diretório do dataset AAPD não encontrado em '{aapd_dir}'.\n"
            f"Para baixar o dataset, execute:\n"
            f"  git clone https://git.uwaterloo.ca/jimmylin/Castor-data.git\n"
            f"E aponte para 'Castor-data/datasets/AAPD' ou coloque a pasta AAPD no diretório do projeto."
        )

    print(f"[DatasetLoader] Carregando AAPD de: '{aapd_dir}' (subset='{subset}')...")

    # Mapeia os arquivos de acordo com o subset solicitado
    files_to_load = []
    subset_lower = subset.lower()

    file_mapping = {
        "train": ["train.tsv", "train.txt"],
        "dev": ["dev.tsv", "val.tsv", "valid.tsv", "dev.txt", "val.txt"],
        "val": ["dev.tsv", "val.tsv", "valid.tsv", "dev.txt", "val.txt"],
        "test": ["test.tsv", "test.txt"]
    }

    if subset_lower in file_mapping:
        found_file = False
        for fname in file_mapping[subset_lower]:
            fpath = os.path.join(aapd_dir, fname)
            if os.path.exists(fpath):
                files_to_load.append(fpath)
                found_file = True
                break
        if not found_file:
            print(f"[DatasetLoader] Aviso: split '{subset}' não encontrado especificamente. Carregando arquivos disponíveis.")
            files_to_load = [os.path.join(aapd_dir, f) for f in os.listdir(aapd_dir) if f.endswith((".tsv", ".txt"))]
    else:
        # subset == 'all'
        priority_files = ["train.tsv", "dev.tsv", "test.tsv"]
        for fname in priority_files:
            fpath = os.path.join(aapd_dir, fname)
            if os.path.exists(fpath):
                files_to_load.append(fpath)
        if not files_to_load:
            files_to_load = [os.path.join(aapd_dir, f) for f in os.listdir(aapd_dir) if f.endswith((".tsv", ".txt"))]

    if not files_to_load:
        raise FileNotFoundError(f"Nenhum arquivo .tsv ou .txt válido encontrado em '{aapd_dir}'.")

    docs: List[str] = []
    targets_multilabel: List[List[int]] = []
    primary_targets: List[int] = []

    cat_to_idx = {cat: i for i, cat in enumerate(AAPD_ARXIV_CATEGORIES)}

    for fpath in files_to_load:
        print(f"[DatasetLoader] Lendo '{os.path.basename(fpath)}'...")
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Detecta se o arquivo é em linhas alternadas (ex: linha 0 texto, linha 1 categorias)
        is_alternating = False
        if len(lines) >= 2:
            second_line_tokens = lines[1].split()
            if all(tok in cat_to_idx for tok in second_line_tokens):
                is_alternating = True

        if is_alternating:
            raw_texts = lines[0::2]
            raw_labels = lines[1::2]
            for text, lbl_str in zip(raw_texts, raw_labels):
                if not text:
                    continue
                cats = lbl_str.split()
                active_indices = [cat_to_idx[c] for c in cats if c in cat_to_idx]
                if not active_indices:
                    active_indices = [0]
                docs.append(clean_text(text))
                targets_multilabel.append(active_indices)
                primary_targets.append(active_indices[0])
        else:
            for line in lines:
                parsed = parse_aapd_tsv_line(line)
                if parsed:
                    active_indices, text = parsed
                    if text:
                        docs.append(text)
                        targets_multilabel.append(active_indices)
                        primary_targets.append(active_indices[0])

    target_names = list(AAPD_ARXIV_CATEGORIES)
    target_labels_text = [
        target_names[t] if t < len(target_names) else f"Category_{t}"
        for t in primary_targets
    ]

    print(f"[DatasetLoader] AAPD carregado com sucesso: {len(docs)} documentos, {len(target_names)} categorias.")

    return DatasetContainer(
        name="aapd",
        docs=docs,
        target_names=target_names,
        targets=primary_targets,
        targets_multilabel=targets_multilabel,
        target_labels_text=target_labels_text,
        metadata={"dir": aapd_dir, "files": files_to_load, "subset": subset}
    )


def load_dataset(
    dataset_name: str = "20newsgroups",
    subset: str = "all",
    dataset_path: Optional[str] = None,
    dataset_remove: Tuple[str, ...] = ("headers", "footers", "quotes")
) -> DatasetContainer:
    """
    Função principal para carregar qualquer dataset suportado pelo nome.

    Args:
        dataset_name: '20newsgroups' ou 'aapd' (insensível a maiúsculas).
        subset: 'all', 'train', 'dev'/'val', ou 'test'.
        dataset_path: Caminho customizado para os arquivos do dataset (para AAPD).
        dataset_remove: Partes a remover (apenas para 20 Newsgroups).

    Returns:
        DatasetContainer contendo docs, targets, target_names, etc.
    """
    normalized_name = dataset_name.lower().strip()

    if normalized_name in ["20newsgroups", "20news", "newsgroups"]:
        return load_20newsgroups_dataset(subset=subset, remove=dataset_remove)
    elif normalized_name in ["aapd", "castor_aapd", "arxiv"]:
        return load_aapd_dataset(dataset_path=dataset_path, subset=subset)
    else:
        raise ValueError(
            f"Dataset desconhecido: '{dataset_name}'. Datasets suportados: '20newsgroups', 'aapd'."
        )


if __name__ == "__main__":
    print("--- Testando DatasetLoader ---")
    
    # 1. Teste 20 Newsgroups
    try:
        ds_20ng = load_dataset("20newsgroups", subset="train")
        print(f"20 Newsgroups: {len(ds_20ng.docs)} docs, {len(ds_20ng.target_names)} classes.")
        print(f"Exemplo 1: {ds_20ng.docs[0][:150]}... | Label: {ds_20ng.target_labels_text[0]}")
    except Exception as e:
        print(f"Erro ao testar 20 Newsgroups: {e}")

    # 2. Teste AAPD
    try:
        ds_aapd = load_dataset("aapd", dataset_path="Castor-data/datasets/AAPD")
        print(f"AAPD: {len(ds_aapd.docs)} docs, {len(ds_aapd.target_names)} classes.")
    except Exception as e:
        print(f"Info AAPD: {e}")
