import sys
import os
import random
import torch
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [BASE_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataset_loader import load_dataset


class TopicModeler:
    def __init__(
        self,
        model_name,
        system_prompt,
        max_tokens,
        num_suggestions,
        output_path,
        temperature=0.1,
        dataset_name="20newsgroups",
        dataset_subset="all",
        dataset_path=None,
        dataset_remove=("headers", "footers", "quotes"),
        diversity_penalty=0.5
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_suggestions = num_suggestions
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_path = dataset_path
        self.dataset_remove = dataset_remove
        self.diversity_penalty = diversity_penalty

        self.model = None
        self.tokenizer = None

        self._load_dataset()

    def _load_dataset(self):
        """Loads the selected dataset (20 Newsgroups or AAPD)."""
        data = load_dataset(
            dataset_name=self.dataset_name,
            subset=self.dataset_subset,
            dataset_path=self.dataset_path,
            dataset_remove=self.dataset_remove
        )
        self.docs = data.docs
        self.target_names = data.target_names
        self.targets = data.targets
        self.targets_multilabel = data.targets_multilabel
        print(f"Documentos carregados: {len(self.docs)}")
        print(f"\nTotal de classificações ({len(self.target_names)}):")
        for i, name in enumerate(self.target_names[:20]):
            print(f"{i+1}. {name}")
        if len(self.target_names) > 20:
            print(f"... e mais {len(self.target_names) - 20} categorias.")


    def load_model(self):
        print(f"\nCarregando modelo '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            # quantização 4-bit para caber em GPUs com menos VRAM
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
        else:
            # CPU: sem bitsandbytes, roda em fp32 (mais lento e mais RAM)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )

        self.model.eval()
        print(f"Modelo carregado. Device: {self.model.device}")

    def _generate_topics(self, doc):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": doc},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                num_beams=self.num_suggestions,
                num_beam_groups=self.num_suggestions,
                diversity_penalty=self.diversity_penalty,
                num_return_sequences=self.num_suggestions,
                return_dict_in_generate=True,
                custom_generate="transformers-community/group-beam-search",   # <- confirme que está aqui
                trust_remote_code=True,                                       # <- e aqui
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs.sequences[:, input_len:]

        topics = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        return topics

    def process_document(self, idx):
        """Processes a single document and generates multiple topic suggestions."""
        doc = self.docs[idx]

        if not doc or not doc.strip():
            print(f"Documento {idx + 1} vazio. Pulando...")
            return None

        print(f"\nProcessando documento {idx + 1}/{len(self.docs)}...")

        topics = self._generate_topics(doc)

        for i, topic in enumerate(topics):
            print(f"  Sugestão {i + 1}: {topic}")

        clean_topics = [t.strip() for t in topics]
        topic_counts = pd.Series(clean_topics).value_counts()

        df_temp = pd.DataFrame({
            "doc_id": idx,
            "topico_sugerido": topic_counts.index,
            "frequencia": topic_counts.values
        })

        df_temp["confianca"] = df_temp["frequencia"] / self.num_suggestions

        return df_temp


def main():
    modeler = TopicModeler(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",  # troque pelo modelo desejado
        system_prompt=" You are a topic analyst. Your task is to identify the main theme of the provided text "
            "and summarize it as a concise topic consisting of 2 to 5 words. "
            "Be specific, use varied vocabulary, and avoid overly generic terms. "
            "Respond with only the topic and no additional text.",
        max_tokens=16,
        num_suggestions=3,
        diversity_penalty=0.5,
        output_path="teste_modulo.csv"
    )

    modeler.load_model()

    random.seed(42)
    sample_indices = random.sample(range(len(modeler.docs)), 3)
    print(f"\nDocumentos sorteados para teste: {sample_indices}")

    resultados = []
    for idx in sample_indices:
        df = modeler.process_document(idx)
        if df is not None:
            resultados.append(df)

    if resultados:
        df_final = pd.concat(resultados, ignore_index=True)
        print("\nResultado final:")
        print(df_final)
        df_final.to_csv(modeler.output_path, index=False)
        print(f"\nSalvo em: {modeler.output_path}")
    else:
        print("\nNenhum documento válido processado.")


if __name__ == "__main__":
    main()