# Basic LLM Project

Este repositório contém a implementação de um sistema RAG (Retrieval-Augmented Generation) básico. O objetivo é integrar modelos de linguagem de grande escala (LLMs) com recuperação de informações de documentos locais, como a documentação do Zabbix.

## 📚 **O que é um RAG System?**

RAG (Retrieval-Augmented Generation) combina técnicas de recuperação de documentos com a geração de texto. Isso permite enriquecer as respostas de modelos pré-treinados, como o GPT, com informações de fontes externas, garantindo maior relevância e precisão.

## 🎯 **Objetivo do Projeto**

- Indexar documentos locais para consulta.
- Recuperar informações relevantes com base em perguntas do usuário.
- Gerar respostas contextuais e enriquecidas utilizando modelos de linguagem.

---

## 🛠️ **Estrutura do Projeto**

```plaintext
basic-llm-project/
├── data/
│   ├── documents/          # Documentos locais para indexação
│   └── index/              # Índices gerados a partir dos documentos
├── scripts/
│   ├── index_documents.py  # Script para criar o índice de documentos
│   └── query_engine.py     # Script para consultas e respostas
├── requirements.txt        # Dependências do projeto
└── README.md               # Descrição do projeto
```

