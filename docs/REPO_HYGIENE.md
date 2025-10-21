# Repo hygiene & dados sensíveis

## Objetivo

Evitar que dados gerados, PII e arquivos pesados entrem no histórico do repositório.

## Política

- **Nunca** versionar `data_ai/`, `data_etl/`, `decrypted_output/` e saídas de notebooks.
- Preservar apenas **código e documentos**; tudo que for artefato/resultado deve ser reproduzível via pipeline.

## Uso rápido

1. Limpar o índice (sem reescrever histórico):

```bash
tools/maintenance/cleanup_repo.sh
git push origin main
```
