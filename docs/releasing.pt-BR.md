# Releasing

[← Voltar ao README](../README.pt-BR.md) · *[English](releasing.md)*

## Checklist de pré-release

1. Confirme que a versão em `pyproject.toml`, `src/llmrivotril/__init__.py`, e
   `CHANGELOG.md` é idêntica.
2. Confirme que o autor do pacote e o nome do projeto no PyPI estão finais.
3. Rode os gates de lint, format, type-check e testes.
4. Rode `python -m build` e `python -m twine check dist/*`.
5. Instale o wheel em um ambiente virtual limpo e rode o smoke test da CLI.
6. Rode as checagens de provider e vector-store listadas na revisão técnica.
7. Faça upload no TestPyPI primeiro ao validar um novo processo de release.

## Release no PyPI

O repositório contém `.github/workflows/release.yml`, que constrói o wheel e a
source distribution e publica com `twine` usando um token de API do PyPI.
Setup único:

1. Gere um token de API no PyPI (abrangente na conta pro primeiro upload;
   restrinja ao projeto `llmrivotril` depois que ele existir lá).
2. Adicione como secret do repositório com o nome `PYPI_API_TOKEN`
   (Settings → Secrets and variables → Actions).

Depois disso, publicar um GitHub Release a partir de uma tag de versão
dispara o workflow automaticamente. `twine upload --skip-existing` torna uma
nova execução do workflow (por exemplo, após uma falha transitória) segura --
não dá erro em arquivos que já foram enviados para aquela versão.

Não reutilize uma versão que já foi enviada: os arquivos de release do PyPI
são imutáveis. Atualize o changelog e a versão do pacote antes de criar uma
nova tag.

## Status atual do release

Em 29/09/2026, a versão `0.1.0` passou pelos gates de qualidade locais e pelo
smoke test de import/CLI em ambiente limpo. O trabalho restante é operação de
release e validação com serviços reais, não uma correção de código crítica
conhecida. Veja a [revisão técnica](technical-review.md#status-validado-em-29092026)
para o checklist completo e as limitações de integração conhecidas.
