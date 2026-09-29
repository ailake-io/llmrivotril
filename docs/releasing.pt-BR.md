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

O repositório contém `.github/workflows/release.yml`. Configure o ambiente
`pypi` do GitHub como um PyPI Trusted Publisher para o repositório e workflow
exatos, depois publique um GitHub Release a partir da tag correspondente. O
workflow constrói tanto o wheel quanto a source distribution e os publica sem
um token de API de longa duração.

Não reutilize uma versão que já foi enviada: os arquivos de release do PyPI
são imutáveis. Atualize o changelog e a versão do pacote antes de criar uma
nova tag.

## Status atual do release

Em 29/09/2026, a versão `0.1.0` passou pelos gates de qualidade locais e pelo
smoke test de import/CLI em ambiente limpo. O trabalho restante é operação de
release e validação com serviços reais, não uma correção de código crítica
conhecida. Veja a [revisão técnica](technical-review.md#status-validado-em-29092026)
para o checklist completo e as limitações de integração conhecidas.
