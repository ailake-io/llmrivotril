# Releasing

[← Voltar ao README](../README.pt-BR.md) · *[English](releasing.md)*

## Checklist de pré-release

1. Confirme que a versão em `pyproject.toml`, `src/llmrivotril/__init__.py`, e
   `CHANGELOG.md` é idêntica.
   O badge de versão do README é obtido dinamicamente do PyPI e não deve ser
   editado manualmente. A suíte de testes também verifica que a versão exposta
   pelo pacote coincide com a versão do `pyproject.toml`.
2. Confirme que o autor do pacote e o nome do projeto no PyPI estão finais.
3. Rode os gates de lint, format, type-check e testes.
4. Rode `python -m build` e `python -m twine check dist/*`.
5. Instale o wheel em um ambiente virtual limpo e rode o smoke test da CLI.
6. Rode as checagens de provider e vector-store listadas na revisão técnica.
7. Faça upload no TestPyPI primeiro ao validar um novo processo de release.

## Release no PyPI

O repositório contém `.github/workflows/pypi-publish.yml`, que constrói o wheel e a
source distribution e publica usando o token legado do PyPI armazenado como
secret `PYPI_API_TOKEN`. O Trusted Publishing está temporariamente pausado
porque a lista de actions permitidas bloqueava as actions de artefatos e a
action de publicação da PyPA, causando `startup_failure` sem jobs. A causa
era a política do repositório, não a autenticação no PyPI.

Setup único:

1. Crie um token de API no PyPI restrito ao projeto `llmrivotril` e salve-o no
   secret de Actions `PYPI_API_TOKEN` do repositório `ailake-io/llmrivotril`.
2. No environment `pypi` do repositório, mantenha o wait timer de cinco minutos.

Depois do setup, publicar um GitHub Release a partir de uma tag de versão
dispara o workflow automaticamente. Para recuperar uma tag antiga com o
workflow atual, execute
`gh workflow run pypi-publish.yml --ref main -f tag=vX.Y.Z`.
Eventos de release usam o workflow do commit da tag; corrigir a `main` não
altera o workflow de uma tag existente. Os arquivos do PyPI são imutáveis;
use uma versão nova para distribuições alteradas.

Mantenha o token até confirmar uma publicação OIDC bem-sucedida. Para voltar
ao OIDC, permita o SHA exato da action de publicação na política de Actions,
configure o workflow e o ambiente `pypi` correspondentes no PyPI e valide uma
publicação antes de revogar o token e excluir o secret do GitHub.

Não reutilize uma versão que já foi enviada: os arquivos de release do PyPI
são imutáveis. Atualize o changelog e a versão do pacote antes de criar uma
nova tag.

## Status atual do release

A versão `0.1.6` foi publicada com sucesso com wheel e source distribution
pelo workflow com token. Alterações em `[Unreleased]` não fazem parte dessa
tag publicada, mesmo quando já incorporadas posteriormente à `main`.
