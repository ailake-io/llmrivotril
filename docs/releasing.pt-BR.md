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
porque o GitHub Actions retorna `startup_failure` sem jobs no workflow OIDC.

Setup único:

1. Nas configurações do projeto no PyPI, abra **Publishing** e adicione um
   trusted publisher do GitHub Actions com:
   - owner: `ailake-io`
   - repository: `llmrivotril`
   - secret: `PYPI_API_TOKEN` configurado nos secrets de Actions do repositório
2. No environment `pypi` do repositório, mantenha o wait timer de cinco minutos.

Depois do setup, publicar um GitHub Release a partir de uma tag de versão
dispara o workflow automaticamente. Uma nova execução deve usar uma versão
nova se algum arquivo já tiver sido enviado, pois os arquivos de uma release do
PyPI são imutáveis.

Não reutilize uma versão que já foi enviada: os arquivos de release do PyPI
são imutáveis. Atualize o changelog e a versão do pacote antes de criar uma
nova tag.

## Status atual do release

A versão `0.1.6` contém o hardening de segurança de persistência/logs. Faça o
merge da PR de release na `main`, aguarde a CI pós-merge e então crie e publique
o GitHub Release `v0.1.6` para disparar o workflow do PyPI. Confirme a versão
publicada no PyPI.
