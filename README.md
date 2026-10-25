# RAG — Questions et réponses sur des documents

Application Python avec une interface Gradio pour charger un document, construire un index de recherche, poser des questions sur son contenu et exporter des paires question/réponse au format JSON.

L’application utilise LlamaIndex pour le traitement et la recherche, des endpoints Hugging Face pour le LLM et les embeddings, et Qdrant en mode local pour le stockage vectoriel. Le modèle configuré est `Qwen/Qwen2.5-VL-7B-Instruct`.

## Prérequis

- Python **3.13.x** et **uv** pour une exécution locale.
- Git, notamment pour installer la dépendance CLIP depuis son dépôt.
- Docker avec Docker Compose pour une exécution en conteneur.
- Make pour les raccourcis du projet ; les commandes équivalentes sont indiquées ci-dessous.
- Un jeton Hugging Face et des endpoints d’inférence accessibles pour le LLM et les embeddings.

## Configuration initiale

Depuis un terminal :

```bash
git clone https://github.com/Val3nt-ML/rag_qa_app.git
cd rag_qa_app
cp .env.example .env
```

Renseigner `HUGGINGFACE_TOKEN` dans `.env`, puis adapter les endpoints dans [src/config.yaml](src/config.yaml). Les adresses présentes dans ce fichier sont des endpoints spécifiques au déploiement ; les remplacer par des endpoints auxquels le jeton a accès.

| Emplacement | Rôle |
| --- | --- |
| `.env` → `HUGGINGFACE_TOKEN` | Authentification de l’application auprès de Hugging Face. |
| `.env` → `HUGGINGFACE_TOKEN_2` | Jeton des essais du notebook ; facultatif pour l’application. |
| `src/config.yaml` → `rag.llm.model_name` | URL de l’endpoint du LLM. |
| `src/config.yaml` → `rag.llm.model` | Identifiant du modèle utilisé pour charger le tokenizer. |
| `src/config.yaml` → `rag.embed_model` | URL de l’endpoint des embeddings. |
| `src/config.yaml` → `rag` | Découpage des documents, recherche, images et nombre de questions par bloc. |
| `src/config.yaml` → `qa_generator` | Seuil percentile utilisé pour sélectionner les blocs de texte. |
| `src/config.yaml` → `app` | Adresse d’écoute, port, partage et mode debug Gradio. |

[src/config.py](src/config.py) valide le YAML avec Pydantic et charge le fichier `.env` à la racine, indépendamment du répertoire de lancement. Les variables déjà définies dans l’environnement sont prioritaires sur `.env`. Docker Compose transmet les valeurs de `.env` au conteneur via `env_file`.

`.env` est ignoré par Git. Seul [.env.example](.env.example), sans valeurs secrètes, doit être versionné.

## Lancement en local

À la racine du dépôt :

```bash
uv sync --locked
uv run --locked python src/app.py
```

Ouvrir ensuite **http://localhost:7860**. Le port et l’adresse d’écoute sont définis dans `src/config.yaml`.

## Lancement avec Docker Compose

Après avoir configuré `.env` et `src/config.yaml` :

```bash
make build-rag
```

Équivalent sans Make :

```bash
docker compose up --build
```

L’interface est accessible sur **http://localhost:7860**. Le service s’appelle `rag_app`. Le Dockerfile installe les dépendances de production avec `uv sync --locked --no-dev --no-install-project` et Compose monte `./src` dans `/app/src`.

Pour relancer une image déjà construite : `make rag`. Pour arrêter les services sans supprimer les conteneurs : `make stop-rag`.

Après une modification des dépendances ou du Dockerfile, reconstruire l’image avec `make build-rag`. Une mise à jour de `.venv` sur la machine hôte ne met pas à jour les dépendances de l’image.

## Utilisation

1. Charger un document dans l’interface. Le traitement actuel utilise le lecteur PDF PyMuPDF4LLM : privilégier les fichiers PDF. L’interface propose aussi TXT, DOCX et Markdown, mais leur traitement n’est pas implémenté par un lecteur dédié dans ce parcours.
2. Cliquer sur **Build Index from document**.
3. Saisir une question, puis cliquer sur **Submit**.
4. Pour exporter des paires question/réponse, choisir leur nombre puis cliquer sur **Generate and Download Question/Answer pairs**.

Les fichiers chargés sont copiés dans `temp_uploads/`, les exports dans `temp_qa/` et la base Qdrant locale dans `qdrant_mm_db/`, relativement au répertoire de lancement. Aucun serveur Qdrant séparé n’est nécessaire.

La configuration Compose actuelle ne monte pas de volume pour ces données : elles ne sont pas conservées lors de la suppression ou de la recréation du conteneur.

## Commandes Make

`make` ou `make help` affiche l’aide.

| Commande | Action |
| --- | --- |
| `make build-rag` | Construit les images et démarre les services avec `docker compose up --build`. |
| `make rag` | Démarre les services avec `docker compose up`. |
| `make stop-rag` | Arrête les services avec `docker compose stop`. |
| `make install-hooks` | Installe le hook Git prek dans le clone courant. |
| `make check` | Lance Ruff, la vérification du formatage et ty via prek. |
| `make test` | Lance `uv run pytest` ; accepte l’absence de tests, mais conserve les véritables échecs. |

## Dépendances et mises à jour

Les dépendances sont déclarées dans [pyproject.toml](pyproject.toml) et leurs versions résolues dans [uv.lock](uv.lock). Le projet utilise uniquement uv pour les gérer.

```bash
# Installer les versions du verrouillage, avec les outils de développement.
uv sync --locked

# Rechercher les dernières versions autorisées et synchroniser l’environnement.
uv sync --upgrade

# Mettre à jour un paquet en particulier.
uv sync --upgrade-package llama-index-vector-stores-qdrant
```

Les contraintes `>=` autorisent les versions plus récentes ; uv doit aussi satisfaire les contraintes des autres dépendances et la version de Python. Versionner `pyproject.toml` et `uv.lock` ensemble après une modification.

Une résolution réussie ne garantit pas que les API des bibliothèques restent compatibles avec le code. Après une mise à jour, lancer les contrôles et les tests, puis vérifier le démarrage de l’application avant de reconstruire l’image.

## Qualité et contribution

Activer les hooks après chaque nouveau clonage :

```bash
uv sync --locked
make install-hooks
```

Avant chaque commit, [prek](.pre-commit-config.yaml) exécute :

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

Un échec bloque le commit. Les hooks sont locaux et contournables ; pour imposer ces contrôles sur GitHub, les exécuter également en CI et les rendre obligatoires dans les règles de la branche.

Pour corriger automatiquement les problèmes pris en charge par Ruff :

```bash
uv run ruff check . --fix
uv run ruff format .
```

Lancer `make test` séparément : pytest ne fait pas partie des hooks actuels. Les tests existants couvrent le chargement et la validation de la configuration ainsi que la priorité des variables d’environnement ; ils ne valident pas les appels aux endpoints ni le parcours RAG complet.

Utiliser des branches dédiées et des messages de commit au format Conventional Commits, par exemple `fix: restore Qdrant compatibility` ou `docs: update setup instructions`.

## Dépannage

- **Erreur d’import après une mise à jour** : vérifier les versions dans `uv.lock`. Une incompatibilité connue peut concerner `IDF_EMBEDDING_MODELS` entre l’intégration Qdrant de LlamaIndex et `qdrant-client`. Aligner leurs versions et vérifier l’import avant de reconstruire Docker :

  ```bash
  uv run --locked python -c "from llama_index.vector_stores.qdrant import QdrantVectorStore"
  ```

- **Endpoint Hugging Face inaccessible** : vérifier le jeton, les droits d’accès et les URL configurées. Les modèles distants nécessitent un accès réseau.
- **Conteneur marqué `unhealthy`** : le healthcheck Compose appelle `curl`, mais le Dockerfile actuel ne l’installe pas. Ce statut peut donc survenir même si l’interface répond ; consulter `docker compose logs rag_app`.
- **Changement du port Gradio** : adapter également la correspondance de ports et le healthcheck dans `docker-compose.yaml`.

## Structure du projet

```text
.
├── .env.example             # Variables à renseigner dans .env
├── .pre-commit-config.yaml  # Contrôles Ruff et ty exécutés par prek
├── Dockerfile
├── docker-compose.yaml
├── Makefile
├── pyproject.toml           # Dépendances et configuration des outils
├── uv.lock                 # Versions et sources résolues par uv
├── data/                   # Répertoire de données du dépôt
├── notebooks/              # Expérimentations
├── src/
│   ├── app.py              # Interface Gradio
│   ├── config.py           # Chargement et validation des paramètres
│   ├── config.yaml         # Paramètres applicatifs
│   ├── logic/
│   │   ├── button.py       # Gestion des boutons et de l’état de l’interface
│   │   ├── qa.py           # Génération et export des paires question/réponse
│   │   └── rag.py          # Construction de l’index et interrogation
│   └── retrieval/
│       └── rag_helper.py   # Lecture PDF, découpage et stockage vectoriel
└── tests/
    └── test_data.py        # Tests de configuration
```
