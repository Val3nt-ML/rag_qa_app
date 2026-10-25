.DEFAULT_GOAL := help

.PHONY: help build-rag rag stop-rag test install-hooks check

help: ## Affiche les commandes disponibles.
	@printf 'Utilisation : make <commande>\n\n'
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z0-9_-]+:.*## / { printf "  %-16s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# À lancer après le clonage pour activer les contrôles à chaque commit.
install-hooks: ## Installe les hooks Git prek (à lancer après le clonage).
	uv run --locked prek install

# Vérifie tout le projet sans créer de commit.
check: ## Lance Ruff et ty sur tout le projet via prek.
	uv run --locked prek run --all-files

# Construit l’image et démarre les services.
build-rag: ## Construit les images Docker et démarre les services.
	docker compose up --build

# Démarre les services avec les images existantes.
rag: ## Démarre les services Docker.
	docker compose up

# Arrête les services du projet sans supprimer les conteneurs.
stop-rag: ## Arrête les services Docker sans supprimer les conteneurs.
	docker compose stop

# Pytest renvoie 5 lorsqu’aucun test n’est trouvé : ce cas est accepté.
test: ## Lance pytest via uv ; accepte l’absence de tests.
	@uv run pytest; status=$$?; \
	if [ "$$status" -eq 5 ]; then \
		echo "Aucun test trouvé."; \
	else \
		exit "$$status"; \
	fi
