#!/bin/bash

# install.sh - Script d'installation pour Sincerity Scoring Script
# Author: Romain
# License: MIT License
# Objectif: Installer les dépendances et lancer le script en une seule commande

echo "=== Début de l'installation pour Sincerity Scoring Script ==="

# Vérification des prérequis système
echo "Vérification des prérequis système..."
if ! command -v python3 &> /dev/null; then
    echo "Erreur : Python 3.8+ n'est pas installé. Installation requise."
    echo "Sur Ubuntu/Debian : sudo apt update && sudo apt install python3 python3-pip"
    echo "Sur Mac : brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
    echo "Erreur : Python 3.8+ requis. Version actuelle : $PYTHON_VERSION"
    exit 1
fi

if ! command -v pip3 &> /dev/null; then
    echo "Installation de pip3..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
fi

echo "Prérequis OK : Python $PYTHON_VERSION et pip3 détectés."

# Mise à jour de pip
echo "Mise à jour de pip..."
pip3 install --upgrade pip

# Installation des dépendances avec configuration minimale
echo "Installation des dépendances..."

# 1. NLTK (Natural Language Toolkit)
echo "Installation de NLTK (v3.8.1)..."
echo "- Config minimum : Python 3.7+"
echo "- Utilité : Tokenisation, stopwords pour analyse de texte"
pip3 install nltk==3.8.1
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
echo "NLTK configuré avec punkt et stopwords."

# 2. Sentence-Transformers
echo "Installation de Sentence-Transformers (v2.2.2)..."
echo "- Config minimum : Python 3.6+, PyTorch 1.6+ (installé automatiquement)"
echo "- Utilité : Embeddings sémantiques pour alignement vision"
pip3 install sentence-transformers==2.2.2
echo "Sentence-Transformers prêt."

# 3. NumPy
echo "Installation de NumPy (v1.24.3)..."
echo "- Config minimum : Python 3.8+"
echo "- Utilité : Calculs numériques (Kalman Filter, tendances)"
pip3 install numpy==1.24.3
echo "NumPy installé."

# 4. Scikit-Learn
echo "Installation de Scikit-Learn (v1.3.0)..."
echo "- Config minimum : Python 3.8+, NumPy 1.17+, SciPy 1.5+ (installés automatiquement)"
echo "- Utilité : TF-IDF pour clarté lexicale"
pip3 install scikit-learn==1.3.0
echo "Scikit-Learn prêt."

# Vérification des installations
echo "Vérification des installations..."
python3 -c "import nltk, sentence_transformers, numpy, sklearn; print('Toutes les dépendances sont installées avec succès !')"

# Lancement du script
echo "Lancement de Sincerity Scoring Script..."
python3 sincerity_score.py

echo "=== Installation et exécution terminées ! ==="
echo "Si des erreurs apparaissent, contactez-moi : [consulting@web3-crypto.xyz]"
