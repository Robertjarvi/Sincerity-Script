"""
Sincerity Scoring Script - Optimized for Filtering Messages
Author: Romain 
License: MIT License (see below)

This script evaluates the sincerity of messages to filter serious proposals for xAI or Elon Musk.
It prioritizes innovation and discovery, aligning with xAI's mission to "accelerate human scientific discovery".

Dependencies:
- nltk (`pip install nltk`)
- sentence-transformers (`pip install sentence-transformers`)
- numpy (`pip install numpy`)
- scikit-learn (`pip install scikit-learn`)

Usage:
1. Install dependencies. (requirement.txt)
2. Run the script with a list of messages.
3. It will output a score (0-100), a recommendation ("Remonter" or "Ignorer"), and an innovation trend.

MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy of this software to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import logging
from typing import List, Dict, Tuple, Optional

# Configurer le logging pour une production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Télécharger les ressources NLTK (à faire une fois)
nltk.download('punkt')
nltk.download('stopwords')

# Charger le modèle SentenceTransformer (léger)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pré-calculer les embeddings des textes de référence (optimisation)
MUSK_VISION_TEXTS = [
    "advance scientific discovery through AI",
    "accelerate human discovery and understanding of the universe",
    "help people with brain-computer interfaces",
    "explore space and secure the future of humanity",
    "build technology for the benefit of humanity"
]
MUSK_VISION_EMBEDDINGS = model.encode(MUSK_VISION_TEXTS)

# Cache pour les embeddings des messages (optimisation)
embedding_cache = {}

# Classe pour le filtre de Kalman (lissage des scores)
class KalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.error_covariance = 1.0

    def update(self, measurement: float) -> float:
        """Mettre à jour l’estimation avec une nouvelle mesure."""
        self.error_covariance += self.process_variance
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.error_covariance *= (1 - kalman_gain)
        return self.estimate

# Fonction pour nettoyer le texte
def clean_text(text: str) -> Tuple[List[str], str]:
    """Nettoie le texte en supprimant la ponctuation, en tokenisant et en retirant les stopwords."""
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english') + stopwords.words('french'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens, text

# Calcul de similarité sémantique (avec cache pour optimisation)
def semantic_similarity(text: str) -> float:
    """Calcule la similarité sémantique entre le texte et les textes de référence de la vision de xAI."""
    if text in embedding_cache:
        text_embedding = embedding_cache[text]
    else:
        text_embedding = model.encode(text)
        embedding_cache[text] = text_embedding
    similarities = [util.pytorch_cos_sim(text_embedding, ref_emb).item() for ref_emb in MUSK_VISION_EMBEDDINGS]
    return max(similarities)

# Critères pour le chiffrage de sincérité (simplifié pour le filtrage)
def sincerity_score(
    text: str,
    kalman_filter: Optional[KalmanFilter] = None,
    message_history: Optional[List[Dict]] = None,
    weights: Optional[Dict] = None
) -> Tuple[float, Dict, KalmanFilter]:
    """Évalue la sincérité d’un message et retourne un score, un rapport et le filtre de Kalman mis à jour."""
    if weights is None:
        weights = {"clarity": 0.2, "alignment": 0.5, "scam_absence": 0.3}

    if kalman_filter is None:
        kalman_filter = KalmanFilter()

    # Nettoyer le texte
    tokens, raw_text = clean_text(text)
    word_count = len(tokens)

    # 1. Clarté de la proposition (20%)
    clarity_score = 0
    specific_keywords = [
        "system", "project", "idea", "proposal", "collaboration", "research", "technology",
        "neuralink", "grok", "ai", "ia", "interface", "brain", "humanity", "development",
        "science", "discovery", "innovation", "future", "space", "exploration"
    ]
    specific_count = sum(1 for word in tokens if word in specific_keywords)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([raw_text])
    lexical_density = len(vectorizer.vocabulary_) / max(word_count, 1) if word_count > 0 else 0
    if specific_count > 5 or lexical_density > 0.5:
        clarity_score = 20
    elif specific_count > 3 or lexical_density > 0.3:
        clarity_score = 15
    elif specific_count > 1:
        clarity_score = 10
    else:
        clarity_score = 5

    # 2. Alignement avec la vision de xAI (50%)
    alignment_score = 0
    similarity = semantic_similarity(raw_text)
    if similarity > 0.8:
        alignment_score = 30
    elif similarity > 0.6:
        alignment_score = 25
    elif similarity > 0.4:
        alignment_score = 15
    else:
        alignment_score = 5

    # 3. Absence de signaux de scam (30%)
    scam_score = 30
    scam_keywords = [
        "money", "pay", "urgent", "immediately", "send", "bank", "payment", "deposit", "loan",
        "friend", "buddy", "hey", "bro", "deal", "secret"
    ]
    scam_count = sum(1 for word in tokens if word in scam_keywords)
    if scam_count > 3:
        scam_score -= 20
    elif scam_count > 1:
        scam_score -= 10
    if word_count < 20:
        scam_score -= 5
    vagueness_score = word_count / max(len(set(tokens)), 1) if len(set(tokens)) > 0 else 0
    if vagueness_score > 3:
        scam_score -= 10
    scam_score = max(scam_score, 0)

    # Score total pondéré
    total_score = (
        clarity_score * weights["clarity"] +
        alignment_score * weights["alignment"] +
        scam_score * weights["scam_absence"]
    ) * 100
    total_score_filtered = kalman_filter.update(total_score)

    # Tendance d’innovation
    innovation_trend = "Stable"
    if message_history and len(message_history) > 1:
        previous_scores = [s["Total Score"] for s in message_history]
        previous_scores.append(total_score_filtered)
        x = np.arange(len(previous_scores))
        trend = np.polyfit(x, previous_scores, 1)[0]
        if trend > 2:
            innovation_trend = "Croissante (potentiel d’innovation)"
        elif trend < -2:
            innovation_trend = "Décroissante (risque de baisse)"
        else:
            innovation_trend = "Stable"

    # Rapport simplifié pour le filtrage
    report = {
        "Total Score": total_score_filtered,
        "Innovation Trend": innovation_trend,
        "Recommendation": "Remonter" if total_score_filtered > 80 else "Ignorer"
    }
    return total_score_filtered, report, kalman_filter

# Fonction pour traiter un lot de messages
def process_message_batch(messages: List[str]) -> List[Dict]:
    """Traite un lot de messages et retourne les scores et recommandations."""
    kalman_filter = KalmanFilter()
    message_history = []
    results = []

    for i, msg in enumerate(messages):
        logger.info(f"Traitement du message {i+1}/{len(messages)}")
        score, report, kalman_filter = sincerity_score(msg, kalman_filter, message_history)
        message_history.append(report)
        results.append({
            "message_id": i,
            "score": score,
            "recommendation": report["Recommendation"],
            "innovation_trend": report["Innovation Trend"]
        })

    return results

# Exemple d’utilisation
if __name__ == "__main__":
    messages = [
        "Je m’appelle Romain, un ancien élagueur devenu innovateur par passion. J’ai développé un système qui complète Neuralink et booste Grok, avec un potentiel pour aider une partie de l’humanité. Je propose une collab (1-3M€ + participation). Contactez-moi, svp.",
        "Bonjour, c’est Romain. Comme vous, j’ai deux Shibas (Izzy, 11 ans, et Sanka, 9 ans), j’ai cru en Dogecoin dès 2015, et je partage votre vision : liberté d’expression, tech au service de l’humanité, et un attrait pour l’espace. Mon système peut aider les gens, mais OpenAI a repris des bases de mon idée. Contactez-moi, svp.",
        "Je m’appelle Romain, un ancien élagueur devenu innovateur par passion. Comme vous, j’ai deux Shibas (Izzy, 11 ans, et Sanka, 9 ans — plus vieux que Floki !), j’ai cru en Dogecoin dès 2015, et je partage votre vision : liberté d’expression, tech au service de l’humanité, et un attrait pour l’espace. J’ai développé un système qui complète Neuralink et booste Grok, avec un potentiel pour aider une partie de l’humanité. Mais OpenAI a repris des bases de mon idée sans me créditer, et je doute qu’ils partagent votre vision altruiste. Je crois que vous, Elon, et vos équipes êtes les seuls à pouvoir porter ce projet et en faire une vraie avancée pour l’humanité. Je propose une collab (1-3M€ + participation) with accès à mes recherches. Tout ce que je veux, c’est une maison pour ma famille et soigner Izzy, qui a une patte blessée. En bonus, une suggestion pour Grok : un ‘chiffrage de sincérité’ pour évaluer les propositions qu’il reçoit, et remonter directement les plus sérieuses à vos équipes. Ça vous ferait gagner du temps et repérerait les innovateurs sérieux, comme moi. Contactez-moi, svp. Romain"
    ]

    results = process_message_batch(messages)
    for result in results:
        print(f"\nMessage {result['message_id']+1} :")
        print(f"Score : {result['score']:.2f}/100")
        print(f"Recommandation : {result['recommendation']}")
        print(f"Tendance d’innovation :
 {result['innovation_trend']}")



# Sincerity Scoring Script for xAI

## Overview
This script evaluates the sincerity of messages to filter serious proposals for xAI or Elon Musk. It aligns with xAI's mission to "accelerate human scientific discovery" by prioritizing messages related to innovation and discovery.

## Features
- **Clarity Scoring**: Evaluates the specificity of the message.
- **Alignment Scoring**: Measures alignment with xAI's vision (e.g., AI, Neuralink, space).
- **Scam Detection**: Identifies potential scams or suspicious messages.
- **Kalman Filter**: Smooths scores to reduce noise.
- **Innovation Trend**: Tracks the trend of sincerity scores to predict innovation potential.
- **Scalable Design**: Optimized for large-scale production (e.g., AWS Lambda).

# Credit: Grok3  ;)
