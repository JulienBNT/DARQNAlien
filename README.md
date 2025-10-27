# DARQN avec CBAM pour Atari Alien

<div align="center">
  <img src="images/alien.jpg" alt="Alien"/>
</div>

## Images du jeu

<div align="center">
  <img src="images/alien-atari.jpg" alt="Alien Atari"/>
</div>

## Architecture du projet

```
DQN/Alien/
├── DARQNALIEN.ipynb          # Notebook principal avec l'implémentation complète
├── DARQNALIENnn.ipynb         # Variante du modèle
├── README.md                  # Documentation du projet
├── logs/                      # Logs TensorBoard des entraînements
└── models/                    # Modèles sauvegardés (.keras)
    ├── best_darqn_cbam_alien.keras
    └── darqn_cbam_alien_ep*.keras
```

## Architecture du code

### 1. Configuration et imports

```python
ENV_NAME = "ALE/Alien-v5"
LEARNING_RATE = 0.00025
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPISODES = 500
UPDATE_TARGET_FREQ = 10
FRAME_STACK = 4
LSTM_UNITS = 256
```

### 2. Module CBAM (Convolutional Block Attention Module)

Le module CBAM applique un mécanisme d'attention sur deux dimensions:

- **Channel Attention**: Identifie les canaux de features les plus importants
  - Global Average Pooling + Global Max Pooling
  - MLP partagé (réduction ratio 8:1)
  - Activation sigmoid
  
- **Spatial Attention**: Identifie les régions spatiales importantes
  - Pooling spatial (avg + max)
  - Convolution 7x7
  - Activation sigmoid

```python
class CBAM(Layer):
    - ratio: Facteur de réduction pour le MLP (défaut: 8)
    - kernel_size: Taille du noyau pour l'attention spatiale (défaut: 7)
```

### 3. Prétraitement des frames

```python
def preprocess_frame(frame):
    - Conversion RGB → Grayscale
    - Redimensionnement 210×160 → 84×84
    - Normalisation [0, 255] → [0, 1]
```

### 4. Architecture DARQN

```
Input: (84, 84, 4)  # Stack de 4 frames

├── Conv2D(32, 8×8, stride=4) + ReLU
├── CBAM(ratio=8, kernel=7)
│
├── Conv2D(64, 4×4, stride=2) + ReLU
├── CBAM(ratio=8, kernel=7)
│
├── Conv2D(64, 3×3, stride=1) + ReLU
├── CBAM(ratio=8, kernel=7)
│
├── Flatten
├── Reshape → (4, features)
│
├── LSTM(256) + Tanh
├── Dropout(0.3)
│
├── Dense(512) + ReLU
├── Dropout(0.2)
│
└── Dense(n_actions) + Linear

Output: Q-values pour chaque action
```

**Caractéristiques**:
- **Attention visuelle**: CBAM après chaque couche convolutionnelle
- **Mémoire temporelle**: LSTM pour capturer les séquences d'états
- **Régularisation**: Dropout pour éviter le surapprentissage
- **Fonction de perte**: Huber loss (robuste aux outliers)

### 5. Composants de l'agent

#### ReplayBuffer
```python
class ReplayBuffer:
    - Capacité: 10,000 transitions
    - Stockage: (state, action, reward, next_state, done)
    - Échantillonnage: Batch aléatoire de 32 transitions
```

#### DARQNAgent
```python
class DARQNAgent:
    - model: Réseau Q principal
    - target_model: Réseau Q cible (stabilisation)
    - replay_buffer: Mémoire des expériences
    - epsilon: Taux d'exploration (1.0 → 0.1)
    - frame_stack: Pile de 4 frames consécutifs
```

**Méthodes principales**:
- `get_action()`: Politique epsilon-greedy
- `train()`: Entraînement sur un batch
- `update_target_model()`: Synchronisation des poids
- `stack_frames()`: Empilement des observations

### 6. Algorithme d'entraînement

```
Pour chaque épisode:
    1. Reset environnement
    2. Initialiser stack de frames
    
    Tant que non terminé:
        a. Sélectionner action (epsilon-greedy)
        b. Exécuter action dans l'environnement
        c. Observer (reward, next_state, done)
        d. Stocker transition dans replay buffer
        e. Échantillonner batch et entraîner
        
    3. Mettre à jour target network (tous les 10 épisodes)
    4. Décroisser epsilon
    5. Sauvegarder si meilleur score
```

**Équation de Bellman utilisée**:

Q(s, a) = r + γ × max Q'(s', a')

Où:
- Q: Réseau principal
- Q': Réseau cible
- γ (gamma): 0.99
- r: Récompense immédiate

### 7. Entraînement et évaluation

#### Fonction d'entraînement
```python
train_darqn():
    - Durée: 500 épisodes
    - Logging: TensorBoard
    - Sauvegarde: Meilleur modèle + checkpoints (tous les 50 épisodes)
    - Affichage: Métriques tous les 10 épisodes
```

#### Visualisation
```python
plot_training_results():
    - Graphique des récompenses par épisode
    - Graphique de la loss par épisode
    - Moyennes mobiles (fenêtre de 10)
    - Statistiques finales
```

#### Test de l'agent
```python
test_agent():
    - Chargement du meilleur modèle
    - Epsilon = 0 (exploitation pure)
    - Mode render optionnel
    - Statistiques sur n épisodes
```

## Utilisation

### Entraînement

Exécuter les cellules dans l'ordre:
1. Imports et configuration
2. Définition du CBAM et architecture
3. Création de l'environnement et de l'agent
4. Lancement de l'entraînement
5. Visualisation des résultats

### Test

Décommenter la dernière ligne de la dernière cellule:
```python
test_rewards = test_agent(render=False)
```

Pour visualiser l'agent en action, utiliser `render=True`.

## Performances attendues

Le modèle devrait montrer:
- Amélioration progressive des récompenses
- Convergence de la loss
- Comportements stratégiques émergents
- Meilleure performance que DQN standard grâce à l'attention CBAM

## Dépendances

```
tensorflow >= 2.x
gymnasium
ale-py
numpy
matplotlib
```

## Références

- **CBAM**: Woo et al. (2018) - "CBAM: Convolutional Block Attention Module"
- **DRQN**: Hausknecht & Stone (2015) - "Deep Recurrent Q-Learning for Partially Observable MDPs"
- **DQN**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
