# Récapitulatif séance 3

## Descripteurs

Pour détecter le pitch, et le timbre on peut utiliser une simple FFT, ESPRIT apporte pas grand chose car on a un instrument auto-oscillant, pas de décroissance
On peut aussi regarder les pics de l'auto-corrélation (non commensurables)

Descripteurs : regarder du côté de la perception, exemple :

- rugosité

## Approche modale

Pour l'approche modale : essayer avec plusieurs modes (harmoniques ou légèrement inharmonique et jouer avec les fréquences pour voir si on peut), ça doit a priori permettre de trouver des régimes différents (apériodiques, chaos...)

On peut aussi essayer de s'approcher d'un son de saxo en plaçant habilement les modes (remarque : on peut utiliser [Openwind](https://openwind.inria.fr) cf [séance 2](recap-seance-2.md))

## Approche guide d'ondes

Lire l'article de Taillard et Kergomard (2010) pour comprendre la méthode itérative (trouver $g$ tq $p^+(t) = g(p^-(t))$). La réponse précise est dans l'annexe A.1.

## Rapport

Pour la prochaine fois : avoir un plan du rapport (format article)

**Conclusion** : où sont les contributions, quels ont été les résultats obtenus

**Différence intro conclusion**, exemple : intro en maths on veut résoudre une équation du 2nd degré, intro = historique (on sait faire 1er degré...) mais pas encore de solution pour toutes les équations du 2nd degré. Conclusion : on a utilisé le déterminant... pour résoudre les 2nd degrés.

Dans l'introduction, il y a une annonce du plan et de ce sur quoi on va travailler, mais c'est surtout dans la conclusion où l'on récapitule les objets utilisés et les résultats. Il faut y ajouter ce qu'on ne savait pas a priori.

Prendre exemple sur les articles d'Acta Acustica par exemple pour le format.

