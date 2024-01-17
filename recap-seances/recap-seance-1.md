# Séance 1

## Rendus

16/02 date de la soutenance
13/02 mardi (18h) rendu du rapport

## Tips

Ca peut être bien de se faire nos présentations à tous les groupes le 15 pour avoir les dernières news pour l'oral individuel.

## Séances

5 : 15/02 à 9h répétition de soutenance

4 : 09/02 à 14h point sur le rapport et derniers avancements

3 : 01/02 à 8h30

2 : 25/01 8h30

1 : 16/01 à 8h30

## Méthodologie

Préparer une liste de questions avant les séances (48h avant c'est bien). Si jamais on est bloqué on peut envoyer un mail, mais on concentre les échanges sur ces séances.

Organisation interne du groupe souple, bien communiquer quand on se sera découpé le travail.

## Technique : résumé des articles

## Rapport biblio

Donner une perspective (se projeter sur la fin et expliquer ce qu'on compte mettre en place sur la base de quel article).

Dans le rapport, bien couvrir tous les champs du projets (ici, pas encore de couverture de la partie contrôle, et signal/descripteurs)


1. Acoustique
   1. Résonateur
    
    - maganza, gibia et schumacher (guides d'ondes, ondes aller-retour et partie réflexion)
    - Missoum, Vergez (approche différente, décomposition modale)

    -> modélisation de l'instrument (cordes & vents)
    -> essayer les 2 approches pour les simulations numériques

    **Questions** : comment passe-t-on de l'impédance d'entrée à la fonction de réflexion (à l'entrée de l'instrument) -> passage à coef de réflexion à la sortie de l'instrument.
    On écoute la pression en sortie et pas l'entrée

    **Réponses** :  FpA pour passer de Ze à Re et inversement (cf notes Paul)
    cf photo pour retrouver le coefficient de réflexion pour un tuyau d'impédance de rayonnement $Z_R = \varepsilon Z_c$.
    Pression en sortie : matrice de transfert

   2. Excitateurs : comment les modéliser.
   Différents excitateurs possibles :
      1. Anche simple
      2. Anche double
      3. Loi de frottement

2. Simulation sur python/matlab et temps réel (s'y mettre rapidement)
   1. Implémenter notre solveur
   2. Comparer la partie qui tourne en temps réel et le code python/matlab pas temps réel :
   Simulation sur python matlab (le séquencer), quand ça marche bien passer sur l'environnement temps réel (max, pure data, faust...). Faire un export avec python/matlab et avec l'environnement temps réel pour comparaison.

3. Partie traitement du signal/descripteurs
    A approfondir

1. Mentionner dans la biblio le contrôle (piloter, sur quelle base on peut s'appuyer (cartographies))
