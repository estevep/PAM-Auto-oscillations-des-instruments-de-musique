# Séance 2

## Répartition du travail

- descripteurs Pharoah et Benjamin
- acoustique coeff réflexion Paul (implem), Hugo
  - Regarder Julius Smith pour guide d'onde
  - Commencer par clarinette & analogie avec le violon
  - Complexifier avec saxo pour géométrie non cylindrique
- Acoustique Van der Pol Charlotte et Victor

## Questions pré-séance

Dans les modèles qu'on a vu avec approche modale, simplification BF avec 1 seul mode:
Pertinence d'ajouter des modes dans la modélisation ? Si oui, comment ? Que devient l'équation ?

- Mise en équation et résolution de l'approche modale avec plusieurs modes ?
- Approche par guide d'onde pour géométrie non cylindrique ?
  - Kelley Lochbaum avec changements de sections : successions de cylindres peut permettre de modéliser ?

- Phénomènes dissipatifs ?
  - Approche modale avec amortissement mais pour l'approche avec guide d'onde on n'en a pas

### Questions et éléments de réponses

Après avoir discuté entre nous, nous avons identifié quelques questions que nous avons encore et dont nous pourrons peut-être discuter ensemble. Ces questions sont aussi destinées à nous-mêmes et allons regarder dans la bibliographie d'ici jeudi si nous trouvons des éléments de réponse.

- Les modèles par approche modale que nous avons étudiés utilisent une simplification basse fréquence et ne gardent qu'un seul mode. Est-il pertinent d'en rajouter ? Si oui comment ? Que devient alors l'équation de type oscillateur de Van der Pol ?
   Contribution des modes c'est bon on a répondu mais nouvelle question : amplitudes modales.

   Idée : considérer qu'on débouche dans un tube idéal (la pièce) ou résistance acoustique

   +- selon si on considère
   Réflexion en dirac avec coeff de réflexion qui vaut +- 1 delta(t-T), si on a un coeff eps tq |eps| < 1, 0 on change pas de section.

   Plutôt réflective (tube de sortie plus petit) si on a un coeff <0

   Pour le saxo il faut travailler sur les ondes sphériques pour l'équation des ondes, retrouver les ondes aller-retour (changement de variable à effectuer).

   Décomposition modale liée à fonction de réflexion (prendre $\delta$ * coeff et trouver la décomposition modale associée)

   La réflexion n'est pas un dirac -> dans le domaine de fourier on n'a plus un gain, ça a une incidence sur les fréquences des modes.

   Décomposition modale bien pratique car si on veut copier un instrument, on prend une mesure, prélever les modes à partir de l'impédance d'entrée (fréquences et facteurs de qualités). Utiliser 

   Partir d'équations simples de la physique (retrouver quart d'onde amortie...), voir ce que ça nous donne et s'en inspirer.

**Implémentation**
Maganza indispensable (rapide à faire et enrichissant).

Openwind peut permettre de calculer les impédances d'entrées pour des géométries qui nous intéressent, en dériver les fréquences et facteurs de qualité, puis voir ce que ça nous donne. Sinon on choisit nous-mêmes les paramètres modaux.

Article (Benjamin connait), pour dériver les modes.

On peut innover en terme de géométrie

- Le modèle par approche modale prend en compte tous les amortissements dans les facteurs de qualité, mais l'approche par guide d'onde ne semble pas en tenir compte. Peut-on modifier le modèle pour les incorporer ?
- Le modèle par guide d'onde semble simple à implémenter avec les simplifications faites par Maganza et al. par exemple, mais comment l'adapter à un instrument à géométrie non cylindrique ? Peut-on utiliser une approche comme celle de Kelly-Lochbaum pour le conduit vocal en découpant le résonateur en sections cylindriques ?

Maganza : tout mécanisme de pertes est constant en fréquences

Terme en racine carrée de la fréquence qui arrivent qui engendrent une complexité fréquentielle

Trajectoires de construites rapidement pour donner des paramètres et 

Passer en temps réel

Multiphoniques : faut placer les modes bien.

### Rapport biblio

Table des matières : inclure intro et conclusion

Intro : détailler un peu plus l'objectif du sujet (on dit synthèse par modèle physique mais pas de prise de recul avec ce qu'il faut faire), décomposer les tâches
Mentionner le temps réel, discuter ce qu'est un instrument virtuel (et en conséquent parler du contrôle : ça doit être un instrument jouable virtuellement).

non linéraire : pas de trait d'union
non-linéarité : trait d'union (substantif)

#### Typo

une paramètre*
décideur au lieu de descripteur
"collé-glissé" -> adhérence ... (stick-flip)

#### Octave du dessus vs du dessous 

Eclaircir la citation de Kergomard.

#### En haut de page 6

Temps réel : il peut y avoir des schémas numériques un peu sérieux qui peuvent être résolus en temps réel

#### Thomas Hélie

Faites gaffe à respecter les consignes (y compris pour le rapport de stage).

Eq 1 : typo sur la formule de la décomposition modale (manque des carrés)

Quand on met une équation centrée, elle fait quand même partie de la phrase.

On commence rarement une phrase par un numéro (que ce soit d'équation, bibliographie, figure...).

Article : Résumé et intro on peut garder. Ce sera un article sur notre travail, donc il y aura pas mal de références mais on ne gardera pas de biblio sous cette forme.


## Questions :

Pour la méthode de Maganza et al. on a compris comment faire pour la partie régime périodique, mais c'est pas encore clair pour le transitoire :

besoin de la fonction G de la non-linéarité tq p+ = G(p-), mais on a que p= F(u)

