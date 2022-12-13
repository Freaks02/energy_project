# energy_project

RAPPORT D’ETUDE – PROJET : ANALYSE DE LA CONSOMMATION D’ENERGIE SUR LE TERRITOIRE FRANÇAIS

Table des matières
Introduction	2
Preprocessing des données	4
Gestion des valeurs manquantes	5
Cas particulier de l’année 2022	8
Ajout de variables d’intérêt : feature engineering	8
Transformation des données « horaires »	8
Visualisation des données	9
Production et consommation française	9
Consommation annuelle par région	9
Production annuelle par régions	11
Répartition de la production d’énergie par type	12
Analyse statistique : recherche de corrélation entre les données	15
Matrice de corrélations	15
Test de Pearson sur les variables quantitatives	16
Recherche de relations entre variables quantitatives et qualitatives : test ANOVA	16
Modèle de prédiction de la consommation	17
Préparation des ensembles d’entrainement et de test pour les modèles de ML	17
Première approche naïve : utilisation de modèle de régression	17
Modèle Prophet adapté aux données type Time.Series	17
Perspectives	20
Conclusion	21
ANNEXE	22
Annexe 1 : Modèle Ridge	22
Annexe 2 : Modèle Lasso	22
Annexe 3 : Modèle ElasticNetCV	23

Introduction

L’énergie est indispensable au développement économique, social et industriel dans tous les pays du monde. Elle fait partie des indicateurs permettant de mesurer les écarts de développement entre les différentes régions et représente l’un des enjeux majeurs actuels du développement durable. 
L’une des problématiques bien connue des acteurs du secteur énergétique est la prédiction de la consommation versus la production afin d’éviter un éventuel blackout.
Ce projet d’analyse de données vise à constater le phasage entre la consommation et la production énergétique au niveau national et au niveau régional.
Pour cela, nous utiliserons la source de données issue de l’ODRE (Open Data Réseaux Energies) qui fournit au pas demi-heure :
La consommation réalisée.
La production selon les différentes filières composant le mix énergétique.
La consommation des pompes dans les Stations de Transfert d'Energie par Pompage (STEP).
Le solde des échanges avec les régions limitrophes.
Ce jeu de données, rafraîchi une fois par jour, présente les données régionales consolidées depuis janvier 2021 et définitives (de janvier 2013 à décembre 2020) issues de l'application éCO2mix. 
Fichier source : lien ici.
Le fichier dispose de 32 variables définissant la consommation par région à l’échelle nationale :
Code Insee Région : il s’agit de l’identification numérique de chaque région basée sur le code Insee
Région : correspond à chacune des régions de France (Normandie, Ile de France, Bretagne etc.)
Nature : il s’agit du type de données, celles-ci peuvent être soit définitives soit consolidées. Les données sont dites consolidées lorsqu'elles ont été vérifiées et complétées (livraison en milieu de M+1). Elles deviennent définitives lorsque tous les partenaires ont transmis et vérifié l'ensemble des comptages, (livraison deuxième trimestre A+1).
Date : sous format année-mois-jour (AAAA-MM-JJ)
Heure : sous format XX : XX. Les données du fichier débutent le 01.01.2013 à 00h00.
Date-heure : valeur définissant le jour et l’heure de relevé des données
Consommation (MW) : valeur chiffrée représentant l’énergie consommée en MW
Thermique (MW) : valeur chiffrée représentant l’énergie produite d’origine thermique en MW
Nucléaire (MW) : valeur chiffrée représentant l’énergie produite d’origine nucléaire en MW
Eolien (MW) : valeur chiffrée représentant l’énergie produite d’origine éolienne en MW
Solaire (MW) : valeur chiffrée représentant l’énergie produite d’origine solaire en MW
Hydraulique (MW) : valeur chiffrée représentant l’énergie produite d’origine hydraulique en MW
Pompage (MW) : correspond aux STEP (stations de transfert d’énergie par pompage). Ce sont des installations hydroélectriques qui puisent aux heures creuses de l'eau dans un bassin inférieur afin de remplir une retenue en amont (lac d'altitude). L'eau est ensuite turbinée aux heures pleines pour produire de l’électricité. Grâce à leur fonction de stockage, ces installations contribuent à maintenir l’équilibre entre production et consommation sur le réseau électrique, tout en limitant les coûts de production lors des pics de consommation. A l’heure actuelle, le transfert d’énergie par pompage hydraulique est la technique la plus mature de stockage stationnaire de l’énergie.
Bioénergies (MW) : valeur chiffrée représentant l’énergie produite issue des energies renouvelables en MW
Echanges physiques (MW) : somme des envois ou des réceptions en MW entre régions
Stockage batterie : représente l’énergie stockée dans des batteries
Déstockage batterie : représente l’énergie déstockée depuis des batteries
Eolien terrestre : représente l’énergie issue de champs d’éoliennes terrestre
Eolien offshore : représente l’énergie issue de champs d’éoliennes offshore
TCO <filière> % : taux de couverture de <filière> en % (production<filière> / production totale)
TCH <filière> % : taux de charge de <filière> en % (volume de production / capacité de production max en service)
Column 30 : colonne sans données

Preprocessing des données

Afin de pouvoir travailler sur les données de manière adéquate, il est indispensable de commencer par nettoyer les données.

Aperçu du dataframe « Energie »
Le jeu de données est constitué de 32 colonnes présentant pour chaque demi-heure la consommation, la production et d’autres variables énergétiques, par régions de France depuis le 01.01.2013 à minuit. 
Il contient : 32 colonnes et 1 927 296 lignes.


Un aperçu rapide des informations du dataframe nous indique que les données sont de type « float64, int64 et object ».


Gestion des valeurs manquantes
La matrice suivante fait apparaitre un nombre important de valeurs manquantes (représenté par les zones blanches dans la matrice). 

Matrice représentant les valeurs manquantes (zone blanche) dans le dataframe
Nous décidons donc de supprimer toutes les colonnes où le taux de valeurs manquantes est supérieur à 75%.
Nous recalculons le taux de valeurs manquantes sur les colonnes restantes (voir tableau ci-dessous), on note que pour certaines variables, nous avons un taux de NaN très bas (0.000006).



En affinant l’analyse, il s’avère que ce sont les 12 premières lignes qui sont vides, nous faisons alors le choix de les supprimer.
Pour les NaN restants, que devons-nous faire ?¶
« nucleaire_mw » : on suppose que ce sont les régions où il n'y a pas de production nucléaire.
« pompage_mw » : idem à la colonne nucléaire
« eolien_mw » : idem au deux autres.
Nous remplaçons donc les NaN par la valeur « zéro ».
Nous recalculons le taux de valeurs manquantes après ce dernier traitement (voir tableau ci-après). 


Nous observons que nous n’avons plus de valeurs manquantes dans notre dataframe. Celui-ci a été nettoyé des NaN.
Cas particulier de l’année 2022
Nous avons décidé de supprimer l’année 2022 car ses données ne sont pas complètes, l’année étant non achevée et les données en cours de compilation.
Ajout de variables d’intérêt : feature engineering
Dans le cadre de notre analyse, nous allons avoir besoin de variables supplémentaires, nous avons donc décidé de creer :
- production totale : somme de l’ensemble des types d’énergies produites
- production énergie green : somme de l’ensemble des énergies renouvelables
Transformation des données « horaires » 
Pour faciliter le travail sur les données, nous avons transformé les données des colonnes « date », « heure » et « date_heure » en type datetime via la fonction to_datetime. 
Par ailleurs, nous avons créé les colonnes « jour », « mois » et « année » pour faciliter le traitement des données par la suite.

Données du dataframe nettoyé
Nous travaillerons donc à partir de ce dataframe nettoyé et préparé selon nos besoins.

Visualisation des données 

Pour la visualisation de nos données, nous avons utilisé Seaborn, Géopandas et Plotly.
Production et consommation française
La France est un pays producteur d’énergie. La majorité de ses besoins sont couverts par sa production propre. Le graphique suivant confirme cette notion : la France produit de l'énergie de manière régulière. Cependant à certaines périodes de l'année, on note que la production n'est pas suffisante pour couvrir l'ensemble des besoins. Dans ces cas-là, la France importe de l'énergie.

Le graphique ci-dessus a été réalisé via « Plotly», il permet d’obtenir les valeurs consommées et produites à chaque instant. 
Consommation annuelle par région
Afin d’affiner l’analyse, nous avons décidé de visualiser la consommation annuelle de chaque région. 

Consommation d’énergie par régions de 2013 à 2021

D’une manière générale, on note que la consommation par région est stable année après année, seul point particulier est l’année 2020 où les régions les plus consommatrices ont subi une légère chute, certainement liée à la crise sanitaire et à la baisse d’activité. 
On peut noter que certaines régions sont plus consommatrices d’énergie, ce qui est le cas pour l’Ile-de-France et l’Auvergne-Rhône-Alpes par ex. 
Ces éléments sont à rapprocher de la démographie et du tissu industriel de chaque zone.


Production annuelle par régions
La production annuelle depuis 2013 par régions présente plus de variations que la consommation.
Nous pouvons noter que certaines régions sont très peu productrices d’énergie alors que d’autres produisent beaucoup. Cela peut être mis en relation avec les infrastructures disponibles sur chacun de ces territoires (centrales nucléaires, centrales hydrauliques, champs d’éoliennes etc.). 

Production d’énergie par régions de 2013 à 2021
Répartition de la production d’énergie par type
Le camembert ci-dessous montre la part des différents types d’énergie produit en France entre 2013 et 2021.

La production d’énergie française est grandement issue de l’énergie nucléaire. En 2ème position arrive l’énergie hydraulique. L’énergie solaire et les bioénergies représentent une part marginale.
Les cartes suivantes permettent d’observer les régions productrices d’énergie nucléaire vs. énergies renouvelables. 


Qu’en est-il de l’évolution de la production au cours des années entre énergie nucléaire et énergies renouvelables ?
Le graphique ci-dessous montre que la production d’énergie renouvelable tend à augmenter, alors que celle du nucléaire est à la baisse (notons que l’année 2020 reste « exceptionnelle » compte tenu de la crise Covid). Cela est lié aux politiques gouvernementales visant à réduire la part du nucléaire dans le portefeuille énergétique français au profit des énergies renouvelables. 

Evolution de la production française d’énergie nucléaire et d’énergie renouvelable de 2013 à 2021
Ainsi, via les différentes visualisations des données fournies, nous avons pu faire ressortir certains éléments :
- la France a une capacité de production d’énergie relativement stable depuis 2013
- la production et la consommation par régions ne sont pas linéaire : certaines régions produisent plus que d’autres tandis que d’autres régions consomment plus que leur production
- le nucléaire est la source principale de production d’énergie en France, bien que celle-ci tend à diminuer au profit des énergies renouvelables

Analyse statistique : recherche de corrélation entre les données

Pour aller plus loin sur l’analyse, nous allons effectuer une recherche de corrélations entre nos différentes variables.
Matrice de corrélations
Nous recherchons ici quelle(s) variable(s) sont le plus corrélées entre elles. La matrice de corrélation ci-dessous nous permet de confirmer nos premières constatations.

Matrice de corrélation
D’après cette figure, nous pouvons noter que la variable « nucléaire » et « total_prod » sont fortement corrélées (coeff. 0.97). Cela confirme nos premières observations quant à l’importance du nucléaire dans la part de la production énergétique totale.
Nous pouvons observer par ailleurs que la variable « hydraulique » est fortement corrélée à la variable « prod_energie_green » (coeff. 0.81), ce qui confirme également nos premières analyses quant à l’importance de l’énergie hydraulique dans la part des énergies renouvelables. 
Test de Pearson sur les variables quantitatives
Nous avons réalisé un test de Pearson afin d'observer si les variables quantitatives sont corrélées entre elles. 
Le résultat ci-dessous est le résultat du test de Pearson entre les variables « total_prod » et « nucléaire ».

Ici, on note que le coeff. de Pearson est proche de 1, donc les variables 'total_prod' et 'nucleaire' sont fortement corrélées, ce qui confirme les résultats précédents sachant que la majorité de la production énergétique française provient du nucléaire.
Grace à la matrice de corrélation des coefficients de Pearson appliquée à l’ensemble du jeu de données, nous avons pu observer que toutes les natures d’énergie sont corrélées avec la production avec plus ou moins d'importance. On note que l’énergie hydraulique est celle qui présente le plus de corrélation avec la production d’énergie renouvelable.

Les tests statistiques confirment bien nos observations antérieures. 
Recherche de relations entre variables quantitatives et qualitatives : test ANOVA 
Nous avons effectué un test ANOVA pour observer la relation entre variables quantitatives (consommation et production énergétiques) et variables qualitatives (région, mois, année).

Nous cherchons à déterminer s’il y a une influence des régions, des mois ou des années sur les types d'énergies produites.

En observant les p-valeurs (<0.05), on a pu conclure que la région de production, le mois et l'année ont des effets significatifs sur les types d'énergie produits : ces 3 variables ont un effet statistique significatif sur la consommation.

Modèle de prédiction de la consommation

Notre objectif est de prédire la consommation énergétique (variable cible quantitative), nous recherchons donc à générer un modèle d’apprentissage supervisé pour une variable quantitative. 
Préparation des ensembles d’entrainement et de test pour les modèles de ML
Dans un premier temps, nous avons éliminé les variables non intéressantes pour le modèle de machine learning et qui pouvaient alourdir le dataset. Nous avons supprimé : « code insee », « region » et « nature ».
Nous avons préparé un dataframe contenant les données par mois afin de pouvoir prédire les consommations mensuelles nationales.
Ce dataframe a été normalisé via StandardScaler car les valeurs pouvaient être très disparates. 
Voyons désormais comment se comporte les modèles présélectionnés compte tenu de nos données ?
Nous avons séparé notre dataset en ensemble d’entrainement et en ensemble de test tel que 20% des données appartiennent à l’ensemble de test. 

Première approche naïve : utilisation de modèle de régression
Dans une première approche, nous avons opté pour un modèle de prédiction type régression avec 3 types de modèle : Ridge, Lasso et ElasticNet.  
Or, il s’est avéré que notre logique était incorrecte car nous avions utilisé la production comme variable explicative. Dans la mesure où la consommation est, de fait, liée à la production, les résultats des modèles (voir annexe) qui approchent les 99% de score nous ont permis de comprendre notre erreur. Il n’est pas pertinent de prédire la consommation en se basant sur la production sachant que la production est elle-même basée sur la prédiction de consommation. 
Nous avons alors retravaillé nos modèles et travaillé sans considérer la variable « consommation ». En ne considérant que les variables « mois », « jour », « année », nous avons pu mettre en pratique un modèle de type Time.Series : le modèle Prophet.
Modèle Prophet adapté aux données type Time.Series
Le modèle Prophet convient à des séries temporelles affectées par des événements ou des saisonnalités liées à l’activité humaine telles que celles dont nous disposons.
Nous avons re-travaillé les données pour qu’elles puissent être utilisées par le modèle :



Nous avons ensuite entrainé le modèle sur notre ensemble d’entrainement prédéfini au préalable. La partie en bleu est le modèle appliqué à l’ensemble d’entrainement et la partie grise est le modèle appliqué à l’ensemble de test.

Résultats du modèle suivant l’ensemble d’entrainement (bleu) et l’ensemble de test (gris)
Nous avons évalué la qualité de la prévision à l’aide des métriques d’évaluation traditionnelles. Nous obtenons :


Le graphique suivant montre le résultat du modèle de prédiction (en orange, valeur « yhat »). Cela confirme que le modèle est robuste.


Graphique montrant les valeurs prédites (orange) versus les valeurs réelles (gris)
Nous réalisons également une prévision pour les données max (en rouge) et min (en vert) : 

Graphique de modélisation intégrant les valeurs max (rouge) et min (vert) de la consommation
Nous appliquons alors ce modèle pour obtenir une prédiction de la consommation à long terme (ici jusqu’en 2023), nous obtenons les résultats suivants :

Modèle de prédiction de la consommation énergétique en France jusqu’en 2023
Nous observons sur ce graphe une prédiction cohérente, qui suit les valeurs de consommation des années précédentes (courbe en bleu). Celle-ci devient toutefois moins précise au cours du temps, avec une plage de valeurs min/max plus étendue (zone en bleu clair).
Ainsi nous avons pu mettre en œuvre un modèle prédictif sur des données de type Time.Series (notion qui n’avait pas été abordé lors des cours) qui nous semble robuste.
Perspectives 
La prédiction de la consommation énergétique est un enjeu majeur pour les acteurs de la filière énergétique. Il est aisé de comprendre les intérêts d’avoir des modèles prédictifs de qualité. 
Une prédiction au plus près de la réalité permet de limiter les pertes, de produire les quantités nécessaires de manière à ce qu’elles soient utilisées de manière efficiente et de diminuer le recours à l’import énergétique. 
Le modèle que nous avons défini pour la prédiction de la consommation nous a permis d’obtenir des résultats satisfaisants, proches de la réalité. 
Un autre modèle est souvent utilisé lorsqu’il s’agit de la prédiction de consommation énergétique : le modèle ARIMA. C’est l’un des modèles le plus connus de la prévision des consommations énergétiques à partir de séries chronologiques. 

Toutefois, ayant eu dans une premier temps une approche « naïve » de nos données, nous n’avons pas pu travailler plus en profondeur sur la prédiction. En effet, une autre possibilité aurait été d’agréger à nos données initiales, des données météorologiques afin de déterminer l’impact de la météo sur la consommation d’énergie. 
Une autre option aurait été de déterminer l’impact saisonnier sur la consommation en fonction des régions. 
Enfin, la taille astronomique du jeu de données (une valeur toutes les 30 min depuis le 1er janvier 2013) a entrainé des difficultés lors de la création de la web app sur Streamlit. Nous avons décidé de diminuer le dataset afin que le modèle puisse tourner plus rapidement sur Streamlit, toutefois, c’est une contrainte qu’il faudrait retravailler par la suite en agrégant les données par jour voire semaine. 
Malheureusement, nous n’avons pas pu aller plus loin à la suite de circonstances particulières, qui ont limité le temps de travail disponible sur ce projet. 
Conclusion
Compte tenu de l’importance pour un pays de la filière énergétique, de nombreux intervenants travaillent sur la prédiction de la consommation énergétique à court, moyen et long terme.
La plupart des études se concentre sur la prévision de la consommation en introduisant d’autres paramètres et en utilisant différents modèles prédictifs selon la nature des données d’entrées et les objectifs visés, ce que nous n’avons pas pu réaliser par manque de temps. 
Toutefois, ce projet nous a permis de mettre en pratique les acquis théoriques dispensés lors de la formation de Data Analyst. Nous avons pu observer que de nombreuses applications étaient possibles dans le cadre de l’analyse de données, et qu’il était possible d’atteindre une compréhension fine de phénomènes particuliers et de définir des modèles prédictifs robustes grâce aux divers modèles de machine learning existants. 
Nous avons également intégré l’importance de sélectionner les données existantes et de les corréler avec les résultats souhaités : une approche naïve de la situation implique des prédictions erronées et une perte de temps non négligeable. Il est essentiel de bien définir les objectifs de l’analyse des données ainsi que les résultats visés. 

ANNEXE

Annexe 1 : Modèle Ridge 

Pour pouvoir prédire la consommation nationale à partir du modèle Ridge, nous avons dans un premier temps utilisé RidgeCV pour déterminer un alpha optimum parmi un ensemble d’alpha prédéterminés : 

Le alpha sélectionné par le modèle est alpha = 0.001.
En faisant travailler le modèle ainsi créé sur les données, nous obtenons les scores suivants :

Les scores obtenus sont très corrects. De même l’erreur quadratique moyenne de prédiction sur l’ensemble d’entrainement et l’ensemble de test sont proches mais nous pouvons nous demander si un autre modèle ne serait pas plus performant.

Annexe 2 : Modèle Lasso

Pour le modèle de Lasso, nous avons travaillé de la même manière que pour le modèle de Ridge. 
Nous avons testé plusieurs alphas pour voir comment le modèle se comportait puis à l’aide de LassoCV, nous avons déterminé le alpha optimum.

Nous avons par la suite souhaité savoir quelles variables le modèle allaient utiliser pour prédire la consommation. Pour cela, nous avons tracé le graphique suivant :

On observe que les coefficients des variables « pompage » et « bioénergies » sont quasiment nuls, cela signifie que ces variables n’ont pas dû être sélectionnées par le modèle.
Une fois le modèle entrainé, nous l’avons testé pour déterminer la prédiction de consommation énergétique nationale. Les résultats obtenus sont les suivants : 

Annexe 3 : Modèle ElasticNetCV

Nous avons créé un model nommé model_en via ElasticNetCV.

Nous avons fixé le L1_ratio ainsi que les alphas que pouvaient prendre le modèle. 
Ensuite, nous avons tracé, pour chaque valeur de L1_ratio, la courbe représentant la moyenne des erreurs MSE obtenues par validation croisée en fonction des valeurs de alpha. 

Les résultats d’efficience du modèle ElasticNetCv sont les suivants : 


Les résultats des 3 modèles approchent les 100%, ces scores étant très élevés, nous avons pu déterminer notre erreur et avons par la suite retravaillé sur un modèle plus pertinent. 

