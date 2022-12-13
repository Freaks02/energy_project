import streamlit as st 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from streamlit_option_menu import option_menu
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.api import anova_lm
from prophet import Prophet
from prophet.plot import plot_plotly

rad=st.sidebar.radio("Menu", ["ACCUEIL", "PREPROCESSING DES DONNEES","ANALYSE DES DONNEES", "PREDICTION DE LA CONSOMMATION ENERGETIQUE", "CONCLUSIONS & PERSPECTIVES"])

if rad=="ACCUEIL":
    st.markdown("<h1 style='text-align: left; color: black;'>ANALYSE DE LA CONSOMMATION D’ENERGIE SUR LE TERRITOIRE FRANÇAIS</h1>", unsafe_allow_html=True)

    st.image("energie.jpg")

    st.markdown('L’énergie est indispensable au développement économique, social et industriel dans tous les pays du monde. Elle fait partie des indicateurs permettant de mesurer les écarts de développement entre les différentes régions et représente l’un des enjeux majeurs actuels du développement durable. L’une des problématiques bien connue des acteurs du secteur énergétique est la prédiction de la consommation versus la production afin d’éviter un éventuel blackout. **Ce projet d’analyse de données vise à constater le phasage entre la consommation et la production énergétique au niveau national et au niveau régional.**')
        
if rad=="PREPROCESSING DES DONNEES":
    st.title("PREPROCESSING DES DONNEES")
    st.write('Les données utilisées sont issues de l’ODRE (Open Data Réseaux Energies) qui fournit au pas demi-heure : la consommation réalisée, la production selon les différentes filières composant le mix énergétique, la consommation des pompes dans les Stations de Transfert d energie, le solde des échanges avec les régions limitrophes.')
    st.write('Ce jeu de données, rafraîchi une fois par jour, présente les données régionales consolidées depuis janvier 2021 et définitives (de janvier 2013 à décembre 2020) issues de l application éCO2mix.')

    df=pd.read_csv('eco2mix-regional-cons-def.csv', sep=';')
    st.write(df.head())

    st.write("dimensions du dataset :", df.shape)

    st.subheader("1. Gestion des valeurs manquantes")
    st.write("La matrice suivante présente le taux de valeurs NaN présents dans le dataset (zones blanches). Nous faisons le choix d'éliminer toutes les colonnes qui presentent un taux superieur à 75%.")

    fig, ax = plt.subplots()
    sns.heatmap(df.isna(), cbar=False, ax=ax)
    st.write(fig)
   

    df_clean=pd.read_csv('df_clean.csv')

    (df_clean.isna().sum()/df_clean.shape[0]).sort_values(ascending=False)
    df_clean = df_clean.drop(axis=0, labels=[0,1,2,3,4,5,6,7,8,9,10,11])


    st.write('En affinant l’analyse, il s’avère que les 12 premières lignes sont vides, nous faisons alors le choix de les supprimer. Enfin, pour les NaN restants, nous decidons de les remplacer par des valeurs nulles.')
    st.write('**Désormais, notre jeu de données ne contient plus aucune valeur manquante.**')

    df_clean['nucleaire_mw'] = df_clean['nucleaire_mw'].fillna(0)
    df_clean['pompage_mw'] = df_clean['pompage_mw'].fillna(0)
    df_clean['eolien_mw'] = df_clean['eolien_mw'].fillna(0)

# calcul de valeurs manquantes ici
    st.write((df_clean.isna().sum()/df_clean.shape[0]).sort_values(ascending=False))

    st.subheader("2. Cas particulier : année 2022")
    st.write('Nous avons décidé de supprimer l’année 2022 car ses données ne sont pas complètes, l’année étant non achevée et les données en cours de compilation.')

    st.subheader("3. Feature engineering")
    st.write('Dans le cadre de notre analyse, nous allons avoir besoin de variables supplémentaires, nous avons donc décidé de créer :')
    st.write('* production totale : somme de l’ensemble des types d’énergies produites')
    st.write('* production énergie green : somme de l’ensemble des énergies renouvelables')

    df_clean['total_prod'] = (df_clean['thermique_mw']
                         + df_clean['nucleaire_mw']
                         + df_clean['eolien_mw']
                         + df_clean['solaire_mw']
                         + df_clean['hydraulique_mw']
                         + df_clean['bioenergies_mw'])

    df_clean['prod_energie_green'] = (df_clean['thermique_mw']
                               + df_clean['eolien_mw']
                               + df_clean['solaire_mw']
                               + df_clean['hydraulique_mw']
                               + df_clean['bioenergies_mw'])


    st.subheader("4. Transformation des données horaires")
    st.write('Pour faciliter le travail sur les données, nous avons transformé les données des colonnes « date », « heure » et « date_heure » en type datetime via la fonction to_datetime.')
    st.write('Par ailleurs, nous avons créé les colonnes « jour », « mois » et « année » pour faciliter le traitement des données par la suite.')

    df_clean['date'] =  pd.to_datetime(df_clean['date'])
    df_clean['heure'] =  pd.to_datetime(df_clean['heure'])
    df_clean['date_heure'] =  pd.to_datetime(df_clean['date_heure'], utc=True)

    st.write(df_clean.head())
    st.write("Nous disposons desormais d'un jeu de données propre nous permettant de réaliser l'ensemble des tests et modèle predictif souhaités.")

if rad=="ANALYSE DES DONNEES":
    st.title("ANALYSE DES DONNEES")
    st.header("I. VISUALISATION GRAPHIQUE DES DONNEES")
    st.subheader("1. Production et consommation française")
    st.write('La France est un pays producteur d’énergie. La majorité de ses besoins sont couverts par sa production propre. Le graphique suivant confirme cette notion : la France produit de l énergie de manière régulière. Cependant à certaines périodes de l année, on note que la production n est pas suffisante pour couvrir l ensemble des besoins. Dans ces cas-là, la France importe de l énergie.')
    st.image('image conso_prod FR.png')

    st.subheader("2. Production par régions")
    st.image('image prod regionale.png', width=600)

    st.subheader("3. Consommation par régions")
    st.image('image conso regionale.png', width=600)
        
    

    st.write('Nous pouvons noter que certaines régions sont très peu productrices d’énergie alors que d’autres produisent beaucoup. Cela peut être mis en relation avec les infrastructures disponibles sur chacun de ces territoires (centrales nucléaires, centrales hydrauliques, champs d’éoliennes etc.)')

    st.subheader("4.Répartition de la production d'énergie par type")
    st.write('La production d’énergie française est grandement issue de l’énergie nucléaire. En 2ème position arrive l’énergie hydraulique. L’énergie solaire et les bioénergies représentent une part marginale.')
    st.image('camembert type energie.png')
    st.write('Les éléments qui ressortent de ces ananlyses sont :')
    st.write('* La France a une capacité de production d’énergie relativement stable depuis 2013.')
    st.write('* La production et la consommation par régions ne sont pas linéaire : certaines régions produisent plus que d’autres tandis que d’autres régions consomment plus que leur production.')
    st.write('* Le nucléaire est la source principale de production d’énergie en France, bien que celle-ci tend à diminuer au profit des énergies renouvelables.')

    st.header("II. STATISTIQUES EXPLORATOIRES DES DONNEES")
    st.write("Nous recherchons ici quelle(s) variable(s) sont le plus corrélées entre elles à l'aide de tests statistiques et de matrice de correlation.")
    st.subheader("1. Matrice de corrélations")

    data = pd.read_csv('df_clean.csv')
    plt.figure(figsize=(18, 18))

    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=False, cmap="RdBu_r", center=0, ax=ax)
    st.write(fig)
   
    
    st.write("Nous pouvons noter que la variable « nucléaire » et « total_prod » sont fortement corrélées (coeff. 0.97). Cela confirme nos premières observations quant à l’importance du nucléaire dans la part de la production énergétique totale. Nous pouvons observer par ailleurs que la variable « hydraulique » est fortement corrélée à la variable « prod_energie_green » (coeff. 0.81), ce qui confirme également nos premières analyses quant à l’importance de l’énergie hydraulique dans la part des énergies renouvelables.") 

    st.subheader("2. Tests statistiques")
    st.write("Les tests statistiques que nous avons réalisé sont :")
    
    st.write(" ***- Le test de Pearson pour déterminer les correlations entre variables qualitatives :***")
    st.write("Nous avons réalisé un test de Pearson afin d'observer si les variables quantitatives sont corrélées entre elles. Le résultat ci-dessous est le résultat du test de Pearson entre les variables « total_prod » et « nucléaire ».")
    st.image("Test PEARSON.png")
    
    st.write("Grace à la matrice de corrélation des coefficients de Pearson appliquée à l’ensemble du jeu de données, nous avons pu observer que toutes les natures d’énergie étaient corrélées avec la production avec plus ou moins d'importance. On note que l’énergie hydraulique est celle qui présente le plus de corrélation avec la production d’énergie renouvelable.")

    
    st.write("***- Le test ANOVA pour déterminer les correlations entre variables qualitatives et quantitatives :***")
    st.image("Test ANOVA.png")
    st.write("En observant les p-valeurs (<0.05), on a pu conclure que la région de production, le mois et l'année ont des effets significatifs sur les types d'énergie produits : ces 3 variables ont un effet statistique significatif sur la consommation.")

if rad=="PREDICTION DE LA CONSOMMATION ENERGETIQUE":
    st.title("PREDICTION DE LA CONSOMMATION ENERGETIQUE")
    st.image("energie 2.jpg")
    st.subheader("1. Préparation des ensembles d'entrainement et de test pour le modèle de Machine Learning")
    st.write("Nous avons éliminé dans un premier temps les données qui n'apportaient rien à l'analyse (ex. code_insee) afin de ne pas alourdir le modèle de prédiction.")
    st.write("Nous avons ensuite préparé un dataframe contenant les données par mois afin de pouvoir prédire les consommations mensuelles nationales. Ce dataframe a été normalisé via StandardScaler.") 
    st.write("Enfin, nous avons séparé notre dataset en ensemble d’entrainement et en ensemble de test tel que approximativement 20% des données appartiennent à l’ensemble de test.") 
    st.text("train = df_prophet[df_prophet['ds'] < '2021-01-01'].copy()")
    st.text("test = df_prophet[df_prophet['ds'] >= '2021-01-01'].copy()")
    
    st.subheader("2. Prédiction de la consomation énergetique via le modèle Prophet adapté aux données Time.Series")
    st.write("Nous avons re-travaillé les données pour qu'elles puissent être intégrées aisément au modèle.")
    
    
    df = pd.read_csv('df_clean.csv')
    dfnew = df[['date_heure','consommation_mw']]
    dfnew = dfnew.loc[dfnew['date_heure'] > '2018-01-01']
    dfnew['date_heure'] = pd.to_datetime(dfnew['date_heure'])
    
    dfnew = dfnew.sort_values(by = 'date_heure')
    dfnew.set_index('date_heure',inplace=True)
    dfnew = dfnew.resample('H').sum()
    dfnew['date_heure'] = dfnew.index
    
    st.write(dfnew.head(5))
    
    st.write("Le modèle a été entrainé et évalué avec des scores satisfaisants, nous pouvons désormais utiliser notre modèle pour prédire la consommation.") 
    st.image("Test Prophet.png")
    
    st.write("**Choisissez le nombre de mois pour lequel vous souhaitez prédire la consommation énergetique française via le curseur ci-dessous.**")
    
    df_prophet = dfnew.copy()
    df_prophet.columns = ['y','ds']
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    train = df_prophet[df_prophet['ds'] < '2021-01-01'].copy()
    test = df_prophet[df_prophet['ds'] >= '2021-01-01'].copy()
    
   
    prophet = Prophet()
    prophet.fit(train)
    prediction = prophet.predict(test)
      
    n_months = st.slider('Nombre de mois de prédiction :', 1, 72)
    
    future = prophet.make_future_dataframe(periods=n_months, freq='M')

    forecast = prophet.predict(future)

    st.write(f'Prédiction de consommations pour {n_months} mois en France')
        
    #Illustrer le résultat des prévisions
    fig =prophet.plot(forecast)
    st.pyplot(fig)
    
    st.write("NB: Compte tenu  de l'importante masse de données générées par éCO2mix, le modèle de prédiciton prend beaucoup de temps avant de sortir un résultat sur Streamlit, c'est pourquoi, exceptionnellement, nous n'avons traité les données qu'à partir de 2018.")
    
if rad=="CONCLUSIONS & PERSPECTIVES":
    st.title("CONCLUSIONS & PERSPECTIVES")
    st.image("energie 3.jpg")
    st.header("PERSPECTIVES")
    st.write("La prédiction de la consommation énergétique est un enjeu majeur pour les acteurs de la filière énergétique. Il est aisé de comprendre les intérêts d’avoir des modèles prédictifs de qualité. **Une prédiction au plus près de la réalité permet de limiter les pertes, de produire les quantités nécessaires de manière à ce qu’elles soient utilisées de manière efficiente et de diminuer le recours à l’import énergétique.**")
    st.write("Pour aller plus loin, il aurait été interessant de modéliser les prédictions grâce à d'autres modèles existants tel que le modèle ARIMA. C’est l’un des modèles les plus connus de la prévision de consommations énergétiques à partir de séries chronologiques. Une autre option aurait été de déterminer l’impact saisonnier sur la consommation en fonction des régions. Malheureusement, nous n’avons pas pu aller plus loin à la suite de circonstances particulières, qui ont limité le temps de travail disponible sur ce projet.") 
    

    st.header("CONCLUSIONS")
    st.write("Compte tenu de l’importance pour un pays de la filière énergétique, de nombreux intervenants travaillent sur la prédiction de la consommation énergétique à court, moyen et long terme. **La plupart des études se concentre sur la prévision de la consommation en introduisant d’autres paramètres et en utilisant différents modèles prédictifs selon la nature des données d’entrées et les objectifs visés, ce que nous n’avons pas pu réaliser par manque de temps.**") 
    st.write("Toutefois, ce projet nous a permis de mettre en pratique les acquis théoriques dispensés lors de la formation de Data Analyst. Nous avons pu observer que de nombreuses applications étaient possibles dans le cadre de l’analyse de données, et qu’il était possible d’atteindre une compréhension fine de phénomènes particuliers et de définir des modèles prédictifs robustes grâce aux divers modèles de machine learning existants.")
    st.write("Nous avons également intégré l’importance de sélectionner les données existantes et de les corréler avec les résultats souhaités : une approche naïve de la situation implique des prédictions erronées et une perte de temps non négligeable. Il est essentiel de bien définir les objectifs de l’analyse des données ainsi que les résultats visés.")

        
st.sidebar.markdown("---")
st.sidebar.markdown("Par Chanel Antoine & Benyamina Khadija")
st.sidebar.markdown(" Formation Bootcamp Data Analyst ")
st.sidebar.markdown("Avril 2022")





