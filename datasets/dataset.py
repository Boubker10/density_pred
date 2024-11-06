import pandas as pd

class Dataset:
    def __init__(self, features_file, target_file):
        self.features_file = features_file
        self.target_file = target_file
        self.data = None
        self.target_data = None

    def load_data(self):
        """Charge les données des fichiers CSV de features et de target et affiche les dates brutes pour inspection."""
        self.data = pd.read_csv(self.features_file)
        self.target_data = pd.read_csv(self.target_file)

        # Afficher les 5 premières valeurs brutes de la colonne `date`
        print("Valeurs brutes de la colonne 'date' dans features :")
        print(self.data['date'].head())
        print("Valeurs brutes de la colonne 'date' dans target :")
        print(self.target_data['date'].head())
        
        return self

    def preprocess(self):
        """
        Prépare les données en effectuant les opérations suivantes :
        - Conversion de la colonne de date en format datetime.
        - Fusion des deux ensembles de données sur la colonne de date.
        - Conversion des colonnes de type object en numériques.
        - Gestion des valeurs manquantes.
        - Suppression des colonnes de type datetime.
        """
        # Convertir la colonne date dans `features` avec le format contenant heure
        self.data['date'] = pd.to_datetime(self.data['date'], format="%d-%b-%y %H:%M:%S", errors='coerce')
        # Extraire uniquement la partie jour
        self.data['date_jour'] = self.data['date'].dt.date
        self.data = self.data.drop(columns=['date'])

        # Convertir la colonne date dans `target` avec le format sans heure
        self.target_data['date'] = pd.to_datetime(self.target_data['date'], format="%m/%d/%Y", errors='coerce')
        self.target_data['date_jour'] = self.target_data['date'].dt.date
        self.target_data = self.target_data.drop(columns=['date'])

        # Vérification de la conversion de la date
        print("Data après conversion de la date (features) avec 'date_jour':")
        print(self.data.head())
        print("Data après conversion de la date (target) avec 'date_jour':")
        print(self.target_data.head())

        # Fusionner les features avec le target sur la colonne `date_jour`
        self.data = pd.merge(self.data, self.target_data, on='date_jour', how='left')
        
        # Supprimer la colonne `date_jour` après la fusion
        self.data = self.data.drop(columns=['date_jour'])

        # Convertir les objets en numériques et gérer les NaN
        for c in self.data.columns:
            if self.data[c].dtype == 'object':
                self.data[c] = pd.to_numeric(self.data[c], errors='coerce')
        
        print("Nombre de valeurs NaN par colonne avant suppression :", self.data.isna().sum())
        self.data = self.data.drop(columns=['depression'])
        for c in self.data.columns:
            self.data[c] = self.data[c].fillna(self.data[c].mean())
        
        print("Dimensions après suppression des NaN :", self.data.shape)
        
        return self

    def get_features_and_target(self, target_column):
        """Sépare les features et la cible."""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return X, y
