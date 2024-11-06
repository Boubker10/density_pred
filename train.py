from datasets.dataset import Dataset
from helper_logger import logger
import joblib
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import config
import pandas as pd
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, X, y, model_path, model_name):
        self.X = X
        self.y = y
        self.model_path = model_path
        self.model_name = model_name
        self.model = xgb.XGBRegressor(objective='reg:squarederror')
        self.results = {}

    def train(self):
        # Paramètres pour GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        # Initialisation de KFold avec 5 replis
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_metrics = []
        best_params_list = []

        # Validation croisée avec KFold
        for fold, (train_index, test_index) in enumerate(kf.split(self.X)):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            # Recherche des meilleurs hyperparamètres avec GridSearchCV
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params_list.append(grid_search.best_params_)

            # Prédictions et calcul des métriques pour le repli courant
            y_pred = best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            fold_metrics.append({"Fold": fold + 1, "RMSE": rmse, "R2": r2})

            # Tracer les résidus pour ce repli
            self.plot_residuals(y_test, y_pred, fold + 1)

        # Calculer la moyenne des métriques sur tous les replis
        avg_rmse = np.mean([m["RMSE"] for m in fold_metrics])
        avg_r2 = np.mean([m["R2"] for m in fold_metrics])
        best_params_avg = {key: np.mean([d[key] for d in best_params_list]) for key in best_params_list[0]}

        # Enregistrer les résultats moyens
        self.results = {
            "Model": self.model_name,
            "Average RMSE": avg_rmse,
            "Average R2": avg_r2,
            "Best Parameters": best_params_avg
        }
        
        # Afficher les métriques moyennes
        logger.info(f"{self.model_name} - Moyenne RMSE: {avg_rmse}, Moyenne R2: {avg_r2}")
        logger.info(f"{self.model_name} - Best Parameters (moyenne sur les replis): {best_params_avg}")
        
        # Sauvegarder le modèle final avec les meilleurs paramètres
        self.save_model(best_model)

    def save_model(self, model):
        joblib.dump(model, self.model_path)
        logger.info(f"Modèle sauvegardé dans {self.model_path}")
    
    def plot_residuals(self, y_test, y_pred, fold):
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red', linestyles='dashed')
        plt.xlabel("Valeurs prédites")
        plt.ylabel("Résidus")
        plt.title(f"Graphique des Résidus pour {self.model_name} - Fold {fold}")
        plt.savefig(f"{self.model_name}_residuals_fold_{fold}.png") 
        plt.close()

if __name__ == "__main__":
    results = []
    dataset_K = Dataset(config.FEATURES_FILE_K, config.TARGET_FILE_K)
    dataset_K.load_data().preprocess()
    X_K, y_K = dataset_K.get_features_and_target(target_column=config.TARGET_COLUMN)
    trainer_K = ModelTrainer(X_K, y_K, model_path=config.MODEL_PATH_K, model_name="Modèle K")
    trainer_K.train()
    results.append(trainer_K.results)

    dataset_L = Dataset(config.FEATURES_FILE_L, config.TARGET_FILE_L)
    dataset_L.load_data().preprocess()
    X_L, y_L = dataset_L.get_features_and_target(target_column=config.TARGET_COLUMN)
    trainer_L = ModelTrainer(X_L, y_L, model_path=config.MODEL_PATH_L, model_name="Modèle L")
    trainer_L.train()
    results.append(trainer_L.results)

    dataset_J = Dataset(config.FEATURES_FILE_J, config.TARGET_FILE_J)
    dataset_J.load_data().preprocess()
    X_J, y_J = dataset_J.get_features_and_target(target_column=config.TARGET_COLUMN)
    trainer_J = ModelTrainer(X_J, y_J, model_path=config.MODEL_PATH_J, model_name="Modèle J")
    trainer_J.train()
    results.append(trainer_J.results)

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)
    logger.info("Les résultats des modèles ont été enregistrés dans model_results.csv")
