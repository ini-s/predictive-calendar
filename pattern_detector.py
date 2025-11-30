from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd


class PatternDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, features: pd.DataFrame):
        if len(features) > 1:
            self.scaler.fit(features)
            self.is_fitted = True

    def calculate_pattern_score(
        self, current_features: dict, historical_features: pd.DataFrame
    ) -> float:
        if len(historical_features) < 2 or not self.is_fitsmartted:
            return 0.0

        try:
            historical_scaled = self.scaler.transform(historical_features)
            current_df = pd.DataFrame(
                [current_features], columns=historical_features.columns
            )
            current_scaled = self.scaler.transform(current_df)

            clustering = DBSCAN(eps=0.5, min_samples=2).fit(historical_scaled)

            if len(set(clustering.labels_)) == 1 and clustering.labels_[0] != -1:
                cluster_center = historical_scaled.mean(axis=0)
                distance = euclidean_distances([cluster_center], current_scaled)[0][0]
                similarity = max(0, 1 - distance)
                return float(similarity)

            else:
                min_distance = np.min(
                    euclidean_distances(historical_scaled, current_scaled)
                )
                return float(max(0, 1 - min_distance))

        except Exception as e:
            print(f"Error in pattern detection: {e}")
            return 0.0
