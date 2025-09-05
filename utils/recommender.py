import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple
from surprise import dump
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')


class EnhancedHybridRecommender:
    def __init__(self, train_path: str, products_path: str,
                 content_model_path: str, svd_model_path: str):
        self.train_path = train_path
        self.products_path = products_path
        self.content_model_path = content_model_path
        self.svd_model_path = svd_model_path
        
        # Initialize attributes
        self.prod_df = None
        self.prod_embeds = None
        self.svd_model = None
        self.global_avg = 3.0
        self.train_df = None
        self.user_history_cache = {}
        self.product_popularity = {}
        self.product_features = {}
        self.skin_profiles: Dict[str, dict] = {}   # NEW

        # Load models and data
        self._load_models()
        self._preload_data()

    # ----------------- LOAD MODELS & DATA -----------------
    def _load_models(self) -> None:
        self.prod_df, self.prod_embeds = joblib.load(self.content_model_path)
        _, self.svd_model = dump.load(self.svd_model_path)

        if hasattr(self.svd_model, 'trainset') and self.svd_model.trainset:
            self.global_avg = self.svd_model.trainset.global_mean

        self.product_id_to_idx = {str(pid): idx for idx, pid in enumerate(self.prod_df["product_id"])}
        self.precompute_product_features()

    def precompute_product_features(self):
        self.product_features = {}
        for _, row in self.prod_df.iterrows():
            product_id = str(row["product_id"])
            self.product_features[product_id] = {
                'brand': row["brand_name"],
                'category': row["tertiary_category"],
                'price': row["price_usd"] if pd.notna(row["price_usd"]) else 0,
                'embedding': self.prod_embeds[self.product_id_to_idx[product_id]]
            }

    def _preload_data(self):
        self.train_df = pd.read_csv(self.train_path, usecols=["author_id", "product_id", "rating"])
        self.prod_df = pd.read_csv(self.products_path)

        user_groups = self.train_df.groupby("author_id")
        for user_id, group in user_groups:
            self.user_history_cache[str(user_id)] = {
                'rated_products': group["product_id"].astype(str).tolist(),
                'ratings': group["rating"].tolist(),
                'avg_rating': group["rating"].mean()
            }

        self.product_popularity = self.train_df['product_id'].astype(str).value_counts().to_dict()

    # ----------------- HYBRID CORE -----------------
    def enhanced_content_similarity(self, target_product_id: str, user_rated_products: List[str]) -> float:
        if target_product_id not in self.product_features or not user_rated_products:
            return 0.0

        target_embed = self.product_features[target_product_id]['embedding']
        similarities = []

        for rated_pid in user_rated_products:
            if rated_pid in self.product_features:
                rated_embed = self.product_features[rated_pid]['embedding']
                cosine_sim = cosine_similarity([target_embed], [rated_embed])[0][0]
                similarities.append(cosine_sim)

        return np.mean(similarities) if similarities else 0.0

    def hybrid_predict(self, user_id: str, product_id: str,
                       content_weight: float = 0.4, collab_weight: float = 0.6) -> Tuple[float, float]:
        user_id, product_id = str(user_id), str(product_id)

        # SVD Prediction
        try:
            svd_prediction = self.svd_model.predict(user_id, product_id)
            svd_pred = max(1.0, min(5.0, svd_prediction.est))
            svd_conf = 0.9 if not svd_prediction.details.get('was_impossible', False) else 0.4
        except:
            svd_pred, svd_conf = self.global_avg, 0.3

        # Content Prediction
        content_pred, content_conf = np.nan, 0.0
        if user_id in self.user_history_cache:
            rated_products = self.user_history_cache[user_id]['rated_products']
            if len(rated_products) >= 2 and product_id in self.product_id_to_idx:
                sim_score = self.enhanced_content_similarity(product_id, rated_products)
                if sim_score > 0.1:
                    content_pred = 1.0 + sim_score * 4.0
                    content_conf = min(1.0, sim_score * 1.8)
                    content_pred = max(1.0, min(5.0, content_pred))

        # Combine
        predictions, confidences, weights = [], [], []
        user_data = self.user_history_cache.get(user_id, {})
        ratio = min(1.0, len(user_data.get('rated_products', [])) / 30)

        if not np.isnan(svd_pred):
            predictions.append(svd_pred)
            confidences.append(svd_conf)
            weights.append(collab_weight * (0.4 + 0.6 * ratio))
        if not np.isnan(content_pred) and content_conf > 0.2:
            predictions.append(content_pred)
            confidences.append(content_conf)
            weights.append(content_weight * (1.0 - 0.6 * ratio))

        if len(predictions) == 2:
            total_conf = sum(c * w for c, w in zip(confidences, weights))
            weighted_pred = sum(p * c * w for p, c, w in zip(predictions, confidences, weights)) / total_conf
            final_conf = total_conf / sum(weights)
        elif len(predictions) == 1:
            weighted_pred, final_conf = predictions[0], confidences[0]
        else:
            weighted_pred = user_data.get('avg_rating', self.global_avg) + np.random.uniform(-0.2, 0.2)
            weighted_pred = max(1.0, min(5.0, weighted_pred))
            final_conf = 0.2

        return max(1.0, min(5.0, weighted_pred)), final_conf

    # ----------------- RECOMMENDATION -----------------
    def generate_recommendations(self, user_id: str, top_n: int = 10,
                                 content_weight: float = 0.4, collab_weight: float = 0.6) -> List[Tuple[str, float, int]]:
        user_id = str(user_id)
        user_rated = self.user_history_cache.get(user_id, {}).get('rated_products', [])
        candidate_products = [pid for pid in self.prod_df["product_id"].astype(str) if pid not in user_rated]

        if not candidate_products:
            return self._get_popular_fallback(top_n)

        recommendations = []
        for product_id in candidate_products:
            try:
                score, conf = self.hybrid_predict(user_id, product_id, content_weight, collab_weight)
                match_percent = self.calculate_match_percentage(score, user_id, product_id)

                # Skin profile adjustment
                multiplier = self.filter_by_skin_profile(product_id, user_id)
                score *= multiplier
                score = max(1.0, min(5.0, score))

                if conf >= 0.5 and match_percent >= 40:
                    recommendations.append((product_id, score, match_percent))
            except:
                continue

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]

    def calculate_match_percentage(self, score: float, user_id: str, product_id: str) -> int:
        user_avg = self.user_history_cache.get(str(user_id), {}).get('avg_rating', self.global_avg)
        if user_avg >= 4.0:
            match = (score - 2.8) / 2.2 * 100
        elif user_avg <= 2.5:
            match = (score - 1.8) / 3.2 * 100
        else:
            if score >= 3.5:
                match = 70 + (score - 3.5) / 1.5 * 30
            elif score >= 2.5:
                match = 40 + (score - 2.5) * 30
            else:
                match = score / 2.5 * 40
        return int(min(100, max(0, match)))

    def _get_popular_fallback(self, top_n: int) -> List[Tuple[str, float, int]]:
        popular = self.train_df.groupby('product_id')['rating'].agg(['count', 'mean']).reset_index()
        popular = popular[popular['count'] >= 10].sort_values(['mean', 'count'], ascending=False)

        result = []
        for _, row in popular.head(top_n).iterrows():
            score = row['mean']
            match = self.calculate_match_percentage(score, "average_user", row['product_id'])
            result.append((str(row['product_id']), score, match))
        return result

    # ----------------- SKIN PROFILE EXTENSION -----------------
    def add_skin_profile(self, user_id: str, profile: dict):
        """Store user's skin type, concern, and budget."""
        self.skin_profiles[str(user_id)] = profile

    def filter_by_skin_profile(self, product_id: str, user_id: str) -> float:
        """Adjust recommendation score by matching skin tags + budget."""
        profile = self.skin_profiles.get(str(user_id))
        if not profile:
            return 1.0

        user_type = profile.get("skin_type", "").lower()
        user_concern = profile.get("skin_concern", "").lower()
        user_budget = profile.get("budget", "")

        product = self.prod_df[self.prod_df["product_id"].astype(str) == str(product_id)].iloc[0]
        product_text = " ".join(map(str, [
            product.get("product_name", ""),
            product.get("combined_features", ""),
            product.get("ingredients", ""),
            product.get("claims", "")
        ]))
        price = product.get("price_usd", 0)

        matched_types, matched_concerns = self._extract_skin_tags(product_text)

        multiplier = 1.0
        if user_type and user_type in matched_types:
            multiplier *= 1.2
        elif user_type:
            multiplier *= 0.9

        if user_concern and user_concern in matched_concerns:
            multiplier *= 1.3
        elif user_concern:
            multiplier *= 0.85

        min_b, max_b = self._budget_range(user_budget)
        multiplier *= 1.1 if min_b <= price <= max_b else 0.7

        return max(0.3, min(multiplier, 2.0))

    def _extract_skin_tags(self, text: str) -> Tuple[List[str], List[str]]:
        types = ["oily", "dry", "sensitive", "normal", "combination"]
        concerns = ["acne", "wrinkle", "dark spot", "redness", "pore", "hydration"]

        text = text.lower()
        matched_types = [t for t in types if re.search(rf"\b{t}\b", text)]
        matched_concerns = [c for c in concerns if re.search(rf"\b{c}\b", text)]

        return matched_types, matched_concerns

    def _budget_range(self, budget: str) -> Tuple[float, float]:
        if budget == "Under $25": 
            return 0, 25
        if budget == "$25-$50": 
            return 25, 50
        if budget == "$50-$100": 
            return 50, 100
        if budget == "Over $100": 
            return 100, float("inf")
        return 0, float("inf")



# ----------------- PLACEHOLDER TEAMMATE CLASSES -----------------
class ContentBasedRecommender:
    def __init__(self, products_path: str):
        self.prod_df = pd.read_csv(products_path)
        print("✅ Content-based recommender initialized")

    def get_recommendations(self, user_id: str, skin_type: str, concerns: list,
                            budget: str, top_n: int = 5) -> List[dict]:
        filtered = self.prod_df.sample(top_n)
        recs = []
        for _, row in filtered.iterrows():
            recs.append({
                'product_id': str(row['product_id']),
                'name': row['product_name'],
                'brand': row['brand_name'],
                'category': row['tertiary_category'],
                'price': row['price_usd'],
                'rating': round(np.random.uniform(3.5, 5.0), 2),
                'match_percent': np.random.randint(70, 95)
            })
        return recs


class CollaborativeRecommender:
    def __init__(self, train_path: str):
        self.train_df = pd.read_csv(train_path)
        print("✅ Collaborative recommender initialized")

    def get_recommendations(self, user_id: str, top_n: int = 5) -> List[dict]:
        popular = self.train_df.groupby('product_id')['rating'].mean().nlargest(top_n).reset_index()
        recs = []
        for _, row in popular.iterrows():
            recs.append({
                'product_id': str(row['product_id']),
                'rating': round(row['rating'], 2),
                'match_percent': np.random.randint(75, 98)
            })
        return recs