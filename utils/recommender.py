import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple
from surprise import dump
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
import traceback
import os
import pickle
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
    def __init__(self, products_path: str, vectorizer, tfidf_matrix):
        self.products_path = products_path
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.prod_df = None
        self._load_data()

    def _load_data(self):
        """Load product data."""
        try:
            self.prod_df = pd.read_csv(self.products_path)
            for c in ["price_usd", "rating", "reviews"]:
                if c in self.prod_df.columns:
                    self.prod_df[c] = pd.to_numeric(self.prod_df[c], errors="coerce")
            self.prod_df["skin_concern"] = self.prod_df.get("skin_concern", "").fillna("").astype(str)
            self.prod_df["skin_type"] = self.prod_df.get("skin_type", "").fillna("").astype(str)
            self.prod_df["product_type"] = self.prod_df.get("product_type", "").fillna("").astype(str)
            if "product_content" not in self.prod_df.columns:
                raise ValueError("Missing 'product_content' column in products data")
            print("✅ Content-based recommender initialized")
        except Exception as e:
            print(f"❌ Error loading product data: {e}")
            raise

    def _to_set(self, x):
        """Convert input to a set of lowercase strings."""
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return set()
        if isinstance(x, (list, tuple, set)):
            return {str(t).strip().lower() for t in x if str(t).strip()}
        return {t.strip().lower() for t in re.split(r"[;,/|]", str(x)) if t.strip()}

    def get_recommendations(self, user_id: str, skin_type: str, concerns: list,
                           budget: str, top_n: int = 5, product_type: str = None,
                           concern_match: str = "all") -> pd.DataFrame:
        """Generate content-based recommendations based on skin profile."""
        print(f"Input: user_id={user_id}, skin_type={skin_type}, concerns={concerns}, budget={budget}, product_type={product_type}, concern_match={concern_match}")
        
        max_price = None
        if budget == "Under $25":
            max_price = 25
        elif budget == "$25-$50":
            max_price = 50
        elif budget == "$50-$100":
            max_price = 100
        elif budget == "Over $100":
            max_price = float("inf")
        elif budget == "No budget limit":
            max_price = float("inf")

        req_product_type = str(product_type).strip().lower() if product_type else None
        req_skin = str(skin_type).strip().lower() if skin_type else None
        req_concern = self._to_set(concerns if concerns else [])
        print(f"Processed inputs: product_type={req_product_type}, skin_type={req_skin}, concerns={req_concern}")

        tokens = []
        if req_product_type:
            tokens.append(req_product_type)
        if req_skin:
            tokens.append(req_skin)
        if req_concern:
            tokens.extend(sorted(req_concern))
        profile_text = " ".join(tokens).strip() or "skincare"
        print(f"Profile text: {profile_text}")

        qv = self.vectorizer.transform([profile_text])
        sims = cosine_similarity(qv, self.tfidf_matrix).ravel()
        print(f"Similarity scores: min={sims.min():.4f}, max={sims.max():.4f}, mean={sims.mean():.4f}")

        price_col = pd.to_numeric(self.prod_df.get("price_usd", np.nan), errors="coerce")
        rating_col = pd.to_numeric(self.prod_df.get("rating", np.nan), errors="coerce").fillna(0.0)
        reviews_col = pd.to_numeric(self.prod_df.get("reviews", 0), errors="coerce").fillna(0).astype(int)

        rows = []
        for i, sim in enumerate(sims):
            row = self.prod_df.iloc[i]

            # Product type filter (skip if None)
            if req_product_type and str(row.get("product_type", "")).strip().lower() != req_product_type:
                continue
            print(f"After product_type filter: {len(rows)+1} products")

            # Skin type filter (skip if None)
            row_skin = str(row.get("skin_type", "")).strip().lower()
            if req_skin and row_skin and row_skin != req_skin:
                continue
            print(f"After skin_type filter: {len(rows)+1} products")

            # Skin concern filter
            row_concern = self._to_set(row.get("skin_concern", ""))
            if req_concern:
                if concern_match == "all":
                    if not req_concern.issubset(row_concern):
                        continue
                else:
                    if row_concern.isdisjoint(req_concern):
                        continue
            print(f"After skin_concern filter: {len(rows)+1} products")

            # Price filter
            p = price_col.iat[i]
            if max_price is not None and (pd.isna(p) or p > float(max_price)):
                continue
            print(f"After price filter: {len(rows)+1} products")

            rows.append({
                "product_id": str(row.get("product_id", "")),
                "product_name": row.get("product_name", ""),
                "brand_name": row.get("brand_name", ""),
                "tertiary_category": row.get("product_type", ""),
                "price_usd": row.get("price_usd", ""),
                "rating": rating_col.iat[i],
                "reviews": reviews_col.iat[i],
                "similarity": float(sim)
            })

        print(f"Total products after filtering: {len(rows)}")
        out = pd.DataFrame(rows)
        if out.empty:
            print("No products matched filters. Returning top products by similarity.")
            out = pd.DataFrame([{
                "product_id": str(row.get("product_id", "")),
                "product_name": row.get("product_name", ""),
                "brand_name": row.get("brand_name", ""),
                "tertiary_category": row.get("product_type", ""),
                "price_usd": row.get("price_usd", ""),
                "rating": rating_col.iat[i],
                "reviews": reviews_col.iat[i],
                "similarity": float(sims[i])
            } for i, row in self.prod_df.iterrows()])
            out = out.sort_values(
                by=["similarity", "rating", "reviews"],
                ascending=[False, False, False]
            ).head(top_n)
            out["similarity"] = out["similarity"].round(4)
            return out

        out = out.sort_values(
            by=["similarity", "rating", "reviews"],
            ascending=[False, False, False]
        ).head(top_n)

        out["similarity"] = out["similarity"].round(4)
        return out



class CollaborativeRecommender:
    def __init__(self, train_path="collaborative_training_data.csv"):
        self.train_path = train_path
        self.model = None
        self.trainset = None
        self.df = None
        self.initialized = False
        self.error_message = ""
        
        print(f"Initializing CollaborativeRecommender with train_path: {train_path}")
        self._initialize()

    def _initialize(self):
        """Initialize the recommender with comprehensive error handling"""
        try:
            print("Step 1: Loading training data...")
            self._load_data()
            print(f"Data loaded, df shape: {self.df.shape if self.df is not None else 'None'}")
            
            print("Step 2: Loading model files...")
            self._load_model()
            
            self.initialized = True
            print("✅ CollaborativeRecommender initialized successfully!")
        except Exception as e:
            self.error_message = f"Initialization failed: {str(e)}"
            print(f"❌ {self.error_message}")
            traceback.print_exc()
            print(f"Current working directory: {os.getcwd()}")

    def _load_data(self):
        """Load and validate training data"""
        try:
            print(f"Checking existence of: {self.train_path}")
            if not os.path.exists(self.train_path):
                raise FileNotFoundError(f"Training data file not found: {self.train_path}")
            
            self.df = pd.read_csv(self.train_path)
            print(f"Training data loaded: {self.df.shape}")
            
            # Validate required columns
            required_columns = ['author_id', 'product_id', 'product_name', 'brand_name', 'rating']
            missing_cols = [col for col in required_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Additional columns that might be missing
            optional_columns = ['skin_type', 'price_usd', 'secondary_category', 'tertiary_category']
            for col in optional_columns:
                if col not in self.df.columns:
                    self.df[col] = 'Unknown'
                    print(f"Added missing column '{col}' with default values")
            
            print(f"Training data validation passed. Users: {self.df['author_id'].nunique()}, Products: {self.df['product_id'].nunique()}")
        except Exception as e:
            print(f"[ERROR] Error loading training data: {str(e)}")
            traceback.print_exc()
            self.df = None  # Explicitly set to None on failure
   
    def _load_model(self):
        """Load SVD model and trainset"""
        model_files = ['svd_model.pkl', 'trainset.pkl']
        missing_files = [f for f in model_files if not os.path.exists(f)]
        
        if missing_files:
            raise FileNotFoundError(f"Model files not found: {missing_files}")
        
        with open('svd_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open('trainset.pkl', 'rb') as f:
            self.trainset = pickle.load(f)
        
        print("Model files loaded successfully")

    def get_user_profile_and_recommendations(self, user_id, n=5, filter_by_favorite_brands=False):
        """Get user profile and recommendations with detailed logging"""
        
        if not self.initialized:
            print(f"❌ Recommender not initialized: {self.error_message}")
            return {}, []
        
        print(f"🔍 Getting recommendations for user: {user_id} (type: {type(user_id)})")
        
        try:
            # Ensure user_id is the right type
            user_id = str(user_id).strip() if user_id else ""
            if not user_id:
                print("❌ Empty user ID provided")
                return {}, []
            
            # Convert to numeric if needed (check your data format)
            try:
                # Try to convert to int first (in case your user_ids are stored as integers)
                numeric_user_id = int(user_id)
                # Check if this numeric version exists in the data
                if numeric_user_id in self.df['author_id'].values:
                    user_id = numeric_user_id
                    print(f"Using numeric user_id: {user_id}")
                else:
                    print(f"Numeric user_id {numeric_user_id} not found, trying string version")
            except ValueError:
                print(f"User ID is not numeric, using as string: '{user_id}'")
            
            # Check user existence
            user_data = self.df[self.df['author_id'] == user_id]
            is_new_user = len(user_data) == 0
            
            print(f"User search result: {len(user_data)} records found")
            if len(user_data) > 0:
                print(f"Sample user data: {user_data.head(2).to_dict('records')}")
            
            # Generate profile
            if is_new_user:
                print("Generating profile for new user")
                profile = {
                    'total_reviews': 0,
                    'avg_rating': 0.0,
                    'skin_type': "Unknown",
                    'favorite_brands': []
                }
                print("Getting popular recommendations for new user...")
                recommendations = self._get_popular_recommendations(n)
            else:
                print("Generating profile for existing user")
                profile = {
                    'total_reviews': len(user_data),
                    'avg_rating': float(user_data['rating'].mean()),
                    'skin_type': user_data['skin_type'].mode().iloc[0] if not user_data['skin_type'].isna().all() else "Unknown",
                    'favorite_brands': user_data['brand_name'].value_counts().head(3).index.tolist()
                }
                print("Getting personalized recommendations...")
                recommendations = self._get_personalized_recommendations(user_id, n)
            
            print(f"Final results: Profile keys: {list(profile.keys())}, Recommendations count: {len(recommendations)}")
            return profile, recommendations
            
        except Exception as e:
            print(f"❌ Error in get_user_profile_and_recommendations: {str(e)}")
            traceback.print_exc()
            return {}, []

    def _get_popular_recommendations(self, n):
        """Get popular recommendations for new users"""
        try:
            print("Calculating popular items...")
            
            # Get items with good ratings and sufficient reviews
            item_stats = self.df.groupby('product_id')['rating'].agg(['mean', 'count']).reset_index()
            popular_items = item_stats[item_stats['count'] >= 5].sort_values(
                by=['mean', 'count'], ascending=[False, False]
            ).head(n)
            
            print(f"Found {len(popular_items)} popular items")
            
            if popular_items.empty:
                print("No popular items found, using any available items")
                popular_items = item_stats.sort_values('mean', ascending=False).head(n)
            
            # Get product details
            recommendations = []
            for _, row in popular_items.iterrows():
                product_info = self.df[self.df['product_id'] == row['product_id']].iloc[0]
                recommendations.append({
                    'product_id': str(row['product_id']),
                    'product_name': str(product_info['product_name']),
                    'brand_name': str(product_info['brand_name']),
                    'price_usd': float(product_info['price_usd']) if pd.notna(product_info['price_usd']) else 0.0,
                    'secondary_category': str(product_info.get('secondary_category', 'Unknown')),
                    'tertiary_category': str(product_info.get('tertiary_category', 'Unknown')),
                    'predicted_rating': float(row['mean'])
                })
            
            print(f"Generated {len(recommendations)} popular recommendations")
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating popular recommendations: {e}")
            traceback.print_exc()
            return []

    def _get_personalized_recommendations(self, user_id, n):
        """Get personalized recommendations using SVD"""
        try:
            print(f"Getting personalized recommendations for user: {user_id}")
            
            # Check if user exists in trainset
            try:
                inner_uid = self.trainset.to_inner_uid(user_id)
                print(f"User found in trainset with inner_uid: {inner_uid}")
            except ValueError:
                print(f"User {user_id} not found in trainset, falling back to popular recommendations")
                return self._get_popular_recommendations(n)
            
            # Get unrated items
            all_items = set(self.trainset.all_items())
            rated_items = set(iid for (iid, _) in self.trainset.ur[inner_uid])
            unrated_items = list(all_items - rated_items)
            
            print(f"Total items: {len(all_items)}, Rated: {len(rated_items)}, Unrated: {len(unrated_items)}")
            
            if not unrated_items:
                print("No unrated items found")
                return []
            
            # Generate predictions
            predictions = []
            for inner_iid in unrated_items[:min(100, len(unrated_items))]:  # Limit for performance
                try:
                    raw_iid = self.trainset.to_raw_iid(inner_iid)
                    pred = self.model.predict(user_id, raw_iid)
                    predictions.append((raw_iid, pred.est))
                except Exception as e:
                    print(f"Error predicting for item {inner_iid}: {e}")
                    continue
            
            print(f"Generated {len(predictions)} predictions")
            
            # Sort and get top recommendations
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:n]
            
            # Format recommendations
            recommendations = []
            for product_id, predicted_rating in top_predictions:
                try:
                    product_info = self.df[self.df['product_id'] == product_id].iloc[0]
                    recommendations.append({
                        'product_id': str(product_id),
                        'product_name': str(product_info['product_name']),
                        'brand_name': str(product_info['brand_name']),
                        'price_usd': float(product_info['price_usd']) if pd.notna(product_info['price_usd']) else 0.0,
                        'secondary_category': str(product_info.get('secondary_category', 'Unknown')),
                        'tertiary_category': str(product_info.get('tertiary_category', 'Unknown')),
                        'predicted_rating': float(np.clip(predicted_rating, 1, 5))
                    })
                except Exception as e:
                    print(f"Error formatting recommendation for product {product_id}: {e}")
                    continue
            
            print(f"Formatted {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating personalized recommendations: {e}")
            traceback.print_exc()
            return []

    def get_available_users(self, limit=10):
        """Get sample user IDs for testing"""
        if self.df is None:
            return []
        return self.df['author_id'].unique()[:limit].tolist()

    def check_user_exists(self, user_id):
        """Check if user exists in training data"""
        if self.df is None:
            return False
        
        # Try both string and numeric versions
        try:
            numeric_user_id = int(user_id)
            return (user_id in self.df['author_id'].values) or (numeric_user_id in self.df['author_id'].values)
        except:
            return user_id in self.df['author_id'].values

    def get_system_info(self):
        """Get system information for debugging"""
        info = {
            'initialized': self.initialized,
            'error_message': self.error_message,
            'model_loaded': self.model is not None,
            'trainset_loaded': self.trainset is not None,
            'data_loaded': self.df is not None
        }
        
        if self.df is not None:
            info.update({
                'data_shape': self.df.shape,
                'unique_users': self.df['author_id'].nunique(),
                'unique_products': self.df['product_id'].nunique(),
                'sample_users': self.df['author_id'].head(5).tolist(),
                'user_id_types': str(self.df['author_id'].dtype)
            })
        
        return info