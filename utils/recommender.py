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
        print("🚀 EnhancedHybridRecommender v3.0 - IMPROVED SCORING LOADED!")
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
        self.skin_profiles: Dict[str, dict] = {}

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

    def get_recommendations_for_new_user(self, skin_type: str, concerns: list, 
                                   budget: str, top_n: int = 10) -> List[Tuple[str, float, int]]:
        """Get full hybrid content-based recommendations for new users across all categories"""
        print(f"🔍 Searching all {len(self.prod_df)} products for: {skin_type} skin, concerns: {concerns}")
        
        min_budget, max_budget = self._budget_range(budget)
        print(f"💰 Budget filter: ${min_budget} - ${max_budget}")
        
        recommendations = []
        budget_filtered_count = 0
        processed_count = 0
        
        # Create a lookup for original ratings
        original_ratings = {}
        
        # Pre-filter by budget to avoid unnecessary processing
        if budget and budget != "":
            budget_filtered_df = self.prod_df[
                (self.prod_df['price_usd'].isna()) | 
                ((self.prod_df['price_usd'] >= min_budget) & (self.prod_df['price_usd'] <= max_budget))
            ]
            print(f"📊 Budget pre-filtering: {len(budget_filtered_df)} products within budget")
        else:
            budget_filtered_df = self.prod_df
        
        for _, product in budget_filtered_df.iterrows():
            product_id = str(product["product_id"])
            processed_count += 1
            
            # Show progress for large datasets
            if processed_count % 500 == 0:
                print(f"⚡ Processed {processed_count}/{len(budget_filtered_df)} products...")
            
            # Create a temporary user profile for filtering
            temp_user_id = "temp_new_user"
            temp_profile = {
                'skin_type': skin_type,
                'concerns': concerns,
                'budget': budget
            }
            
            # Store temp profile and get compatibility score
            original_profile = self.skin_profiles.get(temp_user_id)
            self.skin_profiles[temp_user_id] = temp_profile
            
            try:
                # Use the existing filter_by_skin_profile method
                compatibility_score = self.filter_by_skin_profile(product_id, temp_user_id)
                
                # Skip products with low compatibility (less than 1.0 means filtered out)
                if compatibility_score < 1.0:
                    continue
                
                # Calculate concern score using the SAME accurate method as compatibility calculation
                concern_score = self._calculate_accurate_concern_score(product_id, temp_user_id, concerns)
                
                # Get original product rating for display
                original_rating = float(product.get('rating', 3.5))
                
                # Create sorting score that prioritizes concern score heavily
                # Concern score ranges 0-3+, compatibility ranges 0.3-2.0
                concern_weight = 10.0  # Heavy weight for concern score
                compatibility_weight = 1.0
                final_rating_for_sorting = (concern_score * concern_weight) + (compatibility_score * compatibility_weight)
                
                # Enhanced match percentage calculation based on multiple factors
                concern_match_bonus = min(15, concern_score * 5)  # Up to 15% bonus for high concern relevance
                skin_type_bonus = 5 if compatibility_score >= 1.5 else 0  # Bonus for good skin type match
                
                if compatibility_score >= 1.9:
                    base_match = 85 + (compatibility_score - 1.9) * 50  # 85-95% for exceptional matches
                elif compatibility_score >= 1.7:
                    base_match = 75 + (compatibility_score - 1.7) * 50  # 75-85% for great matches
                elif compatibility_score >= 1.4:
                    base_match = 60 + (compatibility_score - 1.4) * 50  # 60-75% for good matches
                elif compatibility_score >= 1.2:
                    base_match = 45 + (compatibility_score - 1.2) * 75  # 45-60% for decent matches
                else:
                    base_match = max(35, compatibility_score * 35)  # 35-45% for basic matches
                
                match_percent = int(min(95, base_match + concern_match_bonus + skin_type_bonus))
                
                # Store original rating for later lookup
                original_ratings[product_id] = original_rating
                
                # Return: (product_id, final_rating_for_sorting, match_percent) - use final_rating for sorting
                recommendations.append((product_id, final_rating_for_sorting, match_percent))
                
            except Exception as e:
                # Log errors but continue processing
                if processed_count <= 10:  # Only show first few errors to avoid spam
                    print(f"⚠️  Error processing product {product_id}: {str(e)[:100]}...")
                continue
                
            finally:
                # Restore original profile
                if original_profile is not None:
                    self.skin_profiles[temp_user_id] = original_profile
                else:
                    self.skin_profiles.pop(temp_user_id, None)
        
        print(f"📊 Found {len(recommendations)} matching products (filtered out {budget_filtered_count} by budget)")
        
        # Sort by final_rating_for_sorting which now heavily weights concern score
        recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by original_rating, but it's actually final_rating_for_sorting
        
        # Apply category diversity logic for larger requests
        if top_n > 5:
            diverse_recommendations = self._apply_category_diversity(recommendations, top_n, original_ratings)
            # Convert back to app format: (product_id, original_rating, match_percent)
            return [(rec[0], original_ratings.get(rec[0], 3.5), rec[2]) for rec in diverse_recommendations]
        
        # Convert back to app format: (product_id, original_rating, match_percent)  
        final_recommendations = [(rec[0], original_ratings.get(rec[0], 3.5), rec[2]) for rec in recommendations[:top_n]]
        return final_recommendations
    
    def _apply_category_diversity(self, recommendations: List[Tuple[str, float, int]], top_n: int, original_ratings: dict) -> List[Tuple[str, float, int]]:
        """Ensure category diversity for larger product requests"""
        if len(recommendations) <= top_n:
            return recommendations
        
        # Define essential categories in priority order
        essential_categories = [
            "Face Wash & Cleansers",
            "Face Serums", 
            "Moisturizers",
            "Face Sunscreen",
            "Treatments & Masks",
            "Mists & Essences",
            "Eye Care",
            "Exfoliants",
            "Face Oils",
            "Toners"
        ]
        
        selected = []
        category_count = {}
        
        # First pass: Get best product from each essential category
        for category in essential_categories:
            if len(selected) >= top_n:
                break
                
            for product_id, final_rating, match_percent in recommendations:
                if len(selected) >= top_n:
                    break
                    
                # Check if already selected
                if product_id in [p[0] for p in selected]:
                    continue
                    
                # Get product category
                product = self.prod_df[self.prod_df["product_id"].astype(str) == product_id]
                if not product.empty:
                    product_category = product.iloc[0]["tertiary_category"]
                    
                    if product_category == category:
                        selected.append((product_id, final_rating, match_percent))
                        category_count[category] = category_count.get(category, 0) + 1
                        break
        
        # Second pass: Fill remaining slots with best remaining products (max 2 per category)
        for product_id, final_rating, match_percent in recommendations:
            if len(selected) >= top_n:
                break
                
            if product_id in [p[0] for p in selected]:
                continue
                
            product = self.prod_df[self.prod_df["product_id"].astype(str) == product_id]
            if not product.empty:
                product_category = product.iloc[0]["tertiary_category"]
                current_count = category_count.get(product_category, 0)
                
                # Allow max 2 products per category for larger requests
                max_per_category = 2 if top_n > 8 else 1
                if current_count < max_per_category:
                    selected.append((product_id, final_rating, match_percent))
                    category_count[product_category] = current_count + 1
        
        return selected
    
    def _calculate_concern_score_for_ranking(self, product, user_concerns):
        """Calculate concern score for ranking purposes"""
        if not user_concerns:
            return 0.0
            
        # Handle both dict and pandas Series
        if hasattr(product, 'get'):
            product_text = str(product.get("combined_features", "")).lower()
        else:
            product_text = str(product["combined_features"] if "combined_features" in product.index else "").lower()
        
        # Keyword matching
        keyword_matches = 0
        for concern in user_concerns:
            if concern.lower() in product_text:
                keyword_matches += 1
        
        # Simple semantic matching based on common ingredients/keywords
        semantic_score = 0.0
        concern_keywords = {
            'acne': ['salicylic', 'benzoyl', 'niacinamide', 'tea tree', 'zinc'],
            'redness': ['centella', 'aloe', 'ceramide', 'panthenol', 'allantoin'],
            'dryness': ['hyaluronic', 'glycerin', 'ceramide', 'squalane', 'shea'],
            'aging': ['retinol', 'peptides', 'vitamin c', 'collagen', 'antioxidant'],
            'hyperpigmentation': ['vitamin c', 'niacinamide', 'kojic', 'arbutin', 'alpha arbutin']
        }
        
        for concern in user_concerns:
            if concern.lower() in concern_keywords:
                for keyword in concern_keywords[concern.lower()]:
                    if keyword in product_text:
                        semantic_score += 0.1
        
        total_score = keyword_matches + semantic_score
        return total_score
    
    def _normalize_user_input(self, user_concerns: List[str]) -> List[str]:
        """Simple normalization for user dropdown selections that don't match internal names"""
        # Only normalize the few cases where dropdown text differs from internal system
        mapping = {
            'large pores': 'pores',        # Dropdown: "Large pores" → Internal: "pores"  
            'pigmentation': 'hyperpigmentation',  # Dropdown: "Pigmentation" → Internal: "hyperpigmentation"
            'sensitivity': 'redness',      # Dropdown: "Sensitivity" → Internal: "redness"
        }
        
        return [mapping.get(concern.lower(), concern.lower()) for concern in user_concerns]

    def _calculate_accurate_concern_score(self, product_id: str, user_id: str, user_concerns: list) -> float:
        """Calculate concern score using the SAME accurate method as filter_by_skin_profile"""
        if not user_concerns:
            return 0.0
            
        # Get product using the same method as filter_by_skin_profile
        product = self.prod_df[self.prod_df["product_id"].astype(str) == str(product_id)].iloc[0]
        
        # Use the SAME text processing as filter_by_skin_profile
        if 'combined_features' in self.prod_df.columns and pd.notna(product.get("combined_features")):
            product_text = str(product.get("combined_features", ""))
        else:
            # Fallback to manual construction (same as filter_by_skin_profile)
            product_text = " ".join(map(str, [
                product.get("product_name", ""),
                product.get("highlights", ""),
                product.get("ingredients", ""),
                product.get("combined_features", ""),
                product.get("claims", "")
            ]))
        
        # Use the SAME extraction method as filter_by_skin_profile
        matched_types, matched_concerns = self._extract_skin_tags(product_text)
        
        # Normalize user concerns consistently
        normalized_user_concerns = self._normalize_user_input(user_concerns)
        
        # Use the SAME semantic concern matching as filter_by_skin_profile
        semantic_concern_score = self._calculate_semantic_concern_match(normalized_user_concerns, product_text, product_id)
        
        # Use the SAME keyword matching logic as filter_by_skin_profile - but with normalized concerns
        keyword_matches = len([c for c in normalized_user_concerns if c in matched_concerns])
        
        # Return the SAME total concern score as used in filter_by_skin_profile
        total_concern_score = keyword_matches + semantic_concern_score
        
        return total_concern_score

    def enhanced_demo_recommendations(self, user_id: str, top_n: int = 5,
                                 content_weight: float = 0.4, collab_weight: float = 0.6,
                                 selected_product_id: str = None):
        """Enhanced demo recommendations with same-category filtering for new users"""
        user_id = str(user_id)
        user_exists = user_id in self.user_history_cache
        
        if not user_exists:
            print(f"\n🎯 NEW USER DETECTED: {user_id}")
            print("🚀 USING FULL HYBRID CONTENT-BASED FILTERING (SEARCHING ALL CATEGORIES) - VERSION 2.0")
            
            # Get user's skin profile for full hybrid recommendations
            user_profile = self.skin_profiles.get(str(user_id), {})
            user_skin_type = user_profile.get('skin_type', 'combination')
            user_concerns = user_profile.get('concerns', ["acne", "hydration"])
            user_budget = user_profile.get('budget', 'mid')
            
            print(f"👤 User profile: {user_skin_type} skin, concerns: {user_concerns}, budget: {user_budget}")
            
            # Use full hybrid recommendations across all categories
            return self.get_recommendations_for_new_user(user_skin_type, user_concerns, user_budget, top_n)
        else:
            print(f"\n🎯 EXISTING USER: {user_id}")
            print("Using Hybrid Filtering (collaborative + content)")
            return self.generate_recommendations(user_id, top_n, content_weight, collab_weight)
    
    # ----------------- RECOMMENDATION -----------------
    def generate_recommendations(self, user_id: str, top_n: int = 10,
                             content_weight: float = 0.4, collab_weight: float = 0.6) -> List[Tuple[str, float, int]]:
        user_id = str(user_id)
        user_rated = self.user_history_cache.get(user_id, {}).get('rated_products', [])
        candidate_products = [pid for pid in self.prod_df["product_id"].astype(str) if pid not in user_rated]

        if not candidate_products:
            return self._get_popular_fallback(top_n)

        # Check if user is new (not in training data)
        user_exists = user_id in self.user_history_cache
        
        recommendations = []
        for product_id in candidate_products:
            try:
                if not user_exists:
                    # NEW USER: Use content-based approach only
                    content_pred, content_conf = self._content_based_predict(user_id, product_id)
                    score = content_pred
                    conf = content_conf
                else:
                    # EXISTING USER: Use hybrid approach
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

    def _content_based_predict(self, user_id: str, product_id: str) -> Tuple[float, float]:
        """Content-based prediction for new users"""
        if product_id not in self.product_id_to_idx:
            return self.global_avg, 0.3
        
        # Get average rating for this product category
        product_info = self.prod_df[self.prod_df["product_id"].astype(str) == product_id]
        if product_info.empty:
            return self.global_avg, 0.3
        
        category = product_info.iloc[0]["tertiary_category"]
        
        # Get average rating for this category across all users
        category_products = self.prod_df[self.prod_df["tertiary_category"] == category]["product_id"].astype(str)
        category_ratings = []
        
        for cat_product_id in category_products:
            if cat_product_id in self.product_popularity:
                # Use the product's popularity as a proxy for rating
                category_ratings.append(min(5.0, self.product_popularity[cat_product_id] / 1000 + 3.0))
        
        if category_ratings:
            avg_rating = np.mean(category_ratings)
            confidence = min(1.0, len(category_ratings) / 50)  # More products = more confidence
        else:
            avg_rating = self.global_avg
            confidence = 0.3
        
        return avg_rating, confidence

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
        """Enhanced skin profile filtering using semantic similarity"""
        profile = self.skin_profiles.get(str(user_id))
        if not profile:
            return 1.0

        user_type = profile.get("skin_type", "").lower()
        user_concerns = profile.get("concerns", [])
        if isinstance(user_concerns, str):
            user_concerns = [user_concerns]
        
        # Normalize user concerns using centralized method
        normalized_user_concerns = self._normalize_user_input(user_concerns)
        user_budget = profile.get("budget", "")

        product = self.prod_df[self.prod_df["product_id"].astype(str) == str(product_id)].iloc[0]
        # Use the pre-processed combined_features if available, otherwise construct manually
        if 'combined_features' in self.prod_df.columns and pd.notna(product.get("combined_features")):
            product_text = str(product.get("combined_features", ""))
        else:
            # Fallback to manual construction
            product_text = " ".join(map(str, [
                product.get("product_name", ""),
                product.get("highlights", ""),  # This contains "Best for Dry, Combo, Normal Skin"
                product.get("ingredients", ""),
                product.get("combined_features", ""),  # Keep for compatibility
                product.get("claims", "")  # Keep for compatibility
            ]))
        price = product.get("price_usd", 0)

        matched_types, matched_concerns = self._extract_skin_tags(product_text)
        
        # 🎯 SEMANTIC SIMILARITY FOR CONCERNS using normalized concerns
        semantic_concern_score = self._calculate_semantic_concern_match(normalized_user_concerns, product_text, product_id)

        multiplier = 1.0
        
        # ENHANCED skin type filtering with better matching
        if user_type and matched_types:
            if user_type in matched_types:
                multiplier *= 1.4  # Perfect match bonus
            elif "combination" in matched_types or user_type == "combination":
                # Combination skin works well with most products
                multiplier *= 1.1  # Slight bonus for versatile products
            elif (user_type == "dry" and "oily" in matched_types) or \
                 (user_type == "oily" and "dry" in matched_types):
                multiplier *= 0.3  # Penalty for opposite skin types
            else:
                multiplier *= 0.7  # Moderate penalty for other mismatches
        elif user_type and not matched_types:
            # Products that don't specify skin type - treat as universal
            multiplier *= 1.0  # Neutral for universal products

        # 🎯 ENHANCED CONCERN MATCHING: Combine keyword + semantic using normalized concerns
        keyword_matches = len([c for c in normalized_user_concerns if c in matched_concerns])
        
        # Combine keyword matching with semantic similarity
        total_concern_score = keyword_matches + semantic_concern_score
        
        if total_concern_score > 0:
            # Scale the boost based on combined score (honest scoring)
            multiplier *= (1.1 + 0.15 * min(total_concern_score, 3.0))  # Cap at 3 concerns
        elif user_concerns:
            # Penalty only if no keyword OR semantic match
            multiplier *= 0.85

        # Budget filtering
        min_b, max_b = self._budget_range(user_budget)
        multiplier *= 1.1 if min_b <= price <= max_b else 0.7

        return max(0.3, min(multiplier, 2.0))  # Honest maximum

    def _calculate_semantic_concern_match(self, user_concerns: List[str], product_text: str, product_id: str) -> float:
        """Calculate semantic similarity using the same pattern-based approach as final.ipynb"""
        if not user_concerns:
            return 0.0
        
        try:
            # Use centralized concern normalization
            normalized_concerns = self._normalize_user_input(user_concerns)
            
            # Map user concerns to ingredient/description patterns for better matching
            concern_ingredient_map = {
                'acne': ['salicylic acid', 'benzoyl peroxide', 'tea tree', 'zinc', 'bha', 'willow bark', 'sulfur', 
                        'niacinamide', 'azelaic acid', 'adapalene', 'retinoid', 'clay', 'charcoal', 'antibacterial',
                        'antimicrobial', 'anti-acne', 'blemish', 'pore-clearing'],
                'aging': ['retinol', 'peptide', 'collagen', 'bakuchiol', 'matrixyl', 'coenzyme q10', 'anti-aging',
                         'palmitoyl', 'argireline', 'copper peptide', 'growth factor'],
                'dehydration': ['hyaluronic acid', 'glycerin', 'ceramide', 'squalane', 'panthenol', 'moisturizing', 'hydrating',
                               'sodium hyaluronate', 'barrier repair', 'lipid', 'moisture barrier'],
                'redness': ['centella', 'cica', 'allantoin', 'bisabolol', 'green tea', 'calming', 'soothing',
                           'anti-inflammatory', 'sensitive skin', 'rosacea', 'irritation'],
                'hyperpigmentation': ['vitamin c', 'arbutin', 'kojic', 'tranexamic', 'azelaic', 'brightening',
                                     'hydroquinone', 'licorice', 'dark spot', 'even tone'],
                'pores': ['salicylic acid', 'clay', 'charcoal', 'niacinamide', 'bha', 'pore-minimizing', 'refining', 'pore-clearing', 'blackhead', 'whitehead'],
                'oil-control': ['clay', 'charcoal', 'zinc', 'mattifying', 'oil control', 'sebum control', 'shine control'],
                'dullness': ['vitamin c', 'aha', 'glycolic', 'brightening', 'glow', 'radiance', 'luminous', 'revitalizing'],
                'texture': ['aha', 'glycolic', 'lactic', 'resurfacing', 'exfoliating', 'smoothing', 'refining']
            }
            
            text_lower = product_text.lower()
            
            concern_scores = []
            
            for concern in normalized_concerns:  # Use normalized concerns
                # Direct concern match (already handled by keyword matching)
                direct_score = 0.0
                
                # Ingredient-based semantic match with smart combination logic
                semantic_score = 0.0
                if concern in concern_ingredient_map:
                    ingredients = concern_ingredient_map[concern]
                    matches = sum(1 for ingredient in ingredients if ingredient in text_lower)
                    if matches > 0:
                        # Smart ingredient scoring based on skin type compatibility
                        semantic_score = self._calculate_skin_compatible_score(concern, ingredients, text_lower, normalized_concerns)
                
                total_score = direct_score + semantic_score
                if total_score > 0:
                    concern_scores.append(min(1.0, total_score))
            
            final_score = np.mean(concern_scores) if concern_scores else 0.0
            return final_score
            
        except Exception as e:
            return 0.0

    def _calculate_skin_compatible_score(self, concern: str, ingredients: list, text_lower: str, user_concerns: list) -> float:
        """Calculate ingredient score based on skin type and concern combinations"""
        matches = sum(1 for ingredient in ingredients if ingredient in text_lower)
        if matches == 0:
            return 0.0
        
        base_score = matches * 0.2  # Base scoring
        
        # Define ingredient compatibility matrices for different skin combinations
        ingredient_compatibility = {
            # Gentle ingredients suitable for dry/sensitive skin
            'gentle': ['niacinamide', 'azelaic acid', 'zinc', 'tea tree', 'ceramide', 'hyaluronic acid', 
                      'centella', 'allantoin', 'panthenol', 'squalane', 'bakuchiol'],
            
            # Stronger ingredients better for oily/resilient skin  
            'strong': ['salicylic acid', 'benzoyl peroxide', 'glycolic', 'retinol', 'clay', 'charcoal'],
            
            # Multi-benefit ingredients good for combination concerns
            'versatile': ['niacinamide', 'vitamin c', 'peptide', 'hyaluronic acid', 'azelaic acid']
        }
        
        # Get user's likely skin tolerance from their concerns
        user_skin_types = [c for c in user_concerns if c in ['dry', 'oily', 'sensitive', 'combination', 'normal']]
        
        # Smart scoring based on combinations
        if 'acne' in user_concerns:
            if 'dry' in user_skin_types or 'sensitive' in user_skin_types:
                # Dry/sensitive + acne: prefer gentle ingredients
                gentle_matches = sum(1 for ingredient in ingredient_compatibility['gentle'] if ingredient in text_lower)
                if gentle_matches > 0:
                    return min(1.0, base_score + (gentle_matches * 0.3))  # Bonus for gentle
                else:
                    return min(1.0, base_score * 0.8)  # Slight penalty for potentially harsh
            
            elif 'oily' in user_skin_types:
                # Oily + acne: can handle stronger ingredients
                strong_matches = sum(1 for ingredient in ingredient_compatibility['strong'] if ingredient in text_lower)
                if strong_matches > 0:
                    return min(1.0, base_score + (strong_matches * 0.2))  # Bonus for effective
        
        elif 'aging' in user_concerns:
            if 'dry' in user_skin_types or 'sensitive' in user_skin_types:
                # Dry/sensitive + aging: prefer gentle actives
                gentle_aging = ['bakuchiol', 'peptide', 'vitamin c', 'ceramide', 'hyaluronic acid']
                gentle_matches = sum(1 for ingredient in gentle_aging if ingredient in text_lower)
                if gentle_matches > 0:
                    return min(1.0, base_score + (gentle_matches * 0.25))
            
            elif 'oily' in user_skin_types:
                # Oily + aging: can handle retinoids better
                strong_aging = ['retinol', 'glycolic', 'salicylic acid']
                strong_matches = sum(1 for ingredient in strong_aging if ingredient in text_lower)
                if strong_matches > 0:
                    return min(1.0, base_score + (strong_matches * 0.25))
        
        elif 'hyperpigmentation' in user_concerns:
            # Pigmentation: look for proven brightening ingredients
            brightening = ['vitamin c', 'arbutin', 'kojic', 'azelaic acid', 'tranexamic']
            brightening_matches = sum(1 for ingredient in brightening if ingredient in text_lower)
            if brightening_matches > 0:
                return min(1.0, base_score + (brightening_matches * 0.3))
        
        # Multi-concern users: prefer versatile ingredients
        if len(user_concerns) >= 2:
            versatile_matches = sum(1 for ingredient in ingredient_compatibility['versatile'] if ingredient in text_lower)
            if versatile_matches > 0:
                return min(1.0, base_score + (versatile_matches * 0.2))
        
        return min(1.0, base_score)  # Default scoring

    def get_recommendation_debug_info(self, product_id: str, user_id: str) -> dict:
        """Get detailed information about why a product was recommended."""
        profile = self.skin_profiles.get(str(user_id))
        if not profile:
            return {"error": "No profile found for user"}

        user_type = profile.get("skin_type", "").lower()
        user_concerns = profile.get("concerns", [])
        if isinstance(user_concerns, str):
            user_concerns = [user_concerns]
        user_concerns = [c.lower() for c in user_concerns]
        user_budget = profile.get("budget", "")

        product = self.prod_df[self.prod_df["product_id"].astype(str) == str(product_id)].iloc[0]
        # Use the pre-processed combined_features if available, otherwise construct manually
        if 'combined_features' in self.prod_df.columns and pd.notna(product.get("combined_features")):
            product_text = str(product.get("combined_features", ""))
        else:
            # Fallback to manual construction
            product_text = " ".join(map(str, [
                product.get("product_name", ""),
                product.get("highlights", ""),  # This contains "Best for Dry, Combo, Normal Skin"
                product.get("ingredients", ""),
                product.get("combined_features", ""),  # Keep for compatibility  
                product.get("claims", "")  # Keep for compatibility
            ]))
        price = product.get("price_usd", 0)

        matched_types, matched_concerns = self._extract_skin_tags(product_text)
        semantic_score = self._calculate_semantic_concern_match(user_concerns, product_text, product_id)
        
        # Calculate the same multiplier logic for debugging
        multiplier = 1.0
        skin_type_status = "No skin type specified in product"
        concern_status = "No concerns matched"
        budget_status = "Within budget" if self._budget_range(user_budget)[0] <= price <= self._budget_range(user_budget)[1] else "Outside budget"
        
        if user_type and matched_types:
            if user_type in matched_types:
                skin_type_status = f"✅ Perfect match: {user_type} skin"
                multiplier *= 1.3
            else:
                if (user_type == "dry" and "oily" in matched_types) or \
                   (user_type == "oily" and "dry" in matched_types):
                    skin_type_status = f"❌ Opposite skin type: Product for {matched_types}, you have {user_type}"
                    multiplier *= 0.2
                else:
                    skin_type_status = f"⚠️ Different skin type: Product for {matched_types}, you have {user_type}"
                    multiplier *= 0.6
        elif user_type and not matched_types:
            skin_type_status = f"⚪ No skin type specified (neutral for {user_type} skin)"
            
        # Enhanced concern matching
        keyword_matches = len([c for c in user_concerns if c in matched_concerns])
        total_concern_score = keyword_matches + semantic_score
        
        concern_details = []
        if keyword_matches > 0:
            matched_keywords = [c for c in user_concerns if c in matched_concerns]
            concern_details.append(f"Keyword matches: {', '.join(matched_keywords)}")
        if semantic_score > 0:
            concern_details.append(f"Semantic similarity: {semantic_score:.2f}")
            
        if total_concern_score > 0:
            concern_status = f"✅ {total_concern_score:.1f} concern score: {'; '.join(concern_details)}"
            multiplier *= (1.1 + 0.15 * min(total_concern_score, 3.0))
        elif user_concerns:
            concern_status = f"❌ No matches (you: {', '.join(user_concerns)}, product: {', '.join(matched_concerns) if matched_concerns else 'none'})"
            multiplier *= 0.85

        return {
            "user_profile": {
                "skin_type": user_type,
                "concerns": user_concerns,
                "budget": user_budget
            },
            "product_info": {
                "detected_skin_types": matched_types,
                "detected_concerns": matched_concerns,
                "price": price
            },
            "compatibility": {
                "skin_type_status": skin_type_status,
                "concern_status": concern_status,
                "budget_status": budget_status,
                "final_multiplier": round(max(0.3, min(multiplier, 2.0)), 2)
            }
        }

    def _extract_skin_tags(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract skin types and concerns using regex patterns (from final.ipynb)"""
        text = text.lower()
        matched_types = []
        matched_concerns = []
        
        # Skin type patterns
        SKIN_TYPE_PATTERNS = [
            (r"\b(?:good|best)\s*for:\s*oily\b", "oily"),
            (r"\b(?:good|best)\s*for:\s*dry\b", "dry"),
            (r"\b(?:good|best)\s*for:\s*combination\b", "combination"),
            (r"\b(?:good|best)\s*for:\s*sensitive\b", "sensitive"),
            (r"\b(?:good|best)\s*for:\s*normal\b", "normal"),
            (r"\b(oily skin|oily)\b", "oily"),
            (r"\b(dry skin|dry)\b", "dry"),
            (r"\b(combination skin|combination|combo)\b", "combination"),
            (r"\b(sensitive skin|sensitive)\b", "sensitive"),
            (r"\b(normal skin|normal)\b", "normal"),
            (r"\bfor\s+sensitive\s+skin\b", "sensitive"),
            (r"\bsuitable\s+for\s+sensitive\b", "sensitive"),
            (r"\bfor\s+sensitive\b", "sensitive"),
            (r"\bhypoallergenic\b", "sensitive"),
            (r"\bgentle\b", "sensitive"),
        ]
        
        # Skin concern patterns - Enhanced to match normalization
        SKIN_CONCERN_PATTERNS = [
            (r"\b(acne|blemish|breakout|pimple|acne-prone|anti-acne)\b", "acne"),
            (r"\b(pores?|large pores?|enlarged pores?|clogged pores?|pore-minimizing|pore-clearing)\b", "pores"),
            (r"\b(blackhead|whitehead|congestion)\b", "pores"),  # Map to pores for consistency
            (r"\b(dark spot|hyperpigment|discoloration|melasma|sun spot|age spot)\b", "hyperpigmentation"),
            (r"\b(wrinkle|fine line|anti[- ]?aging|firming|loss of firmness|elasticity)\b", "aging"),
            (r"\b(redness|rosacea|irritation|calming|soothing|sensitivity|sensitive)\b", "redness"),
            (r"\b(dryness|dehydration|hydrating|moisturizing|moisturising|barrier|dry skin)\b", "dehydration"),
            (r"\b(dull(ness)?|brighten(ing)?|glow|radiance|luminous|lack of glow|uneven tone|dull skin)\b", "dullness"),
            (r"\b(oil(y| control|iness)|excess oil|greasy|shine|mattifying|sebum|greasy skin|oiliness)\b", "oil-control"),
            (r"\b(uneven texture|rough texture|texture|resurfacing|bumpy|rough|bumpy skin)\b", "texture"),
            (r"\b(dark circle|dark circles)\b", "dark-circles"),
        ]
        
        # Ingredient-based concern mapping
        INGREDIENT_CONCERN_PATTERNS = [
            (r"\b(salicylic acid|beta hydroxy|bha|willow bark|benzoyl peroxide|sulfur|zinc pca|zinc)\b", {"acne","pores","oil-control"}),
            (r"\b(kaolin|bentonite|clay|charcoal)\b", {"pores","oil-control"}),
            (r"\b(tea tree|melaleuca)\b", {"acne"}),
            (r"\b(hyaluronic acid|sodium hyaluronate|glycerin|panthenol|urea|betaine|trehalose|aloe)\b", {"dehydration"}),
            (r"\b(ceramide|ceramides|cholesterol|squalane|squalene|shea|shea butter)\b", {"dehydration"}),
            (r"\b(retinol|retinal|retinoate|bakuchiol|peptide|matrixyl|collagen|coenzyme ?q10|ubiquinone)\b", {"aging"}),
            (r"\b(vitamin ?c|ascorbic|ascorbyl|ethyl ascorbic|magnesium ascorbyl|sodium ascorbyl|alpha arbutin|tranexamic|azelaic|kojic|licorice|glycyrrhiza)\b", {"hyperpigmentation","dullness"}),
            (r"\b(centella|cica|madecassoside|asiaticoside|allantoin|bisabolol|beta glucan|green tea|oat|colloidal oatmeal)\b", {"redness"}),
            (r"\b(aha|glycolic|lactic|mandelic|tartaric|citric|pha|gluconolactone|lactobionic)\b", {"texture","dullness"}),
        ]
        
        # Extract skin types
        for pattern, skin_type in SKIN_TYPE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if skin_type not in matched_types:
                    matched_types.append(skin_type)
        
        # Extract concerns from descriptions
        for pattern, concern in SKIN_CONCERN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if concern not in matched_concerns:
                    matched_concerns.append(concern)
        
        # Extract concerns from ingredients
        for pattern, concerns_set in INGREDIENT_CONCERN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                for concern in concerns_set:
                    if concern not in matched_concerns:
                        matched_concerns.append(concern)

        return matched_types, matched_concerns

    def check_ingredient_conflicts(self, product_recommendations: List[Tuple[str, float, int]]) -> dict:
        """Check for potential ingredient conflicts in recommended products"""
        conflicts = []
        warnings = []
        
        # Define conflicting ingredient combinations
        conflict_rules = {
            'retinol': {
                'conflicts_with': ['aha', 'bha', 'benzoyl peroxide', 'vitamin c'],
                'warning': 'Use retinol separately from acids and vitamin C to avoid irritation'
            },
            'vitamin c': {
                'conflicts_with': ['retinol', 'benzoyl peroxide'],
                'warning': 'Use vitamin C in AM, retinol in PM'
            },
            'benzoyl peroxide': {
                'conflicts_with': ['retinol', 'vitamin c'],
                'warning': 'Use benzoyl peroxide separately from retinol and vitamin C'
            }
        }
        
        # Get ingredients from all recommended products
        product_ingredients = {}
        for product_id, _, _ in product_recommendations:
            try:
                product = self.prod_df[self.prod_df["product_id"].astype(str) == str(product_id)].iloc[0]
                combined_text = str(product.get('combined_features', '')).lower()
                
                found_ingredients = []
                for ingredient in conflict_rules.keys():
                    if ingredient in combined_text:
                        found_ingredients.append(ingredient)
                
                if found_ingredients:
                    product_ingredients[product_id] = {
                        'name': product.get('product_name', 'Unknown'),
                        'ingredients': found_ingredients
                    }
            except:
                continue
        
        # Check for conflicts
        ingredient_products = {}
        for product_id, data in product_ingredients.items():
            for ingredient in data['ingredients']:
                if ingredient not in ingredient_products:
                    ingredient_products[ingredient] = []
                ingredient_products[ingredient].append((product_id, data['name']))
        
        # Find conflicts
        for ingredient, products in ingredient_products.items():
            if ingredient in conflict_rules:
                conflicts_with = conflict_rules[ingredient]['conflicts_with']
                for conflict_ingredient in conflicts_with:
                    if conflict_ingredient in ingredient_products:
                        conflicts.append({
                            'ingredient1': ingredient,
                            'ingredient2': conflict_ingredient,
                            'products1': products,
                            'products2': ingredient_products[conflict_ingredient],
                            'warning': conflict_rules[ingredient]['warning']
                        })
        
        return {
            'has_conflicts': len(conflicts) > 0,
            'conflicts': conflicts,
            'product_ingredients': product_ingredients
        }

    def get_recommendation_debug_info(self, product_id: str, user_id: str) -> dict:
        """Get detailed debug information for why a product was recommended"""
        try:
            profile = self.skin_profiles.get(str(user_id), {})
            if not profile:
                return {"error": "No user profile found"}
            
            product = self.prod_df[self.prod_df["product_id"].astype(str) == str(product_id)]
            if product.empty:
                return {"error": "Product not found"}
            
            product = product.iloc[0]
            
            # Get compatibility breakdown
            compatibility_score = self.filter_by_skin_profile(product_id, user_id)
            concern_score = self._calculate_accurate_concern_score(product_id, user_id, profile.get('concerns', []))
            
            # Extract what the product text says about skin types and concerns
            combined_text = str(product.get('combined_features', ''))
            matched_types, matched_concerns = self._extract_skin_tags(combined_text)
            
            user_skin_type = profile.get('skin_type', '').lower()
            user_concerns = profile.get('concerns', [])
            
            # Determine skin type compatibility status
            if user_skin_type and matched_types:
                if user_skin_type in matched_types:
                    skin_type_status = f"✅ Perfect match for {user_skin_type} skin"
                else:
                    skin_type_status = f"⚪ Product for {', '.join(matched_types)} skin (user has {user_skin_type})"
            elif user_skin_type and not matched_types:
                skin_type_status = f"⚪ No skin type specified (neutral for {user_skin_type} skin)"
            else:
                skin_type_status = "⚪ No skin type information available"
            
            concern_matches = []
            direct_matches = []
            
            # Use the centralized normalization method
            normalized_user_concerns = self._normalize_user_input(user_concerns)
            
            # Check for direct matches using original concerns
            for concern in user_concerns:
                if concern.lower() in combined_text.lower():
                    direct_matches.append(concern)
            
            # Check for normalized concern matches 
            for i, concern in enumerate(normalized_user_concerns):
                if concern.lower() in [c.lower() for c in matched_concerns]:
                    # Use original user concern name for display
                    original_concern = user_concerns[i]
                    if original_concern not in concern_matches:  # Avoid duplicates
                        concern_matches.append(original_concern)
            
            # Get semantic matches
            semantic_score = self._calculate_semantic_concern_match(
                [c.lower() for c in user_concerns], combined_text, product_id
            )
            
            # Create simplified concern status message with semantic score
            all_matches = list(set(direct_matches + concern_matches))  # Combine and remove duplicates
            match_text = f"{', '.join(all_matches)}" if all_matches else "none"
            
            if all_matches:
                concern_status = f"✅ {concern_score:.1f} concern score: Found {match_text}; Semantic: {semantic_score:.2f}"
            else:
                concern_status = f"❌ {concern_score:.1f} concern score: No matches found; Semantic: {semantic_score:.2f}"
            
            return {
                "compatibility_score": compatibility_score,
                "concern_score": concern_score,
                "semantic_score": semantic_score,
                "final_ranking_score": (concern_score * 10.0) + (compatibility_score * 1.0),
                "compatibility": {
                    "skin_type_status": skin_type_status,
                    "concern_status": concern_status
                },
                "concern_matches": concern_matches,
                "direct_matches": direct_matches,
                "skin_type_match": user_skin_type in [t.lower() for t in matched_types],
                "price": product.get('price_usd', 0),
                "budget_range": self._budget_range(profile.get('budget', '')),
                "user_profile": profile,
                "product_addresses": matched_concerns
            }
        except Exception as e:
            return {"error": f"Debug failed: {str(e)}"}

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