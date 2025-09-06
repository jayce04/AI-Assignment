import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple
from surprise import dump
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')
#-------------------------------------------TAN YEN FANG ----------------------------------------------------
"""
Hybrid Skincare Recommendation System
=====================================

This system provides personalized skincare product suggestions 
based on user profiles, skin concerns, and rating history.

CORE ARCHITECTURE:
- Hybrid filtering approach combining content-based and collaborative filtering
- Dynamic weighting system that adapts based on user experience level
- Semantic analysis for intelligent concern matching
- Budget filtering and category diversity for balanced recommendations

DYNAMIC WEIGHTING STRATEGY:
The system intelligently adjusts the importance of content vs collaborative filtering:
- New Users (0 ratings): 70% Content + 30% Collaborative (rely more on skin profile)
- Few Ratings (<10 products): 60% Content + 40% Collaborative (balanced approach)  
- Many Ratings (≥10 products): 40% Content + 60% Collaborative (trust user history more)

CONTENT FILTERING COMPONENTS:
- Skin type compatibility analysis (dry, oily, combination, sensitive, normal)
- Advanced concern matching using keyword detection + semantic similarity
- Ingredient analysis for concern-specific recommendations
- Budget range filtering and price compatibility

COLLABORATIVE FILTERING COMPONENTS:
- SVD (Singular Value Decomposition) matrix factorization for rating prediction
- Skin type popularity analysis (what products work for similar skin types)
- User similarity calculations based on rating patterns
- Global rating trends and product popularity metrics

CONCERN PRIORITY RANKING:
Products are primarily ranked by how well they address user's specific skin concerns,
then refined using hybrid scores for optimal personalization.

CATEGORY DIVERSITY:
For larger recommendation sets (>8 products), the system ensures variety across
essential skincare categories (cleansers, serums, moisturizers, etc.) while
maintaining concern-based ranking priority.
"""

class EnhancedHybridRecommender:
    """
    Advanced Hybrid Recommendation System for Skincare Products
    
    This class implements a sophisticated recommendation engine that combines:
    1. Content-based filtering (skin type + concern matching)
    2. Collaborative filtering (user similarity + product popularity)
    3. Dynamic weighting based on user experience
    4. Semantic similarity for intelligent concern analysis
    """
    
    def __init__(self, train_path: str, products_path: str,
                 content_model_path: str, svd_model_path: str):
        """
        Initialize the hybrid recommendation system
        
        Args:
            train_path: Path to user rating data (CSV with user_id, product_id, rating)
            products_path: Path to product information (CSV with product details)
            content_model_path: Path to pre-trained product embeddings for content similarity
            svd_model_path: Path to pre-trained SVD model for collaborative filtering
        """
        print("HybridRecommender LOADED!")
        self.train_path = train_path
        self.products_path = products_path
        self.content_model_path = content_model_path
        self.svd_model_path = svd_model_path
        
        # Initialize core data structures
        self.prod_df = None                    # Product information DataFrame
        self.prod_embeds = None               # Pre-computed product embeddings
        self.svd_model = None                 # Collaborative filtering model
        self.global_avg = 3.0                # Global average rating fallback
        self.train_df = None                  # User rating history DataFrame
        self.user_history_cache = {}         # Cache user data for performance
        self.product_popularity = {}         # Product popularity metrics
        self.product_features = {}           # Pre-computed product features
        self.skin_profiles: Dict[str, dict] = {}  # User skin profiles storage

        # Load and initialize all models and data
        self._load_models()
        self._preload_data()

    # ----------------- LOAD MODELS & DATA -----------------
    def _load_models(self) -> None:
        """Load pre-trained models and embeddings for recommendation calculations"""
        # Load product data and embeddings for content-based filtering
        self.prod_df, self.prod_embeds = joblib.load(self.content_model_path)
        
        # Load trained SVD model for collaborative filtering
        _, self.svd_model = dump.load(self.svd_model_path)

        # Extract global average rating from trained model
        if hasattr(self.svd_model, 'trainset') and self.svd_model.trainset:
            self.global_avg = self.svd_model.trainset.global_mean

        # Create product ID to embedding index mapping for fast lookups
        self.product_id_to_idx = {str(pid): idx for idx, pid in enumerate(self.prod_df["product_id"])}
        
        # Pre-compute product features to avoid repeated calculations
        self.precompute_product_features()

    def precompute_product_features(self):
        """Pre-compute and cache product features for faster similarity calculations"""
        self.product_features = {}
        for _, row in self.prod_df.iterrows():
            product_id = str(row["product_id"])
            # Store essential product info for quick access during recommendations
            self.product_features[product_id] = {
                'brand': row["brand_name"],           # Product brand for filtering
                'category': row["tertiary_category"], # Product category for diversity
                'price': row["price_usd"] if pd.notna(row["price_usd"]) else 0,  # Price for budget filtering
                'embedding': self.prod_embeds[self.product_id_to_idx[product_id]]  # Vector for similarity
            }

    def _preload_data(self):
        """Load and cache training data and product information"""
        # Load USER RATINGS data for collaborative filtering (user_id -> product_id -> rating)
        self.train_df = pd.read_csv(self.train_path, usecols=["author_id", "product_id", "rating"])
        
        # Load PRODUCT DATA with overall product ratings and details
        self.prod_df = pd.read_csv(self.products_path)

        # Cache USER HISTORY for collaborative filtering
        user_groups = self.train_df.groupby("author_id")
        for user_id, group in user_groups:
            self.user_history_cache[str(user_id)] = {
                'rated_products': group["product_id"].astype(str).tolist(),
                # ⭐ TYPE 1: USER GIVEN RATINGS - Actual 1-5 star ratings users gave to products they tried
                'user_given_ratings': group["rating"].tolist(),
                # ⭐ TYPE 2: USER AVERAGE RATING - Each user's personal rating tendency (harsh vs generous rater)
                'user_avg_rating': group["rating"].mean()
            }

        # Cache PRODUCT POPULARITY for collaborative filtering
        self.product_popularity = self.train_df['product_id'].astype(str).value_counts().to_dict()

    # ----------------- HYBRID CORE ALGORITHMS -----------------
    
    def enhanced_content_similarity(self, target_product_id: str, user_rated_products: List[str]) -> float:
        """
        Calculate content-based similarity between target product and user's rated products
        
        Uses pre-computed embeddings to find products similar to what the user has tried.
        Higher similarity means the target product is more similar to products the user liked.
        
        Args:
            target_product_id: Product we want to calculate similarity for
            user_rated_products: List of product IDs the user has previously rated
            
        Returns:
            Float between 0-1 representing average similarity to user's product history
        """
        if target_product_id not in self.product_features or not user_rated_products:
            return 0.0

        # Get embedding vector for the target product
        target_embed = self.product_features[target_product_id]['embedding']
        similarities = []

        # Calculate cosine similarity with each product the user has rated
        for rated_pid in user_rated_products:
            if rated_pid in self.product_features:
                rated_embed = self.product_features[rated_pid]['embedding']
                # Cosine similarity: 1 = identical products, 0 = completely different
                cosine_sim = cosine_similarity([target_embed], [rated_embed])[0][0]
                similarities.append(cosine_sim)

        # Return average similarity across all user's rated products
        return np.mean(similarities) if similarities else 0.0

    def hybrid_predict(self, user_id: str, product_id: str,
                       content_weight: float = 0.4, collab_weight: float = 0.6) -> Tuple[float, float]:
        """
        Core hybrid prediction method combining collaborative and content-based filtering
        
        This method predicts what rating a user would give to a product by combining:
        1. SVD collaborative filtering (what similar users rated this product)
        2. Content-based similarity (how similar this product is to user's preferences)
        
        Args:
            user_id: User to predict for
            product_id: Product to predict rating for  
            content_weight: Weight for content-based component (0-1)
            collab_weight: Weight for collaborative component (0-1)
            
        Returns:
            Tuple of (predicted_rating, confidence_score)
        """
        user_id, product_id = str(user_id), str(product_id)

        # ===== COLLABORATIVE FILTERING: SVD Matrix Factorization =====
        try:
            svd_prediction = self.svd_model.predict(user_id, product_id)
            # ⭐ TYPE 3: COLLABORATIVE PREDICTED RATING - AI prediction of what user would rate this product (SVD algorithm)
            svd_pred = max(1.0, min(5.0, svd_prediction.est))  # Clamp to valid rating range
            # High confidence if prediction was possible, low if user/product was unknown
            svd_conf = 0.9 if not svd_prediction.details.get('was_impossible', False) else 0.4
        except:
            # Fallback to global average if SVD fails
            svd_pred, svd_conf = self.global_avg, 0.3

        # ===== CONTENT-BASED FILTERING: Similarity to User's History =====
        content_pred, content_conf = np.nan, 0.0
        if user_id in self.user_history_cache:
            rated_products = self.user_history_cache[user_id]['rated_products']
            # Need at least 2 rated products to calculate meaningful similarity
            if len(rated_products) >= 2 and product_id in self.product_id_to_idx:
                sim_score = self.enhanced_content_similarity(product_id, rated_products)
                if sim_score > 0.1:  # Only use if similarity is meaningful
                    # Conservative approach: slight adjustment to user's average rating
                    # ⭐ TYPE 2: USER AVERAGE RATING - Each user's personal rating tendency (harsh vs generous rater)
                    user_avg_rating = self.user_history_cache[user_id]['user_avg_rating']
                    # Scale similarity conservatively: 0.8 to 1.3 multiplier (max 30% boost)
                    content_pred = user_avg_rating * (0.8 + sim_score * 0.5)
                    content_conf = min(0.7, sim_score * 1.2)  # Lower max confidence
                    content_pred = max(1.0, min(5.0, content_pred))  # Clamp to valid range

        # ===== DYNAMIC WEIGHTING BASED ON USER EXPERIENCE =====
        predictions, confidences, weights = [], [], []
        user_data = self.user_history_cache.get(user_id, {})
        # Calculate experience ratio: more ratings = trust collaborative more
        ratio = min(1.0, len(user_data.get('rated_products', [])) / 30)

        if not np.isnan(svd_pred):
            predictions.append(svd_pred)
            confidences.append(svd_conf)
            weights.append(collab_weight * (0.4 + 0.6 * ratio))
        if not np.isnan(content_pred) and content_conf > 0.2:
            predictions.append(content_pred)
            confidences.append(content_conf)
            weights.append(content_weight * (1.0 - 0.6 * ratio))

        # NEW: For NEW users only (not in training data), add light collaborative component
        if len(predictions) == 0 and user_id not in self.user_history_cache and user_id in self.skin_profiles:
            # Get user's skin type from profile
            user_skin_type = self.skin_profiles[user_id].get('skin_type', '')
            if user_skin_type:
                # Light collaborative: popularity among similar skin types
                collab_score, collab_conf = self._get_skin_type_popularity(user_skin_type, product_id)
                # Content-based score
                content_score, content_conf = self._content_based_predict(user_id, product_id)
                
                # Combine: 70% content, 30% collaborative (light weight for new users)
                if content_conf > 0.2:
                    weighted_pred = 0.7 * content_score + 0.3 * collab_score
                    final_conf = 0.7 * content_conf + 0.3 * collab_conf
                else:
                    weighted_pred = collab_score
                    final_conf = collab_conf
                    
                return max(1.0, min(5.0, weighted_pred)), final_conf

        if len(predictions) == 2:
            total_conf = sum(c * w for c, w in zip(confidences, weights))
            weighted_pred = sum(p * c * w for p, c, w in zip(predictions, confidences, weights)) / total_conf
            final_conf = total_conf / sum(weights)
        elif len(predictions) == 1:
            weighted_pred, final_conf = predictions[0], confidences[0]
        else:
            # Fallback for existing users with no good predictions
            # ⭐ TYPE 2: USER AVERAGE RATING - Each user's personal rating tendency (harsh vs generous rater)
            weighted_pred = user_data.get('user_avg_rating', self.global_avg) + np.random.uniform(-0.2, 0.2)
            weighted_pred = max(1.0, min(5.0, weighted_pred))
            final_conf = 0.2

        return max(1.0, min(5.0, weighted_pred)), final_conf

    def get_recommendations_for_new_user(self, skin_type: str, concerns: list, 
                                   budget: str, top_n: int = 10) -> List[Tuple[str, float, int]]:
        """
        Generate personalized recommendations for new users with no rating history
        
        NEW USER STRATEGY (70% Content + 30% Collaborative):
        Since new users have no rating history, we rely heavily on their skin profile
        and supplement with light collaborative signals from similar skin types.
        
        ALGORITHM FLOW:
        1. Filter products by budget constraints
        2. For each product, calculate:
           - Content compatibility (skin type + concern matching)
           - Light collaborative score (popularity among similar skin types)
           - Semantic concern analysis using ingredient matching
        3. Combine scores using 70/30 weighting (content-heavy for new users)
        4. Rank by concern priority (products addressing user's concerns ranked higher)
        5. Apply category diversity for balanced skincare routine
        
        Args:
            skin_type: User's skin type (dry, oily, combination, sensitive, normal)
            concerns: List of skin concerns (acne, aging, dryness, etc.)
            budget: Budget range string ("Under $25", "$25-$50", etc.)
            top_n: Number of recommendations to return
            
        Returns:
            List of tuples: (product_id, product_quality_rating, display_rating)
        """
        print(f"🎯 NEW USER Recommendations for: {skin_type} skin, concerns: {concerns}")
        
        # Parse budget constraints for filtering
        min_budget, max_budget = self._budget_range(budget)
        print(f"💰 Budget filter: ${min_budget} - ${max_budget}")
        
        recommendations = []
        processed_count = 0
        
        # Storage for algorithm metrics (used in diversity calculations)
        original_ratings = {}        # Product quality ratings for display
        collaborative_scores = {}   # Collaborative scores for diversity weighting
        
        # === BUDGET PRE-FILTERING FOR PERFORMANCE ===
        if budget and budget != "":
            budget_filtered_df = self.prod_df[
                (self.prod_df['price_usd'].isna()) |  # Include products with no price data
                ((self.prod_df['price_usd'] >= min_budget) & (self.prod_df['price_usd'] <= max_budget))
            ]
            print(f"📊 Budget pre-filtering: {len(budget_filtered_df)} products within budget")
        else:
            budget_filtered_df = self.prod_df
        
        # === MAIN RECOMMENDATION LOOP ===
        for _, product in budget_filtered_df.iterrows():
            product_id = str(product["product_id"])
            processed_count += 1
            
            # Progress tracking for large datasets
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
                # 🎯 ENHANCED HYBRID SCORING: Content + Light Collaborative
                
                # 1. Content-based compatibility score
                compatibility_score = self.filter_by_skin_profile(product_id, temp_user_id)
                
                # Skip products with low compatibility (less than 1.0 means filtered out)
                if compatibility_score < 1.0:
                    continue
                
                # 2. Light collaborative filtering: skin type popularity
                collab_score, collab_conf = self._get_skin_type_popularity(skin_type, product_id)
                
                # 3. Advanced concern matching
                concern_score = self._calculate_accurate_concern_score(product_id, temp_user_id, concerns)
                
                # 4. TRADITIONAL HYBRID SCORE CALCULATION
                # Pure Hybrid: 70% content + 30% collaborative (for new users)
                content_weight = 0.70
                collab_weight = 0.30
                
                # Content score already includes concern matching via compatibility_score
                # (because filter_by_skin_profile includes concern matching)
                content_score = compatibility_score * 2.5  # Scale 0.3-2.0 → 0.75-5.0
                
                # Boost content score based on concern relevance (this enhances content, not separate)
                if concern_score > 0:
                    concern_boost = min(1.5, 1.0 + (concern_score * 0.15))  # Up to 50% boost
                    content_score *= concern_boost
                
                # ⭐ TYPE 5: HYBRID RANKING SCORE - Internal algorithmic score for ranking products (hidden from users)
                final_hybrid_score = (
                    content_score * content_weight +
                    collab_score * collab_weight
                )
                
                # ⭐ TYPE 4: PRODUCT QUALITY RATING - Overall product rating displayed to users (⭐ shown in UI)
                product_quality_rating = float(product.get('rating', 3.5))
                
                # Store for diversity calculation
                original_ratings[product_id] = product_quality_rating
                collaborative_scores[product_id] = collab_score
                
                # Add to recommendations: (product_id, hybrid_ranking_score, display_rating)
                recommendations.append((product_id, final_hybrid_score, product_quality_rating))
                
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
        
        print(f"📊 Found {len(recommendations)} matching products with hybrid scoring")
        
        # Sort by hybrid score (final_hybrid_score) - CONCERN PRIORITY!
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # 🎯 PRIORITIZE CONCERN SCORES - Apply diversity only for larger requests
        if top_n > 8:  # Only apply diversity for larger requests (>8 products)
            diverse_recommendations = self._apply_enhanced_category_diversity(
                recommendations, top_n, original_ratings, collaborative_scores
            )
        else:
            # For smaller requests (≤8), MAINTAIN CONCERN SCORE RANKING
            diverse_recommendations = recommendations[:top_n]
            print(f"🎯 Maintaining concern-score ranking for {top_n} products (no diversity shuffling)")
        
        # Convert to app format: (product_id, display_rating, display_rating) 
        # Note: App expects 3 values but we only use the first 2
        final_recommendations = [
            (rec[0], rec[2], rec[2])  # (product_id, quality_rating_for_display, unused_duplicate)
            for rec in diverse_recommendations
        ]
        
        print(f"✅ Returning {len(final_recommendations)} diverse, high-quality recommendations")
        return final_recommendations
    
    def _apply_enhanced_category_diversity(self, recommendations: List[Tuple[str, float, int]], 
                                         top_n: int, original_ratings: dict, collaborative_scores: dict) -> List[Tuple[str, float, int]]:
        """Enhanced category diversity with collaborative scoring consideration"""
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
                
            for product_id, hybrid_score, match_percent in recommendations:
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
                        selected.append((product_id, hybrid_score, match_percent))
                        category_count[category] = category_count.get(category, 0) + 1
                        break
        
        # Second pass: Fill remaining slots with best remaining products (max 2 per category)
        # Prioritize products with higher collaborative scores for diversity
        remaining_recs = [(pid, score, match) for pid, score, match in recommendations 
                         if pid not in [p[0] for p in selected]]
        
        # Sort remaining by combination of hybrid score and collaborative score
        remaining_recs.sort(key=lambda x: (
            x[1] + collaborative_scores.get(x[0], 0) * 0.2  # Small boost for popular products
        ), reverse=True)
        
        for product_id, hybrid_score, match_percent in remaining_recs:
            if len(selected) >= top_n:
                break
                
            product = self.prod_df[self.prod_df["product_id"].astype(str) == product_id]
            if not product.empty:
                product_category = product.iloc[0]["tertiary_category"]
                current_count = category_count.get(product_category, 0)
                
                # Allow max 2 products per category for larger requests
                max_per_category = 2 if top_n > 8 else 1
                if current_count < max_per_category:
                    selected.append((product_id, hybrid_score, match_percent))
                    category_count[product_category] = current_count + 1
        
        return selected

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
        
        # DEBUG: For specific problematic products, show calculation details
        if product_id in ["21479"] or total_concern_score == 0.0:  # ELEVATE Retinol Serum had 0.0
            print(f"🔍 DEBUG Product {product_id}:")
            print(f"   User concerns: {user_concerns} → normalized: {normalized_user_concerns}")
            print(f"   Matched concerns: {matched_concerns}")
            print(f"   Keyword matches: {keyword_matches}")
            print(f"   Semantic score: {semantic_concern_score:.2f}")
            print(f"   Total score: {total_concern_score:.2f}")
        
        return total_concern_score

    def enhanced_demo_recommendations(self, user_id: str, top_n: int = 5,
                                 content_weight: float = 0.4, collab_weight: float = 0.6,
                                 selected_product_id: str = None):
        """Enhanced demo recommendations with unified hybrid approach"""
        user_id = str(user_id)
        user_exists = user_id in self.user_history_cache
        
        # ✅ SIMPLIFIED: Always use the main generate_recommendations method
        # which now handles both new and existing users properly with improved scoring
        print(f"🎯 Using unified hybrid approach for user: {user_id}")
        print(f"   User exists in training data: {user_exists}")
        print(f"   Weights: Content={content_weight}, Collaborative={collab_weight}")
        
        return self.generate_recommendations(user_id, top_n, content_weight, collab_weight)
    
    # ----------------- RECOMMENDATION -----------------
    def generate_recommendations(self, user_id: str, top_n: int = 10,
                             content_weight: float = 0.4, collab_weight: float = 0.6) -> List[Tuple[str, float, int]]:
        """
        Generate personalized recommendations for existing users with rating history
        
        EXISTING USER STRATEGY (40% Content + 60% Collaborative):
        Users with rating history get more collaborative filtering weight since we can
        trust their past preferences and find similar users with similar tastes.
        
        ALGORITHM FLOW:
        1. Filter products by budget constraints  
        2. Exclude products the user has already rated (avoid duplicates)
        3. For each remaining product, calculate:
           - Content compatibility (skin type + concern matching)
           - Collaborative prediction using SVD and user similarity  
           - Advanced concern scoring with ingredient analysis
        4. Combine scores using dynamic weighting (more collaborative for experienced users)
        5. Apply concern-weighted ranking (concern relevance * hybrid score)
        6. Apply category diversity for balanced recommendations
        
        Args:
            user_id: Existing user ID from training data
            top_n: Number of recommendations to return
            content_weight: Weight for content-based component (default: 40%)
            collab_weight: Weight for collaborative component (default: 60%)
            
        Returns:
            List of tuples: (product_id, product_quality_rating, display_rating)
        """
        user_id = str(user_id)
        user_rated = self.user_history_cache.get(user_id, {}).get('rated_products', [])
        
        # Retrieve user profile (skin type, concerns, budget)
        profile = self.skin_profiles.get(user_id, {})
        skin_type = profile.get('skin_type', '')
        concerns = profile.get('concerns', [])
        budget = profile.get('budget', '')
        
        print(f"🎯 EXISTING USER Recommendations for: {skin_type} skin, concerns: {concerns}")
        
        # === BUDGET FILTERING (same as new users) ===
        min_budget, max_budget = self._budget_range(budget)
        print(f"💰 Budget filter: ${min_budget} - ${max_budget}")
        
        if budget and budget != "":
            budget_filtered_df = self.prod_df[
                (self.prod_df['price_usd'].isna()) |  # Include products with no price data
                ((self.prod_df['price_usd'] >= min_budget) & (self.prod_df['price_usd'] <= max_budget))
            ]
            print(f"📊 Budget pre-filtering: {len(budget_filtered_df)} products within budget")
        else:
            budget_filtered_df = self.prod_df
        
        # === EXCLUDE ALREADY RATED PRODUCTS ===
        # Don't recommend products the user has already tried
        candidate_products = budget_filtered_df[
            ~budget_filtered_df["product_id"].astype(str).isin(user_rated)
        ]
        
        if candidate_products.empty:
            return self._get_popular_fallback(top_n)
        
        recommendations = []
        original_ratings = {}
        collaborative_scores = {}
        processed_count = 0
        
        for _, product in candidate_products.iterrows():
            product_id = str(product["product_id"])
            processed_count += 1
            
            # Show progress for large datasets
            if processed_count % 500 == 0:
                print(f"⚡ Processed {processed_count}/{len(candidate_products)} products...")
            
            try:
                # 🎯 SAME HYBRID SCORING AS NEW USERS, just different weights
                
                # 1. Content-based compatibility score (same as new users)
                compatibility_score = self.filter_by_skin_profile(product_id, user_id)
                
                # Skip products with low compatibility (same threshold as new users)
                if compatibility_score < 1.0:
                    continue
                
                # 2. Collaborative filtering: USE ACTUAL USER HISTORY (difference from new users)
                collab_score, collab_conf = self.hybrid_predict(user_id, product_id, 0.0, 1.0)
                
                # 3. Advanced concern matching (same as new users) - ENSURE PROPER NORMALIZATION
                user_concerns_for_calc = concerns if isinstance(concerns, list) else [concerns] if concerns else []
                concern_score = self._calculate_accurate_concern_score(product_id, user_id, user_concerns_for_calc)
                
                # 4. CONCERN-PRIORITY HYBRID SCORE CALCULATION 
                # For existing users: Prioritize concern matching more heavily
                content_score = compatibility_score * 2.5  # Same scaling as new users
                
                # Enhanced concern boost for existing users (they know what they want)
                if concern_score > 0:
                    concern_boost = min(2.0, 1.0 + (concern_score * 0.25))  # Up to 100% boost
                    content_score *= concern_boost
                
                # Create a concern-weighted hybrid score
                concern_weight_factor = 1.0 + (concern_score * 0.5)  # Extra weight for high concern matches
                
                # ⭐ TYPE 5: HYBRID RANKING SCORE - Internal algorithmic score for ranking products (hidden from users)
                final_hybrid_score = (
                    (content_score * content_weight + collab_score * collab_weight) * concern_weight_factor
                )
                
                # ⭐ TYPE 4: PRODUCT QUALITY RATING - Overall product rating displayed to users (⭐ shown in UI)
                product_quality_rating = float(product.get('rating', 3.5))
                
                # Store for diversity calculation (same as new users)
                original_ratings[product_id] = product_quality_rating
                collaborative_scores[product_id] = collab_score
                
                # Add to recommendations: (product_id, hybrid_ranking_score, display_rating, concern_score_for_debug)
                recommendations.append((product_id, final_hybrid_score, product_quality_rating, concern_score))
                
            except Exception as e:
                # Log errors but continue processing (same as new users)
                if processed_count <= 10:
                    print(f"⚠️  Error processing product {product_id}: {str(e)[:100]}...")
                continue
        
        print(f"📊 Found {len(recommendations)} matching products with hybrid scoring")
        
        # 🎯 SORT BY CONCERN-WEIGHTED HYBRID SCORE (highest first) - CONCERN PRIORITY!
        # Show debug info for top products
        recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by final_hybrid_score
        
        print(f"🏆 Top 5 Concern-Priority Rankings:")
        for i, (pid, hybrid_score, match_percent, concern) in enumerate(recommendations[:5], 1):
            print(f"   {i}. Product {pid}: Concern={concern:.1f}, Hybrid={hybrid_score:.2f}, Match={match_percent}%")
        
        # 🎯 PRIORITIZE CONCERN SCORES - Apply minimal diversity only for larger requests
        if top_n > 8:  # Only apply diversity for larger requests (>8 products)
            # Need to adjust the diversity function to handle the new tuple format
            simple_recs = [(pid, hybrid_score, match_percent) for pid, hybrid_score, match_percent, _ in recommendations]
            diverse_recommendations = self._apply_enhanced_category_diversity(
                simple_recs, top_n, original_ratings, collaborative_scores
            )
        else:
            # For smaller requests (≤8), MAINTAIN CONCERN SCORE RANKING
            diverse_recommendations = [(rec[0], rec[1], rec[2]) for rec in recommendations[:top_n]]
            print(f"🎯 Maintaining concern-score ranking for {top_n} products (no diversity shuffling)")
        
        # Convert to same format as new users: (product_id, original_rating, rating)
        final_recommendations = [
            (rec[0], rec[2], rec[2])  # (product_id, rating, rating) - third value no longer used
            for rec in diverse_recommendations
        ]
        
        print(f"✅ Returning {len(final_recommendations)} diverse, high-quality recommendations")
        return final_recommendations

    def _get_skin_type_popularity(self, skin_type: str, product_id: str) -> Tuple[float, float]:
        """Get popularity score for a product among users with similar skin type"""
        if product_id not in self.product_id_to_idx:
            return self.global_avg, 0.3
        
        # Find users with similar skin type from skin profiles
        similar_skin_users = []
        for user_id, profile in self.skin_profiles.items():
            if profile.get('skin_type', '').lower() == skin_type.lower():
                similar_skin_users.append(user_id)
        
        if not similar_skin_users:
            # Fallback to general popularity
            popularity_count = self.product_popularity.get(product_id, 0)
            popularity_score = min(5.0, popularity_count / 100 + 2.5)
            return popularity_score, 0.4
        
        # Get ratings for this product from similar skin type users
        similar_ratings = []
        for user_id in similar_skin_users:
            if user_id in self.user_history_cache:
                user_data = self.user_history_cache[user_id]
                if product_id in user_data['rated_products']:
                    # Find the rating for this specific product
                    product_idx = user_data['rated_products'].index(product_id)
                    rating = user_data['ratings'][product_idx]
                    similar_ratings.append(rating)
        
        if similar_ratings:
            avg_rating = np.mean(similar_ratings)
            confidence = min(0.8, len(similar_ratings) / 10)  # More ratings = more confidence
            return avg_rating, confidence
        else:
            # No direct ratings, use popularity-based score
            popularity_count = self.product_popularity.get(product_id, 0)
            popularity_score = min(5.0, popularity_count / 100 + 2.5)
            return popularity_score, 0.3

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

    # ----------------- SKIN PROFILE & CONTENT FILTERING -----------------
    
    def add_skin_profile(self, user_id: str, profile: dict):
        """
        Store user's skin profile for personalized recommendations
        
        Args:
            user_id: Unique identifier for the user
            profile: Dictionary containing skin_type, concerns, and budget
        """
        self.skin_profiles[str(user_id)] = profile

    def filter_by_skin_profile(self, product_id: str, user_id: str) -> float:
        """
        Calculate content-based compatibility between product and user's skin profile
        
        This is the core content filtering method that determines how well a product
        matches the user's skin type, concerns, and budget preferences.
        
        SCORING SYSTEM:
        - Base score: 1.0 (neutral)
        - Skin type match: +40% bonus, mismatch: -30% penalty
        - Concern match: +15% per matched concern (up to 3 concerns)
        - Budget match: +10% if within budget, -30% if outside budget
        - Final range: 0.3 to 2.0 (honest scoring, no inflation)
        
        Args:
            product_id: Product to evaluate compatibility for
            user_id: User whose profile to match against
            
        Returns:
            Float multiplier between 0.3-2.0 representing compatibility strength
        """
        profile = self.skin_profiles.get(str(user_id))
        if not profile:
            return 1.0  # Neutral score if no profile available

        # Extract user preferences from profile
        user_type = profile.get("skin_type", "").lower()
        user_concerns = profile.get("concerns", [])
        if isinstance(user_concerns, str):
            user_concerns = [user_concerns]  # Ensure list format
        
        # Normalize user concerns for better matching consistency
        normalized_user_concerns = self._normalize_user_input(user_concerns)
        user_budget = profile.get("budget", "")

        # Get product information and create searchable text
        product = self.prod_df[self.prod_df["product_id"].astype(str) == str(product_id)].iloc[0]
        
        # Use pre-processed combined features if available, otherwise build manually
        if 'combined_features' in self.prod_df.columns and pd.notna(product.get("combined_features")):
            product_text = str(product.get("combined_features", ""))
        else:
            # Combine all product text fields for comprehensive analysis
            product_text = " ".join(map(str, [
                product.get("product_name", ""),
                product.get("highlights", ""),      # Often contains "Best for Dry, Combo, Normal Skin"
                product.get("ingredients", ""),     # Ingredient list for concern matching
                product.get("combined_features", ""),  # Additional features
                product.get("claims", "")           # Product claims and benefits
            ]))
        
        price = product.get("price_usd", 0)

        # Extract skin types and concerns mentioned in product description
        matched_types, matched_concerns = self._extract_skin_tags(product_text)
        
        # Calculate semantic similarity for advanced concern matching
        semantic_concern_score = self._calculate_semantic_concern_match(
            normalized_user_concerns, product_text, product_id
        )

        # Start with neutral compatibility score
        multiplier = 1.0
        
        # === SKIN TYPE COMPATIBILITY ANALYSIS ===
        if user_type and matched_types:
            if user_type in matched_types:
                multiplier *= 1.4  # Perfect skin type match bonus (+40%)
            elif "combination" in matched_types or user_type == "combination":
                # Combination skin is versatile, works with most products
                multiplier *= 1.1  # Slight bonus for versatile products (+10%)
            elif (user_type == "dry" and "oily" in matched_types) or \
                 (user_type == "oily" and "dry" in matched_types):
                # Opposite skin types - significant penalty
                multiplier *= 0.3  # Strong penalty for incompatible skin types (-70%)
            else:
                multiplier *= 0.7  # Moderate penalty for other mismatches (-30%)
        elif user_type and not matched_types:
            # Product doesn't specify skin type - treat as universal/neutral
            multiplier *= 1.0  # No penalty for universal products

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
        """
        Advanced semantic matching between user concerns and product ingredients/descriptions
        
        This method goes beyond simple keyword matching to understand the relationship
        between skin concerns and the ingredients/technologies that address them.
        
        SEMANTIC INTELLIGENCE:
        - Maps each concern to relevant ingredients (e.g., 'acne' → 'salicylic acid', 'niacinamide')  
        - Considers skin type compatibility (gentle ingredients for sensitive skin)
        - Accounts for ingredient synergies and combinations
        - Provides nuanced scoring based on concern complexity
        
        EXAMPLE MAPPINGS:
        - Acne → salicylic acid, benzoyl peroxide, tea tree, zinc, niacinamide
        - Aging → retinol, peptides, vitamin C, bakuchiol, collagen  
        - Dryness → hyaluronic acid, ceramides, squalane, glycerin
        - Hyperpigmentation → vitamin C, arbutin, kojic acid, tranexamic acid
        
        Args:
            user_concerns: List of normalized user skin concerns
            product_text: Complete product description text to analyze
            product_id: Product ID for debugging (optional)
            
        Returns:
            Float score 0-1 representing semantic relevance to user's concerns
        """
        if not user_concerns:
            return 0.0
        
        try:
            # Use centralized concern normalization for consistency
            normalized_concerns = self._normalize_user_input(user_concerns)
            
            # CONCERN-TO-INGREDIENT KNOWLEDGE BASE
            # Maps skin concerns to ingredients/technologies that address them
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
                'pores': ['salicylic acid', 'clay', 'charcoal', 'niacinamide', 'bha', 'pore-minimizing', 'refining', 
                         'pore-clearing', 'blackhead', 'whitehead'],
                'oil-control': ['clay', 'charcoal', 'zinc', 'mattifying', 'oil control', 'sebum control', 'shine control'],
                'dullness': ['vitamin c', 'aha', 'glycolic', 'brightening', 'glow', 'radiance', 'luminous', 'revitalizing'],
                'texture': ['aha', 'glycolic', 'lactic', 'resurfacing', 'exfoliating', 'smoothing', 'refining']
            }
            
            text_lower = product_text.lower()
            concern_scores = []
            
            # Calculate semantic score for each user concern
            for concern in normalized_concerns:
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



# ----------------- YAP ZI WEN  -----------------
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

# -------------------------------------- CHANG KAR YAN ---------------------------------------
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