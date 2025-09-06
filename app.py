import streamlit as st
import pandas as pd
import numpy as np
from utils.recommender import EnhancedHybridRecommender, CollaborativeRecommender
import plotly.express as px
from streamlit_option_menu import option_menu
import os
import traceback
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Force UTF-8 encoding for the terminal
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Debug file existence
def debug_file_existence():
    """Check if all required files exist"""
    required_files = [
        "products_preprocessed.csv",
        "filtered_skincare_products.csv",
        "collaborative_training_data.csv",
        "models/product_embeddings.pkl",
        "models/surprise_svd_model.pkl",
        "svd_model.pkl",
        "trainset.pkl"
    ]
    
    print("=" * 50)
    print("DEBUGGING FILE EXISTENCE")
    print("=" * 50)
    
    for file_path in required_files:
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        print(f"{file_path}: {'EXISTS' if exists else 'MISSING'} ({size} bytes)")
    
    print("=" * 50)

# Run the debug function
debug_file_existence()

# 页面设置
st.set_page_config(
    page_title="Skincare Recommendation System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 初始化session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'skin_data' not in st.session_state:
    st.session_state.skin_data = {}
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'selected_product_category' not in st.session_state:
    st.session_state.selected_product_category = None

# 加载产品数据
@st.cache_data(show_spinner=True)
def load_products(path="products_preprocessed.csv", extra_path="filtered_skincare_products.csv"):
    try:
        df = pd.read_csv(path)
        # Ensure required columns exist
        for c in ["price_usd", "rating", "reviews"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["product_type"] = df.get("tertiary_category", "").fillna("").astype(str)
        df["skin_concern"] = df.get("skin_concern", "").fillna("").astype(str)
        df["skin_type"] = df.get("skin_type", "").fillna("").astype(str)
        # Ensure product_content exists
        if "product_content" not in df.columns:
            df["product_content"] = (df["product_type"] + " " + 
                                   df["skin_type"] + " " + 
                                   df["skin_concern"]).str.strip()
        
        # Load filtered_skincare_products.csv
        try:
            df_extra = pd.read_csv(extra_path)
            # Select relevant columns
            df_extra = df_extra[["product_id", "size", "highlights", "ingredient"]].copy()
            # Rename 'ingredient' to 'ingredients' if needed
            if "ingredient" in df_extra.columns:
                df_extra = df_extra.rename(columns={"ingredient": "ingredients"})
            # Merge with main DataFrame
            df = df.merge(df_extra, on="product_id", how="left")
            # Fill missing values for new columns
            for c in ["size", "highlights", "ingredients"]:
                df[c] = df[c].fillna("Not specified")
        except Exception as e:
            print(f"Error loading filtered_skincare_products.csv: {e}")
            # Add fallback columns if merge fails
            for c in ["size", "highlights", "ingredients"]:
                df[c] = "Not specified"
        
        print("✅ Product data loaded successfully")
        return df
    except Exception as e:
        print(f"Error loading products: {e}")
        return pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006'],
            'product_name': ['Moisturizing Cream', 'Cleansing Gel', 'Anti-Aging Serum', 
                           'Sunscreen SPF 50', 'Hydrating Toner', 'Acne Treatment'],
            'brand_name': ['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E', 'Brand F'],
            'tertiary_category': ['Moisturizers', 'Cleansers', 'Serums', 
                                'Sunscreens', 'Toners', 'Treatments'],
            'product_type': ['Moisturizers', 'Cleansers', 'Serums', 
                           'Sunscreens', 'Toners', 'Treatments'],
            'skin_type': ['', '', '', '', '', ''],
            'skin_concern': ['', '', '', '', '', ''],
            'product_content': ['Moisturizers', 'Cleansers', 'Serums', 
                              'Sunscreens', 'Toners', 'Treatments'],
            'price_usd': [25.99, 18.50, 32.75, 22.00, 15.99, 28.50],
            'size': ['Not specified'] * 6,
            'highlights': ['Not specified'] * 6,
            'ingredients': ['Not specified'] * 6
        })

# 构建TF-IDF向量化器和矩阵
@st.cache_resource(show_spinner=True)
def build_vectorizer_and_matrix(df: pd.DataFrame):
    try:
        # Use product_content for TF-IDF
        text_series = df.get("product_content", 
                           df.get("product_type", "") + " " + 
                           df.get("skin_type", "") + " " + 
                           df.get("skin_concern", "")).str.strip()
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(text_series.fillna(""))
        print("✅ TF-IDF vectorizer and matrix built successfully")
        return vectorizer, tfidf_matrix
    except Exception as e:
        print(f"Error building TF-IDF matrix: {e}")
        raise

products_df = load_products()
vectorizer, tfidf_matrix = build_vectorizer_and_matrix(products_df)

# Content-based recommender logic with tertiary_category filter
def contentbased_recommender(
    product_type=None,
    skin_type=None,
    skin_concern=None,
    concern_match="all",
    max_price=None,
    n=10
):
    def _to_set(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return set()
        if isinstance(x, (list, tuple, set)):
            return {str(t).strip().lower() for t in x if str(t).strip()}
        return {t.strip().lower() for t in re.split(r"[;,/|]", str(x)) if t.strip()}

    req_type = str(product_type).strip().lower() if product_type else None
    req_category = str(st.session_state.selected_product_category).strip().lower() if st.session_state.selected_product_category else None
    req_skin = str(skin_type).strip().lower() if skin_type else None
    req_concern = _to_set(skin_concern)

    tokens = []
    if req_type: tokens.append(req_type)
    if req_skin: tokens.append(req_skin)
    if req_concern: tokens.extend(sorted(req_concern))
    profile_text = " ".join(tokens).strip() or "skincare"

    qv = vectorizer.transform([profile_text])
    sims = cosine_similarity(qv, tfidf_matrix).ravel()

    price_col = pd.to_numeric(products_df.get("price_usd", np.nan), errors="coerce")
    rating_col = pd.to_numeric(products_df.get("rating", np.nan), errors="coerce").fillna(0.0)
    reviews_col = pd.to_numeric(products_df.get("reviews", 0), errors="coerce").fillna(0).astype(int)

    rows = []
    for i, sim in enumerate(sims):
        row = products_df.iloc[i]

        # Filter by tertiary_category if selected
        if req_category and str(row.get("tertiary_category", "")).strip().lower() != req_category:
            continue

        # Filter by product_type (already mapped from tertiary_category)
        if req_type and str(row.get("product_type", "")).strip().lower() != req_type:
            continue

        row_skin = str(row.get("skin_type", "")).strip().lower()
        if req_skin and row_skin and row_skin != req_skin:
            continue

        row_concern = _to_set(row.get("skin_concern", ""))
        if req_concern:
            if concern_match == "all":
                if not req_concern.issubset(row_concern):
                    continue
            else:
                if row_concern.isdisjoint(req_concern):
                    continue

        p = price_col.iat[i]
        if max_price is not None and (pd.isna(p) or p > float(max_price)):
            continue

        rows.append({
            "product_id": str(row.get("product_id", "")),
            "product_name": row.get("product_name", ""),
            "brand_name": row.get("brand_name", ""),
            "product_type": row.get("product_type", ""),
            "tertiary_category": row.get("tertiary_category", ""),
            "skin_type": row.get("skin_type", ""),
            "skin_concern": row.get("skin_concern", ""),
            "price_usd": row.get("price_usd", ""),
            "rating": rating_col.iat[i],
            "reviews": reviews_col.iat[i],
            "similarity": float(sim),
            "size": row.get("size", "Not specified"),
            "highlights": row.get("highlights", "Not specified"),
            "ingredients": row.get("ingredients", "Not specified")
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame()

    out = out.sort_values(
        by=["similarity", "rating", "reviews"],
        ascending=[False, False, False]
    ).head(n)

    out["similarity"] = out["similarity"].round(4)
    return out

# 初始化推荐系统
@st.cache_resource
def load_recommenders():
    print("Starting to load recommenders...")
    
    hybrid_rec = None
    collab_rec = None
    
    try:
        print("Loading hybrid recommender...")
        hybrid_rec = EnhancedHybridRecommender(
            train_path="collaborative_training_data.csv",
            products_path="products_preprocessed.csv",
            content_model_path="models/product_embeddings.pkl",
            svd_model_path="models/surprise_svd_model.pkl"
        )
        print("✅ Hybrid recommender loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading hybrid recommender: {e}")
        traceback.print_exc()
    
    try:
        print("Loading collaborative recommender...")
        collab_rec = CollaborativeRecommender("collaborative_training_data.csv")
        print(f"CollaborativeRecommender df shape: {collab_rec.df.shape if collab_rec.df is not None else 'None'}")
        print("✅ Collaborative recommender loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading collaborative recommender: {e}")
        traceback.print_exc()
    
    return hybrid_rec, collab_rec

hybrid_rec, collab_rec = load_recommenders()

# 动态提取所有唯一皮肤问题
def all_concerns_unique(df):
    s = df["skin_concern"].fillna("").astype(str)
    uniq = set()
    for txt in s:
        for t in re.split(r"[;,/|]", txt):
            t = t.strip().lower()
            if t:
                uniq.add(t)
    return sorted(uniq)

# 辅助函数定义
def display_recommendation(index, product, rating, similarity=None):
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(f"{index}. {product.get('product_name', 'Unknown Product')}")
            st.write(f"**Brand:** {product.get('brand_name', 'Unknown')}")
            st.write(f"**Category:** {product.get('tertiary_category', 'Unknown')}")
            st.write(f"**Product ID:** {product.get('product_id', 'N/A')}")
        
        with col2:
            st.metric("Rating" if st.session_state.get('selected_model') == 'content' else "Predicted Rating", 
                     f"{rating:.1f}/5")
            if similarity is not None:
                match_percent = round(min(100, max(0, similarity * 100)))
                st.progress(match_percent / 100, text=f"{match_percent}% match")
            st.write(f"**Price:** ${product.get('price_usd', 0):.2f}")
        
        with col3:
            if st.button("Details", key=f"rec_btn_{index}"):
                st.write("**Full Product Info:**")
                st.write(f"**Skin Concern:** {product.get('skin_concern', 'Not specified')}")
                st.write(f"**Reviews:** {product.get('reviews', 'Not specified')}")
                st.json(product)
        
        st.divider()

def display_product_card(product, col):
    with col:
        card = st.container(border=True)
        with card:
            st.subheader(product['product_name'])
            st.write(f"**Brand:** {product['brand_name']}")
            st.write(f"**Category:** {product['tertiary_category']}")
            st.write(f"**Price:** ${product['price_usd']}")
            
            if st.button("Select & Get Recommendations", key=f"select_{product['product_id']}", 
                        use_container_width=True):
                st.session_state.selected_product = product['product_id']
                st.session_state.selected_product_category = product['tertiary_category']
                st.session_state.current_page = 'skin analysis'
                st.rerun()

# CSS for full-width layout and reduced padding
st.markdown("""
    <style>
    .main .block-container {
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: 100% !important;
    }
    .stContainer {
        width: 100% !important;
        padding: 5px !important;
        margin: 0 !important;
    }
    .stButton>button {
        width: 100% !important;
        margin: 2px 0 !important;
    }
    .stApp {
        max-width: 100% !important;
        margin: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 页面路由逻辑
# Navigation bar at the top of each page
st.image("https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=100", width=80)
st.title("🌸 Skincare Recommender")
st.divider()

if st.session_state.current_page == 'home':
    st.header("Discover Your Perfect Skincare")
    st.subheader("Browse our curated collection or get personalized recommendations")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input("🔍 Search products", placeholder="Enter product name or brand")
    with col2:
        selected_category = st.selectbox("Filter by category", 
                                       ["All"] + list(products_df['tertiary_category'].unique()))
    
    filtered_products = products_df.copy()
    if search_query:
        filtered_products = filtered_products[
            filtered_products['product_name'].str.contains(search_query, case=False, na=False) |
            filtered_products['brand_name'].str.contains(search_query, case=False, na=False)
        ]
    if selected_category != "All":
        filtered_products = filtered_products[filtered_products['tertiary_category'] == selected_category]
    
    st.write(f"**Showing {len(filtered_products)} products**")
    
    if len(filtered_products) > 0:
        cols = st.columns(3)
        for idx, (_, product) in enumerate(filtered_products.iterrows()):
            display_product_card(product, cols[idx % 3])
    else:
        st.info("No products found. Try adjusting your search filters.")
    
    st.divider()
    st.write("### Not sure what to choose?")
    if st.button("✨ Get Personalized Recommendations Based on Your Skin Needs", 
                use_container_width=True, type="primary"):
        st.session_state.current_page = 'skin analysis'
        st.session_state.selected_product = None
        st.session_state.selected_product_category = None
        st.rerun()

elif st.session_state.current_page == 'skin analysis':
    st.header("Tell Us About Your Skin")
    
    if st.button("← Back to Products"):
        st.session_state.current_page = 'home'
        st.session_state.selected_product = None
        st.session_state.selected_product_category = None
        st.rerun()
    
    if st.session_state.selected_product:
        product_info = products_df[products_df['product_id'] == st.session_state.selected_product]
        if not product_info.empty:
            product_info = product_info.iloc[0]
            st.info(f"**Selected Product:** {product_info['product_name']} by {product_info['brand_name']} (Category: {st.session_state.selected_product_category})")
    
    with st.form("skin_analysis_form"):
        user_id = st.text_input("User ID", placeholder="Enter your user ID (optional)", 
                               help="Optional for content-based recommendations, required for collaborative")
        
        col1, col2 = st.columns(2)
        with col1:
            skin_type = st.selectbox("Skin Type", ["(any)", "Dry", "Oily", "Combination", "Normal", "Sensitive"],
                                   help="Select your primary skin type (optional)")
        with col2:
            budget = st.selectbox("Budget Preference", ["(any)", "Under $25", "$25-$50", "$50-$100", "Over $100", "No budget limit"],
                                help="Your preferred price range (optional)")
        
        concerns = st.multiselect(
            "Main Skin Concerns",
            all_concerns_unique(products_df),
            help="Select all that apply to you (optional)"
        )
        
        concern_match = st.radio("Concern Match", ["all", "any"], index=1, 
                                help="Match all concerns or any concern (optional)")
        
        num_products = st.slider("Number of Recommendations", 1, 10, 5,
                               help="How many products would you like to see?")
        
        submitted = st.form_submit_button("🎯 Get Personalized Recommendations", type="primary")
        
        if submitted:
            st.session_state.skin_data = {
                'user_id': user_id if user_id else None,
                'skin_type': None if skin_type == "(any)" else skin_type,
                'concerns': concerns if concerns else None,
                'budget': None if budget == "(any)" else budget,
                'num_products': num_products,
                'product_type': st.session_state.selected_product_category,  # Use selected category
                'concern_match': concern_match
            }
            st.session_state.current_page = 'select approach'
            st.rerun()

elif st.session_state.current_page == 'select approach':
    st.header("Choose Your Recommendation Style")
    
    if st.button("← Back to Skin Analysis"):
        st.session_state.current_page = 'skin analysis'
        st.rerun()
    
    st.write("How would you like us to find your perfect skincare match?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Smart Matching", use_container_width=True, help="Based on product ingredients and features"):
            st.session_state.selected_model = 'content'
            st.session_state.current_page = 'recommendations'
            st.rerun()
        st.caption("AI-powered analysis of product ingredients and features")
    
    with col2:
        if st.button("👥 Community Wisdom", use_container_width=True, help="From users with similar skin profiles"):
            st.session_state.selected_model = 'collab' 
            st.session_state.current_page = 'recommendations'
            st.rerun()
        st.caption("Recommendations from users with similar skin concerns")
    
    with col3:
        if st.button("🌟 Best of Both", use_container_width=True, help="Combined AI and community insights"):
            st.session_state.selected_model = 'hybrid'
            st.session_state.current_page = 'recommendations'
            st.rerun()
        st.caption("Advanced AI combining both approaches for optimal results")
    
    with st.expander("ℹ️ Learn about our recommendation methods"):
        st.markdown("""
        **🤖 Smart Matching**  
        Uses artificial intelligence to analyze product ingredients, features, and your skin profile 
        to find scientifically-matched products.
        
        **👥 Community Wisdom**  
        Leverages the collective experience of thousands of users with similar skin types and concerns 
        to recommend proven favorites.
        
        **🌟 Best of Both**  
        Our most advanced approach combining AI analysis with community insights for the most 
        accurate and personalized recommendations.
        """)

elif st.session_state.current_page == 'recommendations':
    st.header("Your Personalized Skincare Recommendations")
    
    if not st.session_state.skin_data:
        st.warning("Please complete the skin analysis first")
        st.session_state.current_page = 'skin analysis'
        st.rerun()
    
    skin_data = st.session_state.skin_data
    model_type = st.session_state.selected_model
    
    # 显示用户输入
    with st.expander("Your Skin Profile"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if skin_data.get('user_id'):
                st.write(f"**User ID:** {skin_data['user_id']}")
            if skin_data.get('skin_type'):
                st.write(f"**Skin Type:** {skin_data['skin_type']}")
            if skin_data.get('product_type'):
                st.write(f"**Product Category:** {skin_data['product_type']}")
        with col2:
            if skin_data.get('budget'):
                st.write(f"**Budget:** {skin_data['budget']}")
            st.write(f"**Number of Products:** {skin_data['num_products']}")
            if skin_data.get('concern_match'):
                st.write(f"**Concern Match:** {skin_data['concern_match'].capitalize()}")
        with col3:
            if skin_data.get('concerns'):
                st.write(f"**Concerns:** {', '.join(skin_data['concerns']) if skin_data['concerns'] else 'None'}")
            st.write(f"**Model:** {model_type.capitalize()}")
    
    # 获取和显示推荐
    st.subheader("Recommended For You")
    
    if model_type == 'hybrid' and hybrid_rec:
        skin_profile_data = {
            'user_id': skin_data['user_id'],
            'skin_type': skin_data['skin_type'],
            'concerns': skin_data['concerns'],
            'budget': skin_data['budget'],
            'concern_match': skin_data['concern_match']
        }
        
        try:
            hybrid_rec.add_skin_profile(skin_data['user_id'], skin_profile_data)
            st.success("✅ Skin profile added successfully!")
        except Exception as e:
            st.error(f"❌ Error adding skin profile: {e}")
            st.write("Falling back to default recommendations without skin filtering")
        
        with st.spinner("Generating hybrid recommendations..."):
            try:
                recommendations = hybrid_rec.generate_recommendations(
                    skin_data['user_id'], 
                    skin_data['num_products']
                )
                
                if recommendations:
                    for i, (product_id, rating, match_percent) in enumerate(recommendations, 1):
                        product_info = products_df[products_df['product_id'].astype(str) == product_id]
                        if not product_info.empty:
                            product_info = product_info.iloc[0]
                            product_dict = product_info.to_dict()
                            # Ensure additional fields are included
                            product_dict.update({
                                'size': product_info.get('size', 'Not specified'),
                                'highlights': product_info.get('highlights', 'Not specified'),
                                'ingredients': product_info.get('ingredients', 'Not specified')
                            })
                            display_recommendation(i, product_dict, rating, match_percent / 100)
                else:
                    st.warning("No recommendations found. Try adjusting your skin profile.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                recommendations = hybrid_rec.generate_recommendations(
                    skin_data['user_id'], 
                    skin_data['num_products']
                )
    
    elif model_type == 'content':
        with st.spinner("🔍 Finding products that match your skin needs..."):
            try:
                max_price = None
                if skin_data['budget'] == "Under $25":
                    max_price = 25
                elif skin_data['budget'] == "$25-$50":
                    max_price = 50
                elif skin_data['budget'] == "$50-$100":
                    max_price = 100
                elif skin_data['budget'] == "Over $100":
                    max_price = float("inf")
                elif skin_data['budget'] == "No budget limit":
                    max_price = float("inf")
                
                recommendations = contentbased_recommender(
                    product_type=skin_data.get('product_type'),
                    skin_type=skin_data.get('skin_type'),
                    skin_concern=skin_data.get('concerns'),
                    concern_match=skin_data.get('concern_match', 'any'),
                    max_price=max_price,
                    n=skin_data['num_products']
                )
                
                if not recommendations.empty:
                    for i, rec in enumerate(recommendations.to_dict('records'), 1):
                        display_recommendation(i, rec, rec['rating'], rec['similarity'])
                    csv = recommendations.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Recommendations (CSV)", 
                        data=csv, 
                        file_name="content_recommendations.csv", 
                        mime="text/csv"
                    )
                else:
                    st.warning("No recommendations found. Try selecting fewer concerns, choosing 'any' for concern match, or leaving skin type and budget as '(any)'.")
            except Exception as e:
                st.error(f"Error generating content-based recommendations: {e}")
                with st.expander("Full Error Details"):
                    st.code(str(e))
                    st.code(traceback.format_exc())
    
    elif model_type == 'collab':
        if not collab_rec:
            st.error("Collaborative Recommender is not available. Please check the system status.")
            st.stop()
        
        with st.spinner("Finding community favorites for your skin type..."):
            if not skin_data.get('user_id'):
                st.error("Please enter a valid User ID.")
            else:
                user_id = str(skin_data['user_id']).strip()
                st.info(f"Searching for recommendations for user: '{user_id}'")
                
                with st.expander("Debug Information (Click to expand)"):
                    st.write(f"**Input User ID:** `{user_id}` (type: {type(user_id)})")
                    user_exists = collab_rec.check_user_exists(user_id)
                    st.write(f"**User exists in training data:** {user_exists}")
                    sample_users = collab_rec.get_available_users(20)
                    st.write(f"**Sample available user IDs ({len(sample_users)} shown):**")
                    st.write(sample_users)
                    if not user_exists:
                        st.warning("User not found in training data. The system will provide popular recommendations.")
                
                try:
                    profile, recommendations = collab_rec.get_user_profile_and_recommendations(
                        user_id, 
                        skin_data['num_products']
                    )
                    
                    if profile:
                        st.subheader("User Profile")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Reviews", profile.get('total_reviews', 0))
                            st.metric("Average Rating", f"{profile.get('avg_rating', 0):.2f}")
                        with col2:
                            st.write(f"**Skin Type:** {profile.get('skin_type', 'Unknown')}")
                            fav_brands = profile.get('favorite_brands', [])
                            st.write(f"**Favorite Brands:** {', '.join(fav_brands[:3]) if fav_brands else 'None'}")
                    else:
                        st.warning("No user profile could be generated.")
                    
                    if recommendations and len(recommendations) > 0:
                        st.subheader("Your Recommendations")
                        for i, rec in enumerate(recommendations, 1):
                            match_percent = (rec['predicted_rating'] / 5.0) * 100
                            # Ensure additional fields are included
                            product_info = products_df[products_df['product_id'].astype(str) == rec['product_id']]
                            if not product_info.empty:
                                rec.update({
                                    'size': product_info.iloc[0].get('size', 'Not specified'),
                                    'highlights': product_info.iloc[0].get('highlights', 'Not specified'),
                                    'ingredients': product_info.iloc[0].get('ingredients', 'Not specified')
                                })
                            display_recommendation(i, rec, rec['predicted_rating'], match_percent / 100)
                        
                        rec_df = pd.DataFrame(recommendations)
                        csv = rec_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Recommendations (CSV)", 
                            data=csv, 
                            file_name="collaborative_recommendations.csv", 
                            mime="text/csv"
                        )
                    else:
                        st.warning("No recommendations were generated.")
                        with st.expander("Troubleshooting"):
                            st.write("**Possible reasons:**")
                            st.write("1. User ID doesn't exist in training data")
                            st.write("2. User has rated all available products")
                            st.write("3. Model files are corrupted")
                            st.write("4. Training data is insufficient")
                            st.write("**Try these solutions:**")
                            st.write("- Use one of the sample user IDs shown above")
                            st.write("- Check if your model files exist and are not corrupted")
                            st.write("- Verify that your training data has sufficient records")
                            if sample_users:
                                st.write("**Quick test - try this user ID:**")
                                st.code(str(sample_users[0]))
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
                    with st.expander("Full Error Details"):
                        st.code(str(e))
                        st.code(traceback.format_exc())
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Get New Recommendations", use_container_width=True):
            st.session_state.current_page = 'select approach'
            st.rerun()
    with col2:
        if st.button("🏠 Start Over", use_container_width=True):
            st.session_state.current_page = 'home'
            st.session_state.selected_product = None
            st.session_state.selected_product_category = None
            st.session_state.skin_data = {}
            st.rerun()

elif st.session_state.current_page == 'about':
    st.header("About Skincare Recommender")
    
    if st.button("← Back to Home"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("""
    ## 🌸 Your Personal Skincare Assistant
    
    Our advanced recommendation system uses machine learning to help you discover 
    skincare products that are perfectly suited to your unique skin needs.
    
    ### How It Works
    1. **Browse Products**: Explore our curated collection of skincare products
    2. **Skin Analysis**: Tell us about your skin type, concerns, and preferences
    3. **Smart Matching**: Choose how you'd like us to find your perfect products
    4. **Personalized Recommendations**: Receive tailored suggestions just for you
    
    ### Our Recommendation Methods
    - **🤖 Smart Matching**: AI-powered analysis of product ingredients and features
    - **👥 Community Wisdom**: Recommendations from users with similar skin profiles  
    - **🌟 Best of Both**: Combined AI and community insights for optimal results
    
    ### Why Trust Us?
    - Scientifically-backed ingredient analysis
    - Real user reviews and experiences
    - Personalized based on your unique skin profile
    - No sponsored recommendations - we're here to help you find what really works
    """)
    
    # System status moved to About page
    with st.expander("System Status"):
        st.write(f"Hybrid Recommender: {'✅' if hybrid_rec else '❌'}")
        st.write(f"Collaborative Recommender: {'✅' if collab_rec else '❌'}")
        if collab_rec and collab_rec.df is not None:
            st.write(f"Training Records: {len(collab_rec.df)}")
            st.write(f"Unique Users: {collab_rec.df['author_id'].nunique()}")
        else:
            st.write("Training Data: ❌")
    
    st.divider()
    st.caption("Built with ❤️ using advanced machine learning algorithms")

if __name__ == "__main__":
    pass