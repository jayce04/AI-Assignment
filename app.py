import streamlit as st
import pandas as pd
import numpy as np
from utils.recommender import EnhancedHybridRecommender, CollaborativeRecommender
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from streamlit_option_menu import option_menu
import re

# ==================== APPLICATION CONFIGURATION ====================
st.set_page_config(
    page_title="Skincare Recommendation System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE INITIALIZATION ====================
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'skin_data' not in st.session_state:
    st.session_state.skin_data = {}

# ==================== DATA LOADING ====================
@st.cache_data(show_spinner=True)
def load_content_products(path="data/CleanedDataSet/products_preprocessed.csv"):
    try:
        df = pd.read_csv(path)
        for c in ["price_usd", "rating", "reviews"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["skin_concern"] = df.get("skin_concern", "").fillna("").astype(str)
        df["skin_type"] = df.get("skin_type", "").fillna("").astype(str)
        df["product_type"] = df.get("product_type", "").fillna("").astype(str)
        return df
    except:
        return pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003'],
            'product_name': ['Moisturizing Cream', 'Cleansing Gel', 'Anti-Aging Serum'],
            'brand_name': ['Brand A', 'Brand B', 'Brand C'],
            'tertiary_category': ['Moisturizers', 'Cleansers', 'Serums'],
            'price_usd': [25.99, 18.50, 32.75],
            'rating': [4.5, 4.0, 4.8],
            'reviews': [100, 80, 120],
            'skin_type': ['Dry', 'Oily', 'Normal'],
            'skin_concern': ['Dehydration', 'Acne', 'Aging'],
            'product_content': ['moisturizer dry dehydration', 'cleanser oily acne', 'serum normal aging']
        })

@st.cache_data(show_spinner=True)
def load_detail_products(path="data/CleanedDataSet/filtered_skincare_products.csv"):
    try:
        products_df = pd.read_csv(path)
        for c in ["price_usd", "rating", "reviews"]:
            if c in products_df.columns:
                products_df[c] = pd.to_numeric(products_df[c], errors="coerce")
        return products_df
    except:
        return pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003'],
            'product_name': ['Moisturizing Cream', 'Cleansing Gel', 'Anti-Aging Serum'],
            'brand_name': ['Brand A', 'Brand B', 'Brand C'],
            'tertiary_category': ['Moisturizers', 'Cleansers', 'Serums'],
            'price_usd': [25.99, 18.50, 32.75],
            'rating': [4.5, 4.0, 4.8],
            'reviews': [100, 80, 120],
            'ingredients': ['["Water", "Glycerin"]', '["Salicylic Acid"]', '["Retinol"]'],
            'highlights': ['Hydrates;Non-greasy', 'Deep cleansing', 'Reduces wrinkles']
        })

content_df = load_content_products("data/CleanedDataSet/products_preprocessed.csv")
detail_products_df = load_detail_products("data/CleanedDataSet/filtered_skincare_products.csv")

@st.cache_resource(show_spinner=True)
def build_vectorizer_and_matrix(product_text: pd.Series):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(product_text.fillna("").astype(str))
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_vectorizer_and_matrix(content_df["product_content"])

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
    req_skin = str(skin_type).strip().lower() if skin_type else None
    req_concern = _to_set(skin_concern)

    tokens = []
    if req_type: tokens.append(req_type)
    if req_skin: tokens.append(req_skin)
    if req_concern: tokens.extend(sorted(req_concern))
    profile_text = " ".join(tokens).strip() or "skincare"

    qv = vectorizer.transform([profile_text])
    sims = cosine_similarity(qv, tfidf_matrix).ravel()

    price_col = pd.to_numeric(content_df.get("price_usd", np.nan), errors="coerce")
    rating_col = pd.to_numeric(content_df.get("rating", np.nan), errors="coerce").fillna(0.0)
    reviews_col = pd.to_numeric(content_df.get("reviews", 0), errors="coerce").fillna(0).astype(int)

    rows = []
    for i, sim in enumerate(sims):
        row = content_df.iloc[i]

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
            "skin_type": row.get("skin_type", ""),
            "skin_concern": row.get("skin_concern", ""),
            "price_usd": row.get("price_usd", ""),
            "rating": rating_col.iat[i],
            "reviews": reviews_col.iat[i],
            "similarity": float(sim)
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

def all_concerns_unique(df):
    s = df["skin_concern"].fillna("").astype(str)
    uniq = set()
    for txt in s:
        for t in re.split(r"[;,/|]", txt):
            t = t.strip().lower()
            if t:
                uniq.add(t)
    return sorted(uniq)

# ==================== HELPER FUNCTIONS ====================

def display_recommendation_hybrid(index, product, rating, user_id=None, recommender=None):
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(f"{index}. {product.get('product_name', 'Product')}")
            st.write(f"**Brand:** {product.get('brand_name', 'Unknown')}")
            st.write(f"**Category:** {product.get('tertiary_category', 'Unknown')}")
            
            if user_id and recommender:
                try:
                    product_id = product.get('product_id')
                    if hasattr(product_id, 'item'):
                        product_id = product_id.item()
                    debug_info = recommender.get_recommendation_debug_info(str(product_id), str(user_id))
                    
                    with st.expander("🔍 Why was this recommended?", expanded=False):
                        compat = debug_info.get('compatibility', {})
                        st.write(f"**Skin Type Match:** {compat.get('skin_type_status', 'N/A')}")
                        st.write(f"**Concern Relevance:** {compat.get('concern_status', 'N/A')}")
                        st.write(f"**Budget:** Within your specified range")
                        
                        compatibility_score = debug_info.get('compatibility_score', 0)
                        if isinstance(compatibility_score, (int, float)) and compatibility_score > 0:
                            st.write(f"**Compatibility Score:** {compatibility_score:.2f}/2.0")
                        
                        if debug_info.get('product_addresses'):
                            st.write(f"**Addresses:** {', '.join(debug_info['product_addresses'])}")
                        
                except Exception as e:
                    pass
        
        with col2:
            st.metric("Rating", f"{rating:.1f}/5")
            st.write("")
        
        with col3:
            st.metric("Price", f"${product.get('price_usd', 0):.2f}")
            if st.button("View Details", key=f"btn_{index}"):
                product_id = product.get('product_id')
                if hasattr(product_id, 'item'):
                    product_id = product_id.item()
                st.session_state.viewing_product = product_id
                st.session_state.current_page = "product_detail"
                st.rerun()
        
        st.divider()

def display_recommendation_content_collab(index, product, rating, similarity=None):
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(f"{index}. {product.get('product_name', 'Unknown Product')}")
            st.write(f"**Brand:** {product.get('brand_name', 'Unknown')}")
            # Use user-selected product_type for Content-Based model
            if st.session_state.get('selected_model') == 'Content-Based' and st.session_state.get('skin_data', {}).get('product_type'):
                category = st.session_state.skin_data['product_type']
            else:
                category = product.get('tertiary_category', 'Unknown')
            st.write(f"**Category:** {category}")
            if st.session_state.get('selected_model') == 'Content-Based':
                skin_concern = product.get('skin_concern', 'N/A')
                if isinstance(skin_concern, str) and skin_concern and skin_concern.lower() != 'nan':
                    concerns = [c.strip() for c in re.split(r"[;,/|]", skin_concern) if c.strip()]
                    st.write(f"**Skin Concern:** {', '.join(concerns) if concerns else 'N/A'}")
                else:
                    st.write("**Skin Concern:** N/A")
            # Modified: Display reviews as integer
            reviews = product.get('reviews', 'N/A')
            if isinstance(reviews, (int, float)) and not pd.isna(reviews):
                reviews = int(reviews)
            st.write(f"**Reviews:** {reviews}")
        
        with col2:
            st.metric("Rating" if st.session_state.get('selected_model') == 'Content-Based' else "Predicted Rating", 
                     f"{rating:.1f}/5")
            if similarity is not None and st.session_state.get('selected_model') == 'Content-Based':
                match_percent = round(min(100, max(0, similarity * 100)))
                st.progress(match_percent / 100, text=f"{match_percent}% match")
        
        with col3:
            st.write(f"Price: ${product.get('price_usd', 0):.2f}")
            if st.button("View Details", key=f"btn_{index}"):
                product_id = product.get('product_id')
                if hasattr(product_id, 'item'):
                    product_id = product_id.item()
                st.session_state.viewing_product = product_id
                st.session_state.current_page = "product_detail"
                st.rerun()
        
        st.divider()

def display_product_card(product, col):
    with col:
        card = st.container(border=True)
        with card:
            st.subheader(product['product_name'])
            st.write(f"**Brand:** {product['brand_name']}")
            st.write(f"**Category:** {product['tertiary_category']}")
            st.write(f"**Price:** ${product['price_usd']}")

# ==================== RECOMMENDER SYSTEM INITIALIZATION ====================

@st.cache_resource
def load_recommenders():
    try:
        hybrid_rec = EnhancedHybridRecommender(
            train_path="data/CleanedDataSet/train_skincare.csv",
            products_path="data/CleanedDataSet/filtered_skincare_products.csv",
            content_model_path="models/product_embeddings.pkl",
            svd_model_path="models/surprise_svd_model.pkl"
        )
        collab_rec = CollaborativeRecommender("data/CleanedDataSet/collaborative_training_data.csv")
        return hybrid_rec, collab_rec
    except Exception as e:
        st.error(f"❌ Error loading recommenders: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None

if 'recommenders_loaded' not in st.session_state:
    with st.spinner('🔄 Loading recommender systems...'):
        hybrid_rec, collab_rec = load_recommenders()
        if hybrid_rec is not None:
            st.session_state.recommenders_loaded = True
            st.success("✅ All recommenders loaded successfully!")
            st.rerun()
else:
    hybrid_rec, collab_rec = load_recommenders()

# ==================== SIDEBAR NAVIGATION ====================

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=100", width=80)
    st.title("🌸 Skincare Recommender")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = 'home'
        st.session_state.selected_model = None
        st.session_state.skin_data = {}
        st.rerun()

    if st.button("💫 Get Recommendations", use_container_width=True):
        st.session_state.current_page = 'model_selection'
        st.rerun()
    
    if st.button("🛍️ All Products", use_container_width=True):
        st.session_state.current_page = 'all_products'
        st.rerun()
    
    st.divider()
    
    if st.session_state.selected_model:
        st.success(f"✅ Model: {st.session_state.selected_model}")
    if st.session_state.skin_data:
        st.success("✅ Skin Profile Complete")
    
    st.caption("Follow the flow: Model → Analysis → Recommendations")

# ==================== PAGE ROUTING AND MAIN CONTENT ====================

if st.session_state.current_page == 'home':
    st.header("🌸 Welcome to Your Skincare Journey")
    st.subheader("Get personalized skincare recommendations tailored to your unique needs")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🎯 How It Works:
        
        1. **🤖 Choose Your Recommendation Model**
           - Content-Based: Based on product similarities
           - Collaborative: Based on users like you  
           - Hybrid: Best of both worlds
        
        2. **📊 Complete Your Skin Analysis**
           - Tell us your skin type and concerns
           - Set your budget preferences
        
        3. **✨ Get Personalized Recommendations**
           - Discover products perfect for your skin
           - See detailed explanations for each recommendation
        """)
        
        st.markdown("### Ready to get started?")
        if st.button("🚀 Start Your Skincare Journey", type="primary", use_container_width=True):
            st.session_state.current_page = 'model_selection'
            st.rerun()
            
    with col2:
        st.image("https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400", 
                caption="Discover your perfect skincare routine")
        
        with st.container():
            st.markdown("### 📈 Our Database")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Products", "10,000+")
                st.metric("Brands", "500+")
            with col_b:
                st.metric("Reviews", "100K+")
                st.metric("Categories", "20+")

elif st.session_state.current_page == 'model_selection':
    st.header("🤖 Choose Your Recommendation Model")
    st.write("Each model has different strengths. Pick the one that best fits your needs:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📝 Content-Based
        **Best for new users**
        
        ✅ Based on product features & ingredients  
        ✅ Works without purchase history  
        ✅ Transparent recommendations  
        ✅ Great for specific skin concerns  
        
        Perfect if you know what ingredients work for your skin.
        """)
        if st.button("Select Content-Based", key="content", use_container_width=True):
            st.session_state.selected_model = "Content-Based"
            st.session_state.current_page = 'input_form'
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 👥 Collaborative  
        **Best for discovering new products**
        
        ✅ Based on users similar to you  
        ✅ Finds hidden gems  
        ✅ Social proof driven  
        ✅ Personalized to your preferences  
        
        Perfect if you want to discover what people like you are loving.
        """)
        if st.button("Select Collaborative", key="collab", use_container_width=True):
            st.session_state.selected_model = "Collaborative"
            st.session_state.current_page = 'input_form'
            st.rerun()
    
    with col3:
        st.markdown("""
        ### 🎯 Hybrid
        **Best overall experience**
        
        ✅ Combines both approaches  
        ✅ Most accurate recommendations  
        ✅ Adapts to your experience level  
        ✅ Balanced and comprehensive  
        
        Perfect for the most accurate and personalized results.
        """)
        if st.button("Select Hybrid", key="hybrid", use_container_width=True):
            st.session_state.selected_model = "Hybrid"
            st.session_state.current_page = 'skin_analysis'
            st.rerun()

elif st.session_state.current_page == 'skin_analysis':
    # Redirect to input_form if Collaborative model is selected
    if st.session_state.selected_model.lower() == 'collaborative':
        st.session_state.current_page = 'input_form'
        st.rerun()
    
    st.header("📊 Complete Your Skin Analysis")
    st.write(f"**Selected Model:** {st.session_state.selected_model}")
    
    if st.button("← Back to Model Selection"):
        st.session_state.current_page = 'model_selection'
        st.rerun()
    
    with st.form("skin_analysis_form"):
        st.subheader("Tell us about your skin")
        
        user_id = st.text_input("User ID", placeholder="Enter your user ID", 
                              help="Required for personalized recommendations")
        
        col1, col2 = st.columns(2)
        with col1:
            skin_type = st.selectbox("Skin Type", 
                                   ["", "Dry", "Oily", "Combination", "Normal", "Sensitive"],
                                   help="Select your primary skin type")
        with col2:
            budget = st.selectbox("Budget Preference", 
                                ["", "Under $25", "$25-$50", "$50-$100", "Over $100"],
                                help="Your preferred price range")
        
        concerns = st.multiselect(
            "Main Skin Concerns",
            ["Acne", "Redness", "Dehydration", "Aging", "Pigmentation", "Sensitivity", "Dullness", "Large pores"],
            help="Select all that apply to you (you can select multiple)"
        )
        
        num_products = st.slider("Number of Recommendations", 1, 10, 5,
                               help="How many products would you like to see?")
        
        submitted = st.form_submit_button("💾 Save Profile & Get Recommendations", type="primary")
        
        if submitted:
            if not all([user_id, skin_type, budget]):
                st.error("Please fill in all required fields (User ID, Skin Type, and Budget)")
            else:
                st.session_state.skin_data = {
                    'user_id': user_id,
                    'skin_type': skin_type,
                    'concerns': concerns,
                    'budget': budget,
                    'num_products': num_products
                }
                st.session_state.current_page = 'recommendations'
                st.rerun()

elif st.session_state.current_page == 'input_form':
    st.header("📊 Complete Your Skin Analysis")
    st.write(f"**Selected Model:** {st.session_state.selected_model}")
    
    model_type = st.session_state.selected_model.lower() if st.session_state.selected_model else None
    
    if st.button("← Back to Selection"):
        st.session_state.current_page = 'model_selection'
        st.rerun()
    
    try:
        with st.form("input_form"):
            user_id = None
            skin_type = None
            product_type = None
            concerns = None
            concern_match = None
            budget = None
            
            if model_type == 'collaborative': 
                sample_users = collab_rec.get_available_users(limit=3)
                user_id = st.text_input("User ID", placeholder="Enter User ID", 
                                       help=f"Enter your User ID for personalized recommendations. Leave blank to get popular recommendations. Sample IDs: {', '.join(map(str, sample_users))}")
            else:  # Content-Based
                col1, col2 = st.columns(2)
                with col1:
                    skin_type = st.selectbox("Skin Type", ["(any)"] + ["Dry", "Oily", "Combination", "Normal", "Sensitive"],
                                           help="Select your primary skin type")
                    product_type = st.selectbox("Product Type", ["(any)"] + sorted(content_df['product_type'].unique()),
                                               help="Select a product category (optional)")
                with col2:
                    budget = st.selectbox("Budget Preference", ["(any)", "Under $25", "Under $50", "Under $100", "No budget limit"],
                                        help="Your preferred price range")
                
                concerns = st.multiselect(
                    "Main Skin Concerns",
                    all_concerns_unique(content_df),
                    help="Select all that apply to you"
                )
                
                concern_match = st.radio("Concern Match", ["all", "any"], index=1, help="Match all concerns or any concern")
            
            num_products = st.slider("Number of Recommendations", 1, 50, 5,
                                   help="How many products would you like to see?")
            
            submitted = st.form_submit_button("💾 Save Profile & Get Recommendations", type="primary")
            
            if submitted:
                if model_type == 'collaborative':
                    if user_id:
                        st.success(f"User ID '{user_id}' submitted. Generating recommendations...")
                    else:
                        st.success("No User ID provided. Generating popular recommendations for new users...")
                    st.session_state.skin_data = {
                        'user_id': user_id.strip() if user_id else None,
                        'num_products': num_products
                    }
                else:
                    st.session_state.skin_data = {
                        'user_id': user_id.strip() if user_id else None,
                        'skin_type': None if skin_type == "(any)" else skin_type,
                        'concerns': concerns if concerns else None,
                        'budget': None if budget == "(any)" else budget,
                        'num_products': num_products,
                        'product_type': None if product_type == "(any)" else product_type,
                        'concern_match': concern_match
                    }
                st.session_state.current_page = 'recommendations'
                st.rerun()
    except Exception as e:
        st.error(f"Error rendering input form: {str(e)}")
        import traceback
        st.write("Detailed error:")
        st.code(traceback.format_exc())

elif st.session_state.current_page == 'recommendations':
    st.header("Your Personalized Skincare Recommendations")
    
    if not st.session_state.skin_data:
        st.warning("Please complete the skin analysis first")
        st.session_state.current_page = 'input_form' if st.session_state.selected_model.lower() == 'collaborative' else 'skin_analysis'
        st.rerun()
    
    skin_data = st.session_state.skin_data
    model_type = st.session_state.selected_model.lower() if st.session_state.selected_model else None

    # Dynamic User Classification
    user_exists = False
    user_rating_count = 0
    
    if model_type == 'collaborative':
        user_id = skin_data.get('user_id')
        if user_id and collab_rec and collab_rec.check_user_exists(user_id):
            user_exists = True
            profile, _ = collab_rec.get_user_profile_and_recommendations(user_id, 0)
            user_rating_count = profile.get('total_reviews', 0)
            print(f"👤 User {user_id}: {user_rating_count} ratings from CollaborativeRecommender")
        else:
            print(f"👤 User ID: {user_id or 'None'}, Exists: {False}, Treating as new user")
    else:
        if hybrid_rec and hasattr(hybrid_rec, 'user_history_cache'):
            user_data = hybrid_rec.user_history_cache.get(str(skin_data['user_id']))
            if user_data:
                user_rating_count = len(user_data.get('rated_products', []))
                user_exists = user_rating_count > 0
            print(f"👤 User {skin_data['user_id']}: {user_rating_count} ratings in recommender cache")
        else:
            try:
                train_df = pd.read_csv("data/CleanedDataSet/train_skincare.csv", 
                                     usecols=['author_id'], 
                                     low_memory=False)
                user_ratings = train_df[train_df['author_id'].astype(str) == str(skin_data['user_id'])]
                user_rating_count = len(user_ratings)
                user_exists = user_rating_count > 0
                print(f"👤 User {skin_data['user_id']}: {user_rating_count} ratings from file")
            except Exception as e:
                print(f"Could not check user history: {e}")
                user_exists = False
                user_rating_count = 0

    # Display Skin Profile only for Content-Based and Hybrid models
    if model_type in ['content-based', 'hybrid']:
        with st.expander("Your Skin Profile"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**User ID:** {skin_data['user_id'] or 'None'}")
                st.write(f"**Skin Type:** {skin_data['skin_type']}")
                if 'product_type' in skin_data and skin_data['product_type']:
                    st.write(f"**Product Type:** {skin_data['product_type'] or 'Any'}")
            with col2:
                st.write(f"**Budget:** {skin_data['budget']}")
                st.write(f"**Number of Products:** {skin_data['num_products']}")
            with col3:
                st.write(f"**Concerns:** {', '.join(skin_data['concerns']) if skin_data['concerns'] else 'None'}")
                if not user_exists:
                    st.write(f"**User Type:** 🆕 New User")
                elif user_rating_count < 10:
                    st.write(f"**User Type:** 👤 Existing User - {user_rating_count} ratings")
                else:
                    st.write(f"**User Type:** 🎯 Experienced User - {user_rating_count} ratings")

    st.subheader("Recommended For You")
    if model_type == 'hybrid':
        st.write("🎯 Higher concern score = Better for your specific skin issues  ⭐ Higher rating = More customers loved this product  💰 All prices are within your budget")
    st.write("")
    st.write("")

    recommendations = []
    recommendations_df = None  # To store DataFrame for CSV download
    
    if model_type == 'hybrid' and hybrid_rec:
        hybrid_rec.add_skin_profile(skin_data['user_id'], {
            'skin_type': skin_data['skin_type'],
            'concerns': skin_data['concerns'],
            'budget': skin_data['budget']
        })
                
        with st.spinner("Generating hybrid recommendations..."):
            try:
                if not user_exists:
                    print(f"🆕 NEW USER: Using content 0.7 + collab 0.3")
                    recommendations = hybrid_rec.get_recommendations_for_new_user(
                        skin_type=skin_data['skin_type'],
                        concerns=skin_data['concerns'], 
                        budget=skin_data['budget'],
                        top_n=skin_data['num_products']
                    )
                elif user_rating_count < 10:
                    print(f"👤 EXISTING USER ({user_rating_count} ratings): Using content 0.6 + collab 0.4")
                    recommendations = hybrid_rec.generate_recommendations(
                        user_id=skin_data['user_id'],
                        top_n=skin_data['num_products'],
                        content_weight=0.6,
                        collab_weight=0.4
                    )
                else:
                    print(f"🎯 EXPERIENCED USER ({user_rating_count} ratings): Using content 0.4 + collab 0.6")
                    recommendations = hybrid_rec.generate_recommendations(
                        user_id=skin_data['user_id'],
                        top_n=skin_data['num_products'],
                        content_weight=0.4,
                        collab_weight=0.6
                    )
                # Convert recommendations to DataFrame for CSV
                if recommendations and isinstance(recommendations, list):
                    recommendations_df = pd.DataFrame([
                        {
                            'product_id': rec[0],
                            'product_name': detail_products_df[detail_products_df['product_id'].astype(str) == str(rec[0])]['product_name'].iloc[0] if not detail_products_df[detail_products_df['product_id'].astype(str) == str(rec[0])].empty else 'Unknown',
                            'brand_name': detail_products_df[detail_products_df['product_id'].astype(str) == str(rec[0])]['brand_name'].iloc[0] if not detail_products_df[detail_products_df['product_id'].astype(str) == str(rec[0])].empty else 'Unknown',
                            'category': detail_products_df[detail_products_df['product_id'].astype(str) == str(rec[0])]['tertiary_category'].iloc[0] if not detail_products_df[detail_products_df['product_id'].astype(str) == str(rec[0])].empty else 'Unknown',
                            'predicted_rating': rec[1],
                            'price_usd': detail_products_df[detail_products_df['product_id'].astype(str) == str(rec[0])]['price_usd'].iloc[0] if not detail_products_df[detail_products_df['product_id'].astype(str) == str(rec[0])].empty else None
                        } for rec in recommendations
                    ])
            except Exception as e:
                st.error(f"Error generating hybrid recommendations: {e}")
                recommendations = []
                
    elif model_type == 'content-based':
        with st.spinner("Generating content-based recommendations..."):
            try:
                max_price = None
                if skin_data['budget'] == "Under $25":
                    max_price = 25
                elif skin_data['budget'] == "Under $50":
                    max_price = 50
                elif skin_data['budget'] == "Under $100":
                    max_price = 100
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
                recommendations_df = recommendations  # Already a DataFrame
                if recommendations.empty:
                    st.warning("No recommendations found. Try selecting fewer concerns, choosing 'any' for concern match, or leaving product type, skin type, and budget as '(any)'.")
            except Exception as e:
                st.error(f"Error generating content-based recommendations: {str(e)}")
                import traceback
                st.write("Detailed error:")
                st.code(traceback.format_exc())
                recommendations = pd.DataFrame()
                
    elif model_type == 'collaborative':
        if not collab_rec:
            st.error("Collaborative Recommender is not available. Please check the system status in the sidebar.")
            st.stop()
        
        with st.spinner("Finding community favorites..."):
            user_id = skin_data.get('user_id')
            try:
                print(f"Calling get_user_profile_and_recommendations for user_id: {user_id}")
                profile, recommendations = collab_rec.get_user_profile_and_recommendations(
                    user_id, 
                    skin_data['num_products']
                )
                
                if profile:
                    st.subheader("User Profile")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("**Total Rating**", profile.get('total_reviews', 0))
                        st.metric("**Average Rating**", f"{profile.get('avg_rating', 0):.2f}")
                    with col2:
                        st.write(f"**Skin Type:** {profile.get('skin_type', 'Unknown')}")
                        fav_brands = profile.get('favorite_brands', [])
                        st.write(f"**Favorite Brands:** {', '.join(fav_brands[:3]) if fav_brands else 'None'}")
                        if not user_exists:
                            st.write("**User Type:** 🆕 New User")
                        elif user_rating_count < 10:
                            st.write(f"**User Type:** 👤 Existing User")
                        else:
                            st.write(f"**User Type:** 🎯 Experienced User")
                else:
                    st.info("Showing popular recommendations for new users.")
                    st.caption("As a new user, you'll see our most popular products based on community ratings.")
                    st.write("**User Type:** 🆕 New User")
                
                if recommendations and len(recommendations) > 0:
                    recommendations_df = pd.DataFrame(recommendations)
                else:
                    st.error("No recommendations generated. Please check the dataset or try a different User ID.")
                    recommendations = []
                    recommendations_df = None
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                import traceback
                st.write("Full Error Details:")
                st.code(traceback.format_exc())
                recommendations = []
                recommendations_df = None

    if recommendations is not None and (isinstance(recommendations, list) and len(recommendations) > 0 or isinstance(recommendations, pd.DataFrame) and not recommendations.empty):
        try:
            if isinstance(recommendations, pd.DataFrame):
                recommendations = recommendations.to_dict('records')
            
            for i, rec in enumerate(recommendations, 1):
                if isinstance(rec, tuple) and len(rec) >= 3:
                    product_id, rating, _ = rec[:3]
                    similarity = None
                elif isinstance(rec, tuple) and len(rec) == 2:
                    product_id, rating = rec
                    similarity = None
                elif isinstance(rec, dict):
                    product_id = rec.get('product_id')
                    rating = rec.get('predicted_rating', rec.get('rating', None))
                    similarity = rec.get('similarity', None)
                else:
                    st.error(f"Unexpected recommendation format at index {i}: {type(rec)}")
                    continue
                
                # Use content_df for Content-Based to get skin_concern, detail_products_df for others
                product_df = content_df if model_type == 'content-based' else detail_products_df
                product_info = product_df[product_df['product_id'].astype(str) == str(product_id)]
                
                if not product_info.empty:
                    product_info = product_info.iloc[0].to_dict()
                    if model_type == 'hybrid':
                        display_recommendation_hybrid(
                            i, product_info, rating, 
                            skin_data['user_id'], hybrid_rec
                        )
                    else:
                        display_recommendation_content_collab(
                            i, product_info, rating, 
                            similarity=similarity if model_type == 'content-based' else None
                        )
                else:
                    st.warning(f"Product ID {product_id} not found in {'content-based' if model_type == 'content-based' else 'product'} catalog.")
        except Exception as e:
            st.error(f"Error displaying recommendations: {str(e)}")
            import traceback
            st.write("Detailed error:")
            st.code(traceback.format_exc())
    else:
        st.warning("No recommendations found. Try adjusting your skin profile or selecting a different model.")
    
    if model_type == 'hybrid' and hybrid_rec:
        try:
            conflicts = hybrid_rec.check_ingredient_conflicts(recommendations)
            if conflicts['has_conflicts']:
                st.warning("⚠️ **Ingredient Conflicts Detected**")
                with st.expander("View potential conflicts", expanded=False):
                    for conflict in conflicts['conflicts']:
                        st.write(f"**{conflict['ingredient1'].title()}** may conflict with **{conflict['ingredient2'].title()}**")
                        st.write(f"💡 {conflict['warning']}")
                        st.write("---")
        except:
            pass
    
    # Moved Download Button to Bottom
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Update Skin Profile", use_container_width=True):
            st.session_state.current_page = 'skin_analysis' if model_type == 'hybrid' else 'input_form'
            st.rerun()
    with col2:
        if st.button("🏠 Start Over", use_container_width=True):
            st.session_state.current_page = 'home'
            st.session_state.skin_data = {}
            st.rerun()
    with col3:
        if recommendations_df is not None and not recommendations_df.empty:
            csv = recommendations_df.to_csv(index=False).encode("utf-8")
            file_name = {
                'hybrid': 'hybrid_recommendations.csv',
                'content-based': 'skincare_recommendations.csv',
                'collaborative': 'collaborative_recommendations.csv'
            }.get(model_type, 'recommendations.csv')
            st.download_button(
                "📥 Download Recommendations (CSV)", 
                data=csv, 
                file_name=file_name, 
                mime="text/csv",
                use_container_width=True
            )

elif st.session_state.current_page == "product_detail":
    if 'viewing_product' in st.session_state and st.session_state.viewing_product is not None:
        product_id = str(st.session_state.viewing_product)
        
        product_info = detail_products_df[detail_products_df['product_id'] == product_id]
        if not product_info.empty:
            product = product_info.iloc[0].to_dict()
            
            st.header(f"📋 {product.get('product_name', '-')}")
            st.info("🔍 You're viewing details of a recommended product")
            
            st.subheader(f"by {product.get('brand_name', '-')}")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Product ID:** {product.get('product_id', '-')}")
                st.write(f"**Brand:** {product.get('brand_name', '-')}")
                st.write(f"**Category:** {product.get('tertiary_category', '-')}")
                st.write(f"**Size:** {product.get('size', '-')}")
                st.write(f"**Price:** ${product.get('price_usd', '-'):.2f}")
                
                # Handle Highlights
                highlights = product.get("highlights", None)
                st.markdown("### ✨ Highlights")
                if pd.isna(highlights) or highlights is None or str(highlights).strip().lower() in ["", "nan"]:
                    st.write("N/A")
                else:
                    st.markdown(f"- {highlights.replace(';', '<br>- ')}", unsafe_allow_html=True)
                
                # Handle Ingredients
                ingredients = product.get("ingredients", None)
                st.markdown("### 🧴 Ingredients")
                if pd.isna(ingredients) or ingredients is None or str(ingredients).strip().lower() in ["", "nan"]:
                    st.write("N/A")
                else:
                    try:
                        import ast
                        if isinstance(ingredients, str) and ingredients.startswith("["):
                            ing_list = ast.literal_eval(ingredients)
                            ing_list = [i.strip() for i in ing_list if i.strip()]
                        else:
                            ing_list = [str(ingredients).strip()]
                        if not ing_list:
                            st.write("N/A")
                        else:
                            for ing in ing_list:
                                st.markdown(f"- {ing}")
                    except Exception as e:
                        st.write("N/A")
                        print(f"Error parsing ingredients for product {product_id}: {str(e)}")
        else:
            st.error(f"Product ID {product_id} not found in product catalog!")
    else:
        st.error("No product selected for viewing!")

    st.divider()
    if st.button("← Back to Recommendations"):
        if 'viewing_product' in st.session_state:
            del st.session_state.viewing_product
        st.session_state.current_page = "recommendations"
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
    
    st.divider()
    st.caption("Built with ❤️ using advanced machine learning algorithms")

elif st.session_state.current_page == 'all_products':
    st.header("🛍️ All Products")
    
    if st.button("← Back to Home"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    try:
        df = detail_products_df
        
        st.subheader(f"Browse Our Collection ({len(df)} Products)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search products...", placeholder="Enter product name, brand, or ingredient")
        
        with col2:
            categories = sorted(df['tertiary_category'].unique()) if 'tertiary_category' in df.columns else []
            selected_category = st.selectbox("Filter by Category", ['All'] + categories)
        
        filtered_df = df.copy()
        
        if search_term:
            mask = (
                df['product_name'].str.contains(search_term, case=False, na=False) |
                df['brand_name'].str.contains(search_term, case=False, na=False)
            )
            if 'notable_effects' in df.columns:
                mask |= df['notable_effects'].str.contains(search_term, case=False, na=False)
            if 'ingredients' in df.columns:
                mask |= df['ingredients'].str.contains(search_term, case=False, na=False)
            filtered_df = df[mask]
        
        if selected_category != 'All' and 'tertiary_category' in df.columns:
            filtered_df = filtered_df[filtered_df['tertiary_category'] == selected_category]
        
        st.write(f"Showing {len(filtered_df)} products")
        
        if not filtered_df.empty:
            for i in range(0, len(filtered_df), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(filtered_df):
                        product = filtered_df.iloc[i + j]
                        display_product_card(product, cols[j])
        else:
            st.info("No products found matching your criteria.")
            
    except Exception as e:
        st.error(f"Error loading products: {str(e)}")

if __name__ == "__main__":
    pass