import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from streamlit_option_menu import option_menu
import os
from utils.recommender import EnhancedHybridRecommender, CollaborativeRecommender

# Add this debugging code at the top of your app.py, right after your imports
# and before the page setup
import os
import traceback
import sys
import os
# Force UTF-8 encoding for the terminal
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def debug_file_existence():
    """Check if all required files exist"""
    required_files = [
        "products_preprocessed.csv",
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

# Modified load_recommenders function with detailed debugging
@st.cache_resource
@st.cache_resource
def load_recommenders():
    print("Starting to load recommenders...")
    
    hybrid_rec = None
    collab_rec = None
    
    # Try to load hybrid recommender
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
    
    # Try to load collaborative recommender
    try:
        print("Loading collaborative recommender...")
        collab_rec = CollaborativeRecommender(r"c:\Users\lucas\Downloads\AI_Assignment\AI-Assignment\collaborative_training_data.csv")
        print(f"CollaborativeRecommender df shape: {collab_rec.df.shape if collab_rec.df is not None else 'None'}")
        print("✅ Collaborative recommender loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading collaborative recommender: {e}")
        traceback.print_exc()
    
    return hybrid_rec, collab_rec

# Replace your existing load_recommenders call with this
hybrid_rec, collab_rec = load_recommenders()

# Add status display in sidebar
# with st.sidebar:
#     st.divider()
#     st.subheader("System Status")
#     st.write(f"Hybrid Recommender: {'✅' if hybrid_rec else '❌'}")
#     st.write(f"Collaborative Recommender: {'✅' if collab_rec else '❌'}")
    
#     if collab_rec and collab_rec.df is not None:
#         st.write(f"Training Records: {len(collab_rec.df)}")
#         st.write(f"Unique Users: {collab_rec.df['author_id'].nunique()}")
#     else:
#         st.write("Training Data: ❌")

# Page setup
st.set_page_config(
    page_title="Skincare Recommendation System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'skin_data' not in st.session_state:
    st.session_state.skin_data = {}

# Load product data and TF-IDF
@st.cache_data(show_spinner=True)
def load_products(path="products_preprocessed.csv"):
    df = pd.read_csv(path)
    for c in ["price_usd", "rating", "reviews"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["skin_concern"] = df.get("skin_concern", "").fillna("").astype(str)
    df["skin_type"] = df.get("skin_type", "").fillna("").astype(str)
    df["product_type"] = df.get("product_type", "").fillna("").astype(str)
    return df

@st.cache_resource(show_spinner=True)
def build_vectorizer_and_matrix(product_text: pd.Series):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(product_text.fillna("").astype(str))
    return vectorizer, tfidf_matrix

df = load_products("products_preprocessed.csv")
vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["product_content"])

# Recommender logic from zw_app.py
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

    price_col = pd.to_numeric(df.get("price_usd", np.nan), errors="coerce")
    rating_col = pd.to_numeric(df.get("rating", np.nan), errors="coerce").fillna(0.0)
    reviews_col = pd.to_numeric(df.get("reviews", 0), errors="coerce").fillna(0).astype(int)

    rows = []
    for i, sim in enumerate(sims):
        row = df.iloc[i]

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

# Dynamically extract concern options
def all_concerns_unique(df):
    s = df["skin_concern"].fillna("").astype(str)
    uniq = set()
    for txt in s:
        for t in re.split(r"[;,/|]", txt):
            t = t.strip().lower()
            if t:
                uniq.add(t)
    return sorted(uniq)

# Helper functions
def display_recommendation(index, product, rating, similarity):
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(f"{index}. {product.get('product_name', 'Product')}")
            st.write(f"**Brand:** {product.get('brand_name', 'Unknown')}")
            st.write(f"**Type:** {product.get('product_type', 'Unknown') if 'product_type' in product else product.get('tertiary_category', 'Unknown')}")
            st.write(f"**Skin Type:** {product.get('skin_type', 'N/A')}")
            st.write(f"**Category:** {product.get('secondary_category', 'N/A')}")
        
        with col2:
            st.metric("Rating", f"{rating:.1f}/5")
            match_percent = round(min(100, max(0, similarity * 100)))
            st.progress(match_percent / 100, text=f"{match_percent}% match")
            st.write(f"**Price:** ${product.get('price_usd', 0):.2f}")
        
        with col3:
            st.write(f"**Predicted Rating:** {product.get('predicted_rating', 'N/A')}")
            if st.button("View Details", key=f"btn_{index}"):
                st.write(f"**Product ID:** {product.get('product_id', 'N/A')}")
        
        st.divider()

def display_product_card(product, col):
    with col:
        card = st.container(border=True)
        with card:
            st.subheader(product['product_name'])
            st.write(f"**Brand:** {product['brand_name']}")
            st.write(f"**Category:** {product['tertiary_category']}")
            st.write(f"**Price:** ${product['price_usd']}")

# Initialize recommenders
@st.cache_resource
def load_recommenders():
    print("Starting to load recommenders...")
    
    hybrid_rec = None
    collab_rec = None
    
    # Try to load hybrid recommender
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
    
    # Try to load collaborative recommender
    try:
        print("Loading collaborative recommender...")
        collab_rec = CollaborativeRecommender("collaborative_training_data.csv")
        print(f"CollaborativeRecommender df shape: {collab_rec.df.shape if collab_rec.df is not None else 'None'}")
        print("✅ Collaborative recommender loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading collaborative recommender: {e}")
        traceback.print_exc()
    
    return hybrid_rec, collab_rec

# Sidebar navigation
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=100", width=80)
    st.title("🌸 Skincare Recommender")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = 'home'
        st.rerun()
    
    if st.button("💫 Get Recommendations", use_container_width=True):
        st.session_state.current_page = 'select_approach'
        st.rerun()
    
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.current_page = 'about'
        st.rerun()
    
    st.divider()
    st.caption("Quick Navigation")

# Page routing
if st.session_state.current_page == 'home':
    st.header("🌸 Discover Your Perfect Skincare")
    st.subheader("Browse our curated collection or get personalized recommendations")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input("🔍 Search products", placeholder="Enter product name or brand")
    with col2:
        selected_category = st.selectbox("Filter by category", 
                                       ["All"] + sorted(df['tertiary_category'].unique()))
    
    filtered_products = df.copy()
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
        st.session_state.current_page = 'select_approach'
        st.rerun()

elif st.session_state.current_page == 'select_approach':
    st.header("Choose Your Recommendation Style")
    
    st.write("How would you like us to find your perfect skincare match?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Smart Matching", use_container_width=True, help="Based on product ingredients and features"):
            st.session_state.selected_model = 'content'
            st.session_state.current_page = 'input_form'
            st.rerun()
        st.caption("AI-powered analysis of product ingredients and features")
    
    with col2:
        if st.button("👥 Community Wisdom", use_container_width=True, help="From users with similar skin profiles"):
            st.session_state.selected_model = 'collab' 
            st.session_state.current_page = 'input_form'
            st.rerun()
        st.caption("Recommendations from users with similar skin concerns")
    
    with col3:
        if st.button("🌟 Best of Both", use_container_width=True, help="Combined AI and community insights"):
            st.session_state.selected_model = 'hybrid'
            st.session_state.current_page = 'input_form'
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

elif st.session_state.current_page == 'input_form':
    model_type = st.session_state.selected_model
    st.header(f"Enter Details for {model_type.capitalize()} Recommendations")
    
    if st.button("← Back to Selection"):
        st.session_state.current_page = 'select_approach'
        st.rerun()
    
    with st.form("input_form"):
        user_id = None
        skin_type = None
        product_type = None
        concerns = None
        concern_match = None
        budget = None
        
        if model_type in ['collab', 'hybrid']:
            user_id = st.text_input("User ID", placeholder="Enter your user ID", help="Required for personalized recommendations")
        
        if model_type in ['content', 'hybrid']:
            col1, col2 = st.columns(2)
            with col1:
                skin_type = st.selectbox("Skin Type", ["(any)"] + ["Dry", "Oily", "Combination", "Normal", "Sensitive"],
                                       help="Select your primary skin type")
                product_type = st.selectbox("Product Type", ["(any)"] + sorted(df['product_type'].unique()),
                                           help="Select a product category (optional)")
            with col2:
                budget = st.selectbox("Budget Preference", ["(any)", "Under $25", "$25-$50", "$50-$100", "Over $100", "No budget limit"],
                                    help="Your preferred price range")
            
            concerns = st.multiselect(
                "Main Skin Concerns",
                all_concerns_unique(df),
                help="Select all that apply to you"
            )
            
            concern_match = st.radio("Concern Match", ["all", "any"], index=1, help="Match all concerns or any concern")
        
        num_products = st.slider("Number of Recommendations", 1, 50, 5,
                               help="How many products would you like to see?")
        
        submitted = st.form_submit_button("🎯 Get Personalized Recommendations", type="primary")
        
        if submitted:
            st.session_state.skin_data = {
                'user_id': user_id,
                'skin_type': None if skin_type == "(any)" else skin_type,
                'concerns': concerns if concerns else None,
                'budget': None if budget == "(any)" else budget,
                'num_products': num_products,
                'product_type': None if product_type == "(any)" else product_type,
                'concern_match': concern_match
            }
            st.session_state.current_page = 'recommendations'
            st.rerun()

elif st.session_state.current_page == 'recommendations':
    st.header("Your Personalized Skincare Recommendations")
    
    if not st.session_state.skin_data:
        st.warning("Please complete the input form first")
        st.session_state.current_page = 'input_form'
        st.rerun()
    
    skin_data = st.session_state.skin_data
    model_type = st.session_state.selected_model
    
    with st.expander("Your Input Profile"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'user_id' in skin_data and skin_data['user_id']:
                st.write(f"**User ID:** {skin_data['user_id']}")
            if 'skin_type' in skin_data and skin_data['skin_type']:
                st.write(f"**Skin Type:** {skin_data['skin_type']}")
            if 'product_type' in skin_data and skin_data['product_type']:
                st.write(f"**Product Type:** {skin_data['product_type'] or 'Any'}")
        with col2:
            if 'budget' in skin_data and skin_data['budget']:
                st.write(f"**Budget:** {skin_data['budget']}")
            st.write(f"**Number of Products:** {skin_data['num_products']}")
            if 'concern_match' in skin_data and skin_data['concern_match']:
                st.write(f"**Concern Match:** {skin_data['concern_match'].capitalize()}")
        with col3:
            if 'concerns' in skin_data and skin_data['concerns']:
                st.write(f"**Concerns:** {', '.join(skin_data['concerns']) if skin_data['concerns'] else 'None'}")
            st.write(f"**Model:** {model_type.capitalize()}")
    
    st.subheader("Recommended For You")
    
    if model_type == 'hybrid' and hybrid_rec:
        skin_profile_data = {
            'user_id': skin_data['user_id'],
            'skin_type': skin_data.get('skin_type'),
            'concerns': skin_data.get('concerns'),
            'budget': skin_data.get('budget')
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
                        product_info = df[df['product_id'].astype(str) == product_id]
                        if not product_info.empty:
                            product_info = product_info.iloc[0]
                            display_recommendation(i, product_info, rating, match_percent / 100)
                else:
                    st.warning("No recommendations found. Try adjusting your skin profile or broadening your filters.")
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                recommendations = hybrid_rec.generate_recommendations(
                    skin_data['user_id'], 
                    skin_data['num_products']
                )
    
    elif model_type == 'content':
        with st.spinner("🔍 Finding products that match your skin needs..."):
            try:
                progress = st.progress(0)
                st.write(f"Generating recommendations for: {skin_data}")
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
                progress.progress(100)
                
                if not recommendations.empty:
                    for i, rec in enumerate(recommendations.to_dict('records'), 1):
                        display_recommendation(i, rec, rec['rating'], rec['similarity'])
                    csv = recommendations.to_csv(index=False).encode("utf-8")
                    st.download_button("Download results (CSV)", data=csv, file_name="skincare_recommendations.csv", mime="text/csv")
                else:
                    st.warning("No recommendations found. Try selecting fewer concerns, choosing 'any' for concern match, or leaving product type, skin type, and budget as '(any)'.")
            except Exception as e:
                st.error(f"Error generating content-based recommendations: {e}")
    
    # Replace your collaborative filtering section in app.py with this:
    elif model_type == 'collab':
        # st.subheader("Community-Based Recommendations")
        
        # Check if collaborative recommender is available
        if not collab_rec:
            st.error("Collaborative Recommender is not available. Please check the system status in the sidebar.")
            st.stop()
        
        # # Show system information
        # with st.expander("System Information"):
        #     system_info = collab_rec.get_system_info()
        #     st.json(system_info)
        
        with st.spinner("Finding community favorites for your skin type..."):
            if not skin_data.get('user_id'):
                st.error("Please enter a valid User ID.")
            else:
                user_id = str(skin_data['user_id']).strip()
                st.info(f"Searching for recommendations for user: '{user_id}'")
                
                # Show debugging information
                # with st.expander("Debug Information (Click to expand)"):
                #     st.write(f"**Input User ID:** `{user_id}` (type: {type(user_id)})")
                    
                #     # Check user existence
                #     user_exists = collab_rec.check_user_exists(user_id)
                #     st.write(f"**User exists in training data:** {user_exists}")
                    
                #     # Show sample user IDs
                #     sample_users = collab_rec.get_available_users(20)
                #     st.write(f"**Sample available user IDs ({len(sample_users)} shown):**")
                #     st.write(sample_users)
                    
                #     if not user_exists:
                #         st.warning("User not found in training data. The system will provide popular recommendations.")
                
                # Get recommendations
                try:
                    # st.write("Generating recommendations...")
                    profile, recommendations = collab_rec.get_user_profile_and_recommendations(
                        user_id, 
                        skin_data['num_products']
                    )
                    
                    # st.write(f"Profile received: {bool(profile)}")
                    # st.write(f"Recommendations received: {len(recommendations) if recommendations else 0}")
                    
                    # Display profile
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
                    
                    # Display recommendations
                    if recommendations and len(recommendations) > 0:
                        st.subheader("Your Recommendations")
                        
                        for i, rec in enumerate(recommendations, 1):
                            with st.container():
                                col1, col2, col3 = st.columns([3, 1, 1])
                                
                                with col1:
                                    st.subheader(f"{i}. {rec.get('product_name', 'Unknown Product')}")
                                    st.write(f"**Brand:** {rec.get('brand_name', 'Unknown')}")
                                    st.write(f"**Category:** {rec.get('tertiary_category', 'Unknown')}")
                                    st.write(f"**Product ID:** {rec.get('product_id', 'N/A')}")
                                
                                with col2:
                                    rating = rec.get('predicted_rating', 0)
                                    st.metric("Predicted Rating", f"{rating:.1f}/5")
                                    price = rec.get('price_usd', 0)
                                    st.write(f"**Price:** ${price:.2f}")
                                
                                with col3:
                                    if st.button(f"Details", key=f"collab_btn_{i}"):
                                        st.write("**Full Product Info:**")
                                        st.json(rec)
                                
                                st.divider()
                        
                        # Download option
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
                        
                        # Troubleshooting suggestions
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
                        import traceback
                        st.code(traceback.format_exc())

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Get New Recommendations", use_container_width=True):
            st.session_state.current_page = 'select_approach'
            st.rerun()
    with col2:
        if st.button("🏠 Start Over", use_container_width=True):
            st.session_state.current_page = 'home'
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
    
    st.divider()
    st.caption("Built with ❤️ using advanced machine learning algorithms")

if __name__ == "__main__":
    pass