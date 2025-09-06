"""
Skincare Recommendation System - Streamlit User Interface
========================================================

This application provides a user-friendly web interface for the advanced skincare 
recommendation system. Users can input their skin profile and get personalized 
product recommendations using state-of-the-art AI algorithms.

KEY FEATURES:
- Interactive skin profile input (skin type, concerns, budget)
- Multiple recommendation algorithms (Hybrid, Content-Based, Collaborative)
- Detailed product information and explanations
- Visual charts and analytics
- Product browsing and filtering
- Recommendation reasoning and transparency

USER FLOW:
1. Home Page: Welcome and navigation
2. Skin Analysis: Input skin type, concerns, and budget preferences  
3. Get Recommendations: Choose algorithm and view personalized suggestions
4. All Products: Browse complete product catalog
5. Product Details: View detailed product information

ALGORITHM INTEGRATION:
- EnhancedHybridRecommender: Main algorithm combining content + collaborative filtering
- ContentBasedRecommender: Pure content filtering based on skin profile
- CollaborativeRecommender: Pure collaborative filtering based on user similarities
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.recommender import EnhancedHybridRecommender, ContentBasedRecommender, CollaborativeRecommender
import plotly.express as px
from streamlit_option_menu import option_menu

# ==================== APPLICATION CONFIGURATION ====================
st.set_page_config(
    page_title="Skincare Recommendation System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE INITIALIZATION ====================
# Initialize session variables for maintaining user state across page interactions
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'    # Current page being viewed
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None    # Chosen recommendation algorithm
if 'skin_data' not in st.session_state:
    st.session_state.skin_data = {}          # User's skin profile data

# ==================== HELPER FUNCTIONS ====================

def display_recommendation(index, product, rating, user_id=None, recommender=None):
    """
    Display a single product recommendation in a formatted card layout
    
    Args:
        index: Recommendation rank (1, 2, 3, etc.)
        product: Product information dictionary
        rating: Product quality rating to display
        user_id: User ID for generating explanation (optional)
        recommender: Recommender instance for generating explanations (optional)
    """
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Product basic information
            st.subheader(f"{index}. {product.get('product_name', 'Product')}")
            st.write(f"**Brand:** {product.get('brand_name', 'Unknown')}")
            st.write(f"**Category:** {product.get('tertiary_category', 'Unknown')}")
            
            # Show recommendation reasoning if available
            if user_id and recommender:
                try:
                    product_id = product.get('product_id')
                    if hasattr(product_id, 'item'):
                        product_id = product_id.item()  # Extract scalar from pandas Series
                    debug_info = recommender.get_recommendation_debug_info(str(product_id), str(user_id))
                    
                    # Expandable section showing why this product was recommended
                    with st.expander("🔍 Why was this recommended?", expanded=False):
                        compat = debug_info.get('compatibility', {})
                        
                        # Show key recommendation factors in user-friendly format
                        st.write(f"**Skin Type Match:** {compat.get('skin_type_status', 'N/A')}")
                        st.write(f"**Concern Relevance:** {compat.get('concern_status', 'N/A')}")
                        st.write(f"**Budget:** Within your specified range")
                        
                        # Show compatibility score if meaningful
                        compatibility_score = debug_info.get('compatibility_score', 0)
                        if isinstance(compatibility_score, (int, float)) and compatibility_score > 0:
                            st.write(f"**Compatibility Score:** {compatibility_score:.2f}/2.0")
                        
                        # Show what specific concerns this product addresses
                        if debug_info.get('product_addresses'):
                            st.write(f"**Addresses:** {', '.join(debug_info['product_addresses'])}")
                        
                except Exception as e:
                    pass  # Gracefully skip debug info if there's an error
        
        with col2:
            # Display product rating
            st.metric("Rating", f"{rating:.1f}/5")
            st.write("")  # Spacing for visual balance
        
        with col3:
            # Display price and action button
            st.metric("Price", f"${product.get('price_usd', 0):.2f}")
            if st.button("View Details", key=f"btn_{index}"):
                # Navigate to detailed product view
                product_id = product.get('product_id')
                if hasattr(product_id, 'item'):
                    product_id = product_id.item()
                st.session_state.viewing_product = product_id
                st.session_state.current_page = "product_detail"
                st.rerun()
        
        st.divider()  # Visual separator between recommendations

def display_product_card(product, col):
    """
    Display a product in a card format for browsing/selection
    
    Args:
        product: Product information dictionary
        col: Streamlit column to display the card in
    """
    with col:
        card = st.container(border=True)
        with card:
            st.subheader(product['product_name'])
            st.write(f"**Brand:** {product['brand_name']}")
            st.write(f"**Category:** {product['tertiary_category']}")
            st.write(f"**Price:** ${product['price_usd']}")
            
            if st.button("Select & Get Recommendations", key=f"select_{product['product_id']}", 
                        use_container_width=True):
                # Store selected product and navigate to skin analysis
                product_id = product['product_id']
                if hasattr(product_id, 'item'):
                    product_id = product_id.item()  # Extract scalar from pandas Series
                st.session_state.selected_product = product_id
                st.session_state.current_page = 'skin analysis'
                st.rerun()

# ==================== RECOMMENDER SYSTEM INITIALIZATION ====================

@st.cache_resource
def load_recommenders():
    """
    Initialize and cache all recommendation algorithms
    
    Uses Streamlit's caching to avoid reloading models on every interaction.
    This function loads the hybrid, content-based, and collaborative recommenders.
    
    Returns:
        Tuple of (hybrid_recommender, content_recommender, collaborative_recommender)
    """
    try:
        # Load the main hybrid recommendation system
        hybrid_rec = EnhancedHybridRecommender(
            train_path="data/CleanedDataSet/train_skincare.csv",           # User rating history
            products_path="data/CleanedDataSet/filtered_skincare_products.csv",  # Product catalog
            content_model_path="models/product_embeddings.pkl",           # Pre-trained embeddings
            svd_model_path="models/surprise_svd_model.pkl"               # Collaborative filtering model
        )
        
        # Load alternative recommendation approaches for comparison
        content_rec = ContentBasedRecommender("data/CleanedDataSet/filtered_skincare_products.csv")
        collab_rec = CollaborativeRecommender("data/CleanedDataSet/train_skincare.csv")
        
        return hybrid_rec, content_rec, collab_rec
        
    except Exception as e:
        st.error(f"❌ Error loading recommenders: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None

# Load recommenders with progress indication (only done once per session)
if 'recommenders_loaded' not in st.session_state:
    with st.spinner('🔄 Loading recommender systems...'):
        hybrid_rec, content_rec, collab_rec = load_recommenders()
        if hybrid_rec is not None:
            st.session_state.recommenders_loaded = True
            st.success("✅ All recommenders loaded successfully!")
            # Auto-refresh to clear the loading message
            st.rerun()
else:
    hybrid_rec, content_rec, collab_rec = load_recommenders()

# 加载产品数据
@st.cache_data
def load_products():
    try:
        products_df = pd.read_csv("data/CleanedDataSet/filtered_skincare_products.csv")
        return products_df
    except:
        return pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006'],
            'product_name': ['Moisturizing Cream', 'Cleansing Gel', 'Anti-Aging Serum', 
                           'Sunscreen SPF 50', 'Hydrating Toner', 'Acne Treatment'],
            'brand_name': ['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E', 'Brand F'],
            'tertiary_category': ['Moisturizers', 'Cleansers', 'Serums', 
                                'Sunscreens', 'Toners', 'Treatments'],
            'price_usd': [25.99, 18.50, 32.75, 22.00, 15.99, 28.50]
        })

products_df = load_products()

# ==================== SIDEBAR NAVIGATION ====================

with st.sidebar:
    # Application branding
    st.image("https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=100", width=80)
    st.title("🌸 Skincare Recommender")
    
    # Main navigation buttons
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = 'home'
        # Reset user flow when returning to home
        st.session_state.selected_model = None
        st.session_state.skin_data = {}
        st.rerun()
    
    if st.button("🛍️ All Products", use_container_width=True):
        st.session_state.current_page = 'all_products'
        st.rerun()
    
    st.divider()
    
    # Progress indicators showing user's current state
    if st.session_state.selected_model:
        st.success(f"✅ Model: {st.session_state.selected_model}")
    if st.session_state.skin_data:
        st.success("✅ Skin Profile Complete")
    
    st.caption("Follow the flow: Model → Analysis → Recommendations")

# ==================== PAGE ROUTING AND MAIN CONTENT ====================

if st.session_state.current_page == 'home':
    # ==================== HOME PAGE ====================
    st.header("🌸 Welcome to Your Skincare Journey")
    st.subheader("Get personalized skincare recommendations tailored to your unique needs")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🎯 How It Works:
        
        1. **🤖 Choose Your Recommendation Model**
           - Content-Based: Based on product similarities
           - Collaborative: Based on users like you  
           - Hybrid: Best of both worlds
        
        2. **� Complete Your Skin Analysis**
           - Tell us your skin type and concerns
           - Set your budget preferences
        
        3. **✨ Get Personalized Recommendations**
           - Discover products perfect for your skin
           - See detailed explanations for each recommendation
        """)
        
        # Quick start button
        st.markdown("### Ready to get started?")
        if st.button("🚀 Start Your Skincare Journey", type="primary", use_container_width=True):
            st.session_state.current_page = 'model_selection'
            st.rerun()
            
    with col2:
        st.image("https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400", 
                caption="Discover your perfect skincare routine")
        
        # Stats or info box
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
            st.session_state.current_page = 'skin_analysis'
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
            st.session_state.current_page = 'skin_analysis'
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
        if st.button("Select Hybrid", key="hybrid", use_container_width=True, type="primary"):
            st.session_state.selected_model = "Hybrid"
            st.session_state.current_page = 'skin_analysis'
            st.rerun()

elif st.session_state.current_page == 'skin_analysis':
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

elif st.session_state.current_page == 'recommendations':
    st.header("Your Personalized Skincare Recommendations")
    
    if not st.session_state.skin_data:
        st.warning("Please complete the skin analysis first")
        st.session_state.current_page = 'skin analysis'
        st.rerun()
    
    skin_data = st.session_state.skin_data
    model_type = st.session_state.selected_model.lower() if st.session_state.selected_model else None

    # 🎯 Dynamic User Classification - USE RECOMMENDER'S CACHED DATA
    user_exists = False
    user_rating_count = 0
    
    if hybrid_rec and hasattr(hybrid_rec, 'user_history_cache'):
        # Use the recommender's already-loaded user cache for efficiency
        user_data = hybrid_rec.user_history_cache.get(str(skin_data['user_id']))
        if user_data:
            user_rating_count = len(user_data.get('rated_products', []))
            user_exists = user_rating_count > 0
        
        print(f"👤 User {skin_data['user_id']}: {user_rating_count} ratings in recommender cache")
    else:
        # Fallback: load from file if recommender cache not available
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

    # Display skin profile summary with user classification
    with st.expander("Your Skin Profile"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**User ID:** {skin_data['user_id']}")
            st.write(f"**Skin Type:** {skin_data['skin_type']}")
        with col2:
            st.write(f"**Budget:** {skin_data['budget']}")
            st.write(f"**Number of Products:** {skin_data['num_products']}")
        with col3:
            st.write(f"**Concerns:** {', '.join(skin_data['concerns']) if skin_data['concerns'] else 'None'}")
            # Show user classification
            if not user_exists:
                st.write(f"**User Type:** 🆕 New User (Content 70% + Collab 30%)")
            elif user_rating_count < 10:
                st.write(f"**User Type:** 👤 Existing User - {user_rating_count} ratings (Content 60% + Collab 40%)")
            else:
                st.write(f"**User Type:** 🎯 Experienced User - {user_rating_count} ratings (Content 40% + Collab 60%)")
    
    # Get and display recommendations
    st.subheader("Recommended For You")
    
    # Add key info directly visible to users
    st.write("🎯 Higher concern score = Better for your specific skin issues  ⭐ Higher rating = More customers loved this product  💰 All prices are within your budget")
    st.write("")  # Add some spacing

    # Generate recommendations based on selected model
    recommendations = []
    
    if model_type == 'hybrid' and hybrid_rec:
        # Add skin profile to recommender
        hybrid_rec.add_skin_profile(skin_data['user_id'], {
            'skin_type': skin_data['skin_type'],
            'concerns': skin_data['concerns'],
            'budget': skin_data['budget']
        })
                
        with st.spinner("Generating hybrid recommendations..."):
            try:
                if not user_exists:
                    # 🆕 NEW USERS (0 ratings): Content-heavy approach
                    print(f"🆕 NEW USER: Using content 0.7 + collab 0.3")
                    recommendations = hybrid_rec.get_recommendations_for_new_user(
                        skin_type=skin_data['skin_type'],
                        concerns=skin_data['concerns'], 
                        budget=skin_data['budget'],
                        top_n=skin_data['num_products']
                    )
                elif user_rating_count < 10:
                    # 👤 EXISTING USERS (Few ratings): Balanced approach  
                    print(f"👤 EXISTING USER ({user_rating_count} ratings): Using content 0.6 + collab 0.4")
                    recommendations = hybrid_rec.generate_recommendations(
                        user_id=skin_data['user_id'],
                        top_n=skin_data['num_products'],
                        content_weight=0.6,  # Still content-heavy but less than new users
                        collab_weight=0.4    # More collaborative than new users
                    )
                else:
                    # 🎯 EXPERIENCED USERS (Many ratings): Collaborative-heavy approach
                    print(f"🎯 EXPERIENCED USER ({user_rating_count} ratings): Using content 0.4 + collab 0.6")
                    recommendations = hybrid_rec.generate_recommendations(
                        user_id=skin_data['user_id'],
                        top_n=skin_data['num_products'],
                        content_weight=0.4,  # Less content weight for experienced users
                        collab_weight=0.6    # Heavy collaborative filtering
                    )
            except Exception as e:
                st.error(f"Error generating hybrid recommendations: {e}")
                recommendations = []
                
    elif model_type == 'content-based' and content_rec:
        with st.spinner("Generating content-based recommendations..."):
            try:
                recommendations = content_rec.get_recommendations_for_skin_profile(
                    skin_type=skin_data['skin_type'],
                    concerns=skin_data['concerns'],
                    budget_min=int(skin_data['budget'].split(' - ')[0]) if ' - ' in skin_data['budget'] else 0,
                    budget_max=int(skin_data['budget'].split(' - ')[1]) if ' - ' in skin_data['budget'] else 200,
                    top_n=skin_data['num_products']
                )
            except Exception as e:
                st.error(f"Error generating content-based recommendations: {e}")
                recommendations = []
                
    elif model_type == 'collaborative' and collab_rec:
        with st.spinner("Generating collaborative recommendations..."):
            try:
                if user_exists:
                    recommendations = collab_rec.get_recommendations(
                        skin_data['user_id'], 
                        skin_data['num_products']
                    )
                else:
                    st.warning("👤 User not found in training data. Using content-based fallback.")
                    if content_rec:
                        recommendations = content_rec.get_recommendations_for_skin_profile(
                            skin_type=skin_data['skin_type'],
                            concerns=skin_data['concerns'],
                            budget_min=int(skin_data['budget'].split(' - ')[0]) if ' - ' in skin_data['budget'] else 0,
                            budget_max=int(skin_data['budget'].split(' - ')[1]) if ' - ' in skin_data['budget'] else 200,
                            top_n=skin_data['num_products']
                        )
            except Exception as e:
                st.error(f"Error generating collaborative recommendations: {e}")
                recommendations = []
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            # Handle different recommendation formats
            if isinstance(rec, tuple) and len(rec) >= 3:
                product_id, rating, _ = rec[:3]  # Ignore match_percent
            elif isinstance(rec, tuple) and len(rec) == 2:
                product_id, rating = rec
            else:
                product_id = rec
                rating = None
                
            product_info = products_df[products_df['product_id'].astype(str) == str(product_id)]
            if not product_info.empty:
                product_info = product_info.iloc[0]
                display_recommendation(i, product_info, rating, 
                                     skin_data['user_id'], hybrid_rec if model_type == 'hybrid' else None)
        
        # Check for ingredient conflicts (only for hybrid)
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
                pass  # Skip conflict checking if there's an error
                        
    else:
        st.warning("No recommendations found. Try adjusting your skin profile or selecting a different model.")
    
    # 行动按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("� Update Skin Profile", use_container_width=True):
            st.session_state.current_page = 'skin_analysis'
            st.rerun()
    with col2:
        if st.button("🏠 Start Over", use_container_width=True):
            st.session_state.current_page = 'home'
            st.session_state.skin_data = {}
            st.rerun()
elif st.session_state.current_page == "product_detail":
    # Product detail view for recommended products
    if 'viewing_product' in st.session_state and st.session_state.viewing_product is not None:
        product_id = str(st.session_state.viewing_product)
        
        # Find the full product information
        product_info = products_df[products_df['product_id'] == product_id]
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
                st.write(f"**Price:** ${product.get('price_usd', '-')}")
                
                # --- Highlights ---
                highlights = product.get("highlights", None)
                if highlights and str(highlights).lower() != "nan":
                    st.markdown("### ✨ Highlights")
                    st.markdown(f"- {highlights.replace(';', '<br>- ')}", unsafe_allow_html=True)
                else:
                    st.markdown("### ✨ Highlights")
                    st.write("-")

                # --- Ingredients ---
                ingredients = product.get("ingredients", None)
                st.markdown("### 🧴 Ingredients")
                if ingredients and str(ingredients).lower() != "nan":
                    try:
                        import ast
                        if isinstance(ingredients, str) and ingredients.startswith("["):
                            ing_list = ast.literal_eval(ingredients)
                            ing_list = [i.strip() for i in ing_list if i.strip()]
                        else:
                            ing_list = [ingredients]
                    except Exception:
                        ing_list = [ingredients]
                    for ing in ing_list:
                        st.markdown(f"- {ing}")
                else:
                    st.write("-")
        else:
            st.error("Product not found!")
    else:
        st.error("No product selected for viewing!")

    st.divider()
    if st.button("← Back to Recommendations"):
        # Clear viewing_product when going back to recommendations
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
    
    # Load and display all products
    try:
        df = load_products()
        
        st.subheader(f"Browse Our Collection ({len(df)} Products)")
        
        # Search and filter options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search products...", placeholder="Enter product name, brand, or ingredient")
        
        with col2:
            # Get unique categories for filter
            categories = sorted(df['tertiary_category'].unique()) if 'tertiary_category' in df.columns else []
            selected_category = st.selectbox("Filter by Category", ['All'] + categories)
        
        # Filter products
        filtered_df = df.copy()
        
        if search_term:
            mask = (
                df['product_name'].str.contains(search_term, case=False, na=False) |
                df['brand_name'].str.contains(search_term, case=False, na=False)
            )
            # Add more search fields if they exist
            if 'notable_effects' in df.columns:
                mask |= df['notable_effects'].str.contains(search_term, case=False, na=False)
            filtered_df = df[mask]
        
        if selected_category != 'All' and 'tertiary_category' in df.columns:
            filtered_df = filtered_df[filtered_df['tertiary_category'] == selected_category]
        
        st.write(f"Showing {len(filtered_df)} products")
        
        # Display products in grid
        if not filtered_df.empty:
            # Display products in rows of 3
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