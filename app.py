import streamlit as st
import pandas as pd
import numpy as np
from utils.recommender import EnhancedHybridRecommender, ContentBasedRecommender, CollaborativeRecommender
import plotly.express as px
from streamlit_option_menu import option_menu

# 页面设置
st.set_page_config(
    page_title="Skincare Recommendation System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
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

# 辅助函数定义
def display_recommendation(index, product, rating, match_percent, user_id=None, recommender=None):
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(f"{index}. {product.get('product_name', 'Product')}")
            st.write(f"**Brand:** {product.get('brand_name', 'Unknown')}")
            st.write(f"**Category:** {product.get('tertiary_category', 'Unknown')}")
            
            # Show debug info if available
            if user_id and recommender:
                try:
                    product_id = product.get('product_id')
                    if hasattr(product_id, 'item'):
                        product_id = product_id.item()
                    debug_info = recommender.get_recommendation_debug_info(str(product_id), str(user_id))
                    
                    # Show why this product was recommended
                    with st.expander("🔍 Why was this recommended?", expanded=False):
                        compat = debug_info.get('compatibility', {})
                        st.write(f"**Skin Type:** {compat.get('skin_type_status', 'N/A')}")
                        st.write(f"**Concerns:** {compat.get('concern_status', 'N/A')}")
                        st.write(f"**Budget:** {compat.get('budget_status', 'N/A')}")
                        st.write(f"**Compatibility Score:** {compat.get('final_multiplier', 'N/A')}")
                        
                        prod_info = debug_info.get('product_info', {})
                        if prod_info.get('detected_skin_types'):
                            st.write(f"**Product targets:** {', '.join(prod_info['detected_skin_types'])} skin")
                        if prod_info.get('detected_concerns'):
                            st.write(f"**Product addresses:** {', '.join(prod_info['detected_concerns'])}")
                except Exception as e:
                    pass  # Skip debug info if there's an error
        
        with col2:
            st.metric("Rating", f"{rating:.1f}/5")
            safe_percent = round(min(100, max(0, match_percent)))
            st.progress(safe_percent / 100, text=f"{safe_percent}% match")
        
        with col3:
            st.metric("Price", f"${product.get('price_usd', 0):.2f}")
            if st.button("View Details", key=f"btn_{index}"):
                # Store the product for viewing details WITHOUT changing the selected product
                product_id = product.get('product_id')
                if hasattr(product_id, 'item'):
                    product_id = product_id.item()
                # Use a separate variable for viewing details
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
            
            if st.button("Select & Get Recommendations", key=f"select_{product['product_id']}", 
                        use_container_width=True):
                # Ensure we store a scalar value, not a pandas Series
                product_id = product['product_id']
                if hasattr(product_id, 'item'):
                    product_id = product_id.item()  # Extract scalar from pandas Series
                st.session_state.selected_product = product_id
                st.session_state.current_page = 'skin analysis'
                st.rerun()

# 初始化推荐系统
@st.cache_resource
def load_recommenders():
    try:
        # Load all components silently
        hybrid_rec = EnhancedHybridRecommender(
            train_path="data/CleanedDataSet/train_skincare.csv",
            products_path="data/CleanedDataSet/filtered_skincare_products.csv",
            content_model_path="models/product_embeddings.pkl",
            svd_model_path="models/surprise_svd_model.pkl"
        )
        
        content_rec = ContentBasedRecommender("data/CleanedDataSet/filtered_skincare_products.csv")
        collab_rec = CollaborativeRecommender("data/CleanedDataSet/train_skincare.csv")
        
        return hybrid_rec, content_rec, collab_rec
        
    except Exception as e:
        st.error(f"❌ Error loading recommenders: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None

# Load recommenders and show status only once
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

# 侧边栏导航 - 简化版本
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=100", width=80)
    st.title("🌸 Skincare Recommender")
    
    # 简化的导航，主要用于快速跳转
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = 'home'
        st.rerun()
    
    if st.button("💫 Get Recommendations", use_container_width=True):
        st.session_state.current_page = 'skin analysis'
        st.rerun()
    
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.current_page = 'about'
        st.rerun()
    
    st.divider()
    st.caption("Quick Navigation")

# 页面路由逻辑
if st.session_state.current_page == 'home':
    st.header("🌸 Discover Your Perfect Skincare")
    st.subheader("Browse our curated collection or get personalized recommendations")
    
    # 搜索和筛选功能
    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input("🔍 Search products", placeholder="Enter product name or brand")
    with col2:
        selected_category = st.selectbox("Filter by category", 
                                       ["All"] + list(products_df['tertiary_category'].unique()))
    
    # 筛选产品
    filtered_products = products_df.copy()
    if search_query:
        filtered_products = filtered_products[
            filtered_products['product_name'].str.contains(search_query, case=False) |
            filtered_products['brand_name'].str.contains(search_query, case=False)
        ]
    if selected_category != "All":
        filtered_products = filtered_products[filtered_products['tertiary_category'] == selected_category]
    
    # 显示产品网格
    st.write(f"**Showing {len(filtered_products)} products**")
    
    if len(filtered_products) > 0:
        cols = st.columns(3)
        for idx, (_, product) in enumerate(filtered_products.iterrows()):
            display_product_card(product, cols[idx % 3])
    else:
        st.info("No products found. Try adjusting your search filters.")
    
    # 个性化推荐按钮
    st.divider()
    st.write("### Not sure what to choose?")
    if st.button("✨ Get Personalized Recommendations Based on Your Skin Needs", 
                use_container_width=True, type="primary"):
        st.session_state.current_page = 'skin analysis'
        st.rerun()

elif st.session_state.current_page == 'skin analysis':
    st.header("Tell Us About Your Skin")
    
    if st.button("← Back to Products"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    # 如果是从产品页面来的，显示选中的产品
    if st.session_state.selected_product is not None:
        # Handle both pandas Series and simple values
        selected_id = st.session_state.selected_product
        if hasattr(selected_id, 'iloc'):
            # It's a pandas object, extract the value
            selected_id = selected_id
        
        product_info = products_df[products_df['product_id'] == selected_id]
        if not product_info.empty:
            product_info = product_info.iloc[0]
            st.info(f"**Selected Product:** {product_info['product_name']} by {product_info['brand_name']}")
    
    with st.form("skin_analysis_form"):
        user_id = st.text_input("User ID", placeholder="Enter your user ID", help="Required for personalized recommendations")
        
        col1, col2 = st.columns(2)
        with col1:
            skin_type = st.selectbox("Skin Type", ["", "Dry", "Oily", "Combination", "Normal", "Sensitive"],
                                   help="Select your primary skin type")
        with col2:
            budget = st.selectbox("Budget Preference", ["", "Under $25", "$25-$50", "$50-$100", "Over $100", "No budget limit"],
                                help="Your preferred price range")
        
        concerns = st.multiselect(
            "Main Skin Concerns",
            ["Acne", "Redness", "Dehydration", "Aging", "Pigmentation", "Sensitivity", "Dullness", "Large pores"],
            help="Select all that apply to you"
        )
        
        num_products = st.slider("Number of Recommendations", 1, 10, 5,
                               help="How many products would you like to see?")
        
        submitted = st.form_submit_button("🎯 Get Personalized Recommendations", type="primary")
        
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
                st.session_state.current_page = 'select approach'
                st.rerun()

elif st.session_state.current_page == 'select approach':
    st.header("Choose Your Recommendation Style")
    
    # Show the selected product info
    if st.session_state.selected_product is not None:
        selected_product_id = st.session_state.selected_product
        if hasattr(selected_product_id, 'item'):
            product_id = selected_product_id.item()
        else:
            product_id = str(selected_product_id)
        
        # Find and display the selected product
        product_info = products_df[products_df['product_id'] == product_id]
        if not product_info.empty:
            product = product_info.iloc[0]
            st.info(f"🎯 **Selected Product:** {product['product_name']} by {product['brand_name']}")
            st.write(f"**Category:** {product['tertiary_category']} | **Price:** ${product['price_usd']:.2f}")
    
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

    # --- Check if user exists in training set (memory efficient) ---
    try:
        # Only load the author_id column to save memory
        train_df = pd.read_csv("data/CleanedDataSet/train_skincare.csv", 
                             usecols=['author_id'], 
                             low_memory=False,
                             chunksize=10000)
        existing_users = set()
        for chunk in train_df:
            existing_users.update(chunk["author_id"].astype(str).unique())
        user_exists = str(skin_data['user_id']) in existing_users
    except Exception as e:
        st.warning(f"Could not check user existence: {e}")
        # Default to treating as new user if we can't check
        user_exists = False

    if model_type == 'hybrid':
        if user_exists:
            st.info("👥 Using Collaborative Filtering (existing user found in dataset)")
        else:
            st.info("🤖 Using Content-Based Filtering (new user, fallback to same-category recommendations)")
    
    # Display skin profile summary
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
            st.write(f"**Model:** {model_type.capitalize()}")
    
    # If coming from a product page, display the selected product
    if st.session_state.selected_product is not None:
        # Ensure selected_product is a string, not a pandas Series
        if hasattr(st.session_state.selected_product, 'iloc'):
            selected_id = st.session_state.selected_product
        else:
            selected_id = st.session_state.selected_product
            
        product_info = products_df[products_df['product_id'] == selected_id]
        if not product_info.empty:
            product_info = product_info.iloc[0]
            st.info(f"**Selected Product:** {product_info['product_name']} by {product_info['brand_name']}")
    # Get and display recommendations
    st.subheader("Recommended For You")
    
        # 获取推荐
    if model_type == 'hybrid' and hybrid_rec:
        # 添加皮肤数据到推荐器 - 确保使用正确的键名
        # With this:
        hybrid_rec.add_skin_profile(skin_data['user_id'], {
            'skin_type': skin_data['skin_type'],
            'concerns': skin_data['concerns'],
            'budget': skin_data['budget']
        })
        
        # 调试信息
        st.write(f"Debug: Adding skin profile for user {skin_data['user_id']}")
        st.write(f"Debug: Skin type = {skin_data['skin_type']}")
        st.write(f"Debug: Concerns = {skin_data['concerns']}")
        st.write(f"Debug: Budget = {skin_data['budget']}")
                
        with st.spinner("Generating hybrid recommendations..."):
            try:
                if not user_exists:
                    # NEW USER: Use full hybrid recommendations (no category restrictions)
                    st.write("🚀 Using Full Hybrid Content-Based Filtering (searching all categories)")
                    
                    recommendations = hybrid_rec.enhanced_demo_recommendations(
                        user_id=skin_data['user_id'],
                        top_n=skin_data['num_products'],
                        content_weight=0.4,
                        collab_weight=0.6,
                        selected_product_id=st.session_state.selected_product if st.session_state.selected_product is not None else None
                    )
                else:
                    # EXISTING USER: Use hybrid approach
                    recommendations = hybrid_rec.generate_recommendations(
                        skin_data['user_id'], 
                        skin_data['num_products'],
                        content_weight=0.4,
                        collab_weight=0.6
                    )

                # 显示推荐结果
                if recommendations:
                    for i, (product_id, rating, match_percent) in enumerate(recommendations, 1):
                        product_info = products_df[products_df['product_id'].astype(str) == product_id]
                        if not product_info.empty:
                            product_info = product_info.iloc[0]
                            display_recommendation(i, product_info, rating, match_percent, 
                                                 skin_data['user_id'], hybrid_rec)
                else:
                    st.warning("No recommendations found. Try adjusting your skin profile.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                # Fallback to basic recommendations
                try:
                    recommendations = hybrid_rec.generate_recommendations(
                        skin_data['user_id'], 
                        skin_data['num_products']
                    )
                except Exception as fallback_error:
                    st.error(f"Fallback also failed: {fallback_error}")
                    recommendations = []
    
    # 行动按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Get New Recommendations", use_container_width=True):
            st.session_state.current_page = 'select approach'
            st.rerun()
    with col2:
        if st.button("🏠 Start Over", use_container_width=True):
            st.session_state.current_page = 'home'
            st.session_state.selected_product = None
            st.session_state.skin_data = {}
            st.rerun()
elif st.session_state.current_page == "product_detail":
    # Check if we're viewing a recommended product or the original selected product
    if 'viewing_product' in st.session_state and st.session_state.viewing_product is not None:
        display_product_id = st.session_state.viewing_product
        # Clear viewing_product so it doesn't interfere with normal flow
        is_viewing_recommended = True
    else:
        display_product_id = st.session_state.selected_product
        is_viewing_recommended = False
    
    if display_product_id is not None:
        # Get the product ID (handle both string and pandas Series)
        if hasattr(display_product_id, 'item'):
            product_id = display_product_id.item()
        else:
            product_id = str(display_product_id)
        
        # Find the full product information
        product_info = products_df[products_df['product_id'] == product_id]
        if not product_info.empty:
            product = product_info.iloc[0].to_dict()
            
            # Show different header if viewing recommended product
            if is_viewing_recommended:
                st.header(f"📋 {product.get('product_name', '-')}")
                st.info("🔍 You're viewing details of a recommended product")
            else:
                st.header(product.get("product_name", "-"))
            
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

if __name__ == "__main__":
    pass