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
def display_recommendation(index, product, rating, match_percent):
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(f"{index}. {product.get('product_name', 'Product')}")
            st.write(f"**Brand:** {product.get('brand_name', 'Unknown')}")
            st.write(f"**Category:** {product.get('tertiary_category', 'Unknown')}")
        
        with col2:
            st.metric("Rating", f"{rating:.1f}/5")
            safe_percent = round(min(100, max(0, match_percent)))
            st.progress(safe_percent / 100, text=f"{safe_percent}% match")
        
        with col3:
            st.metric("Price", f"${product.get('price_usd', 0):.2f}")
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
            
            if st.button("Select & Get Recommendations", key=f"select_{product['product_id']}", 
                        use_container_width=True):
                st.session_state.selected_product = product['product_id']
                st.session_state.current_page = 'skin analysis'
                st.rerun()

# 初始化推荐系统
@st.cache_resource
def load_recommenders():
    try:
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
        st.error(f"Error loading recommenders: {e}")
        return None, None, None

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
    if st.session_state.selected_product:
        product_info = products_df[products_df['product_id'] == st.session_state.selected_product]
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
            st.write(f"**User ID:** {skin_data['user_id']}")
            st.write(f"**Skin Type:** {skin_data['skin_type']}")
        with col2:
            st.write(f"**Budget:** {skin_data['budget']}")
            st.write(f"**Number of Products:** {skin_data['num_products']}")
        with col3:
            st.write(f"**Concerns:** {', '.join(skin_data['concerns']) if skin_data['concerns'] else 'None'}")
            st.write(f"**Model:** {model_type.capitalize()}")
    
    # 获取和显示推荐
    st.subheader("Recommended For You")
    
        # 获取推荐
    if model_type == 'hybrid' and hybrid_rec:
        # 添加皮肤数据到推荐器 - 确保使用正确的键名
        skin_profile_data = {
            'user_id': skin_data['user_id'],
            'skin_type': skin_data['skin_type'],
            'concerns': skin_data['concerns'],  # 确保这是列表
            'budget': skin_data['budget']
        }
        
        # 调试信息
        st.write(f"Debug: Adding skin profile for user {skin_data['user_id']}")
        st.write(f"Debug: Skin type = {skin_data['skin_type']}")
        st.write(f"Debug: Concerns = {skin_data['concerns']}")
        st.write(f"Debug: Budget = {skin_data['budget']}")
        
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
                
                # 显示推荐结果
                if recommendations:
                    for i, (product_id, rating, match_percent) in enumerate(recommendations, 1):
                        product_info = products_df[products_df['product_id'].astype(str) == product_id]
                        if not product_info.empty:
                            product_info = product_info.iloc[0]
                            display_recommendation(i, product_info, rating, match_percent)
                else:
                    st.warning("No recommendations found. Try adjusting your skin profile.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                # 回退到不使用皮肤过滤
                recommendations = hybrid_rec.generate_recommendations(
                    skin_data['user_id'], 
                    skin_data['num_products']
                )
    elif model_type == 'content' and content_rec:
        with st.spinner("🔍 Finding products that match your skin needs..."):
            recommendations = content_rec.get_recommendations(
                skin_data['user_id'],
                skin_data['skin_type'],
                skin_data['concerns'],
                skin_data['budget'],
                skin_data['num_products']
            )
            
            for i, rec in enumerate(recommendations, 1):
                display_recommendation(i, rec, rec['rating'], rec['match_percent'])
    
    elif model_type == 'collab' and collab_rec:
        with st.spinner("🔍 Discovering community favorites for your skin type..."):
            recommendations = collab_rec.get_recommendations(
                skin_data['user_id'],
                skin_data['num_products']
            )
            
            for i, rec in enumerate(recommendations, 1):
                product_info = products_df[products_df['product_id'].astype(str) == rec['product_id']]
                if not product_info.empty:
                    product_info = product_info.iloc[0].to_dict()
                    product_info.update(rec)
                    display_recommendation(i, product_info, rec['rating'], rec['match_percent'])
    
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