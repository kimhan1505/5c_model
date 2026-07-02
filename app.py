import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ==========================================
# 1. CẤU HÌNH TRANG (LỆNH ĐẦU TIÊN)
# ==========================================
st.set_page_config(layout="wide", page_title="App Dự Báo Phân Loại Nhóm", page_icon="🤖")

# ==========================================
# 2. HÀM DÙNG CHUNG (CACHE)
# ==========================================
@st.cache_data
def load_data(file_bytes):
    """Đọc dữ liệu từ file upload"""
    df = pd.read_csv(file_bytes)
    return df

# Khai báo tập biến dựa theo notebook
TARGET_COL = 'PD'
FEATURE_COLS = ['TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'NL1', 'NL2', 'NL3', 'NL4', 
                'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 'V1', 'V2', 'V3', 'V4', 'V5', 
                'V6', 'TS1', 'TS2', 'TS3', 'TS4']

# ==========================================
# 3. THÀNH PHẦN 1: SIDEBAR - CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    uploaded_file = st.file_uploader("Tải file dữ liệu mẫu (CSV)", type=['csv'], help="Tải lên tệp dữ liệu huấn luyện.")
    
    st.subheader("Tham số mô hình AI")
    model_name = st.selectbox("Thuật toán", ["Random Forest Classifier"], help="Thuật toán được chọn mặc định do notebook không ghi nhận bước huấn luyện cụ thể.")
    
    with st.expander("Tham số nâng cao", expanded=True):
        n_estimators = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=300, value=100, step=10)
        random_state = st.number_input("Random State", value=23, step=1, help="Đồng bộ với random_state=23 trong notebook.")
    
    st.divider()
    train_button = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==========================================
# 4. THÀNH PHẦN 2: HEADER - ĐỊNH HƯỚNG
# ==========================================
st.title("🤖 Ứng Dụng Huấn Luyện & Dự Báo (Machine Learning)")
st.caption("Ứng dụng phân loại tự động dựa trên các bộ chỉ số đo lường (TC, NL, DK, V, TS).")

if uploaded_file is None:
    st.info("👋 Vui lòng tải lên file dữ liệu CSV ở cột bên trái (Sidebar) để bắt đầu.")
    st.stop()

# Nạp dữ liệu
try:
    df = load_data(uploaded_file)
    st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
except Exception as e:
    st.error(f"Lỗi khi đọc file: {e}")
    st.stop()

# Kiểm tra cột bắt buộc
missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
if missing_cols:
    st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc: {', '.join(missing_cols)}")
    st.stop()

st.divider()

# ==========================================
# 5. KHỐI TRAIN (XỬ LÝ KHI BẤM NÚT)
# ==========================================
if train_button:
    with st.spinner("Đang xử lý dữ liệu và huấn luyện mô hình..."):
        # Bước 2 & 3: Lấy X, y và Split như trong notebook
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
        
        # Huấn luyện mô hình
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Đánh giá trên tập test
        y_pred = model.predict(X_test)
        
        # Lưu vào session_state
        st.session_state['is_trained'] = True
        st.session_state['model'] = model
        st.session_state['metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'cm': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }
        st.success("✅ Huấn luyện thành công! Xem chi tiết ở các Tab bên dưới.")

# ==========================================
# 6. TẠO CÁC TABS HIỂN THỊ
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa", 
    "🎯 Kết quả Kiểm định", 
    "🔮 Sử dụng Mô hình"
])

# --- THÀNH PHẦN 3: TỔNG QUAN DỮ LIỆU ---
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Số dòng (Mẫu)", df.shape[0])
    col2.metric("Số cột (Đặc trưng)", df.shape[1])
    col3.metric("Kích thước file (MB)", round(uploaded_file.size / (1024 * 1024), 3))
    
    st.subheader("Dữ liệu thô (5 dòng đầu)")
    with st.container(height=300):
        st.dataframe(df.head(50), use_container_width=True)
        
    st.subheader("Thống kê mô tả (Biến đầu vào & Mục tiêu)")
    cols_to_describe = FEATURE_COLS + [TARGET_COL]
    st.dataframe(df[cols_to_describe].describe(), use_container_width=True)

# --- THÀNH PHẦN 4: TRỰC QUAN HÓA ---
with tab2:
    st.subheader("Phân phối dữ liệu")
    
    # Ưu tiên biểu đồ biến mục tiêu trước
    fig_target = px.bar(df[TARGET_COL].value_counts().reset_index(), 
                        x=TARGET_COL, y='count', title=f"Phân phối Biến Mục Tiêu ({TARGET_COL})", 
                        color=TARGET_COL, text='count')
    st.plotly_chart(fig_target, use_container_width=True)
    
    # Multiselect cho các biến đầu vào (do có đến 24 biến)
    selected_vars = st.multiselect("Chọn tối đa 4 biến đầu vào để vẽ biểu đồ phân phối:", 
                                   options=FEATURE_COLS, 
                                   default=FEATURE_COLS[:4])
    
    if len(selected_vars) > 0:
        cols_viz = st.columns(2)
        for idx, var in enumerate(selected_vars[:4]):  # Giới hạn vẽ 4 đồ thị
            with cols_viz[idx % 2]:
                # Các biến này chủ yếu mang giá trị số nguyên / phân loại dạng thứ bậc (1-5)
                fig = px.histogram(df, x=var, title=f"Phân phối biến {var}", color_discrete_sequence=['#636EFA'])
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

# --- THÀNH PHẦN 5: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH ---
with tab3:
    if not st.session_state.get('is_trained', False):
        st.info("⏳ Vui lòng cấu hình và bấm 'Huấn luyện mô hình' ở Sidebar trước.")
    else:
        st.subheader("Chỉ tiêu Đánh giá (Tập kiểm định 10%)")
        metrics = st.session_state['metrics']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{metrics['precision']:.4f}")
        c3.metric("Recall", f"{metrics['recall']:.4f}")
        c4.metric("F1-Score", f"{metrics['f1']:.4f}")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Ma trận nhầm lẫn (Confusion Matrix)**")
            cm_df = pd.DataFrame(metrics['cm'])
            st.dataframe(cm_df, use_container_width=True)
            
        with col_m2:
            st.markdown("**Báo cáo phân loại (Classification Report)**")
            report_df = pd.DataFrame(metrics['report']).transpose()
            st.dataframe(report_df, use_container_width=True)

# --- THÀNH PHẦN 6: SỬ DỤNG MÔ HÌNH ---
with tab4:
    if not st.session_state.get('is_trained', False):
        st.info("⏳ Vui lòng huấn luyện mô hình để kích hoạt chức năng dự báo.")
    else:
        pred_mode = st.radio("Chọn phương thức dự báo:", ["Nhập tay từng mẫu", "Tải file hàng loạt (Batch)"], horizontal=True)
        model = st.session_state['model']
        
        if pred_mode == "Nhập tay từng mẫu":
            st.markdown("### 📝 Nhập thông số")
            with st.form("predict_form"):
                cols_form = st.columns(4)
                input_data = {}
                
                # Render 24 biến X
                for idx, col_name in enumerate(FEATURE_COLS):
                    default_val = int(df[col_name].median())
                    min_val = int(df[col_name].min())
                    max_val = int(df[col_name].max())
                    
                    with cols_form[idx % 4]:
                        input_data[col_name] = st.number_input(f"{col_name}", min_value=min_val, max_value=max_val, value=default_val, step=1)
                
                submit_pred = st.form_submit_button("Dự báo", type="primary")
                
                if submit_pred:
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0].max() * 100
                    st.success(f"🎯 **Kết quả dự báo (Lớp {TARGET_COL}):** {prediction} | **Độ tin cậy:** {prob:.2f}%")
                    
        else:
            st.markdown("### 📁 Dự báo hàng loạt từ File CSV")
            st.caption("File phải có chứa đầy đủ các cột tương tự tập đầu vào.")
            pred_file = st.file_uploader("Tải lên file dữ liệu cần dự báo (Không cần cột PD)", type=['csv'], key="pred_file")
            
            if pred_file is not None:
                new_df = pd.read_csv(pred_file)
                missing_pred_cols = [c for c in FEATURE_COLS if c not in new_df.columns]
                
                if missing_pred_cols:
                    st.error(f"File thiếu các cột: {', '.join(missing_pred_cols)}")
                else:
                    pred_X = new_df[FEATURE_COLS]
                    predictions = model.predict(pred_X)
                    
                    res_df = new_df.copy()
                    res_df[f'Predicted_{TARGET_COL}'] = predictions
                    
                    st.success(f"Đã chấm điểm thành công cho {len(res_df)} dòng dữ liệu!")
                    st.dataframe(res_df.head(10), use_container_width=True)
                    
                    # Nút tải xuống CSV
                    csv = res_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Tải xuống kết quả",
                        data=csv,
                        file_name="ket_qua_du_bao.csv",
                        mime="text/csv",
                    )
