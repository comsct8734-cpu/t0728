import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io

# ==========================================
# 0. Page Config & Session State Initialization
# ==========================================
st.set_page_config(
    page_title="CSV 데이터로 배우는 선형회귀 실험실",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화 (탭 이동 시 학습 모델 및 데이터 유지)
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'simple_model_results' not in st.session_state:
    st.session_state['simple_model_results'] = None
if 'multi_model_results' not in st.session_state:
    st.session_state['multi_model_results'] = None

# ==========================================
# Helper Functions (함수 단위 구현)
# ==========================================

def generate_sample_data() -> pd.DataFrame:
    """학생 실습용 미세먼지(PM2.5) 기상 예제 데이터를 생성합니다."""
    np.random.seed(42)
    n_samples = 120
    
    temperature = np.random.uniform(-5, 32, n_samples)  # 기온 (-5 ~ 32 ℃)
    humidity = np.random.uniform(20, 90, n_samples)      # 습도 (20 ~ 90 %)
    wind_speed = np.random.uniform(0.5, 8.0, n_samples)  # 풍속 (0.5 ~ 8.0 m/s)
    rainfall = np.random.choice([0, 0, 0, 0.5, 2.0, 10.0, 25.0], n_samples) # 강수량 (mm)
    
    # PM2.5 생성 물리 모델 (바람 불면 감소, 습도 높으면 상승, 비 오면 세척 효과)
    pm25 = (
        35.0 
        + 0.6 * temperature 
        + 0.4 * humidity 
        - 4.2 * wind_speed 
        - 1.5 * rainfall 
        + np.random.normal(0, 7.0, n_samples)
    )
    pm25 = np.clip(pm25, 5, 150) # 음수 방지 및 현실적 범위 설정
    
    df = pd.DataFrame({
        'temperature': np.round(temperature, 1),
        'humidity': np.round(humidity, 1),
        'wind_speed': np.round(wind_speed, 1),
        'rainfall': np.round(rainfall, 1),
        'pm25': np.round(pm25, 1)
    })
    return df

def load_csv(uploaded_file) -> pd.DataFrame:
    """CSV 파일을 UTF-8 및 CP949 인코딩으로 안전하게 로드합니다."""
    try:
        bytes_data = uploaded_file.read()
        try:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='cp949')
        return df
    except Exception as e:
        st.error(f"❌ CSV 파일을 읽는 중 오류가 발생했습니다: {str(e)}")
        return None

def validate_data(df: pd.DataFrame):
    """데이터의 행 수 및 숫자형 열 개수를 검증합니다."""
    if df is None:
        return False, "데이터가 존재하지 않습니다."
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return False, "선형회귀를 수행하려면 최소 2개 이상의 숫자형(Numeric) 열이 필요합니다."
    
    if len(df) < 10:
        return False, "데이터 행 수(샘플 수)가 10개 미만입니다. 최소 10개 이상의 데이터가 필요합니다."
        
    return True, f"검증 완료: 총 {len(df)}행, 숫자형 열 {len(numeric_cols)}개"

def calculate_metrics(y_true, y_pred, num_features):
    """모델 평가 지표 (R2, Adj-R2, MAE, MSE, RMSE)를 계산합니다."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    
    # 조정된 R² (Adjusted R-squared)
    if n - num_features - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - num_features - 1)
    else:
        adj_r2 = r2
        
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'R2': r2,
        'Adj_R2': adj_r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }

# ==========================================
# Sidebar UI
# ==========================================
with st.sidebar:
    st.title("🧪 인공지능 기초")
    st.caption("고등학교 2학년 선형회귀 실습 도구")
    st.markdown("---")
    
    st.subheader("📌 주요 용어 요약")
    st.markdown("""
    * **독립변수 ($X$)**: 원인이 되는 입력 변수
    * **종속변수 ($y$)**: 결과가 되는 출력 변수
    * **기울기 ($b_1$)**: $X$가 1 증가할 때 $y$의 변화량
    * **절편 ($b_0$)**: $X=0$일 때 $y$의 기본값
    * **예측값 ($\hat{y}$)**: 모델이 계산한 $y$ 값
    * **잔차 (Residual)**: 실제값($y$) - 예측값($\hat{y}$)
    """)
    st.markdown("---")
    st.info("💡 Tip: 1단계부터 순서대로 학습을 진행해보세요.")

# Main Title
st.title("📊 CSV 데이터로 배우는 선형회귀 실험실")
st.markdown("---")

# ==========================================
# Tabs Setup
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. 학습 안내",
    "2. CSV 데이터 업로드",
    "3. 데이터 탐색",
    "4. 단순선형회귀",
    "5. 다중선형회귀",
    "6. 모델 평가 및 비교"
])

# ------------------------------------------
# Tab 1: 학습 안내
# ------------------------------------------
with tab1:
    st.header("📘 선형회귀 기본 개념 익히기")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 🔍 회귀(Regression)와 선형회귀")
        st.write("""
        * **회귀**: 여러 변수 사이의 관계를 파악하여 연속적인 숫자를 예측하는 기법입니다.
        * **선형회귀**: 데이터의 경향성을 가장 잘 나타내는 **'직선(Line)'**을 찾는 알고리즘입니다.
        """)
        
        st.success("### 🎯 입력과 출력 변수")
        st.write("""
        * **독립변수 ($X$)**: 원인이 되는 데이터 (예: 기온, 풍속, 공부 시간)
        * **종속변수 ($y$)**: 결과가 되는 데이터 (예: 미세먼지 농도, 시험 점수)
        """)
        
    with col2:
        st.warning("### 📐 단순선형회귀 vs 다중선형회귀")
        st.write("""
        * **단순선형회귀**: 독립변수 $X$가 **1개**일 때 사용합니다.
        * **다중선형회귀**: 독립변수 $X$가 **2개 이상**일 때 사용합니다.
        """)
        
        st.error("### ⚠️ 상관관계 vs 인과관계")
        st.write("""
        * **상관관계**: 두 변수가 함께 변하는 경향 (예: 아이스크림 판매량과 수영장 사고 수)
        * **인과관계**: 한 변수가 다른 변수의 직접적인 원인임
        * **주의**: 상관관계가 높다고 해서 반드시 원인과 결과(인과관계)인 것은 아닙니다!
        """)

    st.markdown("---")
    st.subheader("🧮 선형회귀 수식")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**단순선형회귀 수식 ($X$가 1개)**")
        st.latex(r"\hat{y} = b_0 + b_1 x")
        st.caption("👉 $b_1$: 기울기(Coefficient), $b_0$: 절편(Intercept)")
    with c2:
        st.markdown("**다중선형회귀 수식 ($X$가 $n$개)**")
        st.latex(r"\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n")
        st.caption("👉 각 $x_i$ 마다 고유한 기울기 $b_i$가 부여됩니다.")

    st.markdown("---")
    st.subheader("🎯 오차와 잔차(Residual)")
    st.markdown("""
    - **실제값 ($y$)**: 실제 측정된 데이터
    - **예측값 ($\hat{y}$)**: 회귀선 상에서 모델이 예측한 값
    - **잔차 (Residual)** = **실제값 ($y$) - 예측값 ($\hat{y}$)**
    - 선형회귀는 데이터 전체의 **잔차 제곱의 합(SSE)을 최소화**하는 최적의 직선을 찾습니다.
    """)

    with st.expander("❓ [탐구 질문] 선형회귀 기본 개념"):
        st.markdown("""
        1. 독립변수와 종속변수를 실생활 예시로 각각 1가지씩 들어보세요.
        2. 단순선형회귀와 다중선형회귀의 가장 핵심적인 차이점은 무엇인가요?
        3. 상관관계가 높지만 인과관계는 아닌 사례를 떠올려보세요.
        """)

# ------------------------------------------
# Tab 2: CSV 데이터 업로드
# ------------------------------------------
with tab2:
    st.header("📂 데이터 파일 불러오기")
    
    col_up, col_sample = st.columns([2, 1])
    
    with col_up:
        uploaded_file = st.file_uploader("사용할 CSV 파일을 업로드하세요 (UTF-8, CP949 지원)", type=["csv"])
    
    with col_sample:
        st.write(" **실습용 데이터가 없으신가요?**")
        sample_df = generate_sample_data()
        csv_bytes = sample_df.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            label="📥 미세먼지 예제 CSV 다운로드",
            data=csv_bytes,
            file_name="pm25_weather_data.csv",
            mime="text/csv",
            help="기상 변수와 미세먼지(PM2.5) 데이터가 포함된 샘플 CSV를 받습니다."
        )
        if st.button("🚀 예제 데이터 바로 적용하기"):
            st.session_state['df'] = sample_df
            st.success("예제 데이터가 적용되었습니다!")

    if uploaded_file is not None:
        df_loaded = load_csv(uploaded_file)
        if df_loaded is not None:
            st.session_state['df'] = df_loaded

    df = st.session_state['df']

    if df is not None:
        st.markdown("---")
        is_valid, msg = validate_data(df)
        
        if is_valid:
            st.success(f"✅ 데이터 로드 성공! ({msg})")
        else:
            st.error(f"⚠️ {msg}")
        
        if len(df) < 30 and len(df) >= 10:
            st.warning("⚠️ 데이터 행 수가 30개 미만으로 적습니다. 모델 학습 결과 해석 시 주의하세요!")

        st.subheader("🔍 업로드된 데이터 미리보기")
        st.dataframe(df.head(10), use_container_width=True)
        
        # 데이터 요약 정보
        st.subheader("📊 데이터 기본 정보")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전체 행 수", f"{df.shape[0]} 개")
        m2.metric("전체 열 수", f"{df.shape[1]} 개")
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        m3.metric("숫자형 열", f"{len(num_cols)} 개")
        m4.metric("문자형 열", f"{len(cat_cols)} 개")
        
        st.subheader("📌 열별 데이터 타입 및 결측치 현황")
        info_df = pd.DataFrame({
            "데이터 타입": df.dtypes.astype(str),
            "결측치 개수": df.isnull().sum(),
            "결측치 비율(%)": np.round((df.isnull().sum() / len(df)) * 100, 2)
        })
        st.dataframe(info_df.T, use_container_width=True)
        
    else:
        st.info("👆 상단에서 CSV 파일을 업로드하거나 [예제 데이터 바로 적용하기] 버튼을 눌러주세요.")

    with st.expander("❓ [탐구 질문] CSV 데이터 업로드"):
        st.markdown("""
        1. 내가 수집한 데이터에서 독립변수로 적합한 열과 종속변수로 적합한 열은 무엇인가요?
        2. 결측치(Missing Value)가 존재하는 경우 선형회귀 모델에 어떤 영향을 줄까요?
        """)

# ------------------------------------------
# Tab 3: 데이터 탐색 (EDA)
# ------------------------------------------
with tab3:
    st.header("📊 탐색적 데이터 분석 (EDA)")
    
    df = st.session_state['df']
    
    if df is None:
        st.warning("먼저 '2. CSV 데이터 업로드' 탭에서 데이터를 업로드해주세요.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) < 2:
            st.error("숫자형 변수가 2개 이상 필요합니다.")
        else:
            st.subheader("1️⃣ 기술통계량 확인")
            st.dataframe(df[num_cols].describe().T, use_container_width=True)
            
            st.markdown("---")
            st.subheader("2️⃣ 변수별 히스토그램 & 두 변수 산점도")
            
            col_eda1, col_eda2 = st.columns(2)
            
            with col_eda1:
                st.markdown("**히스토그램 (단일 변수 분포)**")
                selected_hist_col = st.selectbox("분포를 볼 변수 선택", num_cols, key="hist_select")
                fig_hist = px.histogram(
                    df, x=selected_hist_col, 
                    title=f"{selected_hist_col} 분포",
                    marginal="rug",
                    color_discrete_sequence=['#4C78A8']
                )
                fig_hist.update_layout(xaxis_title=selected_hist_col, yaxis_title="빈도수")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col_eda2:
                st.markdown("**산점도 (두 변수 간 관계)**")
                x_scatter = st.selectbox("X축 변수 선택", num_cols, index=0, key="scat_x")
                y_scatter = st.selectbox("Y축 변수 선택", num_cols, index=min(1, len(num_cols)-1), key="scat_y")
                
                fig_scat = px.scatter(
                    df, x=x_scatter, y=y_scatter,
                    title=f"{x_scatter} vs {y_scatter} 산점도",
                    color_discrete_sequence=['#E15759'],
                    trendline="ols"
                )
                fig_scat.update_layout(xaxis_title=x_scatter, yaxis_title=y_scatter)
                st.plotly_chart(fig_scat, use_container_width=True)

            # 산점도 체크리스트 질문
            st.info(f"""
            **💡 [{x_scatter}] 와 [{y_scatter}] 의 산점도 관찰 질문:**
            1. 두 변수는 양(+)의 관계인가요, 음(-)의 관계인가요?
            2. 데이터 점들이 하나의 직선에 가깝게 모여 있나요?
            3. 다른 점들과 유난히 떨어진 이상치(Outlier)가 보이나요?
            4. 두 변수의 관계가 관찰된다고 해서 바로 '원인과 결과(인과관계)'라고 할 수 있을까요?
            """)

            st.markdown("---")
            st.subheader("3️⃣ 상관계수(Correlation) 분석")
            
            corr_df = df[num_cols].corr()
            
            col_corr1, col_corr2 = st.columns([1, 1])
            with col_corr1:
                st.markdown("**상관계수 표**")
                st.dataframe(corr_df.style.background_gradient(cmap='coolwarm').format("{:.3f}"), use_container_width=True)
            
            with col_corr2:
                st.markdown("**상관계수 히트맵**")
                fig_heatmap = px.imshow(
                    corr_df, 
                    text_auto=".2f", 
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="상관계수 히트맵"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            st.success("""
            **📖 상관계수($r$)의 범위와 해석 기준:**
            * **$+1.0$에 가까움**: 강한 양의 상관관계 ($X$가 증가하면 $y$도 증가)
            * **$0.0$ 근처**: 상관관계 거의 없음 ($X$와 $y$ 사이에 직선 경향성이 없음)
            * **$-1.0$에 가까움**: 강한 음의 상관관계 ($X$가 증가하면 $y$는 감소)
            """)

    with st.expander("❓ [탐구 질문] 데이터 탐색"):
        st.markdown("""
        1. 산점도는 두 변수 사이에 어떤 관계가 있다고 보여주나요?
        2. 상관계수가 높다고 해서 반드시 선형회귀 모델의 성능이 완벽할까요?
        """)

# ------------------------------------------
# Tab 4: 단순선형회귀
# ------------------------------------------
with tab4:
    st.header("📉 단순선형회귀 (Simple Linear Regression)")
    st.caption("독립변수 1개를 사용하여 종속변수를 예측합니다.")
    
    df = st.session_state['df']
    
    if df is None:
        st.warning("먼저 '2. CSV 데이터 업로드' 탭에서 데이터를 업로드해주세요.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) < 2:
            st.error("숫자형 변수가 최소 2개 이상 필요합니다.")
        else:
            col_sel1, col_sel2, col_sel3 = st.columns(3)
            with col_sel1:
                x_col = st.selectbox("독립변수 (X) 선택", num_cols, index=0, key="simple_x")
            with col_sel2:
                # y는 X와 다른 열을 기본 선택
                default_y_idx = 1 if len(num_cols) > 1 else 0
                y_col = st.selectbox("종속변수 (y) 선택", num_cols, index=default_y_idx, key="simple_y")
            with col_sel3:
                test_size = st.slider("테스트 데이터 비율 (%)", min_value=10, max_value=40, value=20, step=5) / 100.0

            if x_col == y_col:
                st.error("⚠️ 독립변수(X)와 종속변수(y)는 같은 열일 수 없습니다. 서로 다른 열을 선택하세요.")
            else:
                # 1. 결측치 제거 및 데이터 준비
                clean_df = df[[x_col, y_col]].dropna()
                X = clean_df[[x_col]]
                y = clean_df[y_col]
                
                # 2. Train / Test 분리
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # 3. 모델 학습
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # 4. 예측 및 평가
                y_pred_test = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred_test, num_features=1)
                
                # 결과 세션 저장
                coef = model.coef_[0]
                intercept = model.intercept_
                corr_val = clean_df.corr().iloc[0, 1]
                
                st.session_state['simple_model_results'] = {
                    'x_col': x_col,
                    'y_col': y_col,
                    'coef': coef,
                    'intercept': intercept,
                    'metrics': metrics,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred_test': y_pred_test
                }

                st.markdown("---")
                st.subheader("📊 1. 학습 결과 요약")
                
                c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                c_m1.metric("학습 데이터 수", f"{len(X_train)}개")
                c_m2.metric("테스트 데이터 수", f"{len(X_test)}개")
                c_m3.metric("기울기 (Slope, b1)", f"{coef:.4f}")
                c_m4.metric("절편 (Intercept, b0)", f"{intercept:.4f}")

                # 회귀식 표시
                st.info(f"📐 **학습된 단순선형회귀식**:  \n`예측 {y_col} = {coef:.4f} × {x_col} + ({intercept:.4f})`")

                # 기울기 자동 해석 문구
                direction = "증가" if coef > 0 else "감소"
                st.success(f"💡 **기울기 자동 해석**: `{x_col}`이(가) **1** 만큼 증가할 때, `{y_col}` 예측값은 평균적으로 약 **{abs(coef):.4f}** 만큼 **{direction}**합니다. (단, 이는 데이터상의 경향성일 뿐 인과관계를 의미하지 않습니다.)")

                st.markdown("---")
                st.subheader("📈 2. 회귀선 및 잔차 시각화")
                
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    st.markdown("**산점도와 추세선 (Train vs Test)**")
                    fig_simple = go.Figure()
                    
                    # Train Data
                    fig_simple.add_trace(go.Scatter(
                        x=X_train[x_col], y=y_train, mode='markers', name='학습 데이터(Train)',
                        marker=dict(color='#1f77b4', opacity=0.6)
                    ))
                    # Test Data
                    fig_simple.add_trace(go.Scatter(
                        x=X_test[x_col], y=y_test, mode='markers', name='테스트 데이터(Test)',
                        marker=dict(color='#ff7f0e', size=8)
                    ))
                    # Regression Line
                    x_range = np.linspace(X[x_col].min(), X[x_col].max(), 100)
                    y_range = model.predict(x_range.reshape(-1, 1))
                    fig_simple.add_trace(go.Scatter(
                        x=x_range, y=y_range, mode='lines', name='선형 회귀선',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig_simple.update_layout(xaxis_title=x_col, yaxis_title=y_col, title="데이터와 추세선")
                    st.plotly_chart(fig_simple, use_container_width=True)

                with col_p2:
                    st.markdown("**테스트 데이터의 잔차(Residual) 시각화**")
                    fig_res = go.Figure()
                    
                    # Test 데이터 실제 점
                    fig_res.add_trace(go.Scatter(
                        x=X_test[x_col], y=y_test, mode='markers', name='실제값(Test)',
                        marker=dict(color='#ff7f0e', size=8)
                    ))
                    # 회귀선
                    fig_res.add_trace(go.Scatter(
                        x=x_range, y=y_range, mode='lines', name='선형 회귀선',
                        line=dict(color='red', width=2)
                    ))
                    # 잔차 선 그리기
                    for x_i, y_i, y_p in zip(X_test[x_col], y_test, y_pred_test):
                        fig_res.add_trace(go.Scatter(
                            x=[x_i, x_i], y=[y_i, y_p], mode='lines',
                            line=dict(color='gray', width=1, dash='dot'),
                            showlegend=False
                        ))
                        
                    fig_res.update_layout(xaxis_title=x_col, yaxis_title=y_col, title="실제값에서 회귀선까지의 거리 (잔차)")
                    st.plotly_chart(fig_res, use_container_width=True)

                st.markdown("---")
                st.subheader("🔮 3. 새로운 데이터 예측 시뮬레이터")
                
                min_val = float(X[x_col].min())
                max_val = float(X[x_col].max())
                mean_val = float(X[x_col].mean())
                
                input_x = st.number_input(
                    f"새로운 `{x_col}` 값을 입력하세요:",
                    min_value=min_val - (max_val - min_val),
                    max_value=max_val + (max_val - min_val),
                    value=mean_val
                )
                
                pred_y = model.predict([[input_x]])[0]
                
                st.metric(label=f"예측된 `{y_col}` 값", value=f"{pred_y:.2f}")
                st.warning("⚠️ 이 값은 데이터에서 학습한 선형적인 경향을 이용한 예측값이며 실제값과 다를 수 있습니다.")
                
                if pred_y < 0:
                    st.error("💡 **선형회귀의 한계**: 예측 결과가 음수(-)로 나타났습니다. 미세먼지나 가격 등 현실에서는 0 이하가 될 수 없는 변수라도 선형회귀 직선은 음수를 예측할 수 있는 한계가 있습니다.")

    with st.expander("❓ [탐구 질문] 단순선형회귀"):
        st.markdown("""
        1. 회귀선은 모든 데이터 점을 반드시 지나가나요? 지나가지 않는다면 그 이유는 무엇일까요?
        2. 기울기의 부호(양수/음수)는 두 변수 사이의 어떤 관점과 연결되나요?
        3. 잔차가 양수인 경우, 실제값과 예측값 중 어느 것이 더 큰가요?
        """)

# ------------------------------------------
# Tab 5: 다중선형회귀
# ------------------------------------------
with tab5:
    st.header("📉 다중선형회귀 (Multiple Linear Regression)")
    st.caption("여러 개의 독립변수(X)를 동시에 활용하여 종속변수(y)를 예측합니다.")
    
    df = st.session_state['df']
    
    if df is None:
        st.warning("먼저 '2. CSV 데이터 업로드' 탭에서 데이터를 업로드해주세요.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) < 3:
            st.error("다중선형회귀를 진행하려면 최소 3개 이상의 숫자형 열이 필요합니다.")
        else:
            col_m_y, col_m_opt = st.columns([1, 2])
            
            with col_m_y:
                y_multi_col = st.selectbox("종속변수 (y) 선택", num_cols, index=len(num_cols)-1, key="multi_y")
            
            # y 변수를 제외한 후보 X 변수들
            available_x_cols = [c for c in num_cols if c != y_multi_col]
            
            with col_m_opt:
                x_multi_cols = st.multiselect(
                    "독립변수 (X) 선택 (최소 2개 이상)",
                    available_x_cols,
                    default=available_x_cols[:2],
                    key="multi_x"
                )
            
            use_std = st.checkbox("⚙️ 입력 데이터 표준화(StandardScaler) 적용하기", value=False)
            test_size_m = st.slider("테스트 데이터 비율 (%) ", min_value=10, max_value=40, value=20, step=5, key="multi_slider") / 100.0

            if len(x_multi_cols) < 2:
                st.warning("⚠️ 다중선형회귀를 실행하려면 독립변수(X)를 최소 2개 이상 선택해야 합니다.")
            else:
                # 1. 데이터 준비
                clean_m_df = df[x_multi_cols + [y_multi_col]].dropna()
                X_m = clean_m_df[x_multi_cols]
                y_m = clean_m_df[y_multi_col]
                
                # 2. Train / Test Split
                X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y_m, test_size=test_size_m, random_state=42)
                
                # 3. Pipeline / Model Fit
                if use_std:
                    model_m = Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', LinearRegression())
                    ])
                    model_m.fit(X_m_train, y_m_train)
                    coefs = model_m.named_steps['regressor'].coef_
                    intercept_m = model_m.named_steps['regressor'].intercept_
                else:
                    model_m = LinearRegression()
                    model_m.fit(X_m_train, y_m_train)
                    coefs = model_m.coef_
                    intercept_m = model_m.intercept_
                    
                # 4. Predict
                y_m_pred_test = model_m.predict(X_m_test)
                metrics_m = calculate_metrics(y_m_test, y_m_pred_test, num_features=len(x_multi_cols))
                
                st.session_state['multi_model_results'] = {
                    'x_cols': x_multi_cols,
                    'y_col': y_multi_col,
                    'coefs': coefs,
                    'intercept': intercept_m,
                    'metrics': metrics_m,
                    'X_test': X_m_test,
                    'y_test': y_m_test,
                    'y_pred_test': y_m_pred_test,
                    'use_std': use_std
                }

                st.markdown("---")
                st.subheader("📊 1. 다중선형회귀 식 및 계수")
                
                # 회귀식 문자열 생성
                eq_terms = [f"({c:.4f} × {col})" for c, col in zip(coefs, x_multi_cols)]
                eq_str = f"예측 {y_multi_col} = {' + '.join(eq_terms)} + ({intercept_m:.4f})"
                st.info(f"📐 **학습된 다중선형회귀식**:\n`{eq_str}`")

                st.warning("⚠️ **회귀계수 해석 주의사항**: 다중선형회귀의 회귀계수는 **다른 입력 변수들이 일정하다고 가정했을 때** 해당 변수가 1만큼 변할 때의 예측값 변화를 의미합니다. 변수마다 단위(℃, %, m/s 등)가 다르면 계수의 크기만으로 어떤 변수가 더 중요한지 직접 비교하기 어렵습니다.")

                if use_std:
                    st.success("✨ **표준화(StandardScaler) 적용됨**: 모든 입력 변수의 평균을 0, 표준편차를 1로 변환했으므로, 표준화 후의 계수 크기를 통해 변수들의 상대적 영향력을 직접 비교할 수 있습니다!")

                # 회귀계수 표 및 막대그래프
                coef_df = pd.DataFrame({
                    "독립변수(X)": x_multi_cols,
                    "회귀계수(Coefficient)": coefs
                })
                
                col_c1, col_c2 = st.columns([1, 1])
                with col_c1:
                    st.markdown("**변수별 회귀계수 표**")
                    st.dataframe(coef_df.style.format({"회귀계수(Coefficient)": "{:.4f}"}), use_container_width=True)
                    st.metric("절편 (Intercept)", f"{intercept_m:.4f}")
                    
                with col_c2:
                    st.markdown("**회귀계수 크기 비교 막대그래프**")
                    fig_coef = px.bar(
                        coef_df, x="독립변수(X)", y="회귀계수(Coefficient)",
                        color="회귀계수(Coefficient)",
                        title="변수별 회귀계수 시각화",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_coef, use_container_width=True)

                st.markdown("---")
                st.subheader("🔮 2. 다중선형회귀 예측 시뮬레이터")
                st.write("각 변수의 값을 조절하여 종속변수 예측값이 어떻게 달라지는지 확인해보세요.")
                
                input_vals = {}
                col_inputs = st.columns(min(len(x_multi_cols), 4))
                
                for idx, col_name in enumerate(x_multi_cols):
                    c_target = col_inputs[idx % 4]
                    mean_val = float(X_m[col_name].mean())
                    min_v = float(X_m[col_name].min())
                    max_v = float(X_m[col_name].max())
                    input_vals[col_name] = c_target.number_input(
                        f"`{col_name}` 입력",
                        min_value=min_v - (max_v - min_v),
                        max_value=max_v + (max_v - min_v),
                        value=mean_val,
                        key=f"multi_in_{col_name}"
                    )

                input_single_df = pd.DataFrame([input_vals])
                pred_multi_y = model_m.predict(input_single_df)[0]
                
                st.metric(label=f"다중 모델의 예측 `{y_multi_col}` 값", value=f"{pred_multi_y:.2f}")

    with st.expander("❓ [탐구 질문] 다중선형회귀"):
        st.markdown("""
        1. 독립변수를 1개에서 여러 개로 늘렸을 때 모델의 예측 능력은 어떻게 변했나요?
        2. 변수들의 단위가 서로 다를 때 회귀계수의 크기만으로 중요도를 비교하면 왜 안 되나요?
        3. 표준화(Standardization)를 적용하면 회귀계수의 값은 어떻게 바뀌나요?
        """)

# ------------------------------------------
# Tab 6: 모델 평가 및 비교
# ------------------------------------------
with tab6:
    st.header("⚖️ 모델 평가 및 성능 비교")
    
    simple_res = st.session_state.get('simple_model_results')
    multi_res = st.session_state.get('multi_model_results')
    
    if simple_res is None or multi_res is None:
        st.warning("⚠️ 단순선형회귀(4단계)와 다중선형회귀(5단계)를 모두 먼저 학습시켜주세요!")
    else:
        st.subheader("1️⃣ 모델 성능 비교표")
        
        s_m = simple_res['metrics']
        m_m = multi_res['metrics']
        
        comp_df = pd.DataFrame({
            "비교 항목": ["사용한 독립변수(X)", "R² (결정계수)", "조정된 R² (Adj-R²)", "MAE", "MSE", "RMSE"],
            "단순선형회귀": [
                f"{simple_res['x_col']}",
                f"{s_m['R2']:.4f}",
                f"{s_m['Adj_R2']:.4f}",
                f"{s_m['MAE']:.4f}",
                f"{s_m['MSE']:.4f}",
                f"{s_m['RMSE']:.4f}"
            ],
            "다중선형회귀": [
                f"{', '.join(multi_res['x_cols'])}",
                f"{m_m['R2']:.4f}",
                f"{m_m['Adj_R2']:.4f}",
                f"{m_m['MAE']:.4f}",
                f"{m_m['MSE']:.4f}",
                f"{m_m['RMSE']:.4f}"
            ]
        })
        
        st.table(comp_df)

        st.markdown("---")
        st.subheader("📚 평가 지표 지식 상자")
        
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.info("""
            * **MAE (Mean Absolute Error, 평균 절대 오차)**: 실제값과 예측값 차이의 절댓값 평균.
            * **MSE (Mean Squared Error, 평균 제곱 오차)**: 오차를 제곱하여 평균한 값. **큰 오차에 더 큰 벌점**을 줍니다.
            * **RMSE (Root Mean Squared Error, 근사 평균 제곱 오차)**: MSE에 제곱근을 씌워 **원래 y와 동일한 단위**로 맞춘 오차 지표입니다.
            """)
        with c_exp2:
            st.success("""
            * **$R^2$ (결정계수, Coefficient of Determination)**: 모델이 종속변수의 전체 변동성을 몇 %나 설명하는지 나타냄 (1.0에 가까울수록 성능 우수).
            * **조정된 $R^2$ (Adjusted $R^2$)**: 쓸모없는 변수를 무작정 많이 추가할 때 $R^2$가 인위적으로 높아지는 현상을 **방지하기 위해 보정한 지표**.
            """)

        st.markdown("---")
        st.subheader("2️⃣ 진단 그래프 분석 (다중선형회귀 기준)")
        
        col_g1, col_g2 = st.columns(2)
        
        y_true_m = multi_res['y_test']
        y_pred_m = multi_res['y_pred_test']
        residuals_m = y_true_m - y_pred_m
        
        with col_g1:
            st.markdown("**① 실제값 vs 예측값 산점도**")
            fig_act_pred = go.Figure()
            fig_act_pred.add_trace(go.Scatter(
                x=y_true_m, y=y_pred_m, mode='markers',
                marker=dict(color='#2ca02c', size=8),
                name='예측 데이터'
            ))
            # 1:1 대각 기준선 (이상적 예측)
            min_val = min(y_true_m.min(), y_pred_m.min())
            max_val = max(y_true_m.max(), y_pred_m.max())
            fig_act_pred.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', line=dict(color='red', dash='dash'),
                name='이상적 예측선 (y=x)'
            ))
            fig_act_pred.update_layout(
                xaxis_title="실제값 (Actual)",
                yaxis_title="예측값 (Predicted)",
                title="실제값 vs 예측값"
            )
            st.plotly_chart(fig_act_pred, use_container_width=True)
            
        with col_g2:
            st.markdown("**② 잔차 산점도 (Residual Plot)**")
            fig_res_scat = go.Figure()
            fig_res_scat.add_trace(go.Scatter(
                x=y_pred_m, y=residuals_m, mode='markers',
                marker=dict(color='#d62728', size=8),
                name='잔차'
            ))
            fig_res_scat.add_hline(y=0, line_dash="dash", line_color="black")
            fig_res_scat.update_layout(
                xaxis_title="예측값 (Predicted)",
                yaxis_title="잔차 (Actual - Predicted)",
                title="예측값 vs 잔차"
            )
            st.plotly_chart(fig_res_scat, use_container_width=True)

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            st.markdown("**③ 잔차 분포 히스토그램**")
            fig_res_hist = px.histogram(
                x=residuals_m, nbins=15, title="잔차 분포",
                color_discrete_sequence=['#9467bd']
            )
            fig_res_hist.update_layout(xaxis_title="잔차", yaxis_title="빈도수")
            st.plotly_chart(fig_res_hist, use_container_width=True)

        with col_g4:
            st.markdown("💡 **진단 그래프 자동 해석 가이드**")
            st.write("""
            - **실제값 vs 예측값**: 데이터 점들이 **빨간 점선(y=x)**에 가까이 모여 있을수록 모델의 예측 정확도가 뛰어납니다.
            - **잔차 산점도**: 잔차가 0을 중심으로 **위아래 무작위(Random)로 고르게 분포**해야 선형회귀 가정이 잘 만족된 것입니다.
            - **잔차 히스토그램**: 잔차의 모양이 **0을 중심으로 종 모양(정규분포)**에 가까울수록 좋은 모델입니다.
            - **곡선 패턴 발견 시**: 잔차 분포에 U자나 특정 곡선 패턴이 나타난다면 변수 간의 관계가 '비선형(Non-linear)'일 가능성이 높습니다.
            """)

        st.markdown("---")
        st.subheader("📢 최종 모델 선택 고찰")
        
        r2_diff = m_m['R2'] - s_m['R2']
        rmse_diff = s_m['RMSE'] - m_m['RMSE']
        
        st.success(f"""
        **💡 분석 요약:**
        * 단순 모델 대비 다중 모델의 $R^2$ 변화량: **{r2_diff:+.4f}**
        * 단순 모델 대비 다중 모델의 RMSE(오차) 감소량: **{rmse_diff:+.4f}**
        
        **📌 결론 도출 팁:**
        단순히 변수가 많다고 무조건 '좋은 모델'은 아닙니다. 다중선형회귀의 $R^2$가 높아졌더라도 **조정된 $R^2$나 RMSE 오차가 실질적으로 개선되었는지** 확인해야 합니다. 모델의 오차 성능과 **단순성(설명 용이성)** 사이의 균형을 맞추는 것이 중요합니다.
        """)

    with st.expander("❓ [탐구 질문] 모델 평가 및 비교"):
        st.markdown("""
        1. 독립변수를 무조건 많이 넣으면 $R^2$ 지표는 어떻게 변하나요? 이것이 항상 좋은 모델을 의미할까요?
        2. $R^2$가 높아졌는데 RMSE 오차가 오히려 커지는 상황이 발생할 수도 있을까요? 이유를 생각해봅시다.
        3. 이상치(Outlier)를 제거한 후 모델을 다시 학습시키면 회귀식과 평가 지표는 어떻게 변할까요?
        4. 우리가 완성한 이 미세먼지 예측 선형회귀 모델을 실제 기상청 예보에 바로 적용해도 될까요? 한계점은 무엇일까요?
        """)
