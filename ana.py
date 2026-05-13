import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import timedelta, datetime
from sklearn.linear_model import LinearRegression

# 페이지 기본 설정
st.set_page_config(page_title="비트코인(BTC) 대시보드", page_icon="🪙", layout="wide")

# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    try:
        # 파일명을 coin.csv로 변경 (구분자 ; 유지)
        df = pd.read_csv('coin.csv', sep=';')
        
        # 날짜 데이터 변환 (timeOpen 기준)
        df['Date'] = pd.to_datetime(df['timeOpen']).dt.tz_localize(None)
        
        # 날짜순 정렬
        df = df.sort_values('Date').reset_index(drop=True)
        
        # 이동평균선 계산 (7일, 30일)
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()
        
        return df
    except FileNotFoundError:
        st.error("현재 폴더에 'coin.csv' 파일이 없습니다. 파일명을 확인해 주세요.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# 선형 회귀를 이용한 내일 가격 예측 함수
def predict_next_day(df):
    # 날짜를 숫자형으로 변환하여 학습 데이터 생성
    df_pred = df[['Date', 'close']].copy()
    
    # [수정] datetime 객체를 Unix 타임스탬프(숫자)로 안전하게 변환
    # pd.to_numeric은 나노초 단위이므로 10^9로 나눠 초 단위로 맞춤
    df_pred['Timestamp'] = pd.to_numeric(df_pred['Date']) / 10**9
    
    X = df_pred[['Timestamp']].values
    y = df_pred['close'].values
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 내일 날짜 계산 및 예측
    next_day = df_pred['Date'].max() + timedelta(days=1)
    next_timestamp = np.array([[next_day.timestamp()]])
    prediction = model.predict(next_timestamp)[0]
    
    return prediction, next_day

# 메인 함수
def main():
    st.title("🪙 비트코인(BTC) 가격 분석 및 예측 대시보드")
    st.markdown("`coin.csv` 데이터를 분석하고 선형 회귀 모델을 통해 내일의 가격을 예측합니다.")

    # 데이터 로드
    df = load_data()
    
    if df.empty:
        st.warning("분석할 데이터가 없습니다. CSV 파일이 올바른 경로에 있는지 확인하세요.")
        return

    # --- 사이드바 (필터) ---
    st.sidebar.header("🔍 데이터 필터")
    
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    # 날짜 범위 선택기
    selected_date_range = st.sidebar.date_input(
        "날짜 범위 선택",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df.loc[mask]
    else:
        filtered_df = df.copy()

    # --- 예측 섹션 (상단 배치) ---
    st.markdown("---")
    st.subheader("🔮 AI 내일 가격 예측 (선형 회귀)")
    
    # 전체 데이터를 기반으로 학습하여 예측
    pred_price, pred_date = predict_next_day(df)
    today_price = df.iloc[-1]['close']
    diff = pred_price - today_price
    diff_pct = (diff / today_price) * 100
    
    p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
    
    with p_col1:
        st.write(f"**예측 날짜:** {pred_date.date()}")
        st.write(f"**오늘 종가:** ₩ {today_price:,.0f}")
    
    with p_col2:
        color = "red" if diff > 0 else "blue"
        trend = "상승 📈" if diff > 0 else "하락 📉"
        st.markdown(f"**예측 결과:** <span style='color:{color}; font-size:20px; font-weight:bold;'>{trend}</span>", unsafe_allow_html=True)
        st.write(f"**예상 가격:** ₩ {pred_price:,.0f}")

    with p_col3:
        st.info(f"선형 회귀 모델 분석 결과, 내일 가격은 오늘 대비 약 **{abs(diff_pct):.2f}% {trend.split()[0]}**할 것으로 예측됩니다. (단순 추세 분석이므로 참고용으로만 사용하세요.)")

    st.markdown("---")

    # --- 상단 KPI 지표 ---
    st.subheader("💡 핵심 요약 (최신 데이터 기준)")
    
    if not filtered_df.empty:
        latest_data = filtered_df.iloc[-1]
        prev_data = filtered_df.iloc[-2] if len(filtered_df) > 1 else latest_data
        
        price_change = latest_data['close'] - prev_data['close']
        price_change_pct = (price_change / prev_data['close']) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="현재 종가 (Close)",
                value=f"₩ {latest_data['close']:,.0f}",
                delta=f"{price_change:,.0f} ({price_change_pct:.2f}%)"
            )
            
        with col2:
            st.metric(label="시가총액", value=f"₩ {latest_data['marketCap']:,.0f}")
            
        with col3:
            st.metric(label="24시간 거래량", value=f"₩ {latest_data['volume']:,.0f}")
            
        with col4:
            st.metric(label="유통량", value=f"{latest_data['circulatingSupply']:,.0f} BTC")

    # --- 메인 차트: 캔들스틱 및 이동평균선 ---
    st.subheader("📈 가격 추이 및 기술적 분석")
    
    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=filtered_df['Date'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name="가격",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))

    fig_candle.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA7'], 
                                    line=dict(color='orange', width=1.5), name='MA7'))
    fig_candle.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA30'], 
                                    line=dict(color='blue', width=1.5), name='MA30'))

    fig_candle.update_layout(
        xaxis_title="날짜", yaxis_title="가격 (KRW)",
        height=500, xaxis_rangeslider_visible=False,
        template="plotly_white", margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # --- 하단 차트 ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 거래량 추이")
        fig_volume = px.bar(filtered_df, x='Date', y='volume', color_discrete_sequence=['#8884d8'])
        fig_volume.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_volume, use_container_width=True)

    with col2:
        st.subheader("📋 변동성 분석")
        filtered_df['volatility'] = filtered_df['high'] - filtered_df['low']
        fig_vol = px.line(filtered_df, x='Date', y='volatility', color_discrete_sequence=['#ff7300'])
        fig_vol.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_vol, use_container_width=True)

    # --- 데이터 테이블 ---
    with st.expander("원본 데이터 상세 보기"):
        display_cols = ['Date', 'open', 'high', 'low', 'close', 'volume', 'marketCap']
        styled_df = filtered_df[display_cols].copy()
        for col in ['open', 'high', 'low', 'close', 'volume', 'marketCap']:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x:,.0f}")
        st.dataframe(styled_df.set_index('Date'), use_container_width=True)

if __name__ == "__main__":
    main()
