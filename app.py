import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import os

##########################################
# 0. 기본 설정
##########################################

st.set_page_config(
    page_title="임장스쿨 : 서울 부동산실거래가 요인 분석",
    layout="wide"
)

st.title("임장스쿨 : 서울 부동산실거래가 요인 분석")

# 데이터 파일 위치
DATA_DIR = "data"
MAIN_FILE = os.path.join(DATA_DIR, "merged_realestate_transport_plus.csv")

# 국토부 아파트 실거래 (24/25) 파일 경로
APT24_FILE = os.path.join(DATA_DIR, "아파트(매매)_실거래가(24)_20251128161815.xlsx")
APT25_FILE = os.path.join(DATA_DIR, "아파트(매매)_실거래가(25)_20251128161716.xlsx")

##########################################
# 1. 데이터 로드
##########################################

@st.cache_data
def load_main():
    df = pd.read_csv(MAIN_FILE, encoding="utf-8")
    return df

df = load_main()

##########################################
# 1-1. 국토부 아파트 실거래(24/25) 로드
##########################################

def _extract_gu_from_sigungu(s: str) -> str | None:
    """'서울특별시 동대문구 장안동' -> '동대문구'"""
    parts = str(s).split()
    for p in parts:
        if p.endswith("구"):
            return p
    return None

@st.cache_data
def load_apt_trade():
    """국토부 엑셀(24,25)을 읽어서 [자치구, 연도]별 평균가격/거래건수 반환"""
    def read_one(path: str, year: int) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_excel(
            path,
            skiprows=12,          # 안내문/검색조건 행 스킵 (국토부 양식 기준)
            usecols=["시군구", "거래금액(만원)"]
        )
        df = df.dropna(subset=["시군구", "거래금액(만원)"])
        # '55,000' -> 55000
        df["거래금액(만원)"] = (
            df["거래금액(만원)"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        df["거래금액(만원)"] = pd.to_numeric(df["거래금액(만원)"], errors="coerce")
        df = df.dropna(subset=["거래금액(만원)"])
        df["자치구"] = df["시군구"].apply(_extract_gu_from_sigungu)
        df["연도"] = year
        return df[["자치구", "연도", "거래금액(만원)"]].dropna(subset=["자치구"])

    try:
        apt24 = read_one(APT24_FILE, 2024)
        apt25 = read_one(APT25_FILE, 2025)
    except Exception as e:
        return None, str(e)

    all_df = pd.concat([apt24, apt25], ignore_index=True)

    grouped = (
        all_df
        .groupby(["자치구", "연도"])["거래금액(만원)"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "평균가격(만원)", "count": "거래건수"})
    )
    return grouped, None

apt_stats, apt_err = load_apt_trade()

##########################################
# 2. 주요 컬럼 자동 탐색
##########################################

# 자치구 컬럼 추정
gu_col = None
for c in df.columns:
    if c in ["DistrictName", "자치구명", "자치구", "구명"]:
        gu_col = c
        break

# 면적 컬럼 추정
area_col = None
for cand in ["건물면적(㎡)", "전용면적(㎡)", "대지면적(㎡)", "토지면적(㎡)"]:
    if cand in df.columns:
        area_col = cand
        break

# 가격 컬럼 추정
price_col = None
for cand in ["Price (10,000 KRW)", "거래금액", "거래금액(만원)"]:
    if cand in df.columns:
        price_col = cand
        break

# 버스/지하철/병원 거리 컬럼
bus_dist_col = "최근접버스정류장_거리_m" if "최근접버스정류장_거리_m" in df.columns else None
sub_dist_col = "지하철거리(m)" if "지하철거리(m)" in df.columns else None
hosp_dist_col = "최근접병원_거리_m" if "최근접병원_거리_m" in df.columns else None

# 소득 컬럼
income_col = "구_월평균소득_원" if "구_월평균소득_원" in df.columns else None

##########################################
# 3. 사이드바 필터 UI (= 임장 조건)
##########################################

st.sidebar.header("필터 (임장 조건)")

# 자치구 필터
if gu_col is not None:
    gu_list = sorted(df[gu_col].dropna().unique())
    selected_gu = st.sidebar.multiselect("자치구 선택", gu_list, default=gu_list)
else:
    selected_gu = None

# 면적 필터
if area_col is not None:
    min_area = float(df[area_col].min())
    max_area = float(df[area_col].max())
    area_min, area_max = st.sidebar.slider(
        f"면적 범위 선택 ({area_col})",
        min_value=round(min_area, 2),
        max_value=round(max_area, 2),
        value=(round(min_area, 2), round(max_area, 2))
    )
else:
    area_min, area_max = None, None

# 가격 필터
if price_col is not None:
    min_price = float(df[price_col].min())
    max_price = float(df[price_col].max())
    price_min, price_max = st.sidebar.slider(
        f"가격 범위 선택 ({price_col})",
        min_value=round(min_price, 0),
        max_value=round(max_price, 0),
        value=(round(min_price, 0), round(max_price, 0))
    )
else:
    price_min, price_max = None, None

# 버스 거리 필터
if bus_dist_col is not None:
    min_bus = float(df[bus_dist_col].min())
    max_bus = float(df[bus_dist_col].max())
    bus_min, bus_max = st.sidebar.slider(
        "최근접 버스정류장 거리 (m)",
        min_value=round(min_bus, 1),
        max_value=round(max_bus, 1),
        value=(round(min_bus, 1), round(max_bus, 1))
    )
else:
    bus_min, bus_max = None, None

# 지하철 거리 필터
if sub_dist_col is not None:
    min_sub = float(df[sub_dist_col].min())
    max_sub = float(df[sub_dist_col].max())
    sub_min, sub_max = st.sidebar.slider(
        "최근접 지하철역 거리 (m)",
        min_value=round(min_sub, 1),
        max_value=round(max_sub, 1),
        value=(round(min_sub, 1), round(max_sub, 1))
    )
else:
    sub_min, sub_max = None, None

# 병원 거리 필터
if hosp_dist_col is not None:
    min_hosp = float(df[hosp_dist_col].min())
    max_hosp = float(df[hosp_dist_col].max())
    hosp_min, hosp_max = st.sidebar.slider(
        "최근접 병원 거리 (m)",
        min_value=round(min_hosp, 1),
        max_value=round(max_hosp, 1),
        value=(round(min_hosp, 1), round(max_hosp, 1))
    )
else:
    hosp_min, hosp_max = None, None

# 구 소득 필터
if income_col is not None:
    min_inc = float(df[income_col].min())
    max_inc = float(df[income_col].max())
    inc_min, inc_max = st.sidebar.slider(
        f"구 월평균 소득 범위 (원) ({income_col})",
        min_value=round(min_inc, -3),
        max_value=round(max_inc, -3),
        value=(round(min_inc, -3), round(max_inc, -3))
    )
else:
    inc_min, inc_max = None, None

##########################################
# 4. 필터 적용 (임장 조건 반영)
##########################################

cond = pd.Series(True, index=df.index)

if selected_gu is not None and len(selected_gu) > 0 and gu_col is not None:
    cond &= df[gu_col].isin(selected_gu)

if area_col is not None:
    cond &= (df[area_col] >= area_min) & (df[area_col] <= area_max)

if price_col is not None:
    cond &= (df[price_col] >= price_min) & (df[price_col] <= price_max)

if bus_min is not None and bus_dist_col is not None:
    cond &= (df[bus_dist_col] >= bus_min) & (df[bus_dist_col] <= bus_max)

if sub_min is not None and sub_dist_col is not None:
    cond &= (df[sub_dist_col] >= sub_min) & (df[sub_dist_col] <= sub_max)

if hosp_min is not None and hosp_dist_col is not None:
    cond &= (df[hosp_dist_col] >= hosp_min) & (df[hosp_dist_col] <= hosp_max)

if inc_min is not None and income_col is not None:
    cond &= (df[income_col] >= inc_min) & (df[income_col] <= inc_max)

filtered = df[cond].copy()

##########################################
# 5. 탭 구성
##########################################

tab1, tab2, tab3, tab4 = st.tabs([
    "① 요약/표",
    "② 가격 vs 교통·병원·소득",
    "③ 지도에서 위치 보기",
    "④ 임장 분석 (2024 vs 2025 아파트)"
])

############################
# 탭 1: 요약 / 표
############################
with tab1:
    st.subheader("필터 적용된 실거래 목록")

    st.write(f"표시 건수: **{len(filtered)}** / 전체 {len(df)}")
    st.dataframe(filtered.head(200), width="stretch")

    st.markdown("---")
    st.subheader("기본 통계 요약")

    cols_for_desc = []
    for c in [price_col, area_col, bus_dist_col, sub_dist_col, hosp_dist_col, income_col]:
        if c is not None and c in filtered.columns:
            cols_for_desc.append(c)

    if len(cols_for_desc) > 0:
        st.dataframe(filtered[cols_for_desc].describe().T, width="stretch")
    else:
        st.info("요약할 수 있는 수치형 컬럼이 없습니다.")

############################
# 탭 2: 가격 vs 교통·병원·소득 (상관계수 개선)
############################
with tab2:
    st.subheader("가격과 교통·병원 접근성 및 소득의 관계")

    if price_col is None:
        st.warning("⚠ 가격 컬럼을 찾을 수 없어 분석을 진행할 수 없습니다.")
    else:
        # 2-1. 버스 거리 vs 가격
        if bus_dist_col in filtered.columns:
            st.markdown("### (1) 버스정류장 거리 vs 가격")
            bus_price_df = filtered[[bus_dist_col, price_col]].dropna()

            if not bus_price_df.empty:
                bus_chart = (
                    alt.Chart(bus_price_df)
                    .mark_circle(opacity=0.4)
                    .encode(
                        x=alt.X(
                            bus_dist_col,
                            title="버스정류장 최근접 거리 (m)",
                            scale=alt.Scale(domain=[0, float(bus_price_df[bus_dist_col].max())])
                        ),
                        y=alt.Y(
                            price_col,
                            title=price_col,
                            scale=alt.Scale(domain=[0, float(bus_price_df[price_col].max())])
                        )
                    )
                )
                st.altair_chart(bus_chart, width="stretch")
            else:
                st.info("표시할 데이터가 없습니다.")

        # 2-2. 지하철 거리 vs 가격
        if sub_dist_col in filtered.columns:
            st.markdown("### (2) 지하철역 거리 vs 가격")
            sub_price_df = filtered[[sub_dist_col, price_col]].dropna()

            if not sub_price_df.empty:
                sub_chart = (
                    alt.Chart(sub_price_df)
                    .mark_circle(opacity=0.4, color="#ff7f0e")
                    .encode(
                        x=alt.X(
                            sub_dist_col,
                            title="지하철역 최근접 거리 (m)",
                            scale=alt.Scale(domain=[0, float(sub_price_df[sub_dist_col].max())])
                        ),
                        y=alt.Y(
                            price_col,
                            title=price_col,
                            scale=alt.Scale(domain=[0, float(sub_price_df[price_col].max())])
                        )
                    )
                )
                st.altair_chart(sub_chart, width="stretch")
            else:
                st.info("표시할 데이터가 없습니다.")

        # 2-3. 병원 거리 vs 가격
        if hosp_dist_col in filtered.columns:
            st.markdown("### (3) 병원 거리 vs 가격")
            hosp_price_df = filtered[[hosp_dist_col, price_col]].dropna()

            if not hosp_price_df.empty:
                hosp_chart = (
                    alt.Chart(hosp_price_df)
                    .mark_circle(opacity=0.4, color="#2ca02c")
                    .encode(
                        x=alt.X(
                            hosp_dist_col,
                            title="병원 최근접 거리 (m)",
                            scale=alt.Scale(domain=[0, float(hosp_price_df[hosp_dist_col].max())])
                        ),
                        y=alt.Y(
                            price_col,
                            title=price_col,
                            scale=alt.Scale(domain=[0, float(hosp_price_df[price_col].max())])
                        )
                    )
                )
                st.altair_chart(hosp_chart, width="stretch")
            else:
                st.info("표시할 데이터가 없습니다.")

        # 2-4. 구 소득 vs 가격
        if income_col in filtered.columns:
            st.markdown("### (4) 자치구 월평균 소득 vs 가격")
            inc_price_df = filtered[[income_col, price_col]].dropna()

            if not inc_price_df.empty:
                inc_chart = (
                    alt.Chart(inc_price_df)
                    .mark_circle(opacity=0.4, color="#9467bd")
                    .encode(
                        x=alt.X(
                            income_col,
                            title="자치구 월평균 소득 (원)",
                            scale=alt.Scale(domain=[float(inc_price_df[income_col].min()),
                                                    float(inc_price_df[income_col].max())])
                        ),
                        y=alt.Y(
                            price_col,
                            title=price_col,
                            scale=alt.Scale(domain=[0, float(inc_price_df[price_col].max())])
                        )
                    )
                )
                st.altair_chart(inc_chart, width="stretch")
            else:
                st.info("표시할 데이터가 없습니다.")

        # 2-5. 상관계수 (피어슨, 이상치 1~99% 클리핑)
        st.markdown("### (5) 상관계수 (피어슨, 1~99% 분위수 클리핑 적용)")

        corr_cols = []
        for c in [price_col, area_col, bus_dist_col, sub_dist_col, hosp_dist_col, income_col]:
            if c is not None and c in filtered.columns:
                corr_cols.append(c)

        if len(corr_cols) >= 2:
            num_df = filtered[corr_cols].dropna().copy()
            if not num_df.empty:
                # 각 컬럼별로 1~99% 분위수로 클리핑해서 극단값 영향 줄이기
                for c in corr_cols:
                    q_low = num_df[c].quantile(0.01)
                    q_high = num_df[c].quantile(0.99)
                    num_df[c] = num_df[c].clip(lower=q_low, upper=q_high)

                corr_df = num_df.corr(method="pearson")
                st.caption(f"상관분석에 사용된 표본 수: {len(num_df)}")
                st.dataframe(corr_df, width="stretch")
            else:
                st.info("상관을 계산할 수 있는 데이터가 충분하지 않습니다.")
        else:
            st.info("상관을 계산할 수 있는 컬럼이 충분하지 않습니다.")

############################
# 탭 3: 지도 시각화 (가격에 따른 색상 + Heatmap)
############################
with tab3:
    st.subheader("지도에서 실거래 위치 보기 (필터 적용)")

    map_df = filtered.copy()
    if "Latitude" in map_df.columns and "Longitude" in map_df.columns:
        map_df = map_df.rename(columns={
            "Latitude": "lat",
            "Longitude": "lon"
        })

        max_points = 2000
        if len(map_df) > max_points:
            map_df_sample = map_df.sample(max_points, random_state=42)
            st.caption(f"표시 포인트 수: {max_points} (전체 {len(map_df)} 중 샘플링)")
        else:
            map_df_sample = map_df

        # 가격 컬럼이 있을 때: 가격 기반 색상 매핑 + Heatmap
        if price_col is not None and price_col in map_df_sample.columns:
            data = map_df_sample.dropna(subset=["lat", "lon", price_col]).copy()

            if not data.empty:
                # --------------------------
                # 1) 가격 구간별 색상 지정
                # --------------------------
                q1 = data[price_col].quantile(0.33)
                q2 = data[price_col].quantile(0.66)

                def price_to_color(v):
                    if v <= q1:
                        # 낮은 가격 → 파란색
                        return [0, 100, 255]
                    elif v <= q2:
                        # 중간 가격 → 주황색
                        return [255, 140, 0]
                    else:
                        # 높은 가격 → 빨간색
                        return [220, 0, 0]

                colors = data[price_col].apply(price_to_color)
                data["color_r"] = colors.apply(lambda x: x[0])
                data["color_g"] = colors.apply(lambda x: x[1])
                data["color_b"] = colors.apply(lambda x: x[2])

                # 공통 ViewState
                view_state = pdk.ViewState(
                    latitude=float(data["lat"].mean()),
                    longitude=float(data["lon"].mean()),
                    zoom=11,
                    pitch=0
                )

                # Tooltip 내용
                tooltip_fields = []
                if gu_col is not None and gu_col in data.columns:
                    tooltip_fields.append(f"{gu_col}: {{{gu_col}}}")
                tooltip_fields.append(f"{price_col}: {{{price_col}}}")
                tooltip_text = "\n".join(tooltip_fields)

                # --------------------------
                # (A) 가격 색상 Scatter 지도
                # --------------------------
                scatter_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=data,
                    get_position="[lon, lat]",
                    get_radius=100,      # 점 크게
                    radius_min_pixels=2,
                    radius_max_pixels=20,
                    get_fill_color="[color_r, color_g, color_b]",
                    pickable=True,
                    opacity=0.8
                )

                deck_scatter = pdk.Deck(
                    initial_view_state=view_state,
                    layers=[scatter_layer],
                    tooltip={"text": tooltip_text}
                    # map_style 생략 → Streamlit 기본 CARTO 지도를 사용
                )

                st.markdown("### (1) 가격 구간별 색상 지도")
                st.caption(
                    "색상 의미: 낮은 가격 = 파란색, 중간 가격 = 주황색, 높은 가격 = 빨간색\n"
                    f"(구분 기준: {price_col}의 33% / 66% 분위수)"
                )
                st.pydeck_chart(deck_scatter)



############################
# 탭 4: 임장 분석 (국토부 24 vs 25 아파트)
############################
with tab4:
    st.subheader("임장 분석 – 조건 적용 후 자치구별 아파트 가격 (2024 vs 2025)")

    st.caption(
        "· 좌측 필터(면적·가격·교통·병원·소득·자치구)를 모두 적용한 뒤,\n"
        "  그 조건을 만족하는 자치구만 골라서, 국토부 실거래 기준으로 2024년 vs 2025년 아파트 가격을 비교합니다."
    )

    if apt_stats is None:
        st.warning("국토부 아파트 실거래 엑셀을 읽는 데 실패했습니다. data 폴더의 24/25 엑셀 경로를 확인해 주세요.\n"
                   f"에러 메시지: {apt_err}")
    elif gu_col is None:
        st.warning("메인 데이터에서 자치구 컬럼을 찾을 수 없습니다.")
    else:
        # 필터 적용된 데이터 중 아파트만 사용 (임장 타깃)
        apt_df = filtered.copy()
        if "BuildingPurpose" in apt_df.columns:
            apt_df = apt_df[apt_df["BuildingPurpose"].astype(str).str.contains("아파트", na=False)]

        if apt_df.empty:
            st.info("현재 조건을 만족하는 아파트 데이터가 없습니다. 필터를 완화해 주세요.")
        else:
            # 필터된 데이터가 포함하는 자치구 목록
            gu_in_filtered = (
                apt_df[gu_col]
                .dropna()
                .astype(str)
                .unique()
            )

            # 국토부 실거래 중에서 해당 자치구만 추출
            stats_sel = apt_stats[apt_stats["자치구"].isin(gu_in_filtered)].copy()

            if stats_sel["연도"].nunique() < 2:
                st.info("선택된 자치구에 대해 2024와 2025 데이터가 모두 존재하지 않습니다.")
            else:
                # 자치구·연도별 평균가격/거래건수 피벗
                price_pivot = stats_sel.pivot(index="자치구", columns="연도", values="평균가격(만원)")
                count_pivot = stats_sel.pivot(index="자치구", columns="연도", values="거래건수")

                # 24, 25 모두 있는 자치구만
                if not (2024 in price_pivot.columns and 2025 in price_pivot.columns):
                    st.info("일부 연도의 데이터가 부족해 임장 분석이 제한됩니다.")
                else:
                    price_pivot = price_pivot[[2024, 2025]].dropna()
                    count_pivot = count_pivot[[2024, 2025]].reindex(price_pivot.index)

                    price_pivot["증감_만원"] = price_pivot[2025] - price_pivot[2024]
                    price_pivot["증감_퍼센트"] = (price_pivot["증감_만원"] / price_pivot[2024]) * 100

                    # 자치구 중심 좌표 (필터된 아파트들의 평균 좌표)
                    cent = (
                        apt_df
                        .groupby(gu_col)[["Latitude", "Longitude"]]
                        .mean()
                        .reindex(price_pivot.index)
                    )

                    analysis = price_pivot.join(count_pivot, lsuffix="_가격", rsuffix="_건수")
                    analysis = analysis.join(cent)

                    analysis = analysis.reset_index().rename(columns={
                        "자치구": "자치구",
                        2024: "2024_평균가격(만원)",
                        2025: "2025_평균가격(만원)",
                        "2024_가격": "2024_평균가격(만원)",
                        "2025_가격": "2025_평균가격(만원)",
                        "2024_건수": "2024_거래건수",
                        "2025_건수": "2025_거래건수"
                    })

                    analysis_sorted = analysis.sort_values("증감_퍼센트", ascending=False)

                    st.markdown("### 1) 자치구별 아파트 가격 변화 (국토부 2024 → 2025, 조건 적용 후 해당 자치구만)")
                    st.dataframe(
                        analysis_sorted[[
                            "자치구",
                            "2024_평균가격(만원)",
                            "2025_평균가격(만원)",
                            "증감_만원",
                            "증감_퍼센트",
                            "2024_거래건수",
                            "2025_거래건수"
                        ]].round(2),
                        width="stretch"
                    )

                    st.markdown("### 2) 아파트 가격 상승 TOP 10 자치구 (조건 적용 후 대상 자치구 중)")
                    top10 = analysis_sorted.head(10)

                    bar_chart = (
                        alt.Chart(top10)
                        .mark_bar()
                        .encode(
                            x=alt.X("증감_퍼센트:Q", title="증감률 (%)"),
                            y=alt.Y("자치구:N", sort="-x", title="자치구"),
                            color=alt.Color("증감_만원:Q", title="증감액(만원)"),
                            tooltip=[
                                "자치구",
                                "2024_평균가격(만원)",
                                "2025_평균가격(만원)",
                                "증감_만원",
                                "증감_퍼센트",
                                "2024_거래건수",
                                "2025_거래건수"
                            ]
                        )
                    )
                    st.altair_chart(bar_chart, width="stretch")

                    st.markdown("### 3) 자치구별 아파트 가격 변화 지도 (필터 적용 후, 자치구 중심 좌표 기준)")

                    map_df2 = analysis_sorted.dropna(subset=["Latitude", "Longitude"]).copy()
                    map_df2 = map_df2.rename(columns={"Latitude": "lat", "Longitude": "lon"})

                    if not map_df2.empty:
                        st.caption(
                            "· 파란 점 위치는 필터 조건을 만족하는 아파트들의 자치구별 평균 좌표입니다.\n"
                            "· 각 자치구의 24→25년 가격 변화는 위 표/막대그래프를 참고하세요."
                        )
                        st.map(map_df2[["lat", "lon"]])
                    else:
                        st.info("지도에 표시할 좌표 데이터가 없습니다.")
