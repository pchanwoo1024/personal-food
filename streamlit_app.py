import streamlit as st
import requests
from bs4 import BeautifulSoup
import random
import numpy as np
import re
from datetime import datetime, timedelta, time as dtime
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="개인 맞춤 영양 식단 추천 (대전고 급식)")
st.title("🍽️ AI 기반 개인 맞춤 영양 식단 추천 (대전고 3~5월 급식)")

# 1) 사용자 입력
name      = st.text_input("이름")
sex       = st.selectbox("성별", ["M", "F"])
age       = st.slider("나이", 10, 80, 18)
height    = st.slider("키 (cm)", 140, 200, 170)
weight    = st.slider("몸무게 (kg)", 40, 120, 60)
activity  = st.slider("활동량 (1.0~5.0)", 1.0, 5.0, 3.0, 0.1)
wake_time = st.time_input("기상 시간", value=dtime(7,0))
sleep_time = st.time_input("취침 시간", value=dtime(22,0))

allergy_options = [
    "난류","우유","메밀","땅콩","대두","밀",
    "고등어","게","새우","돼지고기","복숭아",
    "토마토","아황산류","호두","닭고기","쇠고기",
    "오징어","조개류","잣"
]
allergies = st.multiselect("알레르기 (복수 선택)", allergy_options)

symptom_options = [
    "눈떨림","피로","두통","근육경련",
    "탈모","불면증","집중력저하","손발저림"
]
symptoms = st.multiselect("현재 증상", symptom_options)

# 2) 급식 메뉴 파싱 (대전고 공식 게시판)
@st.cache_data
"
"def load_menu():
"
"    list_url = "https://djhs.djsch.kr/boardCnts/list.do?boardID=41832&m=020701&s=daejeon"
"
"    dishes = set()
"
"    session = requests.Session()
"
"    session.headers.update({'User-Agent':'Mozilla/5.0'})
"
"    try:
"
"        r = session.get(list_url, timeout=5)
"
"        r.raise_for_status()
"
"        soup = BeautifulSoup(r.text, "html.parser")
"
"        # 게시판 목록
"
"        rows = soup.select("table.boardList tbody tr, table.tableList tbody tr")
"
"        for tr in rows:
"
"            tds = tr.select("td")
"
"            if len(tds) < 2:
"
"                continue
"
"            link = tds[1].find('a')
"
"            if not link or '중식' not in link.text:
"
"                continue
"
"            href = link.get('href')
"
"            detail_url = href if href.startswith('http') else f"https://djhs.djsch.kr{href}"
"
"            try:
"
"                dr = session.get(detail_url, timeout=5)
"
"                dr.raise_for_status()
"
"                dsoup = BeautifulSoup(dr.text, "html.parser")
"
"                content = dsoup.select_one("div.board_conts, div.boardContents, td.board_txt")
"
"                text = content.get_text(separator=",") if content else ''
"
"                for part in re.split('[,·]', text):
"
"                    item = re.sub(r"\([^)]*\)", "", part).strip()
"
"                    if re.fullmatch(r"[가-힣 ]{2,10}", item):
"
"                        dishes.add(item)
"
"            except:
"
"                continue
"
"    except Exception as e:
"
"        st.error(f"급식 목록을 불러오는 중 오류 발생: {e}")
"
"        return []
"
"    return sorted(dishes)

"
"menu_names = load_menu()()
if not menu_names:
    st.error("급식 메뉴를 불러오지 못했습니다. 서버 과부하일 수 있습니다.")
    st.stop()
# Show parsed menu for debug
st.write("🔎 파싱된 급식 메뉴 (총 {}개):".format(len(menu_names)), menu_names)

# 3) 영양 정보 추정 함수
def estimate_nutrition(name: str) -> dict:
    if any(x in name for x in ["밥","죽"]): kcal = 300
    elif any(x in name for x in ["국","찌개","탕"]): kcal = 80
    elif any(x in name for x in ["볶음","조림","구이","스테이크"]): kcal = 250
    elif any(x in name for x in ["전","만두","피자","파스타","면","떡볶이"]): kcal = 200
    else: kcal = 180
    fat  = kcal * 0.2
    prot = kcal * 0.15
    carb = kcal - fat - prot
    return {"name": name, "kcal": int(kcal), "carb": int(carb), "protein": int(prot), "fat": int(fat)}

menu_list = [estimate_nutrition(n) for n in menu_names]

# 4) 추천 실행
if st.button("식단 추천 실행"):
    # BMI, 목표 BMI=22, 목표 체중, TDEE, 소요 기간
    bmi = weight / ((height/100)**2)
    target_bmi = 22.0
    target_weight = target_bmi * ((height/100)**2)
    weeks_needed = abs((target_weight - weight) * 7700 / 500) / 7
    bmr = 10*weight + 6.25*height - 5*age + (5 if sex=='M' else -161)
    tdee = bmr * (1.2 + (activity-1)*0.15)

    # 알레르기 필터링
    filtered = menu_list.copy()
    if allergies:
        filtered = [m for m in filtered if not any(a in m['name'] for a in allergies)]
    if not filtered:
        st.warning("알레르기 조건에 맞는 메뉴가 없습니다.")
        st.stop()

    # 5) 세 가지 조합 생성
    combos = list(combinations(filtered, 3))
    X, y = [], []
    for combo in combos:
        k = sum(i['kcal'] for i in combo)
        c = sum(i['carb'] for i in combo)
        p = sum(i['protein'] for i in combo)
        f = sum(i['fat'] for i in combo)
        X.append([bmi, age, activity, k, c, p, f])
        total = c + p + f + 1e-6
        pr = p / total
        ideal_p = 0.2 + (activity-1)*0.05
        p_score = max(0, 1 - abs(pr - ideal_p))
        k_score = max(0, 1 - abs(k - tdee/3) / (tdee/3))
        y.append(1 if 0.6*k_score + 0.4*p_score >= 0.5 else 0)

    # 6) 모델 학습
    clf = None
    if len(set(y)) > 1:
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        clf.fit(X, y)

    # 7) 평가 및 추천
    recs = []
    for xi, combo in zip(X, combos):
        if clf:
            probs = clf.predict_proba([xi])[0]
            score = probs[1] if len(probs) > 1 else 0.0
        else:
            # fallback score
            k=xi[3]; p=xi[5]; total=xi[4]+p+xi[6]+1e-6; pr=p/total; ideal_p=0.2+(activity-1)*0.05
            p_score=max(0,1-abs(pr-ideal_p)); k_score=max(0,1-abs(k-tdee/3)/(tdee/3)); score=0.6*k_score+0.4*p_score
        recs.append((combo, score))

    # 8) 중복 없는 상위 N개 선택
    # Determine meal count based on wake/sleep
    dt_today = datetime.today()
    wake_dt = datetime.combine(dt_today, wake_time)
    sleep_dt = datetime.combine(dt_today, sleep_time)
    if sleep_dt <= wake_dt:
        sleep_dt += timedelta(days=1)
    awake_duration = (sleep_dt - wake_dt).total_seconds() / 3600
    if awake_duration < 12:
        meal_count = 2
    else:
        meal_count = 3

    selected = []
    used_names = set()
    for combo, score in sorted(recs, key=lambda x: x[1], reverse=True):
        names = [i['name'] for i in combo]
        if any(n in used_names for n in names):
            continue
        selected.append((combo, score))
        used_names.update(names)
        if len(selected) >= meal_count:
            break

    # 9) 식사 시간 설정
    if meal_count == 3:
        b_time = wake_dt + timedelta(hours=1)
        # lunch no later than 16:00
        mid = wake_dt + (sleep_dt - wake_dt)/2
        l_time = min(mid, datetime.combine(dt_today, dtime(16,0)))
        d_time = min(sleep_dt - timedelta(hours=1), datetime.combine(dt_today, dtime(22,0)))
        slots = [b_time, l_time, d_time]
    else:
        b_time = wake_dt + timedelta(hours=1)
        d_time = min(sleep_dt - timedelta(hours=1), datetime.combine(dt_today, dtime(22,0)))
        slots = [b_time, d_time]

    # 10) 출력
    st.subheader(f"{name}님 맞춤 결과")
    st.write(f"- 현재 BMI: {bmi:.2f} | 목표 BMI: {target_bmi} | 목표 체중: {target_weight:.1f}kg")
    st.write(f"- TDEE: {tdee:.0f} kcal | 예상 소요: {weeks_needed:.1f}주")
    st.markdown("### 🍽️ 하루 식단 추천")
    for (combo, score), t in zip(selected, slots):
        items = " + ".join(i['name'] for i in combo)
        kc = sum(i['kcal'] for i in combo)
        st.write(f"{t.strftime('%H:%M')} → **{items}** ({kc} kcal, 적합도 {score:.2f})")
    st.markdown("### ⏰ 증상별 영양소 일정")
    smap = {"눈떨림": [("10:00","마그네슘 300mg")], "피로": [("09:00","비타민 B2 1.4mg")]}    
    for s in symptoms:
        for tt, it in smap.get(s, []):
            st.write(f"{tt} → {it}")
    st.markdown("### ⏰ 연령별 권장 영양소")
    if age < 20:
        amap = [("08:00","칼슘 500mg"),("20:00","비타민 D 10µg")]
    elif age < 50:
        amap = [("09:00","비타민 D 10µg")]
    else:
        amap = [("08:00","칼슘 500mg"),("21:00","비타민 D 20µg")]
    for tt, it in amap:
        st.write(f"{tt} → {it}")
