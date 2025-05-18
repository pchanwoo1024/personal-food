import streamlit as st
import requests
from bs4 import BeautifulSoup
import random
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.title("🍽️ AI 기반 개인 맞춤 영양 식단 추천 (대전고 3-5월 급식)")

# 1) 사용자 입력
name    = st.text_input("이름")
sex     = st.selectbox("성별", ["M", "F"])
age     = st.slider("나이", 10, 80, 18)
height  = st.slider("키 (cm)", 140, 200, 170)
weight  = st.slider("몸무게 (kg)", 40, 120, 60)
activity= st.slider("활동량 (1.0~5.0)", 1.0, 5.0, 3.0, step=0.1)
allergy_options = [
    "난류","우유","메밀","땅콩","대두","밀","고등어",
    "게","새우","돼지고기","복숭아","토마토","아황산류",
    "호두","닭고기","쇠고기","오징어","조개류","잣"
]
allergies = st.multiselect("알레르기 (복수 선택)", allergy_options)
symptom_options = [
    "눈떨림","피로","두통","근육경련",
    "탈모","불면증","집중력저하","손발저림"
]
symptoms = st.multiselect("현재 증상", symptom_options)

# 2) Load and cache menu data
@st.cache_data
def load_menu():
    urls = [f"https://school.koreacharts.com/school/meals/B000013534/{m}.html" for m in ["202503","202504","202505"]]
    dishes = set()
    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            soup = BeautifulSoup(r.text, "html.parser")
            rows = soup.select("table tbody tr td:nth-child(2)")
            for cell in rows:
                items = [i.strip() for i in cell.get_text(separator=",").split(',') if i.strip()]
                dishes.update(items)
        except:
            continue
    return sorted(dishes)

menu_names = load_menu()
if not menu_names:
    st.error("급식 메뉴를 불러오지 못했습니다. 나중에 다시 시도해주세요.")
    st.stop()

# 3) Nutrition estimation
def estimate_nutrition(name: str) -> dict:
    name = name.strip()
    # Base kcal by category
    if any(x in name for x in ["밥","죽"]): kcal = 300
    elif any(x in name for x in ["국","찌개","탕"]): kcal = 80
    elif any(x in name for x in ["볶음","조림","구이","스테이크"]): kcal = 250
    elif any(x in name for x in ["전","만두","피자","파스타","면"]): kcal = 200
    else: kcal = 180
    fat  = kcal * 0.2
    prot = kcal * 0.15
    carb = kcal - fat - prot
    return {"name": name, "kcal": int(kcal), "carb": int(carb), "protein": int(prot), "fat": int(fat)}

menu_list = [estimate_nutrition(n) for n in menu_names]

if st.button("식단 추천 실행"):
    # 4) BMI, 목표, TDEE 계산
    bmi = weight / ((height/100)**2)
    target_bmi = 22.0
    target_weight = target_bmi * ((height/100)**2)
    weeks_needed = abs((target_weight - weight) * 7700 / 500) / 7
    if sex == 'M':
        bmr = 10*weight + 6.25*height - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height - 5*age - 161
    activity_factor = 1.2 + (activity - 1)*0.15
    tdee = bmr * activity_factor

    # 5) Allergy filter
    filtered = menu_list.copy()
    if allergies:
        filtered = [m for m in filtered if not any(a in m['name'] for a in allergies)]
    if not filtered:
        st.warning("알레르기 조건에 맞는 메뉴가 없습니다.")
        st.stop()

    # 6) Generate combos and training data
    combos = list(combinations(filtered, 3))
    X, y = [], []
    for combo in combos:
        kcal = sum(i['kcal'] for i in combo)
        carb = sum(i['carb'] for i in combo)
        prot = sum(i['protein'] for i in combo)
        fat  = sum(i['fat'] for i in combo)
        X.append([bmi, age, activity, kcal, carb, prot, fat])
        total = carb + prot + fat + 1e-6
        p_ratio = prot / total
        ideal_prot = 0.2 + (activity - 1)*0.05
        p_score = max(0.0, 1 - abs(p_ratio - ideal_prot))
        kcal_score = max(0.0, 1 - abs(kcal - tdee/3) / (tdee/3))
        score = 0.6 * kcal_score + 0.4 * p_score
        y.append(1 if score >= 0.5 else 0)

    # 7) Train model if y has both classes
    clf = None
    if len(set(y)) > 1:
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        clf.fit(X, y)

    # 8) Compute probabilities or fallback to score
    results = []
    for xi, combo in zip(X, combos):
        if clf:
            proba = clf.predict_proba([xi])[0]
            prob = proba[1] if len(proba) > 1 else 0.0
        else:
            # fallback: use weighted score
            kcal = xi[3]; prot = xi[5]
            total = xi[4] + prot + xi[6] + 1e-6
            p_ratio = prot / total
            ideal_prot = 0.2 + (activity - 1)*0.05
            p_score = max(0.0, 1 - abs(p_ratio - ideal_prot))
            kcal_score = max(0.0, 1 - abs(kcal - tdee/3) / (tdee/3))
            prob = 0.6 * kcal_score + 0.4 * p_score
        results.append((combo, prob))

    # 9) Top 3 combos
    ranked = sorted(results, key=lambda x: x[1], reverse=True)[:3]
    times = ['07:30 아침', '12:30 점심', '18:30 저녁']

    # 10) Symptom & age nutrients
    symptom_map = {
        '눈떨림': [('10:00','마그네슘 300mg')],
        '피로': [('09:00','비타민 B2 1.4mg')],
        '두통': [('11:00','비타민 B6 2mg')]
    }
    age_map = []
    if age < 20:
        age_map = [('08:00','칼슘 500mg'), ('20:00','비타민 D 10µg')]
    elif age < 50:
        age_map = [('09:00','비타민 D 10µg')]
    else:
        age_map = [('08:00','칼슘 500mg'), ('21:00','비타민 D 20µg')]

    # 11) Display
    st.subheader(f"{name}님 맞춤 결과")
    st.write(f"- 현재 BMI: {bmi:.2f}\n- 목표 BMI: {target_bmi}\n- 목표 체중: {target_weight:.1f}kg\n- 예상 소요: {weeks_needed:.1f}주\n- TDEE: {tdee:.0f} kcal")
    st.markdown("### 🍽️ 하루 식단 추천")
    for combo, prob in zip(ranked, times):
        pass
    for (combo, prob), t in zip(ranked, times):
        items = ' + '.join(i['name'] for i in combo)
        kcal_sum = sum(i['kcal'] for i in combo)
        st.write(f"{t} → **{items}** ({kcal_sum} kcal, 적합도 {prob:.2f})")
    st.markdown("### ⏰ 증상별 영양소 일정")
    for s in symptoms:
        for t, it in symptom_map.get(s, []):
            st.write(f"{t} → {it}")
    st.markdown("### ⏰ 연령별 권장 영양소")
    for t, it in age_map:
        st.write(f"{t} → {it}")
