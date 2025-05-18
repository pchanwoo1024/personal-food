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
allergy_options = ["난류","우유","메밀","땅콩","대두","밀","고등어","게","새우","돼지고기","복숭아","토마토","아황산류","호두","닭고기","쇠고기","오징어","조개류","잣"]
allergies       = st.multiselect("알레르기 (복수 선택)", allergy_options)
symptom_options = ["눈떨림","피로","두통","근육경련","탈모","불면증","집중력저하","손발저림"]
symptoms        = st.multiselect("현재 증상", symptom_options)

# 2) Load and cache menu data
@st.experimental_memo
def load_menu():
    months = ["202503","202504","202505"]
    base_url = "https://school.koreacharts.com/school/meals/B000013534/{}.html"
    dishes = set()
    for m in months:
        url = base_url.format(m)
        try:
            r = requests.get(url, timeout=5)
            soup = BeautifulSoup(r.text, "html.parser")
            # parse table rows
            for tr in soup.select("table tbody tr"):
                tds = tr.select("td")
                if len(tds) < 2:
                    continue
                items = [i.strip() for i in tds[1].get_text(separator=",").split(",") if i.strip()]
                for item in items:
                    dishes.add(item)
        except:
            continue
    return list(dishes)

menu_names = load_menu()
if not menu_names:
    st.error("급식 메뉴를 불러오지 못했습니다.")
    st.stop()

# Nutrition estimation
def estimate_nutrition(name):
    kcal = 200
    if "밥" in name or "죽" in name:
        kcal = 300
    elif any(x in name for x in ["국","찌개","탕"]):
        kcal = 80
    elif any(x in name for x in ["볶음","조림","구이","스테이크"]):
        kcal = 250
    elif any(x in name for x in ["전","만두","피자","파스타","면"]):
        kcal = 200
    fat = kcal * 0.2
    prot = kcal * 0.15
    carb = kcal - fat - prot
    return {"name": name, "kcal": int(kcal), "carb": int(carb), "protein": int(prot), "fat": int(fat)}

menu_list = [estimate_nutrition(n) for n in menu_names]

if st.button("식단 추천 실행"):
    # 3) BMI, 목표, TDEE 계산
    bmi = weight / ((height/100)**2)
    target_bmi = 22.0
    target_weight = target_bmi * ((height/100)**2)
    weeks = abs((target_weight - weight) * 7700 / 500) / 7
    if sex == "M":
        bmr = 10*weight + 6.25*height -5*age +5
    else:
        bmr = 10*weight + 6.25*height -5*age -161
    tdee = bmr * (1.2 + (activity-1)*0.15)

    # 4) Allergy filter
    filtered = [m for m in menu_list if not any(a in m["name"] for a in allergies)] if allergies else menu_list.copy()
    if not filtered:
        st.warning("알레르기 때문에 추천 가능 식단이 없습니다.")
        st.stop()

    # 5) Generate and train on 3-item combos
    combos = list(combinations(filtered, 3))
    X = []; y = []
    for combo in combos:
        kcal = sum(i["kcal"] for i in combo)
        carb = sum(i["carb"] for i in combo)
        prot = sum(i["protein"] for i in combo)
        fat  = sum(i["fat"] for i in combo)
        X.append([bmi, age, activity, kcal, carb, prot, fat])
        total_m = carb+prot+fat+1e-6
        p_ratio = prot/total_m
        ideal_p = 0.2 + (activity-1)*0.05
        p_score = 1 - abs(p_ratio-ideal_p)
        kcal_score = 1 - abs(kcal - tdee/3)/(tdee/3)
        score = 0.6*kcal_score + 0.4*p_score
        y.append(1 if score>0.75 else 0)

    clf = Pipeline([("s",StandardScaler()),("m",RandomForestClassifier(n_estimators=100,random_state=42))])
    clf.fit(X,y)
    probs = [clf.predict_proba([x])[0][1] for x in X]
    ranked = sorted(zip(combos,probs), key=lambda z:-z[1])[:3]
    times = ["07:30 아침","12:30 점심","18:30 저녁"]

    # 6) Symptom & age nutrients
    symptom_map = {"눈떨림":[("10:00","마그네슘 300mg")],"피로":[("09:00","비타민 B2 1.4mg")]}
    age_map = []
    if age<20: age_map=[("08:00","칼슘 500mg"),("20:00","비타민 D 10µg")]
    elif age<50: age_map=[("09:00","비타민 D 10µg")]
    else: age_map=[("08:00","칼슘 500mg"),("21:00","비타민 D 20µg")]

    # 7) Output
    st.subheader(f"{name}님 결과")
    st.write(f"- 현재 BMI: {bmi:.2f}, 목표 BMI: {target_bmi}")
    st.write(f"- 목표 체중: {target_weight:.1f}kg, 소요 기간: {weeks:.1f}주")
    st.write(f"- TDEE: {tdee:.0f} kcal")
    st.markdown("### 🍽️ 하루 식단")
    for (combo,p),t in zip(ranked,times):
        names = " + ".join(i["name"] for i in combo)
        kc = sum(i["kcal"] for i in combo)
        st.write(f"{t} → **{names}** ({kc} kcal, 적합도 {p:.2f})")
    st.markdown("### ⏰ 증상별 영양소")
    for s in symptoms:
        for tt,it in symptom_map.get(s,[]): st.write(f"{tt} → {it}")
    st.markdown("### ⏰ 연령별 영양소")
    for tt,it in age_map: st.write(f"{tt} → {it}")
