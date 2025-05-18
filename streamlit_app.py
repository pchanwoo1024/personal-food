import streamlit as st
import random
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.title("🍽️ AI 기반 개인 맞춤 영양 식단 추천 (Generic Dishes)")

# 1) 사용자 입력
name    = st.text_input("이름")
sex     = st.selectbox("성별", ["M", "F"])
age     = st.slider("나이", 10, 80, 18)
height  = st.slider("키 (cm)", 140, 200, 170)
weight  = st.slider("몸무게 (kg)", 40, 120, 60)
activity= st.slider("활동량 (1.0~5.0)", 1.0, 5.0, 3.0, step=0.1)
allergy_options  = ["밀","대두","우유","달걀","닭고기","돼지고기","생선","땅콩","메밀"]
allergies        = st.multiselect("알레르기 (복수 선택)", allergy_options)
symptom_options  = ["눈떨림","피로","두통","근육경련","탈모","불면증","집중력저하","손발저림"]
symptoms         = st.multiselect("현재 증상", symptom_options)

if st.button("식단 추천 실행"):
    # 2) BMI·BMR·TDEE 계산
    bmi           = weight / ((height/100)**2)
    target_weight = 21.0 * ((height/100)**2)
    weeks_needed  = abs((target_weight - weight) * 7700 / 500) / 7
    if sex == "M":
        bmr = 10*weight + 6.25*height -5*age +5
    else:
        bmr = 10*weight + 6.25*height -5*age -161
    activity_factor = 1.2 + (activity-1)*0.15
    tdee = bmr * activity_factor

    # 3) Generic dish list with translations and nutrition
    nutrition_map = {
        "Kimchi Jjigae":    {"name":"김치찌개","kcal":150,"carb":5,"protein":6,"fat":10},
        "Doenjang Jjigae":  {"name":"된장찌개","kcal":180,"carb":8,"protein":7,"fat":12},
        "Miyeok Guk":       {"name":"미역국","kcal":50,"carb":3,"protein":2,"fat":1},
        "Bulgogi":          {"name":"불고기","kcal":300,"carb":10,"protein":25,"fat":15},
        "Japchae":          {"name":"잡채","kcal":280,"carb":40,"protein":7,"fat":12},
        "Jeyuk Bokkeum":    {"name":"제육볶음","kcal":350,"carb":15,"protein":20,"fat":25},
        "Curry Rice":       {"name":"카레라이스","kcal":450,"carb":60,"protein":10,"fat":15},
        "Dubu Jorim":       {"name":"두부조림","kcal":200,"carb":8,"protein":12,"fat":12},
        "Gyeran Jjim":      {"name":"계란찜","kcal":120,"carb":2,"protein":10,"fat":8},
        "Jjambbong":        {"name":"짬뽕","kcal":550,"carb":70,"protein":15,"fat":20},
        "Ramen":            {"name":"라면","kcal":500,"carb":60,"protein":10,"fat":20},
        "Bibimbap":         {"name":"비빔밥","kcal":600,"carb":80,"protein":12,"fat":18},
        "Tteokbokki":       {"name":"떡볶이","kcal":400,"carb":65,"protein":6,"fat":10},
        "Pizza":            {"name":"피자","kcal":700,"carb":80,"protein":25,"fat":30},
        "Pasta":            {"name":"파스타","kcal":650,"carb":75,"protein":20,"fat":25},
        "Burger":           {"name":"햄버거","kcal":550,"carb":45,"protein":30,"fat":25},
        "Sandwich":         {"name":"샌드위치","kcal":350,"carb":40,"protein":15,"fat":15}
    }
    dishes = list(nutrition_map.values())
    # Random sample of 7 dishes
    meals = random.sample(dishes, min(7, len(dishes)))
    for m in meals:
        m["tags"] = [m["name"]]

    # 4) 알레르기 필터링
    filtered = [m for m in meals if not any(a in m["name"] for a in allergies)] if allergies else meals.copy()
    if not filtered:
        st.warning("입력하신 알레르기 때문에 추천 가능한 음식이 없습니다.")
        st.stop()

    # 5) Generate combos of 3 items
    combos = list(combinations(filtered, 3))
    X = []; y = []
    for combo in combos:
        kcal = sum(m["kcal"] for m in combo)
        carb = sum(m["carb"] for m in combo)
        prot = sum(m["protein"] for m in combo)
        fat  = sum(m["fat"] for m in combo)
        X.append([bmi, age, activity, kcal, carb, prot, fat])
        # scoring
        total_macros = carb + prot + fat + 1e-6
        prot_ratio = prot / total_macros
        ideal_prot = 0.20 + (activity-1)*0.05
        p_score = 1 - abs(prot_ratio - ideal_prot)
        kcal_score = 1 - abs(kcal - tdee/3) / (tdee/3)
        total_score = 0.6 * kcal_score + 0.4 * p_score
        y.append(1 if total_score > 0.75 else 0)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    clf.fit(X, y)

    # 6) Rank combos
    probs = [clf.predict_proba([feat])[0][1] for feat in X]
    ranked = sorted(zip(combos, probs), key=lambda x: -x[1])[:3]
    times = ["07:30 아침", "12:30 점심", "18:30 저녁"]

    # 7) Symptom & age-based nutrients
    symptom_map = {
        "눈떨림":[("10:00","마그네슘 300mg")], "피로":[("09:00","비타민 B2 1.4mg")],
        # Extend as needed...
    }
    age_map = []
    if age < 20:
        age_map = [("08:00","칼슘 500mg"),("20:00","비타민 D 10µg")]
    elif age < 50:
        age_map = [("09:00","비타민 D 10µg")]
    else:
        age_map = [("08:00","칼슘 500mg"),("21:00","비타민 D 20µg")]

    # 8) Output
    st.subheader(f"{name}님 맞춤 식단")
    for (combo, prob), t in zip(ranked, times):
        items = " + ".join(m["name"] for m in combo)
        kcal_sum = sum(m["kcal"] for m in combo)
        st.write(f"{t} → {items} ({kcal_sum} kcal, 적합도 {prob:.2f})")

    st.markdown("### ⏰ 증상별 영양소 일정")
    for s in symptoms:
        for t, item in symptom_map.get(s, []):
            st.write(f"{t} → {item}")

    st.markdown("### ⏰ 연령별 권장 영양소")
    for t, item in age_map:
        st.write(f"{t} → {item}")
