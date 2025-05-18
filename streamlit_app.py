import streamlit as st
import random
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.title("🍽️ AI 기반 개인 맞춤 영양 식단 추천")

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
    # 2) BMI·BMR·TDEE·목표체중·기간 계산
    bmi           = weight / ((height/100)**2)
    target_weight = 21.0 * ((height/100)**2)
    weeks_needed  = abs((target_weight - weight) * 7700 / 500) / 7

    if sex == "M":
        bmr = 10*weight + 6.25*height -5*age +5
    else:
        bmr = 10*weight + 6.25*height -5*age -161
    activity_factor = 1.2 + (activity-1)*0.15
    tdee = bmr * activity_factor

    # 3) TheMealDB API로 한국 음식 리스트 가져오기
    try:
        r = requests.get("https://www.themealdb.com/api/json/v1/1/filter.php?a=Korean", timeout=5)
        data = r.json().get("meals", [])
    except:
        data = []
    # 영문 이름 → 한국어+영양정보 매핑
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
        "Salad":            {"name":"샐러드","kcal":150,"carb":10,"protein":5,"fat":10}
    }
    available = [nutrition_map[m["strMeal"]] for m in data if m["strMeal"] in nutrition_map]
    if not available:
        st.error("인터넷에서 한국 음식 정보를 가져오지 못했습니다.")
        st.stop()

    meals = random.sample(available, min(5, len(available)))

    # 4) 알레르기 필터링
    filtered = [m for m in meals if not any(a in m["name"] for a in allergies)] if allergies else meals.copy()
    if not filtered:
        st.warning("⚠️ 입력하신 알레르기 때문에 추천 가능한 음식이 없습니다.")
        st.stop()

    # 5) 라벨링 & 모델 학습
    X_train = [[bmi, age, activity, m["kcal"], m["carb"], m["protein"], m["fat"]] for m in filtered]
    y_train = []
    for m in filtered:
        total_macros = m["carb"] + m["protein"] + m["fat"] + 1e-6
        protein_ratio = m["protein"] / total_macros
        ideal_protein = 0.20 + (activity - 1)*0.05
        protein_score = 1 - abs(protein_ratio - ideal_protein)
        kcal_score    = 1 - abs(m["kcal"] - tdee) / tdee
        total_score   = 0.6 * kcal_score + 0.4 * protein_score
        y_train.append(1 if total_score > 0.75 else 0)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    clf.fit(X_train, y_train)
    classes = clf.named_steps["model"].classes_

    def get_prob(feats):
        proba = clf.predict_proba([feats])[0]
        return proba[list(classes).index(1)] if 1 in classes else 0.0

    # 6) 세 끼 식단 추천
    scores = [(m, get_prob([bmi, age, activity, m["kcal"], m["carb"], m["protein"], m["fat"]])) for m in filtered]
    scores.sort(key=lambda x: -x[1])
    plan = scores[:3]
    times = ["07:30 아침", "12:30 점심", "18:30 저녁"]

    # 7) 결과 출력
    st.markdown("### 🍽️ 하루 식단 계획")
    for (meal, prob), time in zip(plan, times):
        st.write(f"{time} → **{meal['name']}** ({meal['kcal']} kcal, 적합도 {prob:.2f})")

