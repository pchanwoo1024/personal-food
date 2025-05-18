import streamlit as st
import random
import requests
import numpy as np
from itertools import combinations
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

    # 3) Fetch Korean meals
    try:
        resp = requests.get("https://www.themealdb.com/api/json/v1/1/filter.php?a=Korean", timeout=5)
        meals_data = resp.json().get("meals") or []
    except:
        meals_data = []
    cat_nut = {
        "Jjigae": {"suffix":"찌개","kcal":180,"carb":8,"protein":7,"fat":12},
        "Guk":    {"suffix":"국","kcal":80,"carb":5,"protein":3,"fat":2},
        "Bulgogi":{"name":"불고기","kcal":300,"carb":10,"protein":25,"fat":15},
        "Japchae":{"name":"잡채","kcal":280,"carb":40,"protein":7,"fat":12},
        "Bokkeum":{"suffix":"볶음","kcal":350,"carb":15,"protein":20,"fat":25},
        "Curry":  {"suffix":"카레","kcal":450,"carb":60,"protein":10,"fat":15},
        "Dubu":   {"suffix":"두부","kcal":200,"carb":8,"protein":12,"fat":12},
        "Gyeran": {"suffix":"계란찜","kcal":120,"carb":2,"protein":10,"fat":8},
        "Salad":  {"name":"샐러드","kcal":150,"carb":10,"protein":5,"fat":10}
    }
    available = []
    for m in meals_data:
        meal_en = m.get("strMeal","")
        for key,nut in cat_nut.items():
            if key in meal_en:
                if "name" in nut:
                    name_kr = nut["name"]
                else:
                    prefix = meal_en.replace(key,"").strip()
                    name_kr = prefix + nut["suffix"]
                available.append({"name":name_kr,
                                  "kcal":nut["kcal"],
                                  "carb":nut["carb"],
                                  "protein":nut["protein"],
                                  "fat":nut["fat"]})
                break
    if not available:
        st.error("한국 음식 정보를 가져오지 못했습니다.")
        st.stop()

    # 4) allergy filter
    meals = random.sample(available, min(7,len(available)))
    filtered = [m for m in meals if not any(a in m["name"] for a in allergies)] if allergies else meals
    if not filtered:
        st.warning("알레르기로 추천 불가")
        st.stop()

    # 5) combos of 3
    combos = list(combinations(filtered,3))
    X = []; y=[]
    for combo in combos:
        kcal=sum(m["kcal"] for m in combo)
        carb=sum(m["carb"] for m in combo)
        prot=sum(m["protein"] for m in combo)
        fat=sum(m["fat"] for m in combo)
        X.append([bmi,age,activity,kcal,carb,prot,fat])
        total_macros=carb+prot+fat+1e-6
        prot_ratio=prot/total_macros
        ideal_prot=0.2+(activity-1)*0.05
        p_score=1-abs(prot_ratio-ideal_prot)
        kcal_score=1-abs(kcal-tdee*0.33)/(tdee*0.33)
        score=0.6*kcal_score+0.4*p_score
        y.append(1 if score>0.75 else 0)
    clf=Pipeline([("scaler",StandardScaler()),
                  ("model",RandomForestClassifier(n_estimators=100,random_state=42))])
    clf.fit(X,y)
    probs=[clf.predict_proba([feat])[0][1] for feat in X]
    ranked=sorted(zip(combos,probs),key=lambda x:-x[1])[:3]
    times=["07:30","12:30","18:30"]

    # nutrient schedules
    symptom_map={"눈떨림":[("10:00","마그네슘 300mg")],"피로":[("09:00","비타민 B2 1.4mg")]}
    age_map=[]
    if age<20: age_map=[("08:00","칼슘 500mg"),("20:00","비타민 D 10µg")]
    elif age<50: age_map=[("09:00","비타민 D 10µg")]
    else: age_map=[("08:00","칼슘 500mg"),("21:00","비타민 D 20µg")]

    # output
    st.subheader(f"{name}님 식단")
    for (combo,prob),t in zip(ranked,times):
        items=" + ".join(m["name"] for m in combo)
        kcal_sum=sum(m["kcal"] for m in combo)
        st.write(f"{t} → {items} ({kcal_sum} kcal, 적합도 {prob:.2f})")
    st.markdown("### 영양소 섭취 일정")
    for s in symptoms:
        for t,i in symptom_map.get(s,[]): st.write(f"{t} → {i}")
    st.markdown("### 연령별 권장")
    for t,i in age_map: st.write(f"{t} → {i}")
