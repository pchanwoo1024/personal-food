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

allergy_options  = ["난류","우유","메밀","땅콩","대두","밀","고등어","게","새우","돼지고기","복숭아","토마토","아황산류","호두","닭고기","쇠고기","오징어","조개류","잣"]
allergies        = st.multiselect("알레르기 (복수 선택)", allergy_options)

symptom_options  = ["눈떨림","피로","두통","근육경련","탈모","불면증","집중력저하","손발저림"]
symptoms         = st.multiselect("현재 증상", symptom_options)

if st.button("식단 추천 실행"):
    # 2) BMI · 목표체중(BMI=22) · TDEE · 기간 계산
    bmi = weight / ((height/100)**2)
    target_bmi = 22.0
    target_weight = target_bmi * ((height/100)**2)
    weeks_needed = abs((target_weight - weight) * 7700 / 500) / 7
    if sex == "M": bmr = 10*weight + 6.25*height - 5*age + 5
    else:          bmr = 10*weight + 6.25*height - 5*age - 161
    activity_factor = 1.2 + (activity - 1)*0.15
    tdee = bmr * activity_factor

    # 3) 대전고 3-5월 급식 메뉴 파싱
    base_url = "https://school.koreacharts.com/school/meals/B000013534/{month}.html"
    dishes = set()
    for m in ["202503", "202504", "202505"]:
        try:
            r = requests.get(base_url.format(month=m), timeout=5)
            soup = BeautifulSoup(r.text, "html.parser")
            # 메뉴 텍스트 추출
            for td in soup.find_all(text=lambda t: '[' in t and ']' in t):
                # skip headers
                continue
            # 실제 메뉴는 <li> 태그 내부에 있음
            for li in soup.select('li'):
                text = li.get_text()
                # 괄호 이외 메뉴명만 추출
                if '(' in text:
                    name = text.split('(')[0].strip()
                else:
                    name = text.strip()
                # 길이 2 이상, 공백 아닌 것만
                if len(name) >= 2 and not name.isdigit():
                    dishes.add(name)
        except:
            pass
    if not dishes:
        st.error("급식 메뉴를 불러오지 못했습니다.")
        st.stop()
    # 4) 영양 정보 자동 생성 함수
    def estimate_nutrition(name):
        kcal = 200
        if any(x in name for x in ["밥","죽"]): kcal = 300
        elif any(x in name for x in ["국","찌개","탕"]): kcal = 100
        elif any(x in name for x in ["볶음","조림","구이","스테이크"]): kcal = 250
        elif any(x in name for x in ["전","만두","피자","파스타"]): kcal = 200
        fat = kcal * 0.2
        protein = kcal * 0.15
        carb = kcal - fat - protein
        return {"name":name, "kcal":int(kcal), "carb":int(carb), "protein":int(protein), "fat":int(fat)}

    menu_list = [estimate_nutrition(d) for d in dishes]
    # 5) 알레르기 필터링
    filtered = [m for m in menu_list if not any(a in m['name'] for a in allergies)] if allergies else menu_list
    if not filtered:
        st.warning("⚠️ 추천 가능한 음식이 없습니다.")
        st.stop()

    # 6) 3가지 조합 생성 및 학습
    combos = list(combinations(filtered, 3))
    X, y = [], []
    for combo in combos:
        kcal = sum(m['kcal'] for m in combo)
        carb = sum(m['carb'] for m in combo)
        prot = sum(m['protein'] for m in combo)
        fat  = sum(m['fat'] for m in combo)
        X.append([bmi, age, activity, kcal, carb, prot, fat])
        # 스코어
        total_macros = carb+prot+fat+1e-6
        pr = prot/total_macros
        ideal_pr = 0.2 + (activity-1)*0.05
        p_score = 1 - abs(pr - ideal_pr)
        kcal_score = 1 - abs(kcal - tdee/3)/(tdee/3)
        score = 0.6*kcal_score + 0.4*p_score
        y.append(1 if score > 0.75 else 0)

    clf = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
    clf.fit(X, y)

    # 7) 상위 3개 추천
    probs = [clf.predict_proba([x])[0][1] for x in X]
    ranked = sorted(zip(combos, probs), key=lambda x: -x[1])[:3]
    times = ["07:30 아침", "12:30 점심", "18:30 저녁"]

    # 8) 증상·연령별 영양소
    symptom_map = {"눈떨림":[("10:00","마그네슘 300mg")], "피로":[("09:00","비타민 B2 1.4mg")]}  # 생략
    age_map = []
    if age < 20: age_map = [("08:00","칼슘 500mg"),("20:00","비타민 D 10µg")]
    elif age < 50: age_map = [("09:00","비타민 D 10µg")]
    else: age_map = [("08:00","칼슘 500mg"),("21:00","비타민 D 20µg")]

    # 9) 출력
    st.subheader(f"{name}님 맞춤 결과")
    st.write(f"- 현재 BMI: {bmi:.2f}")
    st.write(f"- 목표 BMI: {target_bmi}, 목표 체중: {target_weight:.1f}kg")
    st.write(f"- TDEE: {tdee:.0f} kcal, 예상 소요: {weeks_needed:.1f}주")
    st.markdown("### 🍽️ 하루 식단 계획")
    for (combo, prob), t in zip(ranked, times):
        items = " + ".join(m['name'] for m in combo)
        kcal_sum = sum(m['kcal'] for m in combo)
        st.write(f"{t} → **{items}** ({kcal_sum} kcal, 적합도 {prob:.2f})")
    st.markdown("### ⏰ 증상별 영양소 일정")
    for s in symptoms:
        for tt,it in symptom_map.get(s,[]): st.write(f"{tt} → {it}")
    st.markdown("### ⏰ 연령별 권장 영양소")
    for tt,it in age_map: st.write(f"{tt} → {it}")
