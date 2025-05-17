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
age     = st.slider("나이", 16, 18, 17)
height  = st.slider("키 (cm)", 140, 200, 170)
weight  = st.slider("몸무게 (kg)", 40, 120, 60)
activity= st.slider("활동량 (1.0~5.0)", 1.0, 5.0, 3.0, step=0.1)
allergy_options  = ["밀", "대두", "우유", "닭고기", "돼지고기", "생선", "무", "배추", "카레"]
allergies       = st.multiselect("알레르기 (복수 선택)", allergy_options)
symptom_options = ["눈떨림", "피로", "두통", "근육경련"]
symptoms        = st.multiselect("현재 증상", symptom_options)

if st.button("식단 추천 실행"):
    # 2) BMI·BMR·TDEE·목표체중·기간 계산
    bmi = weight / ((height/100)**2)
    target_weight = 21.0 * ((height/100)**2)
    weeks_needed = abs((target_weight - weight) * 7700 / 500) / 7

    if sex == "M":
        bmr = 10*weight + 6.25*height - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height - 5*age - 161
    activity_factor = 1.2 + (activity - 1)*0.15
    tdee = bmr * activity_factor

    # 3) 외부 API에서 랜덤 식품 가져오기
    def fetch_random_foods(n=5):
        meals = []
        for _ in range(n):
            page = random.randint(1, 50)
            url = (
                "https://world.openfoodfacts.org/cgi/search.pl"
                f"?action=process&json=true&page={page}&page_size=20"
                "&fields=product_name,nutriments,ingredients_text"
            )
            try:
                r = requests.get(url, timeout=5)
                data = r.json()
            except Exception:
                continue
            prods = data.get("products", [])
            valid = [p for p in prods if p.get("product_name") and p.get("nutriments")]
            if not valid:
                continue
            p = random.choice(valid)
            nut = p["nutriments"]
            meals.append({
                "name": p["product_name"],
                "kcal": nut.get("energy-kcal_100g", 0),
                "carb": nut.get("carbohydrates_100g", 0),
                "protein": nut.get("proteins_100g", 0),
                "fat": nut.get("fat_100g", 0),
                "tags": p.get("ingredients_text", "").split(",")
            })
        return meals

    meals = fetch_random_foods(5)
    if not meals:
        meals = [
          {"name":"통곡물빵+달걀","kcal":350,"carb":40,"protein":20,"fat":10,"tags":["밀","달걀"]},
          {"name":"닭가슴살샐러드","kcal":300,"carb":10,"protein":35,"fat":8,"tags":["닭고기","채소"]},
          {"name":"연어스테이크","kcal":450,"carb":0,"protein":30,"fat":35,"tags":["생선"]},
          {"name":"두부스테이크","kcal":280,"carb":8,"protein":25,"fat":15,"tags":["대두"]},
          {"name":"과일스무디","kcal":200,"carb":45,"protein":2,"fat":1,"tags":["과일"]}
        ]

    # 4) 알레르기 필터
    filtered = [
        m for m in meals
        if not any(a in tag for a in allergies for tag in m["tags"])
    ] if allergies else meals.copy()
    if not filtered:
        st.warning("⚠️ 알레르기 때문에 추천 가능한 음식이 없습니다.")
        st.stop()

    # 5) 라벨링 & 모델 학습
    diffs = [abs(m["kcal"] - tdee) for m in filtered]
    top2 = np.argsort(diffs)[:2]
    X_train = [[m["kcal"],m["carb"],m["protein"],m["fat"]] for m in filtered]
    y_train = [1 if i in top2 else 0 for i in range(len(filtered))]
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
    scores = [(m, get_prob([m["kcal"],m["carb"],m["protein"],m["fat"]])) for m in filtered]
    scores.sort(key=lambda x: -x[1])
    plan = scores[:3]
    times = ["07:30 아침", "12:30 점심", "18:30 저녁"]

    # 7) 증상별 영양소 스케줄
    symptom_vitamin_map = {
        "눈떨림":   [("10:00","마그네슘 300mg"),("14:30","비타민 B6 2mg")],
        "피로":     [("09:00","비타민 B2 1.4mg"),("13:00","비타민 C 100mg")],
        "두통":     [("11:00","마그네슘 300mg"),("15:00","비타민 B2 1.4mg")],
        "근육경련": [("08:00","칼슘 500mg"),("20:00","마그네슘 300mg")]
    }
    nutri_schedule = []
    for s in symptoms:
        nutri_schedule.extend(symptom_vitamin_map.get(s, []))

    # 8) 결과 출력
    st.subheader(f"{name}님 맞춤 결과")
    st.write(f"- BMI: {bmi:.2f}")
    st.write(f"- 목표 체중: {target_weight:.1f} kg")
    st.write(f"- TDEE: {tdee:.0f} kcal")
    st.write(f"- 예상 소요 기간: 약 {weeks_needed:.1f}주")

    st.markdown("### 🍽️ 하루 식단 계획")
    for (meal, prob), time in zip(plan, times):
        st.write(f"{time} → **{meal['name']}** ({meal['kcal']} kcal, 적합도 {prob:.2f})")

    if nutri_schedule:
        st.markdown("### ⏰ 영양소 섭취 일정")
        for time, item in nutri_schedule:
            st.write(f"- {time} → {item}")

    st.markdown("### 🔍 기타 음식 적합도")
    for m, p in scores[3:]:
        label = "✅ 추천" if p >= 0.5 else "🔸 참고"
        st.write(f"{label} {m['name']} ({m['kcal']} kcal, 적합도 {p:.2f})")
