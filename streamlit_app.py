import streamlit as st
import requests, itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.title("🍽️ AI 기반 개인 맞춤 영양 식단 추천 (다양한 패턴)")

# 1) 사용자 입력
name      = st.text_input("이름")
sex       = st.selectbox("성별", ["M", "F"])
age       = st.slider("나이", 10, 80, 18)
height    = st.slider("키 (cm)", 140, 200, 170)
weight    = st.slider("몸무게 (kg)", 40, 120, 60)
activity  = st.slider("활동량 (1.0~5.0)", 1.0, 5.0, 3.0, step=0.1)
allergies = st.multiselect("알레르기", ["밀","대두","우유","달걀","닭고기","돼지고기","생선","땅콩","메밀"])
symptoms  = st.multiselect("현재 증상", ["피로","두통","불면증","집중력저하"])

if st.button("식단 추천 실행"):
    # 2) 기초대사량·TDEE 계산
    bmi = weight/((height/100)**2)
    if sex=="M":
        bmr = 10*weight + 6.25*height -5*age +5
    else:
        bmr = 10*weight + 6.25*height -5*age -161
    tdee = bmr * (1.2 + (activity-1)*0.15)
    per_meal = tdee/3

    # 3) 동적 매크로 비율
    carb_r, prot_r, fat_r = 0.5, 0.25, 0.25
    if activity>=4: prot_r+=0.05; carb_r-=0.025; fat_r-=0.025
    if "피로" in symptoms: carb_r+=0.05; fat_r-=0.05
    if "불면증" in symptoms: fat_r+=0.05; carb_r-=0.05
    s = carb_r+prot_r+fat_r
    carb_r, prot_r, fat_r = carb_r/s, prot_r/s, fat_r/s

    # 4) 음식 카테고리 정의
    Grains   = [{"name":"백미밥","kcal":300,"carb":65,"protein":4,"fat":1},
                {"name":"현미밥","kcal":250,"carb":55,"protein":5,"fat":2}]
    Stews    = [{"name":"김치찌개","kcal":150,"carb":5,"protein":6,"fat":10},
                {"name":"된장찌개","kcal":180,"carb":8,"protein":7,"fat":12}]
    Proteins = [{"name":"닭가슴살","kcal":200,"carb":0,"protein":40,"fat":2},
                {"name":"연어구이","kcal":250,"carb":0,"protein":30,"fat":12}]
    Sides    = [{"name":"두부조림","kcal":200,"carb":8,"protein":12,"fat":12},
                {"name":"계란찜","kcal":120,"carb":2,"protein":10,"fat":8},
                {"name":"샐러드","kcal":150,"carb":10,"protein":5,"fat":10}]
    Fruits   = [{"name":"사과","kcal":80,"carb":20,"protein":0,"fat":0},
                {"name":"바나나","kcal":100,"carb":25,"protein":1,"fat":0}]

    # 5) 알레르기 필터링
    def filt(lst):
        return [m for m in lst if not any(a in m["name"] for a in allergies)]
    Grains, Stews, Proteins, Sides, Fruits = map(filt, (Grains,Stews,Proteins,Sides,Fruits))
    if not (Grains and Stews and Proteins and Sides):
        st.error("알레르기 필터 후 선택 가능한 음식이 부족합니다.")
        st.stop()

    # 6) 패턴별 조합 생성
    patterns = [
        ("Grains","Stews","Sides"),
        ("Grains","Proteins","Sides"),
        ("Grains","Sides","Fruits")
    ]
    cats = {"Grains":Grains,"Stews":Stews,"Proteins":Proteins,"Sides":Sides,"Fruits":Fruits}
    combos = []
    for pat in patterns:
        for combo in itertools.product(*(cats[p] for p in pat)):
            kc = sum(item["kcal"]    for item in combo)
            cb = sum(item["carb"]    for item in combo)
            pr = sum(item["protein"] for item in combo)
            ft = sum(item["fat"]     for item in combo)
            feats = [bmi, age, activity, kc, cb, pr, ft]
            combos.append({"combo":combo, "feat":feats, "kcal":kc})

    # 7) 라벨링 & 모델 학습
    X, y = [], []
    for c in combos:
        kc, cb, pr, ft = c["feat"][3:]
        pr_ratio = pr/(cb+pr+ft+1e-6)
        ideal_pr = prot_r
        score = 0.6*(1-abs(kc-per_meal)/per_meal) + 0.4*(1-abs(pr_ratio-ideal_pr))
        X.append(c["feat"]); y.append(1 if score>0.75 else 0)
    clf = Pipeline([("scaler",StandardScaler()),
                    ("model",RandomForestClassifier(n_estimators=100,random_state=42))])
    clf.fit(X,y); classes=clf.named_steps["model"].classes_

    def get_prob(feat):
        p=clf.predict_proba([feat])[0]
        return p[list(classes).index(1)] if 1 in classes else 0.0

    # 8) 추천 우선순위 산정
    ranked = []
    for c in combos:
        p = get_prob(c["feat"])
        err = abs(c["kcal"]-per_meal)
        ranked.append((p, -err, c))
    ranked.sort(reverse=True)
    plan = ranked[:3]

    # 9) 결과 출력
    st.markdown(f"**TDEE:** {tdee:.0f} kcal  |  **매크로:** {carb_r:.2f}/{prot_r:.2f}/{fat_r:.2f}")
    st.markdown("### 🍽️ 하루 식단 계획")
    times = ["07:30 아침","12:30 점심","18:30 저녁"]
    for (p,_,c), t in zip(plan, times):
        names = " + ".join(m["name"] for m in c["combo"])
        st.write(f"{t} → **{names}** ({c['kcal']} kcal, 적합도 {p:.2f})")
        st.write(f"   - 탄:{c['feat'][4]}g 단:{c['feat'][5]}g 지:{c['feat'][6]}g")
