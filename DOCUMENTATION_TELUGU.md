# MLOps డయాబెటిస్ ప్రిడిక్షన్ - పూర్తి డాక్యుమెంటేషన్ (తెలుగు)

## ప్రాజెక్ట్ పరిచయం
ఈ ప్రాజెక్ట్ మెషిన్ లెర్నింగ్, FastAPI, Docker, మరియు Kubernetes ఉపయోగించి డయాబెటిస్ అంచనా అప్లికేషన్ కోసం పూర్తి MLOps పైప్‌లైన్‌ను ప్రదర్శిస్తుంది.

---

## విషయ సూచిక
1. [ప్రాజెక్ట్ ఆర్కిటెక్చర్](#ప్రాజెక్ట్-ఆర్కిటెక్చర్)
2. [దశల వారీగా అమలు](#దశల-వారీగా-అమలు)
3. [కోడ్ వివరణ](#కోడ్-వివరణ)
4. [డిప్లాయ్మెంట్ గైడ్](#డిప్లాయ్మెంట్-గైడ్)
5. [అప్లికేషన్ టెస్టింగ్](#అప్లికేషన్-టెస్టింగ్)

---

## ప్రాజెక్ట్ ఆర్కిటెక్చర్

```
┌─────────────────┐
│  యూజర్ రిక్వెస్ట్ │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kubernetes     │
│  LoadBalancer   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Pods   │
│  (2 రెప్లికాలు)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML మోడల్        │
│  (Random Forest)│
└─────────────────┘
```

---

## దశల వారీగా అమలు

### దశ 1: మోడల్ శిక్షణ (train.py)

**ఉద్దేశం**: రోగి డేటా ఆధారంగా డయాబెటిస్‌ను అంచనా వేయడానికి మెషిన్ లెర్నింగ్ మోడల్‌ను శిక్షణ ఇవ్వడం.

**కోడ్**:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# డేటాసెట్‌ను లోడ్ చేయడం
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("✅ కాలమ్స్:", df.columns.tolist())

# డేటాను సిద్ధం చేయడం
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# విభజన
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# మోడల్ శిక్షణ
model = RandomForestClassifier()
model.fit(X_train, y_train)

# సేవ్ చేయడం
joblib.dump(model, "diabetes_model.pkl")
print("✅ మోడల్ diabetes_model.pkl గా సేవ్ చేయబడింది")
```

**వివరణ**:

1. **లైబ్రరీలను ఇంపోర్ట్ చేయడం**:
   - `pandas`: డేటా మానిప్యులేషన్ కోసం
   - `train_test_split`: డేటాను శిక్షణ మరియు పరీక్ష సెట్‌లుగా విభజించడానికి
   - `RandomForestClassifier`: మెషిన్ లెర్నింగ్ అల్గోరిథం
   - `joblib`: శిక్షణ పొందిన మోడల్‌ను సేవ్ చేయడానికి

2. **డేటాసెట్ లోడ్ చేయడం**:
   - GitHub నుండి డయాబెటిస్ డేటాసెట్‌ను లోడ్ చేస్తుంది
   - డేటాసెట్‌లో రోగి సమాచారం మరియు డయాబెటిస్ ఫలితాలు ఉంటాయి

3. **ఫీచర్లను సిద్ధం చేయడం**:
   - `X`: ఇన్‌పుట్ ఫీచర్లు (గర్భాలు, గ్లూకోజ్, రక్తపోటు, BMI, వయస్సు)
   - `y`: టార్గెట్ వేరియబుల్ (ఫలితం - 0 లేదా 1, డయాబెటిస్‌ను సూచిస్తుంది)

4. **డేటాను విభజించడం**:
   - 80% శిక్షణ కోసం
   - 20% పరీక్ష కోసం
   - `random_state=42` పునరుత్పత్తిని నిర్ధారిస్తుంది

5. **మోడల్ శిక్షణ**:
   - Random Forest Classifier ఉపయోగిస్తుంది
   - Random Forest బహుళ నిర్ణయ చెట్లను సృష్టిస్తుంది మరియు వాటి అంచనాలను కలుపుతుంది

6. **మోడల్‌ను సేవ్ చేయడం**:
   - శిక్షణ పొందిన మోడల్‌ను `diabetes_model.pkl` గా సేవ్ చేస్తుంది
   - ఈ ఫైల్ API ద్వారా ఉపయోగించబడుతుంది

**ఎలా రన్ చేయాలి**:
```bash
python train.py
```

---

### దశ 2: API అభివృద్ధి (main.py)

**ఉద్దేశం**: శిక్షణ పొందిన మోడల్‌ను ఉపయోగించి అంచనాలను అందించడానికి REST API సృష్టించడం.

**కోడ్**:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("diabetes_model.pkl")

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int

@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is live"}

@app.post("/predict")
def predict(data: DiabetesInput):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    prediction = model.predict(input_data)[0]
    return {"diabetic": bool(prediction)}
```

**వివరణ**:

1. **లైబ్రరీలను ఇంపోర్ట్ చేయడం**:
   - `FastAPI`: APIలను నిర్మించడానికి ఆధునిక వెబ్ ఫ్రేమ్‌వర్క్
   - `BaseModel`: డేటా ధృవీకరణ కోసం
   - `joblib`: సేవ్ చేసిన మోడల్‌ను లోడ్ చేయడానికి
   - `numpy`: అర్రే ఆపరేషన్స్ కోసం

2. **FastAPI ప్రారంభించడం**:
   - `app = FastAPI()`: API అప్లికేషన్‌ను సృష్టిస్తుంది

3. **మోడల్ లోడ్ చేయడం**:
   - `diabetes_model.pkl` నుండి గతంలో శిక్షణ పొందిన మోడల్‌ను లోడ్ చేస్తుంది

4. **ఇన్‌పుట్ స్కీమా నిర్వచించడం**:
   - `DiabetesInput` క్లాస్ ఆశించిన ఇన్‌పుట్ ఫార్మాట్‌ను నిర్వచిస్తుంది
   - డేటా ధృవీకరణను నిర్ధారిస్తుంది (సరైన రకాలు)
   - ఫీల్డ్స్: గర్భాలు (int), గ్లూకోజ్ (float), రక్తపోటు (float), BMI (float), వయస్సు (int)

5. **రూట్ ఎండ్‌పాయింట్** (`GET /`):
   - సాధారణ హెల్త్ చెక్ ఎండ్‌పాయింట్
   - API రన్ అవుతోందని నిర్ధారించే సందేశాన్ని తిరిగి ఇస్తుంది

6. **ప్రిడిక్షన్ ఎండ్‌పాయింట్** (`POST /predict`):
   - JSON ఫార్మాట్‌లో రోగి డేటాను అంగీకరిస్తుంది
   - ఇన్‌పుట్‌ను numpy అర్రేగా మారుస్తుంది
   - అంచనా వేయడానికి మోడల్‌ను ఉపయోగిస్తుంది
   - ఫలితాన్ని JSON గా తిరిగి ఇస్తుంది: `{"diabetic": true/false}`

**స్థానికంగా ఎలా రన్ చేయాలి**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**API టెస్ట్ చేయడం**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "BMI": 33.6, "Age": 50}'
```

---

### దశ 3: డిపెండెన్సీలు (requirements.txt)

**ఉద్దేశం**: ప్రాజెక్ట్‌కు అవసరమైన అన్ని Python ప్యాకేజీలను జాబితా చేయడం.

**కోడ్**:
```
fastapi
uvicorn[standard]
scikit-learn
pandas
joblib
```

**వివరణ**:

1. **fastapi**: API నిర్మించడానికి వెబ్ ఫ్రేమ్‌వర్క్
2. **uvicorn[standard]**: FastAPI రన్ చేయడానికి ASGI సర్వర్ (ప్రామాణిక ఫీచర్లతో)
3. **scikit-learn**: మెషిన్ లెర్నింగ్ లైబ్రరీ (RandomForestClassifier కలిగి ఉంటుంది)
4. **pandas**: డేటా మానిప్యులేషన్ మరియు విశ్లేషణ
5. **joblib**: Python ఆబ్జెక్ట్‌ల సమర్థవంతమైన సీరియలైజేషన్ (మోడల్‌లను సేవ్/లోడ్ చేయడానికి)

---

### దశ 4: కంటైనరైజేషన్ (Dockerfile)

**ఉద్దేశం**: స్థిరమైన డిప్లాయ్మెంట్ కోసం అప్లికేషన్‌ను Docker కంటైనర్‌లో ప్యాకేజ్ చేయడం.

**కోడ్**:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**వివరణ**:

1. **FROM python:3.10**:
   - Python 3.10 ఇన్‌స్టాల్ చేసిన బేస్ ఇమేజ్
   - రన్‌టైమ్ వాతావరణాన్ని అందిస్తుంది

2. **WORKDIR /app**:
   - కంటైనర్ లోపల వర్కింగ్ డైరెక్టరీని `/app`గా సెట్ చేస్తుంది
   - తదుపరి అన్ని కమాండ్‌లు ఈ డైరెక్టరీ నుండి రన్ అవుతాయి

3. **COPY . /app**:
   - ప్రస్తుత డైరెక్టరీ నుండి అన్ని ఫైల్‌లను కంటైనర్‌లోని `/app`కు కాపీ చేస్తుంది
   - కలిగి ఉంటుంది: main.py, train.py, requirements.txt, diabetes_model.pkl

4. **RUN pip install -r requirements.txt**:
   - అన్ని Python డిపెండెన్సీలను ఇన్‌స్టాల్ చేస్తుంది
   - ఇమేజ్ బిల్డ్ సమయంలో రన్ అవుతుంది (రన్‌టైమ్‌లో కాదు)

5. **CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]**:
   - కంటైనర్ ప్రారంభమైనప్పుడు రన్ చేయడానికి కమాండ్
   - FastAPI సర్వర్‌ను ప్రారంభిస్తుంది
   - `--host 0.0.0.0`: కంటైనర్ వెలుపల నుండి సర్వర్‌ను యాక్సెస్ చేయగలిగేలా చేస్తుంది
   - `--port 8000`: పోర్ట్ 8000లో రన్ అవుతుంది

**Docker ఇమేజ్ బిల్డ్ చేయడం**:
```bash
docker build -t anildocker54321/diabetes-prediction-model:latest .
```

**Docker Hub కు పుష్ చేయడం**:
```bash
docker login
docker push anildocker54321/diabetes-prediction-model:latest
```

**కంటైనర్‌ను స్థానికంగా రన్ చేయడం**:
```bash
docker run -p 8000:8000 anildocker54321/diabetes-prediction-model:latest
```

---

### దశ 5: Kubernetes డిప్లాయ్మెంట్ (k8s-deploy.yml)

**ఉద్దేశం**: అధిక లభ్యత మరియు లోడ్ బ్యాలెన్సింగ్‌తో Kubernetes కు అప్లికేషన్‌ను డిప్లాయ్ చేయడం.

**కోడ్**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diabetes-api
  labels:
    app: diabetes-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: diabetes-api
  template:
    metadata:
      labels:
        app: diabetes-api
    spec:
      containers:
      - name: diabetes-api
        image: anildocker54321/diabetes-prediction-model:latest
        ports:
        - containerPort: 8000
        imagePullPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  name: diabetes-api-service
spec:
  selector:
    app: diabetes-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

**వివరణ**:

#### భాగం 1: డిప్లాయ్మెంట్

1. **apiVersion: apps/v1**:
   - డిప్లాయ్మెంట్ రిసోర్స్‌ల కోసం Kubernetes API వెర్షన్

2. **kind: Deployment**:
   - దీన్ని డిప్లాయ్మెంట్ రిసోర్స్‌గా నిర్వచిస్తుంది
   - పాడ్ సృష్టి మరియు స్కేలింగ్‌ను నిర్వహిస్తుంది

3. **metadata**:
   - `name: diabetes-api`: డిప్లాయ్మెంట్ పేరు
   - `labels`: రిసోర్స్‌లను నిర్వహించడానికి ట్యాగ్‌లు

4. **spec.replicas: 2**:
   - 2 ఒకే విధమైన పాడ్‌లను సృష్టిస్తుంది
   - అధిక లభ్యత మరియు లోడ్ పంపిణీని అందిస్తుంది

5. **selector.matchLabels**:
   - ఈ డిప్లాయ్మెంట్‌కు ఏ పాడ్‌లు చెందుతాయో గుర్తిస్తుంది
   - `app: diabetes-api` లేబుల్‌తో పాడ్‌లను సరిపోల్చుతుంది

6. **template**:
   - పాడ్ స్పెసిఫికేషన్‌ను నిర్వచిస్తుంది
   - పాడ్‌లను సృష్టించడానికి బ్లూప్రింట్

7. **containers**:
   - `name: diabetes-api`: కంటైనర్ పేరు
   - `image`: ఉపయోగించాల్సిన Docker ఇమేజ్
   - `containerPort: 8000`: కంటైనర్ ద్వారా బహిర్గతం చేయబడిన పోర్ట్
   - `imagePullPolicy: Always`: రిజిస్ట్రీ నుండి ఎల్లప్పుడూ తాజా ఇమేజ్‌ను పుల్ చేస్తుంది

#### భాగం 2: సర్వీస్

1. **apiVersion: v1**:
   - సర్వీస్ రిసోర్స్‌ల కోసం Kubernetes API వెర్షన్

2. **kind: Service**:
   - దీన్ని సర్వీస్ రిసోర్స్‌గా నిర్వచిస్తుంది
   - నెట్‌వర్కింగ్ మరియు లోడ్ బ్యాలెన్సింగ్‌ను అందిస్తుంది

3. **metadata.name: diabetes-api-service**:
   - సర్వీస్ పేరు

4. **spec.selector**:
   - `app: diabetes-api` లేబుల్‌తో పాడ్‌లకు ట్రాఫిక్‌ను రూట్ చేస్తుంది

5. **ports**:
   - `protocol: TCP`: TCP ప్రోటోకాల్‌ను ఉపయోగిస్తుంది
   - `port: 80`: బాహ్య పోర్ట్ (యూజర్లు యాక్సెస్ చేసేది)
   - `targetPort: 8000`: అంతర్గత పోర్ట్ (కంటైనర్ పోర్ట్)

6. **type: LoadBalancer**:
   - బాహ్య లోడ్ బ్యాలెన్సర్‌ను సృష్టిస్తుంది
   - పాడ్‌ల మధ్య ట్రాఫిక్‌ను పంపిణీ చేస్తుంది
   - ఒకే ఎంట్రీ పాయింట్‌ను అందిస్తుంది

---

## డిప్లాయ్మెంట్ గైడ్

### ముందస్తు అవసరాలు
- Docker ఇన్‌స్టాల్ చేయబడింది
- Kubernetes క్లస్టర్ (Minikube/Docker Desktop/Cloud)
- kubectl కాన్ఫిగర్ చేయబడింది

### దశల వారీగా డిప్లాయ్మెంట్

#### 1. మోడల్‌ను శిక్షణ ఇవ్వండి
```bash
python train.py
```
అవుట్‌పుట్: `diabetes_model.pkl` ఫైల్ సృష్టించబడింది

#### 2. Docker ఇమేజ్ బిల్డ్ చేయండి
```bash
docker build -t anildocker54321/diabetes-prediction-model:latest .
```

#### 3. Docker Hub కు పుష్ చేయండి
```bash
docker login
docker push anildocker54321/diabetes-prediction-model:latest
```

#### 4. Kubernetes క్లస్టర్ ప్రారంభించండి (Minikube)
```bash
minikube start
```

#### 5. Kubernetes కు డిప్లాయ్ చేయండి
```bash
kubectl apply -f k8s-deploy.yml
```

#### 6. డిప్లాయ్మెంట్ స్థితిని తనిఖీ చేయండి
```bash
kubectl get pods
kubectl get services
```

#### 7. అప్లికేషన్‌ను యాక్సెస్ చేయండి

**ఎంపిక A: Minikube సర్వీస్**
```bash
minikube service diabetes-api-service
```

**ఎంపిక B: పోర్ట్ ఫార్వర్డ్**
```bash
kubectl port-forward service/diabetes-api-service 8000:80
```
యాక్సెస్: http://localhost:8000

**ఎంపిక C: Minikube టన్నెల్**
```bash
minikube tunnel
```
యాక్సెస్: http://localhost

---

## అప్లికేషన్ టెస్టింగ్

### 1. హెల్త్ చెక్
```bash
curl http://localhost:8000/
```
ఆశించిన ప్రతిస్పందన:
```json
{"message": "Diabetes Prediction API is live"}
```

### 2. అంచనా వేయడం
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "BMI": 33.6,
    "Age": 50
  }'
```

ఆశించిన ప్రతిస్పందన:
```json
{"diabetic": true}
```

### 3. వేరే డేటాతో టెస్ట్ చేయడం
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 1,
    "Glucose": 85,
    "BloodPressure": 66,
    "BMI": 26.6,
    "Age": 31
  }'
```

ఆశించిన ప్రతిస్పందన:
```json
{"diabetic": false}
```

---

## పర్యవేక్షణ మరియు నిర్వహణ

### లాగ్‌లను చూడండి
```bash
kubectl logs -l app=diabetes-api
```

### అప్లికేషన్‌ను స్కేల్ చేయండి
```bash
kubectl scale deployment diabetes-api --replicas=5
```

### అప్లికేషన్‌ను అప్‌డేట్ చేయండి
```bash
# కొత్త ఇమేజ్ బిల్డ్ చేయండి
docker build -t anildocker54321/diabetes-prediction-model:v2 .
docker push anildocker54321/diabetes-prediction-model:v2

# డిప్లాయ్మెంట్ అప్‌డేట్ చేయండి
kubectl set image deployment/diabetes-api diabetes-api=anildocker54321/diabetes-prediction-model:v2
```

### డిప్లాయ్మెంట్ తొలగించండి
```bash
kubectl delete -f k8s-deploy.yml
```

---

## సమస్యా పరిష్కారం

### పాడ్‌లు ప్రారంభం కావడం లేదు
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### ఇమేజ్ పుల్ ఎర్రర్‌లు
- Docker Hub లో ఇమేజ్ ఉందో ధృవీకరించండి
- k8s-deploy.yml లో ఇమేజ్ పేరును తనిఖీ చేయండి
- Docker Hub క్రెడెన్షియల్స్ సరైనవో నిర్ధారించండి

### సర్వీస్ యాక్సెస్ చేయలేకపోతోంది
```bash
kubectl get services
minikube service list
```

---

## ప్రాజెక్ట్ నిర్మాణం
```
diabetes-prediction/
├── main.py                 # FastAPI అప్లికేషన్
├── train.py                # మోడల్ శిక్షణ స్క్రిప్ట్
├── requirements.txt        # Python డిపెండెన్సీలు
├── Dockerfile              # కంటైనర్ కాన్ఫిగరేషన్
├── k8s-deploy.yml          # Kubernetes డిప్లాయ్మెంట్
├── diabetes_model.pkl      # శిక్షణ పొందిన మోడల్ (జనరేట్ చేయబడింది)
└── README.md               # ప్రాజెక్ట్ డాక్యుమెంటేషన్
```

---

## ముఖ్య భావనలు

### MLOps పైప్‌లైన్
1. **మోడల్ శిక్షణ**: ML మోడల్‌ను సృష్టించి శిక్షణ ఇవ్వడం
2. **API అభివృద్ధి**: మోడల్‌ను REST API లో చుట్టడం
3. **కంటైనరైజేషన్**: Docker లో ప్యాకేజ్ చేయడం
4. **ఆర్కెస్ట్రేషన్**: Kubernetes తో డిప్లాయ్ చేయడం
5. **పర్యవేక్షణ**: పనితీరు మరియు లాగ్‌లను ట్రాక్ చేయడం

### ప్రయోజనాలు
- **స్కేలబిలిటీ**: మరిన్ని అభ్యర్థనలను సులభంగా నిర్వహించడం
- **అధిక లభ్యత**: బహుళ రెప్లికాలు అప్‌టైమ్‌ను నిర్ధారిస్తాయి
- **పోర్టబిలిటీ**: Docker/Kubernetes రన్ అయ్యే చోట ఎక్కడైనా రన్ అవుతుంది
- **నిర్వహణ**: సులభమైన అప్‌డేట్‌లు మరియు రోల్‌బ్యాక్‌లు
- **పర్యవేక్షణ**: అంతర్నిర్మిత లాగింగ్ మరియు హెల్త్ చెక్‌లు

---

## ముగింపు
ఈ ప్రాజెక్ట్ ఆధునిక DevOps పద్ధతులు మరియు సాధనాలను ఉపయోగించి మోడల్ శిక్షణ నుండి ఉత్పత్తి డిప్లాయ్మెంట్ వరకు పూర్తి MLOps వర్క్‌ఫ్లోను ప్రదర్శిస్తుంది.
