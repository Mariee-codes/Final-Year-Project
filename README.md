# ShambaIQ: an AI advisor for small-scale Kenyan farmers

Group 3 final-year project. A farmer photographs a sick leaf and gets a diagnosis,
plus planting advice, a price outlook and buyer matches, in English or Swahili.

## What is already working

| Part | Status |
| --- | --- |
| FastAPI backend with all endpoints | Working, tested |
| Crop Doctor inference, confidence threshold, EN/SW advice | Working; needs a trained model |
| Crop Doctor training script (lab + field evaluation, ONNX export) | Ready to run on Colab/Kaggle |
| Planting Advisor (Open-Meteo weather + crop rules + soil pH) | Working |
| Price Forecaster (naive, seasonal naive, gradient boosting) | Working on synthetic data; needs real KAMIS data |
| Buyer Matcher (distance, quantity, price scoring) | Working |
| Weekly dashboard linking all modules | Working |
| Mobile web page (English / Kiswahili) | Basic version |
| Tests | 24 passing |

## Project structure

```
shambaiq/
  backend/app/main.py              API endpoints
  backend/app/modules/
    crop_doctor.py                 disease model inference + advice
    planting_advisor.py            weather + soil rules
    price_forecaster.py            KAMIS price models
    buyer_matcher.py               match scoring
    dashboard.py                   combines modules into a weekly plan
  backend/app/data/
    treatments.json                treatment advice (EN/SW) per disease
    crop_requirements.json         planting rules per crop
  ml/
    train_crop_doctor.py           train + evaluate + export the disease model
    evaluate_prices.py             compare price models honestly
    make_sample_kamis.py           creates SYNTHETIC test price data
  data/                            datasets go here (not committed)
  models/                          trained model files go here
  frontend/index.html              mobile web app
  tests/                           automated tests
```

## Getting started (first day)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python ml/make_sample_kamis.py     # synthetic prices so the API runs
pytest -q                          # should show 24 passed
uvicorn backend.app.main:app --reload
```

Open http://127.0.0.1:8000 for the app and http://127.0.0.1:8000/docs for the API.

## Step 1: train the Crop Doctor (Google Colab, free GPU)

1. Download the PlantVillage dataset (colour images) from Kaggle.
2. Keep only the folders starting with `Tomato___`, `Potato___` and `Corn_(maize)___`
   and put them in `data/plantvillage/`.
3. Build the field test set in `data/field/`, using the **same folder names**:
   PlantDoc images for matching classes, plus photos you take on Kenyan farms.
   Never train on these unless you deliberately fine-tune and keep a separate test part.
4. In Colab: `pip install torch torchvision onnx scikit-learn` then
   `python ml/train_crop_doctor.py --data data/plantvillage --field data/field --epochs 12`
5. Copy `models/crop_doctor.onnx` and `models/labels.json` into this project's `models/` folder.
   The API's `/diagnose` endpoint switches on automatically.
6. Put `models/report.json` (lab accuracy vs field accuracy) in your report.

## Step 2: real price data

Export maize, tomato and potato prices from KAMIS (kamis.kilimo.go.ke) as CSV with columns
`date, commodity, county, market, wholesale_price`, then:

```bash
python ml/evaluate_prices.py --csv data/kamis_export.csv --crop maize --county Nakuru
KAMIS_CSV=data/kamis_export.csv uvicorn backend.app.main:app --reload
```

Results from the synthetic file only prove the code runs. Do not report them.
Next step: add a SARIMA model (statsmodels) to `price_forecaster.evaluate()`, as planned in the proposal.

## Before the pilot: things to verify

- `treatments.json`: every treatment step and `yield_impact` value is a placeholder;
  check each with an extension officer or a KALRO guide.
- `crop_requirements.json`: check seasons, rainfall and pH ranges for your pilot county.
- Replace the in-memory storage in `main.py` with PostgreSQL tables.
- Get informed consent before collecting farm photos or farmer data (Data Protection Act, 2019).

## Suggested task split for Group 3

| Member | Main area |
| --- | --- |
| 1 | Crop Doctor: data collection, training, lab vs field evaluation |
| 2 | Field photo collection and labelling; farmer interviews and pilot |
| 3 | Price Forecaster with real KAMIS data |
| 4 | Planting Advisor and Buyer Matcher; database |
| 5 | Front end (English/Swahili), integration, testing, report |

## Sprint plan (about three weeks each, over six months)

1. Set up repo, environment, requirements interviews, start collecting field photos
2. Train first Crop Doctor model; measure lab accuracy
3. Field test set complete; measure field accuracy; fine-tune; connect to API
4. Real KAMIS data; price model evaluation
5. Planting Advisor with real county data; buyer matching with test users
6. Front end polish, Swahili review, database
7. Farmer pilot (about 20 farmers), usability survey
8. Evaluation, final report and demo
