"""
DerekSafe — Privacy-Preserving Anti-Fraud via Federated Learning (Hackathon MVP)
================================================================================
Banks A, B and C jointly train one fraud-detection model. Each bank trains on
its own data locally; only (noised, encrypted) weight updates reach the
central aggregator, which averages them (FedAvg) into a global model.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Everything here is simulated in-process with scikit-learn. Comments marked
"PRODUCTION:" show where the real federated-learning infrastructure plugs in.
"""

import hashlib
import time
import warnings
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

st.set_page_config(page_title="DerekSafe", page_icon="🛡️", layout="wide")

# =============================================================================
# 1. TRANSLATIONS (EN / RU / KZ)
# =============================================================================
T = {
    "en": {
        "tagline": "Federated anti-fraud intelligence · one shared model, zero shared data",
        "badge": "Secure Aggregation · TLS 1.3 · ε-Differential Privacy",
        "tab_network": "🛰 Network Control",
        "tab_monitor": "🔎 Transaction Monitor",
        "tab_privacy": "🛡 Privacy Shield",
        # Network
        "net_title": "Federated Training Network",
        "net_desc": "Each bank trains the model on its own data, inside its own perimeter. "
                    "Only encrypted weight updates travel to the aggregator — raw transactions never leave the bank.",
        "bank": "Bank {x}",
        "bank_records": "{n:,} local transactions",
        "bank_pattern": "Local fraud pattern: {p}",
        "data_stays": "🔒 Data stays on-premise",
        "p_cross_border": "cross-border card fraud",
        "p_night_burst": "night-time burst transfers",
        "p_account_takeover": "account takeover",
        "aggregator": "Central Aggregator",
        "aggregator_sub": "Secure FedAvg · sees only encrypted weights",
        "rounds_label": "Federated rounds",
        "start_training": "▶ Start Training",
        "reset_model": "↺ Reset model",
        "phase_upload": "Round {r}/{n} · Banks → Aggregator: encrypted weight updates",
        "phase_broadcast": "Round {r}/{n} · Aggregator → Banks: new global model",
        "net_idle": "Network idle · press «Start Training»",
        "net_synced": "Global model synced to all banks",
        "training_done": "Training complete · global model accuracy {acc:.1%}",
        "chart_title": "Model accuracy by round (test set covering all 3 fraud patterns)",
        "series_fed": "Federated model (A+B+C)",
        "series_local": "Bank A alone",
        "axis_round": "Round",
        "axis_acc": "Accuracy",
        "m_accuracy": "Global accuracy",
        "m_uplift": "Uplift vs. single bank",
        "m_rounds": "Rounds completed",
        "m_shared": "Raw records shared",
        "chart_empty": "The accuracy chart appears after the first training run.",
        # Monitor
        "mon_title": "Live Transaction Monitor",
        "mon_desc": "Every transaction is scored locally by the federated model. "
                    "Client identifiers are masked; nothing is sent outside the bank.",
        "live_feed": "Live feed",
        "simulate_attack": "⚠ Simulate Fraud Attack",
        "reset_feed": "↺ Reset feed",
        "alert_title": "Fraud Blocked. Pattern matched. Client data NOT exposed.",
        "alert_detail": "{id} · {amount} · {merchant} · risk score {score:.0%}",
        "attack_story": "Target: Bank A. This pattern was learned from Bank B and Bank C through "
                        "federated training — Bank A never saw their data.",
        "untrained_warn": "Global model not trained yet — scoring uses fallback rules. "
                          "Run training in «Network Control» for ML-based detection.",
        "col_time": "Time", "col_id": "Tx ID", "col_bank": "Bank", "col_client": "Client",
        "col_amount": "Amount", "col_merchant": "Merchant", "col_country": "Country",
        "col_risk": "Risk", "col_status": "Status",
        "st_ok": "✓ Approved", "st_review": "⚑ Review", "st_blocked": "⛔ BLOCKED",
        "m_scanned": "Transactions scanned", "m_blocked": "Fraud blocked",
        "m_exposed": "Client records exposed", "m_latency": "Avg. scoring latency",
        "mer_groceries": "Groceries", "mer_restaurants": "Restaurants", "mer_fuel": "Fuel",
        "mer_electronics": "Electronics", "mer_travel": "Travel", "mer_p2p": "P2P transfer",
        "mer_gaming": "Online gaming", "mer_crypto": "Crypto exchange",
        # Privacy
        "priv_title": "Privacy Shield",
        "priv_desc": "Differential Privacy adds calibrated Gaussian noise to every bank's weight update "
                     "before it is encrypted and sent. Even the aggregator cannot reverse-engineer an "
                     "individual client from the model.",
        "noise_slider": "Differential Privacy Noise Level",
        "privacy_level": "Privacy Level: {p}% — {label}",
        "lvl_none": "No protection: updates may leak client data",
        "lvl_weak": "Weak: re-identification risk",
        "lvl_moderate": "Moderate protection",
        "lvl_strong": "Strong protection",
        "lvl_full": "Data fully anonymized",
        "m_epsilon": "Privacy budget ε",
        "m_sigma": "Noise multiplier σ",
        "m_cost": "Est. accuracy cost",
        "update_chart_title": "What the aggregator receives: Bank A's weight update per feature",
        "series_raw": "Raw update (never leaves bank)",
        "series_noised": "DP-protected update (sent)",
        "axis_feature": "Feature", "axis_value": "Δ weight",
        "priv_applied": "This noise level is applied to the next training run in «Network Control».",
        "guarantees_title": "Privacy guarantees",
        "g1": "Raw transactions never leave the bank",
        "g2": "Weight updates encrypted end-to-end (Secure Aggregation)",
        "g3": "Differential Privacy noise on every update",
        "g4": "Client IDs masked in every dashboard view",
        "f_amount": "Amount", "f_night": "Night hours", "f_merchant_risk": "Merchant risk",
        "f_distance": "Distance from home", "f_velocity": "Tx velocity",
        "f_foreign": "Foreign", "f_new_device": "New device", "f_bias": "Bias",
        "footer": "DerekSafe MVP · simulated data · demo build",
    },
    "ru": {
        "tagline": "Федеративный антифрод · одна общая модель, ноль переданных данных",
        "badge": "Защищённая агрегация · TLS 1.3 · ε-дифференциальная приватность",
        "tab_network": "🛰 Управление сетью",
        "tab_monitor": "🔎 Мониторинг транзакций",
        "tab_privacy": "🛡 Щит приватности",
        "net_title": "Сеть федеративного обучения",
        "net_desc": "Каждый банк обучает модель на своих данных внутри своего периметра. "
                    "В агрегатор уходят только зашифрованные обновления весов — сырые транзакции никогда не покидают банк.",
        "bank": "Банк {x}",
        "bank_records": "{n:,} локальных транзакций",
        "bank_pattern": "Локальная схема мошенничества: {p}",
        "data_stays": "🔒 Данные остаются в банке",
        "p_cross_border": "трансграничные операции по картам",
        "p_night_burst": "ночные серии переводов",
        "p_account_takeover": "захват аккаунта",
        "aggregator": "Центральный агрегатор",
        "aggregator_sub": "Secure FedAvg · видит только зашифрованные веса",
        "rounds_label": "Раунды обучения",
        "start_training": "▶ Начать обучение",
        "reset_model": "↺ Сбросить модель",
        "phase_upload": "Раунд {r}/{n} · Банки → Агрегатор: зашифрованные обновления весов",
        "phase_broadcast": "Раунд {r}/{n} · Агрегатор → Банки: новая глобальная модель",
        "net_idle": "Сеть в ожидании · нажмите «Начать обучение»",
        "net_synced": "Глобальная модель синхронизирована со всеми банками",
        "training_done": "Обучение завершено · точность глобальной модели {acc:.1%}",
        "chart_title": "Точность модели по раундам (тестовая выборка со всеми 3 схемами)",
        "series_fed": "Федеративная модель (A+B+C)",
        "series_local": "Только Банк A",
        "axis_round": "Раунд",
        "axis_acc": "Точность",
        "m_accuracy": "Глобальная точность",
        "m_uplift": "Прирост vs. один банк",
        "m_rounds": "Пройдено раундов",
        "m_shared": "Передано сырых записей",
        "chart_empty": "График точности появится после первого запуска обучения.",
        "mon_title": "Мониторинг транзакций в реальном времени",
        "mon_desc": "Каждая транзакция оценивается федеративной моделью локально. "
                    "Идентификаторы клиентов замаскированы; ничего не покидает банк.",
        "live_feed": "Живой поток",
        "simulate_attack": "⚠ Симулировать атаку",
        "reset_feed": "↺ Сбросить ленту",
        "alert_title": "Мошенничество заблокировано. Схема распознана. Данные клиента НЕ раскрыты.",
        "alert_detail": "{id} · {amount} · {merchant} · риск {score:.0%}",
        "attack_story": "Цель: Банк A. Эта схема выучена на данных Банка B и Банка C через "
                        "федеративное обучение — Банк A никогда не видел их данных.",
        "untrained_warn": "Глобальная модель ещё не обучена — используются резервные правила. "
                          "Запустите обучение во вкладке «Управление сетью».",
        "col_time": "Время", "col_id": "ID транзакции", "col_bank": "Банк", "col_client": "Клиент",
        "col_amount": "Сумма", "col_merchant": "Категория", "col_country": "Страна",
        "col_risk": "Риск", "col_status": "Статус",
        "st_ok": "✓ Одобрено", "st_review": "⚑ Проверка", "st_blocked": "⛔ ЗАБЛОКИРОВАНО",
        "m_scanned": "Проверено транзакций", "m_blocked": "Заблокировано атак",
        "m_exposed": "Раскрыто записей клиентов", "m_latency": "Средняя задержка",
        "mer_groceries": "Продукты", "mer_restaurants": "Рестораны", "mer_fuel": "Топливо",
        "mer_electronics": "Электроника", "mer_travel": "Путешествия", "mer_p2p": "P2P-перевод",
        "mer_gaming": "Онлайн-игры", "mer_crypto": "Криптобиржа",
        "priv_title": "Щит приватности",
        "priv_desc": "Дифференциальная приватность добавляет откалиброванный гауссов шум к обновлению весов "
                     "каждого банка до шифрования и отправки. Даже агрегатор не может восстановить данные "
                     "отдельного клиента по модели.",
        "noise_slider": "Уровень шума дифференциальной приватности",
        "privacy_level": "Уровень приватности: {p}% — {label}",
        "lvl_none": "Нет защиты: обновления могут раскрыть данные клиентов",
        "lvl_weak": "Слабая: риск деанонимизации",
        "lvl_moderate": "Умеренная защита",
        "lvl_strong": "Сильная защита",
        "lvl_full": "Данные полностью анонимизированы",
        "m_epsilon": "Бюджет приватности ε",
        "m_sigma": "Множитель шума σ",
        "m_cost": "Ожид. потеря точности",
        "update_chart_title": "Что получает агрегатор: обновление весов Банка A по признакам",
        "series_raw": "Исходное обновление (не покидает банк)",
        "series_noised": "Обновление с DP-защитой (отправляется)",
        "axis_feature": "Признак", "axis_value": "Δ веса",
        "priv_applied": "Этот уровень шума применяется при следующем запуске обучения во вкладке «Управление сетью».",
        "guarantees_title": "Гарантии приватности",
        "g1": "Сырые транзакции не покидают банк",
        "g2": "Обновления весов зашифрованы (Secure Aggregation)",
        "g3": "Шум дифференциальной приватности в каждом обновлении",
        "g4": "ID клиентов замаскированы на всех экранах",
        "f_amount": "Сумма", "f_night": "Ночное время", "f_merchant_risk": "Риск мерчанта",
        "f_distance": "Удалённость", "f_velocity": "Частота операций",
        "f_foreign": "Зарубежная", "f_new_device": "Новое устройство", "f_bias": "Смещение",
        "footer": "DerekSafe MVP · симулированные данные · демо-версия",
    },
    "kz": {
        "tagline": "Федеративтік антифрод · бір ортақ модель, деректер берілмейді",
        "badge": "Қорғалған агрегация · TLS 1.3 · ε-дифференциалды құпиялылық",
        "tab_network": "🛰 Желіні басқару",
        "tab_monitor": "🔎 Транзакциялар мониторингі",
        "tab_privacy": "🛡 Құпиялылық қалқаны",
        "net_title": "Федеративтік оқыту желісі",
        "net_desc": "Әр банк модельді өз деректерінде, өз периметрі ішінде оқытады. "
                    "Агрегаторға тек шифрланған салмақ жаңартулары жіберіледі — бастапқы транзакциялар банктен ешқашан шықпайды.",
        "bank": "Банк {x}",
        "bank_records": "{n:,} жергілікті транзакция",
        "bank_pattern": "Жергілікті алаяқтық үлгісі: {p}",
        "data_stays": "🔒 Деректер банкте қалады",
        "p_cross_border": "шекарааралық карта алаяқтығы",
        "p_night_burst": "түнгі сериялық аударымдар",
        "p_account_takeover": "аккаунтты басып алу",
        "aggregator": "Орталық агрегатор",
        "aggregator_sub": "Secure FedAvg · тек шифрланған салмақтарды көреді",
        "rounds_label": "Оқыту раундтары",
        "start_training": "▶ Оқытуды бастау",
        "reset_model": "↺ Модельді қалпына келтіру",
        "phase_upload": "{r}/{n} раунд · Банктер → Агрегатор: шифрланған салмақ жаңартулары",
        "phase_broadcast": "{r}/{n} раунд · Агрегатор → Банктер: жаңа ғаламдық модель",
        "net_idle": "Желі күту режимінде · «Оқытуды бастау» батырмасын басыңыз",
        "net_synced": "Ғаламдық модель барлық банктермен синхрондалды",
        "training_done": "Оқыту аяқталды · ғаламдық модель дәлдігі {acc:.1%}",
        "chart_title": "Раундтар бойынша модель дәлдігі (барлық 3 үлгісі бар тест жиыны)",
        "series_fed": "Федеративтік модель (A+B+C)",
        "series_local": "Тек Банк A",
        "axis_round": "Раунд",
        "axis_acc": "Дәлдік",
        "m_accuracy": "Ғаламдық дәлдік",
        "m_uplift": "Бір банкпен салыстырғандағы өсім",
        "m_rounds": "Өткен раундтар",
        "m_shared": "Берілген бастапқы жазбалар",
        "chart_empty": "Дәлдік графигі алғашқы оқытудан кейін пайда болады.",
        "mon_title": "Нақты уақыттағы транзакциялар мониторингі",
        "mon_desc": "Әр транзакцияны федеративтік модель жергілікті түрде бағалайды. "
                    "Клиент идентификаторлары жасырылған; банктен ештеңе шықпайды.",
        "live_feed": "Тікелей ағын",
        "simulate_attack": "⚠ Алаяқтық шабуылын модельдеу",
        "reset_feed": "↺ Лентаны тазалау",
        "alert_title": "Алаяқтық бұғатталды. Үлгі анықталды. Клиент деректері АШЫЛМАДЫ.",
        "alert_detail": "{id} · {amount} · {merchant} · тәуекел {score:.0%}",
        "attack_story": "Нысана: Банк A. Бұл үлгі Банк B мен Банк C деректерінен федеративтік оқыту "
                        "арқылы үйренілді — Банк A олардың деректерін ешқашан көрген жоқ.",
        "untrained_warn": "Ғаламдық модель әлі оқытылмаған — резервтік ережелер қолданылады. "
                          "«Желіні басқару» қойындысында оқытуды іске қосыңыз.",
        "col_time": "Уақыт", "col_id": "Транзакция ID", "col_bank": "Банк", "col_client": "Клиент",
        "col_amount": "Сома", "col_merchant": "Санат", "col_country": "Ел",
        "col_risk": "Тәуекел", "col_status": "Мәртебе",
        "st_ok": "✓ Мақұлданды", "st_review": "⚑ Тексеру", "st_blocked": "⛔ БҰҒАТТАЛДЫ",
        "m_scanned": "Тексерілген транзакциялар", "m_blocked": "Бұғатталған алаяқтық",
        "m_exposed": "Ашылған клиент жазбалары", "m_latency": "Орташа кідіріс",
        "mer_groceries": "Азық-түлік", "mer_restaurants": "Мейрамханалар", "mer_fuel": "Жанармай",
        "mer_electronics": "Электроника", "mer_travel": "Саяхат", "mer_p2p": "P2P аударым",
        "mer_gaming": "Онлайн ойындар", "mer_crypto": "Криптобиржа",
        "priv_title": "Құпиялылық қалқаны",
        "priv_desc": "Дифференциалды құпиялылық әр банктің салмақ жаңартуына шифрлау мен жіберуден бұрын "
                     "калибрленген Гаусс шуын қосады. Тіпті агрегатор да модельден жеке клиенттің "
                     "деректерін қалпына келтіре алмайды.",
        "noise_slider": "Дифференциалды құпиялылық шу деңгейі",
        "privacy_level": "Құпиялылық деңгейі: {p}% — {label}",
        "lvl_none": "Қорғаныс жоқ: жаңартулар клиент деректерін ашуы мүмкін",
        "lvl_weak": "Әлсіз: қайта сәйкестендіру қаупі",
        "lvl_moderate": "Орташа қорғаныс",
        "lvl_strong": "Күшті қорғаныс",
        "lvl_full": "Деректер толығымен анонимделген",
        "m_epsilon": "Құпиялылық бюджеті ε",
        "m_sigma": "Шу көбейткіші σ",
        "m_cost": "Болжамды дәлдік шығыны",
        "update_chart_title": "Агрегатор не алады: Банк A салмақ жаңартуы (белгілер бойынша)",
        "series_raw": "Бастапқы жаңарту (банктен шықпайды)",
        "series_noised": "DP-қорғалған жаңарту (жіберіледі)",
        "axis_feature": "Белгі", "axis_value": "Δ салмақ",
        "priv_applied": "Бұл шу деңгейі «Желіні басқару» қойындысындағы келесі оқытуда қолданылады.",
        "guarantees_title": "Құпиялылық кепілдіктері",
        "g1": "Бастапқы транзакциялар банктен шықпайды",
        "g2": "Салмақ жаңартулары шифрланған (Secure Aggregation)",
        "g3": "Әр жаңартуда дифференциалды құпиялылық шуы",
        "g4": "Клиент ID барлық көріністерде жасырылған",
        "f_amount": "Сома", "f_night": "Түнгі уақыт", "f_merchant_risk": "Мерчант тәуекелі",
        "f_distance": "Үйден қашықтық", "f_velocity": "Операция жиілігі",
        "f_foreign": "Шетелдік", "f_new_device": "Жаңа құрылғы", "f_bias": "Ығысу",
        "footer": "DerekSafe MVP · модельденген деректер · демо нұсқа",
    },
}


def t(key: str, **kw) -> str:
    """Translate a key into the active language (falls back to English)."""
    lang = st.session_state.get("lang", "EN").lower()
    text = T.get(lang, T["en"]).get(key, T["en"].get(key, key))
    return text.format(**kw) if kw else text


# =============================================================================
# 2. MOCK DATA GENERATION
# =============================================================================
FEATURES = ["amount", "night", "merchant_risk", "distance", "velocity", "foreign", "new_device"]

# merchant -> base risk score
MERCHANTS = {"groceries": 0.10, "restaurants": 0.15, "fuel": 0.10, "electronics": 0.40,
             "travel": 0.35, "p2p": 0.50, "gaming": 0.60, "crypto": 0.85}
LEGIT_MIX = [0.30, 0.20, 0.15, 0.10, 0.08, 0.10, 0.04, 0.03]
FRAUD_MIX = [0.04, 0.03, 0.03, 0.25, 0.10, 0.20, 0.15, 0.20]
TAKEOVER_MIX = [0.01, 0.01, 0.01, 0.07, 0.02, 0.40, 0.08, 0.40]
FOREIGN_COUNTRIES = ["TR", "AE", "GE", "CN", "US", "DE", "TH"]
KZT_PER_USD = 470

# Each bank has a *different* dominant fraud pattern (non-IID data). That is
# exactly why federation helps: no single bank sees the whole fraud picture.
BANKS = {
    "A": dict(n=12000, fraud_rate=0.04, profile="cross_border", seed=11),
    "B": dict(n=9000, fraud_rate=0.05, profile="night_burst", seed=22),
    "C": dict(n=7000, fraud_rate=0.03, profile="account_takeover", seed=33),
}


def sample_transactions(n, rng, fraud=False, profile=None):
    """Generate n synthetic raw transactions (legit or fraud of a given profile)."""
    if n <= 0:
        return pd.DataFrame()
    keys = list(MERCHANTS)
    mix = FRAUD_MIX if fraud else LEGIT_MIX
    if profile == "account_takeover":
        mix = TAKEOVER_MIX
    merchant = rng.choice(keys, size=n, p=mix)
    merchant_risk = np.clip(np.array([MERCHANTS[m] for m in merchant]) + rng.normal(0, 0.12, n), 0, 1)

    if not fraud:
        amount = rng.lognormal(3.3, 0.9, n)
        hour = rng.normal(14, 4, n).astype(int) % 24
        distance = rng.exponential(8, n)
        velocity = rng.poisson(1, n)
        foreign = rng.random(n) < 0.04
        new_device = rng.random(n) < 0.05
    else:
        # generic fraud: weak signals on everything
        amount = rng.lognormal(4.1, 1.1, n)
        hour = rng.integers(0, 24, n)
        distance = rng.exponential(20, n)
        velocity = rng.poisson(1.6, n)
        foreign = rng.random(n) < 0.08
        new_device = rng.random(n) < 0.10
        # strong, bank-specific signal
        if profile == "cross_border":
            foreign = rng.random(n) < 0.90
            distance = rng.exponential(1500, n)
        elif profile == "night_burst":
            hour = rng.integers(0, 6, n)
            velocity = rng.poisson(8, n)
        elif profile == "account_takeover":
            new_device = rng.random(n) < 0.92
        elif profile == "attack":  # demo attack: night burst + takeover
            hour = rng.integers(1, 5, n)
            velocity = rng.poisson(9, n) + 3
            new_device = np.ones(n, dtype=bool)
            amount = rng.lognormal(6.3, 0.3, n)

    country = np.where(foreign, rng.choice(FOREIGN_COUNTRIES, n), "KZ")
    return pd.DataFrame({
        "amount_usd": amount, "hour": hour, "merchant": merchant,
        "merchant_risk": merchant_risk, "distance_km": distance,
        "velocity_1h": velocity, "foreign": foreign.astype(int),
        "new_device": new_device.astype(int), "country": country,
        "is_fraud": int(fraud),
    })


def featurize(df: pd.DataFrame) -> np.ndarray:
    """Shared feature schema. All banks agree on this transform up front, so
    no data-dependent scaler (which would need raw statistics) is required.
    PRODUCTION: if you need standardisation, compute mean/var with federated
    analytics or secure aggregation instead of pooling data."""
    return np.column_stack([
        np.log1p(df["amount_usd"]) / 6.0,
        (df["hour"] < 6).astype(float),
        df["merchant_risk"],
        np.log1p(df["distance_km"]) / 8.0,
        np.minimum(df["velocity_1h"], 15) / 10.0,
        df["foreign"].astype(float),
        df["new_device"].astype(float),
    ])


@st.cache_data(show_spinner=False)
def load_federation():
    """Create each bank's private dataset + a balanced test set.
    PRODUCTION: this data never exists in one place. Each bank's dataset
    lives in its own data warehouse; only the training client runs there."""
    banks = {}
    for k, cfg in BANKS.items():
        rng = np.random.default_rng(cfg["seed"])
        n_fraud = int(cfg["n"] * cfg["fraud_rate"])
        df = pd.concat([sample_transactions(cfg["n"] - n_fraud, rng),
                        sample_transactions(n_fraud, rng, fraud=True, profile=cfg["profile"])])
        df = df.sample(frac=1, random_state=cfg["seed"])
        banks[k] = (featurize(df), df["is_fraud"].to_numpy())

    # Test set: 50% legit, 50% fraud split evenly across the three patterns.
    # PRODUCTION: each bank evaluates on its own hold-out and reports only metrics.
    rng = np.random.default_rng(99)
    parts = [sample_transactions(1500, rng)] + [
        sample_transactions(500, rng, fraud=True, profile=p)
        for p in ("cross_border", "night_burst", "account_takeover")]
    test = pd.concat(parts)
    return banks, (featurize(test), test["is_fraud"].to_numpy())


# =============================================================================
# 3. FEDERATED LEARNING SIMULATION (FedAvg + DP)
# =============================================================================
LR = 0.01            # local learning rate
LOCAL_BATCH = 400    # transactions per local round
CLIP_NORM = 1.0      # L2 clipping bound for each update (needed for DP)
NOISE_SCALE = 0.25   # σ=1 → noise std = 0.25·CLIP_NORM per coordinate
SIGMA_MAX = 1.0      # noise multiplier at slider = 100


def local_update(w, b, X, y, rng):
    """One round of local training at a bank, starting from the global model.
    PRODUCTION: this function runs *inside the bank* as a Flower / FATE /
    NVFlare / PySyft client. It pulls the global weights, trains on local data
    and returns only the new weights — never X or y."""
    idx = rng.choice(len(X), size=min(LOCAL_BATCH, len(X)), replace=False)
    clf = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=LR, alpha=1e-4,
                        max_iter=1, tol=None, class_weight="balanced",
                        random_state=int(rng.integers(1_000_000)))
    clf.fit(X[idx], y[idx], coef_init=w.reshape(1, -1), intercept_init=np.array([b]))
    return clf.coef_.ravel(), float(clf.intercept_[0])


def privatize(delta, sigma, rng):
    """Differential Privacy: clip the update, then add Gaussian noise.
    PRODUCTION: use DP-SGD per-example clipping (Opacus / TF-Privacy) and a
    privacy accountant (RDP/moments) to track the real ε over all rounds."""
    norm = np.linalg.norm(delta)
    if norm > CLIP_NORM:
        delta = delta * (CLIP_NORM / norm)
    if sigma > 0:
        delta = delta + rng.normal(0, sigma * CLIP_NORM * NOISE_SCALE, size=delta.shape)
    return delta


def federated_round(w, b, banks, sigma, rng):
    """One FedAvg round: local training → DP → (encryption) → weighted average."""
    deltas, sizes = [], []
    for X, y in banks.values():
        w_k, b_k = local_update(w, b, X, y, rng)
        d = privatize(np.append(w_k - w, b_k - b), sigma, rng)
        # PRODUCTION: encrypt `d` here before it leaves the bank — Secure
        # Aggregation masks (Bonawitz et al., 2017) or homomorphic encryption
        # (TenSEAL / Paillier), so the server only ever sees the *sum*.
        deltas.append(d)
        sizes.append(len(X))
    # PRODUCTION: this averaging runs on the aggregator server (Flower
    # `FedAvg` strategy), over decrypted *aggregate* only, over gRPC + mTLS.
    avg = np.average(deltas, axis=0, weights=sizes)
    return w + avg[:-1], b + avg[-1]


def accuracy(w, b, X, y):
    return float((((X @ w + b) > 0).astype(int) == y).mean())


def risk_score(df: pd.DataFrame) -> np.ndarray:
    """Score transactions with the global model (or rules if untrained).
    PRODUCTION: each bank runs inference locally on its own copy of the model."""
    X = featurize(df)
    if st.session_state.global_w is not None:
        z = X @ st.session_state.global_w + st.session_state.global_b
        return 1 / (1 + np.exp(-z))
    # fallback rules before the model is trained
    s = (0.15 + 0.35 * X[:, 1] + 0.25 * X[:, 2] + 0.2 * X[:, 4] + 0.2 * X[:, 5]
         + 0.25 * X[:, 6] + 0.2 * np.maximum(X[:, 0] - 0.7, 0))
    return np.clip(s, 0, 0.99)


# =============================================================================
# 4. PRIVACY MATH (illustrative mapping for the dashboard)
# =============================================================================
def privacy_stats(noise_pct: int):
    x = noise_pct / 100
    sigma = x * SIGMA_MAX
    pct = round(99.9 * x ** 0.35, 1)
    eps = float("inf") if sigma == 0 else 0.9 / sigma   # PRODUCTION: from an RDP accountant
    cost = 7.0 * sigma ** 2                             # empirical, from offline runs
    if noise_pct == 0:
        label = "lvl_none"
    elif pct < 60:
        label = "lvl_weak"
    elif pct < 85:
        label = "lvl_moderate"
    elif pct < 99:
        label = "lvl_strong"
    else:
        label = "lvl_full"
    return sigma, pct, eps, cost, label


# =============================================================================
# 5. SESSION STATE
# =============================================================================
def init_state():
    ss = st.session_state
    ss.setdefault("lang", "EN")
    ss.setdefault("noise", 70)
    ss.setdefault("live", True)
    ss.setdefault("history", [])          # [{round, fed, local}]
    ss.setdefault("global_w", None)
    ss.setdefault("global_b", 0.0)
    ss.setdefault("tx", pd.DataFrame())
    ss.setdefault("scanned", 0)
    ss.setdefault("blocked", 0)
    ss.setdefault("alert", None)
    ss.setdefault("rng", np.random.default_rng(7))


init_state()
BANK_DATA, TEST_SET = load_federation()

# =============================================================================
# 6. STYLING
# =============================================================================
st.markdown("""
<style>
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #13284d 0%, #0B1220 55%) fixed; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.ds-brand { display:flex; align-items:center; gap:14px; }
.ds-logo { width:46px; height:46px; border-radius:12px; display:flex; align-items:center;
  justify-content:center; font-size:24px; background:linear-gradient(135deg,#2563EB,#1E3A8A);
  box-shadow:0 0 24px rgba(59,130,246,.45); }
.ds-name { font-size:28px; font-weight:700; color:#F1F5F9; line-height:1; }
.ds-name span { color:#60A5FA; }
.ds-tag { color:#94A3B8; font-size:14px; margin-top:4px; }
.ds-badge { display:inline-block; margin-top:8px; padding:4px 10px; border-radius:999px;
  font-size:12px; color:#93C5FD; background:rgba(59,130,246,.12); border:1px solid rgba(59,130,246,.35); }
.ds-card { background:#111A2E; border:1px solid #1E2A44; border-radius:14px; padding:16px 18px; }
.ds-card h4 { margin:0 0 6px 0; color:#E2E8F0; font-size:16px; }
.ds-muted { color:#94A3B8; font-size:13px; }
.ds-ok { color:#34D399; font-size:13px; }
div[data-testid="stMetric"] { background:#111A2E; border:1px solid #1E2A44; border-radius:14px; padding:12px 16px; }
div[data-testid="stMetricValue"] { color:#F1F5F9; }
button[data-baseweb="tab"] p { font-size:15px; font-weight:600; }
.ds-alert { border-radius:14px; padding:16px 20px; margin:8px 0 14px 0;
  background:linear-gradient(90deg,rgba(220,38,38,.25),rgba(127,29,29,.15));
  border:1px solid #DC2626; box-shadow:0 0 28px rgba(220,38,38,.25); animation:ds-pulse 1.2s ease-out 2; }
.ds-alert .t { color:#FECACA; font-weight:700; font-size:17px; }
.ds-alert .d { color:#FCA5A5; font-size:13px; margin-top:4px; font-family:ui-monospace,monospace; }
.ds-alert .s { color:#E2E8F0; font-size:13px; margin-top:6px; }
@keyframes ds-pulse { 0%{box-shadow:0 0 0 rgba(220,38,38,.8);} 100%{box-shadow:0 0 36px rgba(220,38,38,.0);} }
.ds-privacy { font-size:22px; font-weight:700; padding:16px 20px; border-radius:14px;
  background:#0F1D38; border:1px solid #1D4ED8; color:#BFDBFE; }
.ds-footer { color:#475569; font-size:12px; text-align:center; margin-top:32px; }
</style>
""", unsafe_allow_html=True)


def flat(s: str) -> str:
    """Strip indentation/blank lines so Markdown doesn't turn HTML into code blocks."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def html(s: str, target=st):
    target.markdown(flat(s), unsafe_allow_html=True)


def network_html(phase: str, status: str) -> str:
    """Animated network diagram: 3 banks ⇄ central aggregator.
    phase: idle | upload | broadcast | synced"""
    xs = {"A": 16, "B": 50, "C": 84}
    moving = phase in ("upload", "broadcast")
    up = phase == "upload"
    line_color = "#3B82F6" if moving else ("#10B981" if phase == "synced" else "#334155")
    agg_glow = "0 0 40px rgba(59,130,246,.65)" if moving else "0 0 18px rgba(59,130,246,.25)"

    keyframes, packets, lines, nodes = [], [], [], []
    for k, x in xs.items():
        lines.append(f'<line x1="{x}" y1="26" x2="50" y2="74" stroke="{line_color}" '
                     f'stroke-width="0.6" stroke-dasharray="1.5 1.5" vector-effect="non-scaling-stroke" '
                     f'style="stroke-width:2px"/>')
        nodes.append(f'''<div style="position:absolute;left:{x}%;top:4%;transform:translateX(-50%);
            width:150px;text-align:center;background:#111A2E;border:1px solid {line_color};
            border-radius:12px;padding:10px 6px;">
            <div style="font-size:20px">🏦</div>
            <div style="color:#E2E8F0;font-weight:700;font-size:14px">{t("bank", x=k)}</div>
            <div style="color:#64748B;font-size:11px">{t("data_stays")}</div></div>''')
        if moving:
            a, b_ = ((x, 30), (50, 66)) if up else ((50, 66), (x, 30))
            color = "#F59E0B" if up else "#10B981"
            name = f"pkt{k}{'u' if up else 'd'}"
            keyframes.append(f"@keyframes {name} {{ 0%{{left:{a[0]}%;top:{a[1]}%;opacity:0}} "
                             f"15%{{opacity:1}} 85%{{opacity:1}} 100%{{left:{b_[0]}%;top:{b_[1]}%;opacity:0}} }}")
            for i, delay in enumerate((0, 0.45)):
                packets.append(f'<div style="position:absolute;transform:translate(-50%,-50%);'
                               f'animation:{name} 0.9s linear {delay}s infinite;font-size:12px;'
                               f'padding:2px 7px;border-radius:999px;background:{color};color:#0B1220;'
                               f'font-weight:700;white-space:nowrap;">🔒 Δw</div>')

    return f'''<style>{" ".join(keyframes)}</style>
    <div style="position:relative;height:310px;background:#0D1629;border:1px solid #1E2A44;border-radius:16px;overflow:hidden;">
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%">{"".join(lines)}</svg>
    {"".join(nodes)}
    <div style="position:absolute;left:50%;top:66%;transform:translateX(-50%);width:250px;text-align:center;
    background:linear-gradient(135deg,#1E3A8A,#1D4ED8);border-radius:14px;padding:10px;box-shadow:{agg_glow};">
    <div style="font-size:20px">🛡️</div>
    <div style="color:#fff;font-weight:700;font-size:14px">{t("aggregator")}</div>
    <div style="color:#BFDBFE;font-size:11px">{t("aggregator_sub")}</div></div>
    {"".join(packets)}
    <div style="position:absolute;left:12px;bottom:8px;color:#94A3B8;font-size:12px;">{status}</div>
    </div>'''.replace("\n", " ").strip()


def accuracy_chart(history):
    df = pd.DataFrame(history)
    long = pd.concat([
        pd.DataFrame({"round": df["round"], "acc": df["fed"], "series": t("series_fed")}),
        pd.DataFrame({"round": df["round"], "acc": df["local"], "series": t("series_local")}),
    ])
    return (alt.Chart(long).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("round:Q", title=t("axis_round"), axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("acc:Q", title=t("axis_acc"), scale=alt.Scale(domain=[0.4, 1.0]),
                axis=alt.Axis(format="%")),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=[t("series_fed"), t("series_local")],
                                        range=["#3B82F6", "#64748B"]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("round:Q", title=t("axis_round")), "series:N",
                 alt.Tooltip("acc:Q", title=t("axis_acc"), format=".1%")],
    ).properties(height=320, title=t("chart_title"))
     .configure(background="transparent")
     .configure_title(color="#E2E8F0", fontSize=14, anchor="start")
     .configure_axis(labelColor="#94A3B8", titleColor="#94A3B8", gridColor="#1E2A44", domainColor="#334155")
     .configure_legend(labelColor="#CBD5E1"))


# =============================================================================
# 7. HEADER + LANGUAGE SWITCHER
# =============================================================================
h_left, h_right = st.columns([5, 1.3])
with h_right:
    st.radio("🌐", ["EN", "RU", "KZ"], key="lang", horizontal=True)
with h_left:
    html(f'''<div class="ds-brand"><div class="ds-logo">🛡️</div><div>
    <div class="ds-name">Derek<span>Safe</span></div>
    <div class="ds-tag">{t("tagline")}</div>
    <div class="ds-badge">🔐 {t("badge")}</div></div></div>''')
st.write("")

tab_net, tab_mon, tab_priv = st.tabs([t("tab_network"), t("tab_monitor"), t("tab_privacy")])

# =============================================================================
# TAB 1 — NETWORK CONTROL
# =============================================================================
with tab_net:
    st.subheader(t("net_title"))
    st.caption(t("net_desc"))

    cols = st.columns(3)
    for col, (k, cfg) in zip(cols, BANKS.items()):
        with col:
            html(f'''<div class="ds-card"><h4>🏦 {t("bank", x=k)}</h4>
            <div class="ds-muted">{t("bank_records", n=cfg["n"])}</div>
            <div class="ds-muted">{t("bank_pattern", p=t("p_" + cfg["profile"]))}</div>
            <div class="ds-ok">{t("data_stays")}</div></div>''')
    st.write("")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        n_rounds = st.slider(t("rounds_label"), 5, 30, 15)
    with c2:
        st.write("")
        start = st.button(t("start_training"), type="primary", width="stretch")
    with c3:
        st.write("")
        if st.button(t("reset_model"), width="stretch"):
            st.session_state.history = []
            st.session_state.global_w = None
            st.session_state.global_b = 0.0
            st.rerun()

    anim = st.empty()
    metrics_slot = st.empty()
    chart_slot = st.empty()

    def render_metrics():
        hist = st.session_state.history
        with metrics_slot.container():
            m1, m2, m3, m4 = st.columns(4)
            fed = hist[-1]["fed"] if hist else 0.5
            loc = hist[-1]["local"] if hist else 0.5
            m1.metric(t("m_accuracy"), f"{fed:.1%}",
                      f"{(fed - hist[0]['fed']) * 100:+.1f} pp" if len(hist) > 1 else None)
            m2.metric(t("m_uplift"), f"{(fed - loc) * 100:+.1f} pp")
            m3.metric(t("m_rounds"), max(len(hist) - 1, 0))
            m4.metric(t("m_shared"), "0")

    def render_chart():
        if len(st.session_state.history) > 1:
            chart_slot.altair_chart(accuracy_chart(st.session_state.history), width="stretch")
        else:
            chart_slot.info(t("chart_empty"))

    if start:
        sigma = privacy_stats(st.session_state.noise)[0]
        rng = np.random.default_rng(int(time.time()))
        Xt, yt = TEST_SET
        n_feat = len(FEATURES)
        w, b = np.zeros(n_feat), 0.0            # PRODUCTION: server initialises & broadcasts w0
        w_loc, b_loc = np.zeros(n_feat), 0.0    # baseline: Bank A training alone
        X_a, y_a = BANK_DATA["A"]
        st.session_state.history = [{"round": 0, "fed": accuracy(w, b, Xt, yt),
                                     "local": accuracy(w_loc, b_loc, Xt, yt)}]
        for r in range(1, n_rounds + 1):
            anim.markdown(network_html("upload", t("phase_upload", r=r, n=n_rounds)), unsafe_allow_html=True)
            w, b = federated_round(w, b, BANK_DATA, sigma, rng)
            # Baseline gets the same update clipping (but no noise) for a fair comparison.
            w_new, b_new = local_update(w_loc, b_loc, X_a, y_a, rng)
            d_loc = privatize(np.append(w_new - w_loc, b_new - b_loc), 0.0, rng)
            w_loc, b_loc = w_loc + d_loc[:-1], b_loc + d_loc[-1]
            time.sleep(0.45)
            anim.markdown(network_html("broadcast", t("phase_broadcast", r=r, n=n_rounds)), unsafe_allow_html=True)
            st.session_state.history.append({"round": r, "fed": accuracy(w, b, Xt, yt),
                                             "local": accuracy(w_loc, b_loc, Xt, yt)})
            st.session_state.global_w, st.session_state.global_b = w, b
            render_metrics()
            render_chart()
            time.sleep(0.35)
        st.toast(t("training_done", acc=st.session_state.history[-1]["fed"]), icon="✅")

    phase = "synced" if st.session_state.global_w is not None else "idle"
    anim.markdown(network_html(phase, t("net_synced") if phase == "synced" else t("net_idle")),
                  unsafe_allow_html=True)
    render_metrics()
    render_chart()

# =============================================================================
# TAB 2 — TRANSACTION MONITOR
# =============================================================================
MAX_ROWS = 25


def mask_client(rng) -> str:
    """Pseudonymous, masked client ID. PRODUCTION: salted HMAC per bank."""
    h = hashlib.sha256(rng.bytes(8)).hexdigest()
    return f"KZ••{h[:2].upper()}•••{h[-4:].upper()}"


def new_transactions(n, rng, attack=False) -> pd.DataFrame:
    """Draw new 'live' transactions and score them with the global model."""
    if attack:
        df = sample_transactions(1, rng, fraud=True, profile="attack")
        df["bank"] = "A"
    else:
        rows = []
        for _ in range(n):
            bank = rng.choice(list(BANKS))
            organic_fraud = rng.random() < 0.03
            rows.append(sample_transactions(1, rng, fraud=organic_fraud,
                                            profile=BANKS[bank]["profile"] if organic_fraud else None)
                        .assign(bank=bank))
        df = pd.concat(rows)
    df["risk"] = risk_score(df)
    df["status"] = np.select([df["risk"] >= 0.85, df["risk"] >= 0.6], ["blocked", "review"], "ok")
    if attack:
        df["status"] = "blocked"
    df["time"] = datetime.now().strftime("%H:%M:%S")
    df["tx_id"] = [f"TX-{rng.integers(16**6):06X}" for _ in range(len(df))]
    df["client"] = [mask_client(rng) for _ in range(len(df))]
    df["attack"] = attack
    return df


def push_tx(df):
    ss = st.session_state
    ss.tx = pd.concat([df, ss.tx]).head(MAX_ROWS).reset_index(drop=True)
    ss.scanned += len(df)
    ss.blocked += int((df["status"] == "blocked").sum())


if st.session_state.tx.empty:
    push_tx(new_transactions(12, st.session_state.rng))


def styled_table(df: pd.DataFrame):
    status_txt = {"ok": t("st_ok"), "review": t("st_review"), "blocked": t("st_blocked")}
    view = pd.DataFrame({
        t("col_time"): df["time"],
        t("col_id"): df["tx_id"],
        t("col_bank"): df["bank"].map(lambda k: t("bank", x=k)),
        t("col_client"): df["client"],
        t("col_amount"): (df["amount_usd"] * KZT_PER_USD).map(lambda v: f"₸ {v:,.0f}"),
        t("col_merchant"): df["merchant"].map(lambda m: t("mer_" + m)),
        t("col_country"): df["country"],
        t("col_risk"): df["risk"].map(lambda v: f"{v:.0%}"),
        t("col_status"): df["status"].map(status_txt),
    })
    statuses = df["status"].tolist()
    attacks = df["attack"].tolist()

    def row_style(row):
        s, a = statuses[row.name], attacks[row.name]
        if s == "blocked":
            css = "background-color:#7F1D1D;color:#FEE2E2;font-weight:700" if a else \
                  "background-color:#4C1D1D;color:#FECACA"
        elif s == "review":
            css = "background-color:#3B2F0B;color:#FDE68A"
        else:
            css = "color:#CBD5E1"
        return [css] * len(row)

    return view.style.apply(row_style, axis=1)


with tab_mon:
    st.subheader(t("mon_title"))
    st.caption(t("mon_desc"))
    if st.session_state.global_w is None:
        st.warning(t("untrained_warn"), icon="ℹ️")

    b1, b2, b3 = st.columns([1.4, 1, 1])
    with b1:
        attack = st.button(t("simulate_attack"), type="primary", width="stretch")
    with b2:
        if st.button(t("reset_feed"), width="stretch"):
            st.session_state.tx = pd.DataFrame()
            st.session_state.scanned = st.session_state.blocked = 0
            st.session_state.alert = None
            push_tx(new_transactions(12, st.session_state.rng))
    with b3:
        st.toggle(t("live_feed"), key="live")

    if attack:
        df_attack = new_transactions(1, st.session_state.rng, attack=True)
        push_tx(df_attack)
        row = df_attack.iloc[0]
        st.session_state.alert = dict(id=row["tx_id"], amount=row["amount_usd"] * KZT_PER_USD,
                                      merchant=row["merchant"], score=float(row["risk"]))
        st.session_state.skip_tick = True   # keep the red attack row at the top this run

    @st.fragment(run_every=2.5 if st.session_state.live else None)
    def live_monitor():
        ss = st.session_state
        if ss.live and not ss.get("skip_tick"):
            push_tx(new_transactions(int(ss.rng.integers(1, 3)), ss.rng))
        ss.skip_tick = False

        if ss.alert:
            a = ss.alert
            detail = t("alert_detail", id=a["id"], amount="₸ {:,.0f}".format(a["amount"]),
                       merchant=t("mer_" + a["merchant"]), score=a["score"])
            html(f'''<div class="ds-alert">
            <div class="t">⛔ {t("alert_title")}</div>
            <div class="d">{detail}</div>
            <div class="s">{t("attack_story")}</div></div>''')

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t("m_scanned"), f"{ss.scanned:,}")
        m2.metric(t("m_blocked"), ss.blocked)
        m3.metric(t("m_exposed"), "0")
        m4.metric(t("m_latency"), f"{ss.rng.integers(9, 15)} ms")
        st.dataframe(styled_table(ss.tx), hide_index=True, width="stretch", height=560)

    live_monitor()

# =============================================================================
# TAB 3 — PRIVACY SHIELD
# =============================================================================
with tab_priv:
    st.subheader(t("priv_title"))
    st.caption(t("priv_desc"))

    left, right = st.columns([1.5, 1])
    with left:
        st.slider(t("noise_slider"), 0, 100, key="noise", format="%d%%")
        sigma, pct, eps, cost, label = privacy_stats(st.session_state.noise)
        color = {"lvl_none": "#F87171", "lvl_weak": "#FB923C", "lvl_moderate": "#FACC15",
                 "lvl_strong": "#60A5FA", "lvl_full": "#34D399"}[label]
        level_text = t("privacy_level", p="{:.1f}".format(pct), label=t(label))
        html(f'''<div class="ds-privacy" style="border-color:{color}">
        🛡️ <span style="color:{color}">{level_text}</span></div>''')
        st.progress(min(pct / 100, 1.0))

        m1, m2, m3 = st.columns(3)
        m1.metric(t("m_epsilon"), "∞" if eps == float("inf") else f"{eps:.2f}")
        m2.metric(t("m_sigma"), f"{sigma:.2f}")
        m3.metric(t("m_cost"), f"−{cost:.1f} pp")
        st.caption("ℹ️ " + t("priv_applied"))

    with right:
        dp_color, dp_icon = ("#34D399", "✔") if sigma > 0 else ("#F87171", "✖")
        html(f'''<div class="ds-card"><h4>🔐 {t("guarantees_title")}</h4>
        <div class="ds-ok">✔ {t("g1")}</div>
        <div class="ds-ok">✔ {t("g2")}</div>
        <div class="ds-ok" style="color:{dp_color}">{dp_icon} {t("g3")}</div>
        <div class="ds-ok">✔ {t("g4")}</div></div>''')

    # Demo: what one bank's update looks like before vs after DP noise.
    demo_rng = np.random.default_rng(5)  # fixed seed → only the noise level changes the picture
    X_a, y_a = BANK_DATA["A"]
    w0 = st.session_state.global_w if st.session_state.global_w is not None else np.zeros(len(FEATURES))
    b0 = st.session_state.global_b
    w1, b1_ = local_update(w0, b0, X_a, y_a, demo_rng)
    raw = np.append(w1 - w0, b1_ - b0)
    raw = raw * min(1.0, CLIP_NORM / max(np.linalg.norm(raw), 1e-12))
    noised = raw + demo_rng.normal(0, sigma * CLIP_NORM * NOISE_SCALE, raw.shape) if sigma > 0 else raw
    feat_names = [t("f_" + f) for f in FEATURES] + [t("f_bias")]
    upd = pd.concat([
        pd.DataFrame({"feature": feat_names, "value": raw, "series": t("series_raw")}),
        pd.DataFrame({"feature": feat_names, "value": noised, "series": t("series_noised")}),
    ])
    chart = (alt.Chart(upd).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("feature:N", title=t("axis_feature"), sort=feat_names, axis=alt.Axis(labelAngle=-30)),
        xOffset="series:N",
        y=alt.Y("value:Q", title=t("axis_value")),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=[t("series_raw"), t("series_noised")],
                                        range=["#64748B", "#3B82F6"]),
                        legend=alt.Legend(orient="bottom")),
    ).properties(height=320, title=t("update_chart_title"))
     .configure(background="transparent")
     .configure_title(color="#E2E8F0", fontSize=14, anchor="start")
     .configure_axis(labelColor="#94A3B8", titleColor="#94A3B8", gridColor="#1E2A44", domainColor="#334155")
     .configure_legend(labelColor="#CBD5E1"))
    st.altair_chart(chart, width="stretch")

html(f'<div class="ds-footer">{t("footer")}</div>')
