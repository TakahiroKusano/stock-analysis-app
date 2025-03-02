import os
import pandas as pd
import yfinance as yf
import concurrent.futures
import talib
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # フォルダがなければ作成

@app.post("/upload/")
async def upload_csv(
    file: UploadFile = File(...),
    # 指標を適用するかどうかの選択（デフォルト: True）
    use_sma: bool = Form(True),
    use_macd: bool = Form(True),
    use_rsi: bool = Form(True),
    use_bb: bool = Form(True),
    use_stoch: bool = Form(True),
    use_adx: bool = Form(True),
    use_candlestick: bool = Form(True),
    # 各指標のパラメータ（デフォルト値）
    sma_short: int = Form(20),
    sma_long: int = Form(50),
    ema_fast: int = Form(12),
    ema_slow: int = Form(26),
    macd_signal: int = Form(9),
    rsi_period: int = Form(14),
    rsi_threshold: int = Form(50),
    bb_period: int = Form(20),
    stoch_k: int = Form(14),
    stoch_d: int = Form(3),
    adx_period: int = Form(14),
):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(file_path, encoding="utf-8")
    df["ティッカー"] = df["コード"].astype(str) + ".T"
    ticker_list = df["ティッカー"].tolist()

    print("📊 テクニカル分析用の株価データを一括取得中...")
    try:
        tech_data = yf.download(ticker_list, period="6mo", auto_adjust=False, progress=False)
        if tech_data.empty:
            return {"error": "取得した株価データが空です。銘柄コードを確認してください"}
    except Exception as e:
        return {"error": f"株価データ取得中にエラー発生: {e}"}

    def check_technical(ticker):
        try:
            data = tech_data["Close"][ticker].dropna()
            if len(data) < max(sma_long, ema_slow, rsi_period, bb_period, adx_period):
                return None
            
            # 指標を必要に応じて計算
            indicators = {}

            if use_sma:
                indicators["sma_short_vals"] = data.rolling(window=sma_short).mean()
                indicators["sma_long_vals"] = data.rolling(window=sma_long).mean()
            
            if use_macd:
                indicators["ema_fast_vals"] = data.ewm(span=ema_fast, adjust=False).mean()
                indicators["ema_slow_vals"] = data.ewm(span=ema_slow, adjust=False).mean()
                indicators["macd"] = indicators["ema_fast_vals"] - indicators["ema_slow_vals"]
                indicators["signal"] = indicators["macd"].ewm(span=macd_signal, adjust=False).mean()

            if use_rsi:
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
                rs = gain / (loss + 1e-10)
                indicators["rsi"] = 100 - (100 / (1 + rs))
            
            if use_bb:
                indicators["upper_band"], _, indicators["lower_band"] = talib.BBANDS(data, timeperiod=bb_period)
            
            if use_stoch:
                indicators["slowk"], indicators["slowd"] = talib.STOCH(data, data, data, fastk_period=stoch_k, slowk_period=stoch_d, slowd_period=stoch_d)

            if use_adx:
                indicators["adx"] = talib.ADX(data, data, data, timeperiod=adx_period)
            
            if use_candlestick:
                indicators["bullish_engulfing"] = talib.CDLENGULFING(data, data, data, data)
            
            # シグナルの判定
            conditions = []

            if use_sma and use_macd:
                conditions.append(indicators["sma_short_vals"].iloc[-1] > indicators["sma_long_vals"].iloc[-1])
                conditions.append(indicators["macd"].iloc[-1] > indicators["signal"].iloc[-1])
            
            if use_rsi:
                conditions.append(indicators["rsi"].iloc[-1] > rsi_threshold)
            
            if use_bb:
                conditions.append(data.iloc[-1] > indicators["upper_band"].iloc[-1])
            
            if use_stoch:
                conditions.append(indicators["slowk"].iloc[-1] > indicators["slowd"].iloc[-1])
            
            if use_adx:
                conditions.append(indicators["adx"].iloc[-1] > 25)
            
            if use_candlestick:
                conditions.append(indicators["bullish_engulfing"].iloc[-1] == 1)

            # すべての条件が True なら上昇トレンド銘柄とする
            if all(conditions):
                return ticker
        except Exception:
            return None
        return None

    print("📊 テクニカル分析を実行中...")
    uptrend_stocks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(check_technical, ticker_list)
        uptrend_stocks = [ticker for ticker in results if ticker]

    print(f"✅ テクニカル分析の結果、{len(uptrend_stocks)} 銘柄が上昇トレンド")

    output_file = os.path.join(UPLOAD_FOLDER, "technical_analysis_result.csv")
    filtered_df = df[df["ティッカー"].isin(uptrend_stocks)]
    filtered_df.to_csv(output_file, index=False, encoding="utf-8")

    return {
        "message": "テクニカル分析完了",
        "csv_file": output_file,
        "uptrend_stocks": uptrend_stocks
    }

@app.get("/download/")
async def download_csv():
    csv_path = os.path.join(UPLOAD_FOLDER, "technical_analysis_result.csv")
    if not os.path.exists(csv_path):
        return {"error": "CSV ファイルが見つかりません"}
    return FileResponse(csv_path, media_type="text/csv", filename="technical_analysis_result.csv")





