@st.cache_data(show_spinner=False)
def load_predictions(path: Path) -> pd.DataFrame:
    # Проверяем, есть ли локальный файл Parquet
    parquet_path = path.with_suffix('.parquet')
    
    # Пробуем загрузить с Google Drive
    try:
        file_id = st.secrets.get("GDRIVE_URL", "")
        if file_id:
            import requests, io
            # Если в secrets сохранен только ID
            if "http" not in file_id:
                url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
            else:
                url = file_id
            
            with st.spinner("Загрузка данных с Google Drive..."):
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                # Читаем Parquet из байтов
                df = pd.read_parquet(io.BytesIO(r.content))
                return df
    except Exception as e:
        st.warning(f"Не удалось загрузить с Google Drive: {e}")
    
    # Пробуем локальный Parquet
    if parquet_path.exists():
        with st.spinner("Загрузка локального Parquet..."):
            df = pd.read_parquet(parquet_path)
            return df
    
    # Пробуем локальный CSV (медленно)
    if path.exists():
        with st.spinner("Загрузка CSV (может занять время)..."):
            df = pd.read_csv(path, low_memory=False)
            # Конвертируем колонки
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df["fire_probability"] = pd.to_numeric(df["fire_probability"], errors="coerce")
            df["fire"] = df.get("fire", 0).fillna(0).astype(int)
            df = df.dropna(subset=["date", "lat", "lon", "fire_probability"])
            df["date_only"] = df["date"].dt.date
            return df
    
    st.error("Файл данных не найден")
    return pd.DataFrame()
