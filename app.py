
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="PrévisionFacile", layout="wide")
st.title("📈 PrévisionFacile – Prédisez vos ventes simplement")

st.markdown("Chargez un fichier CSV contenant deux colonnes : `date` et `sales`.")

uploaded_file = st.file_uploader("📤 Charger votre fichier CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Mapping intelligent des colonnes
        column_map = {
            'Date': 'date',
            'DATE': 'date',
            'Ventes': 'sales',
            'ventes': 'sales',
            'total': 'sales',
            'Total': 'sales'
        }

        df.columns = [column_map.get(col, col) for col in df.columns]

        expected_columns = {'date', 'sales'}
        if not expected_columns.issubset(df.columns):
            st.error("❌ Le fichier doit contenir deux colonnes nommées `date` et `sales` (ou noms reconnus comme équivalents).")
            st.markdown("Exemple de format valide :")
            st.dataframe(pd.DataFrame({ "date": ["2023-01-01", "2023-01-02"], "sales": [120, 135] }))
        else:
            # Conversion de la colonne date
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                st.error(f"❌ Erreur dans le format des dates : {e}")
                st.stop()

            st.success("✅ Fichier valide. Données chargées avec succès.")

            st.subheader("Aperçu des données")
            st.line_chart(df.set_index("date")["sales"])

            df_prophet = df.rename(columns={"date": "ds", "sales": "y"})
            model = Prophet()
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            st.subheader("Prévision sur les 30 prochains jours")
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            forecast_output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
            csv = forecast_output.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Télécharger les prévisions", data=csv, file_name="forecast_30_days.csv")

    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
else:
    st.info("Veuillez charger un fichier pour commencer.")
