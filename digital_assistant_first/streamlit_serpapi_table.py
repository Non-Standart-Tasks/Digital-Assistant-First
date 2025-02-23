from src.utils.check_serp_response import check_api_keys
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

# Заголовок приложения
st.title("Проверка статуса API-ключей")
st.write("Это приложение позволяет просматривать и обновлять статус ваших API-ключей.")

# Загрузка начального DataFrame
df = check_api_keys('api_keys_status.csv')

# Отображение текущей таблицы
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_default_column(min_column_width=100)
gb.configure_grid_options(domLayout='normal')
grid_options = gb.build()

AgGrid(
    df,
    gridOptions=grid_options,
    fit_columns_on_grid_load=True,
    columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
    height=450,
    width='100%',
    key="initial_grid"
)

# Раздел для добавления новых записей
st.subheader("Добавить новую запись")