import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import unicodedata


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Reporte de Camión",
    page_icon="🚛",
    layout="wide"
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .titulo-principal {
            font-size: 2.2rem;
            font-weight: 700;
            color: #222222;
            margin-bottom: 0;
        }

        .subtitulo {
            font-size: 1rem;
            color: #6b7280;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
        }

        .card-info {
            background-color: #f7f9fb;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px 18px;
            height: 100%;
        }

        .card-label {
            color: #6b7280;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 5px;
        }

        .card-value {
            color: #111827;
            font-size: 1.15rem;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background-color: #f7f9fb;
            border: 1px solid #e5e7eb;
            padding: 15px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCIONES
# ============================================================

def normalizar_texto(texto):
    """
    Convierte encabezados a una forma comparable.
    Ejemplo:
    'Latitud GPS' -> 'latitud gps'
    """
    texto = str(texto).strip().lower()

    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )

    return texto


def buscar_columna(df, candidatos):
    """
    Busca automáticamente una columna según una lista
    de posibles nombres.
    """

    columnas_normalizadas = {
        normalizar_texto(col): col
        for col in df.columns
    }

    # Primero intenta coincidencia exacta
    for candidato in candidatos:
        candidato_norm = normalizar_texto(candidato)

        if candidato_norm in columnas_normalizadas:
            return columnas_normalizadas[candidato_norm]

    # Después intenta coincidencia parcial
    for candidato in candidatos:
        candidato_norm = normalizar_texto(candidato)

        for col_norm, col_original in columnas_normalizadas.items():
            if candidato_norm in col_norm:
                return col_original

    return None


def leer_csv_seguro(archivo):
    """
    Intenta leer CSV con distintas codificaciones
    y separadores.
    """

    codificaciones = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
        "cp1252"
    ]

    ultimo_error = None

    for encoding in codificaciones:
        try:
            archivo.seek(0)

            df = pd.read_csv(
                archivo,
                encoding=encoding,
                sep=None,
                engine="python"
            )

            return df

        except Exception as e:
            ultimo_error = e

    raise ultimo_error


def convertir_numero(serie):
    """
    Convierte coordenadas a número contemplando
    coma o punto decimal.
    """

    return pd.to_numeric(
        serie.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )


def construir_fecha_hora(df):
    """
    Busca una columna de fecha/hora.
    Si fecha y hora vienen separadas las combina.
    """

    columna_datetime = buscar_columna(
        df,
        [
            "fecha hora",
            "fecha/hora",
            "datetime",
            "timestamp",
            "date time",
            "fecha y hora",
            "hora fecha"
        ]
    )

    if columna_datetime:
        return pd.to_datetime(
            df[columna_datetime],
            errors="coerce",
            dayfirst=True
        )

    columna_fecha = buscar_columna(
        df,
        [
            "fecha",
            "date"
        ]
    )

    columna_hora = buscar_columna(
        df,
        [
            "hora",
            "time",
            "hora de inicio"
        ]
    )

    if columna_fecha and columna_hora:
        valores = (
            df[columna_fecha].astype(str).str.strip()
            + " "
            + df[columna_hora].astype(str).str.strip()
        )

        return pd.to_datetime(
            valores,
            errors="coerce",
            dayfirst=True
        )

    if columna_fecha:
        return pd.to_datetime(
            df[columna_fecha],
            errors="coerce",
            dayfirst=True
        )

    return pd.Series(pd.NaT, index=df.index)


def calcular_distancia_km(lat1, lon1, lat2, lon2):
    """
    Fórmula de Haversine.
    Calcula distancia aproximada entre dos coordenadas.
    """

    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0088

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c


def calcular_distancia_recorrido(df):
    """
    Suma la distancia entre puntos GPS consecutivos.
    """

    if len(df) < 2:
        return 0.0

    distancia_total = 0

    for i in range(1, len(df)):

        lat1 = df.iloc[i - 1]["_latitud"]
        lon1 = df.iloc[i - 1]["_longitud"]

        lat2 = df.iloc[i]["_latitud"]
        lon2 = df.iloc[i]["_longitud"]

        distancia = calcular_distancia_km(
            lat1,
            lon1,
            lat2,
            lon2
        )

        # Filtro de saltos GPS absurdos.
        # No une dos puntos si el salto supera 300 km.
        if distancia <= 300:
            distancia_total += distancia

    return distancia_total


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🚛 Reporte de Camión")

    st.markdown("---")

    st.subheader("📂 Carga de archivos")

    archivo_ubicaciones = st.file_uploader(
        "Historial de ubicaciones",
        type=["csv"],
        help="CSV que contiene las posiciones GPS del camión."
    )

    archivo_motor = st.file_uploader(
        "Datos del motor",
        type=["csv"],
        help="CSV que contiene los parámetros de funcionamiento del camión."
    )

    st.markdown("---")


# ============================================================
# ENCABEZADO
# ============================================================

st.markdown(
    '<div class="titulo-principal">🚛 Reporte de operación del camión</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitulo">'
    'Análisis de recorrido y funcionamiento de la unidad'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ESPERA DE DATOS
# ============================================================

if archivo_ubicaciones is None:

    st.info(
        "👈 Cargá en el sidebar el CSV de historial de ubicaciones "
        "para comenzar el análisis."
    )

    st.stop()


# ============================================================
# LEER UBICACIONES
# ============================================================

try:

    df_ubicaciones = leer_csv_seguro(
        archivo_ubicaciones
    )

except Exception as e:

    st.error(
        f"No se pudo leer el archivo de ubicaciones: {e}"
    )

    st.stop()


# ============================================================
# IDENTIFICAR COLUMNAS
# ============================================================

col_latitud = buscar_columna(
    df_ubicaciones,
    [
        "latitud",
        "latitude",
        "lat",
        "gps latitude",
        "posición latitud",
        "position latitude"
    ]
)

col_longitud = buscar_columna(
    df_ubicaciones,
    [
        "longitud",
        "longitude",
        "lon",
        "lng",
        "long",
        "gps longitude",
        "posición longitud",
        "position longitude"
    ]
)

col_serie = buscar_columna(
    df_ubicaciones,
    [
        "número de serie de la máquina",
        "numero de serie de la maquina",
        "número de serie",
        "numero de serie",
        "serial number",
        "serie máquina",
        "machine serial"
    ]
)


# ============================================================
# VALIDACIÓN GPS
# ============================================================

if col_latitud is None or col_longitud is None:

    st.error(
        "❌ No pude identificar automáticamente las columnas "
        "de latitud y longitud."
    )

    st.write(
        "**Columnas encontradas en el CSV:**"
    )

    st.write(
        list(df_ubicaciones.columns)
    )

    st.stop()


# ============================================================
# LIMPIEZA UBICACIONES
# ============================================================

df_ubicaciones["_latitud"] = convertir_numero(
    df_ubicaciones[col_latitud]
)

df_ubicaciones["_longitud"] = convertir_numero(
    df_ubicaciones[col_longitud]
)

df_ubicaciones["_fecha_hora"] = construir_fecha_hora(
    df_ubicaciones
)


# Eliminar filas sin coordenadas
df_ubicaciones = df_ubicaciones.dropna(
    subset=[
        "_latitud",
        "_longitud"
    ]
).copy()


# Validar rango geográfico
df_ubicaciones = df_ubicaciones[
    df_ubicaciones["_latitud"].between(-90, 90)
    & df_ubicaciones["_longitud"].between(-180, 180)
].copy()


if df_ubicaciones.empty:

    st.error(
        "El archivo no contiene coordenadas GPS válidas."
    )

    st.stop()


# Orden cronológico
if df_ubicaciones["_fecha_hora"].notna().any():

    df_ubicaciones = (
        df_ubicaciones
        .sort_values("_fecha_hora")
        .reset_index(drop=True)
    )

else:

    df_ubicaciones = (
        df_ubicaciones
        .reset_index(drop=True)
    )


# ============================================================
# LEER DATOS MOTOR
# ============================================================

df_motor = None

if archivo_motor is not None:

    try:

        df_motor = leer_csv_seguro(
            archivo_motor
        )

    except Exception as e:

        st.warning(
            f"El archivo de motor no pudo leerse: {e}"
        )


# ============================================================
# PERÍODO DE ANÁLISIS
# ============================================================

fecha_inicio = None
fecha_fin = None

if df_ubicaciones["_fecha_hora"].notna().any():

    fecha_inicio = df_ubicaciones["_fecha_hora"].min()
    fecha_fin = df_ubicaciones["_fecha_hora"].max()


# Si el motor amplía el período, también lo analizamos
if df_motor is not None:

    df_motor["_fecha_hora"] = construir_fecha_hora(
        df_motor
    )

    if df_motor["_fecha_hora"].notna().any():

        motor_inicio = df_motor["_fecha_hora"].min()
        motor_fin = df_motor["_fecha_hora"].max()

        if fecha_inicio is None:
            fecha_inicio = motor_inicio
        else:
            fecha_inicio = min(
                fecha_inicio,
                motor_inicio
            )

        if fecha_fin is None:
            fecha_fin = motor_fin
        else:
            fecha_fin = max(
                fecha_fin,
                motor_fin
            )


# ============================================================
# MOSTRAR PERÍODO EN SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("📅 Período de análisis")

    if fecha_inicio is not None:

        st.write(
            "**Desde:**",
            fecha_inicio.strftime(
                "%d/%m/%Y %H:%M"
            )
        )

        st.write(
            "**Hasta:**",
            fecha_fin.strftime(
                "%d/%m/%Y %H:%M"
            )
        )

    else:

        st.warning(
            "No fue posible identificar las fechas."
        )


# ============================================================
# IDENTIFICAR UNIDAD
# ============================================================

numero_serie = "No identificado"

if col_serie is not None:

    series_validas = (
        df_ubicaciones[col_serie]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(series_validas) > 0:
        numero_serie = series_validas[0]


# ============================================================
# DISTANCIA GPS
# ============================================================

distancia_total = calcular_distancia_recorrido(
    df_ubicaciones
)


# ============================================================
# TARJETAS PRINCIPALES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🚛 Unidad",
        numero_serie
    )


with col2:

    st.metric(
        "📍 Puntos GPS",
        f"{len(df_ubicaciones):,}".replace(",", ".")
    )


with col3:

    st.metric(
        "🛣️ Distancia GPS",
        f"{distancia_total:,.1f} km"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


with col4:

    if fecha_inicio is not None:

        dias = (
            fecha_fin - fecha_inicio
        ).total_seconds() / 86400

        st.metric(
            "📅 Período",
            f"{dias:.1f} días".replace(".", ",")
        )

    else:

        st.metric(
            "📅 Período",
            "-"
        )


st.markdown("---")


# ============================================================
# MAPA
# ============================================================

st.subheader("📍 Recorrido de la unidad")

st.caption(
    "Los puntos del historial GPS se unen siguiendo "
    "su orden cronológico."
)


# Centro del mapa
centro_lat = (
    df_ubicaciones["_latitud"]
    .mean()
)

centro_lon = (
    df_ubicaciones["_longitud"]
    .mean()
)


mapa = folium.Map(
    location=[
        centro_lat,
        centro_lon
    ],
    zoom_start=7,
    control_scale=True,
    tiles="OpenStreetMap"
)


# ============================================================
# RECORRIDO
# ============================================================

coordenadas = (
    df_ubicaciones[
        [
            "_latitud",
            "_longitud"
        ]
    ]
    .values
    .tolist()
)


folium.PolyLine(
    coordenadas,
    color="#2563EB",
    weight=4,
    opacity=0.85,
    tooltip="Recorrido del camión"
).add_to(mapa)


# ============================================================
# INICIO
# ============================================================

inicio = df_ubicaciones.iloc[0]

texto_inicio = "<b>Inicio del recorrido</b>"

if pd.notna(inicio["_fecha_hora"]):

    texto_inicio += (
        "<br>"
        + inicio["_fecha_hora"]
        .strftime("%d/%m/%Y %H:%M")
    )


folium.Marker(
    location=[
        inicio["_latitud"],
        inicio["_longitud"]
    ],
    popup=folium.Popup(
        texto_inicio,
        max_width=250
    ),
    tooltip="Inicio",
    icon=folium.Icon(
        color="green",
        icon="play"
    )
).add_to(mapa)


# ============================================================
# FIN
# ============================================================

fin = df_ubicaciones.iloc[-1]

texto_fin = "<b>Fin del recorrido</b>"

if pd.notna(fin["_fecha_hora"]):

    texto_fin += (
        "<br>"
        + fin["_fecha_hora"]
        .strftime("%d/%m/%Y %H:%M")
    )


folium.Marker(
    location=[
        fin["_latitud"],
        fin["_longitud"]
    ],
    popup=folium.Popup(
        texto_fin,
        max_width=250
    ),
    tooltip="Fin",
    icon=folium.Icon(
        color="red",
        icon="stop"
    )
).add_to(mapa)


# ============================================================
# AJUSTAR MAPA AL RECORRIDO
# ============================================================

lat_min = df_ubicaciones["_latitud"].min()
lat_max = df_ubicaciones["_latitud"].max()

lon_min = df_ubicaciones["_longitud"].min()
lon_max = df_ubicaciones["_longitud"].max()


if (
    lat_min != lat_max
    or lon_min != lon_max
):

    mapa.fit_bounds(
        [
            [lat_min, lon_min],
            [lat_max, lon_max]
        ]
    )


# ============================================================
# MOSTRAR MAPA
# ============================================================

st_folium(
    mapa,
    width=None,
    height=650,
    returned_objects=[]
)


# ============================================================
# REFERENCIAS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("🟢 **Inicio del recorrido**")

with c2:
    st.markdown("🔵 **Trayecto GPS**")

with c3:
    st.markdown("🔴 **Fin del recorrido**")


# ============================================================
# INFORMACIÓN TÉCNICA
# ============================================================

with st.expander(
    "🔎 Ver información del archivo de ubicaciones"
):

    st.write(
        f"**Columna de latitud detectada:** {col_latitud}"
    )

    st.write(
        f"**Columna de longitud detectada:** {col_longitud}"
    )

    st.write(
        f"**Cantidad de registros válidos:** "
        f"{len(df_ubicaciones)}"
    )

    st.dataframe(
        df_ubicaciones.head(50),
        use_container_width=True
    )
