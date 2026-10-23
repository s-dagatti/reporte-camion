import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import unicodedata
from math import radians, sin, cos, sqrt, atan2


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
# FUNCIONES GENERALES
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
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    return texto


def buscar_columna(df, candidatos):
    """
    Busca automáticamente una columna según una lista
    de posibles nombres.

    Primero busca coincidencias exactas.
    Después busca coincidencias parciales.
    """

    columnas_normalizadas = {
        normalizar_texto(columna): columna
        for columna in df.columns
    }

    # Coincidencia exacta
    for candidato in candidatos:

        candidato_normalizado = normalizar_texto(candidato)

        if candidato_normalizado in columnas_normalizadas:
            return columnas_normalizadas[candidato_normalizado]

    # Coincidencia parcial
    for candidato in candidatos:

        candidato_normalizado = normalizar_texto(candidato)

        for columna_normalizada, columna_original in columnas_normalizadas.items():

            if candidato_normalizado in columna_normalizada:
                return columna_original

    return None


def leer_csv_seguro(archivo):
    """
    Intenta leer el CSV utilizando distintas
    codificaciones y detección automática del separador.
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

            # Limpiar espacios en los encabezados
            df.columns = [
                str(columna).strip()
                for columna in df.columns
            ]

            return df

        except Exception as error:

            ultimo_error = error

    raise ultimo_error


def convertir_numero(serie):
    """
    Convierte una serie a valores numéricos.

    Contempla:
    - punto decimal;
    - coma decimal;
    - espacios;
    - valores vacíos.
    """

    texto = (
        serie
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
    )

    # Si un valor tiene coma pero no punto,
    # interpretamos la coma como separador decimal.
    mascara_coma_decimal = (
        texto.str.contains(",", regex=False)
        & ~texto.str.contains(".", regex=False)
    )

    texto.loc[mascara_coma_decimal] = (
        texto.loc[mascara_coma_decimal]
        .str.replace(",", ".", regex=False)
    )

    # Si tiene punto y coma, se asume que la coma
    # puede ser un separador de miles.
    texto.loc[~mascara_coma_decimal] = (
        texto.loc[~mascara_coma_decimal]
        .str.replace(",", "", regex=False)
    )

    return pd.to_numeric(
        texto,
        errors="coerce"
    )


# ============================================================
# FECHA Y HORA
# ============================================================

def construir_fecha_hora(df):
    """
    Construye una fecha-hora cronológica.

    Prioridad:
    1. Columna única de fecha y hora.
    2. Columnas de fecha y hora separadas.
    3. Columna de fecha sin hora.
    """

    columna_datetime = buscar_columna(
        df,
        [
            "fecha hora",
            "fecha/hora",
            "fecha y hora",
            "fecha de ubicación",
            "fecha de ubicacion",
            "fecha posición",
            "fecha posicion",
            "datetime",
            "timestamp",
            "date time"
        ]
    )

    if columna_datetime is not None:

        texto = (
            df[columna_datetime]
            .astype(str)
            .str.strip()
        )

        return pd.to_datetime(
            texto,
            errors="coerce",
            dayfirst=True
        )

    columna_fecha = buscar_columna(
        df,
        [
            "fecha",
            "date",
            "día",
            "dia"
        ]
    )

    columna_hora = buscar_columna(
        df,
        [
            "hora",
            "time",
            "hora de ubicación",
            "hora de ubicacion",
            "hora posición",
            "hora posicion"
        ]
    )

    if columna_fecha is not None and columna_hora is not None:

        fecha_hora_texto = (
            df[columna_fecha]
            .astype(str)
            .str.strip()
            + " "
            + df[columna_hora]
            .astype(str)
            .str.strip()
        )

        return pd.to_datetime(
            fecha_hora_texto,
            errors="coerce",
            dayfirst=True
        )

    if columna_fecha is not None:

        return pd.to_datetime(
            df[columna_fecha],
            errors="coerce",
            dayfirst=True
        )

    return pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns]"
    )


# ============================================================
# DISTANCIA GEOGRÁFICA
# ============================================================

def calcular_distancia_km(latitud_1, longitud_1, latitud_2, longitud_2):
    """
    Calcula la distancia en kilómetros entre dos
    coordenadas mediante la fórmula de Haversine.
    """

    radio_tierra_km = 6371.0088

    latitud_1 = radians(latitud_1)
    longitud_1 = radians(longitud_1)

    latitud_2 = radians(latitud_2)
    longitud_2 = radians(longitud_2)

    diferencia_latitud = latitud_2 - latitud_1
    diferencia_longitud = longitud_2 - longitud_1

    a = (
        sin(diferencia_latitud / 2) ** 2
        + cos(latitud_1)
        * cos(latitud_2)
        * sin(diferencia_longitud / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return radio_tierra_km * c


def calcular_variables_movimiento(df):
    """
    Calcula para cada punto:

    - tiempo desde el punto anterior;
    - distancia desde el punto anterior;
    - velocidad implícita;
    - motivo de corte del recorrido.
    """

    df = df.copy()

    df["_delta_min"] = (
        df["_fecha_hora"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    distancias = [0.0]

    for indice in range(1, len(df)):

        punto_anterior = df.iloc[indice - 1]
        punto_actual = df.iloc[indice]

        distancia = calcular_distancia_km(
            punto_anterior["_latitud"],
            punto_anterior["_longitud"],
            punto_actual["_latitud"],
            punto_actual["_longitud"]
        )

        distancias.append(distancia)

    df["_distancia_anterior_km"] = distancias

    horas_transcurridas = (
        df["_delta_min"]
        .div(60)
    )

    df["_velocidad_implicita_kmh"] = (
        df["_distancia_anterior_km"]
        .div(horas_transcurridas)
    )

    # Evitar divisiones por cero o infinitos.
    df["_velocidad_implicita_kmh"] = (
        df["_velocidad_implicita_kmh"]
        .replace(
            [float("inf"), float("-inf")],
            pd.NA
        )
    )

    return df


def crear_tramos(
    df,
    max_gap_minutos,
    max_velocidad_kmh,
    max_salto_km
):
    """
    Divide el historial en tramos independientes.

    Se inicia un tramo nuevo cuando:

    - es el primer punto;
    - el tiempo entre puntos supera el límite;
    - la velocidad implícita supera el límite;
    - la distancia entre dos puntos supera el límite.
    """

    df = df.copy()

    condicion_primer_punto = (
        df["_delta_min"].isna()
    )

    condicion_gap_temporal = (
        df["_delta_min"] > max_gap_minutos
    )

    condicion_velocidad = (
        df["_velocidad_implicita_kmh"] > max_velocidad_kmh
    )

    condicion_salto = (
        df["_distancia_anterior_km"] > max_salto_km
    )

    df["_nuevo_tramo"] = (
        condicion_primer_punto
        | condicion_gap_temporal
        | condicion_velocidad
        | condicion_salto
    )

    df["_motivo_corte"] = ""

    df.loc[
        condicion_gap_temporal,
        "_motivo_corte"
    ] = "Brecha temporal"

    df.loc[
        condicion_velocidad,
        "_motivo_corte"
    ] = "Velocidad implícita excesiva"

    df.loc[
        condicion_salto,
        "_motivo_corte"
    ] = "Salto geográfico excesivo"

    df.loc[
        condicion_primer_punto,
        "_motivo_corte"
    ] = "Inicio del historial"

    df["_tramo"] = (
        df["_nuevo_tramo"]
        .cumsum()
        .astype(int)
    )

    return df


def calcular_distancia_recorrido(df):
    """
    Suma las distancias solamente entre puntos
    pertenecientes al mismo tramo.
    """

    if len(df) < 2:
        return 0.0

    distancia_total = 0.0

    for indice in range(1, len(df)):

        tramo_anterior = df.iloc[indice - 1]["_tramo"]
        tramo_actual = df.iloc[indice]["_tramo"]

        if tramo_actual != tramo_anterior:
            continue

        distancia_total += df.iloc[indice]["_distancia_anterior_km"]

    return distancia_total


def formatear_numero(valor, decimales=1):
    """
    Formatea números con punto de miles y coma decimal.
    """

    formato = f"{valor:,.{decimales}f}"

    return (
        formato
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


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
        help="CSV que contiene los parámetros de funcionamiento del motor."
    )

    st.markdown("---")

    st.subheader("🧭 Control del recorrido")

    max_gap_minutos = st.number_input(
        "Cortar tramo si no hay datos durante más de",
        min_value=1,
        max_value=1440,
        value=60,
        step=5,
        format="%d",
        help="Valor expresado en minutos."
    )

    max_velocidad_kmh = st.number_input(
        "Velocidad implícita máxima",
        min_value=20,
        max_value=500,
        value=160,
        step=10,
        format="%d",
        help=(
            "Si el desplazamiento entre dos registros exige una "
            "velocidad superior, no se dibuja la línea entre ellos."
        )
    )

    max_salto_km = st.number_input(
        "Salto máximo entre puntos",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        format="%d",
        help=(
            "Si dos registros consecutivos están separados por una "
            "distancia mayor, se inicia un tramo nuevo."
        )
    )

    st.markdown("---")


# ============================================================
# ENCABEZADO
# ============================================================

st.markdown(
    """
    <div class="titulo-principal">
        🚛 Reporte de operación del camión
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitulo">
        Análisis de recorrido y funcionamiento de la unidad
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ESPERA DE DATOS
# ============================================================

if archivo_ubicaciones is None:

    st.info(
        "👈 Cargá en el sidebar el CSV de historial de "
        "ubicaciones para comenzar el análisis."
    )

    st.stop()


# ============================================================
# LEER HISTORIAL DE UBICACIONES
# ============================================================

try:

    df_ubicaciones_original = leer_csv_seguro(
        archivo_ubicaciones
    )

except Exception as error:

    st.error(
        f"No se pudo leer el archivo de ubicaciones: {error}"
    )

    st.stop()


df_ubicaciones = df_ubicaciones_original.copy()


# ============================================================
# IDENTIFICAR COLUMNAS
# ============================================================

col_latitud = buscar_columna(
    df_ubicaciones,
    [
        "latitud",
        "latitude",
        "gps latitude",
        "posición latitud",
        "posicion latitud",
        "position latitude",
        "lat"
    ]
)

col_longitud = buscar_columna(
    df_ubicaciones,
    [
        "longitud",
        "longitude",
        "gps longitude",
        "posición longitud",
        "posicion longitud",
        "position longitude",
        "lng",
        "lon"
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
        "serie maquina",
        "machine serial",
        "vin"
    ]
)


# ============================================================
# VALIDAR COLUMNAS GPS
# ============================================================

if col_latitud is None or col_longitud is None:

    st.error(
        "❌ No pude identificar automáticamente las columnas "
        "de latitud y longitud."
    )

    st.write("**Columnas encontradas en el CSV:**")

    st.code(
        "\n".join(
            str(columna)
            for columna in df_ubicaciones.columns
        )
    )

    st.stop()


# ============================================================
# CONSTRUIR FECHA-HORA Y COORDENADAS
# ============================================================

df_ubicaciones["_fecha_hora"] = construir_fecha_hora(
    df_ubicaciones
)

df_ubicaciones["_latitud"] = convertir_numero(
    df_ubicaciones[col_latitud]
)

df_ubicaciones["_longitud"] = convertir_numero(
    df_ubicaciones[col_longitud]
)


# ============================================================
# CONTROL DE REGISTROS DESCARTADOS
# ============================================================

cantidad_original = len(df_ubicaciones)

cantidad_sin_fecha = int(
    df_ubicaciones["_fecha_hora"].isna().sum()
)

cantidad_sin_coordenadas = int(
    (
        df_ubicaciones["_latitud"].isna()
        | df_ubicaciones["_longitud"].isna()
    ).sum()
)


# ============================================================
# ELIMINAR REGISTROS INVÁLIDOS
# ============================================================

df_ubicaciones = (
    df_ubicaciones
    .dropna(
        subset=[
            "_fecha_hora",
            "_latitud",
            "_longitud"
        ]
    )
    .copy()
)


# ============================================================
# VALIDAR RANGOS GEOGRÁFICOS
# ============================================================

df_ubicaciones = df_ubicaciones[
    df_ubicaciones["_latitud"].between(-90, 90)
    & df_ubicaciones["_longitud"].between(-180, 180)
].copy()


if df_ubicaciones.empty:

    st.error(
        "No quedaron registros válidos después de procesar "
        "la fecha, la hora y las coordenadas."
    )

    st.stop()


# ============================================================
# ORDENAR CRONOLÓGICAMENTE
# ============================================================

df_ubicaciones = (
    df_ubicaciones
    .sort_values(
        by="_fecha_hora",
        ascending=True,
        kind="stable"
    )
    .reset_index(drop=True)
)


# ============================================================
# ELIMINAR DUPLICADOS EXACTOS
# ============================================================

cantidad_antes_duplicados = len(df_ubicaciones)

df_ubicaciones = (
    df_ubicaciones
    .drop_duplicates(
        subset=[
            "_fecha_hora",
            "_latitud",
            "_longitud"
        ],
        keep="first"
    )
    .reset_index(drop=True)
)

duplicados_eliminados = (
    cantidad_antes_duplicados
    - len(df_ubicaciones)
)


# ============================================================
# CALCULAR MOVIMIENTO Y CREAR TRAMOS
# ============================================================

df_ubicaciones = calcular_variables_movimiento(
    df_ubicaciones
)

df_ubicaciones = crear_tramos(
    df=df_ubicaciones,
    max_gap_minutos=max_gap_minutos,
    max_velocidad_kmh=max_velocidad_kmh,
    max_salto_km=max_salto_km
)


# ============================================================
# LEER DATOS DEL MOTOR
# ============================================================

df_motor = None

if archivo_motor is not None:

    try:

        df_motor = leer_csv_seguro(
            archivo_motor
        )

        df_motor["_fecha_hora"] = construir_fecha_hora(
            df_motor
        )

    except Exception as error:

        st.warning(
            f"El archivo de motor no pudo leerse: {error}"
        )

        df_motor = None


# ============================================================
# PERÍODO DEL HISTORIAL GPS
# ============================================================

fecha_inicio_gps = (
    df_ubicaciones["_fecha_hora"].min()
)

fecha_fin_gps = (
    df_ubicaciones["_fecha_hora"].max()
)


# El período principal se basa en el historial GPS.
fecha_inicio = fecha_inicio_gps
fecha_fin = fecha_fin_gps


# Si existe archivo de motor, podemos ampliar el período general.
if (
    df_motor is not None
    and df_motor["_fecha_hora"].notna().any()
):

    fecha_inicio_motor = (
        df_motor["_fecha_hora"].min()
    )

    fecha_fin_motor = (
        df_motor["_fecha_hora"].max()
    )

    fecha_inicio = min(
        fecha_inicio,
        fecha_inicio_motor
    )

    fecha_fin = max(
        fecha_fin,
        fecha_fin_motor
    )


# ============================================================
# MOSTRAR PERÍODO EN SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("📅 Período de análisis")

    st.write(
        "**Desde:**",
        fecha_inicio.strftime("%d/%m/%Y %H:%M:%S")
    )

    st.write(
        "**Hasta:**",
        fecha_fin.strftime("%d/%m/%Y %H:%M:%S")
    )

    st.caption(
        "El orden del recorrido se determina con la fecha "
        "y la hora de cada posición GPS."
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
        .str.strip()
    )

    series_validas = series_validas[
        series_validas.ne("")
    ].unique()

    if len(series_validas) > 0:
        numero_serie = series_validas[0]


# ============================================================
# INDICADORES
# ============================================================

distancia_total = calcular_distancia_recorrido(
    df_ubicaciones
)

cantidad_puntos = len(df_ubicaciones)

cantidad_tramos = int(
    df_ubicaciones["_tramo"].nunique()
)

cantidad_cortes = int(
    max(cantidad_tramos - 1, 0)
)

duracion_dias = (
    fecha_fin_gps - fecha_inicio_gps
).total_seconds() / 86400


# ============================================================
# TARJETAS PRINCIPALES
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "🚛 Unidad",
        numero_serie
    )

with col2:

    st.metric(
        "📍 Puntos GPS",
        f"{cantidad_puntos:,}".replace(",", ".")
    )

with col3:

    st.metric(
        "🛣️ Distancia GPS",
        f"{formatear_numero(distancia_total, 1)} km"
    )

with col4:

    st.metric(
        "🧭 Tramos",
        cantidad_tramos
    )

with col5:

    st.metric(
        "📅 Período",
        f"{formatear_numero(duracion_dias, 1)} días"
    )


st.markdown("---")


# ============================================================
# MAPA
# ============================================================

st.subheader("📍 Recorrido de la unidad")

st.caption(
    "Los puntos GPS se ordenan cronológicamente. La línea se "
    "interrumpe ante brechas temporales, saltos geográficos o "
    "velocidades implícitas superiores a los límites configurados."
)


centro_latitud = (
    df_ubicaciones["_latitud"].mean()
)

centro_longitud = (
    df_ubicaciones["_longitud"].mean()
)


mapa = folium.Map(
    location=[
        centro_latitud,
        centro_longitud
    ],
    zoom_start=7,
    control_scale=True,
    tiles="OpenStreetMap",
    prefer_canvas=True
)


# ============================================================
# DIBUJAR LOS TRAMOS
# ============================================================

colores_tramos = [
    "#00A6A6",
    "#2563EB",
    "#7C3AED",
    "#EA580C",
    "#0891B2",
    "#16A34A"
]


for numero_tramo, df_tramo in df_ubicaciones.groupby(
    "_tramo",
    sort=True
):

    if len(df_tramo) < 2:
        continue

    coordenadas_tramo = (
        df_tramo[
            [
                "_latitud",
                "_longitud"
            ]
        ]
        .values
        .tolist()
    )

    color_tramo = colores_tramos[
        (int(numero_tramo) - 1) % len(colores_tramos)
    ]

    fecha_tramo_inicio = (
        df_tramo["_fecha_hora"]
        .min()
        .strftime("%d/%m/%Y %H:%M")
    )

    fecha_tramo_fin = (
        df_tramo["_fecha_hora"]
        .max()
        .strftime("%d/%m/%Y %H:%M")
    )

    tooltip_tramo = (
        f"Tramo {numero_tramo} | "
        f"{fecha_tramo_inicio} a {fecha_tramo_fin}"
    )

    folium.PolyLine(
        locations=coordenadas_tramo,
        color=color_tramo,
        weight=4,
        opacity=0.9,
        tooltip=tooltip_tramo,
        smooth_factor=1
    ).add_to(mapa)


# ============================================================
# MARCADOR DE INICIO
# ============================================================

inicio = df_ubicaciones.iloc[0]

texto_inicio = (
    "<b>Inicio del historial</b>"
    f"<br>{inicio['_fecha_hora'].strftime('%d/%m/%Y %H:%M:%S')}"
    f"<br>Latitud: {inicio['_latitud']:.6f}"
    f"<br>Longitud: {inicio['_longitud']:.6f}"
)

folium.Marker(
    location=[
        inicio["_latitud"],
        inicio["_longitud"]
    ],
    popup=folium.Popup(
        texto_inicio,
        max_width=280
    ),
    tooltip="Inicio del historial",
    icon=folium.Icon(
        color="green",
        icon="play",
        prefix="fa"
    )
).add_to(mapa)


# ============================================================
# MARCADOR DE FIN
# ============================================================

fin = df_ubicaciones.iloc[-1]

texto_fin = (
    "<b>Fin del historial</b>"
    f"<br>{fin['_fecha_hora'].strftime('%d/%m/%Y %H:%M:%S')}"
    f"<br>Latitud: {fin['_latitud']:.6f}"
    f"<br>Longitud: {fin['_longitud']:.6f}"
)

folium.Marker(
    location=[
        fin["_latitud"],
        fin["_longitud"]
    ],
    popup=folium.Popup(
        texto_fin,
        max_width=280
    ),
    tooltip="Fin del historial",
    icon=folium.Icon(
        color="red",
        icon="stop",
        prefix="fa"
    )
).add_to(mapa)


# ============================================================
# MARCAR INICIOS DE TRAMOS
# ============================================================

for numero_tramo, df_tramo in df_ubicaciones.groupby(
    "_tramo",
    sort=True
):

    if int(numero_tramo) == 1:
        continue

    primer_punto_tramo = df_tramo.iloc[0]

    motivo = primer_punto_tramo["_motivo_corte"]

    popup_corte = (
        f"<b>Inicio del tramo {numero_tramo}</b>"
        f"<br>{primer_punto_tramo['_fecha_hora'].strftime('%d/%m/%Y %H:%M:%S')}"
        f"<br>Motivo: {motivo}"
    )

    folium.CircleMarker(
        location=[
            primer_punto_tramo["_latitud"],
            primer_punto_tramo["_longitud"]
        ],
        radius=6,
        color="#F59E0B",
        fill=True,
        fill_color="#F59E0B",
        fill_opacity=1,
        popup=folium.Popup(
            popup_corte,
            max_width=300
        ),
        tooltip=f"Inicio del tramo {numero_tramo}"
    ).add_to(mapa)


# ============================================================
# AJUSTAR MAPA A LOS DATOS
# ============================================================

latitud_minima = (
    df_ubicaciones["_latitud"].min()
)

latitud_maxima = (
    df_ubicaciones["_latitud"].max()
)

longitud_minima = (
    df_ubicaciones["_longitud"].min()
)

longitud_maxima = (
    df_ubicaciones["_longitud"].max()
)


if (
    latitud_minima != latitud_maxima
    or longitud_minima != longitud_maxima
):

    mapa.fit_bounds(
        [
            [
                latitud_minima,
                longitud_minima
            ],
            [
                latitud_maxima,
                longitud_maxima
            ]
        ],
        padding=[
            20,
            20
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

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("🟢 **Inicio del historial**")

with c2:
    st.markdown("🔵 **Trayectos GPS válidos**")

with c3:
    st.markdown("🟠 **Inicio de un tramo nuevo**")

with c4:
    st.markdown("🔴 **Fin del historial**")


# ============================================================
# INFORMACIÓN TÉCNICA
# ============================================================

with st.expander(
    "🔎 Ver control del historial de ubicaciones"
):

    st.write("### Columnas detectadas")

    st.write(
        f"**Latitud:** {col_latitud}"
    )

    st.write(
        f"**Longitud:** {col_longitud}"
    )

    st.write(
        f"**Número de serie:** "
        f"{col_serie if col_serie is not None else 'No detectado'}"
    )

    st.write("### Control de registros")

    control1, control2, control3, control4 = st.columns(4)

    with control1:

        st.metric(
            "Registros originales",
            cantidad_original
        )

    with control2:

        st.metric(
            "Sin fecha-hora",
            cantidad_sin_fecha
        )

    with control3:

        st.metric(
            "Sin coordenadas",
            cantidad_sin_coordenadas
        )

    with control4:

        st.metric(
            "Duplicados eliminados",
            duplicados_eliminados
        )

    st.write("### Orden cronológico procesado")

    columnas_control = [
        "_fecha_hora",
        "_latitud",
        "_longitud",
        "_delta_min",
        "_distancia_anterior_km",
        "_velocidad_implicita_kmh",
        "_tramo",
        "_motivo_corte"
    ]

    columnas_control = [
        columna
        for columna in columnas_control
        if columna in df_ubicaciones.columns
    ]

    st.dataframe(
        df_ubicaciones[columnas_control].head(200),
        use_container_width=True,
        hide_index=True
    )

    st.write("### Cortes detectados")

    df_cortes = df_ubicaciones[
        df_ubicaciones["_nuevo_tramo"]
    ].copy()

    # El primer registro es el inicio del historial,
    # no un corte real.
    if len(df_cortes) > 1:

        st.dataframe(
            df_cortes[
                [
                    "_fecha_hora",
                    "_latitud",
                    "_longitud",
                    "_delta_min",
                    "_distancia_anterior_km",
                    "_velocidad_implicita_kmh",
                    "_tramo",
                    "_motivo_corte"
                ]
            ].iloc[1:],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No se detectaron interrupciones adicionales "
            "en el recorrido."
        )

    st.caption(
        f"Se generaron {cantidad_tramos} tramos y "
        f"{cantidad_cortes} cortes entre posiciones."
    )
