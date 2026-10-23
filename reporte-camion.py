import streamlit as st
import pandas as pd
import folium
import unicodedata
import numpy as np

from math import radians, sin, cos, sqrt, atan2
from streamlit_folium import st_folium
from branca.colormap import LinearColormap


# ============================================================
# CONFIGURACIÓN GENERAL
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

    texto = str(texto).strip().lower()

    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    return " ".join(texto.split())


def buscar_columna(df, candidatos):

    columnas_normalizadas = {
        normalizar_texto(columna): columna
        for columna in df.columns
    }

    # Coincidencia exacta
    for candidato in candidatos:

        candidato_normalizado = normalizar_texto(
            candidato
        )

        if candidato_normalizado in columnas_normalizadas:

            return columnas_normalizadas[
                candidato_normalizado
            ]

    # Coincidencia parcial
    for candidato in candidatos:

        candidato_normalizado = normalizar_texto(
            candidato
        )

        for (
            columna_normalizada,
            columna_original
        ) in columnas_normalizadas.items():

            if candidato_normalizado in columna_normalizada:

                return columna_original

    return None


def leer_csv_seguro(archivo):

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

            df.columns = [
                str(columna).strip()
                for columna in df.columns
            ]

            return df

        except Exception as error:

            ultimo_error = error

    raise ultimo_error


def convertir_numero(serie):

    texto = (
        serie
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
    )

    mascara_coma_decimal = (
        texto.str.contains(",", regex=False)
        & ~texto.str.contains(".", regex=False)
    )

    texto.loc[mascara_coma_decimal] = (
        texto.loc[mascara_coma_decimal]
        .str.replace(",", ".", regex=False)
    )

    texto.loc[~mascara_coma_decimal] = (
        texto.loc[~mascara_coma_decimal]
        .str.replace(",", "", regex=False)
    )

    return pd.to_numeric(
        texto,
        errors="coerce"
    )


def formatear_numero(valor, decimales=1):

    if pd.isna(valor):
        return "-"

    formato = f"{valor:,.{decimales}f}"

    return (
        formato
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


# ============================================================
# FUNCIONES DE FECHA
# ============================================================

def convertir_fecha_hora(serie):

    return pd.to_datetime(
        serie.astype(str).str.strip(),
        errors="coerce",
        dayfirst=True
    )


def construir_fecha_hora_gps(df):

    columna_datetime = buscar_columna(
        df,
        [
            "fecha hora",
            "fecha/hora",
            "fecha y hora",
            "datetime",
            "timestamp",
            "date time"
        ]
    )

    if columna_datetime is not None:

        return convertir_fecha_hora(
            df[columna_datetime]
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

    if (
        columna_fecha is not None
        and columna_hora is not None
    ):

        texto_fecha_hora = (
            df[columna_fecha]
            .astype(str)
            .str.strip()
            + " "
            + df[columna_hora]
            .astype(str)
            .str.strip()
        )

        return convertir_fecha_hora(
            texto_fecha_hora
        )

    if columna_fecha is not None:

        return convertir_fecha_hora(
            df[columna_fecha]
        )

    return pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns]"
    )


def construir_intervalo_motor(df):

    df = df.copy()

    columna_fecha_inicio = buscar_columna(
        df,
        [
            "fecha de inicio",
            "fecha inicio",
            "start date"
        ]
    )

    columna_hora_inicio = buscar_columna(
        df,
        [
            "hora de inicio",
            "hora inicio",
            "start time"
        ]
    )

    columna_fecha_fin = buscar_columna(
        df,
        [
            "fecha de terminación",
            "fecha de terminacion",
            "fecha de fin",
            "fecha fin",
            "end date"
        ]
    )

    columna_hora_fin = buscar_columna(
        df,
        [
            "tiempo final",
            "hora final",
            "hora de terminación",
            "hora de terminacion",
            "end time"
        ]
    )

    if (
        columna_fecha_inicio is None
        or columna_hora_inicio is None
    ):

        raise ValueError(
            "No se encontraron las columnas de fecha "
            "y hora inicial del archivo del motor."
        )

    texto_inicio = (
        df[columna_fecha_inicio]
        .astype(str)
        .str.strip()
        + " "
        + df[columna_hora_inicio]
        .astype(str)
        .str.strip()
    )

    df["_periodo_inicio"] = convertir_fecha_hora(
        texto_inicio
    )

    if (
        columna_fecha_fin is not None
        and columna_hora_fin is not None
    ):

        texto_fin = (
            df[columna_fecha_fin]
            .astype(str)
            .str.strip()
            + " "
            + df[columna_hora_fin]
            .astype(str)
            .str.strip()
        )

        df["_periodo_fin"] = convertir_fecha_hora(
            texto_fin
        )

    else:

        df["_periodo_fin"] = (
            df["_periodo_inicio"]
            + pd.Timedelta(hours=1)
            - pd.Timedelta(seconds=1)
        )

    mascara_fin_invalido = (
        df["_periodo_fin"].isna()
        | (
            df["_periodo_fin"]
            < df["_periodo_inicio"]
        )
    )

    df.loc[
        mascara_fin_invalido,
        "_periodo_fin"
    ] = (
        df.loc[
            mascara_fin_invalido,
            "_periodo_inicio"
        ]
        + pd.Timedelta(hours=1)
        - pd.Timedelta(seconds=1)
    )

    return df


# ============================================================
# FUNCIONES GEOGRÁFICAS
# ============================================================

def calcular_distancia_km(
    latitud_1,
    longitud_1,
    latitud_2,
    longitud_2
):

    radio_tierra_km = 6371.0088

    latitud_1 = radians(latitud_1)
    longitud_1 = radians(longitud_1)
    latitud_2 = radians(latitud_2)
    longitud_2 = radians(longitud_2)

    diferencia_latitud = (
        latitud_2 - latitud_1
    )

    diferencia_longitud = (
        longitud_2 - longitud_1
    )

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

    df = df.copy()

    df["_delta_min"] = (
        df["_fecha_hora"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    distancias = [0.0]

    for indice in range(1, len(df)):

        anterior = df.iloc[indice - 1]
        actual = df.iloc[indice]

        distancia = calcular_distancia_km(
            anterior["_latitud"],
            anterior["_longitud"],
            actual["_latitud"],
            actual["_longitud"]
        )

        distancias.append(distancia)

    df["_distancia_anterior_km"] = distancias

    df["_velocidad_implicita_kmh"] = np.nan

    mascara_tiempo_positivo = (
        df["_delta_min"] > 0
    )

    df.loc[
        mascara_tiempo_positivo,
        "_velocidad_implicita_kmh"
    ] = (
        df.loc[
            mascara_tiempo_positivo,
            "_distancia_anterior_km"
        ]
        / (
            df.loc[
                mascara_tiempo_positivo,
                "_delta_min"
            ]
            / 60
        )
    )

    df["_velocidad_implicita_kmh"] = (
        df["_velocidad_implicita_kmh"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    return df


def crear_tramos(
    df,
    max_gap_minutos,
    max_velocidad_kmh,
    max_salto_km,
    max_salto_misma_hora_km
):

    df = df.copy()

    primer_punto = (
        df["_delta_min"].isna()
    )

    brecha_temporal = (
        df["_delta_min"] > max_gap_minutos
    )

    tiempo_negativo = (
        df["_delta_min"] < 0
    )

    velocidad_excesiva = (
        df["_velocidad_implicita_kmh"].notna()
        & (
            df["_velocidad_implicita_kmh"]
            > max_velocidad_kmh
        )
    )

    salto_excesivo = (
        df["_distancia_anterior_km"]
        > max_salto_km
    )

    timestamp_repetido_con_salto = (
        (df["_delta_min"] == 0)
        & (
            df["_distancia_anterior_km"]
            > max_salto_misma_hora_km
        )
    )

    df["_nuevo_tramo"] = (
        primer_punto
        | brecha_temporal
        | tiempo_negativo
        | velocidad_excesiva
        | salto_excesivo
        | timestamp_repetido_con_salto
    )

    df["_motivo_corte"] = ""

    df.loc[
        primer_punto,
        "_motivo_corte"
    ] = "Inicio del historial"

    df.loc[
        brecha_temporal,
        "_motivo_corte"
    ] = "Brecha temporal"

    df.loc[
        tiempo_negativo,
        "_motivo_corte"
    ] = "Orden temporal negativo"

    df.loc[
        velocidad_excesiva,
        "_motivo_corte"
    ] = "Velocidad implícita excesiva"

    df.loc[
        salto_excesivo,
        "_motivo_corte"
    ] = "Salto geográfico excesivo"

    df.loc[
        timestamp_repetido_con_salto,
        "_motivo_corte"
    ] = "Timestamp repetido con salto geográfico"

    df["_tramo"] = (
        df["_nuevo_tramo"]
        .cumsum()
        .astype(int)
    )

    return df


def calcular_distancia_recorrido(df):

    if len(df) < 2:
        return 0.0

    mascara_mismo_tramo = (
        df["_tramo"]
        == df["_tramo"].shift(1)
    )

    return (
        df.loc[
            mascara_mismo_tramo,
            "_distancia_anterior_km"
        ]
        .sum()
    )


# ============================================================
# FUNCIONES DEL MOTOR
# ============================================================

def obtener_valor_metrica(
    grupo,
    texto_categoria=None,
    texto_serie=None
):

    mascara = pd.Series(
        True,
        index=grupo.index
    )

    if texto_categoria is not None:

        mascara &= grupo[
            "_categoria_norm"
        ].str.contains(
            normalizar_texto(texto_categoria),
            na=False,
            regex=False
        )

    if texto_serie is not None:

        mascara &= grupo[
            "_serie_norm"
        ].str.contains(
            normalizar_texto(texto_serie),
            na=False,
            regex=False
        )

    valores = grupo.loc[
        mascara,
        "_valor_num"
    ].dropna()

    if valores.empty:
        return np.nan

    return valores.sum()


def obtener_valor_metrica_exacto(
    grupo,
    categoria_exacta=None,
    serie_exacta=None
):

    mascara = pd.Series(
        True,
        index=grupo.index
    )

    if categoria_exacta is not None:

        mascara &= (
            grupo["_categoria_norm"]
            == normalizar_texto(categoria_exacta)
        )

    if serie_exacta is not None:

        mascara &= (
            grupo["_serie_norm"]
            == normalizar_texto(serie_exacta)
        )

    valores = grupo.loc[
        mascara,
        "_valor_num"
    ].dropna()

    if valores.empty:
        return np.nan

    return valores.sum()


def procesar_motor(df_motor):

    df_motor = df_motor.copy()

    columna_categoria = buscar_columna(
        df_motor,
        [
            "categoría",
            "categoria",
            "category"
        ]
    )

    columna_serie = buscar_columna(
        df_motor,
        [
            "serie",
            "series"
        ]
    )

    columna_valor = buscar_columna(
        df_motor,
        [
            "valor",
            "value"
        ]
    )

    if (
        columna_categoria is None
        or columna_serie is None
        or columna_valor is None
    ):

        raise ValueError(
            "No se pudieron identificar Categoría, Serie y Valor "
            "en el archivo del motor."
        )

    df_motor = construir_intervalo_motor(
        df_motor
    )

    df_motor["_categoria_norm"] = (
        df_motor[columna_categoria]
        .apply(normalizar_texto)
    )

    df_motor["_serie_norm"] = (
        df_motor[columna_serie]
        .apply(normalizar_texto)
    )

    df_motor["_valor_num"] = convertir_numero(
        df_motor[columna_valor]
    )

    df_motor = df_motor.dropna(
        subset=[
            "_periodo_inicio",
            "_periodo_fin"
        ]
    ).copy()

    registros = []

    for (
        periodo_inicio,
        periodo_fin
    ), grupo in df_motor.groupby(
        [
            "_periodo_inicio",
            "_periodo_fin"
        ],
        dropna=False,
        sort=True
    ):

        combustible_total_l = obtener_valor_metrica_exacto(
            grupo,
            categoria_exacta="Combustible consumido en período",
            serie_exacta="Combustible consumido"
        )

        combustible_trabajando_l = obtener_valor_metrica_exacto(
            grupo,
            categoria_exacta="Combustible consumido",
            serie_exacta="Trabajando"
        )

        combustible_reposo_l = obtener_valor_metrica_exacto(
            grupo,
            categoria_exacta="Combustible consumido",
            serie_exacta="En reposo"
        )

        combustible_transporte_l = obtener_valor_metrica_exacto(
            grupo,
            categoria_exacta="Combustible consumido",
            serie_exacta="Transporte"
        )

        horas_trabajando = obtener_valor_metrica(
            grupo,
            texto_categoria="utilización",
            texto_serie="trabajando"
        )

        horas_reposo = obtener_valor_metrica(
            grupo,
            texto_categoria="utilización",
            texto_serie="en reposo"
        )

        horas_transporte = obtener_valor_metrica(
            grupo,
            texto_categoria="utilización",
            texto_serie="transporte"
        )

        velocidad_trabajando_kmh = obtener_valor_metrica(
            grupo,
            texto_categoria="velocidad de avance promedio",
            texto_serie="trabajando"
        )

        distancia_periodo_km = obtener_valor_metrica(
            grupo,
            texto_categoria="distancia"
        )

        consumo_reposo_archivo = obtener_valor_metrica(
            grupo,
            texto_categoria="consumo promedio de combustible",
            texto_serie="en reposo"
        )

        registros.append(
            {
                "_periodo_inicio": periodo_inicio,
                "_periodo_fin": periodo_fin,
                "_combustible_total_l": combustible_total_l,
                "_combustible_trabajando_l": combustible_trabajando_l,
                "_combustible_reposo_l": combustible_reposo_l,
                "_combustible_transporte_l": combustible_transporte_l,
                "_horas_trabajando": horas_trabajando,
                "_horas_reposo": horas_reposo,
                "_horas_transporte": horas_transporte,
                "_velocidad_trabajando_kmh": velocidad_trabajando_kmh,
                "_distancia_periodo_km": distancia_periodo_km,
                "_consumo_reposo_l_h_archivo": consumo_reposo_archivo
            }
        )

    df_periodos = pd.DataFrame(
        registros
    )

    if df_periodos.empty:

        raise ValueError(
            "No fue posible construir los intervalos del motor."
        )

    # Distancia estimada durante estado trabajando.
    df_periodos["_distancia_trabajando_km"] = (
        df_periodos["_velocidad_trabajando_kmh"]
        * df_periodos["_horas_trabajando"]
    )

    # ========================================================
    # RENDIMIENTO TRABAJANDO EN KM/L
    # ========================================================

    df_periodos["_rendimiento_trabajando_km_l"] = np.nan

    mascara_rendimiento_trabajo_valido = (
        df_periodos["_combustible_trabajando_l"] > 0.01
    ) & (
        df_periodos["_distancia_trabajando_km"] > 0.1
    )

    df_periodos.loc[
        mascara_rendimiento_trabajo_valido,
        "_rendimiento_trabajando_km_l"
    ] = (
        df_periodos.loc[
            mascara_rendimiento_trabajo_valido,
            "_distancia_trabajando_km"
        ]
        / df_periodos.loc[
            mascara_rendimiento_trabajo_valido,
            "_combustible_trabajando_l"
        ]
    )

    # Rendimiento total del intervalo.
    df_periodos["_rendimiento_total_km_l"] = np.nan

    mascara_rendimiento_total_valido = (
        df_periodos["_combustible_total_l"] > 0.01
    ) & (
        df_periodos["_distancia_periodo_km"] > 0.1
    )

    df_periodos.loc[
        mascara_rendimiento_total_valido,
        "_rendimiento_total_km_l"
    ] = (
        df_periodos.loc[
            mascara_rendimiento_total_valido,
            "_distancia_periodo_km"
        ]
        / df_periodos.loc[
            mascara_rendimiento_total_valido,
            "_combustible_total_l"
        ]
    )

    # Consumo en reposo l/h.
    df_periodos["_consumo_reposo_l_h"] = (
        df_periodos["_consumo_reposo_l_h_archivo"]
    )

    mascara_reposo_calculable = (
        df_periodos["_consumo_reposo_l_h"].isna()
        & (
            df_periodos["_horas_reposo"] > 0
        )
        & (
            df_periodos["_combustible_reposo_l"] >= 0
        )
    )

    df_periodos.loc[
        mascara_reposo_calculable,
        "_consumo_reposo_l_h"
    ] = (
        df_periodos.loc[
            mascara_reposo_calculable,
            "_combustible_reposo_l"
        ]
        / df_periodos.loc[
            mascara_reposo_calculable,
            "_horas_reposo"
        ]
    )

    return df_motor, df_periodos


# ============================================================
# COMBINAR GPS Y MOTOR POR INTERVALO TEMPORAL
# ============================================================

def asignar_motor_a_gps(
    df_gps,
    df_periodos
):

    df_gps = (
        df_gps
        .sort_values(
            [
                "_fecha_hora",
                "_orden_original"
            ],
            kind="stable"
        )
        .copy()
    )

    df_periodos = (
        df_periodos
        .sort_values("_periodo_inicio")
        .copy()
    )

    columnas_motor = [
        "_periodo_inicio",
        "_periodo_fin",
        "_combustible_total_l",
        "_combustible_trabajando_l",
        "_combustible_reposo_l",
        "_combustible_transporte_l",
        "_horas_trabajando",
        "_horas_reposo",
        "_horas_transporte",
        "_velocidad_trabajando_kmh",
        "_distancia_periodo_km",
        "_distancia_trabajando_km",
        "_rendimiento_trabajando_km_l",
        "_rendimiento_total_km_l",
        "_consumo_reposo_l_h"
    ]

    df_cruzado = pd.merge_asof(
        df_gps,
        df_periodos[columnas_motor],
        left_on="_fecha_hora",
        right_on="_periodo_inicio",
        direction="backward"
    )

    # merge_asof asigna el último inicio anterior.
    # Esta comprobación confirma que el punto GPS esté
    # realmente incluido en el intervalo del motor.
    mascara_fuera_intervalo = (
        df_cruzado["_periodo_fin"].isna()
        | (
            df_cruzado["_fecha_hora"]
            > df_cruzado["_periodo_fin"]
        )
    )

    columnas_valores_motor = [
        columna
        for columna in columnas_motor
        if columna not in [
            "_periodo_inicio",
            "_periodo_fin"
        ]
    ]

    df_cruzado.loc[
        mascara_fuera_intervalo,
        columnas_valores_motor
    ] = np.nan

    df_cruzado.loc[
        mascara_fuera_intervalo,
        [
            "_periodo_inicio",
            "_periodo_fin"
        ]
    ] = pd.NaT

    return df_cruzado


# ============================================================
# EVENTOS DE REPOSO
# ============================================================

def generar_eventos_reposo(
    df_periodos,
    df_gps,
    minimo_combustible_l,
    minimo_horas_reposo
):

    eventos = []

    mascara_reposo = (
        (
            df_periodos["_combustible_reposo_l"]
            >= minimo_combustible_l
        )
        | (
            df_periodos["_horas_reposo"]
            >= minimo_horas_reposo
        )
    )

    periodos_reposo = df_periodos[
        mascara_reposo
    ].copy()

    for _, periodo in periodos_reposo.iterrows():

        inicio = periodo["_periodo_inicio"]
        fin = periodo["_periodo_fin"]

        puntos_intervalo = df_gps[
            (
                df_gps["_fecha_hora"] >= inicio
            )
            & (
                df_gps["_fecha_hora"] <= fin
            )
        ].copy()

        if puntos_intervalo.empty:
            continue

        # Seleccionamos la posición con menor movimiento
        # dentro del intervalo como punto representativo.
        puntos_intervalo["_movimiento_reposo"] = (
            puntos_intervalo["_distancia_anterior_km"]
            .fillna(0)
        )

        puntos_intervalo = (
            puntos_intervalo
            .sort_values(
                [
                    "_movimiento_reposo",
                    "_fecha_hora",
                    "_orden_original"
                ],
                kind="stable"
            )
        )

        posicion = puntos_intervalo.iloc[0]

        eventos.append(
            {
                "_periodo_inicio": inicio,
                "_periodo_fin": fin,
                "_latitud": posicion["_latitud"],
                "_longitud": posicion["_longitud"],
                "_fecha_posicion": posicion["_fecha_hora"],
                "_combustible_reposo_l": periodo[
                    "_combustible_reposo_l"
                ],
                "_horas_reposo": periodo[
                    "_horas_reposo"
                ],
                "_consumo_reposo_l_h": periodo[
                    "_consumo_reposo_l_h"
                ]
            }
        )

    return pd.DataFrame(eventos)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🚛 Reporte de Camión")

    st.markdown("---")

    st.subheader("📂 Carga de archivos")

    archivo_ubicaciones = st.file_uploader(
        "Historial de ubicaciones",
        type=["csv"]
    )

    archivo_motor = st.file_uploader(
        "Datos del motor",
        type=["csv"]
    )

    st.markdown("---")

    st.subheader("🧭 Control del recorrido")

    max_gap_minutos = st.number_input(
        "Brecha temporal máxima, minutos",
        min_value=1,
        max_value=1440,
        value=60,
        step=5
    )

    max_velocidad_kmh = st.number_input(
        "Velocidad implícita máxima, km/h",
        min_value=20,
        max_value=500,
        value=160,
        step=10
    )

    max_salto_km = st.number_input(
        "Salto máximo entre puntos, km",
        min_value=1,
        max_value=1000,
        value=100,
        step=10
    )

    max_salto_misma_hora_km = st.number_input(
        "Salto máximo con timestamp repetido, km",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.5,
        format="%.1f"
    )

    st.markdown("---")

    st.subheader("🅿️ Eventos de reposo")

    minimo_combustible_reposo_l = st.number_input(
        "Combustible mínimo en reposo, l",
        min_value=0.0,
        max_value=1000.0,
        value=0.05,
        step=0.05,
        format="%.2f"
    )

    minimo_horas_reposo = st.number_input(
        "Tiempo mínimo en reposo, h",
        min_value=0.0,
        max_value=24.0,
        value=0.05,
        step=0.05,
        format="%.2f"
    )


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
        Recorrido, rendimiento de combustible y estados de operación
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# VALIDAR ARCHIVOS
# ============================================================

if archivo_ubicaciones is None:

    st.info(
        "👈 Cargá el historial de ubicaciones para comenzar."
    )

    st.stop()


if archivo_motor is None:

    st.info(
        "👈 Cargá también el archivo de datos del motor."
    )

    st.stop()


# ============================================================
# LEER GPS
# ============================================================

try:

    df_ubicaciones = leer_csv_seguro(
        archivo_ubicaciones
    )

    df_ubicaciones["_orden_original"] = np.arange(
        len(df_ubicaciones)
    )

except Exception as error:

    st.error(
        f"No se pudo leer el historial GPS: {error}"
    )

    st.stop()


col_latitud = buscar_columna(
    df_ubicaciones,
    [
        "latitud",
        "latitude",
        "gps latitude",
        "posición latitud",
        "posicion latitud",
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
        "machine serial",
        "vin"
    ]
)


if col_latitud is None or col_longitud is None:

    st.error(
        "No se detectaron las columnas de latitud y longitud."
    )

    st.write(list(df_ubicaciones.columns))

    st.stop()


cantidad_gps_original = len(df_ubicaciones)

df_ubicaciones["_fecha_hora"] = construir_fecha_hora_gps(
    df_ubicaciones
)

df_ubicaciones["_latitud"] = convertir_numero(
    df_ubicaciones[col_latitud]
)

df_ubicaciones["_longitud"] = convertir_numero(
    df_ubicaciones[col_longitud]
)

cantidad_sin_fecha = int(
    df_ubicaciones["_fecha_hora"].isna().sum()
)

cantidad_sin_coordenadas = int(
    (
        df_ubicaciones["_latitud"].isna()
        | df_ubicaciones["_longitud"].isna()
    ).sum()
)

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

df_ubicaciones = df_ubicaciones[
    df_ubicaciones["_latitud"].between(-90, 90)
    & df_ubicaciones["_longitud"].between(-180, 180)
].copy()

df_ubicaciones = (
    df_ubicaciones
    .sort_values(
        by=[
            "_fecha_hora",
            "_orden_original"
        ],
        kind="stable"
    )
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


if df_ubicaciones.empty:

    st.error(
        "No quedaron registros GPS válidos."
    )

    st.stop()


# ============================================================
# LEER Y PROCESAR MOTOR
# ============================================================

try:

    df_motor_original = leer_csv_seguro(
        archivo_motor
    )

    (
        df_motor_largo,
        df_periodos_motor
    ) = procesar_motor(
        df_motor_original
    )

except Exception as error:

    st.error(
        f"No se pudo procesar el archivo del motor: {error}"
    )

    st.stop()


# ============================================================
# COMBINAR GPS Y MOTOR
# ============================================================

df_ubicaciones = asignar_motor_a_gps(
    df_gps=df_ubicaciones,
    df_periodos=df_periodos_motor
)

# Restaurar orden cronológico definitivo.
df_ubicaciones = (
    df_ubicaciones
    .sort_values(
        by=[
            "_fecha_hora",
            "_orden_original"
        ],
        kind="stable"
    )
    .reset_index(drop=True)
)

df_ubicaciones = calcular_variables_movimiento(
    df_ubicaciones
)

df_ubicaciones = crear_tramos(
    df=df_ubicaciones,
    max_gap_minutos=max_gap_minutos,
    max_velocidad_kmh=max_velocidad_kmh,
    max_salto_km=max_salto_km,
    max_salto_misma_hora_km=max_salto_misma_hora_km
)


# ============================================================
# EVENTOS DE REPOSO
# ============================================================

df_eventos_reposo = generar_eventos_reposo(
    df_periodos=df_periodos_motor,
    df_gps=df_ubicaciones,
    minimo_combustible_l=minimo_combustible_reposo_l,
    minimo_horas_reposo=minimo_horas_reposo
)


# ============================================================
# PERÍODO COMÚN
# ============================================================

fecha_inicio_gps = (
    df_ubicaciones["_fecha_hora"].min()
)

fecha_fin_gps = (
    df_ubicaciones["_fecha_hora"].max()
)

fecha_inicio_motor = (
    df_periodos_motor["_periodo_inicio"].min()
)

fecha_fin_motor = (
    df_periodos_motor["_periodo_fin"].max()
)

fecha_inicio_analisis = max(
    fecha_inicio_gps,
    fecha_inicio_motor
)

fecha_fin_analisis = min(
    fecha_fin_gps,
    fecha_fin_motor
)


if fecha_inicio_analisis > fecha_fin_analisis:

    st.error(
        "Los archivos GPS y motor no tienen un período temporal común."
    )

    st.stop()


df_periodos_analisis = df_periodos_motor[
    (
        df_periodos_motor["_periodo_fin"]
        >= fecha_inicio_analisis
    )
    & (
        df_periodos_motor["_periodo_inicio"]
        <= fecha_fin_analisis
    )
].copy()


# ============================================================
# INDICADORES
# ============================================================

numero_serie = "No identificado"

if col_serie is not None:

    valores_serie = (
        df_ubicaciones[col_serie]
        .dropna()
        .astype(str)
        .str.strip()
    )

    valores_serie = valores_serie[
        valores_serie.ne("")
    ].unique()

    if len(valores_serie) > 0:
        numero_serie = valores_serie[0]


combustible_total_l = (
    df_periodos_analisis[
        "_combustible_total_l"
    ]
    .fillna(0)
    .sum()
)

combustible_trabajando_l = (
    df_periodos_analisis[
        "_combustible_trabajando_l"
    ]
    .fillna(0)
    .sum()
)

combustible_reposo_l = (
    df_periodos_analisis[
        "_combustible_reposo_l"
    ]
    .fillna(0)
    .sum()
)

combustible_transporte_l = (
    df_periodos_analisis[
        "_combustible_transporte_l"
    ]
    .fillna(0)
    .sum()
)

distancia_motor_km = (
    df_periodos_analisis[
        "_distancia_periodo_km"
    ]
    .fillna(0)
    .sum()
)

distancia_trabajando_km = (
    df_periodos_analisis[
        "_distancia_trabajando_km"
    ]
    .fillna(0)
    .sum()
)

horas_reposo_total = (
    df_periodos_analisis[
        "_horas_reposo"
    ]
    .fillna(0)
    .sum()
)

distancia_gps_km = calcular_distancia_recorrido(
    df_ubicaciones
)


# KPI total en km/l.
rendimiento_total_km_l = (
    distancia_motor_km / combustible_total_l
    if combustible_total_l > 0.01
    else np.nan
)

# KPI trabajando en km/l.
rendimiento_trabajando_km_l = (
    distancia_trabajando_km
    / combustible_trabajando_l
    if combustible_trabajando_l > 0.01
    else np.nan
)

consumo_reposo_l_h = (
    combustible_reposo_l / horas_reposo_total
    if horas_reposo_total > 0
    else np.nan
)


with st.sidebar:

    st.markdown("---")

    st.subheader("📅 Período común analizado")

    st.write(
        "**Desde:**",
        fecha_inicio_analisis.strftime(
            "%d/%m/%Y %H:%M:%S"
        )
    )

    st.write(
        "**Hasta:**",
        fecha_fin_analisis.strftime(
            "%d/%m/%Y %H:%M:%S"
        )
    )


# ============================================================
# KPIS
# ============================================================

st.subheader("📊 Indicadores generales")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🚛 Unidad",
        numero_serie
    )

with col2:

    st.metric(
        "⛽ Combustible total",
        f"{formatear_numero(combustible_total_l, 1)} l"
    )

with col3:

    st.metric(
        "⚙️ Combustible trabajando",
        f"{formatear_numero(combustible_trabajando_l, 1)} l"
    )

with col4:

    st.metric(
        "🅿️ Combustible en reposo",
        f"{formatear_numero(combustible_reposo_l, 1)} l"
    )


col5, col6, col7, col8 = st.columns(4)

with col5:

    st.metric(
        "🚚 Combustible transporte",
        f"{formatear_numero(combustible_transporte_l, 1)} l"
    )

with col6:

    st.metric(
        "📈 Rendimiento trabajando",
        (
            f"{formatear_numero(rendimiento_trabajando_km_l, 2)} km/l"
            if pd.notna(rendimiento_trabajando_km_l)
            else "-"
        )
    )

with col7:

    st.metric(
        "⏸️ Consumo en reposo",
        (
            f"{formatear_numero(consumo_reposo_l_h, 2)} l/h"
            if pd.notna(consumo_reposo_l_h)
            else "-"
        )
    )

with col8:

    st.metric(
        "🛣️ Distancia motor",
        f"{formatear_numero(distancia_motor_km, 1)} km"
    )


with st.expander(
    "Ver indicadores complementarios"
):

    comp1, comp2, comp3, comp4 = st.columns(4)

    with comp1:

        st.metric(
            "Distancia GPS estimada",
            f"{formatear_numero(distancia_gps_km, 1)} km"
        )

    with comp2:

        st.metric(
            "Distancia trabajando estimada",
            f"{formatear_numero(distancia_trabajando_km, 1)} km"
        )

    with comp3:

        st.metric(
            "Rendimiento total",
            (
                f"{formatear_numero(rendimiento_total_km_l, 2)} km/l"
                if pd.notna(rendimiento_total_km_l)
                else "-"
            )
        )

    with comp4:

        st.metric(
            "Tramos GPS",
            int(df_ubicaciones["_tramo"].nunique())
        )


st.markdown("---")


# ============================================================
# ESCALA DE COLORES EN KM/L
# ============================================================

valores_rendimiento_validos = (
    df_periodos_analisis[
        "_rendimiento_trabajando_km_l"
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna()
)

# Filtrar extremos para evitar que un valor aislado
# distorsione toda la escala.
if not valores_rendimiento_validos.empty:

    limite_inferior = (
        valores_rendimiento_validos.quantile(0.01)
    )

    limite_superior = (
        valores_rendimiento_validos.quantile(0.99)
    )

    valores_rendimiento_validos = (
        valores_rendimiento_validos[
            (
                valores_rendimiento_validos
                >= limite_inferior
            )
            & (
                valores_rendimiento_validos
                <= limite_superior
            )
        ]
    )


if len(valores_rendimiento_validos) >= 2:

    rendimiento_minimo = float(
        valores_rendimiento_validos.quantile(0.05)
    )

    rendimiento_p25 = float(
        valores_rendimiento_validos.quantile(0.25)
    )

    rendimiento_p50 = float(
        valores_rendimiento_validos.quantile(0.50)
    )

    rendimiento_p75 = float(
        valores_rendimiento_validos.quantile(0.75)
    )

    rendimiento_maximo = float(
        valores_rendimiento_validos.quantile(0.95)
    )

    valores_indice = [
        rendimiento_minimo,
        rendimiento_p25,
        rendimiento_p50,
        rendimiento_p75,
        rendimiento_maximo
    ]

    # LinearColormap necesita valores estrictamente crecientes.
    for indice in range(1, len(valores_indice)):

        if valores_indice[indice] <= valores_indice[indice - 1]:

            valores_indice[indice] = (
                valores_indice[indice - 1]
                + 0.000001
            )

    # Menos km/l = rojo.
    # Más km/l = verde.
    escala_rendimiento = LinearColormap(
        colors=[
            "#DC2626",
            "#F97316",
            "#FACC15",
            "#84CC16",
            "#16A34A"
        ],
        index=valores_indice,
        vmin=valores_indice[0],
        vmax=valores_indice[-1],
        caption="Rendimiento trabajando, kilómetros por litro"
    )

else:

    escala_rendimiento = None


# ============================================================
# MAPA
# ============================================================

st.subheader(
    "📍 Recorrido y rendimiento de combustible"
)

st.caption(
    "Cada posición GPS se agrupa con el intervalo temporal "
    "correspondiente del archivo del motor. Los segmentos con "
    "mayor rendimiento en km/l aparecen verdes; los de menor "
    "rendimiento aparecen rojos."
)


mapa = folium.Map(
    location=[
        df_ubicaciones["_latitud"].mean(),
        df_ubicaciones["_longitud"].mean()
    ],
    zoom_start=7,
    control_scale=True,
    tiles="OpenStreetMap",
    prefer_canvas=True
)


# ============================================================
# DIBUJAR SEGMENTOS COLOREADOS
# ============================================================

segmentos_dibujados = 0
segmentos_descartados = 0
segmentos_con_rendimiento = 0
segmentos_sin_rendimiento = 0

descartados_por_tramo = 0
descartados_por_tiempo = 0
descartados_por_distancia = 0
descartados_por_velocidad = 0


for indice in range(1, len(df_ubicaciones)):

    anterior = df_ubicaciones.iloc[indice - 1]
    actual = df_ubicaciones.iloc[indice]

    delta_minutos = actual["_delta_min"]

    distancia_segmento_km = actual[
        "_distancia_anterior_km"
    ]

    velocidad_segmento_kmh = actual[
        "_velocidad_implicita_kmh"
    ]

    mismo_tramo = (
        anterior["_tramo"]
        == actual["_tramo"]
    )

    tiempo_valido = (
        pd.notna(delta_minutos)
        and delta_minutos >= 0
        and delta_minutos <= max_gap_minutos
    )

    distancia_valida = (
        pd.notna(distancia_segmento_km)
        and distancia_segmento_km <= max_salto_km
    )

    velocidad_valida = (
        pd.isna(velocidad_segmento_kmh)
        or velocidad_segmento_kmh <= max_velocidad_kmh
    )

    if not mismo_tramo:

        descartados_por_tramo += 1
        segmentos_descartados += 1
        continue

    if not tiempo_valido:

        descartados_por_tiempo += 1
        segmentos_descartados += 1
        continue

    if not distancia_valida:

        descartados_por_distancia += 1
        segmentos_descartados += 1
        continue

    if not velocidad_valida:

        descartados_por_velocidad += 1
        segmentos_descartados += 1
        continue

    rendimiento_segmento = actual[
        "_rendimiento_trabajando_km_l"
    ]

    if (
        escala_rendimiento is not None
        and pd.notna(rendimiento_segmento)
        and rendimiento_segmento > 0
    ):

        rendimiento_para_color = min(
            max(
                float(rendimiento_segmento),
                escala_rendimiento.vmin
            ),
            escala_rendimiento.vmax
        )

        color_segmento = escala_rendimiento(
            rendimiento_para_color
        )

        rendimiento_texto = (
            f"{formatear_numero(rendimiento_segmento, 2)} km/l"
        )

        segmentos_con_rendimiento += 1

    else:

        color_segmento = "#9CA3AF"
        rendimiento_texto = "Sin dato comparable"

        segmentos_sin_rendimiento += 1

    inicio_periodo = actual["_periodo_inicio"]
    fin_periodo = actual["_periodo_fin"]

    if (
        pd.notna(inicio_periodo)
        and pd.notna(fin_periodo)
    ):

        texto_intervalo = (
            f"{inicio_periodo.strftime('%d/%m/%Y %H:%M')}"
            f" a "
            f"{fin_periodo.strftime('%H:%M')}"
        )

    else:

        texto_intervalo = (
            "Sin intervalo de motor asociado"
        )

    popup_segmento = (
        "<b>Rendimiento durante el trabajo</b>"
        f"<br>Fecha GPS: "
        f"{actual['_fecha_hora'].strftime('%d/%m/%Y %H:%M:%S')}"
        f"<br>Intervalo motor: {texto_intervalo}"
        f"<br>Rendimiento: {rendimiento_texto}"
        f"<br>Combustible trabajando: "
        f"{formatear_numero(actual['_combustible_trabajando_l'], 2)} l"
        f"<br>Distancia trabajando estimada: "
        f"{formatear_numero(actual['_distancia_trabajando_km'], 2)} km"
        f"<br>Distancia GPS del segmento: "
        f"{formatear_numero(distancia_segmento_km, 3)} km"
        f"<br>Tiempo entre posiciones: "
        f"{formatear_numero(delta_minutos, 2)} min"
        f"<br>Velocidad implícita: "
        f"{formatear_numero(velocidad_segmento_kmh, 1)} km/h"
        f"<br>Tramo GPS: {int(actual['_tramo'])}"
    )

    folium.PolyLine(
        locations=[
            [
                anterior["_latitud"],
                anterior["_longitud"]
            ],
            [
                actual["_latitud"],
                actual["_longitud"]
            ]
        ],
        color=color_segmento,
        weight=5,
        opacity=0.92,
        tooltip=(
            f"Rendimiento trabajando: {rendimiento_texto}"
        ),
        popup=folium.Popup(
            popup_segmento,
            max_width=380
        )
    ).add_to(mapa)

    segmentos_dibujados += 1


if escala_rendimiento is not None:

    escala_rendimiento.add_to(mapa)


# ============================================================
# EVENTOS DE REPOSO
# ============================================================

if not df_eventos_reposo.empty:

    grupo_reposo = folium.FeatureGroup(
        name="Eventos de reposo",
        show=True
    )

    for _, evento in df_eventos_reposo.iterrows():

        litros_reposo = evento[
            "_combustible_reposo_l"
        ]

        horas_reposo = evento[
            "_horas_reposo"
        ]

        consumo_reposo = evento[
            "_consumo_reposo_l_h"
        ]

        popup_reposo = (
            "<b>🅿️ Evento de reposo</b>"
            f"<br>Inicio: "
            f"{evento['_periodo_inicio'].strftime('%d/%m/%Y %H:%M')}"
            f"<br>Fin: "
            f"{evento['_periodo_fin'].strftime('%d/%m/%Y %H:%M')}"
            f"<br>Tiempo en reposo: "
            f"{formatear_numero(horas_reposo, 2)} h"
            f"<br>Combustible consumido: "
            f"{formatear_numero(litros_reposo, 2)} l"
            f"<br>Consumo promedio: "
            f"{formatear_numero(consumo_reposo, 2)} l/h"
        )

        radio = 6

        if pd.notna(litros_reposo):

            radio = min(
                14,
                max(
                    6,
                    6 + float(litros_reposo)
                )
            )

        folium.CircleMarker(
            location=[
                evento["_latitud"],
                evento["_longitud"]
            ],
            radius=radio,
            color="#6D28D9",
            weight=2,
            fill=True,
            fill_color="#8B5CF6",
            fill_opacity=0.85,
            tooltip=(
                f"Reposo: "
                f"{formatear_numero(litros_reposo, 2)} l"
            ),
            popup=folium.Popup(
                popup_reposo,
                max_width=330
            )
        ).add_to(grupo_reposo)

    grupo_reposo.add_to(mapa)


# ============================================================
# MARCADORES DE INICIO Y FIN
# ============================================================

inicio = df_ubicaciones.iloc[0]
fin = df_ubicaciones.iloc[-1]

folium.Marker(
    location=[
        inicio["_latitud"],
        inicio["_longitud"]
    ],
    tooltip="Inicio del historial",
    popup=folium.Popup(
        (
            "<b>Inicio del historial</b>"
            f"<br>{inicio['_fecha_hora'].strftime('%d/%m/%Y %H:%M:%S')}"
        ),
        max_width=280
    ),
    icon=folium.Icon(
        color="green",
        icon="play",
        prefix="fa"
    )
).add_to(mapa)

folium.Marker(
    location=[
        fin["_latitud"],
        fin["_longitud"]
    ],
    tooltip="Fin del historial",
    popup=folium.Popup(
        (
            "<b>Fin del historial</b>"
            f"<br>{fin['_fecha_hora'].strftime('%d/%m/%Y %H:%M:%S')}"
        ),
        max_width=280
    ),
    icon=folium.Icon(
        color="red",
        icon="stop",
        prefix="fa"
    )
).add_to(mapa)


# ============================================================
# MARCADORES DE NUEVO TRAMO
# ============================================================

for numero_tramo, df_tramo in df_ubicaciones.groupby(
    "_tramo",
    sort=True
):

    if int(numero_tramo) == 1:
        continue

    primer_punto = df_tramo.iloc[0]

    folium.CircleMarker(
        location=[
            primer_punto["_latitud"],
            primer_punto["_longitud"]
        ],
        radius=5,
        color="#F59E0B",
        fill=True,
        fill_color="#F59E0B",
        fill_opacity=1,
        tooltip=f"Inicio del tramo {numero_tramo}",
        popup=folium.Popup(
            (
                f"<b>Inicio del tramo {numero_tramo}</b>"
                f"<br>{primer_punto['_fecha_hora'].strftime('%d/%m/%Y %H:%M:%S')}"
                f"<br>Motivo: {primer_punto['_motivo_corte']}"
            ),
            max_width=320
        )
    ).add_to(mapa)


# ============================================================
# AJUSTAR Y MOSTRAR MAPA
# ============================================================

mapa.fit_bounds(
    [
        [
            df_ubicaciones["_latitud"].min(),
            df_ubicaciones["_longitud"].min()
        ],
        [
            df_ubicaciones["_latitud"].max(),
            df_ubicaciones["_longitud"].max()
        ]
    ],
    padding=[
        20,
        20
    ]
)

folium.LayerControl(
    collapsed=False
).add_to(mapa)

st_folium(
    mapa,
    width=None,
    height=680,
    returned_objects=[]
)


# ============================================================
# CONTROL DE SEGMENTOS
# ============================================================

control1, control2, control3, control4 = st.columns(4)

with control1:

    st.metric(
        "Segmentos analizados",
        max(len(df_ubicaciones) - 1, 0)
    )

with control2:

    st.metric(
        "Segmentos coloreados",
        segmentos_con_rendimiento
    )

with control3:

    st.metric(
        "Segmentos sin rendimiento",
        segmentos_sin_rendimiento
    )

with control4:

    st.metric(
        "Segmentos descartados",
        segmentos_descartados
    )


with st.expander(
    "Ver motivos de descarte"
):

    motivo1, motivo2, motivo3, motivo4 = st.columns(4)

    with motivo1:

        st.metric(
            "Cambio de tramo",
            descartados_por_tramo
        )

    with motivo2:

        st.metric(
            "Tiempo inválido",
            descartados_por_tiempo
        )

    with motivo3:

        st.metric(
            "Distancia excesiva",
            descartados_por_distancia
        )

    with motivo4:

        st.metric(
            "Velocidad excesiva",
            descartados_por_velocidad
        )


# ============================================================
# REFERENCIAS
# ============================================================

ref1, ref2, ref3, ref4, ref5 = st.columns(5)

with ref1:
    st.markdown("🟢 **Mayor km/l**")

with ref2:
    st.markdown("🔴 **Menor km/l**")

with ref3:
    st.markdown("🟣 **Reposo**")

with ref4:
    st.markdown("🟠 **Nuevo tramo**")

with ref5:
    st.markdown("⚪ **Sin dato del motor**")


# ============================================================
# DETALLE DE EVENTOS DE REPOSO
# ============================================================

st.markdown("---")

st.subheader("🅿️ Detalle de eventos de reposo")

if df_eventos_reposo.empty:

    st.info(
        "No se detectaron eventos de reposo que superen "
        "los límites configurados."
    )

else:

    tabla_reposo = df_eventos_reposo[
        [
            "_periodo_inicio",
            "_periodo_fin",
            "_horas_reposo",
            "_combustible_reposo_l",
            "_consumo_reposo_l_h",
            "_latitud",
            "_longitud"
        ]
    ].copy()

    tabla_reposo = tabla_reposo.rename(
        columns={
            "_periodo_inicio": "Inicio",
            "_periodo_fin": "Fin",
            "_horas_reposo": "Horas en reposo",
            "_combustible_reposo_l": "Combustible en reposo, l",
            "_consumo_reposo_l_h": "Consumo en reposo, l/h",
            "_latitud": "Latitud",
            "_longitud": "Longitud"
        }
    )

    st.dataframe(
        tabla_reposo,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Inicio": st.column_config.DatetimeColumn(
                format="DD/MM/YYYY HH:mm"
            ),
            "Fin": st.column_config.DatetimeColumn(
                format="DD/MM/YYYY HH:mm"
            ),
            "Horas en reposo": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Combustible en reposo, l": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Consumo en reposo, l/h": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Latitud": st.column_config.NumberColumn(
                format="%.6f"
            ),
            "Longitud": st.column_config.NumberColumn(
                format="%.6f"
            )
        }
    )


# ============================================================
# CONTROL TÉCNICO
# ============================================================

with st.expander(
    "🔎 Ver control técnico de la combinación"
):

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "GPS originales",
            cantidad_gps_original
        )

    with c2:

        st.metric(
            "GPS válidos",
            len(df_ubicaciones)
        )

    with c3:

        st.metric(
            "Sin fecha",
            cantidad_sin_fecha
        )

    with c4:

        st.metric(
            "Sin coordenadas",
            cantidad_sin_coordenadas
        )

    puntos_con_motor = int(
        df_ubicaciones["_periodo_inicio"]
        .notna()
        .sum()
    )

    puntos_con_rendimiento = int(
        df_ubicaciones[
            "_rendimiento_trabajando_km_l"
        ]
        .notna()
        .sum()
    )

    porcentaje_cruce = (
        puntos_con_motor
        / len(df_ubicaciones)
        * 100
    )

    porcentaje_rendimiento = (
        puntos_con_rendimiento
        / len(df_ubicaciones)
        * 100
    )

    control_cruce1, control_cruce2 = st.columns(2)

    with control_cruce1:

        st.metric(
            "Puntos GPS con intervalo del motor",
            (
                f"{puntos_con_motor} de {len(df_ubicaciones)} "
                f"({formatear_numero(porcentaje_cruce, 1)} %)"
            )
        )

    with control_cruce2:

        st.metric(
            "Puntos GPS con km/l válido",
            (
                f"{puntos_con_rendimiento} de {len(df_ubicaciones)} "
                f"({formatear_numero(porcentaje_rendimiento, 1)} %)"
            )
        )

    st.write("### Control del recorrido combinado")

    columnas_control_gps = [
        "_orden_original",
        "_fecha_hora",
        "_latitud",
        "_longitud",
        "_delta_min",
        "_distancia_anterior_km",
        "_velocidad_implicita_kmh",
        "_tramo",
        "_periodo_inicio",
        "_periodo_fin",
        "_combustible_trabajando_l",
        "_distancia_trabajando_km",
        "_rendimiento_trabajando_km_l"
    ]

    st.dataframe(
        df_ubicaciones[
            columnas_control_gps
        ].head(500),
        use_container_width=True,
        hide_index=True
    )

    st.write("### Intervalos temporales del motor")

    columnas_control_motor = [
        "_periodo_inicio",
        "_periodo_fin",
        "_combustible_total_l",
        "_combustible_trabajando_l",
        "_combustible_reposo_l",
        "_combustible_transporte_l",
        "_horas_trabajando",
        "_horas_reposo",
        "_velocidad_trabajando_kmh",
        "_distancia_periodo_km",
        "_distancia_trabajando_km",
        "_rendimiento_trabajando_km_l",
        "_rendimiento_total_km_l",
        "_consumo_reposo_l_h"
    ]

    st.dataframe(
        df_periodos_motor[
            columnas_control_motor
        ],
        use_container_width=True,
        hide_index=True
    )
