# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:50:32 2026

@author: Enzo
"""

###############################################################################
## PROYECTO OPENALEX
###############################################################################

###############################################################################
# OBJETIVO: CONSTRUIR UN SISTEMA DE GESTIÓN DE BASES DE DATOS RELACIONAL
# USANDO DATOS ALMACENADOS EN LA BASE DE DATOS DE NATURALEZA OPEN SOURCE OPENALEX
##############################################################################

import requests
import pandas as pd
import numpy as np
import io
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sqlalchemy import create_engine
import time
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
from datetime import date

###############################################################################
# Estrategia para obtener los works del Perú usando la paginación compleja
# Se genera un filtro desde el url, considerando solo los works del Perú
# El presente url presenta todos los works generados en donde hay participación de investigadores peruanos 
###############################################################################

API_KEY = "v9Ys1yuVJl3kNJYVql74Ip"          # <- pon tu key aquí
MAILTO  = "enzo.lpge@gmail.com"  # <- opcional pero recomendado

cursor = "*"
trabajo_peru = []

session = requests.Session()

MAX_RETRIES = 8

while cursor:
    params = {
        "filter": "authorships.institutions.country_code:PE",
        "per_page": 200,
        "cursor": cursor,
        "api_key": API_KEY,
        "mailto": MAILTO,
        # opcional: reduce tamaño por request
        # "select": "id,doi,display_name,publication_year,type,cited_by_count,authorships"
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get("https://api.openalex.org/works", params=params, timeout=90)

            # rate limit
            if r.status_code == 429:
                time.sleep(2 + attempt)
                continue

            # errores temporales del servidor
            if r.status_code in (500, 502, 503, 504):
                time.sleep(2 + attempt)
                continue

            # otros errores: muestra causa real y corta
            if r.status_code != 200:
                print("STATUS:", r.status_code)
                print("URL:", r.url)
                print("BODY:", r.text[:400])
                r.raise_for_status()

            data = r.json()
            results = data.get("results", [])
            trabajo_peru.extend(results)
            cursor = data.get("meta", {}).get("next_cursor")

            break  # éxito, salimos del retry loop

        except (ChunkedEncodingError, ConnectionError, ReadTimeout) as e:
            # backoff progresivo y reintento
            if attempt == MAX_RETRIES - 1:
                raise  # ya no quedan reintentos
            time.sleep(2 + attempt)

    # pausa leve (reduce probabilidad de cortes)
    time.sleep(0.1)

print("Total works descargados:", len(trabajo_peru))


total_works = len(trabajo_peru)

works_100_authors = 0
works_50plus_authors = 0

for work in trabajo_peru:
    
    authorships = work.get("authorships", [])
    n_authors = len(authorships)
    
    if n_authors == 100:
        works_100_authors += 1
        
    if n_authors >= 50:
        works_50plus_authors += 1


print("Total works:", total_works)
print("Works con exactamente 100 autores (posible truncamiento):", works_100_authors)
print("Works con >=50 autores:", works_50plus_authors)

print("Porcentaje posible truncamiento:",
      round(works_100_authors / total_works * 100, 3), "%")


###############################################################################
# Estrategia para obtener los autores peruanos usando la paginación compleja
# Se genera un filtro desde el url, considerando solo los autores peruanos
# El presente url presenta todos los autores cuyo país es Perú
###############################################################################

# Esta línea de código, me permite identificar si mi conexión con la APi está activa o no

API_KEY = "v9Ys1yuVJl3kNJYVql74Ip"          # <- pon tu key aquí
MAILTO  = "enzo.lpge@gmail.com"  # <- opcional pero recomendado

params = {
    "per_page": 1,
    "api_key": API_KEY,
    "mailto": MAILTO,
}

r = requests.get("https://api.openalex.org/authors", params=params)

print("STATUS:", r.status_code)
print("BODY:", r.text)

# Esta parte de código me permite identificar cual de los filtros debo considerar

def test_filter(filtro):
    params = {
        "filter": filtro,
        "per_page": 1,
        "api_key": API_KEY,
        "mailto": MAILTO,
    }
    r = requests.get("https://api.openalex.org/authors", params=params, timeout=60)
    print("\nFILTER:", filtro)
    print("STATUS:", r.status_code)
    print("URL:", r.url)
    print("BODY:", r.text[:300])  # muestra el mensaje exacto del servidor

filters = [
    "last_known_institution.country_code:PE",
    "last_known_institutions.country_code:PE",
    "affiliations.institution.country_code:PE",
    "last_known_institutions.id:I4210108322",  # prueba de estructura (ejemplo)
]

for f in filters:
    test_filter(f)

# Script para descargar todos los investigadores con afiliación Perú

API_KEY = "v9Ys1yuVJl3kNJYVql74Ip"          # <- pon tu key aquí
MAILTO  = "enzo.lpge@gmail.com"

session = requests.Session()

cursor = "*"
autores_peru = []

while cursor:
    params = {
        "filter": "last_known_institutions.country_code:PE",
        "per_page": 200,
        "cursor": cursor,
        "api_key": API_KEY,
        "mailto": MAILTO
    }

    r = session.get("https://api.openalex.org/authors", params=params, timeout=60)

    if r.status_code != 200:
        print("STATUS:", r.status_code)
        print("URL:", r.url)
        print("BODY:", r.text[:400])
        r.raise_for_status()

    data = r.json()
    autores_peru.extend(data.get("results", []))
    cursor = data.get("meta", {}).get("next_cursor")

    time.sleep(0.1)

print("Autores descargados:", len(autores_peru))


###############################################################################
# Se calcula los keys de la lista trabajo_peru
###############################################################################
type(trabajo_peru)

keys_unicos = set()
for trabajo in trabajo_peru:
    keys_unicos.update(trabajo.keys())

print(keys_unicos)


###############################################################################
# Se identifica los keys de la lista autores_peru
###############################################################################
type(autores_peru)

llaves = set()
for autor in autores_peru:
    llaves.update(autor.keys())

print(llaves)
type(llaves)


###############################################################################
# Se construye un dataframe que contiene cuatro campos relacionados con las
# publicaciones
###############################################################################

# Se construye la lista publicacion que contiene atributos de las publicaciones científicas
publicacion = ["id", "doi", "title", "publication_year"]


# Se almacena en un dataframe las publicaciones de la lista publicacion 
df_publicacion1 = pd.DataFrame.from_records(trabajo_peru, columns=publicacion)
df_publicacion1["id"].nunique()
df_publicacion1 = df_publicacion1.drop_duplicates(subset=["id"])

# Función para extraer el DOI, devolviendo None en caso de error
def extraer_doi(url):
    try:
        # Dividimos la cadena por "https://doi.org/" y tomamos la segunda parte
        parts = url.split("https://doi.org/")
        if len(parts) > 1:
            return parts[1]
        else:
            return None  # Devuelve None si no se encuentra el DOI
    except Exception:
        return None  # Devuelve None en caso de cualquier otro error


# Aplicamos la función a la columnda doi del dataframe df_publicacion1
df_publicacion1['doi'] = df_publicacion1['doi'].apply(extraer_doi)

# Se cuenta la cantidad de publicaciones científicas únicas
df_publicacion1["doi"].nunique()


# Ahora se condensa en un dataframe el id de la publicación con los ids
publicacion1 = ["id", "ids"]
df_publicacion_codigo = pd.DataFrame.from_records(trabajo_peru, columns=publicacion1)
df_publicacion_codigo = df_publicacion_codigo.drop_duplicates(subset=["id"])


# Se analiza el tipo de dato que es la columna ids
print(df_publicacion_codigo["ids"][50])
type(df_publicacion_codigo["ids"][10])

df_publicacion_codigo['openalex'] = df_publicacion_codigo["ids"].apply(lambda x: x.get('openalex', False) if isinstance(x, dict) else False)
df_publicacion_codigo['mag'] = df_publicacion_codigo["ids"].apply(lambda x: x.get('mag', False) if isinstance(x, dict) else False)
df_publicacion_codigo['pmid'] = df_publicacion_codigo["ids"].apply(lambda x: x.get('pmid', False) if isinstance(x, dict) else False)

# Se construye una función para quedarnos solo con el código pmid para cada publicación

def extraer_pmid(url):
    try:
        # Dividimos la cadena por "https://pubmed.ncbi.nlm.nih.gov/" y tomamos la segunda parte
        parts = url.split("https://pubmed.ncbi.nlm.nih.gov/")
        if len(parts) > 1:
            return parts[1]
        else:
            return None  # Devuelve None si no se encuentra el DOI
    except Exception:
        return None  # Devuelve None en caso de cualquier otro error


# Aplicamos la función a la columnda doi del dataframe df_publicacion1
df_publicacion_codigo['pmid'] = df_publicacion_codigo['pmid'].apply(extraer_pmid)


df_publicacion_codigo = df_publicacion_codigo[["id", "mag", "pmid"]]

# Se integran los dataframes df_publicacion y df_publicacion_codigo
publicaciones = pd.merge(df_publicacion1, df_publicacion_codigo, on="id")
publicaciones["id"].nunique()

publicaciones = publicaciones.drop_duplicates(subset=["id"])
publicaciones.columns

# Se organiza la información contenida en la tabla publicaciones
publicaciones = publicaciones[["id", "doi", "mag", "pmid", "title", "publication_year"]]
publicaciones.dtypes

# Convertimos la columna 'publication_year' a datetime, asignando 01-01 como día y mes por defecto
publicaciones['publication_year'] = pd.to_datetime(publicaciones['publication_year'], format='%Y')

# Si solo quieres mantener el año visible, no se puede almacenar directamente un solo año como datetime
# pero puedes trabajar con el 'datetime' y luego extraer el año si es necesario
publicaciones['publication_year'] = publicaciones['publication_year'].dt.year
publicaciones.dtypes


###############################################################################
# SE obtiene información adicional de las publicaciones científicas
###############################################################################
detalle = ["id", "type", "language","biblio","is_retracted","open_access","cited_by_count","institutions_distinct_count", "countries_distinct_count"]
df_detalle = pd.DataFrame.from_records(trabajo_peru, columns=detalle)
df_detalle = df_detalle.drop_duplicates(subset=["id"])


# Se analiza el tipo de dato del atributo biblio del dataframe df_detalle
print(df_detalle["biblio"][100])

df_detalle["volume"] = df_detalle["biblio"].apply(lambda x: x.get('volume', False) if isinstance(x, dict) else False)
df_detalle["issue"] = df_detalle["biblio"].apply(lambda x: x.get('issue', False) if isinstance(x, dict) else False)
df_detalle["first_page"] = df_detalle["biblio"].apply(lambda x: x.get('first_page', False) if isinstance(x, dict) else False)
df_detalle["last_page"] = df_detalle["biblio"].apply(lambda x: x.get('last_page', False) if isinstance(x, dict) else False)
df_detalle["acceso"] = df_detalle["open_access"].apply(lambda x: x.get('is_oa', False) if isinstance(x, dict) else False)


# Se eliminan atributos del dataframe df_detalle
del df_detalle["biblio"]
del df_detalle["open_access"]

###############################################################################
# Se construye la tabla Publicaciones
###############################################################################

# Se realiza una fusión entre publicaciones y df_detalle
tbl_publicaciones = pd.merge(publicaciones, df_detalle, on="id")
# Se analiza la cantidad de atributos que contiene el dataframe tabla_publicaciones
tbl_publicaciones.shape
tbl_publicaciones.columns
tbl_publicaciones["id"].nunique()
# Se renombre un campo de la tabla tbl_publicaciones
tbl_publicaciones.rename(columns=({"id":"publication_id"}), inplace=True)


# Se comprende la estructura (Data profiling) del dataframe tbl_publicaciones
tbl_publicaciones.info()
tbl_publicaciones.describe()
tbl_publicaciones.describe(include="all")
tbl_publicaciones.nunique()
tbl_publicaciones.head(10)
tbl_publicaciones.isna().sum()
tbl_publicaciones.columns

# Se construye un reporte de validación semántica automatizado para el dataframe tbl_publicaciones

def validar_semantica_tbl_publicaciones(df: pd.DataFrame) -> dict:
    """
    Etapa 2: Validación semántica (NO limpia, NO modifica).
    Devuelve conteos de violaciones + ejemplos de filas problemáticas.
    Columnas esperadas:
    publication_id, doi, mag, pmid, title, publication_year, type, language,
    is_retracted, cited_by_count, institutions_distinct_count, countries_distinct_count,
    volume, issue, first_page, last_page, acceso
    """
    required = [
        "publication_id","doi","mag","pmid","title","publication_year","type","language",
        "is_retracted","cited_by_count","institutions_distinct_count","countries_distinct_count",
        "volume","issue","first_page","last_page","acceso"
    ]
    missing_cols = [c for c in required if c not in df.columns]

    out = {
        "missing_columns": missing_cols,
        "n_rows": int(len(df)),
        "violations": {},
        "samples": {}
    }
    if missing_cols:
        return out

    d = df.copy()

    # Helpers
    def s(col):  # string normalized view
        return d[col].astype("string").str.strip()

    def sample(mask, cols=None, n=8):
        cols = cols or required
        return d.loc[mask, cols].head(n)

    # 1) publication_id: no vacío + unicidad
    pid = s("publication_id")
    mask_pid_empty = pid.isna() | (pid == "")
    out["violations"]["publication_id_empty"] = int(mask_pid_empty.sum())
    out["samples"]["publication_id_empty"] = sample(mask_pid_empty, ["publication_id","title","publication_year"])

    mask_pid_dup = d["publication_id"].duplicated(keep=False) & (~mask_pid_empty)
    out["violations"]["publication_id_duplicates"] = int(mask_pid_dup.sum())
    out["samples"]["publication_id_duplicates"] = sample(mask_pid_dup, ["publication_id","doi","title","publication_year"])

    # 2) publication_year: rango plausible
    year = pd.to_numeric(d["publication_year"], errors="coerce")
    current_year = pd.Timestamp.today().year
    mask_year_bad = year.isna() | (year < 1800) | (year > current_year + 1)
    out["violations"]["publication_year_invalid"] = int(mask_year_bad.sum())
    out["samples"]["publication_year_invalid"] = sample(mask_year_bad, ["publication_id","publication_year","title"])

    # 3) cited_by_count: numérico, no negativo
    cbc = pd.to_numeric(d["cited_by_count"], errors="coerce")
    mask_cbc_bad = cbc.isna() | (cbc < 0)
    out["violations"]["cited_by_count_invalid"] = int(mask_cbc_bad.sum())
    out["samples"]["cited_by_count_invalid"] = sample(mask_cbc_bad, ["publication_id","cited_by_count","title"])

    # 4) institutions_distinct_count / countries_distinct_count: numérico, no negativo
    for col in ["institutions_distinct_count","countries_distinct_count"]:
        x = pd.to_numeric(d[col], errors="coerce")
        mask_bad = x.isna() | (x < 0)
        out["violations"][f"{col}_invalid"] = int(mask_bad.sum())
        out["samples"][f"{col}_invalid"] = sample(mask_bad, ["publication_id",col,"title"])

    # 5) is_retracted / acceso: binarios 0/1 (o booleanos)
    for col in ["is_retracted","acceso"]:
        x = d[col]
        # admite bool o 0/1
        if x.dtype == bool:
            mask_bad = pd.Series([False]*len(d), index=d.index)
        else:
            xn = pd.to_numeric(x, errors="coerce")
            mask_bad = xn.isna() | (~xn.isin([0,1]))
        out["violations"][f"{col}_invalid_not_binary"] = int(mask_bad.sum())
        out["samples"][f"{col}_invalid_not_binary"] = sample(mask_bad, ["publication_id",col,"title"])

    # 6) type: dominio permitido (ajústalo si quieres más)
    allowed_types = {"article","review","book-chapter","book"}
    typ = s("type").str.lower()
    mask_type_bad = typ.isna() | (typ == "") | (~typ.isin(allowed_types))
    out["violations"]["type_invalid_outside_domain"] = int(mask_type_bad.sum())
    out["samples"]["type_invalid_outside_domain"] = sample(mask_type_bad, ["publication_id","type","title","publication_year"])
    out["violations"]["type_distribution_top20"] = typ.value_counts(dropna=False).head(20).to_dict()

    # 7) language: ISO-639-1 (2 letras) o vacío/NA
    lang = s("language").str.lower()
    mask_lang_bad = ~(lang.str.fullmatch(r"[a-z]{2}").fillna(False) | lang.isna() | (lang == ""))
    out["violations"]["language_invalid_not_iso2"] = int(mask_lang_bad.sum())
    out["samples"]["language_invalid_not_iso2"] = sample(mask_lang_bad, ["publication_id","language","title"])

    # 8) doi: forma básica (10.xxxx/...) o doi.org/10...
    doi = s("doi").str.lower()
    looks_like_doi = doi.str.contains(r"(https?://doi\.org/)?10\.\d{4,9}/", regex=True, na=False)
    mask_doi_bad = ~(looks_like_doi | doi.isna() | (doi == ""))
    out["violations"]["doi_invalid_format"] = int(mask_doi_bad.sum())
    out["samples"]["doi_invalid_format"] = sample(mask_doi_bad, ["publication_id","doi","title"])

    # 9) mag / pmid: suelen ser IDs; aceptamos vacío/NA o dígitos (string numérico)
    for col in ["mag","pmid"]:
        x = s(col)
        mask_bad = ~(
            x.isna() | (x == "") | x.str.fullmatch(r"\d+").fillna(False)
        )
        out["violations"][f"{col}_invalid_not_numeric_string"] = int(mask_bad.sum())
        out["samples"][f"{col}_invalid_not_numeric_string"] = sample(mask_bad, ["publication_id",col,"title"])

    # 10) title: no vacío (si tu modelo exige título)
    title = s("title")
    mask_title_empty = title.isna() | (title == "")
    out["violations"]["title_empty"] = int(mask_title_empty.sum())
    out["samples"]["title_empty"] = sample(mask_title_empty, ["publication_id","title","doi","publication_year"])

    # 11) volume/issue/first_page/last_page: etiquetas; validación ligera
    # 11.1 páginas coherentes si ambas son numéricas
    fp = s("first_page")
    lp = s("last_page")
    fp_num = pd.to_numeric(fp, errors="coerce")
    lp_num = pd.to_numeric(lp, errors="coerce")
    comparable = fp_num.notna() & lp_num.notna()
    mask_pages_bad = comparable & (lp_num < fp_num)
    out["violations"]["pages_inconsistent_last_lt_first"] = int(mask_pages_bad.sum())
    out["samples"]["pages_inconsistent_last_lt_first"] = sample(mask_pages_bad, ["publication_id","first_page","last_page","title"])

    return out


# --- USO ---
rep = validar_semantica_tbl_publicaciones(tbl_publicaciones)
rep["violations"]
rep["samples"]["publication_year_invalid"]

# Se realiza un esquema de limpieza de los datos del dataframe tbl_publicaciones
tbl_publicaciones = tbl_publicaciones.copy()

# eliminar publication_id vacío
tbl_publicaciones = tbl_publicaciones[tbl_publicaciones["publication_id"].notna()]

# eliminar duplicados por clave
tbl_publicaciones = tbl_publicaciones.drop_duplicates(subset=["publication_id"])

# corregir tipos
tbl_publicaciones["publication_year"] = pd.to_numeric(tbl_publicaciones["publication_year"], errors="coerce")
tbl_publicaciones = tbl_publicaciones[tbl_publicaciones["publication_year"].between(1800, 2026)]

tbl_publicaciones["cited_by_count"] = pd.to_numeric(tbl_publicaciones["cited_by_count"], errors="coerce").fillna(0)

# Se vuelve a utilizar el reporte de validación semántica
rep = validar_semantica_tbl_publicaciones(tbl_publicaciones)
rep["violations"]


tbl_publicaciones["is_retracted"] = (
    tbl_publicaciones["is_retracted"]
    .astype(str)
    .str.lower()
    .map({"true": 1, "false": 0})
)

tbl_publicaciones["acceso"] = (
    tbl_publicaciones["acceso"]
    .astype(str)
    .str.lower()
    .map({"true": 1, "false": 0})
)


# Una vez analizada la estructura de mi dataframe, establecido un reporte de validación semántica, y limpiados los datos
# se procede a tipear los datos
schema = {"publication_id":"string",
          "doi":"string",
          "mag":"string",
          "pmid":"string",
          "title":"string",
          "publication_year":"Int64",
          "type":"string",
          "language":"string",
          "is_retracted":"Int64",
          "cited_by_count":"Int64",
          "institutions_distinct_count":"Int64",
          "countries_distinct_count":"Int64",
          "volume":"string",
          "issue":"string",
          "first_page":"string",
          "last_page":"string",
          "acceso":"Int64"
          }


tbl_publicaciones = tbl_publicaciones.astype(schema)
tbl_publicaciones.info()
tbl_publicaciones.describe()
tbl_publicaciones.shape

# Se analiza un caso particular para validar la descarga
caso = tbl_publicaciones[tbl_publicaciones["publication_id"]=="https://openalex.org/W2073840323"]
caso = tbl_publicaciones[tbl_publicaciones["publication_id"]=="https://openalex.org/W1234567890"]

###############################################################################
# Se construye la tabla ODS
###############################################################################

ods = ["id", "sustainable_development_goals"]
# Se almacena en un dataframe el código de las publicaciones con sus codigos ods 
df_ods = pd.DataFrame.from_records(trabajo_peru, columns=ods)
df_ods = df_ods.drop_duplicates(subset=["id"])

# Eliminar filas del dataframe df_ods donde la columna tiene listas vacías
df_ods = df_ods[df_ods["sustainable_development_goals"].map(len) > 0]

# Se arregla el indice con el propósito de manipular la información
df_ods.reset_index(inplace=True)
del df_ods["index"]

# Se analiza el tipo de dato de un valor típico de la fila de la columna sustainable_development_goals
ods_dic = df_ods["sustainable_development_goals"][500]
print(ods_dic) 
type(ods_dic)

# Listas para almacenar los datos de los autores y las afiliaciones
goals = []

# Iterar sobre cada elemento de df_ods1["sustainable_development_goals"] 
for i in range(len(df_ods["sustainable_development_goals"])):
    try:
        ods_dic = df_ods["sustainable_development_goals"][i]        
        # Iterar sobre cada elemento de la lista de diccionarios
        for goal in ods_dic:
            # Datos del autor
            goal_info = {
                'publication_id': df_ods['id'][i],
                'goal_id': goal.get("id", ""),
                "display_name": goal.get("display_name", ""),
                "score": goal.get("score", "")
            }
            goals.append(goal_info)
            
    except (KeyError, ValueError, SyntaxError, IndexError) as e:
        print("Error al procesar el índice", i, ":", e)

# Crear DataFrame consolidado
base_ods = pd.DataFrame(goals)
base_ods.columns

# Se analiza la estructura del dataframe base_ods
base_ods.info()
base_ods.shape
base_ods.dtypes
base_ods.describe()
base_ods.nunique()
base_ods.head(10)
base_ods.columns

# Se analiza la presencia de duplicados
base_ods[["publication_id", "goal_id"]].nunique()

duplicados = base_ods.duplicated(subset=["publication_id", "goal_id"])
print("¿Existen duplicados?:", duplicados.any())

# Se realiza un tipeo de los datos que integran el dataframe base_ods
schema1 = {"publication_id":"string",
           "goal_id":"string",
           "display_name":"string",
           "score":"float64"}

base_ods = base_ods.astype(schema1)
base_ods.info()
base_ods.dtypes


# Se construye la tabla ods
tbl_ods = base_ods[["goal_id", "display_name"]]
tbl_ods = tbl_ods.drop_duplicates(subset=["goal_id"])
tbl_ods.rename(columns=({"display_name":"display_name_ods"}), inplace=True)

# Se analiza la estructura del dataframe tbl_ods
tbl_ods.info()
tbl_ods.describe()
tbl_ods.nunique()
tbl_ods.head(10)
tbl_ods.isna().sum()
tbl_ods.columns


# Se eliminan identificadores goal_id vacios
tbl_ods = tbl_ods[tbl_ods["goal_id"].notna()]

# Se eliminan duplicados del identificador goal_id
tbl_ods = tbl_ods.drop_duplicates(subset=["goal_id"])


# Se procede a tipear los datos
schema2 = {"goal_id":"string",
           "display_name_ods":"string"}

tbl_ods = tbl_ods.astype(schema2)
tbl_ods.info()

###############################################################################
# Se construye la tabla ODSPublicacion
###############################################################################
tbl_odsPub = base_ods[["publication_id", "goal_id", "score"]]
tbl_odsPub.shape
tbl_odsPub.columns

# Se analiza la estructura de este dataframe
tbl_odsPub.info()
tbl_odsPub.describe()
tbl_odsPub.nunique()

# Se analiza la presencia de duplicados
duplicados = tbl_odsPub.duplicated(subset=["publication_id", "goal_id"])
print("¿Existen duplicados?:", duplicados.any())


###############################################################################
# SE CONSTRUYE LA TABLA Investigadores
###############################################################################

autores = ["id", "doi", "title","authorships"]
df_investigador = pd.DataFrame.from_records(trabajo_peru, columns=autores)
df_investigador = df_investigador.drop_duplicates(subset=["id"])


# Se observa un registro de la columna authorships del dataframe df_investigador
print(df_investigador["authorships"][100])

# Se crea una lista para almacenar datos
esquema = []

# Iterar sobre cada elemento de coautores["authorships"] 
for i in range(len(df_investigador["authorships"])):
    try:
        # Obtener el string de la lista de diccionarios en el índice i
        lista_autor = df_investigador["authorships"][i]
               
        # Iterar sobre cada elemento de la lista de diccionarios
        for item in lista_autor:
            try:
                author_position = item.get('author_position', None)
                author_id = item['author'].get('id', None)
                author_name = item['author'].get('display_name', None)
                author_orcid = item['author'].get('orcid', None)
                countries = item.get('countries', [])
                is_corresponding = item.get('is_corresponding', None)
                raw_author_name = item.get('raw_author_name', None)
                raw_affiliation_strings = item.get('raw_affiliation_strings', [])
        
                for institution in item.get('institutions', []):
                    try:
                        institution_id = institution.get('id', None)
                        institution_name = institution.get('display_name', None)
                        institution_ror = institution.get('ror', None)
                        institution_country_code = institution.get('country_code', None)
                        institution_type = institution.get('type', None)
                        
                        author_info = {
                            'publication_id': df_investigador['id'][i],
                            'doi': df_investigador['doi'][i],
                            'title': df_investigador['title'][i],
                            'author_position': author_position,
                            'author_id': author_id,
                            'author_name': author_name,
                            'author_orcid': author_orcid,
                            'affiliation_countries': countries,
                            'institution_id': institution_id,
                            'institution_name': institution_name,
                            'institution_ror': institution_ror,
                            'institution_country_code': institution_country_code,
                            'institution_type': institution_type,
                            'is_corresponding': is_corresponding,
                            'raw_author_name': raw_author_name,
                            'raw_affiliation_strings': raw_affiliation_strings
                        }
                        esquema.append(author_info)
                    except Exception as e:
                        print(f"Error processing institution data: {e}")
        
            except Exception as e:
                print(f"Error processing author data: {e}")
           
    except (KeyError, ValueError, SyntaxError, IndexError) as e:
        print("Error al procesar el índice", i, ":", e)
                
      
# Convertir la lista de diccionarios en un DataFrame
base = pd.DataFrame(esquema)
base.shape
base.info()
base.columns

caso = base[base["doi"]=="10.1590/s0034-75901993000600010"]
caso = base[base["publication_id"]=="https://openalex.org/W2073840323"]

# Se considera solo los atributos relacionados con los investigadores del dataframe base
tab_investigadores = base[["author_id", "author_orcid", "author_name"]]


# Función para extraer el orcid, devolviendo None en caso de error
def extraer_orcid(url):
    try:
        parts = url.split("https://orcid.org/")
        if len(parts) > 1:
            return parts[1]
        else:
            return None  # Devuelve None si no se encuentra el DOI
    except Exception:
        return None  # Devuelve None en caso de cualquier otro error

tab_investigadores['author_orcid'] = tab_investigadores['author_orcid'].apply(extraer_orcid)

# Solo se consideran los valores unicos del dataframe investigador_openalex
tab_investigadores = tab_investigadores.drop_duplicates(subset=["author_id"])


# Se utliza el objeto autores_peru para obtener más información de los investigadores que consignaron
# como última afiliación a una entidad peruana
type(autores_peru)
dir(autores_peru)

# Con el propósito de no generar inconvenientes, se cambia el nombre de dos elementos de la lista autores_peru
# Se analiza el primer elemento de la lista autores_peru
print(autores_peru[0])
type(autores_peru[0])
 
# Se cambian algunos nombres de elementos de la lista de diccionarios autores_peru
for caso1 in autores_peru:
    if "id" in caso1:
        caso1["id_autor"] = caso1.pop("id")
        
        
for caso2 in autores_peru:
    if "ids" in caso2:
        caso2["ids_autor"] = caso2.pop("ids")


# Se crea una lista con los campos id y ids
list_autor = ["id_autor", "ids_autor", "works_count", "cited_by_count", "summary_stats"]
df_autor_codigo = pd.DataFrame.from_records(autores_peru, columns=list_autor)

# Se identifica que tipo de objeto es df_autor_codigo
type(df_autor_codigo)
# Se analiza un caso específico
print(df_autor_codigo["ids_autor"][600])

# Se cuentan la cantidad de registros que tiene el dataframe df_autor_codigo
print(df_autor_codigo["id_autor"].nunique())

# Se construyen nuevas columnas para el dataframe df_autor_codigo asociadas con la identificación del autor
df_autor_codigo['codigo_openalex'] = df_autor_codigo["ids_autor"].apply(lambda x: x.get('openalex', False) if isinstance(x, dict) else False)
df_autor_codigo['codigo_orcid'] = df_autor_codigo["ids_autor"].apply(lambda x: x.get('orcid', False) if isinstance(x, dict) else False)
df_autor_codigo['codigo_scopus'] = df_autor_codigo["ids_autor"].apply(lambda x: x.get('scopus', False) if isinstance(x, dict) else False)

# Se construyen nuevas columnas para el dataframe asociadas con la producción científica del investigador
df_autor_codigo["2yr_mean_citedness"] = df_autor_codigo["summary_stats"].apply(lambda x: x.get("2yr_mean_citedness", False) if isinstance(x, dict) else False)
df_autor_codigo["h_index"] = df_autor_codigo["summary_stats"].apply(lambda x: x.get("h_index", False) if isinstance(x, dict) else False)
df_autor_codigo["i10_index"] = df_autor_codigo["summary_stats"].apply(lambda x: x.get("i10_index", False) if isinstance(x, dict) else False)


# Se eliminan columnas del dataframe df_autor_codigo y summary_stats
del df_autor_codigo["ids_autor"]
del df_autor_codigo["summary_stats"]


# El dataframe df_autor_codigo contiene los códigos de los autores
# Se analizan otros campos de la lista autores_peru
campo_autor = ["id_autor", "works_count", "cited_by_count"]
df_campo_autor = pd.DataFrame.from_records(autores_peru, columns=campo_autor)

# Se analiza el tipo de dato del dataframe df_campo_autor
#df_campo_autor.dtypes
#print(df_campo_autor["works_count"][10])
#type(df_campo_autor["works_count"][10])
#type(df_campo_autor["counts_by_year"][100])
#print(df_campo_autor["counts_by_year"][100])

# Se realiza una fusion entre el dataframe df_autor_codigo y el dataframe df_campo_autor
fusion_autor = pd.merge(df_autor_codigo, df_campo_autor, on="id_autor")
fusion_autor.columns


def extract_author_id(link):
    try:
        if not link or not isinstance(link, str):
            return None  # Si el registro es False, None o no es una cadena
        # Dividir la URL y extraer el valor de authorID
        parts = link.split("authorID=")
        if len(parts) > 1:
            author_id = parts[1].split("&")[0]  # Tomar solo el valor de authorID
            return author_id
        return None  # Si no se encuentra authorID
    except Exception as e:
        return None  # Manejar cualquier error inesperado

# Aplicar la función a la columna 'links'
df_autor_codigo["scopus"] = df_autor_codigo["codigo_scopus"].apply(extract_author_id)

# Se renombra la columna id_autor del dataframe df_autor_codigo a author_id
df_autor_codigo.rename(columns=({"id_autor":"author_id"}), inplace=True)


# El dataframe df_autor_codigo se fusiona con el dataframe tab_investigadores
# Se construye la tabla investigadores
tbl_investigadores = pd.merge(tab_investigadores, df_autor_codigo, on="author_id", how="left")
tbl_investigadores.shape
tbl_investigadores.columns

# Se consideran un conjunto de atributos del dataframe tbl_investigadores
tbl_investigadores = tbl_investigadores[["author_id", "author_orcid","author_name", "works_count",
                                         "cited_by_count", "2yr_mean_citedness", "h_index", "i10_index"]]


# Se construyen dos tablas diferencias en función a tbl_investigadores
estado = tbl_investigadores.copy()

tbl_investigadores = estado[["author_id", "author_orcid", "author_name"]]

tbl_metricas_investigador = estado[["author_id","works_count",
                                         "cited_by_count", "2yr_mean_citedness", "h_index", "i10_index"]]

# Se construye un campo que muestra la fecha actual de las métricas
tbl_metricas_investigador["fecha"] = date.today()
tbl_metricas_investigador.columns


# Se analiza la estructura y naturaleza del dataframe tbl_investigadores
tbl_investigadores.shape
tbl_investigadores.columns
tbl_investigadores.nunique()

# Solo se consideran los registros únicos
tbl_investigadores = tbl_investigadores.drop_duplicates(subset=["author_id"])
tbl_investigadores.nunique()


# Se analiza la estructura del dataframe tbl_investigadores
tbl_investigadores.shape
tbl_investigadores.dtypes
tbl_investigadores.info()
tbl_investigadores.describe()

# Se identifica la presencia de duplicados en el dataframe tbl_investigadores
duplicados = tbl_investigadores.duplicated(subset=["author_id"])
print("¿Existen duplicados?:", duplicados.any())

# Se identifica la presencia de nulos en la columna author_id
tbl_investigadores["author_id"].isna().sum()
tbl_investigadores = tbl_investigadores.dropna(subset=["author_id"])


# Se realiza un tipeo de los datos que integran el dataframe tbl_investigadores
schema3 = {"author_id":"string",
           "author_orcid":"string",
           "author_name":"string"}


tbl_investigadores = tbl_investigadores.astype(schema3)
tbl_investigadores.info()


# Por otro lado, se analiza también la estructura del dataframe tbl_metricas_investigador
tbl_metricas_investigador.shape
tbl_metricas_investigador.dtypes
tbl_metricas_investigador.info()
tbl_metricas_investigador.describe()

# Se identifica la presencia de duplicados en el dataframe tbl_metricas_investigador
duplicados = tbl_metricas_investigador.duplicated(subset=["author_id"])
print("¿Existen duplicados?:", duplicados.any())

# Se identifica la presencia de nulos en la columna author_id
tbl_metricas_investigador["author_id"].isna().sum()
tbl_metricas_investigador = tbl_metricas_investigador.dropna(subset=["author_id"])

# Se renombra algunas columnas del dataframe tbl_metricas_investigador
tbl_metricas_investigador.rename(columns=({"2yr_mean_citedness":"mean_citedness_2yr"}), inplace=True)


# Se realiza un tipeo de los datos que integran el dataframe tbl_metricas_investigador
schema4 = {"author_id":"string",
           "works_count":"Int64",
           "cited_by_count":"Int64",
           "mean_citedness_2yr":"float64",
           "h_index":"Int64",
           "i10_index":"Int64"
           }

tbl_metricas_investigador = tbl_metricas_investigador.astype(schema4)

# Convertir fecha correctamente
tbl_metricas_investigador["fecha"] = pd.to_datetime(tbl_metricas_investigador["fecha"])

tbl_metricas_investigador.info()


###############################################################################
# SE CONSTRUYE LA TABLA Afiliaciones
###############################################################################

# Se considera el dataframe base
base.columns

tab_afiliaciones = base[["institution_id", "institution_name", "institution_country_code", "institution_type"]]
tbl_afiliaciones = tab_afiliaciones.drop_duplicates(subset=["institution_id"])

# Se analiza la estructura del dataframe tbl_afiliaciones
tbl_afiliaciones.shape
tbl_afiliaciones.dtypes
tbl_afiliaciones.info()
tbl_afiliaciones.columns

# Se identifica la presencia de duplicados en el dataframe tbl_afiliaciones
duplicados = tbl_afiliaciones.duplicated(subset=["institution_id"])
print("¿Existen duplicados?:", duplicados.any())


# Se identifica la presencia de nulos en la columna institution_id
tbl_afiliaciones["institution_id"].isna().sum()
tbl_afiliaciones = tbl_afiliaciones.dropna(subset=["institution_id"])

# Se realiza un tipeo de los datos que integran el dataframe tbl_afiliaciones
schema5 = {"institution_id":"string",
           "institution_name":"string",
           "institution_country_code":"string",
           "institution_type":"string"
           }

tbl_afiliaciones = tbl_afiliaciones.astype(schema5)
tbl_afiliaciones.info()


###############################################################################
# SE CONSTRUYE LA TABLA PublicacionesInvestigadores
###############################################################################

base.columns
tbl_pub_inv = base[["publication_id", "author_id", "author_position"]]
# Se aplica un filtro considerando autores por investigación
tbl_pub_inv = tbl_pub_inv.drop_duplicates(subset=['publication_id', 'author_id'])
tbl_pub_inv.info()

# Se calcula la cantidad de nulos del dataframe tbl_pub_inv
tbl_pub_inv["author_id"].isna().sum()
tbl_pub_inv["publication_id"].isna().sum()

tbl_pub_inv = tbl_pub_inv.dropna(subset=["author_id"])

schema6 = {"publication_id":"string",
           "author_id":"string",
           "author_position":"string"}


tbl_pub_inv = tbl_pub_inv.astype(schema6)
tbl_pub_inv.info()


tbl_pub_inv = tbl_pub_inv.copy()

tbl_pub_inv["publication_id"] = tbl_pub_inv["publication_id"].astype("string").str.strip()
tbl_pub_inv["author_id"] = tbl_pub_inv["author_id"].astype("string").str.strip()
tbl_pub_inv["author_position"] = tbl_pub_inv["author_position"].astype("string").str.strip()

tbl_pub_inv = tbl_pub_inv[
    tbl_pub_inv["publication_id"].isin(tbl_publicaciones["publication_id"]) &
    tbl_pub_inv["author_id"].isin(tbl_investigadores["author_id"])
]

###############################################################################
# SE CREA LA TABLA tbl_publInvestigaAfil
###############################################################################
publicainv.columns

# Se crea la tabla PublInvestigaAfil
tbl_pubinvestigaafil = publicainv[["publication_id", "doi", "author_id", "author_name", "institution_id"]]
caso1 = tbl_pubinvestigaafil[tbl_pubinvestigaafil["publication_id"]=="https://openalex.org/W1000036331"]

# Se desarrolla un ejercio para examinar la representatividad de publicaciones científicas 
# Se desarrolla un merge con tabla tabla_afiliaciones
tabla_pubinvafil = pd.merge(tbl_pubinvestigaafil, tbl_afiliaciones, on="institution_id")

# Se obtiene la información para el Perú
peru = tabla_pubinvafil[tabla_pubinvafil["institution_country_code"]=="PE"]
peru.institution_name.value_counts()

###############################################################################
# SE CREA LA TABLA ProgramaBeca
###############################################################################

# La tabla ProgramaBeca almacena información de las becas de programas de maestria
# y doctorado otorgados por el Prociencia
# Para lograr este integración, se utilizan las bases elaboradas previamente

# Se utiliza la información almacenada en ctvitae
ctvitae = pd.read_csv("sit_consulta_cti_vitae.csv",encoding='utf-8', delimiter = ",")
ctvitae.columns

# Se utiliza el archivo que contiene información sobre los becarios de maestria
def int_to_str(value):
    return str(value)

# Especifica el diccionario de conversión en el parámetro converters
converters = {"DNI": int_to_str}

maestria = pd.read_excel("info_becarios_maestria.xlsx", sheet_name="Sheet1", header=0, converters=converters)
maestria.columns
maestria.dtypes

converters1 = {"codigo_scopus":int_to_str, "DNI":int_to_str}
doctorado = pd.read_excel("info_becarios_doctorado.xlsx", sheet_name="Sheet1", header=0, converters=converters1)
doctorado.columns
doctorado.dtypes
# Se convierte la columna DNI de int a string del dataframe doctorado
#doctorado["DNI"] = doctorado["DNI"].apply(lambda x: str(x))


# Del dataframe ctvitae se considera un conjunto de atributos
datos = ctvitae[["Nro de Documento de Identidad", "id_perfil_scopus"]]
datos.rename(columns=({"Nro de Documento de Identidad":"DNI",
                       "id_perfil_scopus":"codigo_scopus"}), inplace=True)

datos = datos.drop_duplicates(subset=["DNI"])

# Se fusiona el dataframe datos con el dataframe maestria
dato_maestria = pd.merge(maestria, datos, on="DNI", how="left")
dato_maestria.columns
dato_maestria["codigo_scopus"].count()

# Se arma la tabla de programa de becas a partir de dato_maestria y doctorado
base1 = doctorado[["Número de Contrato", "esqueFinanc", "AÑO CONVOCATORIA", "PAÍS EN DONDE SE REALIZA LA SUBVENCIÓN"]]
# Se eliminan los registros duplicados del dataframe base1
base1 = base1.drop_duplicates(subset=["Número de Contrato"])

base2 = dato_maestria[["Número de Contrato", "esqueFinanc", "AÑO CONVOCATORIA", "PAÍS EN DONDE SE REALIZA LA SUBVENCIÓN"]]
base2 = base2.drop_duplicates(subset=["Número de Contrato"])

# Usando los dataframes base1 y base2, se crea la tabla programa
tbl_programa = pd.concat([base1, base2], ignore_index=True)
tbl_programa.columns

# Se renombran algunas columnas del dataframe tabla_programa
tbl_programa.rename(columns=({"Número de Contrato":"id_contrato_programa",
                                "AÑO CONVOCATORIA":"año_convocatoria",
                                "PAÍS EN DONDE SE REALIZA LA SUBVENCIÓN":"Pais"}), inplace=True)

# Se analiza la distribución de convocatorias por paises
tbl_programa.Pais.value_counts()

###############################################################################
# Se crea la tabla tbl_Becarios
###############################################################################

# Se crea la tabla becarios de programas de maestria y doctorado
becario1 = doctorado[["DNI","codigo_scopus", "codigo_renacyt","GENERO", "año_inicio_subvencion", "año_fin_subvencion", "Línea de Investigación"]]
becario2 = dato_maestria[["DNI", "codigo_scopus", "GÉNERO", "año_inicio_subvencion", "año_fin_subvencion", "Línea de Investigación"]]
becario2.rename(columns=({"GÉNERO":"GENERO"}), inplace=True)

becario1 = becario1.drop_duplicates(subset=["DNI"])
becario2 = becario2.drop_duplicates(subset=["DNI"])

# Usando los dataframes becario1 y becario2, se crea la tabla becario
tbl_becarios = pd.concat([becario1, becario2], ignore_index=True)
tbl_becarios["DNI"].nunique()
tbl_becarios = tbl_becarios.drop_duplicates(subset=["DNI"])
tbl_becarios.columns

###############################################################################
# Se crea la tabla BecaBecarios
###############################################################################
caso1 = doctorado[["Número de Contrato", "DNI", "codigo_scopus"]]
caso2 = dato_maestria[["Número de Contrato", "DNI", "codigo_scopus"]]

tbl_programa_becarios = pd.concat([caso1, caso2], ignore_index=True)
tbl_programa_becarios.columns
tbl_programa_becarios.rename(columns=({"Número de Contrato":"id_contrato_programa"}), inplace=True)


###############################################################################
# Se crea una nueva base de datos o esquema y se almacena la información de mis
# dataframes en esa nueva base de datos
###############################################################################

# Se establece la conexión entre python y MySQL
usuario = "root"          # Usuario de MySQL
contraseña = "garabombo2468"  # Contraseña de MySQL
host = "localhost"        # Dirección del servidor (localhost para uso local)
base_datos = "db_open_peru"  # Nombre de la base de datos


# Crear la conexión usando pymysql de python
engine = create_engine(f"mysql+pymysql://{usuario}:{contraseña}@{host}/{base_datos}")

# Se sube la tabla tbl_publicaciones a mi database db_open_peru
tbl_publicaciones.to_sql(
    "tbl_publicaciones",
    engine,
    schema="db_open_peru",
    if_exists="append",
    index=False
)

# Se sube la tabla tbl_ods a mi database db_open_peru
tbl_ods.to_sql(
    "tbl_ods",
    engine,
    schema="db_open_peru",
    if_exists="append",
    index=False)


# Se sube la tabla tbl_odsPub a mi database db_open_peru
tbl_odsPub.to_sql(
    "tbl_odspub",
    engine,
    schema="db_open_peru",
    if_exists="append",
    index=False)


# Se sube la tabla tbl_investigadores a mi database db_open_peru
tbl_investigadores.to_sql(
    "tbl_investigadores",
    engine,
    schema="db_open_peru",
    if_exists="append",
    index=False)


# Se sube la tabla tbl_metricas_investigador a mi database db_open_peru
tbl_metricas_investigador.to_sql(
    "tbl_metricas_investigador",
    engine,
    schema="db_open_peru",
    if_exists="append",
    index=False)


# Se sube la tabla tbl_afiliaciones a mi database db_open_peru
tbl_afiliaciones.to_sql(
    "tbl_afiliaciones",
    engine,
    schema="db_open_peru",
    if_exists="append",
    index=False)


# Se sube la tabla tbl_pub_inv a mi database db_open_peru
tbl_pub_inv.to_sql(
    "tbl_pub_inv",
    engine,
    schema="db_open_peru",
    if_exists="append",
    index=False)


###############################################################################
# Se realiza el procedimiento inverso, es decir, se convierten las tablas de mi
# MariaDB en dataframes
###############################################################################

# Ahora se trae la tabla sobre publicaciones científicas de SQL a Python
df = pd.read_sql(
    "SELECT * FROM db_open_peru.tbl_publicaciones",
    con=engine
)

# Se realiza un shape del dataframe df
df.shape















