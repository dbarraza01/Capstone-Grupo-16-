"""
comordibipy.py
================================================================================
Script de Inspección y Simulación Interactiva del Índice de Charlson y Elixhauser.
Muestra cómo la librería `comorbidipy` procesa los códigos ICD-10 de un paciente
y calcula sus scores de comorbilidad bajo las reglas de Quan et al. (2005).
================================================================================
"""

import sys
import pandas as pd
import numpy as np
import comorbidipy
from comorbidipy import comorbidity

# Traducciones al español de las categorías para facilitar el entendimiento clínico
CHARLSON_DESC_ES = {
    'aids': 'SIDA / VIH',
    'ami': 'Infarto Agudo de Miocardio',
    'canc': 'Neoplasia Maligna (Cáncer activo)',
    'cevd': 'Enfermedad Cerebrovascular',
    'chf': 'Insuficiencia Cardíaca Congestiva',
    'copd': 'Enfermedad Pulmonar Crónica (EPOC)',
    'dementia': 'Demencia',
    'diab': 'Diabetes sin complicaciones crónicas',
    'diabwc': 'Diabetes con complicaciones crónicas',
    'hp': 'Hemiplejia o Paraplejia',
    'metacanc': 'Tumor Sólido Metastásico',
    'mld': 'Enfermedad Hepática Leve',
    'msld': 'Enfermedad Hepática Moderada o Grave',
    'pud': 'Enfermedad Úlcera Péptica',
    'pvd': 'Enfermedad Vascular Periférica',
    'rend': 'Enfermedad Renal',
    'rheumd': 'Enfermedad Reumática'
}

ELIXHAUSER_DESC_ES = {
    'aids': 'SIDA / VIH',
    'alcohol': 'Abuso de alcohol',
    'blane': 'Anemia por pérdida de sangre',
    'carit': 'Arritmias cardíacas',
    'chf': 'Insuficiencia cardíaca congestiva',
    'coag': 'Coagulopatía',
    'cpd': 'Enfermedad pulmonar crónica',
    'dane': 'Anemia por deficiencia',
    'depre': 'Depresión',
    'diabc': 'Diabetes complicada',
    'diabunc': 'Diabetes no complicada',
    'drug': 'Abuso de drogas',
    'fed': 'Trastornos de fluidos y electrolitos',
    'hypc': 'Hipertensión complicada',
    'hypothy': 'Hipotiroidismo',
    'hypunc': 'Hipertensión no complicada',
    'ld': 'Enfermedad hepática',
    'lymph': 'Linfoma',
    'metacanc': 'Cáncer metastásico',
    'obes': 'Obesidad',
    'ond': 'Otros trastornos neurológicos',
    'para': 'Parálisis',
    'pcd': 'Trastornos de la circulación pulmonar',
    'psycho': 'Psicosis',
    'pud': 'Úlcera péptica sin hemorragia',
    'pvd': 'Trastornos vasculares periféricos',
    'rf': 'Insuficiencia renal',
    'rheumd': 'Artritis reumatoide / Enfermedad colágeno vascular',
    'solidtum': 'Tumor sólido sin metástasis',
    'valv': 'Enfermedad valvular',
    'wloss': 'Pérdida de peso'
}

def mostrar_tablas_pesos():
    """Imprime las tablas de pesos oficiales cargadas por la librería."""
    print("=" * 105)
    print(" 1. TABLAS DE PESOS COMPARATIVAS (ICD-10 QUAN)")
    print("=" * 105)

    # --- Charlson Weights ---
    print("\n[PESOS DEL ÍNDICE DE CHARLSON (Classic 'charlson' vs Quan 'quan')]")
    print("Nota: El proyecto usa el weighting 'quan' por defecto en la librería.")
    ch_classic = comorbidipy.weights.weights['charlson_icd10_quan']['charlson']
    ch_quan = comorbidipy.weights.weights['charlson_icd10_quan']['quan']
    charlson_names = comorbidipy.colnames.get_colnames('charlson')
    
    rows_ch = []
    for key in ch_classic.keys():
        rows_ch.append({
            "Clave": key,
            "Descripción (Español)": CHARLSON_DESC_ES.get(key, "Desconocido"),
            "Peso Classic": ch_classic[key],
            "Peso Quan": ch_quan[key],
            "Descripción (Inglés)": charlson_names.get(key, "").strip()
        })
    df_ch = pd.DataFrame(rows_ch).sort_values("Peso Classic", ascending=False)
    print(df_ch.to_string(index=False))

    # --- Elixhauser Weights (van Walraven) ---
    print("\n" + "-" * 105)
    print("[PESOS DEL ÍNDICE DE ELIXHAUSER (Ponderación van Walraven - 'vw')]")
    elix_raw = comorbidipy.weights.weights['elixhauser_icd10_quan']['vw']
    elix_names = comorbidipy.colnames.get_colnames('elixhauser')
    
    rows_el = []
    for key, weight in elix_raw.items():
        rows_el.append({
            "Clave": key,
            "Descripción (Español)": ELIXHAUSER_DESC_ES.get(key, "Desconocido"),
            "Peso (vw)": weight,
            "Descripción (Inglés)": elix_names.get(key, "").strip()
        })
    df_el = pd.DataFrame(rows_el).sort_values("Peso (vw)", ascending=False)
    print(df_el.to_string(index=False))
    print("\n" + "=" * 105)


def simular_proceso_paciente(nombre, codigos_crudos, weighting_esquema="quan"):
    """
    Simula paso a paso el algoritmo de comorbilidad para un paciente ficticio.
    """
    print(f"\n⚡ PROCESANDO PACIENTE: {nombre} (Esquema de Pesos: '{weighting_esquema}')")
    print(f"   Códigos de entrada: {codigos_crudos}")
    
    # 1. Limpieza y preprocesamiento de códigos
    codigos_limpios = []
    for c in codigos_crudos:
        clean = c.replace(".", "").strip().upper()
        codigos_limpios.append((c, clean))
    
    # Deduplicar
    codigos_unicos = sorted(list(set([clean for _, clean in codigos_limpios])))
    print(f"   1. Limpieza y Deduplicación:")
    for orig, clean in codigos_limpios:
        print(f"      - '{orig}' -> '{clean}'")
    print(f"      Códigos únicos limpios a evaluar: {codigos_unicos}")

    # 2. Mapeo de códigos a categorías clínicas (Quan ICD-10)
    mapa_charlson = comorbidipy.mapping.mapping['charlson_icd10_quan']
    charlson_weights = comorbidipy.weights.weights['charlson_icd10_quan'][weighting_esquema]

    categorias_detectadas = {}

    print(f"\n   2. Mapeo de códigos a categorías (Charlson):")
    for codigo in codigos_unicos:
        matched = False
        for cat_key, prefijos in mapa_charlson.items():
            for prefijo in prefijos:
                # Comprobar si el código del paciente empieza con el prefijo de la categoría
                if codigo.startswith(prefijo):
                    if cat_key not in categorias_detectadas:
                        categorias_detectadas[cat_key] = []
                    categorias_detectadas[cat_key].append(codigo)
                    matched = True
                    print(f"      👉 Código '{codigo}' coincide con prefijo '{prefijo}' -> Categoría '{cat_key}' ({CHARLSON_DESC_ES[cat_key]})")
                    break # ya coincidió con esta categoría, ir a la siguiente
        if not matched:
            print(f"      ❌ Código '{codigo}' no mapea a ninguna categoría de Charlson.")

    # 3. Aplicación de Jerarquías de Exclusión (Evitar doble conteo)
    print(f"\n   3. Aplicación de Jerarquías de Exclusión (en una copia para calcular el score):")
    
    # Clonamos para ver cambios
    presencia = {cat: 1 for cat in categorias_detectadas.keys()}
    for cat in charlson_weights.keys():
        if cat not in presencia:
            presencia[cat] = 0

    # Las reglas de exclusión/jerarquía de la librería
    jerarquias = [
        ('msld', 'mld', "Enfermedad Hepática Grave desactiva Leve"),
        ('diabwc', 'diab', "Diabetes con Complicaciones desactiva Diabetes sin Complicaciones"),
        ('metacanc', 'canc', "Cáncer Metastásico desactiva Cáncer Activo Localizado")
    ]

    for mayor, menor, razon in jerarquias:
        if presencia.get(mayor, 0) == 1 and presencia.get(menor, 0) == 1:
            presencia[menor] = 0
            print(f"      ⚠️  Jerarquía Activa: {razon} -> {menor.upper()} se pone en 0 para el puntaje (pero mantiene su bandera 1 original en el DataFrame de salida).")
        elif presencia.get(menor, 0) == 1:
            print(f"      ✓ Categoría {menor.upper()} activa (no hay comorbilidad mayor de este tipo).")

    # 4. Suma de Pesos
    print(f"\n   4. Cálculo del Score Final de Charlson:")
    suma_total = 0
    detalles_suma = []
    
    for cat, activo in presencia.items():
        if activo == 1:
            w = charlson_weights[cat]
            suma_total += w
            detalles_suma.append(f"{cat.upper()} ({CHARLSON_DESC_ES[cat]}) = +{w}")
            
    if detalles_suma:
        print("      " + " + ".join(detalles_suma) + f" = {suma_total}")
    else:
        print("      Ninguna comorbilidad detectada. Score = 0")
        
    # 5. Validación Oficial con la librería
    records = []
    for c in codigos_unicos:
        records.append({'id': 'paciente_test', 'code': c, 'age': 0})
    df_long = pd.DataFrame(records)
    
    df_res = comorbidity(
        df_long,
        id='id',
        code='code',
        age='age',
        score='charlson',
        icd='icd10',
        variant='quan',
        weighting=weighting_esquema
    )
    
    score_libreria = int(df_res['comorbidity_score'].iloc[0])
    print(f"\n   5. Validación de salida contra la librería:")
    print(f"      ✓ Score de nuestra simulación paso a paso: {suma_total}")
    print(f"      ✓ Score entregado por comorbidipy.comorbidity(): {score_libreria}")
    assert suma_total == score_libreria, f"ERROR: La simulación manual difiere de la librería. Esperado {score_libreria}, obtenido {suma_total}"
    print(f"      ✅ ¡Ambos coinciden a la perfección!")
    print("-" * 80)


def main():
    # Mostrar las tablas de pesos al inicio
    mostrar_tablas_pesos()

    print("\n" + "=" * 105)
    print(" 2. SIMULACIÓN DEL PROCESAMIENTO PASO A PASO CON PACIENTES DE PRUEBA")
    print("=" * 105)

    # Caso 1: Paciente Cardiovascular y Renal común (Esquema por defecto 'quan')
    simular_proceso_paciente(
        nombre="Paciente A (Cardiovascular y Renal)",
        codigos_crudos=["I50.9", "N18.9"],
        weighting_esquema="quan"
    )

    # Caso 2: Paciente con Jerarquía de Diabetes (Evaluado en ambos esquemas de pesos)
    # - En el esquema 'charlson' (classic), diabwc pesa 2, y canc pesa 2 (Total = 4)
    # - En el esquema 'quan' (default), diabwc pesa 1, y canc pesa 2 (Total = 3)
    simular_proceso_paciente(
        nombre="Paciente B (Jerarquía Diabetes + Cáncer de Pulmón)",
        codigos_crudos=["E11.9", "E11.21", "C34.90"],
        weighting_esquema="charlson"
    )
    simular_proceso_paciente(
        nombre="Paciente B (Jerarquía Diabetes + Cáncer de Pulmón)",
        codigos_crudos=["E11.9", "E11.21", "C34.90"],
        weighting_esquema="quan"
    )

    # Caso 3: Paciente con Códigos sin Puntos, Repetición y Cáncer Metastásico
    # - 'I219' y 'I21.9' se limpian y deduplican.
    # - En el esquema 'charlson' (classic), ami pesa 1, metacanc pesa 6 (Total = 7)
    # - En el esquema 'quan' (default), ami pesa 0, metacanc pesa 6 (Total = 6)
    simular_proceso_paciente(
        nombre="Paciente C (Repetidos, sin puntos y Cáncer Metastásico)",
        codigos_crudos=["I219", "I21.9", "C18.9", "C78.7"],
        weighting_esquema="charlson"
    )
    simular_proceso_paciente(
        nombre="Paciente C (Repetidos, sin puntos y Cáncer Metastásico)",
        codigos_crudos=["I219", "I21.9", "C18.9", "C78.7"],
        weighting_esquema="quan"
    )

if __name__ == "__main__":
    main()
