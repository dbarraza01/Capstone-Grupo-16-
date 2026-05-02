# =====================================================
# DICCIONARIO ICD-10-PCS — Los 7 caracteres explicados
# =====================================================

'''
Esto es 100% provisorio, aún no se está usando este diccionario
'''

# CARÁCTER 1: Sección (ya tienes esto en p_caract_1)
seccion_pcs = {
    '0': 'Cirugía médica y quirúrgica (Medical & Surgical)',
    '1': 'Obstetricia',
    '2': 'Colocación (Placement)',
    '3': 'Administración (ej: transfusiones, infusiones)',
    '4': 'Medición y monitoreo',
    '5': 'Asistencia y rendimiento extracorpóreo (ej: ECMO, diálisis)',
    '6': 'Terapias extracorpóreas',
    '7': 'Osteopatía',
    '8': 'Otros procedimientos',
    '9': 'Quiropráctica',
    'B': 'Diagnóstico por imagen (Imaging)',
    'C': 'Medicina nuclear',
    'D': 'Oncología radioterápica',
    'F': 'Rehabilitación física y audiología',
    'G': 'Salud mental',
    'H': 'Tratamiento de abuso de sustancias',
    'X': 'Nueva tecnología',
}

# CARÁCTER 2: Sistema corporal (solo aplica para sección 0 — la más común)
sistema_corporal_pcs = {
    '0': 'Sistema nervioso central',
    '1': 'Sistema nervioso periférico',
    '2': 'Corazón y grandes vasos',
    '3': 'Arterias superiores',
    '4': 'Arterias inferiores',
    '5': 'Venas superiores',
    '6': 'Venas inferiores',
    '7': 'Sistema linfático y hemático',
    '8': 'Ojo',
    '9': 'Oído, nariz y seno',
    'B': 'Sistema respiratorio',
    'C': 'Boca y garganta',
    'D': 'Sistema gastrointestinal',
    'F': 'Sistema hepatobiliar y páncreas',
    'G': 'Sistema endocrino',
    'H': 'Piel y mama',
    'J': 'Tejido subcutáneo y fascia',
    'K': 'Músculos',
    'L': 'Tendones',
    'M': 'Bursas y ligamentos',
    'N': 'Huesos craneales y faciales',
    'P': 'Huesos superiores',
    'Q': 'Huesos inferiores',
    'R': 'Articulaciones superiores',
    'S': 'Articulaciones inferiores',
    'T': 'Sistema urinario',
    'U': 'Sistema reproductor femenino',
    'V': 'Sistema reproductor masculino',
    'W': 'Regiones anatómicas generales',
    'X': 'Regiones anatómicas, extremidades superiores',
    'Y': 'Regiones anatómicas, extremidades inferiores',
}

# CARÁCTER 3: Operación raíz (lo que SE HACE — el más clínico)
operacion_raiz_pcs = {
    '0': 'Alteración (Alteration)',
    '1': 'Derivación/Bypass',
    '2': 'Cambio de dispositivo (Change)',
    '3': 'Control de hemorragia (Control)',
    '4': 'Creación (Creation)',
    '5': 'Destrucción de tejido (Destruction)',
    '6': 'Desarticulación (Detachment)',
    '7': 'Dilatación (Dilation)',
    '8': 'División (Division)',
    '9': 'Drenaje (Drainage)',
    'B': 'Escisión parcial (Excision)',
    'C': 'Extirpación de materia sólida (Extirpation)',
    'D': 'Extracción (Extraction)',
    'F': 'Fragmentación (Fragmentation)',
    'G': 'Fusión (Fusion)',
    'H': 'Inserción de dispositivo (Insertion)',
    'J': 'Inspección (Inspection)',
    'K': 'Mapeo (Map)',
    'L': 'Oclusión (Occlusion)',
    'M': 'Reimplante (Reattachment)',
    'N': 'Liberación/Neurólisis (Release)',
    'P': 'Remoción de dispositivo (Removal)',
    'Q': 'Reparación (Repair)',
    'R': 'Reemplazo (Replacement)',
    'S': 'Reposición (Reposition)',
    'T': 'Resección total (Resection)',
    'V': 'Restricción (Restriction)',
    'W': 'Revisión (Revision)',
    'X': 'Transferencia (Transfer)',
    'Y': 'Trasplante (Transplantation)',
}

# CARÁCTER 5: Abordaje quirúrgico (cómo se accede al sitio)
abordaje_pcs = {
    '0': 'Abierto (Open)',
    '3': 'Percutáneo',
    '4': 'Percutáneo endoscópico (laparoscopía)',
    '7': 'Apertura natural o artificial',
    '8': 'Apertura natural o artificial endoscópica',
    'F': 'Apertura natural + asistencia percutánea endoscópica',
    'X': 'Externo (sobre la piel)',
}


# Ejemplo de uso (descomentar para probar):
# print(decodificar_pcs('0DB64Z3'))
# → sección 0 (cirugía), sistema D (gastrointestinal),
#   operación B (excisión), abordaje 4 (percutáneo endoscópico),
#   dispositivo Z (ninguno), calificador 3 (diagnóstico)