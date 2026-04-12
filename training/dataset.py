"""
Dataset multi-dominio para el notebook de Transfer Learning RSL - v3.
MEJORAS v3:
  1. Inputs traducidos/adaptados al espanol academico (elimina code-switching Spanglish)
  2. 30 ejemplos adicionales en columnas debiles (Estimacion del efecto, Escala psicometrica,
     Estrategia de reclutamiento, Diseno de cohorte, Terapia evaluada)
  3. Total: 100 ejemplos originales -> ~300 aumentados
CLAVE: Cada ejemplo usa un nombre de columna DISTINTO y UNICO.
"""

DATASET = [
    # ================================================================
    # DOMINIO 1: MEDICINA / SALUD
    # ================================================================
    {"input":"Extrae [Diseno del ensayo]: Se realizo un ensayo controlado aleatorizado doble ciego comparando metformina vs placebo en 312 pacientes con diabetes tipo 2 durante 52 semanas. El endpoint primario fue la reduccion de HbA1c.",
     "output":"ECA doble ciego: metformina vs. placebo, n=312 diabeticos tipo 2, 52 semanas. Endpoint: reduccion de HbA1c."},

    {"input":"Extrae [Tipo de sintesis]: Esta meta-analisis sintetiza 34 ECA (n=12,400) sobre el efecto de suplementos de omega-3 en mortalidad cardiovascular publicados entre 2000 y 2023 usando modelo de efectos aleatorios DerSimonian-Laird.",
     "output":"Meta-analisis con efectos aleatorios (DerSimonian-Laird): 34 ECA, n=12,400. Outcome: mortalidad cardiovascular por omega-3 (2000-2023)."},

    {"input":"Extrae [Tratamiento farmacologico evaluado]: Se evaluo sertralina (50-200mg/dia) comparada con placebo. La fluoxetina y el escitalopram se mencionan como alternativas en la introduccion pero no se probaron.",
     "output":"EVALUADO: sertralina (50-200mg/dia) vs. placebo. MENCIONADOS (no evaluados): fluoxetina y escitalopram - alternativas citadas en la introduccion."},

    {"input":"Extrae [Terapia psicologica evaluada]: Se prueban TCC (12 sesiones semanales) y MBSR (8 semanas). La terapia psicodinamica se cita como alternativa pero no se compara empiricamente en este estudio.",
     "output":"EVALUADAS: TCC (12 sesiones semanales) y MBSR (8 semanas). MENCIONADA (no evaluada): terapia psicodinamica - alternativa citada, sin evaluacion empirica."},

    {"input":"Extrae [Criterios de elegibilidad]: Adultos 40-75 anos con hipertension (PA>140/90), reclutados en 8 hospitales espanoles. Exclusiones: insuficiencia renal (TFG<60), embarazo, ACV previo en ultimos 5 anos.",
     "output":"Inclusion: adultos 40-75 anos, hipertension (PA>140/90), 8 hospitales espanoles. Exclusion: insuficiencia renal (TFG<60), embarazo, ACV previo en ultimos 5 anos."},

    {"input":"Extrae [Endpoint primario del estudio]: Endpoint compuesto: muerte cardiovascular + IAM no fatal + ACV no fatal a los 36 meses, adjudicado por comite ciegado.",
     "output":"Endpoint primario compuesto: muerte cardiovascular + IAM no fatal + ACV no fatal a 36 meses, adjudicado por comite cegado."},

    {"input":"Extrae [Tamano del efecto reportado]: Reduccion de HbA1c: intervencion -1.2% vs control -0.3% (p<0.001, d de Cohen=0.78, NNT=8, IC95%: 5-14). Sin diferencia en episodios hipoglucemicos.",
     "output":"Efecto: HbA1c -1.2% (intervencion) vs. -0.3% (control), p<0.001, d=0.78, NNT=8 (IC95%: 5-14). Sin diferencias en hipoglucemia."},

    {"input":"Extrae [Posicion de los autores sobre primera linea]: Los autores concluyen explicitamente que la TCC debe reemplazar a los antidepresivos como tratamiento de primera linea para depresion leve-moderada dado que tiene eficacia equivalente y menos efectos secundarios.",
     "output":"Los autores concluyen explicitamente que la TCC debe reemplazar a los antidepresivos como primera linea en depresion leve-moderada, por eficacia equivalente y menos efectos secundarios."},

    {"input":"Extrae [Amenazas a la validez reconocidas]: Los autores declaran: seguimiento de 6 meses insuficiente para enfermedad cronica; imposible cegar por naturaleza conductual; posible sesgo de seleccion por inscripcion voluntaria.",
     "output":"Amenazas reconocidas: (1) seguimiento insuficiente (6 meses) para pronostico cronico, (2) cegamiento imposible por intervencion conductual, (3) sesgo de seleccion por inscripcion voluntaria."},

    {"input":"Extrae [Diseno de cohorte]: Cohorte prospectiva de 2,340 enfermeras seguidas 10 anos con controles de salud anuales y cuestionarios bianuales. Resultado: incidencia de enfermedad cardiovascular segun tipo de turno laboral.",
     "output":"Cohorte prospectiva: 2,340 enfermeras seguidas 10 anos (controles anuales + cuestionarios bianuales). Outcome: incidencia de enfermedad cardiovascular segun tipo de turno."},

    # --- Columnas debiles adicionales (medicina) ---
    {"input":"Extrae [Estimacion del efecto del tratamiento]: El grupo tratado mostro reduccion del 34% en la incidencia de eventos cardiacos vs control (HR=0.66, IC95%: 0.54-0.81, p<0.001, NNT=18). No se observaron diferencias en mortalidad total.",
     "output":"Tratamiento: reduccion del 34% en eventos cardiacos (HR=0.66, IC95%: 0.54-0.81, p<0.001, NNT=18). Sin diferencia en mortalidad total."},

    {"input":"Extrae [Escala de medicion del dolor]: Dolor evaluado con Escala Visual Analoga (EVA, 0-100mm) y Cuestionario Brief Pain Inventory (BPI) en 3 momentos: basal, 6 semanas y 12 semanas. Ambas validadas en espanol.",
     "output":"EVA (0-100mm) + BPI, aplicados en basal, 6 y 12 semanas. Ambas escalas validadas en espanol."},

    {"input":"Extrae [Terapia evaluada como intervencion unica]: Solo la EMDR fue evaluada en este ensayo. La introduccion cita terapia de exposicion, ISRS y TCC como alternativas para el TEPT, pero no forman parte del diseno experimental.",
     "output":"EVALUADA: EMDR (unica intervencion del ensayo). ALTERNATIVAS (no evaluadas, solo revision): terapia de exposicion, ISRS y TCC para TEPT."},

    {"input":"Extrae [Estrategia de reclutamiento clinico]: Reclutamiento consecutivo en 5 consultas de atencion primaria de Madrid durante 18 meses (enero 2021-junio 2022). Se excluyo a pacientes con comorbilidades psiquiatricas o tratamiento farmacologico activo.",
     "output":"Reclutamiento consecutivo: 5 consultas de atencion primaria, Madrid, 18 meses (ene.2021-jun.2022). Exclusion: comorbilidades psiquiatricas o farmacoterapia activa."},

    {"input":"Extrae [Definicion operacional de remision]: Remision definida como puntuacion HDRS-17 menor o igual a 7 puntos mantenida durante al menos 4 semanas consecutivas, confirmada en dos evaluaciones independientes.",
     "output":"Remision: HDRS-17 <=7 durante >=4 semanas consecutivas, confirmada en 2 evaluaciones independientes."},

    # ================================================================
    # DOMINIO 2: EDUCACION
    # ================================================================
    {"input":"Extrae [Modelo pedagogico implementado]: Aula invertida: videos en casa (15-20 min) mas tiempo de clase para resolucion colaborativa de problemas. El grupo control recibio clase magistral tradicional.",
     "output":"Intervencion: aula invertida - videos en casa (15-20 min) + clase para resolucion colaborativa. Control: clase magistral tradicional."},

    {"input":"Extrae [Alcance geografico y temporal de la revision]: Revision sistematica (PRISMA 2020): 62 estudios sobre gamificacion en educacion superior, 14 paises, 8 disciplinas, 2010-2023, fuentes en ingles y espanol.",
     "output":"Revision sistematica (PRISMA 2020): 62 estudios, 14 paises, 8 disciplinas, 2010-2023. Idiomas: ingles y espanol."},

    {"input":"Extrae [Plataforma de IA comparada]: ChatGPT (GPT-4, temperatura=0.7) fue evaluado como herramienta de retroalimentacion en escritura frente a revision por pares y retroalimentacion del instructor. Grammarly y Turnitin se mencionan en el marco teorico pero no se comparan.",
     "output":"EVALUADA: ChatGPT (GPT-4, temperatura=0.7) vs. evaluacion por pares vs. retroalimentacion del instructor. MENCIONADAS (no evaluadas): Grammarly y Turnitin en el marco teorico."},

    {"input":"Extrae [Descripcion de la muestra estudiantil]: 342 estudiantes de primer ano de estadistica introductoria en una universidad publica colombiana. Excluidos: quienes tenian experiencia previa en programacion o matematicas a nivel calculo.",
     "output":"342 estudiantes de primer ano en estadistica introductoria, universidad publica colombiana. Exclusion: programacion previa o matematicas avanzadas (nivel calculo)."},

    {"input":"Extrae [Operacionalizacion del rendimiento academico]: El rendimiento se midio con: (1) nota de examen final (0-100), (2) completacion del curso (binaria), (3) escala de autoeficacia academica de Bandura (Likert 1-7, alfa=0.89).",
     "output":"Rendimiento academico: (1) examen final (0-100), (2) completacion del curso (binaria), (3) autoeficacia academica (Bandura, Likert 1-7, alfa=0.89)."},

    {"input":"Extrae [Diferencia cuantitativa entre grupos]: El aula invertida obtuvo 8.2 puntos mas en el examen final (IC95%: 3.1-13.3, p=0.003, d de Cohen=0.67). Completacion del curso: 91% (invertida) vs 77% (control).",
     "output":"Examen final: aula invertida +8.2 puntos (IC95%: 3.1-13.3, p=0.003, d=0.67). Completacion: 91% (invertida) vs. 77% (control)."},

    {"input":"Extrae [Limitacion de generalizabilidad]: El diseno monocentrico limita la validez externa. Los autores reconocen el efecto Hawthorne como amenaza potencial y la ausencia de asignacion aleatoria como limitacion metodologica.",
     "output":"Limitaciones declaradas: (1) diseno en una sola institucion (baja validez externa), (2) efecto Hawthorne reconocido, (3) sin asignacion aleatoria a grupos."},

    {"input":"Extrae [Duracion y estructura del programa de tutoria]: Tutoria entre pares: sesiones de 30 minutos semanales durante 16 semanas. Los tutores recibieron 4 horas de entrenamiento previo. Estudiantes de cuartil superior ensenaron a cuartil inferior.",
     "output":"Estructura: sesiones de 30 min semanales, 16 semanas. Tutores: entrenamiento previo de 4 horas. Seleccion: estudiantes cuartil superior ensenan a cuartil inferior."},

    {"input":"Extrae [Instrumentos de evaluacion bilingue]: Pruebas estandarizadas SIELE (espanol) y Cambridge B2 (ingles) mas entrevistas semiestructuradas (n=90, submuestra), aplicadas en los anos 1, 2 y 3 del programa.",
     "output":"Instrumentos: SIELE (espanol) + Cambridge B2 (ingles) + entrevistas semi-estructuradas (n=90 submuestra), aplicados en anos 1, 2 y 3."},

    {"input":"Extrae [Plataformas no evaluadas en el estudio]: Coursera, Khan Academy y Duolingo se citan en la revision de literatura como alternativas de aprendizaje digital. Este estudio evalua exclusivamente el programa presencial de tutoria entre pares.",
     "output":"EVALUADO: programa de tutoria entre pares presencial. MENCIONADAS (no evaluadas): Coursera, Khan Academy, Duolingo - alternativas digitales en la revision de literatura."},

    # --- Columnas debiles adicionales (educacion) ---
    {"input":"Extrae [Estrategia de reclutamiento estudiantil]: Muestreo por conveniencia en 6 institutos de educacion secundaria de Lima Metropolitana. Los directores de cada institucion autorizaron el acceso durante el horario escolar en el segundo trimestre de 2022.",
     "output":"Muestreo por conveniencia: 6 institutos de secundaria, Lima Metropolitana. Acceso autorizado por direccion, horario escolar, segundo trimestre 2022."},

    {"input":"Extrae [Escala de motivacion academica aplicada]: Motivacion medida con la Escala de Motivacion Educativa (EME-S, 28 items, 7 subescalas tipo Likert 1-7). La consistencia interna global fue alfa=0.87 y las subescalas oscilaron entre 0.74 y 0.91.",
     "output":"EME-S (28 items, 7 subescalas Likert 1-7). Consistencia interna: alfa global=0.87; subescalas: 0.74-0.91."},

    # ================================================================
    # DOMINIO 3: PSICOLOGIA
    # ================================================================
    {"input":"Extrae [Diseno correlacional]: Encuesta transversal en linea con n=1,204 adultos examinando la asociacion entre uso diario de redes sociales (horas/dia), soledad (UCLA-3) y ansiedad (GAD-7).",
     "output":"Encuesta transversal online, correlacional: n=1,204 adultos. Variables: uso diario de redes (horas), soledad (UCLA-3) y ansiedad (GAD-7)."},

    {"input":"Extrae [Comparacion de intervenciones psicologicas]: ECA de 3 brazos: MBSR (8 semanas) vs TCC (12 semanas) vs lista de espera. Todos los participantes diagnosticados con TAG por psicologo licenciado. Farmacoterapia excluida por diseno.",
     "output":"ECA de 3 brazos: MBSR (8 sem.) vs. TCC (12 sem.) vs. control lista de espera. Diagnostico TAG por psicologo licenciado. Farmacoterapia excluida por diseno."},

    {"input":"Extrae [Perfil diagnostico de la muestra]: 89 universitarios con ansiedad ante examenes (TAI>52), edad media 21.3 anos (DE=2.1), 67% mujeres, reclutados en 3 centros de orientacion universitaria, sin compensacion economica.",
     "output":"n=89, universitarios con ansiedad ante examen (TAI>52), M=21.3 anos (DE=2.1), 67% mujeres, 3 centros de orientacion. Sin compensacion monetaria."},

    {"input":"Extrae [Indicador psicofisiologico]: Cortisol salival (ug/dL) medido en 3 momentos: al despertar (T0), post-sesion (T1) y recuperacion 30 minutos (T2), mediante ensayo ELISA. Muestras almacenadas a -20 grados Celsius.",
     "output":"Cortisol salival (ug/dL) via ELISA: T0 (despertar), T1 (post-sesion), T2 (recuperacion 30 min). Muestras almacenadas a -20C."},

    {"input":"Extrae [Escala psicometrica principal de resultado]: Resultado primario: STAI-Y. Reduccion media: MBSR M=-18.3 (DE=4.2) vs TCC M=-11.2 (p=0.02) vs lista de espera M=-2.1 (p<0.001). Sin diferencias grupales en cortisol.",
     "output":"Escala primaria STAI-Y - Reduccion media: MBSR M=-18.3 (DE=4.2) > TCC M=-11.2 (p=0.02) > espera M=-2.1 (p<0.001). Sin diferencias en cortisol."},

    {"input":"Extrae [Interpretacion causal de los autores]: Los autores concluyen que el MBSR produce reduccion de ansiedad mas rapida que la TCC a 3 meses pero reconocen explicitamente que no pueden descartar que la TCC genere reestructuracion cognitiva mas duradera a largo plazo.",
     "output":"Los autores concluyen: MBSR reduce ansiedad mas rapidamente que TCC a 3 meses, pero reconocen la posibilidad de que TCC produzca reestructuracion cognitiva mas duradera a largo plazo."},

    {"input":"Extrae [Contrapeso de condiciones experimentales]: Diseno intra-sujeto: 60 participantes completaron tareas cognitivas bajo ruido (70dB), silencio (<30dB) y musica clasica (60dB), contrabalanceadas mediante cuadrado latino.",
     "output":"Intra-sujeto: 60 participantes x 3 condiciones (ruido 70dB, silencio <30dB, musica clasica 60dB). Contrapeso: cuadrado latino."},

    {"input":"Extrae [Terapia evaluada en ensayo unico]: La EMDR es el unico tratamiento evaluado en este ensayo aleatorizado. La revision de literatura discute la terapia de exposicion, los ISRS y la TCC como alternativas para el TEPT, pero no forman parte del diseno.",
     "output":"EVALUADA: EMDR (unica intervencion del ensayo). ENFOQUES ALTERNATIVOS (no evaluados, solo revision): terapia de exposicion, ISRS y TCC para TEPT."},

    {"input":"Extrae [Tipo de contribucion del articulo]: Articulo puramente conceptual que propone una tipologia de 5 estadios de adiccion digital. Sin recogida de datos. Contribucion: marco teorico para investigacion empirica futura.",
     "output":"Contribucion teorica/conceptual: tipologia de 5 estadios de adiccion digital. Sin datos empiricos. Marco para investigacion empirica futura."},

    {"input":"Extrae [Sesgo metodologico reconocido]: Los autores indican que el muestreo de conveniencia universitario limita la generalizabilidad clinica; las medidas de autoinforme de ansiedad son susceptibles a la deseabilidad social; la intervencion de 8 semanas puede no capturar cambio duradero.",
     "output":"Sesgos reconocidos: (1) muestreo por conveniencia - baja generalizabilidad clinica, (2) autoinforme susceptible a deseabilidad social, (3) intervencion de 8 semanas puede no capturar cambio duradero."},

    # --- Columnas debiles adicionales (psicologia) ---
    {"input":"Extrae [Estimacion del tamano del efecto psicologico]: La intervencion de mindfulness mostro un tamano del efecto grande sobre la ansiedad rasgo (d=0.82, IC95%: 0.61-1.03) y moderado sobre la ansiedad estado (d=0.54, IC95%: 0.34-0.74).",
     "output":"Efecto mindfulness: ansiedad rasgo d=0.82 (IC95%: 0.61-1.03, grande); ansiedad estado d=0.54 (IC95%: 0.34-0.74, moderado)."},

    {"input":"Extrae [Escala de bienestar psicologico empleada]: Bienestar psicologico evaluado con las Escalas de Bienestar Psicologico de Ryff (EBPR, 84 items, 6 dimensiones). Version espanola validada por Van Dierendonck (alfa total=0.83).",
     "output":"EBPR de Ryff: 84 items, 6 dimensiones. Version espanola de Van Dierendonck, alfa total=0.83."},

    {"input":"Extrae [Estrategia de reclutamiento clinico cualitativo]: Muestreo teorico en psicoterapia: participantes seleccionados segun criterio de maxima variacion (genero, edad, diagnostico), hasta alcanzar saturacion teorica en la entrevista 21.",
     "output":"Muestreo teorico de maxima variacion (genero, edad, diagnostico). Saturacion teorica alcanzada en entrevista 21."},

    # ================================================================
    # DOMINIO 4: ECONOMIA
    # ================================================================
    {"input":"Extrae [Estrategia de identificacion causal]: Diferencias en diferencias con efectos fijos bidireccionales, aprovechando incrementos escalonados del salario minimo en 12 estados de EE.UU. (2015-2022) como experimento cuasi-natural.",
     "output":"DiD con efectos fijos bidireccionales: variacion escalonada del salario minimo en 12 estados EE.UU. (2015-2022) como experimento cuasi-natural."},

    {"input":"Extrae [Politica publica evaluada]: Programa de microcredito CREDIPYME (2019) para pymes manufactureras. Los programas FONDOEMPRESA y PROEMPLEO se mencionan como potenciales confusores en las comprobaciones de robustez pero no se analizan directamente.",
     "output":"EVALUADO: CREDIPYME 2019 - microcredito para PYMEs manufactureras. MENCIONADOS (como robustez, no analizados): FONDOEMPRESA y PROEMPLEO - confundidores potenciales."},

    {"input":"Extrae [Marco muestral de las empresas]: 2,180 pymes manufactureras peruanas (ingresos <5M USD, registro SUNAT 2020). Excluidas: microempresas con menos de 5 empleados y empresas con morosos crediticios previos (2015-2019).",
     "output":"Marco muestral: 2,180 PYMEs manufactureras peruanas (ingresos <US$5M, SUNAT 2020). Exclusiones: microempresas <5 empleados y empresas con mora crediticia (2015-2019)."},

    {"input":"Extrae [Variables de resultado empresarial]: Resultado primario: supervivencia a 24 meses (binaria, datos administrativos). Secundarios: crecimiento de ingresos (%/ano, registros fiscales), variacion de empleo (planillas ESSALUD), tasa de repago (%).",
     "output":"Resultado primario: supervivencia 24 meses (binaria, datos administrativos). Secundarios: crecimiento de ingresos (%/ano, SUNAT), variacion de empleo (planillas ESSALUD), tasa de repago (%)."},

    {"input":"Extrae [Estimacion del efecto del tratamiento]: PSM-ATT: los beneficiarios del microcredito mostraron 23% mayor supervivencia a 24 meses (p<0.01) y prima de crecimiento de ingresos del 18%. Empleo: sin efecto significativo (ATT=0.8 trabajadores, p=0.31).",
     "output":"ATT (PSM): supervivencia 24m +23% (p<0.01), ingresos +18%. Empleo: sin efecto significativo (ATT=0.8 trabajadores, p=0.31)."},

    {"input":"Extrae [Modelo de series de tiempo empleado]: GARCH(1,1) para varianza condicional del VIX y retornos diarios de Bitcoin. VAR(3) para interacciones dinamicas. Prueba de causalidad de Granger. Muestra: 2017-2023 (1,560 dias).",
     "output":"Modelos: GARCH(1,1) para varianza condicional + VAR(3) para interacciones dinamicas. Prueba de causalidad de Granger. Muestra: 2017-2023 (n=1,560 dias)."},

    {"input":"Extrae [Supuesto critico del estimador y amenaza]: El supuesto de tendencias paralelas del DiD se verifica mediante graficos de estudio de eventos y el estimador Callaway-SantAnna. No se puede descartar completamente la existencia de tendencias pre-tratamiento en estados con aumento anticipado.",
     "output":"Supuesto critico: tendencias paralelas en DiD. Pruebas de robustez: event study + estimador Callaway-SantAnna. Amenaza no descartada completamente para estados con aumento anticipado."},

    {"input":"Extrae [Metricas de evaluacion de portafolios]: ROI, VPN (tasa de descuento 10%) y TIR calculados para 12 portafolios. El indice de Sharpe y el CVaR se discuten como preferibles pero no se calculan por falta de datos completos de distribucion de retornos.",
     "output":"Calculadas: ROI, VPN (tasa 10%), TIR - 12 portafolios. Sharpe y CVaR omitidos por datos insuficientes para distribucion de retornos completa."},

    {"input":"Extrae [Recomendacion de politica explicita]: Los autores recomiendan combinar el microcredito con capacitacion empresarial obligatoria (12 horas) y acceso a mercados, citando evidencia de que el credito solo no genera empleo.",
     "output":"Recomendacion explicita: combinar microcredito con capacitacion empresarial obligatoria (12h) y acceso a mercados, dado que el credito solo no genera empleo."},

    {"input":"Extrae [Restriccion de generalizabilidad economica]: El analisis se limita al sector formal (empresas registradas en SUNAT). Las pymes informales - estimadas en el 65% del sector - quedan excluidas por falta de datos administrativos. Los autores senalan esto como limitacion mayor.",
     "output":"Limitacion mayor declarada: analisis restringido al sector formal (registradas en SUNAT). PYMEs informales - estimadas ~65% del sector - excluidas por ausencia de datos administrativos."},

    # --- Columnas debiles adicionales (economia) ---
    {"input":"Extrae [Estimacion de la elasticidad precio]: La elasticidad precio de la demanda de transporte publico estimada fue de -0.31 (IC95%: -0.44 a -0.18), indicando demanda inelastica. El efecto fue mayor en usuarios de bajos ingresos (-0.52).",
     "output":"Elasticidad precio demanda transporte: -0.31 (IC95%: -0.44 a -0.18, inelastica). Mayor efecto en usuarios bajos ingresos (-0.52)."},

    {"input":"Extrae [Escala de percepcion de bienestar financiero]: Bienestar financiero medido con las 10 preguntas de la Consumer Financial Protection Bureau Financial Well-Being Scale (CFPB-FWB). Puntuaciones de 0 a 100; mayor puntuacion indica mayor bienestar.",
     "output":"CFPB-FWB: 10 items, rango 0-100 (mayor = mejor bienestar financiero)."},

    # ================================================================
    # DOMINIO 5: CIENCIAS AMBIENTALES
    # ================================================================
    {"input":"Extrae [Diseno del experimento de campo]: Diseno de bloques completos al azar en 40 parcelas andinas. 3 tratamientos: biocarbono (10t/ha), cultivos de cobertura (Vicia faba) y labranza cero. 2 ciclos de cultivo (2021-2022).",
     "output":"RCBD en 40 parcelas andinas: (1) biocarbono 10t/ha, (2) cultivos de cobertura (Vicia faba), (3) labranza cero. 2 ciclos de cultivo (2021-2022)."},

    {"input":"Extrae [Medida de conservacion marina evaluada]: Area Marina Protegida (AMP) comunitaria establecida en 2019 en la Bahia de Sechura, Peru (45 km2). Se evaluan zonas de exclusion de mareas y areas sin extraccion. La veda nacional de arrastre comercial se cita solo como contexto de politica nacional.",
     "output":"EVALUADA: AMP comunitaria de Bahia de Sechura (Peru, 45 km2, est. 2019): zonas de exclusion tidal y areas sin extraccion. CONTEXTO NACIONAL (no evaluado): veda nacional de arrastre comercial."},

    {"input":"Extrae [Red de muestreo hidrologico]: 78 estaciones de muestreo estratificadas por altitud (baja <500m, media 500-2000m, alta >2000m) en 3 periodos hidrologicos (julio-seco, noviembre-transicion, febrero-lluvioso). Cuenca del Rimac (Lima).",
     "output":"78 estaciones estratificadas por altitud (baja/media/alta) x 3 periodos hidrologicos (seco-julio, transicion-noviembre, lluvias-febrero). Cuenca del Rimac (Lima)."},

    {"input":"Extrae [Indices de diversidad biologica]: Macroinvertebrados: riqueza de especies (S), Shannon-Wiener (H'), equidad de Pielou (J'). Calidad del agua: oxigeno disuelto (mg/L), pH, turbidez (NTU), nitratos (mg/L), conductividad (uS/cm).",
     "output":"Biodiversidad: S (riqueza), H' (Shannon-Wiener), J' (equidad de Pielou). Calidad del agua: OD, pH, turbidez, NO3, conductividad."},

    {"input":"Extrae [Comparacion de tecnicas de restauracion de suelo]: Carbono organico ganado en 2 ciclos: cultivos de cobertura +41% > biocarbono +28% (p=0.015) >> labranza cero +4% (no significativo vs linea base, p=0.42).",
     "output":"Carbono organico ganado a 2 ciclos: cultivos de cobertura +41% > biocarbono +28% (p=0.015) >> labranza cero +4% (no significativo, p=0.42)."},

    {"input":"Extrae [Condicionantes de la efectividad del AMP]: Los autores concluyen que las AMP restauran la biomasa de peces en 3 anos solo cuando la adhesion comunitaria supera el 80% y la frecuencia de patrullaje gubernamental es al menos 2 veces por mes. Sin ambas condiciones, la efectividad cae aproximadamente un 60%.",
     "output":"Los autores concluyen: las AMP restauran biomasa en 3 anos si y solo si: adhesion comunitaria >80% Y patrullaje gubernamental >=2/mes. Sin ambas condiciones, efectividad cae ~60%."},

    {"input":"Extrae [Escala temporal de validez de resultados]: Los autores declaran explicitamente que las observaciones de 2 ciclos son insuficientes para afirmaciones sobre carbono del suelo que operan en escalas decadales. Recomiendan monitoreo de 5 a 10 anos.",
     "output":"Limitacion temporal declarada: 2 ciclos insuficientes para afirmaciones sobre carbono del suelo (escala decadal). Los autores recomiendan monitoreo de 5-10 anos."},

    {"input":"Extrae [Unidad funcional y sistema LCA]: Unidad funcional: 1 kg de alimento entregado al consumidor. Frontera del sistema: cuna a tumba. 5 materiales: vidrio, plastico HDPE, aluminio, carton y bioplastico PLA. Software: SimaPro 9.0 con Ecoinvent 3.8.",
     "output":"Unidad funcional: 1 kg alimento al consumidor final. Frontera: cuna a tumba. 5 materiales: vidrio, HDPE, aluminio, carton, PLA. SimaPro 9.0 + Ecoinvent 3.8."},

    {"input":"Extrae [Categorias de impacto ambiental del LCA]: Potencial de calentamiento global (kg CO2-eq), demanda acumulada de energia (MJ), huella de escasez hidrica (m3 eq) y uso de suelo (m2-ano), evaluados por unidad funcional por material.",
     "output":"Categorias de impacto evaluadas por unidad funcional: GWP (kg CO2-eq), demanda energetica acumulada (MJ), huella de escasez hidrica (m3 eq), uso de suelo (m2-ano)."},

    {"input":"Extrae [Alcance de las politicas energeticas revisadas]: Las politicas de despliegue de energia solar fotovoltaica y los subsidios eolicos offshore son las 2 intervenciones principales. Las politicas nuclear, hidroelectrica y de gas natural aparecen solo en la seccion de antecedentes como contexto de mezcla energetica de otros paises.",
     "output":"REVISADAS: politicas de despliegue solar fotovoltaico y subsidios eolicos offshore. CONTEXTO (no revisadas): politicas nuclear, hidroelectrica y gas - mezcla energetica de otros paises en el marco teorico."},

    # ================================================================
    # DOMINIO 6: TECNOLOGIA GENERAL
    # ================================================================
    {"input":"Extrae [Conjunto de algoritmos de aprendizaje automatico evaluados]: CART, SVM (nucleo RBF, C=1, gamma='scale'), MLP (2 capas ocultas, 64-32 unidades, ReLU), Random Forest (500 arboles), XGBoost (eta=0.1, profundidad maxima=6, n=500) para prediccion de reingreso a 30 dias.",
     "output":"Algoritmos: CART, SVM (RBF, C=1), MLP (64-32 unidades, ReLU), Random Forest (n=500), XGBoost (eta=0.1, depth=6, n=500) para prediccion de reingreso a 30 dias."},

    {"input":"Extrae [Artefacto tecnologico desarrollado y evaluado]: HealthTrack (app Android/iOS) para autocontrol de diabetes tipo 2 con registro de glucosa, recordatorios de medicacion y seguimiento dietetico. MyFitnessPal y Fitbit se citan en el trabajo relacionado como soluciones existentes pero no se comparan.",
     "output":"EVALUADO: HealthTrack (app Android/iOS) para autogestion diabetes T2D - modulos de glucosa, medicacion y dieta. MENCIONADOS (no evaluados): MyFitnessPal y Fitbit en trabajo relacionado."},

    {"input":"Extrae [Caracteristicas de los datos clinicos]: 1,847 registros de historia clinica electronica de un hospital terciario de Mexico (2018-2021). Inclusion: adultos mayores de 18 anos con al menos 2 ingresos previos. Datos faltantes: 8.3%, imputados con MICE (m=10).",
     "output":"1,847 registros EHR de hospital terciario Mexico (2018-2021). Inclusion: >=18 anos, >=2 ingresos previos. Datos faltantes: 8.3%, imputados con MICE (m=10)."},

    {"input":"Extrae [Protocolo de evaluacion de modelos predictivos]: Validacion cruzada de 10 particiones estratificada repetida 5 veces. Metricas: AUROC, F1-macro, Brier score, curva de calibracion. Umbral de decision: 0.5. Sin validacion externa (limitacion monocentrica).",
     "output":"Evaluacion: 10-fold VC estratificado x5 repeticiones. Metricas: AUROC, F1-macro, Brier score, curva de calibracion. Umbral=0.5. Sin validacion externa (limitacion monocentrica)."},

    {"input":"Extrae [Modelo con mejor rendimiento]: XGBoost: AUROC=0.87 (IC95%: 0.83-0.91), significativamente superior a la regresion logistica de referencia (AUROC=0.71, prueba de DeLong p<0.001). RF: 0.84, MLP: 0.82, SVM: 0.79, CART: 0.71.",
     "output":"Mejor modelo: XGBoost - AUROC=0.87 (IC95%: 0.83-0.91), supera LR baseline (AUROC=0.71, p<0.001 DeLong). RF: 0.84, MLP: 0.82, SVM: 0.79, CART: 0.71."},

    {"input":"Extrae [Paradigma de investigacion del prototipo IoT]: Investigacion en Ciencia del Diseno (DSR): diseno de arquitectura IoT para agricultura inteligente, implementacion del prototipo y evaluacion en 3 fincas reales durante 4 meses con 12 nodos de sensores cada una.",
     "output":"DSR: diseno de arquitectura IoT (agricultura inteligente) -> prototipo -> evaluacion en 3 fincas reales (4 meses, 12 nodos por finca)."},

    {"input":"Extrae [Posicion de los autores sobre costo computacional]: Los autores concluyen explicitamente que los modelos de tipo transformer superan a los baselines de aprendizaje automatico pero son computacionalmente prohibitivos para despliegue en entornos de borde (memoria >2GB, inferencia >500ms), recomendando versiones destiladas para uso en tiempo real.",
     "output":"Los autores concluyen: transformers superan baselines ML pero son prohibitivos en entornos edge (>2GB memoria, >500ms inferencia). Recomiendan versiones destiladas para uso en tiempo real."},

    {"input":"Extrae [Alcance de la evaluacion del prototipo IoT]: El sistema solo se probo en condiciones de invernadero controlado. La escalabilidad mas alla de 50 nodos, la seguridad (autenticacion y cifrado) y la tolerancia a fallos se declaran explicitamente fuera del alcance del estudio.",
     "output":"Alcance declarado: evaluacion en invernadero controlado solo. Fuera del alcance (declarado): escalabilidad >50 nodos, seguridad (autenticacion, cifrado) y tolerancia a fallos."},

    {"input":"Extrae [Pipeline NLP comparado]: Se comparan formalmente BERT-base no sensible a mayusculas, ClinicalBERT (preentrenado en MIMIC-III), BioBERT (preentrenado en PubMed) y la linea de referencia TF-IDF+SVM. Los autores mencionan GPT-4 como trabajo futuro prometedor pero sin resultados empiricos.",
     "output":"COMPARADOS: BERT-base, ClinicalBERT (MIMIC-III), BioBERT (PubMed), baseline TF-IDF+SVM. MENCIONADO (futuro, sin resultados): GPT-4."},

    {"input":"Extrae [Metricas de rendimiento operacional del sistema]: Latencia P95 (ms), rendimiento (solicitudes/seg), utilizacion de CPU (%) y energia por inferencia (mWh) medidos bajo cargas de 10, 100 y 1,000 usuarios concurrentes en una instancia AWS t3.medium.",
     "output":"Metricas operacionales (10/100/1000 usuarios en AWS t3.medium): latencia P95 (ms), throughput (req/s), CPU (%), energia/inferencia (mWh)."},

    # --- Columnas debiles adicionales (tecnologia) ---
    {"input":"Extrae [Estimacion del ahorro computacional]: La cuantizacion INT8 del modelo redujo el uso de memoria del 68% (de 2.1 GB a 0.67 GB) y mejoro la latencia de inferencia de 312ms a 89ms (3.5x mas rapido), con una degradacion del AUROC de solo 0.003.",
     "output":"Cuantizacion INT8: memoria -68% (2.1GB -> 0.67GB), latencia mejoro 3.5x (312ms -> 89ms), degradacion AUROC=0.003."},

    {"input":"Extrae [Estrategia de reclutamiento de participantes en el estudio de usabilidad]: Reclutamiento mediante anuncio en redes sociales profesionales (LinkedIn, ResearchGate), dirigido a profesionales de salud con experiencia en sistemas de informacion clinica. Se seleccionaron los primeros 30 respondentes que cumplieron los criterios de inclusion.",
     "output":"Reclutamiento en LinkedIn y ResearchGate, dirigido a profesionales de salud con experiencia en SIC. Seleccion: primeros 30 respondentes elegibles."},

    # ================================================================
    # DOMINIO 7: CIENCIAS SOCIALES
    # ================================================================
    {"input":"Extrae [Tecnica de analisis de contenido]: Analisis de contenido cuantitativo con software LIWC en 3,240 articulos de prensa (15 medios, 5 paises, 2015-2022). Unidad de analisis: articulo. Dimensiones: lenguaje afectivo y cognitivo.",
     "output":"Analisis de contenido cuantitativo (LIWC): 3,240 articulos de 15 medios en 5 paises (2015-2022). Unidad de analisis: articulo. Dimensiones: lenguaje afectivo y cognitivo."},

    {"input":"Extrae [Metodos etnograficos empleados]: Trabajo de campo de 18 meses en 3 comunidades amazonicas: observacion participante (diarios de campo), 47 entrevistas semiestructuradas (grabadas en audio, transcritas, analisis tematico) y recoleccion de artefactos.",
     "output":"Etnografia: 18 meses de campo en 3 comunidades amazonicas. Metodos: observacion participante (diarios de campo), 47 entrevistas semiestructuradas (grabadas + analisis tematico), coleccion de artefactos."},

    {"input":"Extrae [Intervencion formal evaluada vs citadas]: Taller de alfabetizacion mediatica de 12 horas evaluado con prueba de conocimiento antes y despues (d de Cohen=0.82). Las campanas en redes sociales y la reforma curricular se citan en la literatura relacionada como estrategias complementarias, pero no se prueban.",
     "output":"EVALUADO: taller de alfabetizacion mediatica (12h, pre/post test, d=0.82). CITADAS (no evaluadas): campanas en redes sociales y reforma curricular - estrategias complementarias en la literatura."},

    {"input":"Extrae [Estrategia de reclutamiento y muestreo]: Muestra aleatoria estratificada de usuarios de Instagram y TikTok de Buenos Aires (n=428, 18-35 anos, 52% mujeres), estratificada por plataforma y frecuencia de uso (diario vs ocasional). Encuesta en linea, marzo-abril 2023.",
     "output":"Muestreo aleatorio estratificado (plataforma x frecuencia): n=428, Buenos Aires, 18-35 anos, 52% mujeres. Encuesta online, marzo-abril 2023."},

    {"input":"Extrae [Instrumentos de polarizacion politica validados]: Escala de polarizacion afectiva (0-100, adaptada de Iyengar et al. 2019), indice de confianza mediatica Reuters Institute y Likert de 5 puntos de consumo de medios partidistas. Todas validadas para el contexto latinoamericano.",
     "output":"Instrumentos validados (contexto latinoamericano): escala polarizacion afectiva 0-100 (adaptada de Iyengar et al. 2019), indice confianza Reuters Institute, Likert 5pt consumo de medios partidistas."},

    {"input":"Extrae [Hallazgo principal sobre encuadre mediatico]: Los medios de derecha presentaron 3.2 veces mas encuadre negativo de politicas climaticas (chi2=48.3, gl=4, p<0.001, V de Cramer=0.18). Sin diferencias significativas de encuadre entre paises (F=1.2, p=0.31).",
     "output":"Hallazgo: medios de derecha muestran 3.2x mas encuadre negativo de politicas climaticas (chi2=48.3, p<0.001, V=0.18). Sin diferencias significativas entre paises (F=1.2, p=0.31)."},

    {"input":"Extrae [Implicacion normativa explicita]: Los autores recomiendan explicitamente que la alfabetizacion mediatica sea obligatoria como desarrollo profesional para todo el profesorado de secundaria, basandose en la mejora significativa en la deteccion de desinformacion (d=0.82) y el analisis de coste-efectividad (42 dolares por hora de capacitacion).",
     "output":"Recomendacion normativa explicita: alfabetizacion mediatica obligatoria para todo el profesorado de secundaria, con base en d=0.82 y costo-efectividad ($42 USD/hora de capacitacion)."},

    {"input":"Extrae [Reflexividad del investigador reconocida]: Los autores declaran la posicion del investigador como academico externo no indigena como posible fuente de sesgo de observacion. Afirman que la ventana de 18 meses es insuficiente para observar cambio cultural multigeneracional.",
     "output":"Reflexividad declarada: posicion del investigador como academico externo no indigena - riesgo de sesgo de observacion reconocido. Limitacion temporal: 18 meses insuficientes para cambio cultural multigeneracional."},

    {"input":"Extrae [Diseno secuencial de metodos mixtos]: Mixto secuencial explicativo: Fase 1 - encuesta en linea (n=634, muestreo por cuota por edad/genero/region). Fase 2 - grupos focales (6 grupos, n=48, muestreo intencional) para interpretar los hallazgos de la Fase 1 sobre percepcion de la IA.",
     "output":"Mixtos secuencial explicativo: Fase 1 - encuesta online (n=634, cuota por edad/genero/region). Fase 2 - grupos focales (6 grupos, n=48, muestreo intencional) para interpretar Fase 1."},

    {"input":"Extrae [Estrategia de moderacion en plataformas comparada]: Prueba A/B de 3 estrategias de moderacion: filtrado algoritmico solitario, moderacion humana e hibrido (el algoritmo senala publicaciones que pasan a revision humana). Las leyes de difamacion y la regulacion gubernamental se citan como alternativas de politica publica fuera del alcance de la prueba A/B.",
     "output":"A/B test de 3 estrategias: (1) filtrado algoritmico puro, (2) moderacion humana, (3) hibrido (algoritmo + cola de revision humana). FUERA DEL ALCANCE (citadas): leyes de difamacion y regulacion gubernamental."},

    # --- Columnas debiles adicionales (ciencias sociales) ---
    {"input":"Extrae [Estimacion de la brecha de genero]: Las mujeres obtuvieron 8.3 puntos menos que los hombres en la prueba de razonamiento matematico (IC95%: 4.1-12.5, p<0.001, d=0.41). La brecha se redujo en un 60% cuando se controlo por acceso a recursos educativos.",
     "output":"Brecha de genero en razonamiento matematico: -8.3 puntos (IC95%: 4.1-12.5, p<0.001, d=0.41). Brecha se reduce 60% al controlar por acceso a recursos educativos."},

    {"input":"Extrae [Escala de capital social aplicada]: Capital social medido con la Escala de Redes Sociales de Wenger (RAS, 15 items, 5 dimensiones), validada en contexto hispanohablante (alpha=0.81). Puntuacion de 0 a 60; mayor puntuacion indica mayor integracion social.",
     "output":"RAS de Wenger: 15 items, 5 dimensiones, validada en espanol (alpha=0.81). Rango 0-60 (mayor = mayor integracion social)."},
    # ================================================================
    # DOMINIO 8: EJEMPLOS [SIN INFORMACION] — ANTI-ALUCINACION
    # Ensenian al modelo a responder cuando el texto NO contiene datos
    # para la columna pedida. Critico para fidelidad en produccion.
    # ================================================================
    {"input": "Extrae [Modelo predictivo utilizado]: Este articulo analiza el impacto del salario minimo en el empleo. Los autores utilizaron un diseno de diferencias en diferencias con datos de panel de 12 estados.",
     "output": "[SIN INFORMACION] El texto no describe ningun modelo predictivo; el diseno es cuasi-experimental (DiD)."},

    {"input": "Extrae [Escala psicometrica de ansiedad]: El estudio evalua el efecto de la lluvia acida en biodiversidad de macroinvertebrados en rios andinos usando indices de Shannon-Wiener y Pielou.",
     "output": "[SIN INFORMACION] El texto no contiene escalas psicometricas; se trata de un estudio ambiental con indices ecologicos."},

    {"input": "Extrae [Tasa de mortalidad cardiovascular]: Los autores comparan tres algoritmos de machine learning (Random Forest, XGBoost, SVM) para predecir el abandono escolar en educacion secundaria.",
     "output": "[SIN INFORMACION] El texto no reporta datos de mortalidad cardiovascular; el dominio es prediccion educativa."},

    {"input": "Extrae [Variables de resultado farmacologico]: Este articulo desarrolla un marco conceptual para analizar la adopcion de tecnologia movil en PYMEs de paises en desarrollo. No incluye datos empiricos.",
     "output": "[SIN INFORMACION] El articulo es teorico/conceptual y no reporta variables de resultado farmacologico."},

    {"input": "Extrae [Endpoint primario del ensayo]: La investigacion utiliza analisis de redes sociales para estudiar la difusion de desinformacion en Twitter durante elecciones presidenciales en Mexico 2024.",
     "output": "[SIN INFORMACION] El texto no es un ensayo clinico; no existe endpoint primario. Se trata de un estudio observacional en redes sociales."},

    {"input": "Extrae [Dosis del farmaco evaluado]: El articulo realiza una revision sistematica de estrategias pedagogicas para la educacion STEM en educacion primaria, analizando 45 estudios de 2010 a 2023.",
     "output": "[SIN INFORMACION] El articulo es una revision sistematica en educacion; no evalua farmacos ni dosis."},

    {"input": "Extrae [Modelo GARCH o VAR utilizado]: El estudio compara la eficacia de tres intervenciones psicologicas (TCC, ACT, mindfulness) para el tratamiento del trastorno de ansiedad generalizada en adultos.",
     "output": "[SIN INFORMACION] El texto describe un ECA de intervenciones psicologicas; no utiliza modelos de series de tiempo financieras."},

    {"input": "Extrae [Indice de Shannon-Wiener de biodiversidad]: El articulo evalua el desempeno de cuatro modelos de lenguaje grande (GPT-4, Claude, Gemini, Llama-3) en tareas de razonamiento matematico.",
     "output": "[SIN INFORMACION] El articulo compara LLMs para razonamiento matematico; no reporta indices de biodiversidad."},

    {"input": "Extrae [Tamano del efecto de la intervencion]: Los autores describen el diseno y la implementacion de un prototipo de sensor IoT para monitoreo de humedad del suelo en cultivos de papa. No presentan resultados de campo.",
     "output": "[SIN INFORMACION] El articulo describe el diseno del prototipo pero no reporta resultados empiricos ni tamano del efecto."},

    {"input": "Extrae [Criterios de inclusion y exclusion de participantes]: Este trabajo propone una arquitectura de microservicios para sistemas de recomendacion en e-commerce. Es un articulo de ingenieria de software sin participantes humanos.",
     "output": "[SIN INFORMACION] No existen criterios de inclusion/exclusion: el articulo no involucra participantes humanos; es un estudio tecnico de arquitectura de software."},

    {"input": "Extrae [Estrategia de reclutamiento de pacientes]: El articulo presenta un analisis bibliometrico de la produccion cientifica sobre inteligencia artificial en el Peru durante 2015-2023 usando VOSviewer.",
     "output": "[SIN INFORMACION] El articulo es un bibliometrico; no recluta pacientes ni participantes experimentales."},

    {"input": "Extrae [Metricas de rendimiento del modelo de ML]: El estudio es una etnografia de 18 meses sobre practicas de subsistencia en comunidades indigenas amazonicas. Metodos: observacion participante y entrevistas.",
     "output": "[SIN INFORMACION] El articulo es un estudio etnografico cualitativo; no desarrolla ni evalua modelos de machine learning."},

    {"input": "Extrae [Tamano de la muestra del estudio]: El articulo discute la importancia de definir correctamente los criterios de inclusion en estudios clinicos y propone una guia metodologica. No reporta un estudio empirico propio.",
     "output": "[SIN INFORMACION] El articulo es metodologico/conceptual; no reporta un estudio empirico con muestra propia."},

    {"input": "Extrae [Resultado primario de la intervencion educativa]: La investigacion describe el desarrollo y validacion de un cuestionario para medir competencias digitales en docentes universitarios (fase de diseno, n=8 expertos).",
     "output": "[SIN INFORMACION] El articulo reporta la fase de diseno/validacion del instrumento, no resultados de una intervencion educativa implementada."},

    {"input": "Extrae [Porcentaje de reduccion de CO2 logrado]: El estudio es una revision narrativa de politicas de precios al carbono en America Latina, discutiendo sus limitaciones y oportunidades. No reporta datos empiricos originales.",
     "output": "[SIN INFORMACION] El articulo es una revision narrativa de politicas; no reporta datos empiricos de reduccion de CO2."},

    {"input": "Extrae [Escala de depresion utilizada en el screening]: El articulo analiza la viabilidad financiera de proyectos de energia solar en zonas rurales de Bolivia usando TIR, VPN y periodo de retorno.",
     "output": "[SIN INFORMACION] El articulo es de evaluacion financiera de proyectos energeticos; no utiliza escalas de depresion."},

    {"input": "Extrae [Algoritmo de clasificacion de imagenes medicas]: Los autores realizan un experimento de campo evaluando el efecto de tres densidades de siembra en el rendimiento del arroz en la costa del Peru.",
     "output": "[SIN INFORMACION] El articulo es un experimento agricola de campo; no desarrolla ni aplica algoritmos de clasificacion de imagenes medicas."},

    {"input": "Extrae [Intervalo de confianza del efecto principal]: Los autores concluyen que la intervencion fue efectiva y estadisticamente significativa (p<0.05), pero no reportan el intervalo de confianza en el resumen ni en los resultados principales.",
     "output": "[SIN INFORMACION] El texto reporta significancia estadistica (p<0.05) pero no proporciona el intervalo de confianza del efecto principal."},

    {"input": "Extrae [Nombre del dataset publico utilizado]: El articulo describe un sistema de deteccion de fraude bancario usando redes neuronales. Los autores mencionan que usaron datos reales de un banco europeo bajo acuerdo de confidencialidad.",
     "output": "[SIN INFORMACION] Los datos son propietarios bajo confidencialidad; no se utiliza un dataset publico nombrable."},

    {"input": "Extrae [Porcentaje de acuerdo entre evaluadores]: El articulo reporta el proceso de analisis de contenido pero omite los indices de confiabilidad entre codificadores, mencionando solo que 'se alcanzaron acuerdos satisfactorios'.",
     "output": "[SIN INFORMACION] El texto menciona acuerdos entre evaluadores de forma cualitativa pero no reporta el porcentaje o coeficiente especifico (ej. kappa de Cohen)."},

    {"input": "Extrae [Protocolo de quimioterapia evaluado]: Este articulo estudia la adopcion de practicas agiles (Scrum, Kanban) en empresas de desarrollo de software en Chile. Analiza 120 equipos de desarrollo.",
     "output": "[SIN INFORMACION] El articulo es sobre metodologias agiles en desarrollo de software; no evalua protocolos de quimioterapia ni ningun tratamiento oncologico."},

    {"input": "Extrae [Tasa de supervivencia a 5 anos]: El estudio analiza el sentido del humor como estrategia pedagogica en el aula universitaria, usando grupos focales (n=6) y observacion de clases.",
     "output": "[SIN INFORMACION] El articulo es un estudio cualitativo sobre pedagogia y humor; no reporta tasas de supervivencia clinica."},

    {"input": "Extrae [Coeficiente de Gini del pais]: El articulo evalua la eficacia de un chatbot de IA para soporte tecnico en una empresa de telecomunicaciones, midiendo satisfaccion del usuario (CSAT) y tiempo de resolucion.",
     "output": "[SIN INFORMACION] El articulo es sobre evaluacion de chatbot corporativo; no reporta el coeficiente de Gini ni datos de desigualdad economica."},

    {"input": "Extrae [Criterios DSM-5 para el diagnostico]: El trabajo presenta un modelo matematico para optimizar rutas de distribucion de vacunas en zonas rurales usando programacion entera mixta.",
     "output": "[SIN INFORMACION] El articulo es de investigacion de operaciones y logistica; no aplica criterios DSM-5 ni realiza diagnosticos psiquiatricos."},

    {"input": "Extrae [Fraccion de eyeccion ventricular izquierda]: El articulo analiza el impacto de las redes sociales en la autoestima de adolescentes usando la Escala de Autoestima de Rosenberg (EAR) en 340 estudiantes.",
     "output": "[SIN INFORMACION] El articulo estudia autoestima en adolescentes; no mide parametros cardiacos como la fraccion de eyeccion ventricular."},

    {"input": "Extrae [Tipo de reactor quimico utilizado]: La investigacion realiza un analisis de correspondencias multiples sobre los factores asociados a la desercion universitaria en universidades publicas del Peru.",
     "output": "[SIN INFORMACION] El articulo es un estudio estadistico sobre desercion universitaria; no describe ningun reactor quimico."},


]
