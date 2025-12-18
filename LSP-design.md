> Diseño de integración con LSP (propuesta, sin código ni dependencias)

  1) Interfaz conceptual DiagnosticsProvider

  - Responsabilidad: abstraer cualquier fuente de diagnósticos (LSP, analizador propio,
  herramientas de build/test).
  - Contrato conceptual (no código):
      - name: identificador de la fuente (ej. “pyright”, “tsserver”, “pytest-check”).
      - supported_languages: lista de lenguajes o vacía si es genérico.
      - fetch_diagnostics(workdir, files=None) -> List[Diagnostic]: produce diagnósticos con
  su metadata; puede ser incremental o full.
      - Debe declarar alcance de los diagnósticos (file-level, project-level, global).
      - Debe proporcionar una estimación de confianza (autoasignada por la fuente) y permitir
  degradarla externamente.
      - No ejecuta acciones; solo entrega señales.

  2) Modelo de diagnóstico (neutral, desacoplado)

  - Diagnostic (estructura conceptual):
      - source: “lsp:<server>”, “analyzer”, “tool:<name>”.
      - severity: {error, warning, info, hint}.
      - message: texto humano resumido.
      - location: {file, line, column} opcional; puede ser project o global para alcance
  mayor.
      - confidence: {high, medium, low} asignado por la fuente o ajustado por políticas.
      - suggested_action (opcional): texto corto (“revisar import”, “ejecutar tests”, “ignorar
  si intencional”).
      - evidence: breve referencia a la señal (ej. “LSP semantic token”, “compilador”,
  “análisis heurístico”).
      - No se asume veracidad; es una hipótesis con contexto.

  3) Política de confianza

  - Reglas explícitas (conceptuales):
      - Severidad mayor → prioriza revisión, pero no ejecución automática.
      - Múltiples fuentes coherentes sobre el mismo archivo/fragmento → aumenta confianza.
      - Contradicciones con ProjectState (p.ej., archivo inexistente, ya corregido) → degradar
  confianza o ignorar.
      - Diagnósticos sin ubicación clara o sin reproducibilidad → tratar como hints (baja
  confianza).
      - Antes de actuar, pedir confirmación humana si:
          - implica cambio destructivo
          - severidad baja/mediana sin soporte de más evidencia
          - hay contradicción o ambigüedad entre diagnósticos
      - El agente puede descartar diagnósticos individuales o agruparlos por archivo/tema
  para priorizar.

  4) Integración en el ciclo análisis → plan → acción → evaluación

  - Análisis: recopilar diagnósticos como insumos, etiquetarlos con confianza y severidad.
  - Plan: usar diagnósticos para priorizar archivos/problemas, pero siempre proponer pasos con
  justificación (“LSP reporta X en Y, confianza media”).
  - Acción: ejecutar solo tras confirmación o cuando la confianza es alta y el cambio es
  seguro; la acción puede ser “no hacer nada” o “pedir aclaración”.
  - Evaluación: verificar si el diagnóstico desaparece o mejora la evidencia; si persiste,
  reconsiderar, pedir guía, o etiquetar como posiblemente falso positivo.

  5) Comportamientos seguros y trazables

  - El agente puede:
      - Rechazar diagnósticos de confianza baja o contradictorios.
      - Agrupar y resumir diagnósticos para no abrumar.
      - Pedir confirmación humana antes de actuar, especialmente con cambios amplios.
      - Detenerse si no hay evidencia suficiente o si los diagnósticos son ruidosos.
  - Humildad: siempre comunicar que los diagnósticos son hipótesis, no hechos; preferir “no
  claro” antes que asumir.
  - Trazabilidad: registrar qué diagnósticos influyeron en qué decisiones, y qué se descartó
  (y por qué).

  Flujo ejemplo (conceptual)

  1. DiagnosticsProvider(pyright) devuelve: error en src/app.py:42, confianza alta.
  2. Agente en fase de plan: “Priorizar src/app.py porque pyright reporta error (alta).”
  3. Acción propuesta: revisar bloque señalado; si el cambio es seguro y confirmado, aplicar;
  si no, pedir guía.
  4. Evaluación: re-correr diagnósticos (o re-run tests). Si desaparece, marcar como resuelto;
  si persiste, degradar confianza o pedir intervención humana.

  Alcance y límites

  - Sin LSP implementado ni dependencias nuevas.
  - Sin cambios de comportamiento actual; es un contrato de diseño listo para futura
  implementación.
  - Mantiene la filosofía de humildad y control humano.

