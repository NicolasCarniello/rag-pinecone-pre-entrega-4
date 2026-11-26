"""Genera data/changelog.pdf (documento de ejemplo de varias páginas).

Se versiona el PDF ya generado; este script queda para que sea reproducible:
    pip install reportlab && python scripts/generar_changelog_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "changelog.pdf"

PAGES = [
    ("andespay-python — Changelog (versiones 2.4 y 2.3)", [
        ("Versión 2.4.0 — marzo de 2026", [
            "Nuevo: AsyncAndesPayClient soporta HTTP/2 con el extra andespay[async].",
            "Nuevo: client.payments.search() permite buscar pagos por metadata con una sintaxis de consulta.",
            "Nuevo: WebhookVerifier acepta una lista de signing secrets para rotarlos sin cortes.",
            "Cambio: el valor por defecto de max_retries pasó de 3 a 2.",
            "Corrección: RateLimitError ahora expone el atributo retry_after como entero.",
        ]),
        ("Versión 2.3.0 — noviembre de 2025", [
            "Cambio incompatible: client.refunds.list() usa paginación por cursor. Se eliminaron los parámetros page y per_page; usar cursor y limit.",
            "Nuevo: el parámetro max_items en todos los métodos list.",
            "Deprecado: PaymentIntent.confirm() queda deprecado en favor de client.payments.create(capture=True). Emite DeprecationWarning y se eliminará en la 3.0.",
            "Nuevo: el comando andespay trigger del CLI para disparar eventos de prueba.",
        ]),
    ]),
    ("Versiones 2.2 y 2.1", [
        ("Versión 2.2.0 — julio de 2025", [
            "Nuevo: soporte para la moneda UYU.",
            "Nuevo: client.with_options() para cambiar timeout y max_retries en una sola llamada.",
            "Cambio: la tolerancia por defecto de WebhookVerifier bajó de 600 a 300 segundos.",
            "Corrección: el SDK ya no reintenta peticiones POST que no tienen idempotency_key; ahora genera una automáticamente.",
        ]),
        ("Versión 2.1.0 — marzo de 2025", [
            "Nuevo: el logger estándar andespay y la variable de entorno ANDESPAY_LOG.",
            "Nuevo: EnvironmentMismatchError cuando la clave no corresponde al entorno.",
            "Corrección: las API keys se enmascaran en los mensajes de excepción.",
        ]),
    ]),
    ("Versión 2.0 y política de soporte", [
        ("Versión 2.0.0 — octubre de 2024", [
            "Cambio incompatible: se requiere Python 3.10 o superior (antes 3.8).",
            "Cambio incompatible: se reemplazó requests por httpx y los modelos pasaron a pydantic v2.",
            "Cambio incompatible: los montos son enteros en centavos; los float lanzan ValidationError (AP-2003).",
            "Nuevo: cliente asíncrono AsyncAndesPayClient.",
        ]),
        ("Política de soporte de versiones", [
            "Cada versión mayor recibe correcciones de seguridad durante 18 meses desde la salida de la siguiente versión mayor.",
            "La rama 1.x dejó de recibir actualizaciones en abril de 2026.",
            "Las funciones deprecadas emiten DeprecationWarning durante al menos una versión menor antes de eliminarse.",
        ]),
    ]),
]


def main() -> None:
    styles = getSampleStyleSheet()
    story = []
    for p, (title, sections) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Title"]))
        story.append(Paragraph(
            "Documentación ficticia de una librería de Python, creada como dataset de ejemplo para el sistema RAG.",
            styles["Italic"]))
        for heading, items in sections:
            story.append(Spacer(1, 12))
            story.append(Paragraph(heading, styles["Heading2"]))
            for item in items:
                story.append(Paragraph(f"• {item}", styles["BodyText"]))
        if p < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(OUT), pagesize=A4, title="andespay-python changelog").build(story)
    print(f"Generado {OUT}")


if __name__ == "__main__":
    main()
