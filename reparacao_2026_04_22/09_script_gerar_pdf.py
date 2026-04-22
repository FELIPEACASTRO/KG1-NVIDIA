"""
Script para consolidar reparacao_2026_04_22 em um PDF unico para anexo em reclamacoes.

Uso:
    pip install reportlab markdown
    python 09_script_gerar_pdf.py

Output:
    reparacao_2026_04_22/RELATORIO_COMPLETO_FELIPE.pdf
"""

import os
import sys
from pathlib import Path

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table,
        TableStyle, KeepTogether,
    )
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
except ImportError:
    print("ERRO: instale reportlab: pip install reportlab")
    sys.exit(1)


PASTA = Path(__file__).parent
OUTPUT = PASTA / "RELATORIO_COMPLETO_FELIPE.pdf"


# ============================================================
# Configurar estilo
# ============================================================
styles = getSampleStyleSheet()

estilo_titulo = ParagraphStyle(
    "Titulo", parent=styles["Heading1"],
    fontName="Helvetica-Bold", fontSize=18, leading=22,
    textColor=HexColor("#1a1a1a"), spaceAfter=14, alignment=TA_CENTER,
)

estilo_h1 = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontName="Helvetica-Bold", fontSize=14, leading=18,
    textColor=HexColor("#1a237e"), spaceBefore=14, spaceAfter=8,
)

estilo_h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=12, leading=16,
    textColor=HexColor("#283593"), spaceBefore=10, spaceAfter=6,
)

estilo_body = ParagraphStyle(
    "Body", parent=styles["BodyText"],
    fontName="Helvetica", fontSize=10, leading=14,
    alignment=TA_JUSTIFY, spaceAfter=6,
)

estilo_code = ParagraphStyle(
    "Code", parent=styles["BodyText"],
    fontName="Courier", fontSize=8, leading=10,
    backColor=HexColor("#f5f5f5"), borderColor=HexColor("#ccc"),
    borderWidth=0.5, borderPadding=4,
    leftIndent=6, rightIndent=6, spaceAfter=4,
)


# ============================================================
# Parser simples de Markdown para Paragraph
# ============================================================
def md_to_elements(texto):
    """Converte Markdown basico em lista de elementos ReportLab."""
    elementos = []
    linhas = texto.split("\n")
    i = 0
    while i < len(linhas):
        linha = linhas[i].rstrip()

        # Codigo em bloco
        if linha.startswith("```"):
            i += 1
            code_lines = []
            while i < len(linhas) and not linhas[i].startswith("```"):
                code_lines.append(linhas[i])
                i += 1
            if code_lines:
                codigo_texto = "<br/>".join(
                    l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace(" ", "&nbsp;")
                    for l in code_lines
                )
                elementos.append(Paragraph(codigo_texto, estilo_code))
            i += 1
            continue

        # Headings
        if linha.startswith("### "):
            elementos.append(Paragraph(
                linha[4:].replace("&", "&amp;"), estilo_h2,
            ))
        elif linha.startswith("## "):
            elementos.append(Paragraph(
                linha[3:].replace("&", "&amp;"), estilo_h1,
            ))
        elif linha.startswith("# "):
            elementos.append(Paragraph(
                linha[2:].replace("&", "&amp;"), estilo_h1,
            ))
        # Lista
        elif linha.startswith("- ") or linha.startswith("* "):
            item = linha[2:].replace("&", "&amp;")
            item = item.replace("**", "")  # strip bold markers
            elementos.append(Paragraph(f"&bull; {item}", estilo_body))
        # Linha em branco
        elif not linha.strip():
            elementos.append(Spacer(1, 0.15 * cm))
        # Body
        else:
            # Escape + bold
            texto_linha = linha.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Simple bold parsing
            while "**" in texto_linha:
                texto_linha = texto_linha.replace("**", "<b>", 1)
                texto_linha = texto_linha.replace("**", "</b>", 1)
            elementos.append(Paragraph(texto_linha, estilo_body))

        i += 1
    return elementos


# ============================================================
# Construir PDF
# ============================================================
def build_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        title="Reparacao Felipe - Caso Anthropic/Colab 2026-04-22",
        author="Felipe Andrade de Castro",
    )

    story = []

    # Capa
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph(
        "RELATORIO CONSOLIDADO<br/>"
        "CASO FELIPE ANDRADE DE CASTRO<br/>"
        "vs. ANTHROPIC PBC / GOOGLE (COLAB PRO+)",
        estilo_titulo,
    ))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "Data do relatorio: 22 de abril de 2026<br/>"
        "Periodo dos fatos: 21-22 de abril de 2026<br/>"
        "Fundamento legal: CDC (Lei 8.078/90) Art. 14, 20, 37",
        ParagraphStyle("SubTitle", parent=estilo_body, alignment=TA_CENTER, fontSize=11),
    ))
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(
        "Documento destinado a: Procon estadual / "
        "consumidor.gov.br / Juizado Especial Civel do domicilio do Autor / "
        "Anthropic Legal (legal@anthropic.com) / "
        "Google Colab Support (colab-help@google.com)",
        ParagraphStyle("Dest", parent=estilo_body, alignment=TA_CENTER),
    ))
    story.append(PageBreak())

    # Indice
    story.append(Paragraph("INDICE", estilo_h1))
    indice = [
        "1. Resumo executivo (INDEX)",
        "2. Cronologia tecnica dos defeitos",
        "3. Planilha detalhada de prejuizo",
        "4. Mensagem enviada Anthropic in-app (registro)",
        "5. Email enviado Colab (registro)",
        "6. Reclamacao consumidor.gov.br (texto submetido)",
        "7. Escalacao pre-judicial Anthropic Legal (se aplicavel)",
        "8. Peticao inicial JEC (se aplicavel)",
    ]
    for i in indice:
        story.append(Paragraph(i, estilo_body))
    story.append(PageBreak())

    # Para cada arquivo, adicionar ao PDF
    arquivos_ordenados = [
        ("00_INDEX.md", "1. RESUMO EXECUTIVO"),
        ("07_cronologia_tecnica.md", "2. CRONOLOGIA TECNICA DOS DEFEITOS"),
        # Planilha sera renderizada separadamente como tabela
        ("01_mensagem_anthropic_inapp.md", "4. MENSAGEM ANTHROPIC IN-APP"),
        ("03_email_colab_refund.md", "5. EMAIL COLAB PRO+"),
        ("04_reclamacao_consumidor_gov.md", "6. RECLAMACAO CONSUMIDOR.GOV.BR"),
        ("02_email_anthropic_legal.md", "7. ESCALACAO PRE-JUDICIAL"),
        ("05_peticao_jec.md", "8. PETICAO INICIAL JEC"),
    ]

    for fname, section_title in arquivos_ordenados:
        fpath = PASTA / fname
        if not fpath.exists():
            continue

        story.append(Paragraph(section_title, estilo_h1))
        story.append(Spacer(1, 0.3 * cm))

        texto = fpath.read_text(encoding="utf-8")
        elementos = md_to_elements(texto)
        story.extend(elementos)
        story.append(PageBreak())

    # Planilha de prejuizo como tabela
    story.append(Paragraph("3. PLANILHA DETALHADA DE PREJUIZO", estilo_h1))
    story.append(Spacer(1, 0.3 * cm))
    csv_path = PASTA / "06_planilha_prejuizo.csv"
    if csv_path.exists():
        import csv as csv_module

        with open(csv_path, encoding="utf-8") as f:
            reader = csv_module.reader(f)
            rows = list(reader)

        if rows:
            # Truncar colunas longas para caber na pagina
            max_col_len = 40
            rows_trunc = [
                [
                    (c if len(c) < max_col_len else c[: max_col_len - 3] + "...")
                    for c in row
                ]
                for row in rows
            ]
            t = Table(rows_trunc[:30], repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#283593")),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(t)
    story.append(PageBreak())

    # Build
    doc.build(story)
    print(f"[OK] PDF gerado: {OUTPUT}")
    print(f"     Tamanho: {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build_pdf()
