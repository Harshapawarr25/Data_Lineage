from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import services.storage as storage


def _text_pdf(text: str) -> bytes:
    content = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 300 144] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    cross_reference = len(document)
    document.extend(
        f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    )
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode())
    document.extend(
        f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\n"
        f"startxref\n{cross_reference}\n%%EOF".encode()
    )
    return bytes(document)


def test_gated_app_flow_persists_and_exposes_dashboard(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIRECTORY", tmp_path)
    script = Path(__file__).resolve().parents[1] / "main.py"
    app = AppTest.from_file(script).run(timeout=30)
    assert not app.exception
    assert "Data Quality" not in app.radio[0].options

    app.get("file_uploader")[0].set_value(
        (
            "reference.csv",
            b"region,churned,monthly_charges\n"
            b"North,0,10\nSouth,1,20\nNorth,0,11\nSouth,1,21\n",
            "text/csv",
        )
    ).run(timeout=30)
    app.get("file_uploader")[1].set_value(
        (
            "current.csv",
            b"region,churned,monthly_charges\n"
            b"Central,0,-2\nSouth,1,30\nNorth,0,11\nCentral,1,40\n",
            "text/csv",
        )
    ).run(timeout=30)
    next(
        button for button in app.button if button.label.startswith("Run Analysis")
    ).click().run(timeout=30)
    assert not app.exception
    assert "Data Quality" in app.radio[0].options
    assert "Drift Detection" not in app.radio[0].options

    next(
        button for button in app.button if button.label == "Run quality checks"
    ).click().run(timeout=30)
    assert not app.exception
    assert "Drift Detection" in app.radio[0].options
    assert "Dashboard" not in app.radio[0].options

    next(
        button for button in app.button if button.label == "Run drift checks"
    ).click().run(timeout=30)
    assert not app.exception
    assert "Dashboard" in app.radio[0].options
    app.radio[0].set_value("Dashboard").run(timeout=30)
    assert not app.exception
    assert any(
        item.label == "Download self-contained HTML report"
        for item in app.get("download_button")
    )
    app.radio[0].set_value("Project Library").run(timeout=30)
    assert not app.exception
    assert len(app.get("file_uploader")) == 1
    app.get("file_uploader")[0].set_value(
        (
            "project-notes.pdf",
            _text_pdf("Oberva searchable test phrase"),
            "application/pdf",
        )
    ).run(timeout=30)
    assert not app.exception
    assert len(app.session_state["project_documents"]) == 1
    document = next(iter(app.session_state["project_documents"].values()))
    assert "Oberva searchable test phrase" in document["text"]
    app.get("text_input")[0].set_value("searchable test").run(timeout=30)
    assert any("searchable test" in item.value for item in app.get("text"))

    runs = storage.list_runs()
    assert len(runs) == 1
    run_id = runs[0]["run_id"]
    saved_current = pd.read_csv(runs[0]["current_path"])
    assert saved_current.loc[0, "region"] == "Central"
    assert storage.get_run(run_id)["status"] == "drift_complete"

    refreshed_app = AppTest.from_file(script).run(timeout=30)
    assert not refreshed_app.exception
    assert "Dashboard" in refreshed_app.radio[0].options
    refreshed_app.radio[0].set_value("Data Lineage").run(timeout=30)
    assert not refreshed_app.exception
