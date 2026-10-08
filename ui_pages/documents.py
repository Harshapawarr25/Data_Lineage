from __future__ import annotations

import hashlib
import io

import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ui_pages.chrome import page_title


def render() -> None:
    if "project_documents" not in st.session_state:
        st.session_state["project_documents"] = {}
    if "project_pdfs_widget_version" not in st.session_state:
        st.session_state["project_pdfs_widget_version"] = 0

    page_title(
        "Project document library",
        "Upload searchable PDFs, scan extracted text, and keep project context close to your analysis.",
    )
    uploaded_pdfs = st.file_uploader(
        "Add PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"project-pdfs-{st.session_state['project_pdfs_widget_version']}",
        help="Text-based PDFs are supported. Scanned pages need OCR before their text can be searched.",
    )
    for uploaded_pdf in uploaded_pdfs:
        pdf_bytes = uploaded_pdf.getvalue()
        document_id = hashlib.sha256(pdf_bytes).hexdigest()
        if document_id in st.session_state["project_documents"]:
            continue
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
            if reader.is_encrypted:
                st.error(
                    f"{uploaded_pdf.name} is encrypted. Upload an unlocked PDF."
                )
                continue
            page_texts = [page.extract_text() or "" for page in reader.pages]
        except (PdfReadError, ValueError, EOFError) as error:
            st.error(f"Could not read {uploaded_pdf.name}: {error}")
            continue
        metadata = reader.metadata
        st.session_state["project_documents"][document_id] = {
            "name": uploaded_pdf.name,
            "title": metadata.title if metadata and metadata.title else uploaded_pdf.name,
            "page_texts": page_texts,
            "page_count": len(page_texts),
            "text": "\n\n".join(
                f"--- Page {index + 1} ---\n{text}"
                for index, text in enumerate(page_texts)
            ),
        }

    documents = st.session_state["project_documents"]
    if not documents:
        st.info("Your library is empty. Choose one or more PDFs above.")
        return
    query = st.text_input(
        "Search across project PDFs",
        placeholder="Search a phrase, requirement, or keyword...",
    ).strip()
    st.caption(f"{len(documents)} document(s) in this session")
    for document_id, document in list(documents.items()):
        matching_pages = [
            (index, text)
            for index, text in enumerate(document["page_texts"])
            if not query or query.casefold() in text.casefold()
        ]
        if query and not matching_pages:
            continue
        with st.container(border=True):
            details, actions = st.columns([4, 1])
            with details:
                st.subheader(document["title"])
                st.caption(f"{document['name']} - {document['page_count']} page(s)")
            with actions:
                if st.button("Remove", key=f"remove-{document_id}"):
                    del st.session_state["project_documents"][document_id]
                    st.session_state["project_pdfs_widget_version"] += 1
                    st.rerun()
            if not any(document["page_texts"]):
                st.warning(
                    "No selectable text was found. This may be a scanned PDF; "
                    "run OCR and upload the searchable copy."
                )
            elif query:
                st.markdown(f"**Matches on {len(matching_pages)} page(s)**")
                for page_index, text in matching_pages[:10]:
                    match_index = text.casefold().find(query.casefold())
                    start = max(0, match_index - 180)
                    excerpt = text[start : match_index + len(query) + 260]
                    st.caption(f"Page {page_index + 1}")
                    st.text(excerpt.strip())
                if len(matching_pages) > 10:
                    st.caption("Showing the first 10 matching pages.")
            else:
                preview_text = next(
                    (text.strip() for text in document["page_texts"] if text.strip()),
                    "",
                )
                st.caption("Extracted text preview")
                st.text(preview_text[:1000] + ("…" if len(preview_text) > 1000 else ""))
            st.download_button(
                "Download extracted text",
                data=document["text"].encode("utf-8"),
                file_name=f"{document['name'].rsplit('.', 1)[0]}.txt",
                mime="text/plain",
                key=f"download-{document_id}",
            )
    if query and not any(
        query.casefold() in document["text"].casefold()
        for document in documents.values()
    ):
        st.info("No matching text found in the current PDF library.")
