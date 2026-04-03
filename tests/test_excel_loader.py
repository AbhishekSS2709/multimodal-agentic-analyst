"""Tests for src/ingestion/excel_loader.py — written before implementation (TDD)."""

from __future__ import annotations

import pytest
import openpyxl

from src.ingestion.excel_loader import load_excel
from src.ingestion.pdf_loader import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_xlsx(tmp_path):
    """Create a .xlsx with 2 sheets, each containing header + data rows."""
    xlsx_path = tmp_path / "sample.xlsx"
    wb = openpyxl.Workbook()

    # Sheet 1: Sales
    ws1 = wb.active
    ws1.title = "Sales"
    ws1.append(["Product", "Revenue"])
    ws1.append(["Widget A", 1000])
    ws1.append(["Widget B", 2000])

    # Sheet 2: Inventory
    ws2 = wb.create_sheet("Inventory")
    ws2.append(["Item", "Quantity", "Location"])
    ws2.append(["Bolt", 500, "Shelf A"])
    ws2.append(["Nut", 300, "Shelf B"])

    wb.save(str(xlsx_path))
    return str(xlsx_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadExcelReturnsDocuments:
    """load_excel returns one Document per sheet."""

    def test_load_excel_returns_documents(self, sample_xlsx):
        result = load_excel(sample_xlsx)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(doc, Document) for doc in result)


class TestLoadExcelExtractsData:
    """load_excel formats sheet content with headers and row values."""

    def test_load_excel_extracts_data(self, sample_xlsx):
        result = load_excel(sample_xlsx)
        texts = [doc.text for doc in result]

        # Sheet names appear as section headers
        assert any("Sales" in t for t in texts)
        assert any("Inventory" in t for t in texts)

        # Column headers are present
        assert any("Product" in t and "Revenue" in t for t in texts)
        assert any("Item" in t and "Quantity" in t for t in texts)

        # Row data is present
        assert any("Widget A" in t for t in texts)
        assert any("Bolt" in t for t in texts)


class TestLoadExcelMetadata:
    """load_excel sets required metadata fields for each sheet document."""

    def test_load_excel_metadata(self, sample_xlsx):
        result = load_excel(sample_xlsx)
        assert len(result) == 2

        sheet_names = {doc.metadata["sheet_name"] for doc in result}
        assert sheet_names == {"Sales", "Inventory"}

        for doc in result:
            meta = doc.metadata
            assert meta["file_type"] == "xlsx"
            assert "source" in meta
            assert "sheet_name" in meta
            assert "row_count" in meta
            assert "column_count" in meta
            assert meta["modality"] == "table"
            assert isinstance(meta["row_count"], int)
            assert isinstance(meta["column_count"], int)

        # source field on the Document dataclass
        assert result[0].source == sample_xlsx

        # Verify row/column counts for known sheets
        sales_doc = next(d for d in result if d.metadata["sheet_name"] == "Sales")
        assert sales_doc.metadata["row_count"] == 2     # 2 data rows (excl. header)
        assert sales_doc.metadata["column_count"] == 2  # Product, Revenue

        inv_doc = next(d for d in result if d.metadata["sheet_name"] == "Inventory")
        assert inv_doc.metadata["row_count"] == 2       # 2 data rows (excl. header)
        assert inv_doc.metadata["column_count"] == 3    # Item, Quantity, Location


class TestLoadNonexistentExcelRaises:
    """load_excel raises FileNotFoundError for missing files."""

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_excel("/nonexistent/path/missing.xlsx")
