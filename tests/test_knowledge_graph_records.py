"""Triple extraction from structured log records.

The existing rules expect prose ("Supplier X delayed shipment Y"), but the
corpus is pipe-delimited records, so extraction yielded 4 triples from 451
documents and the graph specialist had almost nothing to traverse.  These are
verbatim lines from data/dispatch_log.txt and data/machine_failure_logs.txt.

Entity-valued fields are normalised the existing way (whitespace ->
underscore); free-text fields such as Root Cause stay readable, because the
synthesizer quotes them straight into the answer.
"""

import unittest

DISPATCH = (
    "[2023-01-10 07:00] DISPATCH-1002 | Warehouse: Cleveland-W1 | "
    "Destination: Precision Dynamics LLC - Assembly Plant | "
    "Items: CNC Spindle Bearings x200 | Carrier: National Express | "
    "Status: IN_TRANSIT | ETA: 2023-01-14 | Supplier: Apex Materials | "
    "Notes: Shipment delayed at origin - supplier warehouse picking backlog"
)

DELAY_UPDATE = (
    "[2023-01-10 16:45] DISPATCH-1002 | Status Update: DELAYED | "
    "Revised ETA: 2023-01-18 | Reason: Apex Materials warehouse reported "
    "inventory discrepancy, recount in progress | Escalation: Level 1"
)

MACHINE = (
    "[2023-03-05 16:45] MACHINE: CNC-Mill-07 | LOCATION: Chicago Plant | "
    "SEVERITY: MEDIUM | Issue: Spindle bearing noise increasing | "
    "Downtime: 0hrs | Root Cause: CNC Spindle Bearings on order from "
    "Apex Materials (ORD-2023-0022), order delayed | "
    "Production Impact: Machine flagged for reduced speed operation"
)


def _triples(text):
    from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
    return KnowledgeGraphBuilder().extract_triples(text)


def _pairs(text):
    return {(s, p, o) for s, p, o, _ in _triples(text)}


class TestDispatchRecords(unittest.TestCase):

    def test_supplier_is_linked_to_the_dispatch(self):
        found = _pairs(DISPATCH)
        self.assertIn(("DISPATCH-1002", "supplied_by", "Apex_Materials"), found)

    def test_destination_is_linked(self):
        subjects = {(s, p) for s, p, _ in _pairs(DISPATCH)}
        self.assertIn(("DISPATCH-1002", "shipped_to"), subjects)

    def test_carrier_is_linked(self):
        self.assertIn(("DISPATCH-1002", "carried_by", "National_Express"),
                      _pairs(DISPATCH))

    def test_status_update_is_linked(self):
        self.assertIn(("DISPATCH-1002", "has_status", "DELAYED"),
                      _pairs(DELAY_UPDATE))

    def test_delay_reason_is_captured(self):
        """'Why are there dispatch delays?' must be answerable from this."""
        objs = [o for s, p, o in _pairs(DELAY_UPDATE) if p == "delayed_because"]
        self.assertTrue(objs, "no delayed_because triple")
        self.assertIn("apex materials", " ".join(objs).lower())

    def test_noisy_fields_are_not_entities(self):
        """Timestamps and ETAs would swamp the graph with junk nodes."""
        preds = {p for _, p, _ in _pairs(DISPATCH)}
        self.assertNotIn("eta", preds)


class TestMachineRecords(unittest.TestCase):

    def test_machine_location_is_linked(self):
        self.assertIn(("CNC-Mill-07", "located_in", "Chicago_Plant"),
                      _pairs(MACHINE))

    def test_root_cause_is_captured(self):
        objs = [o for s, p, o in _pairs(MACHINE) if p == "failed_due_to"]
        self.assertTrue(objs, "no failed_due_to triple")
        self.assertIn("apex materials", " ".join(objs).lower())

    def test_machine_is_the_subject_not_the_timestamp(self):
        subjects = {s for s, _, _ in _pairs(MACHINE)}
        self.assertIn("CNC-Mill-07", subjects)
        self.assertFalse([s for s in subjects if s.startswith("2023-")])


class TestExistingProseStillWorks(unittest.TestCase):
    """The record extractor must not displace the original rules."""

    def test_prose_rule_still_matches(self):
        found = _pairs("Supplier Apex Materials delayed shipment ORD-1234.")
        self.assertTrue(any(p == "delayed" for _, p, _ in found), found)


if __name__ == "__main__":
    unittest.main()
