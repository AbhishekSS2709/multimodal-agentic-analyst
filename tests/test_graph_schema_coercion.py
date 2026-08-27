"""Structured-output schemas must tolerate provider quirks.

Different models serialise booleans differently. qwen/qwen3.6-27b returns
`"true"` where the schema declares a boolean, and Groq rejects the tool call
with:

    tool call validation failed: parameters for tool GradeDocuments did not
    match schema: errors: [`/relevant`: expected boolean, but got string

which the node catches as "LLM unavailable" and answers heuristically instead.
Fifty of sixty-seven fallbacks in one evaluation run came from this alone, so
the LLM arm silently measured the heuristic path.
"""

import unittest

from src.graph.schemas import GradeDocuments, Verification


def _field_schema(model, field):
    return model.model_json_schema()["properties"][field]


def _accepts_string(schema):
    if schema.get("type") == "string":
        return True
    return any(opt.get("type") == "string"
               for opt in schema.get("anyOf", []) + schema.get("oneOf", []))


class TestEmittedJsonSchema(unittest.TestCase):
    """The rejection is server-side, before Pydantic runs.

    Groq validates tool-call arguments against the JSON Schema we send, so
    Pydantic's own lax coercion never gets a chance. The *schema* has to permit
    the string form or the call 400s.
    """

    def test_grade_relevant_schema_permits_string(self):
        self.assertTrue(_accepts_string(_field_schema(GradeDocuments, "relevant")),
                        "schema rejects string booleans server-side")

    def test_verification_grounded_schema_permits_string(self):
        self.assertTrue(_accepts_string(_field_schema(Verification, "grounded")))

    def test_verification_relevant_schema_permits_string(self):
        self.assertTrue(_accepts_string(_field_schema(Verification, "relevant")))


class TestBooleanCoercion(unittest.TestCase):

    def test_grade_accepts_real_booleans(self):
        self.assertTrue(GradeDocuments(relevant=True).relevant)
        self.assertFalse(GradeDocuments(relevant=False).relevant)

    def test_grade_accepts_string_true(self):
        for value in ("true", "True", "TRUE", "yes", "1"):
            self.assertTrue(GradeDocuments(relevant=value).relevant,
                            f"{value!r} should coerce to True")

    def test_grade_accepts_string_false(self):
        for value in ("false", "False", "FALSE", "no", "0"):
            self.assertFalse(GradeDocuments(relevant=value).relevant,
                             f"{value!r} should coerce to False")

    def test_verification_booleans_coerce(self):
        v = Verification(grounded="true", relevant="false", score=0.5)
        self.assertTrue(v.grounded)
        self.assertFalse(v.relevant)

    def test_scores_given_as_strings_coerce(self):
        """Same models also emit numbers as strings."""
        self.assertEqual(GradeDocuments(relevant="true", score="0.75").score, 0.75)

    def test_unparseable_value_still_raises(self):
        """Coercion must not swallow genuinely broken output."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GradeDocuments(relevant="maybe-ish")


if __name__ == "__main__":
    unittest.main()


class TestLabelStyleGrades(unittest.TestCase):
    """Smaller models answer the question instead of filling the field.

    Asked whether a document is relevant, `google/gemma-4-E4B-it` returns the
    string "Not Relevant". That is an unambiguous grade, and discarding it sent
    the node to its heuristic grader while the run still looked like an LLM one.
    """

    def _grade(self, value):
        from src.graph.schemas import GradeDocuments
        return GradeDocuments(relevant=value).relevant

    def test_relevant_label_reads_as_true(self):
        for value in ("Relevant", "relevant", " RELEVANT "):
            self.assertIs(self._grade(value), True, value)

    def test_not_relevant_label_reads_as_false(self):
        for value in ("Not Relevant", "not relevant", "Irrelevant", "not_relevant"):
            self.assertIs(self._grade(value), False, value)

    def test_still_refuses_an_ungradeable_string(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._grade("possibly, in some respects")
