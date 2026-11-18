import unittest

class AdapterClassificationContractTests(unittest.TestCase):
    def classify(self, samples, stable):
        any_contact = any(any(s.get("contacts", [])) for s in samples)
        if any_contact:
            return True, "success" if stable else "partial_contact"
        return False, "no_contact"

    def test_success_episode(self):
        self.assertEqual(
            self.classify([{"contacts": [1, 1, 1]}], True),
            (True, "success"),
        )

    def test_partial_contact_is_retained(self):
        self.assertEqual(
            self.classify([{"contacts": [1, 0, 0]}], False),
            (True, "partial_contact"),
        )

    def test_no_contact_is_audit_only(self):
        self.assertEqual(
            self.classify([{"contacts": [0, 0, 0]}], False),
            (False, "no_contact"),
        )

if __name__ == "__main__":
    unittest.main()
