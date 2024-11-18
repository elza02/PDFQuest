import unittest
from app.Rag_with_py import retrieving_chunks, llm_generation

class TestRAG(unittest.TestCase):
    def test_retrieving_chunks(self):
        """Test retrieving_chunks function."""
        query = "qu'est ce que ca veut dire université?"
        chunks = retrieving_chunks(query)
        self.assertTrue(isinstance(chunks, str))
        self.assertGreater(len(chunks), 0)  # Ensure chunks are returned

    def test_llm_generation(self):
        """Test llm_generation function."""
        question = "Ca veut dire quoi universite"
        context = "les droits d'un étudiant sont : La liberté d'information et d'expression dans les enceintes et locaux des établissements d'enseignement."
        response = llm_generation(question, context)
        self.assertTrue(isinstance(response, str))
        self.assertGreater(len(response), 0)  # Ensure a response is generated

if __name__ == "__main__":
    unittest.main()
