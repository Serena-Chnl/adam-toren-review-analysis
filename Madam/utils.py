 def download_nltk_resources() -> bool:
     """
     Downloads required NLTK resources for text processing.
     Returns True if successful, False otherwise.
     """
     try:
         for resource, download_name in [
             ('corpora/stopwords', 'stopwords'),
             ('corpora/wordnet', 'wordnet'),
             ('corpora/omw-1.4', 'omw-1.4'),
             ('tokenizers/punkt', 'punkt'),
             ('tokenizers/punkt_tab', 'punkt_tab')
         ]:
             try:
                 nltk.data.find(resource)
             except LookupError:
                 nltk.download(download_name, quiet=True)
         return True
     except Exception as e:
         print(f"Error downloading NLTK resources: {e}")
         return False