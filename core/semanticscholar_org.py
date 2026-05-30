import requests
from config import config

class SemanticScholar:
    def __init__(self, paper_id, api_key=config.SEMANTICSCHOLAR_API_KEY, base_url="https://api.semanticscholar.org/graph/v1/paper/"):
        self.paper_id = paper_id
        self.api_key = api_key
        self.base_url = base_url

        self.abstract = None
        self.doi = None
        self.tldr = None
        self.reference_count = 0
        self.citation_count = 0
        self.is_open_access = None

        # if paper_id != None:
        #     self.get_details()

    def get_details(self):
        """
        Fetches abstract, doi, and tldr for the initialized paper_id.
        """
        endpoint = f"{self.base_url}{self.paper_id}"
        
        params = {
            "fields": "abstract,externalIds,tldr,referenceCount,isOpenAccess"
        }
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        response = requests.get(endpoint, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            # Extracting the specific fields requested
            self.abstract = data.get("abstract", "")
            self.doi = data.get("externalIds", {}).get("DOI")
            self.tldr = data.get("tldr", {}).get("text") if data.get("tldr") else None
            self.reference_count = data.get("referenceCount", 0)
            self.citation_count = data.get("citationCount", 0)
            self.is_open_access = data.get("isOpenAccess")
            return {
                "abstract": data.get("abstract"),
                "doi": data.get("externalIds", {}).get("DOI"),
                "tldr": data.get("tldr", {}).get("text") if data.get("tldr") else None
            }
        else:
            response.raise_for_status()


class SemanticScholarDOI(SemanticScholar):
    def __init__(self, doi, api_key=None):
        super().__init__(doi, api_key, "https://api.semanticscholar.org/graph/v1/paper/DOI:")