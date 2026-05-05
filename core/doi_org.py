import requests

BASE_URL = "https://citation.doi.org/metadata"

class DoiOrg():
    def __init__(self, doi):
        self.doi: str = doi
        self.raw = self.getJSONFromDOI()

        self.title = self.raw.get("title")
        self.authors = self.extractAuthors()
        self.page = self.raw.get("page")
        self.proceedings_subject = self.raw.get("proceedings-subject")
        self.abstract = self.raw.get("abstract")
        self.score = self.raw.get("score")

    def getJSONFromDOI(self) -> dict:
        try:
            res = requests.get(BASE_URL, params={"doi": self.doi})

            if res.ok:
                return res.json()
            
            return {}
        
        except Exception as e:
            print("getJSONFromDOI -", e)
            return {}

    def extractAuthors(self):
        authors = self.raw.get("author")
        return Authors(authors)


class Authors():
    def __init__(self, raw: list):
        self.raw = raw

        self.authors = self.extractJSON()

    def extractJSON(self):        
        raw: list = self.raw

        if not raw:
            return []
        
        res = []
        for author in raw:
            given = author.get("given") if author.get("given") else ""
            family = author.get("family") if author.get("family") else ""

            res.append(f"{given} {family}")

        return res
    
    def __str__(self):
        if not self.authors:
            return ""
        
        if len(self.authors) == 1:
            return str(self.authors[0])

        return ", ".join(self.authors) 
    
    def __len__(self):
        return len(self.authors)