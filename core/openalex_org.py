import requests

BASE_URL = "https://api.openalex.org/works"

class OpenAlex:
    def __init__(self, doi):
        self.doi = doi
        self.raw = self.get_JSON_from_doi()

        self.title = self.raw.get("title")
        self.display_name = self.raw.get("display_name")
        
        self.authors = Authors(self.raw.get("authorships") or [])
        
        self.primary_topic = (self.raw.get("primary_topic") or {}).get("display_name")
        
        self.num_of_awards = len(self.raw.get("awards") or [])

        self.abstract = self.reconstruct_abstract()
        
        self.keywords = self.get_keywords()
        self.concepts = self.get_concepts()

    def get_JSON_from_doi(self, in_doi=None) -> dict:
        doi = in_doi if in_doi else self.doi

        try:
            res = requests.get(f"{BASE_URL}/doi:{doi}")
            if res.ok:
                return res.json()
            return {}
        except Exception as e:
            print(f"get_JSON_from_doi - {e}")
            return {}
        
    def get_keywords(self):
        raw_list = self.raw.get("keywords") or []
        names =  [keyword.get("display_name") for keyword in raw_list if keyword.get("display_name")]
        return ', '.join(names) if names else ""
        
    def get_concepts(self):
        raw_list = self.raw.get("concepts") or []
        names =  [concept.get("display_name") for concept in raw_list if concept.get("display_name")]
        return ', '.join(names) if names else ""

    def reconstruct_abstract(self):
        inverted_index = self.raw.get("abstract_inverted_index", {})

        if inverted_index == None:
            return ""
        
        max_index = -1
        for positions in inverted_index.values():
            if positions:
                max_index = max(max_index, max(positions))
                
        if max_index == -1:
            return ""
            
        words_list = [""] * (max_index + 1)
        
        for word, positions in inverted_index.items():
            for pos in positions:
                words_list[pos] = word
                
        reconstructed_text = " ".join(words_list)
        return reconstructed_text


class Authors:
    def __init__(self, raw):
        self.raw: list = raw if raw else []
        self.authors = self.get_authors()

    def get_authors(self):
        names = []
        for author_info in self.raw:
            author = author_info.get("author") or {}
            author_name = author.get("display_name")
            
            if author_name:
                names.append(author_name)
                
        return names
    
    def __str__(self):
        if not self.authors:
            return ""
        return ", ".join(self.authors)
    
    def __len__(self):
        return int(len(self.authors))