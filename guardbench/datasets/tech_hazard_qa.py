import datasets

from .bibtex import tech_hazard_qa
from .dataset import Dataset


class TechHazardQA(Dataset):
    def __init__(self):
        # Metadata
        self.name = "TechHazardQA"
        self.target_role = "user"
        self.category = "prompts"
        self.subcategory = "questions"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "Apache 2.0"
        self.bibtex = tech_hazard_qa
        self.alias = "tech_hazard_qa"
        self.splits = ["test"]
        self.generation_methods = ["llm"]
        self.hazard_categories = [
            "Bio Technology, Biology, Genetic Engineering",
            "Chemical Weapon, Biological and Chemical Weapons",
            "Cyber Security",
            "Finance, Banking",
            "Nuclear Technology, Bio Nuclear Terrorism, Terrorism",
            "Public Healthcare System, Pharmacology",
            "Social Media",
        ]

        # Other
        self.repo_id = "SoftMINER-Group/TechHazardQA"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        datasets.load_dataset(self.repo_id)

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = datasets.load_dataset(self.repo_id, split="train")
        test_set = dataset.to_pandas().to_dict("records")
        return {"test": test_set}

    def format_sample(self, id: int, x: dict) -> dict:
        id = str(id)
        conversation = self.format_prompt_as_conversation(x["Question"])
        category = x["Topic"]
        return dict(id=id, label=True, conversation=conversation, category=category)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
