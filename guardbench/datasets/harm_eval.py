import datasets

from .bibtex import harm_eval
from .dataset import Dataset


class HarmEval(Dataset):
    def __init__(self):
        # Metadata
        self.name = "HarmEval"
        self.target_role = "user"
        self.category = "prompts"
        self.subcategory = "questions"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "Apache 2.0"
        self.bibtex = harm_eval
        self.alias = "harm_eval"
        self.splits = ["test"]
        self.generation_methods = ["human"]
        self.hazard_categories = [
            "Adult Content",
            "Child Abuse Content",
            "Economic Harm",
            "Fraud/Deception",
            "Hate/Harass/Violence",
            "Illegal Activity",
            "Malware",
            "Physical Harm",
            "Political Campaigning",
            "Privacy Violation Activity",
            "Tailored Financial Advice",
        ]

        # Other
        self.repo_id = "SoftMINER-Group/HarmEval"

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
