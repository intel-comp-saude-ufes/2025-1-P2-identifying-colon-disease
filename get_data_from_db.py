import os
import kagglehub


def dowload_data():
    # Download latest version (kagglehub will handle caching automaticamente)
    path = kagglehub.dataset_download("francismon/curated-colon-dataset-for-deep-learning")
    print(f"Dataset downloaded/cached at: {path}")
    return path


class DataLoader:
    def __init__(self):
        print("Initializing DataLoader...")
        self.path = dowload_data()
        
        # Definição das categorias
        self.categories = {
            0: "normal",
            1: "ulcerative_colitis",
            2: "polyps",
            3: "esophagitis"
        }

        # Carrega todos os arquivos em um dicionário estruturado
        self.data = {
            split: {
                idx: os.listdir(os.path.join(self.path, split, f"{idx}_{name}"))
                for idx, name in self.categories.items()
            }
            for split in ["train", "test", "val"]
        }

    # -------- Métodos auxiliares --------
    def _get_split_data(self, split: str, with_labels: bool = True):
        """Retorna dados de um split (train/test/val), opcionalmente com labels"""
        files = []
        for idx, name in self.categories.items():
            imgs = self.data[split][idx]
            base_path = os.path.join(self.path, split, f"{idx}_{name}")
            if with_labels:
                files.extend([(os.path.join(base_path, f), idx) for f in imgs])
            else:
                files.extend([os.path.join(base_path, f) for f in imgs])
        return files

    # -------- Métodos públicos --------
    def get_train_data(self):
        return self._get_split_data("train", with_labels=True)

    def get_test_data(self):
        return self._get_split_data("test", with_labels=True)

    def get_validation_data_with_labels(self):
        return self._get_split_data("val", with_labels=True)

    def get_train_data_by_category(self):
        """Retorna os arquivos de treino organizados por categoria"""
        return {
            self.categories[idx]: self.data["train"][idx]
            for idx in self.categories
        }

    def get_validation_data(self):
        """Retorna os arquivos de validação organizados por categoria"""
        return {
            self.categories[idx]: self.data["val"][idx]
            for idx in self.categories
        }

    def get_category_counts(self):
        """Conta de imagens em cada categoria e split"""
        return {
            f"{split}_{self.categories[idx]}": len(files)
            for split, cats in self.data.items()
            for idx, files in cats.items()
        }