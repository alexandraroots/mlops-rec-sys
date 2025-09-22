import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(torch.nn.Module):
    def __init__(self, num_recommendations: int = 10, device: str = "cpu") -> None:
        super(Model, self).__init__()
        torch.manual_seed(42)
        self._item_embeddings = torch.rand((10000, 32), device=device)
        self._num_recommendations = num_recommendations

    def forward(self, user_history: torch.Tensor) -> torch.Tensor:
        user_embedding = self._item_embeddings[user_history].mean(axis=0)
        scores = user_embedding @ self._item_embeddings.T
        topk = torch.topk(scores, k=self._num_recommendations)
        return topk.indices


def export_to_onnx(model: Model, name_model: str = "recommendation_model.onnx"):
    """
    Экспорт модели PyTorch в ONNX формат
    """
    torch.manual_seed(42)
    model.eval()

    dummy_input = torch.tensor([1, 5, 10, 15], dtype=torch.long)

    torch.onnx.export(
        model,
        dummy_input,
        f"./models/{name_model}",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["user_history"],
        output_names=["recommendations"],
        dynamic_axes={"user_history": {0: "history_length"}},
        verbose=False,
    )

    logger.info("✅ Модель успешно экспортирована в ONNX")


if __name__ == "__main__":
    model = Model()
    export_to_onnx(model)
