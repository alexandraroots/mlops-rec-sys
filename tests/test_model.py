import pytest
import torch
import numpy as np
import onnxruntime as ort

from scripts.export_model import Model, export_to_onnx


@pytest.fixture(scope="module")
def torch_model():
    torch.manual_seed(42)
    model = Model(num_recommendations=5)
    model.eval()
    return model


@pytest.fixture(scope="module")
def ort_session(torch_model):
    export_to_onnx(model=torch_model, name_model='test_model.onnx')
    return ort.InferenceSession("./models/test_model.onnx")


@pytest.mark.parametrize(
    "test_input",
    [
        torch.tensor([1, 5, 10, 15], dtype=torch.long),
        torch.tensor([10, 20, 30, 40], dtype=torch.long),
        torch.tensor([5, 15, 25], dtype=torch.long),
        torch.tensor([100, 200], dtype=torch.long),
        torch.tensor([42], dtype=torch.long),
    ],
)
def test_onnx_model_correctness(torch_model, ort_session, test_input):
    """
    Сравниваем PyTorch и ONNX модели
    """
    with torch.no_grad():
        torch_output = torch_model(test_input)

    onnx_input = test_input.cpu().numpy()
    onnx_output = ort_session.run(None, {"user_history": onnx_input})[0]

    torch_numpy = torch_output.cpu().numpy()

    assert onnx_output.shape == torch_numpy.shape, (
        f"Несовпадение форм: PyTorch {torch_numpy.shape}, ONNX {onnx_output.shape}"
    )
    assert np.array_equal(torch_numpy, onnx_output), (
        f"Несовпадение значений: PyTorch {torch_numpy}, ONNX {onnx_output}"
    )


def test_model_properties(torch_model, ort_session):
    """
    Проверка количества рекомендаций, типов данных и уникальности
    """
    test_input = torch.tensor([1, 2, 3], dtype=torch.long)

    with torch.no_grad():
        torch_output = torch_model(test_input)

    onnx_output = ort_session.run(None, {"user_history": test_input.numpy()})[0]

    assert len(torch_output) == 5
    assert len(onnx_output) == 5

    assert torch_output.dtype == torch.int64
    assert onnx_output.dtype == np.int64

    assert len(set(torch_output.tolist())) == len(torch_output)
    assert len(set(onnx_output.tolist())) == len(onnx_output)


@pytest.mark.parametrize(
    "test_input",
    [
        torch.tensor([999], dtype=torch.long),
    ],
)
def test_edge_cases(torch_model, ort_session, test_input):
    """
    Проверка крайних случаев
    """
    with torch.no_grad():
        torch_out = torch_model(test_input)

    onnx_out = ort_session.run(None, {"user_history": test_input.numpy()})[0]

    assert torch_out.shape == onnx_out.shape
    assert np.array_equal(torch_out.cpu().numpy(), onnx_out)
