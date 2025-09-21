import logging
import os
import time
from concurrent import futures

import grpc
import numpy as np
import onnxruntime as ort
import proto.recommendation_pb2 as recommendation_pb2
from proto.recommendation_pb2_grpc import RecommenderServicer, add_RecommenderServicer_to_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommenderService(RecommenderServicer):
    def __init__(self, model_path: str):
        """
        Инициализация сервиса с загрузкой ONNX модели

        Args:
            model_path: путь к ONNX модели
        """
        logger.info(f"🔄 Загрузка ONNX модели из {model_path}...")
        try:
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            logger.info("✅ Модель успешно загружена")

            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            logger.info(f"📥 Входы модели: {[inp.name for inp in inputs]}")
            logger.info(f"📤 Выходы модели: {[out.name for out in outputs]}")

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise

    def Recommend(self, request, context):
        """
        Обработка gRPC запроса для получения рекомендаций
        """
        try:
            item_ids = list(request.item_ids)

            if not item_ids:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Пустой список item_ids")
                logger.warning("⚠️ Получен пустой список item_ids")
                return recommendation_pb2.RecommendResponse()

            logger.info(f"📥 Получен запрос: {item_ids}")

            input_data = np.array(item_ids, dtype=np.int64)

            recommendations = self._get_recommendations(input_data)

            logger.info(f"📤 Отправляем рекомендации: {recommendations.tolist()}")

            return recommendation_pb2.RecommendResponse(item_ids=recommendations.tolist())

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке запроса: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Ошибка сервера: {str(e)}")
            return recommendation_pb2.RecommendResponse()

    def _get_recommendations(self, input_data: np.ndarray) -> np.ndarray:
        """
        Получение рекомендаций от ONNX модели
        """
        outputs = self.session.run(None, {self.input_name: input_data})

        recommendations = outputs[0]

        if len(recommendations.shape) > 1:
            recommendations = recommendations.flatten()

        return recommendations.astype(np.int32)


def serve():
    """
    Запуск gRPC сервера
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, "models", "recommendation_model.onnx")

    if not os.path.exists(model_path):
        logger.error(f"❌ Модель не найдена по пути: {model_path}")
        return

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[("grpc.max_send_message_length", 50 * 1024 * 1024), ("grpc.max_receive_message_length", 50 * 1024 * 1024)],
    )

    service = RecommenderService(model_path)

    add_RecommenderServicer_to_server(service, server)

    port = 50051
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    logger.info(f"🚀 gRPC сервер запущен на порту {port}")
    logger.info(f"📁 Модель загружена из: {model_path}")
    logger.info("📡 Ожидаем запросы...")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("🛑 Остановка сервера...")
        server.stop(0)


if __name__ == "__main__":
    serve()
