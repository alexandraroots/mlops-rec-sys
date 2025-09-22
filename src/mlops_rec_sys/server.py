import logging
import os
import time
from concurrent import futures

import grpc
import numpy as np
import onnxruntime as ort
import proto.recommendation_pb2 as recommendation_pb2
from prometheus_client import Counter, Gauge, Summary, start_http_server
from proto.recommendation_pb2_grpc import RecommenderServicer, add_RecommenderServicer_to_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_TIME = Summary("grpc_request_duration_seconds", "Time spent processing gRPC request")
REQUEST_COUNT = Counter("grpc_requests_total", "Total number of gRPC requests", ["method", "code"])
REQUESTS_IN_PROGRESS = Gauge("grpc_requests_in_progress", "Number of gRPC requests in progress", ["method"])


class LoggingInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        start = time.time()
        response = continuation(handler_call_details)
        duration = time.time() - start
        logger.info(f"[gRPC] {handler_call_details.method} - {duration:.4f}s")
        return response


class RecommenderService(RecommenderServicer):
    def __init__(self, model_path: str):
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

    @REQUEST_TIME.time()
    def Recommend(self, request, context):
        """
        Обработка gRPC запроса для получения рекомендаций
        """
        method = "Recommender/Recommend"
        try:
            item_ids = list(request.item_ids)

            if not item_ids:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Пустой список item_ids")
                REQUEST_COUNT.labels(method=method, code="INVALID_ARGUMENT").inc()
                logger.warning("⚠️ Получен пустой список item_ids")
                return recommendation_pb2.RecommendResponse()

            logger.info(f"📥 Получен запрос: {item_ids}")

            input_data = np.array(item_ids, dtype=np.int64)

            recommendations = self._get_recommendations(input_data)

            logger.info(f"📤 Отправляем рекомендации: {recommendations.tolist()}")
            REQUEST_COUNT.labels(method=method, code="OK").inc()

            return recommendation_pb2.RecommendResponse(item_ids=recommendations.tolist())

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке запроса: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Ошибка сервера: {str(e)}")
            REQUEST_COUNT.labels(method=method, code="INTERNAL").inc()
            return recommendation_pb2.RecommendResponse()
        finally:
            REQUESTS_IN_PROGRESS.labels(method=method).dec()

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
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, "models", "recommendation_model.onnx")

    if not os.path.exists(model_path):
        logger.error(f"❌ Модель не найдена по пути: {model_path}")
        return

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )

    service = RecommenderService(model_path)

    add_RecommenderServicer_to_server(service, server)

    grpc_port = 50051
    metrics_port = 8000

    server.add_insecure_port(f"[::]:{grpc_port}")
    server.start()

    start_http_server(metrics_port)
    logger.info(f"🚀 gRPC сервер запущен на порту {grpc_port}")
    logger.info(f"📊 Метрики Prometheus доступны на порту {metrics_port} (/metrics)")
    logger.info(f"📁 Модель загружена из: {model_path}")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("🛑 Остановка сервера...")
        server.stop(0)


if __name__ == "__main__":
    serve()
