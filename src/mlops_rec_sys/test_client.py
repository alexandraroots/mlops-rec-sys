import grpc
import sys
import os
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
proto_dir = os.path.join(current_dir, 'proto')
sys.path.insert(0, proto_dir)
sys.path.insert(0, current_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    from proto import recommendation_pb2
    from proto import recommendation_pb2_grpc

    logger.info("✅ Proto модули успешно импортированы")
except ImportError as e:
    logger.info(f"❌ Ошибка импорта: {e}")
    logger.info("Поиск proto файлов...")

    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.py'):
                logger.info(f"Найден: {os.path.join(root, file)}")
    exit(1)


def test_recommendation():
    logger.info("🔍 Тестирование gRPC сервера...")

    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = recommendation_pb2_grpc.RecommenderStub(channel)

        request = recommendation_pb2.RecommendRequest(
            item_ids=[1001, 1002, 1003, 1004, 1005]
        )

        logger.info(f"📤 Отправляем запрос: {request.item_ids}")

        response = stub.Recommend(request)

        logger.info("✅ Успешный ответ!")
        logger.info(f"📥 Полученные рекомендации: {response.item_ids}")

    except grpc.RpcError as e:
        logger.info(f"❌ gRPC ошибка: {e.code()} - {e.details()}")
    except Exception as e:
        logger.info(f"❌ Общая ошибка: {str(e)}")


if __name__ == '__main__':
    test_recommendation()