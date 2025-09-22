import logging
import sys

import grpc
from proto import recommendation_pb2, recommendation_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recommender(server_address: str):
    with grpc.insecure_channel(server_address) as channel:
        stub = recommendation_pb2_grpc.RecommenderStub(channel)

        test_inputs = [
            [1, 2, 3],
            [10, 20, 30, 40],
            [5, 15, 25],
            [],
            [42],
        ]

        for i, user_history in enumerate(test_inputs, 1):
            request = recommendation_pb2.RecommendRequest(item_ids=user_history)
            logging.info(f"\nТест #{i}: input={user_history}")
            try:
                response = stub.Recommend(request)
                logger.info(f"✅ Output: {list(response.item_ids)}")
            except grpc.RpcError as e:
                logger.info(f"❌ Ошибка: {e.code()} - {e.details()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.info("Usage: python test_grpc_client.py <server_address:port>")
        sys.exit(1)
    server_addr = sys.argv[1]
    recommender(server_addr)
