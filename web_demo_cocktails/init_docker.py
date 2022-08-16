from views import app
from config import DOCKER_PORT, DEBUG, DOCKER_HOST


if __name__ == "__main__":
    app.run(host=DOCKER_HOST, port=DOCKER_PORT, debug=DEBUG)
