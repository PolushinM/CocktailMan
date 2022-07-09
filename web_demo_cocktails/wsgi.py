from views import app
from config import DEBUG, WSGI_HOST, WSGI_PORT

if __name__ == "__main__":
    app.run(host=WSGI_HOST, port=WSGI_PORT, debug=DEBUG)
