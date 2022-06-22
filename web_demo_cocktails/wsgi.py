from app import app
from app_config import config

if __name__ == "__main__":
    app.run(debug=config['debug'])
