from werkzeug.middleware.proxy_fix import ProxyFix
from app.main import app

# Make the flask app aware of the reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_port=1, x_proto=1, x_prefix=1)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=False, processes=1)
